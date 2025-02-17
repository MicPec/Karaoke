import difflib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

import lyricsgenius
import torch
import torchaudio
import whisper
from demucs.apply import apply_model
from demucs.pretrained import get_model
from dotenv import load_dotenv

load_dotenv()


def download_lyrics(song_title: str, artist_name: str = "") -> str:
    genius = lyricsgenius.Genius(os.getenv("GENIUS_API_KEY"))
    song = genius.search_song(song_title, artist_name)
    return song.lyrics


CACHE_DIR = Path("cache")
STRIP_CHARS = " ,.!?()[]/\\\"'"


MatchType = Enum(
    "MatchType",
    {
        "NONE": "NONE",
        "DIRECT": "DIRECT",
        "SIMILAR": "SIMILAR",
        "JOIN": "JOIN",
        "PARTIAL": "PARTIAL",
        "MISSING": "MISSING",
        "EXCESS": "EXCESS",
    },
)


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0

    def __str__(self):
        return self.text

    @property
    def duration(self):
        return self.end - self.start


Segment = TypeVar("Segment")


@dataclass
class Segment:
    text: str
    words: list[Word]
    match_type: MatchType = MatchType.NONE
    match_confidence: float = 0.0

    def __str__(self):
        return self.text

    def stretch(self, value: float, backward: bool = False):
        duration = self.words[-1].end - self.words[0].start
        new_duration = duration + value
        scale = new_duration / duration

        for word in self.words:
            if not backward:
                word.start = self.words[0].start + (word.start - self.words[0].start) * scale
                word.end = self.words[0].start + (word.end - self.words[0].start) * scale
            else:
                word.start = self.words[-1].end - (self.words[-1].end - word.start) * scale
                word.end = self.words[-1].end - (self.words[-1].end - word.end) * scale

    def shift(self, value: float):
        for word in self.words:
            word.start += value
            word.end += value

    def rebase(self, base: float):
        diff = base - self.start
        for word in self.words:
            word.start += diff
            word.end += diff

    def insert_word(self, word: Word, index: int):
        word.start = self.start + self.duration / len(self.words) * index
        word.end = self.start + self.duration / len(self.words) * (index + 1)
        for w in self.words[index + 1 :]:
            w.start += self.duration / len(self.words)
            w.end += self.duration / len(self.words)
        self.words.insert(index, word)
        self.text = " ".join([word.text for word in self.words])

    def remove_word(self, index: int):
        for w in self.words[index + 1 :]:
            w.start -= self.duration / len(self.words)
            w.end -= self.duration / len(self.words)
        self.words.pop(index)
        self.text = " ".join([word.text for word in self.words])

    def split(self, index: int) -> Segment:
        if index == 0 or index >= len(self.words):
            return None
        new_segment = Segment(
            text=" ".join([word.text for word in self.words[index:]]),
            words=self.words[index:],
            match_type=MatchType.PARTIAL,
            match_confidence=self.match_confidence,
        )
        self.words = self.words[:index]
        self.text = " ".join([word.text for word in self.words])
        return new_segment

    def join(self, other: Segment, *, rebase_time: float = 0.0):
        if rebase_time != 0.0:
            other.rebase(rebase_time)
        if self.end > other.start:
            raise ValueError("Cannot join segments that overlap")
            # other.rebase(self.end)
        self.words.extend(other.words)
        self.match_type = MatchType.JOIN
        self.match_confidence = max(self.match_confidence, other.match_confidence)
        self.text = " ".join([word.text for word in self.words])

    @property
    def raw(self):
        return " ".join([word.text.strip(STRIP_CHARS).lower() for word in self.words])

    @property
    def words_count(self):
        return len(self.words)

    @property
    def start(self):
        if not self.words:
            return 0.0
        return self.words[0].start

    @property
    def end(self):
        if not self.words:
            return 0.0
        return self.words[-1].end

    @property
    def duration(self) -> float:
        """Return the duration of the lyrics slice."""
        if not self.words:
            return 0.0
        return self.end - self.start


Lyrics = TypeVar("Lyrics")


class Lyrics:
    def __init__(self, transcription: dict) -> None:
        self.transcription = transcription
        self.segments: list[Segment] = []
        self.segments = self._prepare_transcription()

    def __repr__(self) -> str:
        return f"Lyrics({self.segments})"

    def __str__(self) -> str:
        return "\n".join([str(segment) for segment in self.segments])

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Segment:
        return self.segments[index]

    def _prepare_transcription(self):
        segments = []
        for seg in self.transcription["segments"]:
            segment = Segment(
                text=seg["text"].strip(STRIP_CHARS),
                words=[
                    Word(
                        text=word["word"].strip(STRIP_CHARS),
                        start=word["start"],
                        end=word["end"],
                        probability=word.get("probability", 0.5),
                    )
                    for word in seg["words"]
                ],
                match_type=MatchType.NONE,
            )
            segments.append(segment)
        return segments

    def clean_str(self, text: str) -> str:
        return re.sub(r"[\{\[\(].*?[\}\]\)]", "", text).strip(STRIP_CHARS).lower()

    def get_segment_by_time(self, time: float) -> Segment | None:
        for segment in self.segments:
            if segment.start <= time <= segment.end:
                return segment
        return None

    def get_word_by_time(self, time: float) -> Word | None:
        for segment in self.segments:
            for word in segment.words:
                if word.start <= time <= word.end:
                    return word
        return None

    def get_word_by_pos(self, pos: int) -> Word | None:
        curr_pos = 0
        for segment in self.segments:
            for word in segment.words:
                if curr_pos == pos:
                    return word
                curr_pos += 1
        return None

    def split_segment(self, segment: Segment, *, start: int, end: int) -> list[Segment]:
        if start == 0 and end >= len(segment.words):
            return [segment]
        if start == 0:
            new_segment = segment.split(end)
            if new_segment:
                self.segments.insert(self.segments.index(segment) + 1, new_segment)
                return [segment, new_segment]
            return [segment]
        if end >= len(segment.words):
            new_segment = segment.split(start)
            if new_segment:
                self.segments.insert(self.segments.index(segment) + 1, new_segment)
                return [segment, new_segment]
            return [segment]
        if start > 0 and end < len(segment.words):
            result = []
            new_segment = segment.split(start)
            if new_segment:
                self.segments.insert(self.segments.index(segment) + 1, new_segment)
                result.extend([segment, new_segment])
                new_segment2 = new_segment.split(end - start)
                if new_segment2:
                    result.append(new_segment2)
                    self.segments.insert(self.segments.index(new_segment) + 1, new_segment2)
        return result

    @property
    def words(self):
        return [w for segment in self.segments for w in segment.words]

    def find_best_match(self, search_text: str, *, similarity_threshold: float, max_try_num: int = 2) -> Segment | None:
        st = self.clean_str(search_text)
        best_match_score = 0.0
        best_matched_segment = None
        try_num = 0
        while try_num < max_try_num:
            for segment in self.segments:
                if segment.match_type in (MatchType.DIRECT,):
                    continue
                similarity = difflib.SequenceMatcher(None, st, segment.raw).ratio()
                if similarity == 1.0:
                    segment.text = search_text
                    segment.match_type = MatchType.DIRECT
                    segment.match_confidence = 1.0
                    return segment
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_matched_segment = segment

            if best_match_score >= similarity_threshold:
                if best_matched_segment.match_confidence < best_match_score:
                    best_matched_segment.text = search_text
                    best_matched_segment.match_type = MatchType.SIMILAR
                    best_matched_segment.match_confidence = best_match_score
                return best_matched_segment
            try_num += 1
            similarity_threshold -= 0.1
        return None

    def split_seg_by_text(self, segment: Segment, text: str, *, similarity_threshold: float) -> Segment | None:
        search_text = text.split()

        best_score = 0
        best_match = -1

        for pos in range(len(segment.words) - len(text.split()) + 1):
            subtext = " ".join([segment.words[pos + i].text for i in range(len(search_text))])
            similarity = difflib.SequenceMatcher(None, subtext, text).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = pos

        if best_match >= 0 and best_score >= similarity_threshold:
            start = best_match
            end = best_match + len(search_text)
            return self.split_segment(segment, start=start, end=end)

        return None

    def verify_next_segment(self, segment: Segment, text: str, *, similarity_threshold: float) -> Segment | None:
        pos = self.segments.index(segment)
        if pos < len(self.segments) - 1:
            joined_text = " ".join([segment.text, self.segments[pos + 1].text])
            similarity = difflib.SequenceMatcher(None, joined_text, text).ratio()
            if similarity >= similarity_threshold:
                result = segment.join(self.segments[pos + 1])
                self.segments.remove(self.segments[pos + 1])
                return result
        return None

    def to_dict(self) -> dict:
        return {
            "segments": [
                {
                    "text": segment.text,
                    "words": [{"text": w.text, "start": w.start, "end": w.end} for w in segment.words],
                    "match_type": segment.match_type.value,
                    "match_confidence": segment.match_confidence,
                }
                for segment in self.segments
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> Lyrics:
        segments = [
            Segment(
                text=segment["text"],
                words=[Word(text=word["text"], start=word["start"], end=word["end"]) for word in segment["words"]],
                match_type=MatchType(segment["match_type"]),
                match_confidence=segment["match_confidence"],
            )
            for segment in data["segments"]
        ]


class LyricsAligner:
    def __init__(self, audio_path: Path) -> None:
        self.audio_file = audio_path
        self.cache_dir = CACHE_DIR / self.audio_file.stem
        self.vocals_path = self.cache_dir / f"{self.audio_file.stem}.vocals.mp3"
        self.instr_path = self.cache_dir / f"{self.audio_file.stem}.instr.mp3"
        self.lyrics_path = self.cache_dir / f"{self.audio_file.stem}.lyrics.txt"
        self.transcription_path = self.cache_dir / f"{self.audio_file.stem}.transcription.json"
        self.aligned_lyrics_path = self.cache_dir / f"{self.audio_file.stem}.aligned_lyrics.json"
        self.transcription = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process_audio(self, overwrite: bool = False):
        if self.vocals_path.exists() and self.instr_path.exists() and not overwrite:
            print("Audio already processed.")
        else:
            self.process_audio_file()

        raw_lyrics = ""
        if self.lyrics_path.exists():
            raw_lyrics = self.get_lyrics(self.lyrics_path)
        else:
            print(f"No lyrics file found at {self.lyrics_path}")
            # TODO: download lyrics
            return None

        print("Transcribing audio...")
        self.transcription = self.transcribe_audio()

        print("Aligning lyrics...")
        lyrics = Lyrics(self.transcription)
        # lyrics.create_lyric_slices(raw_lyrics)
        aligned = self.align(lyrics, raw_lyrics)
        self.save_alignment(aligned, self.aligned_lyrics_path)
        return aligned

    def align(self, lyrics: Lyrics, raw_lyrics: str, *, max_try_num: int = 2) -> Lyrics:
        raw_lyrics = [line.strip() for line in raw_lyrics.splitlines()]
        try_num = 0
        completed = False

        while not completed and try_num < max_try_num:
            print(f"\nAttempt {try_num + 1} to fix alignment")
            # pass 1: find best matches
            for line in raw_lyrics:
                if not line.strip():
                    continue
                segment = lyrics.find_best_match(line, similarity_threshold=0.9, max_try_num=3)
                if not segment:
                    print(f"Couldn't find match for: {line}")
                    continue
                print(f"Matched: {line} ->> {segment}")

            # pass 2: process segments where there's no match
            for segment in lyrics.segments:
                if segment.match_type not in (MatchType.DIRECT, MatchType.SIMILAR):
                    print(f"Processing segment: {segment}")
                    for line in raw_lyrics:
                        seg = lyrics.verify_next_segment(segment, line, similarity_threshold=0.8)
                        if seg:
                            print(f"Matched: {line} ->> {seg.text}")
                            continue
                        seg = lyrics.split_seg_by_text(segment, line, similarity_threshold=0.8)
                        if seg and len(seg) > 1:
                            print(f"Matched: {line} ->> {[seg.text for seg in seg]}")
                            continue

            try_num += 1
        return lyrics

    def get_vocals(self):
        return self.vocals_path

    def get_instr(self):
        return self.instr_path

    def get_aligned_lyrics(self):
        if self.aligned_lyrics_path.exists():
            print(f"Reading aligned lyrics from {self.aligned_lyrics_path}")
            with open(self.aligned_lyrics_path, "r") as f:
                return json.load(f)
        else:
            print(f"No aligned lyrics file found at {self.aligned_lyrics_path}")
            return None

    def get_lyrics(self, lyrics_path: Path) -> str:
        if lyrics_path.exists():
            with open(lyrics_path, "r") as f:
                return self.prepare_lyrics(f.read())
        else:
            print(f"No lyrics file found at {lyrics_path}")
            return None

    def prepare_lyrics(self, lyrics: str = None) -> str:
        lyrics = re.sub(r"[\{\[\(].*?[\}\]\)]", "", lyrics).strip(STRIP_CHARS).lower()
        return lyrics

    def process_audio_file(self):
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

        print(f"Processing {self.audio_file}")
        try:
            model = get_model("htdemucs")
            model.eval()

            if torch.cuda.is_available():
                model.cuda()

            audio, sr = torchaudio.load(self.audio_file)

            # Ensure audio is stereo
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio], dim=0)  # Convert mono to stereo
            elif audio.shape[0] > 2:
                audio = audio[:2]  # Take first two channels if more than stereo
            audio = audio.unsqueeze(0)

            if torch.cuda.is_available():
                audio = audio.cuda()

            try:
                with torch.no_grad():
                    sources = apply_model(model, audio, split=True, progress=True)

                # Get the index of vocals
                vocals_idx = model.sources.index("vocals")
                vocals = sources[0, vocals_idx]
                # Sum all other sources for instrumental
                instrumental = torch.zeros_like(vocals)
                for i, source in enumerate(model.sources):
                    if i != vocals_idx:  # Skip vocals
                        instrumental += sources[0, i]

                # Safe vocal enhancement
                vocals = torch.clamp(vocals * 1.5, -1, 1)  # Increase vocal presence safely
                vocals = vocals + torch.clamp(vocals - vocals.roll(1, -1), -0.5, 0.5) * 0.3  # Enhance clarity
                vocals = vocals + torch.clamp(vocals - vocals.roll(2, -1), -0.2, 0.2) * 0.1  # Enhance harmonicity

            finally:
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if torch.cuda.is_available():
                vocals = vocals.cpu()
                instrumental = instrumental.cpu()

            print(f"Saving vocals shape: {vocals.shape}, instrumental shape: {instrumental.shape}")
            self.save_track(self.vocals_path, vocals, sr)
            self.save_track(self.instr_path, instrumental, sr)

        except Exception as e:
            print(f"Error processing audio file: {e}")
            raise

    def save_track(self, path: Path, audio: torch.Tensor, sr: int = 44100):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torchaudio.save(path, audio, sr)
            print(f"Saved {path}")
        except IOError:
            print(f"Error saving {path}")
            raise

    def save_transcription(self, path: Path, transcription: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(transcription, f, indent=2)
            print(f"Saved transcription to {path}")

    def load_transcription(self, path: Path) -> dict:
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def transcribe_audio(self, model_size: str = "large") -> dict:
        if result := self.load_transcription(self.transcription_path):
            return result

        try:
            print("Loading model...")
            model = whisper.load_model(model_size)
            print("Processing audio...")

            if not self.vocals_path.exists():
                raise FileNotFoundError(f"Vocals file not found: {self.vocals_path}")

            audio = whisper.load_audio(str(self.vocals_path))
            result = model.transcribe(
                audio,
                word_timestamps=True,
                hallucination_silence_threshold=0.1,
            )
            self.save_transcription(self.transcription_path, result)
            return result
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise
        finally:
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_alignment(self, aligned: Lyrics, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(aligned.to_dict(), f, indent=1)

    def load_alignment(self, path: Path) -> Lyrics:
        if not path.exists():
            return None
        with open(path, "r") as f:
            return Lyrics.from_dict(json.load(f))
