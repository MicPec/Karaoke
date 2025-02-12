import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
from pathlib import Path
import whisper
import difflib
import json
import re
from dataclasses import dataclass
from typing import TypeVar


CACHE_DIR = Path("cache")
STRIP_CHARS = " ,.!?()[]\"'"


@dataclass
class Word:
    text: str
    start: float
    end: float

    def __str__(self):
        return self.text

    @property
    def duration(self):
        return self.end - self.start


LyricsSlice = TypeVar("LyricsSlice")


@dataclass
class LyricsSlice:
    words: list[Word]
    text: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> LyricsSlice:
        return cls(
            words=[Word(w["text"], w["start"], w["end"]) for w in data["words"]],
            text=data.get("raw_text", None),
        )

    def __str__(self):
        return " ".join(word.text.strip() for word in self.words)

    # def append(self, slice: LyricsSlice, rebase: bool = True) -> None:
    #     if rebase:
    #         slice.rebase(self.end)
    #     self.words += slice.words

    def stretch(self, value: float, backward: bool = False):
        duration = self.words[-1].end - self.words[0].start
        new_duration = duration + value
        scale = new_duration / duration

        for word in self.words:
            if not backward:
                word.start = (
                    self.words[0].start + (word.start - self.words[0].start) * scale
                )
                word.end = (
                    self.words[0].start + (word.end - self.words[0].start) * scale
                )
            else:
                word.start = (
                    self.words[-1].end - (self.words[-1].end - word.start) * scale
                )
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

    def to_dict(self) -> dict:
        """Convert LyricsSlice to dictionary format."""
        return {
            "words": [
                {"text": w.text, "start": w.start, "end": w.end} for w in self.words
            ],
            "raw_text": self.text,
        }


class Lyrics:
    def __init__(self, transcription: dict) -> None:
        self.raw_lyrics: list | None = None
        self.slices: list[LyricsSlice] = []
        self.words = []
        for i, seg in enumerate(transcription["segments"]):
            for word in transcription["segments"][i]["words"]:
                if word["probability"] > 0.3:
                    self.words.append(
                        Word(
                            word["word"].strip(STRIP_CHARS), word["start"], word["end"]
                        )
                    )

    def __repr__(self) -> str:
        return " ".join([word.text for word in self.words])

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> LyricsSlice:
        return self.slices[index]

    def get_time_slice(self, start: float, end: float) -> LyricsSlice:
        result = [
            word for word in self.words if word.start >= start and word.end <= end
        ]
        return LyricsSlice(result)

    def get_words_pos_slice(self, start: int, end: int) -> LyricsSlice:
        result = self.words[start:end]
        return LyricsSlice(result)

    def get_word_at_time(self, time: float) -> Word:
        for word in self.words:
            if word.start <= time <= word.end:
                return word
        return None

    def get_text_slice(
        self, text: str, *, similarity_threshold: float, after: float = -1.0
    ) -> LyricsSlice:
        search_sentence = " ".join([w.strip(STRIP_CHARS).lower() for w in text.split()])
        sentence_len = len(search_sentence.split())
        print(f"Searching for: {search_sentence}")
        best_sentence = None
        best_similarity = 0.0

        start_idx = next(
            (i for i, word in enumerate(self.words) if word.start >= after), None
        )
        if start_idx is None or start_idx + sentence_len > len(self.words):
            return LyricsSlice([])
        try_number = 0
        while try_number < 2:
            for pos in range(start_idx, len(self.words) - sentence_len + 1):
                sentence = self.get_words_pos_slice(pos, pos + sentence_len)

                similarity = difflib.SequenceMatcher(
                    None, search_sentence, str(sentence).lower()
                ).ratio()

                if similarity == 1:
                    return LyricsSlice(sentence.words, text)

                if similarity > best_similarity:
                    best_sentence = sentence
                    best_similarity = similarity

            if best_similarity >= similarity_threshold:
                return LyricsSlice(best_sentence.words, text)
            try_number += 1
            start_idx = 0

        return LyricsSlice([])

    def create_lyric_slices(self, lyrics: str, *, similarity_threshold: float = 0.8):
        self.raw_lyrics = lyrics.splitlines()
        self.slices = []
        last_slice_end = 0.0
        for line in self.raw_lyrics:
            slice = self.get_text_slice(
                line, after=last_slice_end, similarity_threshold=similarity_threshold
            )
            self.slices.append(slice)
            last_slice_end = slice.end

        return self.slices

    def check_time_overlap(self, slice1, slice2) -> bool:
        if slice1.end > slice2.start and slice1.start < slice2.end:
            return True
        else:
            return False

    def check_time_continuation(self, slice1, slice2) -> bool:
        # if not slice1.words or not slice2.words:
        #     return True  # Consider empty slices as valid continuations
        if slice1.end <= slice2.start:
            return True
        else:
            return False

    def check_alignment(self):
        for i in range(len(self.slices) - 1):
            if self.check_time_overlap(self.slices[i], self.slices[i + 1]):
                print("Overlap detected:", self.slices[i], self.slices[i + 1])
                return False
            if not self.check_time_continuation(self.slices[i], self.slices[i + 1]):
                print("Continuation problem:", self.slices[i], self.slices[i + 1])
                return False
        return True

    # def fix_alignment(
    #     self, max_try: int = 100, min_gap: float = 0.1, max_stretch: float = 0.8
    # ):
    #     # FIXME: this implementation is not working properly
    #     if not self.slices:
    #         return self.slices

    #     try_num = 0
    #     while not self.check_alignment() and try_num < max_try:
    #         try_num += 1
    #         modified = False

    #         # First pass: Fix major overlaps and gaps
    #         for i in range(len(self.slices) - 1):
    #             pass
    #     return self.slices


class LyricsAligner:
    def __init__(self, audio_path: Path) -> None:
        self.audio_file = audio_path
        self.cache_dir = CACHE_DIR / self.audio_file.stem
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vocals_path = self.cache_dir / f"{self.audio_file.stem}.vocals.mp3"
        self.instr_path = self.cache_dir / f"{self.audio_file.stem}.instr.mp3"
        self.lyrics_path = self.cache_dir / f"{self.audio_file.stem}.lyrics.txt"
        self.transcription_path = (
            self.cache_dir / f"{self.audio_file.stem}.transcription.json"
        )
        self.aligned_lyrics_path = (
            self.cache_dir / f"{self.audio_file.stem}.aligned_lyrics.json"
        )
        self.transcription = None

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
        lyrics = re.sub(r"[\{\[\(].*?[\}\]\)]", "", lyrics)
        return lyrics

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
            return None

        print("Transcribing audio...")
        self.transcription = self.transcribe_audio()

        print("Aligning lyrics...")
        lyrics = Lyrics(self.transcription)
        lyrics.create_lyric_slices(raw_lyrics)
        aligned = self.fix_alignment(lyrics)
        self.save_alignment(aligned, self.aligned_lyrics_path)
        return aligned

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
                vocals = torch.clamp(
                    vocals * 1.5, -1, 1
                )  # Increase vocal presence safely
                vocals = (
                    vocals + torch.clamp(vocals - vocals.roll(1, -1), -0.5, 0.5) * 0.3
                )  # Enhance clarity
                vocals = (
                    vocals + torch.clamp(vocals - vocals.roll(2, -1), -0.2, 0.2) * 0.1
                )  # Enhance harmonicity

            finally:
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if torch.cuda.is_available():
                vocals = vocals.cpu()
                instrumental = instrumental.cpu()

            print(
                f"Saving vocals shape: {vocals.shape}, instrumental shape: {instrumental.shape}"
            )
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

    def fix_alignment(self, lyrics: Lyrics, max_try_num: int = 10):
        try_num = 0
        while not lyrics.check_alignment() and try_num < max_try_num:
            for i in range(len(lyrics) - 1):
                if lyrics.check_time_overlap(lyrics[i], lyrics[i + 1]):
                    gap = abs(lyrics[i + 1].start - lyrics[i].end)
                    print(f"{lyrics[i].duration=} \t{lyrics[i + 1].duration=} \t{gap=}")
                    lyrics[i].stretch(gap / 2)
                    lyrics[i + 1].stretch(gap / 2, back=True)
                    print(f"{lyrics[i].duration=} \t{lyrics[i + 1].duration=}")
                #  FIXME: check for empty lines
                if not lyrics.check_time_continuation(lyrics[i], lyrics[i + 1]):
                    lyrics[i + 1].rebase(lyrics[i].end + 0.1)
                    print(f"{str(lyrics[i])=} \t{str(lyrics[i + 1])=}")
                    print(f"{lyrics[i].end=} \t{lyrics[i + 1].start=}")

            try_num += 1
        return lyrics

    def save_alignment(self, aligned: Lyrics, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([slice.to_dict() for slice in aligned], f, indent=2)
