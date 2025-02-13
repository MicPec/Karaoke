import os
from pathlib import Path
import time
import threading
from lyrics_aligner import LyricsAligner, Word, LyricsSlice
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
from typing import List, Optional


class KaraokePlayer:
    def __init__(self, audio_file: str):
        self.audio_file = Path(audio_file)
        self.cache_dir = Path("cache") / self.audio_file.stem
        os.makedirs(self.cache_dir, exist_ok=True)
        self.aligner = LyricsAligner(self.audio_file)

    def process_audio(self, force: bool = False) -> Optional[List[LyricsSlice]]:
        """Process audio and get aligned lyrics using LyricsAligner."""
        if not force:
            # Try to load existing lyrics first
            lyrics_data = self.aligner.get_aligned_lyrics()

            if lyrics_data is not None:
                return [LyricsSlice.from_dict(slice_data) for slice_data in lyrics_data]

        print("Processing audio and aligning lyrics...")
        lyrics_data = self.aligner.process_audio(overwrite=force)
        if lyrics_data:
            return lyrics_data  # Already LyricsSlice objects
        return None

    def play_audio(self):
        """Load and mix the audio files for playback."""
        # Get paths from aligner
        vocals_path = self.aligner.get_vocals()
        instr_path = self.aligner.get_instr()

        vocals = AudioSegment.from_mp3(str(vocals_path))
        instrumental = AudioSegment.from_mp3(str(instr_path))

        vocals = vocals - 15  # Reduce vocals
        mixed = instrumental.overlay(vocals)

        # Play with pydub
        play(mixed)

    def play_karaoke(self, aligned_lyrics: List[LyricsSlice]):
        """Play audio and display synchronized lyrics."""
        # Play audio in a separate thread
        threading.Thread(target=self.play_audio, daemon=True).start()
        start_time = time.time()

        try:
            while True:
                current_time = time.time() - start_time

                # Find current, previous and next lyrics
                prev_line = curr_line = next_line = None
                curr_idx = -1

                # Find current line
                for i, slice in enumerate(aligned_lyrics):
                    if not slice.words:  # Skip empty slices
                        continue
                    if slice.start <= current_time <= slice.end:
                        curr_line = slice.text
                        curr_idx = i
                        break

                # Find previous and next lines
                if curr_idx >= 0:
                    prev_line = (
                        aligned_lyrics[curr_idx - 1].text
                        if aligned_lyrics[curr_idx - 1].text
                        else aligned_lyrics[curr_idx].text
                    )
                    next_line = (
                        aligned_lyrics[curr_idx + 1].text
                        if curr_idx + 1 < len(aligned_lyrics)
                        and aligned_lyrics[curr_idx + 1].text
                        else aligned_lyrics[curr_idx].text
                    )
                # Clear screen and display lyrics
                # raise Exception()
                print("\033[H\033[J", end="")  # Clear terminal
                print(f"Time: {current_time:.1f}s")
                print("\nLyrics:")
                print("-" * 50)

                if prev_line:
                    print(f"Previous: {prev_line}")
                if curr_line:
                    print(f"\033[1m>>> {curr_line} <<<\033[0m")
                if next_line:
                    print(f"Next: {next_line}")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nPlayback stopped")
        finally:
            # Clean up temporary file
            temp_wav = self.cache_dir / "temp_mixed.wav"
            if temp_wav.exists():
                temp_wav.unlink()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple Karaoke Player")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force audio transcription even if it exists",
    )
    args = parser.parse_args()

    try:
        # Initialize player and process audio
        player = KaraokePlayer(args.audio)
        aligned_lyrics = player.process_audio(force=args.force)

        if not aligned_lyrics:
            print("No lyrics found or processing failed")
            return

        # Show lyrics preview
        print("\nAligned Lyrics Preview:")
        print("-" * 50)
        for slice in aligned_lyrics:
            if slice.words:
                start = slice.words[0].start
                end = slice.words[-1].end
                print(f"[{start:.1f}s - {end:.1f}s] {slice}")

        # Start karaoke mode
        print("\nStarting karaoke mode... (Press Ctrl+C to exit)")
        player.play_karaoke(aligned_lyrics)

    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    # except Exception as e:
    #     print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
