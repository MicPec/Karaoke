import os
from pathlib import Path
import time
import threading
from lyrics_aligner import LyricsAligner, Word, Segment, MatchType
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

    def process_audio(self, force: bool = False) -> list[Segment] | None:
        """Process audio and get aligned lyrics using LyricsAligner."""
        if not force:
            # Try to load existing lyrics first
            lyrics_data = self.aligner.load_alignment()

            if lyrics_data is not None:
                return lyrics_data

        print("Processing audio and aligning lyrics...")
        lyrics_data = self.aligner.process_audio(overwrite=force)
        return lyrics_data

    def play_audio(self):
        """Play the mixed audio file."""
        # Load audio files
        vocals = AudioSegment.from_file(self.aligner.get_vocals())
        instrumental = AudioSegment.from_file(self.aligner.get_instr())

        # Mix vocals and instrumental
        mixed = vocals.overlay(instrumental)

        # Play with pydub
        play(mixed)

    def play_karaoke(self, aligned_lyrics: list[Segment]):
        """Play audio and display synchronized lyrics."""
        # Play audio in a separate thread
        threading.Thread(target=self.play_audio, daemon=True).start()

        # Display lyrics
        start_time = time.time()
        last_displayed = None

        try:
            while True:
                current_time = time.time() - start_time
                for segment in aligned_lyrics:
                    if (
                        segment.start <= current_time <= segment.end
                        and segment != last_displayed
                    ):
                        print(f"\033[K{segment}")  # Clear line and print lyrics
                        last_displayed = segment
                        break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped playback.")


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
