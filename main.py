import os
from pathlib import Path
import time
import threading
from lyrics_aligner import LyricsAligner, Word, Segment, MatchType, download_lyrics
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
from typing import List, Optional
from tinytag import TinyTag


class KaraokePlayer:
    def __init__(
        self,
        audio_file: str,
        song_title: Optional[str] = None,
        song_author: Optional[str] = None,
    ):
        self.audio_file = Path(audio_file)
        self.cache_dir = Path("cache") / self.audio_file.stem
        os.makedirs(self.cache_dir, exist_ok=True)
        self.aligner = LyricsAligner(self.audio_file)
        self.song_title = song_title or self.audio_file.stem
        self.song_author = song_author

    def process_audio(self, force: bool = False) -> list[Segment] | None:
        """Process audio and get aligned lyrics using LyricsAligner."""
        if self.aligner.get_lyrics(self.aligner.lyrics_path) is None or force:
            print("Downloading lyrics...")
            lyrics = download_lyrics(self.song_title, self.song_author or "")
            with open(self.aligner.lyrics_path, "w") as f:
                f.write(lyrics)

        if not force:
            # Try to load existing lyrics first
            lyrics_data = self.aligner.load_alignment(self.aligner.aligned_lyrics_path)

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

        vocals = vocals - 12
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
                    if segment.start <= current_time <= segment.end and segment != last_displayed:
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
    parser.add_argument(
        "--title",
        help="Song title (if different from audio filename)",
    )
    parser.add_argument(
        "--author",
        help="Song author/artist name",
    )
    args = parser.parse_args()

    try:
        # Initialize player and process audio
        audio_tags = TinyTag.get(args.audio)

        title = audio_tags.title if args.title is None else args.title
        author = audio_tags.artist if args.author is None else args.author

        player = KaraokePlayer(args.audio, title, author)
        aligned_lyrics = player.process_audio(force=args.force)

        if not aligned_lyrics:
            print("No lyrics found or processing failed")
            return

        # Show lyrics preview
        print("\nAligned Lyrics Preview:")
        print("-" * 50)
        for segment in aligned_lyrics:
            start = segment.start
            end = segment.end
            print(f"[{start:.1f}s - {end:.1f}s] {segment.text}")

        # Start karaoke mode
        print("\nStarting karaoke mode... (Press Ctrl+C to exit)")
        player.play_karaoke(aligned_lyrics)

    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    # except Exception as e:
    #     print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
