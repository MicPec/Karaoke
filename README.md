# Karaoke Player

(WORK IN PROGRESS)

A Python-based Karaoke player that automatically aligns lyrics with audio and provides a synchronized karaoke experience.

## Features

- Automatic lyrics alignment with audio
- Vocal and instrumental track separation
- Real-time lyrics display synchronized with music
- Support for custom lyrics and song metadata
- Caching system for processed audio and alignments

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MicPec/Karaoke.git
cd Karaoke
```

2. Install dependencies using `uv`:

```bash
uv sync
uv pip install -r pyproject.toml
```

3. Create a `.env` file and add GENIUS_API_KEY:

```bash
GENIUS_API_KEY = YOUR_GENIUS_API_KEY
```

The key can be obtained from the [Genius API](https://genius.com/api-clients/new) website.

## Usage

Run the karaoke player with an audio file:

```bash
uv run main.py --author [author] --title [title] path/to/audio.mp3
```

### Optional Arguments

- `--force`: Force audio transcription even if cached version exists
- `--title`: Specify song title (if different from audio filename)
- `--author`: Specify song author/artist name

Example:

```bash
uv run main.py mysong.mp3 --title "My Favorite Song" --author "Famous Artist"
```

## How It Works

1. **Audio Processing**:

   - Separates vocals from instrumental track
   - Transcribes vocals to text
   - Aligns transcribed text with provided lyrics

2. **Lyrics Alignment**:

   - Uses advanced text alignment algorithms
   - Matches transcribed words with provided lyrics
   - Creates time-stamped lyric segments

3. **Playback**:
   - Plays audio in a separate thread
   - Displays synchronized lyrics in real-time

## Testing

Run tests using pytest:

```bash
uv run -m pytest
```

## Project Structure

- `main.py`: Main application entry point and KaraokePlayer class
- `lyrics_aligner.py`: Core lyrics alignment functionality
- `tests/`: Test suite directory

## Cache Management

Processed audio files and alignments are cached in the `cache/` directory to improve performance on subsequent runs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
