import pytest
from pathlib import Path
from lyrics_aligner import Lyrics, LyricsSlice, Word


@pytest.fixture
def sample_words():
    return [
        Word("hello", 0.0, 0.5),
        Word("world", 0.6, 1.0),
        Word("this", 1.2, 1.5),
        Word("is", 1.6, 1.8),
        Word("a", 1.9, 2.0),
        Word("test", 2.1, 2.5),
    ]


@pytest.fixture
def lyrics(sample_words):
    mock_transcription = {
        "segments": [
            {
                "words": [
                    {
                        "word": word.text,
                        "start": word.start,
                        "end": word.end,
                        "probability": 1.0,
                    }
                    for word in sample_words
                ]
            }
        ]
    }
    lyrics = Lyrics(mock_transcription)
    return lyrics


class TestWord:
    def test_word_creation(self):
        word = Word("test", 1.0, 2.0)
        assert word.text == "test"
        assert word.start == 1.0
        assert word.end == 2.0

    def test_word_duration(self):
        word = Word("test", 1.0, 2.0)
        assert word.duration == 1.0

    def test_word_str_representation(self):
        word = Word("test", 1.0, 2.0)
        assert str(word) == "test"


class TestLyricsSlice:
    def test_empty_slice(self):
        slice = LyricsSlice([])
        assert slice.words == []
        assert slice.text == None
        assert slice.start == 0
        assert slice.end == 0

    def test_slice_with_words(self, sample_words):
        slice = LyricsSlice(sample_words[:2])  # "hello world"
        assert len(slice.words) == 2
        assert str(slice) == "hello world"
        assert slice.start == 0.0
        assert slice.end == 1.0

    def test_slice_with_raw_text(self, sample_words):
        slice = LyricsSlice(sample_words[:2], "custom text")
        assert len(slice.words) == 2
        assert str(slice) == "hello world"
        assert slice.text == "custom text"
        assert slice.start == 0.0
        assert slice.end == 1.0

    def test_slice_with_custom_text(self, sample_words):
        slice = LyricsSlice(sample_words[:2], "custom text")
        assert slice.text == "custom text"

    def test_slice_shift(self, sample_words):
        slice = LyricsSlice(sample_words[:2])
        original_start = slice.start
        original_end = slice.end
        shift_amount = 1.0

        slice.shift(shift_amount)
        assert slice.start == original_start + shift_amount
        assert slice.end == original_end + shift_amount

    def test_slice_stretch(self, sample_words):
        slice = LyricsSlice(sample_words[:2])
        original_start = slice.start
        original_end = slice.end

        stretch_amount = 1.5
        slice.stretch(stretch_amount)
        assert slice.start == original_start
        assert slice.end == original_end + stretch_amount

        stretch_amount = -1.5
        slice.stretch(stretch_amount)
        assert slice.start == original_start
        assert slice.end == original_end

    def test_slice_stretch_backward(self, sample_words):
        slice = LyricsSlice(sample_words[:2])
        original_start = slice.start
        original_end = slice.end

        stretch_amount = 1.5
        slice.stretch(stretch_amount, backward=True)
        assert slice.start == original_start - stretch_amount
        assert slice.end == original_end

        stretch_amount = -1.5
        slice.stretch(stretch_amount, backward=True)
        assert slice.start == original_start
        assert slice.end == original_end

    def test_slice_rebase(self, sample_words):
        slice = LyricsSlice(sample_words[2:5])
        original_start = slice.start
        original_end = slice.end

        new_start = 0
        slice.rebase(new_start)
        assert slice.start == new_start
        assert slice.end == original_end + (new_start - original_start)

        new_start = original_start
        slice.rebase(new_start)
        assert slice.start == original_start
        assert slice.end == original_end


class TestLyrics:
    def test_get_word_at_time(self, lyrics):
        # Test exact match
        word = lyrics.get_word_at_time(0.3)
        assert word.text == "hello"

        # Test no word at time
        word = lyrics.get_word_at_time(0.55)
        assert word is None

    def test_get_words_pos_slice(self, lyrics):
        # Test exact match
        slice = lyrics.get_words_pos_slice(0, 2)
        assert len(slice.words) == 2
        assert slice.words[0].text == "hello"
        assert slice.words[1].text == "world"

        # Test middle match
        slice = lyrics.get_words_pos_slice(2, 4)
        assert len(slice.words) == 2
        assert slice.words[0].text == "this"
        assert slice.words[1].text == "is"

        # Test no match
        slice = lyrics.get_words_pos_slice(10, 11)
        assert len(slice.words) == 0

    def test_get_text_slice(self, lyrics):
        # Test exact match
        slice = lyrics.get_text_slice("Hello world", similarity_threshold=1.0)
        assert str(slice) == "hello world"
        assert slice.text == "Hello world"
        assert len(slice.words) == 2

        # Test partial match with threshold
        slice = lyrics.get_text_slice("Helo wrld", similarity_threshold=0.5)
        assert len(slice.words) == 2
        assert slice.words[0].text == "hello"
        assert slice.words[1].text == "world"

        # Test no match with threshold
        slice = lyrics.get_text_slice("helo wrld", similarity_threshold=1.0)
        assert slice is None

        # Test match after specific time
        slice = lyrics.get_text_slice("this is", similarity_threshold=1.0, after=1.0)
        assert len(slice.words) == 2
        assert slice.words[0].text == "this"
        assert slice.words[1].text == "is"

        # Test match after specific time
        slice = lyrics.get_text_slice(
            "this is test", similarity_threshold=0.7, after=1.0
        )
        assert len(slice.words) == 3
        assert slice.words[0].text == "this"
        assert slice.words[1].text == "is"

    def test_get_slice_at_time(self, lyrics):
        # Test exact time range
        slice = lyrics.get_time_slice(0.0, 1.0)
        assert len(slice.words) == 2
        assert slice.words[0].text == "hello"
        assert slice.words[1].text == "world"

        # Test partial time range
        slice = lyrics.get_time_slice(0.7, 1.3)
        assert len(slice.words) == 0

        # Test no match with threshold
        slice = lyrics.get_time_slice(0.5, 1.5)
        assert len(slice.words) == 2
        assert slice.words[0].text == "world"
        assert slice.words[1].text == "this"

    def test_create_lyric_slices(self, lyrics):
        lyrics_text = "hello world\nthis is a test"
        lyrics.create_lyric_slices(lyrics_text)

        assert len(lyrics.slices) == 2
        assert lyrics.slices[0].text == "hello world"
        assert len(lyrics.slices[0].words) == 2
        assert lyrics.slices[1].text == "this is a test"
        assert len(lyrics.slices[1].words) == 4

    def test_check_alignment(self, lyrics):
        # Create two non-overlapping slices
        slice1 = LyricsSlice([Word("hello", 0.0, 0.5)])
        slice2 = LyricsSlice([Word("world", 0.6, 1.0)])
        lyrics.slices = [slice1, slice2]

        assert lyrics.check_alignment() is True

        # Create overlapping slices
        slice1 = LyricsSlice([Word("hello", 0.0, 0.7)])
        slice2 = LyricsSlice([Word("world", 0.6, 1.0)])
        lyrics.slices = [slice1, slice2]

        assert lyrics.check_alignment() is False


if __name__ == "__main__":
    pytest.main([__file__])
