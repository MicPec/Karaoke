import pytest
from pathlib import Path
from lyrics_aligner import Lyrics, Segment, Word, MatchType


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
                "text": " ".join(word.text for word in sample_words),
                "words": [
                    {
                        "word": word.text,
                        "start": word.start,
                        "end": word.end,
                        "probability": 1.0,
                    }
                    for word in sample_words
                ],
            }
        ]
    }
    print(f"Creating lyrics with transcription: {mock_transcription}")
    lyrics = Lyrics(mock_transcription)
    print(f"Created lyrics: {lyrics}")
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


class TestSegment:
    def test_empty_segment(self):
        segment = Segment(text="", words=[], match_type=MatchType.NONE)
        assert segment.words == []
        assert segment.text == ""
        assert segment.start == 0
        assert segment.end == 0

    def test_segment_with_words(self, sample_words):
        segment = Segment(
            text="hello world", words=sample_words[:2], match_type=MatchType.NONE
        )
        assert len(segment.words) == 2
        assert str(segment) == "hello world"
        assert segment.start == 0.0
        assert segment.end == 1.0

    def test_segment_match_type(self, sample_words):
        segment = Segment(
            text="hello world", words=sample_words[:2], match_type=MatchType.DIRECT
        )
        assert segment.match_type == MatchType.DIRECT

    def test_segment_match_confidence(self, sample_words):
        segment = Segment(
            text="hello world",
            words=sample_words[:2],
            match_type=MatchType.SIMILAR,
            match_confidence=0.95,
        )
        assert segment.match_confidence == 0.95

    def test_segment_split(self, sample_words):
        segment = Segment(
            text="Hello world this is",
            words=sample_words[:4],  # Hello world this is
            match_type=MatchType.NONE,
        )
        # Split in the middle
        new_segment = segment.split(2)  # Split after "world"
        assert len(new_segment.words) == 2
        assert str(new_segment) == "this is"
        assert len(segment.words) == 2
        assert str(segment) == "hello world"


class TestLyrics:
    def test_lyrics_initialization(self, lyrics):
        assert len(lyrics.segments) > 0
        assert isinstance(lyrics.segments[0], Segment)

    def test_lyrics_str_representation(self, lyrics):
        assert isinstance(str(lyrics), str)
        assert len(str(lyrics)) > 0

    def test_lyrics_len(self, lyrics):
        assert len(lyrics) == len(lyrics.segments)

    def test_lyrics_getitem(self, lyrics):
        assert isinstance(lyrics[0], Segment)

    def test_get_segment_by_time(self, lyrics):
        segment = lyrics.get_segment_by_time(0.3)
        assert isinstance(segment, Segment)
        assert segment.start <= 0.3 <= segment.end

    def test_lyrics_split_segment(self, lyrics):
        # Get the first segment which should contain all sample words
        original_segment = lyrics.segments[0]
        original_len = len(lyrics.segments)

        # Test splitting in the middle
        segments = lyrics.split_segment(original_segment, start=2, end=5)
        print([str(segment) for segment in lyrics.segments])
        assert len(segments) == 3
        assert len(lyrics.segments) == original_len + 2
        assert segments[0].text == "hello world"
        assert segments[1].text == "this is a"
        assert segments[2].text == "test"
        assert lyrics.segments[0].text == "hello world"
        assert lyrics.segments[1].text == "this is a"
        assert lyrics.segments[2].text == "test"

        # Test splitting at start
        original_segment = lyrics.segments[0]  # "hello world"
        segments = lyrics.split_segment(original_segment, start=0, end=1)
        print([str(segment) for segment in lyrics.segments])
        assert len(segments) == 2
        assert len(lyrics.segments) == original_len + 2 + 1
        assert segments[0].text == "hello"
        assert segments[1].text == "world"
        assert lyrics.segments[0].text == "hello"
        assert lyrics.segments[1].text == "world"
        assert lyrics.segments[2].text == "this is a"
        assert lyrics.segments[3].text == "test"

        # Test splitting at end
        original_segment = lyrics.segments[2]  # "this is a"
        print([str(segment) for segment in lyrics.segments])
        segments = lyrics.split_segment(original_segment, start=1, end=4)
        assert len(segments) == 2
        assert len(lyrics.segments) == original_len + 2 + 1 + 1
        assert segments[0].text == "this"
        assert segments[1].text == "is a"
        assert lyrics.segments[0].text == "hello"
        assert lyrics.segments[1].text == "world"
        assert lyrics.segments[2].text == "this"
        assert lyrics.segments[3].text == "is a"
        assert lyrics.segments[4].text == "test"

        # Test no split case (when start=0 and end=len)
        no_split = lyrics.split_segment(segments[0], start=0, end=2)
        assert len(no_split) == 1
        assert no_split[0].text == "this"
        assert len(lyrics.segments) == original_len + 2 + 1 + 1


if __name__ == "__main__":
    pytest.main([__file__])
