import pytest

from course_copilot.ingestion import split_text


def test_split_text_normalizes_whitespace_and_preserves_text():
    assert split_text("  alpha\n beta\t gamma  ") == ["alpha beta gamma"]


def test_split_text_chunks_long_inputs_with_overlap():
    text = " ".join(f"token{index}" for index in range(250))
    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    overlap = max(
        count
        for count in range(1, min(len(chunks[0]), len(chunks[1])) + 1)
        if chunks[0][-count:] == chunks[1][:count]
    )
    assert overlap >= 100


def test_split_text_returns_no_chunks_for_empty_input():
    assert split_text(" \n\t ") == []
