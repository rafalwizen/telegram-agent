"""Tests for the Telegram message splitter and token estimation."""

from tgagent.splitter import est_tokens, split_for_telegram


def test_short_text_is_single_chunk():
    assert split_for_telegram("hello world") == ["hello world"]


def test_empty_text_returns_no_chunks():
    assert split_for_telegram("   ") == []


def test_long_text_is_split_within_limit():
    text = "This is a sentence about something. " * 100  # ~3500*... chars
    chunks = split_for_telegram(text, limit=100)
    assert len(chunks) > 1
    # fence padding can add a few chars; keep well below the hard 4096 limit
    assert all(len(c) <= 110 for c in chunks)


def test_split_prefers_paragraph_boundaries():
    paragraphs = ["Paragraph one.\n" + "a" * 50, "Paragraph two.\n" + "b" * 50]
    text = "\n\n".join(paragraphs)
    chunks = split_for_telegram(text, limit=70)
    assert len(chunks) == 2
    assert chunks[0].startswith("Paragraph one.")


def test_code_fences_stay_balanced():
    text = "intro\n```python\n" + "x = 1\n" * 60 + "```\nsummary"
    chunks = split_for_telegram(text, limit=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0


def test_est_tokens_latin():
    assert est_tokens("hello world") == 2  # 11 chars // 4
    assert est_tokens("") == 0
