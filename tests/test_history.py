"""Tests for the SQLite history store and context building."""

import pytest

from tgagent.history import History


@pytest.fixture
def history(tmp_path):
    h = History(tmp_path / "bot.db")
    yield h
    h.close()


def test_append_and_build_context(history):
    history.append(1, "user", "hello")
    history.append(1, "assistant", "hi there")
    messages = history.build_context(1, "SYS", 10_000)
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"]
    assert messages[1]["content"] == "hello"


def test_budget_trims_oldest_turns(history):
    for i in range(10):
        history.append(1, "user", f"message number {i} " + "x" * 40)
        history.append(1, "assistant", "ack " + "y" * 10)
    messages = history.build_context(1, "SYS", budget_tokens=60)
    roles = [m["role"] for m in messages[1:]]
    assert len(roles) < 20  # some turns were dropped
    # the newest turn (user message + reply) is always kept
    assert any("message number 9" in m["content"] for m in messages)


def test_reset_clears_history(history):
    history.append(1, "user", "hello")
    history.reset(1)
    assert history.build_context(1, "SYS", 100) == [{"role": "system", "content": "SYS"}]


def test_chats_are_isolated(history):
    history.append(1, "user", "for chat one")
    history.append(2, "user", "for chat two")
    messages = history.build_context(1, "SYS", 10_000)
    assert len(messages) == 2
    assert messages[1]["content"] == "for chat one"


def test_stats(history):
    history.append(1, "user", "hello world")  # est 2 tokens
    stats = history.stats(1)
    assert stats["messages"] == 1
    assert stats["est_tokens"] == 2
