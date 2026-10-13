"""Tests for conversation import/export parsing (Claude Code JSONL and JSON arrays)."""

import json

import pytest

from tgagent.conversation_io import (
    MAX_MESSAGES,
    ConversationIOError,
    messages_to_json,
    parse_claude_jsonl,
    parse_conversation,
    parse_json_messages,
)


def _cc_line(role: str, content, **extra) -> str:
    entry = {"type": role, "message": {"role": role, "content": content}}
    entry.update(extra)
    return json.dumps(entry, ensure_ascii=False)


def test_parse_claude_jsonl_keeps_dialogue_only():
    text = "\n".join(
        [
            json.dumps({"type": "mode", "mode": "normal"}),
            _cc_line("user", "Hello, help me with X"),
            _cc_line(
                "assistant",
                [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Sure!"}],
            ),
            _cc_line(
                "user", [{"type": "tool_result", "tool_use_id": "t1", "content": "raw output"}]
            ),
            _cc_line("assistant", [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]),
            _cc_line("assistant", [{"type": "text", "text": "Second part"}]),
            "{broken json",
            _cc_line("user", "<command-name>/plan</command-name><command-args>go</command-args>"),
            _cc_line("user", "sidechain chatter", isSidechain=True),
            _cc_line("user", "meta entry", isMeta=True),
        ]
    )
    # Consecutive assistant texts merge; tool/thinking/command/meta/sidechain lines drop.
    assert parse_claude_jsonl(text) == [
        ("user", "Hello, help me with X"),
        ("assistant", "Sure!\n\nSecond part"),
    ]


def test_parse_claude_jsonl_strips_system_reminders():
    text = _cc_line("user", "<system-reminder>noise</system-reminder>\n\nReal question?")
    assert parse_claude_jsonl(text) == [("user", "Real question?")]


def test_parse_claude_jsonl_merges_consecutive_roles():
    text = "\n".join(
        [
            _cc_line("user", "one"),
            _cc_line("user", "two"),
            _cc_line("assistant", "answer"),
        ]
    )
    assert parse_claude_jsonl(text) == [("user", "one\n\ntwo"), ("assistant", "answer")]


def test_parse_claude_jsonl_no_dialogue_raises():
    with pytest.raises(ConversationIOError):
        parse_claude_jsonl(json.dumps({"type": "mode", "mode": "plan"}))


def test_parse_claude_jsonl_caps_at_max_messages():
    extra_pairs = 100
    total_pairs = MAX_MESSAGES + extra_pairs
    lines = []
    for i in range(total_pairs):
        lines.append(_cc_line("user", f"msg {i}"))
        lines.append(_cc_line("assistant", f"reply {i}"))
    parsed = parse_claude_jsonl("\n".join(lines))
    assert len(parsed) == MAX_MESSAGES
    # The cap counts individual messages, so the newest MAX_MESSAGES survive:
    # 2 * total_pairs messages - MAX_MESSAGES kept = the tail starts mid-conversation.
    first_kept_pair = total_pairs - MAX_MESSAGES // 2
    assert parsed[0] == ("user", f"msg {first_kept_pair}")


def test_parse_json_messages_roundtrip():
    original = [("user", "hi"), ("assistant", "hello"), ("user", "bye")]
    serialized = messages_to_json(original)
    assert parse_json_messages(serialized) == original
    # Dispatcher picks the JSON-array branch automatically.
    assert parse_conversation(serialized) == original


def test_parse_json_messages_accepts_text_blocks():
    data = [{"role": "assistant", "content": [{"type": "text", "text": "block reply"}]}]
    assert parse_json_messages(json.dumps(data)) == [("assistant", "block reply")]


def test_parse_errors():
    with pytest.raises(ConversationIOError):
        parse_conversation("")
    with pytest.raises(ConversationIOError):
        parse_conversation("just some plain text file")
    with pytest.raises(ConversationIOError):
        parse_json_messages('{"role": "user", "content": "not a list"}')
    with pytest.raises(ConversationIOError):
        parse_json_messages(json.dumps([{"role": "system", "content": "bad role"}]))
    with pytest.raises(ConversationIOError):
        parse_json_messages(json.dumps([{"role": "user", "content": ""}]))
