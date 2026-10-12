"""Tests for the agent tool-calling loop, using a fake GLM client."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tgagent.agent import Agent
from tgagent.history import History


class FakeGLM:
    """Duck-typed stand-in for GLMClient that replays scripted responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict], list | None]] = []

    async def chat(self, messages, *, model=None, tools=None, max_tokens=4000):
        self.calls.append(([dict(m) for m in messages], tools))
        return self._responses.pop(0)


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str, arguments: dict | str):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=raw))


async def test_tool_round_trip(tmp_path: Path, make_cfg):
    cfg = make_cfg(max_tool_iters=4)
    fake = FakeGLM(
        [
            _message(
                tool_calls=[
                    _tool_call(
                        "c1", "write_file", {"path": "scripts/hi.py", "content": "print('hi')"}
                    )
                ]
            ),
            _message(tool_calls=[_tool_call("c2", "run_python", {"path": "scripts/hi.py"})]),
            _message(content="The script ran and printed 'hi'."),
        ]
    )
    history = History(tmp_path / "bot.db")
    agent = Agent(cfg, fake, history)
    try:
        out = await agent.handle_turn(1, "write and run a script that prints hi")
    finally:
        history.close()

    assert out == "The script ran and printed 'hi'."
    assert (cfg.workspace_dir / "scripts" / "hi.py").exists()
    # Second model call saw the tool result for the write, third saw the run output.
    second_messages, _ = fake.calls[1]
    write_result = next(m for m in second_messages if m["role"] == "tool")
    assert "wrote" in write_result["content"]
    third_messages, _ = fake.calls[2]
    run_result = [m for m in third_messages if m["role"] == "tool"][-1]
    assert "exit_code=0" in run_result["content"]


async def test_invalid_json_arguments(tmp_path: Path, make_cfg):
    cfg = make_cfg(max_tool_iters=4)
    fake = FakeGLM(
        [
            _message(tool_calls=[_tool_call("c1", "read_file", '{"path": broken json')]),
            _message(content="Recovered."),
        ]
    )
    history = History(tmp_path / "bot.db")
    agent = Agent(cfg, fake, history)
    try:
        out = await agent.handle_turn(1, "read something")
    finally:
        history.close()

    assert out == "Recovered."
    second_messages, _ = fake.calls[1]
    tool_result = next(m for m in second_messages if m["role"] == "tool")
    assert "ERROR" in tool_result["content"]


async def test_max_iterations_returns_notice(tmp_path: Path, make_cfg):
    cfg = make_cfg(max_tool_iters=2)
    # The model never stops asking for tools.
    endless_tool_call = _message(tool_calls=[_tool_call("c1", "list_files", {})])
    fake = FakeGLM([endless_tool_call, endless_tool_call, endless_tool_call])
    history = History(tmp_path / "bot.db")
    agent = Agent(cfg, fake, history)
    try:
        out = await agent.handle_turn(1, "loop forever")
    finally:
        history.close()

    assert "tool-use limit" in out
    # The GLM client was called exactly max_tool_iters times.
    assert len(fake.calls) == 2


async def test_plain_answer_without_tools(tmp_path: Path, make_cfg):
    cfg = make_cfg()
    fake = FakeGLM([_message(content="Just an answer.")])
    history = History(tmp_path / "bot.db")
    agent = Agent(cfg, fake, history)
    try:
        out = await agent.handle_turn(1, "hi")
    finally:
        history.close()

    assert out == "Just an answer."
    assert len(fake.calls) == 1
