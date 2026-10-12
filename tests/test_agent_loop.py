"""Tests for the agent tool-calling loop, using a fake GLM client."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgagent.agent import Agent, AgentError
from tgagent.glm_client import GLMError
from tgagent.history import History


class FakeGLM:
    """Duck-typed stand-in for GLMClient that replays scripted responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict], str | None, list | None]] = []
        self.vision_calls: list[tuple[str, list[dict], list[str]]] = []

    async def chat(self, messages, *, model=None, tools=None, max_tokens=4000):
        self.calls.append(([dict(m) for m in messages], model, tools))
        return self._responses.pop(0)

    async def vision_chat(self, system, messages, image_data_urls):
        self.vision_calls.append((system, [dict(m) for m in messages], list(image_data_urls)))
        return "A blue rectangle."


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
    second_messages, _, _ = fake.calls[1]
    write_result = next(m for m in second_messages if m["role"] == "tool")
    assert "wrote" in write_result["content"]
    third_messages, _, _ = fake.calls[2]
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
    second_messages, _, _ = fake.calls[1]
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


async def test_image_turn_uses_vision_chat(tmp_path: Path, make_cfg):
    from PIL import Image

    cfg = make_cfg()
    image = cfg.workspace_dir / "photo.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 40), (10, 120, 200)).save(image, format="PNG")

    fake = FakeGLM([])
    history = History(tmp_path / "bot.db")
    agent = Agent(cfg, fake, history)
    try:
        out = await agent.handle_turn(1, "what is this?", image_paths=[image])
    finally:
        history.close()

    assert out == "A blue rectangle."
    # The text/tool endpoint was not used for the image turn.
    assert fake.calls == []
    assert len(fake.vision_calls) == 1
    system, messages, urls = fake.vision_calls[0]
    assert system  # non-empty system prompt
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "what is this?"
    assert len(urls) == 1
    assert urls[0].startswith("data:image/jpeg;base64,")


class _RaisingGLM:
    def __init__(self, error: GLMError) -> None:
        self._error = error

    async def chat(self, messages, *, model=None, tools=None, max_tokens=4000):
        raise self._error


async def test_balance_error_gets_friendly_message(tmp_path: Path, make_cfg):
    glm = _RaisingGLM(
        GLMError(
            "Error code: 429 - {'error': {'code': '1113',"
            " 'message': 'Insufficient balance or no resource package. Please recharge.'}}"
        )
    )
    history = History(tmp_path / "bot.db")
    agent = Agent(make_cfg(), glm, history)
    try:
        with pytest.raises(AgentError, match="balance"):
            await agent.handle_turn(1, "hello")
    finally:
        history.close()


async def test_window_limit_gets_friendly_message(tmp_path: Path, make_cfg):
    glm = _RaisingGLM(GLMError("Error code: 429 - usage limit reached for the 5-hour window"))
    history = History(tmp_path / "bot.db")
    agent = Agent(make_cfg(), glm, history)
    try:
        with pytest.raises(AgentError, match="5-hour window"):
            await agent.handle_turn(1, "hello")
    finally:
        history.close()
