# telegram-agent

Personal Telegram AI assistant powered by GLM (Zhipu / Z.ai), running on your own VPS
in Docker. You chat with it like with any AI chat app; it accepts text, photos and
documents (PDF/DOCX/XLSX/CSV/…). The agent has a private **workspace** where your
files land and where it can write and run its own Python scripts (pandas, matplotlib
preinstalled) inside a sandbox with time/RAM limits and no access to secrets.

Runs entirely on a **GLM Coding Plan** (e.g. Lite) — the same subscription and
5-hour windows you use with Claude Code; no pay-as-you-go balance required.

## Features

- Telegram long-polling — no open inbound ports, no TLS certificates
- Single-user by design: only whitelisted Telegram user ids can talk to the bot
- Persistent conversation history per chat (SQLite) with token-budget context trimming
- Tool-calling agent: `list_files`, `read_file`, `write_file`, `extract_text`, `run_python`
- Images analyzed through the Anthropic-compatible endpoint of the Coding Plan
- Conversation import/export: continue a Claude Code session inside Telegram (`/import`),
  back up or move a bot conversation anywhere (`/export`)
- Sandboxed script execution: `python -I`, whitelisted environment (no API keys),
  CPU/RAM/file-size rlimits, wall-clock timeout, own process group
- Docker deployment: non-root container, mem/cpu caps, healthcheck, auto-restart

## Status

- [x] Phase 1 — scaffold, Telegram connection, user allowlist
- [x] Phase 2 — GLM chat core with history
- [x] Phase 3 — attachments and document extraction
- [x] Phase 4 — agent tools and Python sandbox
- [x] Phase 5 — vision, voice, Docker deployment

## Setup

### 1. Telegram bot

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Message [@userinfobot](https://t.me/userinfobot) to learn your numeric Telegram id.

### 2. GLM API key

**Recommended: GLM Coding Plan** (Lite or higher) — the same plan you use with
Claude Code. Everything (chat, tools, image analysis) runs within the plan's
5-hour windows:

| Variable | Value |
|---|---|
| `GLM_API_KEY` | your Coding Plan key (the one Claude Code uses) |
| `GLM_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` |
| `GLM_MODEL` | `glm-5.2` |
| `GLM_ANTHROPIC_BASE_URL` | `https://api.z.ai/api/anthropic` (images; already the default) |

Note: the bot **shares the 5-hour window quota with your Claude Code sessions**.

Alternative: pay-as-you-go API (`https://api.z.ai/api/paas/v4`, requires balance).
China platform: `https://open.bigmodel.cn/api/coding/paas/v4` and `.../api/anthropic`.

### 3. Local development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
pip install pytest pytest-asyncio ruff   # dev tools
cp .env.example .env             # fill in TELEGRAM_BOT_TOKEN, ALLOWED_TELEGRAM_IDS, GLM_API_KEY
make run                         # or: python -m tgagent
```

Requires Python 3.11+.

### 4. VPS deployment (Docker)

```bash
git clone https://github.com/rafalwizen/telegram-agent.git
cd telegram-agent
cp .env.example .env && nano .env   # fill in secrets
mkdir -p data/workspace data/bot    # bind-mount targets
sudo chown -R 1000:1000 data        # container runs as uid 1000 (appuser)
docker compose up -d --build
docker compose logs -f
```

That's it — the bot connects out to Telegram (long-polling), so no ports to open.
Updating: `git pull && docker compose up -d --build`.

Health: the bot touches `data/bot/heartbeat` every minute; the Docker healthcheck
fails if it is stale for >10 min (watchdog for silent hangs).

## Configuration

All settings come from environment variables (see `.env.example`): bot token, allowed
user ids, GLM key/base URLs/model, agent limits (`MAX_TOOL_ITERS`,
`CONTEXT_TOKEN_BUDGET`) and sandbox limits (`PY_TIMEOUT`, `PY_MEM_MB`).
Voice notes are not supported.

## Workspace layout

```
data/workspace/          # agent's playground (bind-mounted into the container)
├── inbox/<date>/        # files you send land here
├── scripts/             # agent-written Python scripts
└── out/                 # agent-written outputs (reports, charts)
data/bot/bot.db          # conversation history (SQLite)
```

## Moving conversations (/import, /export)

**Claude Code → bot.** Claude Code stores every session as a JSONL file:

```
Linux/macOS:  ~/.claude/projects/<project>/<session-id>.jsonl
Windows:      %USERPROFILE%\.claude\projects\<project>\<session-id>.jsonl
```

(`<project>` is the working directory with non-alphanumeric characters replaced by `-`.
Sessions are kept for 30 days by default.) Send that file to the bot and either reply to it
with `/import`, or send `/import` first and then the file. The bot parses the dialogue
(user/assistant text — thinking, tool calls and command wrappers are skipped) and loads it
into the chat's history, **replacing** the current conversation. Note: Claude Code's own
`/export` produces a rendered text transcript for humans — it cannot be imported; use the
raw `.jsonl` session file.

**Bot → anywhere.** `/export` sends the whole conversation as a JSON file
(`[{"role": "user"|"assistant", "content": "..."}]`). Import it back later, or feed it to
any other tool that accepts this format. If an imported conversation exceeds the context
budget, the oldest turns are trimmed automatically (`build_context` keeps the newest).

## Security model

- **Allowlist first**: every update is checked against `ALLOWED_TELEGRAM_IDS`; others are ignored.
- **Path confinement**: all file tools resolve paths inside the workspace root
  (`..`, absolute paths and symlink escapes rejected).
- **Script sandbox**: agent scripts run as the container's unprivileged user, in their own
  process group, with `python -I`, a scrubbed environment (PATH/HOME/LANG only — never the
  bot token or GLM key), RLIMIT_CPU/AS/FSIZE/NPROC and a wall-clock timeout; the container
  is capped at 2 GB RAM / 1.5 CPU.
- **No inbound ports**: long-polling only.
- **Known residual risks** (accepted for a single-owner deployment): sandbox scripts share
  the bot's uid and have outbound network access. Upgrade path if that ever matters: a
  separate executor container with `network_mode: none`.

## Project layout

```
src/tgagent/
├── config.py        # env → typed config, fail-fast
├── bot.py           # Application factory, heartbeat
├── handlers.py      # Telegram routing, auth, UX flow
├── glm_client.py    # GLM chat + vision (OpenAI/Anthropic-compatible endpoints)
├── agent.py         # tool-calling loop, vision routing
├── tools.py         # tool specs + dispatch
├── conversation_io.py # /import + /export: Claude Code JSONL and JSON parsing
├── sandbox.py       # path confinement + run_python subprocess sandbox
├── extract.py       # PDF/DOCX/XLSX/CSV/TXT extraction
├── attachments.py   # Telegram file downloads → inbox
├── vision.py        # image → base64 data URL with downscaling
├── history.py       # SQLite history + context building
├── splitter.py      # 4096-char message splitting
└── prompts.py       # system prompt
```

## Tests

```bash
make test    # pytest
make lint    # ruff
```
