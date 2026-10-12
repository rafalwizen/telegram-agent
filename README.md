# telegram-agent

Personal Telegram AI assistant powered by GLM (Zhipu / Z.ai), with a sandboxed
workspace on your own VPS. You chat with it like with any AI chat app, send files
for analysis (PDF, DOCX, CSV, images, voice notes), and the agent can write and
run its own Python scripts inside a confined workspace directory.

## Status

- [x] Phase 1 — scaffold, Telegram connection, user allowlist, echo mode
- [x] Phase 2 — GLM chat core with history
- [x] Phase 3 — attachments and document extraction
- [x] Phase 4 — agent tools and Python sandbox
- [ ] Phase 5 — vision, voice, Docker deployment

## Quick start (local dev)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # fill in TELEGRAM_BOT_TOKEN, ALLOWED_TELEGRAM_IDS, GLM_API_KEY
make run
```

Requires Python 3.11+.

## Security model (summary)

- Only whitelisted Telegram user ids (`ALLOWED_TELEGRAM_IDS`) can talk to the bot.
- All file operations are confined to the workspace directory.
- Agent-written scripts run with a scrubbed environment (no API keys), CPU/RAM/file-size
  limits and a wall-clock timeout.

Details will be documented here as the remaining phases land.
