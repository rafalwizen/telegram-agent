"""Entry point: ``python -m tgagent``."""

from __future__ import annotations

import logging
import sys
import time

import telegram.error

from .agent import Agent
from .bot import build_application
from .config import ConfigError, load_config
from .glm_client import GLMClient
from .history import History
from .logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Wait time before reconnecting after a transient Telegram network error.
RECONNECT_DELAY_S = 30


def main() -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
    setup_logging(cfg.log_level)

    # Ensure runtime directories exist (workspace with its conventional subdirs).
    for sub in ("inbox", "scripts", "out"):
        (cfg.workspace_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    history = History(cfg.data_dir / "bot.db")
    glm = GLMClient(cfg)
    agent = Agent(cfg, glm, history)
    try:
        while True:
            # A fresh Application per attempt: run_polling() may only be called once.
            application = build_application(cfg, agent, history, glm)
            try:
                # Drop messages sent while the bot was down; we only want fresh updates.
                application.run_polling(drop_pending_updates=True)
                return  # clean shutdown (SIGTERM/SIGINT)
            except (telegram.error.TimedOut, telegram.error.NetworkError) as exc:
                # Transient connectivity problems must not crash the bot;
                # Docker's restart policy stays as the last line of defence.
                logger.error(
                    "Telegram network error (%s); reconnecting in %d s",
                    exc,
                    RECONNECT_DELAY_S,
                )
                time.sleep(RECONNECT_DELAY_S)
    finally:
        history.close()


if __name__ == "__main__":
    main()
