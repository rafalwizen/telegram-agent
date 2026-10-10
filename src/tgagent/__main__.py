"""Entry point: ``python -m tgagent``."""

from __future__ import annotations

import sys

from .bot import build_application
from .config import ConfigError, load_config
from .logging_setup import setup_logging


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

    application = build_application(cfg)
    # Drop messages sent while the bot was down; we only want fresh updates.
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
