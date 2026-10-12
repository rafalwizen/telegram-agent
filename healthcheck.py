"""Container healthcheck: fail (exit 1) when the bot's heartbeat file is stale or missing."""

from __future__ import annotations

import os
import sys
import time

MAX_AGE_S = int(os.environ.get("HEARTBEAT_MAX_AGE", "600"))
HEARTBEAT_PATH = os.environ.get("HEARTBEAT", "/app/data/heartbeat")

try:
    age = time.time() - os.path.getmtime(HEARTBEAT_PATH)
except OSError:
    sys.exit(1)

sys.exit(0 if age < MAX_AGE_S else 1)
