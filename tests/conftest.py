"""Shared test setup — isolate data dir and make `sana_engine` importable."""
import os
import sys
import tempfile
from pathlib import Path

# IMGS_DATA must be set before sana_engine/server are imported anywhere: it
# redirects data/images/thumbs + history.json away from the real data dir.
_TEST_DATA = tempfile.mkdtemp(prefix="imgs-test-data-")
os.environ.setdefault("IMGS_DATA", _TEST_DATA)
os.environ.setdefault("IMGS_PORT", "47999")
# The Host-guard middleware trusts loopback hostnames only; tests connect
# via Starlette's default "testserver" base URL.
os.environ.setdefault("IMGS_TRUSTED_HOSTS", "testserver")

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
