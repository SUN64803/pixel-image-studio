#!/usr/bin/env bash
# Run the Electron UI smoke tests (tests/ui/run.js).
#
# Needs a display. On a headless machine install xvfb and this script will use
# `xvfb-run` automatically. Electron comes from electron/node_modules; set
# ELECTRON=/path/to/electron to use another binary.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

electron="${ELECTRON:-$root/electron/node_modules/.bin/electron}"
if [[ ! -x "$electron" ]]; then
  echo "error: Electron not found at $electron" >&2
  echo "       run: cd electron && npm install" >&2
  exit 1
fi

runner=()
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    runner=(xvfb-run -a)
  else
    echo "error: no display and xvfb-run is not installed" >&2
    echo "       install xvfb or run from a desktop session" >&2
    exit 1
  fi
fi

export PIXELSMITH_ROOT="$root"
cd "$root"
exec "${runner[@]}" "$electron" "$here/run.js"
