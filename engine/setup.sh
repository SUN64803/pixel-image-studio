#!/usr/bin/env bash
# PixelSmith Image Studio — ติดตั้ง engine (Python venv + โมเดล SANA)
#
#   bash engine/setup.sh                    ดาวน์โหลดโมเดลทางการ (~9 GB)
#   bash engine/setup.sh --from-pth PATH    แปลง .pth ในเครื่องแทนการโหลด transformer
#   bash engine/setup.sh --check            ตรวจอย่างเดียว
#   bash engine/setup.sh --no-model         ติดตั้ง venv + packages เท่านั้น
#
#   IMGS_VENV=/path/to/venv  เลือกตำแหน่ง venv (ค่าเริ่มต้น engine/.venv) —
#                            ใช้โดยตัวติดตั้งในแอป (AppImage) เพื่อวาง venv ใน home
#
# ต้องต่ออินเทอร์เน็ตครั้งนี้ครั้งเดียว (torch + CUDA libs ~5 GB + โมเดล ~9 GB)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '\033[1;36m[ImageStudio]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ImageStudio]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[ImageStudio]\033[0m %s\n' "$*"; exit 1; }

PYTHON_BIN="${PYTHON:-$(command -v python3 || true)}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "ไม่พบ python3 — ติดตั้ง Python 3.11/3.12 ก่อน"

PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
say "Python $PY_VER ($("$PYTHON_BIN" -c 'import sys;print(sys.executable)'))"

# ── flags ────────────────────────────────────────────────────────────────────
NO_MODEL=0
EXTRA_ARGS=()
for a in "$@"; do
  if [ "$a" = "--no-model" ]; then NO_MODEL=1; else EXTRA_ARGS+=("$a"); fi
done

# ── uv (เร็วกว่า pip มาก; ใช้สร้าง venv/ติดตั้ง) ─────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    warn "ไม่พบ uv — กำลังติดตั้งให้ user (ไม่ต้อง sudo)…"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || die "ติดตั้ง uv ไม่สำเร็จ"
      export PATH="$HOME/.local/bin:$PATH"
    else
      die "ไม่มี curl — ติดตั้ง uv เองหรือใช้ pip install --user uv"
    fi
  fi
fi
say "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# ── venv ─────────────────────────────────────────────────────────────────────
VENV_DIR="${IMGS_VENV:-$HERE/.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  say "สร้าง venv ($VENV_DIR)…"
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
else
  say "มี venv อยู่แล้ว — ใช้ตัวเดิม ($VENV_DIR)"
fi
VENV_PY="$VENV_DIR/bin/python"

# ── dependencies ─────────────────────────────────────────────────────────────
say "ติดตั้ง dependencies (torch + CUDA libs ~5 GB — ครั้งแรกเท่านั้น)…"
uv pip install \
  --python "$VENV_PY" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements.txt

# ── model ────────────────────────────────────────────────────────────────────
if [ "$NO_MODEL" = "1" ]; then
  say "ข้ามการเตรียมโมเดล (--no-model)"
else
  say "เตรียมโมเดล SANA…"
  "$VENV_PY" prepare_model.py "${EXTRA_ARGS[@]}"
fi

echo
say "ติดตั้งเสร็จแล้ว"
echo "   เปิดโปรแกรม : bash electron/linux-setup.sh"
echo "   หรือรัน engine ตรง ๆ : $VENV_PY engine/server.py"
