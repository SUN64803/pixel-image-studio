#!/usr/bin/env bash
# PixelSmith Image Studio — ตัวช่วยติดตั้ง/เปิดโปรแกรมบน Linux (ใช้ได้กับทุก distro ที่มี Node)
#
#   bash linux-setup.sh         ติดตั้ง dependency แล้วเปิดโปรแกรม
#   bash linux-setup.sh dist    สร้าง AppImage ใน ./dist
#   bash linux-setup.sh check   ตรวจสภาพแวดล้อมเฉย ๆ ไม่ทำอะไร
#
# สคริปต์นี้ไม่แตะไฟล์ภาพ/ประวัติของคุณ — แค่ติดตั้ง dependency แล้วเรียก npm

set -u
cd "$(dirname "$0")" || exit 1

MODE="${1:-start}"
say()  { printf '\033[1;36m[ImageStudio]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ImageStudio]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[ImageStudio]\033[0m %s\n' "$*"; exit 1; }

# ── Node ─────────────────────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
  warn "ไม่พบ Node.js บนเครื่องนี้ — ติดตั้งก่อนด้วย:"
  echo
  echo "  sudo apt update"
  echo "  sudo apt install -y curl ca-certificates"
  echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
  echo "  sudo apt install -y nodejs"
  echo
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if [ "$NODE_MAJOR" -lt 18 ] 2>/dev/null; then
  die "Node $(node -v) เก่าเกินไป — Electron 32 ต้องใช้ Node 18 ขึ้นไป (แนะนำ 20)"
fi
say "Node $(node -v) · npm $(npm -v)"

# ── โครงสร้างโปรแกรม ────────────────────────────────────────────────────────
for f in main.js preload.js package.json icon.png ../host/index.html ../studio/index.html ../engine/server.py; do
  [ -e "$f" ] || warn "ไม่พบ $f — ไฟล์โปรแกรมอาจแตกซิปมาไม่ครบ"
done
[ -e ../studio/index.html ] || die "ไม่พบ ../studio/index.html"
[ -e ../engine/server.py ] || die "ไม่พบ ../engine/server.py"

# ── engine (Python venv + โมเดล) ────────────────────────────────────────────
VENV_PY="../engine/.venv/bin/python"
MODEL_DIR="../models/SANA1.5_1.6B_1024px_diffusers"
if [ ! -x "$VENV_PY" ]; then
  warn "ยังไม่ได้ติดตั้ง engine (ไม่พบ $VENV_PY)"
  if [ "$MODE" != "check" ] && [ -t 0 ]; then
    echo
    read -r -p "ติดตั้ง engine + ดาวน์โหลดโมเดลตอนนี้เลยไหม? (~11 GB, ต้องต่อเน็ต) [y/N] " ans
    case "$ans" in
      y|Y) bash ../engine/setup.sh || die "ติดตั้ง engine ไม่สำเร็จ" ;;
      *) warn "ข้าม — โปรแกรมจะเปิดได้ แต่จะยังสร้างภาพไม่ได้จนกว่าจะรัน: bash ../engine/setup.sh" ;;
    esac
  else
    echo "  ติดตั้งด้วย: bash ../engine/setup.sh"
  fi
elif [ ! -f "$MODEL_DIR/model_index.json" ]; then
  warn "engine พร้อม แต่ยังไม่มีไฟล์โมเดล — จะดาวน์โหลดอัตโนมัติเมื่อกดสร้างภาพครั้งแรก"
  echo "  หรือเตรียมล่วงหน้า: bash ../engine/setup.sh"
else
  say "engine + โมเดลพร้อมใช้งาน"
fi

if [ "$MODE" = "check" ]; then
  say "ตรวจแล้ว ดูเหมือนพร้อมใช้งาน — เปิดด้วย: npm start"
  exit 0
fi

# ── dependency ───────────────────────────────────────────────────────────────
if [ ! -d node_modules ] || [ ! -d node_modules/electron ]; then
  say "ติดตั้ง dependency (ดาวน์โหลด Electron ~100 MB — ครั้งแรกครั้งเดียว)…"
  npm install || die "npm install ล้มเหลว — ดูข้อความ error ด้านบน (เน็ต? พื้นที่ว่าง?)"
else
  say "dependency ติดตั้งไว้แล้ว — ข้ามการติดตั้ง"
fi

# ── sandbox (Ubuntu 24.04 / Mint 22 ล็อก namespace ทำให้ Electron เปิดไม่ขึ้น) ──
SB="node_modules/electron/dist/chrome-sandbox"
if [ -f "$SB" ]; then
  MODE_BITS="$(stat -c '%a' "$SB" 2>/dev/null || echo '')"
  if [ "$MODE_BITS" != "4755" ]; then
    say "ตั้งสิทธิ์ chrome-sandbox (ต้องใช้รหัสผ่าน sudo — ข้ามได้ด้วย Ctrl+C แล้วรัน './node_modules/.bin/electron . --no-sandbox')"
    sudo chown root:root "$SB" 2>/dev/null && sudo chmod 4755 "$SB" 2>/dev/null \
      && say "ซ่อม sandbox เรียบร้อย" \
      || warn "ซ่อม sandbox ไม่สำเร็จ — ถ้าเปิดโปรแกรมไม่ขึ้น ให้รัน: npx electron . --no-sandbox"
  fi
fi

# ── เปิด ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "dist" ]; then
  say "สร้าง AppImage ด้วย electron-builder (ต้องต่อเน็ต)…"
  npm run dist
  say "เสร็จแล้ว — ไฟล์อยู่ใน ./dist"
  echo "   หมายเหตุ: AppImage ไม่ได้ฝัง Python venv/โมเดล ให้รัน engine/setup.sh ในโฟลเดอร์โปรแกรมก่อนเปิด"
else
  say "เปิดโปรแกรม…"
  echo "   (ครั้งต่อไปไม่ต้องพิมพ์คำสั่งอีก — สร้างไอคอนเปิดโปรแกรมด้วย: bash create-launcher.sh)"
  npm start
fi
