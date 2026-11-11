#!/usr/bin/env bash
# เปิด PixelSmith Image Studio ตรง ๆ — ตัวนี้คือสิ่งที่ไอคอนในเมนู/เดสก์ท็อปเรียก
# (ไม่ต้องเปิด terminal ไม่ต้องพิมพ์ npm ทุกครั้ง) ส่ง argument ต่อให้ Electron ได้ เช่น
#   ./launch.sh --no-sandbox
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" || exit 1

fail() {
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --width=480 --title="PixelSmith Image Studio" --text="$1"
  else
    printf '%s\n' "$1" >&2
  fi
  exit 1
}

if [ ! -x node_modules/.bin/electron ]; then
  fail "PixelSmith Image Studio ยังไม่ได้ติดตั้ง dependency (โฟลเดอร์ node_modules)

เปิด terminal แล้วรัน:
  cd \"$PWD\"
  npm install

(หรือรันสคริปต์ที่ให้มา: bash linux-setup.sh)"
fi

for f in ../host/index.html ../studio/index.html ../engine/server.py; do
  [ -e "$f" ] || fail "ไม่พบไฟล์ $f

โปรแกรมต้องมีโฟลเดอร์เหล่านี้อยู่ด้วยกันเสมอ:
  electron/ , host/ , studio/ , engine/"
done

if [ ! -x ../engine/.venv/bin/python ]; then
  fail "ยังไม่ได้ติดตั้ง engine (ไม่พบ ../engine/.venv)

เปิด terminal แล้วรัน (ดาวน์โหลดครั้งแรก ~11 GB):
  cd \"$(cd .. && pwd)\"
  bash engine/setup.sh

จากนั้นเปิดโปรแกรมนี้ใหม่"
fi

exec ./node_modules/.bin/electron . "$@"
