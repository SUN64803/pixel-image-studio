#!/usr/bin/env bash
# สร้างไอคอนเปิดโปรแกรม (Linux Mint / XFCE) — รันครั้งเดียว ไม่ต้องรันซ้ำอีก
#
#   bash create-launcher.sh
#
# มันจะสร้าง:
#   ~/.local/share/applications/pixelsmith-image.desktop   ← เมนูเริ่ม (Graphics)
#   ~/Desktop/PixelSmith Image Studio.desktop              ← ไอคอนบนเดสก์ท็อป (ถ้ามีโฟลเดอร์ Desktop)
#   ~/.local/share/icons/hicolor/512x512/apps/pixelsmith-image.png
#
# ไอคอนจะเรียก launch.sh ในโฟลเดอร์นี้ (ไม่มีหน้าต่าง terminal โผล่มา)
# ถ้าย้ายโฟลเดอร์โปรแกรมไปที่อื่น ให้รันสคริปต์นี้อีกครั้งเพื่ออัปเดตพาธ

set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
ID="pixelsmith-image"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/512x512/apps"
PIXMAPS="$HOME/.local/share/pixmaps"

say() { printf '\033[1;36m[ImageStudio]\033[0m %s\n' "$*"; }

[ -x "$DIR/launch.sh" ] || chmod +x "$DIR/launch.sh" 2>/dev/null || true

mkdir -p "$APPS" "$ICONS" "$PIXMAPS"

if [ -f "$DIR/icon.png" ]; then
  cp -f "$DIR/icon.png" "$ICONS/$ID.png"
  cp -f "$DIR/icon.png" "$PIXMAPS/$ID.png"
  say "ติดตั้งไอคอนแล้ว"
else
  say "ไม่พบ icon.png — จะใช้ไอคอนมาตรฐานของ Electron ไปก่อน"
fi

DESKTOP_FILE="$APPS/$ID.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=PixelSmith Image Studio
Name[th]=PixelSmith Image Studio (สร้างภาพด้วย SANA)
GenericName=AI image generator
Comment=Local image generation with SANA 1.5 — offline, on this machine's GPU
Comment[th]=สร้างภาพด้วย SANA 1.5 — ออฟไลน์ ใช้ GPU ในเครื่องนี้
Exec=bash "$DIR/launch.sh"
Path=$DIR
Icon=$ID
Terminal=false
Categories=Graphics;2DGraphics;Utility;
Keywords=image;ai;sana;diffusion;generate;pixelsmith;
StartupNotify=true
StartupWMClass=pixelsmith-image
EOF
chmod +x "$DESKTOP_FILE"
say "สร้างเมนูแล้ว: $DESKTOP_FILE"

if [ -d "$HOME/Desktop" ]; then
  cp -f "$DESKTOP_FILE" "$HOME/Desktop/PixelSmith Image Studio.desktop"
  chmod +x "$HOME/Desktop/PixelSmith Image Studio.desktop"
  say "วางไอคอนบนเดสก์ท็อปแล้ว (~/Desktop)"
fi

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" >/dev/null 2>&1
command -v gtk-update-icon-cache  >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1

echo
say "เสร็จแล้ว เปิดโปรแกรมได้ 3 ทาง:"
echo "   • เมนู → Graphics → PixelSmith Image Studio"
echo "   • ดับเบิลคลิกไอคอนบนเดสก์ท็อป (ถ้าไม่ขึ้น ให้ออกจากระบบแล้วเข้าใหม่ 1 ครั้ง)"
echo "   • หรือรันตรง ๆ: bash \"$DIR/launch.sh\""
