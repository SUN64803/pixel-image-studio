#!/usr/bin/env python3
"""
Prepare the SANA 1.5 · 1.6B · 1024px pipeline for PixelSmith Image Studio.

    python prepare_model.py                     # download ~9 GB (first run)
    python prepare_model.py --from-pth PATH     # convert a local .pth instead
    python prepare_model.py --check             # only report what is present

Run it with the engine venv:
    engine/.venv/bin/python engine/prepare_model.py
(or simply: bash engine/setup.sh)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import sana_engine as se


def disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024**3)
    except OSError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the SANA model for Image Studio")
    ap.add_argument("--from-pth", default=None, help="convert this .pth instead of downloading the transformer")
    ap.add_argument("--check", action="store_true", help="only report status, do not download")
    args = ap.parse_args()

    print(f"model dir : {se.MODEL_DIR}")
    print(f"data dir  : {se.DATA}")
    print(f"free disk : {disk_free_gb(se.DATA):.1f} GB")
    if se.torch is not None:
        info = se.gpu_info()
        print(f"torch     : {info.get('torch')} (cuda {info.get('cuda')})")
        print(f"gpu       : {info.get('gpu') or 'CPU only'}")
        if info.get("vram_total"):
            print(f"vram      : {info['vram_total'] / 1024**3:.1f} GB")
    else:
        print("torch     : ยังไม่ได้ติดตั้ง — รัน bash engine/setup.sh ก่อน")

    ready = se.files_ready()
    print(f"files     : {'ครบแล้ว' if ready else 'ยังไม่ครบ'}")

    if args.check:
        return 0 if ready else 1

    if ready and not args.from_pth:
        print("ไม่ต้องทำอะไร — โมเดลพร้อมใช้งานแล้ว")
        return 0

    try:
        result = se.ensure_model_files(from_pth=args.from_pth, log=print)
    except Exception as exc:  # noqa: BLE001
        print(f"\nล้มเหลว: {exc}", file=sys.stderr)
        print(
            "\nถ้าดาวน์โหลดไม่เสร็จ ให้รันซ้ำ (ไฟล์ที่โหลดแล้วจะถูกใช้ต่อ) "
            "หรือตรวจพื้นที่ว่าง/อินเทอร์เน็ต",
            file=sys.stderr,
        )
        return 1

    print(f"\nเสร็จแล้ว ({result}) — โมเดลพร้อมที่ {se.MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
