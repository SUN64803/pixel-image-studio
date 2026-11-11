#!/usr/bin/env python3
"""Real-ESRGAN (RRDBNet) upscaler — pure PyTorch, no `basicsr` dependency.

Weights: RealESRGAN_x4plus.pth (67 MB) from the official release.  The card has
10 GB VRAM, so anything larger than one tile is processed in overlapping tiles
and stitched back together — this keeps peak VRAM around 2–4 GB and produces
the same result as a single pass (the overlap is cropped off).  Outputs above
4096 px use 256 px tiles: at 512 px the allocator reserves ~9 GB of the card.

Only the RRDBNet generator is implemented; the discriminator from the original
training code is not needed for inference.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path(os.environ.get("IMGS_MODELS") or (ROOT / "models"))
UPSCALE_DIR = MODELS_ROOT / "upscalers"
MODEL_PATH = UPSCALE_DIR / "RealESRGAN_x4plus.pth"

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
MODEL_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
MODEL_BYTES = 67_040_989
MODEL_LABEL = "Real-ESRGAN ×4"
MODEL_ID = "realesrgan-x4plus"

# 2048² input ×4 = 8192² output (~805 MB fp32) — the practical ceiling on a
# 10 GB card / 15 GB RAM machine.
MAX_OUTPUT_SIDE = 8192
DEFAULT_TILE = 512
DEFAULT_TILE_PAD = 16

# Above 4096 px per side the caching allocator fragments badly with 512 px
# tiles: measured peak *reserved* VRAM is 9.2 GB of the 10 GB card.  256 px
# tiles cost ~10% more time but stay under 4 GB.
LARGE_OUTPUT_SIDE = 4096
LARGE_OUTPUT_TILE = 256


def pick_tile(width: int, height: int, scale: int = 4, requested=None) -> int:
    """Tile size for an upscale: honour an explicit request, else pick by size."""
    if requested:
        return max(128, min(1024, int(requested)))
    if max(width, height) * scale > LARGE_OUTPUT_SIDE:
        return LARGE_OUTPUT_TILE
    return DEFAULT_TILE


# ── network ──────────────────────────────────────────────────────────────────
class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (the Real-ESRGAN generator body)."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBNet(nn.Module):
    """Generator architecture of RealESRGAN_x4plus (scale ×4)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ── weights ─────────────────────────────────────────────────────────────────
def weights_ready() -> bool:
    try:
        return MODEL_PATH.is_file() and MODEL_PATH.stat().st_size == MODEL_BYTES
    except OSError:
        return False


def ensure_weights(progress=None) -> Path:
    """Download the official weights once, verifying the SHA-256."""
    if weights_ready():
        return MODEL_PATH

    UPSCALE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    digest = hashlib.sha256()
    done = 0
    req = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "pixelsmith-image-studio"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    if digest.hexdigest() != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "ไฟล์โมเดล upscale เสียหาย (checksum ไม่ตรง) — ลองดาวน์โหลดใหม่"
        )
    os.replace(tmp, MODEL_PATH)
    return MODEL_PATH


# ── tiling ──────────────────────────────────────────────────────────────────
class UpscaleCancelled(Exception):
    """Raised between tiles when the user cancels an upscale job."""


def tile_process(
    model,
    img: torch.Tensor,
    scale: int = 4,
    tile: int = DEFAULT_TILE,
    tile_pad: int = DEFAULT_TILE_PAD,
    on_progress=None,
    cancel=None,
) -> torch.Tensor:
    """Run `model` over overlapping tiles and stitch the result.

    `img` is (1, 3, H, W) in [0, 1].  The padded border of each tile is cropped
    from the output, so seams do not appear.
    """
    _, _, h, w = img.shape
    out = img.new_zeros((1, img.shape[1], h * scale, w * scale))
    tiles_x = (w + tile - 1) // tile
    tiles_y = (h + tile - 1) // tile
    total = tiles_x * tiles_y
    done = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if cancel is not None and cancel.is_set():
                raise UpscaleCancelled()
            ix0, ix1 = x, min(x + tile, w)
            iy0, iy1 = y, min(y + tile, h)
            px0, px1 = max(ix0 - tile_pad, 0), min(ix1 + tile_pad, w)
            py0, py1 = max(iy0 - tile_pad, 0), min(iy1 + tile_pad, h)
            patch = img[:, :, py0:py1, px0:px1]
            patch_out = model(patch)
            ox0, ox1 = ix0 * scale, ix1 * scale
            oy0, oy1 = iy0 * scale, iy1 * scale
            sx = (ix0 - px0) * scale
            sy = (iy0 - py0) * scale
            out[:, :, oy0:oy1, ox0:ox1] = patch_out[
                :, :, sy : sy + (iy1 - iy0) * scale, sx : sx + (ix1 - ix0) * scale
            ]
            done += 1
            if on_progress is not None:
                on_progress(done, total)
    return out


# ── upscaler ────────────────────────────────────────────────────────────────
class RealESRGANUpscaler:
    def __init__(self, device: str | None = None, tile: int = DEFAULT_TILE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tile = tile
        self._model: RRDBNet | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return weights_ready()

    def info(self) -> dict:
        return {
            "id": MODEL_ID,
            "label": MODEL_LABEL,
            "scale": 4,
            "max_output": MAX_OUTPUT_SIDE,
            "downloaded": weights_ready(),
            "bytes": MODEL_BYTES,
        }

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            path = ensure_weights()
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = (
                checkpoint.get("params_ema")
                or checkpoint.get("params")
                or checkpoint
            )
            model = RRDBNet()
            model.load_state_dict(state, strict=True)
            model.eval()
            self._model = model.to(self.device)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def upscale(
        self,
        image: Image.Image,
        scale: int = 4,
        tile: int | None = None,
        on_progress=None,
        cancel=None,
    ) -> Image.Image:
        if scale not in (2, 4):
            raise ValueError("scale ต้องเป็น 2 หรือ 4")
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if max(w, h) * 4 > MAX_OUTPUT_SIDE:
            raise ValueError(
                f"ภาพใหญ่เกินไป: {w}×{h} ×4 = {w * 4}×{h * 4} "
                f"(สูงสุด {MAX_OUTPUT_SIDE}px ต่อด้าน)"
            )

        self.load()
        model = self._model
        assert model is not None

        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        tile = tile or self.tile
        if tile and tile > 0 and (w > tile or h > tile):
            out = tile_process(
                model,
                tensor,
                scale=4,
                tile=tile,
                tile_pad=DEFAULT_TILE_PAD,
                on_progress=on_progress,
                cancel=cancel,
            )
        else:
            out = model(tensor)
            if on_progress is not None:
                on_progress(1, 1)

        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        result = Image.fromarray((out * 255.0).round().astype(np.uint8), "RGB")
        if scale != 4:
            result = result.resize((w * scale, h * scale), Image.LANCZOS)
        return result


UPSCALER = RealESRGANUpscaler()
