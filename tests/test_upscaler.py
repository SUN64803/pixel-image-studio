"""Real-ESRGAN upscaler — architecture, tiling and validation (no GPU needed)."""
import numpy as np
import pytest
import torch
from PIL import Image

import upscaler as upr


class _Nearest4x(torch.nn.Module):
    """Stand-in for RRDBNet: local, so tiled and full passes must be identical."""

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_rrdb_forward_shape():
    model = upr.RRDBNet(num_feat=8, num_block=1, num_grow_ch=4).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 8, 8))
    assert tuple(out.shape) == (1, 3, 32, 32)


def test_tile_process_matches_single_pass():
    torch.manual_seed(0)
    img = torch.rand(1, 3, 70, 90)
    with torch.inference_mode():
        full = _Nearest4x()(img)
        tiled = upr.tile_process(_Nearest4x(), img, scale=4, tile=32, tile_pad=8)
    assert tuple(tiled.shape) == (1, 3, 280, 360)
    assert torch.equal(full, tiled)


def test_tile_process_reports_progress():
    seen = []
    img = torch.rand(1, 3, 40, 40)
    with torch.inference_mode():
        upr.tile_process(
            _Nearest4x(),
            img,
            scale=4,
            tile=16,
            tile_pad=4,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    assert seen[-1] == (9, 9)  # 3×3 tiles


def test_tile_process_honours_cancel():
    import threading

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(upr.UpscaleCancelled):
        upr.tile_process(
            _Nearest4x(), torch.rand(1, 3, 32, 32), tile=16, cancel=cancel
        )


def test_upscale_rejects_bad_scale():
    up = upr.RealESRGANUpscaler(device="cpu")
    with pytest.raises(ValueError):
        up.upscale(Image.new("RGB", (8, 8)), scale=3)


def test_upscale_rejects_too_large_before_loading():
    up = upr.RealESRGANUpscaler(device="cpu")
    with pytest.raises(ValueError, match="ใหญ่เกินไป"):
        up.upscale(Image.new("RGB", (2049, 8)), scale=4)


def test_pick_tile_by_output_size():
    assert upr.pick_tile(1024, 1024, scale=4) == upr.DEFAULT_TILE
    assert upr.pick_tile(2048, 2048, scale=2) == upr.DEFAULT_TILE  # 4096 px output
    assert upr.pick_tile(2048, 2048, scale=4) == upr.LARGE_OUTPUT_TILE  # 8192 px


def test_pick_tile_honours_and_clamps_request():
    assert upr.pick_tile(2048, 2048, scale=4, requested=512) == 512
    assert upr.pick_tile(1024, 1024, scale=4, requested=64) == 128
    assert upr.pick_tile(1024, 1024, scale=4, requested=4096) == 1024
    assert upr.pick_tile(1024, 1024, scale=4, requested=None) == upr.DEFAULT_TILE


def test_info_shape():
    info = upr.UPSCALER.info()
    assert info["scale"] == 4
    assert info["max_output"] == upr.MAX_OUTPUT_SIDE
    assert info["id"] == "realesrgan-x4plus"


def test_weights_ready_checks_size(tmp_path, monkeypatch):
    fake = tmp_path / "RealESRGAN_x4plus.pth"
    fake.write_bytes(b"x" * 10)
    monkeypatch.setattr(upr, "MODEL_PATH", fake)
    assert upr.weights_ready() is False
