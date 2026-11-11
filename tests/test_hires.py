"""Tests for hires fix parameter logic and upscaling."""
from PIL import Image

import sana_engine as se


def _hires(enabled=True, scale=1.5, strength=0.35, **extra):
    return {"hires": {"enabled": enabled, "scale": scale, "strength": strength}, **extra}


def test_hires_basic_sdxl():
    out = se._hires_params(_hires(), "sdxl", 1024, 1024)
    assert out == {"scale": 1.5, "strength": 0.35, "width": 1536, "height": 1536}


def test_hires_aspect_ratio_preserved_with_cap():
    # 1152×864 at 2× would exceed 2048 → capped keeping the ratio
    out = se._hires_params(_hires(scale=2.0), "sdxl", 1152, 864)
    assert (out["width"], out["height"]) == (2048, 1536)


def test_hires_clamps_scale_and_strength():
    out = se._hires_params(_hires(scale=9.9, strength=0.99), "sdxl", 1024, 1024)
    assert out["scale"] == 2.0
    assert out["strength"] == 0.75
    assert (out["width"], out["height"]) == (2048, 2048)


def test_hires_disabled_cases():
    assert se._hires_params(_hires(enabled=False), "sdxl", 1024, 1024) is None
    assert se._hires_params({}, "sdxl", 1024, 1024) is None
    assert se._hires_params(None, "sdxl", 1024, 1024) is None
    # scale 1.0 → no growth → disabled
    assert se._hires_params(_hires(scale=1.0), "sdxl", 1024, 1024) is None


def test_hires_wrong_family_or_inpaint():
    assert se._hires_params(_hires(), "flux", 1024, 1024) is None
    assert se._hires_params(_hires(inpaint=True), "sdxl", 1024, 1024) is None


def test_hires_sd15_small_sizes():
    out = se._hires_params(_hires(scale=1.5), "sd15", 512, 512)
    assert (out["width"], out["height"]) == (768, 768)


def test_upscale_image():
    img = Image.new("RGB", (64, 48), (10, 20, 30))
    up = se._upscale_image(img, 128, 96)
    assert up.size == (128, 96)
    assert up.getpixel((0, 0)) == (10, 20, 30)


def test_hires_families_constant():
    assert se.HIRES_FAMILIES == ("sana", "sdxl", "sd15")
