"""Tests for model discovery, family detection and size helpers."""
import json

import pytest
from safetensors.torch import save_file
import torch

import sana_engine as se


# ── norm_size ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [(1024, 1024), (1000, 992), (256, 256), (100, 256), (5000, 2048), (None, 1024), ("abc", 1024)],
)
def test_norm_size(value, expected):
    assert se.norm_size(value) == expected


def test_norm_size_custom_fallback():
    assert se.norm_size(None, fallback=512) == 512


# ── _slug ────────────────────────────────────────────────────────────────────
def test_slug_shape():
    slug = se._slug(__import__("pathlib").Path("/tmp/My Cool Model!.safetensors"), "sdxl")
    assert slug.startswith("sdxl-my-cool-model-")
    digest = slug.rsplit("-", 1)[1]
    assert len(digest) == 8 and all(c in "0123456789abcdef" for c in digest)


def test_slug_deterministic_per_path():
    from pathlib import Path

    a = se._slug(Path("/x/foo.safetensors"), "sd15")
    b = se._slug(Path("/x/foo.safetensors"), "sd15")
    c = se._slug(Path("/y/foo.safetensors"), "sd15")
    assert a == b and a != c


# ── family detection from weight keys ────────────────────────────────────────
@pytest.mark.parametrize(
    "keys,expected",
    [
        (["double_blocks.0.img_attn.qkv.weight"], None),  # Flux → unsupported
        (["single_transformer_blocks.0.x"], None),  # Flux → unsupported
        (["conditioner.embedders.1.model.transformer.ln_1.weight"], "sdxl"),
        (["lora_te2.foo"], "sdxl"),
        (["text_encoder_2.bar"], "sdxl"),
        (["caption_projection.linear.weight"], "sana"),
        (["ff.conv_inverted.weight"], "sana"),
        (["transformer_blocks.0.attn1.qkv.weight", "transformer.norm.weight"], "sana"),
        (["lora_te_text_model.encoder.layers.0.mlp.fc1.weight"], "sd15"),
        (["cond_stage_model.transformer.text_model.weight"], "sd15"),
        (["lora_unet.down_blocks.0.conv.weight"], "sd15"),
        (["random.unrelated.key"], None),
    ],
)
def test_family_from_keys(keys, expected):
    assert se._family_from_keys(keys) == expected


# ── family from diffusers dir ────────────────────────────────────────────────
def test_family_from_dir(tmp_path):
    d = tmp_path / "mypipeline"
    d.mkdir()
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    assert se._family_from_dir(d) == "sdxl"

    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaPipeline"}), encoding="utf-8"
    )
    assert se._family_from_dir(d) == "sana"

    (d / "model_index.json").write_text("not json at all", encoding="utf-8")
    assert se._family_from_dir(d) is None
    assert se._family_from_dir(tmp_path / "missing") is None


# ── files_ready ──────────────────────────────────────────────────────────────
def test_files_ready_missing_dir(tmp_path):
    assert se.files_ready(tmp_path) is False


# ── discover_models on a fake scan dir ───────────────────────────────────────
def test_discover_models_scans_custom_dirs(tmp_path, monkeypatch):
    base = tmp_path / "scan"
    base.mkdir()

    # diffusers directory (SDXL)
    d = base / "cool-xl"
    d.mkdir()
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    # single-file checkpoint (SD1.5 keys)
    save_file(
        {"cond_stage_model.transformer.text_model.weight": torch.zeros(1)},
        str(base / "old-school.safetensors"),
    )
    # unknown checkpoint must be skipped
    save_file({"random.unrelated.key": torch.zeros(1)}, str(base / "mystery.safetensors"))
    # .ckpt disabled
    (base / "legacy.ckpt").write_bytes(b"x")

    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    models = se.discover_models()

    by_name = {m["name"]: m for m in models}
    assert by_name["cool-xl"]["family"] == "sdxl"
    assert by_name["cool-xl"]["kind"] == "diffusers"
    assert by_name["old-school"]["family"] == "sd15"
    assert by_name["old-school"]["kind"] == "single"
    assert "mystery" not in by_name
    # always contains the built-in SANA entry first
    assert models[0]["family"] == "sana"
    # models sorted by (family, name) after the first entry
    tail = models[1:]
    assert tail == sorted(tail, key=lambda m: (m["family"], m["name"].lower()))


def test_capabilities_complete():
    for family in ("sana", "sdxl", "sd15"):
        caps = se.CAPABILITIES[family]
        for key in ("negative_prompt", "enhance", "lora", "regional", "inpaint", "hires", "samplers"):
            assert key in caps, f"missing {key} for {family}"
    assert se.CAPABILITIES["sana"]["hires"] is True
    assert se.CAPABILITIES["sdxl"]["hires"] is True
    assert se.CAPABILITIES["sd15"]["hires"] is True
    assert se.CAPABILITIES["sdxl"]["regional"] is True
    assert se.CAPABILITIES["sana"]["regional"] is False
    assert se.CAPABILITIES["sd15"]["inpaint"] is False
