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
        (["double_blocks.0.img_attn.qkv.weight"], "flux"),
        (["single_transformer_blocks.0.x"], "flux"),
        (["distilled_guidance_layer.in_proj.weight"], "chroma"),
        (["double_blocks.0.x", "distilled_guidance_layer.layers.0.x"], "chroma"),
        (["joint_blocks.0.x", "text_encoder_3.x"], "sd3"),
        (["double_layers.0.x"], "auraflow"),
        (["noise_refiner.0.x"], "lumina"),
        (["adaln_single.emb.x", "transformer_blocks.0.ff.net.0.proj.weight"], "pixart"),
        # SDXL also uses ff.net (GEGLU) — must not be mistaken for PixArt
        (["unet.down_blocks.0.x", "transformer_blocks.0.ff.net.0.proj.weight"], "sdxl"),
        (
            [
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
                "lora_unet_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
            ],
            "sdxl",
        ),
        (["conditioner.embedders.1.model.transformer.ln_1.weight"], "sdxl"),
        (["lora_te2.foo"], "sdxl"),
        (["text_encoder_2.bar"], "sdxl"),
        (["caption_projection.linear.weight"], "sana"),
        (["ff.conv_inverted.weight"], "sana"),
        (["transformer_blocks.0.attn1.qkv.weight", "transformer.norm.weight"], "sana"),
        (["cond_stage_model.model.token_embedding.weight"], "sd2"),
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


def test_family_from_dir_extra_architectures(tmp_path):
    for class_name, family in (
        ("StableDiffusion3Pipeline", "sd3"),
        ("PixArtSigmaPipeline", "pixart"),
        ("Lumina2Text2ImgPipeline", "lumina2"),
        ("HunyuanDiTPipeline", "hunyuan"),
        ("KolorsPipeline", "kolors"),
        ("AuraFlowPipeline", "auraflow"),
        ("FluxPipeline", "flux"),
        ("ChromaPipeline", "chroma"),
    ):
        d = tmp_path / family
        d.mkdir()
        (d / "model_index.json").write_text(
            json.dumps({"_class_name": class_name}), encoding="utf-8"
        )
        assert se._family_from_dir(d) == family, class_name


def test_dir_info_generic_and_rejected(tmp_path):
    # An installed diffusers pipeline we have no explicit profile for is still
    # usable through the generic loader.
    generic = tmp_path / "cascade"
    generic.mkdir()
    (generic / "model_index.json").write_text(
        json.dumps({"_class_name": "StableCascadeCombinedPipeline"}), encoding="utf-8"
    )
    family, class_name, reason = se._dir_info(generic)
    assert (family, class_name, reason) == ("generic", "StableCascadeCombinedPipeline", "")

    # Video/audio pipelines are not image generators.
    video = tmp_path / "video"
    video.mkdir()
    (video / "model_index.json").write_text(
        json.dumps({"_class_name": "CogVideoXPipeline"}), encoding="utf-8"
    )
    family, _class_name, reason = se._dir_info(video)
    assert family is None and reason

    # Unknown class that the installed diffusers build does not have.
    missing = tmp_path / "missing-class"
    missing.mkdir()
    (missing / "model_index.json").write_text(
        json.dumps({"_class_name": "TotallyMadeUpPipeline"}), encoding="utf-8"
    )
    family, _class_name, reason = se._dir_info(missing)
    assert family is None and "ไม่รู้จัก" in reason


def test_single_file_sana_is_listed_but_unsupported(tmp_path, monkeypatch):
    base = tmp_path / "scan"
    base.mkdir()
    save_file(
        {"transformer_blocks.0.ff.conv_inverted.weight": torch.zeros(1)},
        str(base / "sana-single.safetensors"),
    )
    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    entry = {m["name"]: m for m in se.discover_models()}["sana-single"]
    assert entry["family"] == "sana"
    assert entry["supported"] is False
    assert "diffusers" in entry["unsupported_reason"]


def test_generic_pipeline_is_listed_and_loadable(tmp_path, monkeypatch):
    base = tmp_path / "scan"
    base.mkdir()
    d = base / "cascade"
    d.mkdir()
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableCascadeCombinedPipeline"}), encoding="utf-8"
    )
    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    entry = {m["name"]: m for m in se.discover_models()}["cascade"]
    assert entry["family"] == "generic"
    assert entry["supported"] is True
    assert entry["pipeline_class"] == "StableCascadeCombinedPipeline"


def test_flux_chroma_checkpoints_listed_as_unsupported(tmp_path, monkeypatch):
    base = tmp_path / "scan"
    base.mkdir()
    save_file(
        {"double_blocks.0.img_attn.qkv.weight": torch.zeros(1)},
        str(base / "flux-dev.safetensors"),
    )
    save_file(
        {"distilled_guidance_layer.in_proj.weight": torch.zeros(1)},
        str(base / "chroma.safetensors"),
    )
    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    by_name = {m["name"]: m for m in se.discover_models()}
    assert by_name["flux-dev"]["family"] == "flux"
    assert by_name["flux-dev"]["supported"] is False
    assert by_name["chroma"]["family"] == "chroma"
    assert by_name["chroma"]["supported"] is False
    assert by_name["chroma"]["unsupported_reason"]


def test_filter_pipe_kwargs():
    class Loose:
        def __call__(self, prompt, num_inference_steps=1, **kwargs):
            return None

    class Strict:
        def __call__(self, prompt, num_inference_steps=1):
            return None

    kwargs = {"prompt": "x", "num_inference_steps": 2, "negative_prompt": "y"}
    assert se._filter_pipe_kwargs(Loose(), kwargs) == kwargs
    assert se._filter_pipe_kwargs(Strict(), kwargs) == {
        "prompt": "x",
        "num_inference_steps": 2,
    }


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
    # unknown checkpoint is still listed (with a reason) instead of vanishing
    save_file({"random.unrelated.key": torch.zeros(1)}, str(base / "mystery.safetensors"))
    # .ckpt disabled
    (base / "legacy.ckpt").write_bytes(b"x")

    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    models = se.discover_models()

    by_name = {m["name"]: m for m in models}
    assert by_name["cool-xl"]["family"] == "sdxl"
    assert by_name["cool-xl"]["kind"] == "diffusers"
    assert by_name["cool-xl"]["supported"] is True
    assert by_name["old-school"]["family"] == "sd15"
    assert by_name["old-school"]["kind"] == "single"
    assert by_name["mystery"]["supported"] is False
    assert by_name["mystery"]["unsupported_reason"]
    # always contains the built-in SANA entry first
    assert models[0]["family"] == "sana"
    # supported models come first, then sorted by (family, name)
    tail = models[1:]
    assert tail == sorted(
        tail,
        key=lambda m: (0 if m.get("supported") else 1, m["family"], m["name"].lower()),
    )


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


def test_nested_checkpoint_and_non_model_dirs(tmp_path, monkeypatch):
    base = tmp_path / "models"
    base.mkdir()
    nested = base / "checkpoints" / "sub" / "deeper"
    nested.mkdir(parents=True)
    save_file(
        {"conditioner.embedders.1.model.x": torch.zeros(1)},
        str(nested / "nested-xl.safetensors"),
    )
    # LoRA / upscaler folders must never be listed as checkpoints
    loras = base / "loras"
    loras.mkdir()
    save_file({"lora_unet_down_blocks_0": torch.zeros(1)}, str(loras / "style.safetensors"))
    # files inside a diffusers folder are shards, not standalone checkpoints
    diff = base / "checkpoints" / "diffusers-model"
    diff.mkdir()
    (diff / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    save_file({"unet.down_blocks.0.x": torch.zeros(1)}, str(diff / "diffusion_pytorch_model.safetensors"))

    monkeypatch.setattr(se, "MODEL_SCAN_DIRS", [base])
    by_name = {m["name"]: m for m in se.discover_models()}
    assert by_name["nested-xl"]["family"] == "sdxl"
    assert by_name["nested-xl"]["supported"] is True
    assert "style" not in by_name
    assert "diffusion_pytorch_model" not in by_name
    assert by_name["diffusers-model"]["kind"] == "diffusers"


def test_trainable_flag_follows_family():
    from pathlib import Path

    def entry_for(family):
        path = Path(f"/tmp/{family}-entry")
        kind = "diffusers" if family == "sana" else "single"
        if kind == "single":
            path = path.with_suffix(".safetensors")
        return se._model_entry(path, family, kind, Path("/tmp"))

    assert entry_for("sana")["trainable"] is True
    assert entry_for("sdxl")["trainable"] is True
    for family in ("sd15", "pixart", "generic", "flux", "chroma", "unknown"):
        assert entry_for(family)["trainable"] is False, family


def test_hires_families_only_with_img2img():
    assert "sana" in se.HIRES_FAMILIES
    assert "sdxl" in se.HIRES_FAMILIES
    assert "sd2" in se.HIRES_FAMILIES
    assert "sd3" in se.HIRES_FAMILIES
    assert "kolors" in se.HIRES_FAMILIES
    for family in se.HIRES_FAMILIES:
        assert se.CAPABILITIES[family]["hires"] is True
        assert family == "sana" or "img2img" in se.PIPELINE_CLASSES.get(family, {})


def test_select_model_rejects_unsupported(monkeypatch):
    entries = [
        se._model_entry(
            __import__("pathlib").Path("/tmp/good-xl.safetensors"),
            "sdxl",
            "single",
            __import__("pathlib").Path("/tmp"),
        ),
        se._model_entry(
            __import__("pathlib").Path("/tmp/flux-x.safetensors"),
            "flux",
            "single",
            __import__("pathlib").Path("/tmp"),
        ),
    ]
    monkeypatch.setattr(se, "discover_models", lambda: entries)
    manager = se.ModelManager()
    assert manager.current_id == entries[0]["id"]
    with pytest.raises(ValueError) as exc:
        manager.select_model(entries[1]["id"])
    assert "ใช้ไม่ได้" in str(exc.value)
    assert manager.current_id == entries[0]["id"]
