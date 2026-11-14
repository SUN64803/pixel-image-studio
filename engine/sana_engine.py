#!/usr/bin/env python3
"""
PixelSmith Image Studio — SANA engine.

Everything that touches torch / diffusers lives here, so the web server
(server.py) stays small and the model can be prepared from a terminal
(prepare_model.py) without ever starting the HTTP API.

Model: SANA 1.5 · 1.6B · 1024px (NVIDIA / Efficient-Large-Model).
Weights are the official diffusers conversion of
    Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth
Optionally a local .pth can be converted instead of downloading the
transformer (see convert_pth_to_transformer / prepare_model.py --from-pth).
"""
from __future__ import annotations

import functools
import gc
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
DATA = Path(os.environ.get("IMGS_DATA") or (ROOT / "data"))
IMAGES = DATA / "images"
THUMBS = DATA / "thumbs"
# IMGS_MODELS relocates model/LoRA storage (used by packaged AppImage builds,
# where the app bundle itself is read-only and everything lives in the user's
# data home instead)
MODELS_ROOT = Path(os.environ.get("IMGS_MODELS") or (ROOT / "models"))
MODEL_DIR = Path(
    os.environ.get("SANA_MODEL_DIR") or (MODELS_ROOT / "SANA1.5_1.6B_1024px_diffusers")
)
REPO_ID = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
MODEL_LABEL = "SANA 1.5 · 1.6B · 1024px"
MODEL_ID = "sana15-1.6b-1024"

for _d in (DATA, IMAGES, THUMBS, MODELS_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

# extra model / LoRA folders (Forge, A1111, external disks …). Configure with
# models/dirs.json ({"model_dirs": [...], "lora_dirs": [...]}) or env vars
# IMGS_MODEL_DIRS / IMGS_LORA_DIRS (os.pathsep separated).
def _path_list(value: str) -> list[Path]:
    return [Path(p).expanduser() for p in value.split(os.pathsep) if p.strip()]


def _extra_dirs() -> tuple[list[Path], list[Path]]:
    cfg_models: list[Path] = []
    cfg_loras: list[Path] = []
    cfg_file = MODELS_ROOT / "dirs.json"
    try:
        cfg = json.loads(cfg_file.read_text("utf-8"))
        cfg_models = [Path(x).expanduser() for x in cfg.get("model_dirs", []) if str(x).strip()]
        cfg_loras = [Path(x).expanduser() for x in cfg.get("lora_dirs", []) if str(x).strip()]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[models] อ่าน {cfg_file} ไม่ได้: {exc}", flush=True)
    cfg_models += _path_list(os.environ.get("IMGS_MODEL_DIRS", ""))
    cfg_loras += _path_list(os.environ.get("IMGS_LORA_DIRS", ""))
    return cfg_models, cfg_loras


_EXTRA_MODEL_DIRS, _EXTRA_LORA_DIRS = _extra_dirs()
MODEL_SCAN_DIRS = [MODELS_ROOT, MODELS_ROOT / "checkpoints", *_EXTRA_MODEL_DIRS]
LORA_SCAN_DIRS = [MODELS_ROOT / "loras", *_EXTRA_LORA_DIRS]

# ── model families ──────────────────────────────────────────────────────────
# `family` is the architecture profile the engine knows how to run.  The list
# is not limited to the models we happen to have tested: any diffusers pipeline
# directory whose `_class_name` exists in the installed diffusers build is
# listed as family="generic" and loaded with that class.  `supported=False`
# entries are still shown in the UI (with a reason) instead of disappearing —
# notably Flux/Chroma, whose 9–12B weights do not fit 10 GB VRAM / 15 GB RAM.
def _caps(
    enhance: bool = False,
    regional: bool = False,
    inpaint: bool = False,
    hires: bool = False,
    negative: bool = True,
    lora: bool = True,
) -> dict:
    return {
        "negative_prompt": negative,
        "enhance": enhance,
        "lora": lora,
        "regional": regional,
        "inpaint": inpaint,
        "hires": hires,
        "samplers": ["dpm", "euler"],
    }


def _defaults(width: int, height: int, steps: int, guidance: float, enhance: bool = False) -> dict:
    return {
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "sampler": "dpm",
        "enhance": enhance,
    }


_TOO_HEAVY = (
    "ต้องใช้ VRAM/RAM มากกว่าที่การ์ดนี้มี (10 GB VRAM / 15 GB RAM) — "
    "Flux/Chroma ~9–12B ต้องรัน fp8 ซึ่ง GPU นี้ไม่รองรับ"
)

FAMILY_PROFILES: dict[str, dict] = {
    "sana": {
        "label": "SANA",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 20, 4.5, True),
        "caps": _caps(enhance=True, hires=True),
    },
    "sdxl": {
        "label": "SDXL",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(1024, 1024, 28, 5.0),
        "caps": _caps(regional=True, inpaint=True, hires=True),
    },
    "sd15": {
        "label": "SD 1.5",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(512, 512, 25, 7.0),
        "caps": _caps(hires=True),
    },
    "sd2": {
        "label": "SD 2.x",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(768, 768, 25, 7.5),
        "caps": _caps(hires=True),
    },
    "sd3": {
        "label": "SD 3 / 3.5",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 28, 4.5),
        "caps": _caps(hires=True),
    },
    "pixart": {
        "label": "PixArt",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(1024, 1024, 20, 4.5),
        "caps": _caps(),
    },
    "lumina": {
        "label": "Lumina",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 30, 4.0),
        "caps": _caps(),
    },
    "lumina2": {
        "label": "Lumina 2",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 30, 4.0),
        "caps": _caps(),
    },
    "hunyuan": {
        "label": "HunyuanDiT",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 30, 5.0),
        "caps": _caps(),
    },
    "kolors": {
        "label": "Kolors",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(1024, 1024, 28, 5.0),
        "caps": _caps(hires=True),
    },
    "auraflow": {
        "label": "AuraFlow",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 25, 3.5),
        "caps": _caps(),
    },
    "amused": {
        "label": "Amused",
        "supported": True,
        "dtype": "float16",
        "defaults": _defaults(512, 512, 12, 3.5),
        "caps": _caps(),
    },
    "cogview3": {
        "label": "CogView3",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 50, 5.0),
        "caps": _caps(),
    },
    "cogview4": {
        "label": "CogView4",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 30, 5.0),
        "caps": _caps(),
    },
    "wuerstchen": {
        "label": "Würstchen",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 30, 4.0),
        "caps": _caps(),
    },
    "generic": {
        "label": "Diffusers",
        "supported": True,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 28, 4.5),
        "caps": _caps(),
    },
    "flux": {
        "label": "Flux",
        "supported": False,
        "reason": _TOO_HEAVY,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 20, 3.5),
        "caps": _caps(),
    },
    "chroma": {
        "label": "Chroma",
        "supported": False,
        "reason": _TOO_HEAVY,
        "dtype": "bfloat16",
        "defaults": _defaults(1024, 1024, 20, 3.5),
        "caps": _caps(),
    },
    "unknown": {
        "label": "Unknown",
        "supported": False,
        "reason": "ตรวจไม่พบสถาปัตยกรรมที่รองรับ",
        "dtype": "float16",
        "defaults": _defaults(1024, 1024, 28, 4.5),
        "caps": _caps(),
    },
}

FAMILY_DEFAULTS = {k: v["defaults"] for k, v in FAMILY_PROFILES.items()}
FAMILY_LABEL = {k: v["label"] for k, v in FAMILY_PROFILES.items()}
CAPABILITIES = {k: v["caps"] for k, v in FAMILY_PROFILES.items()}

# Text-to-image / img2img classes per family.  `generic` entries carry the
# class name discovered from model_index.json instead.
PIPELINE_CLASSES: dict[str, dict[str, str]] = {
    "sana": {"t2i": "SanaPipeline"},
    "sdxl": {
        "t2i": "StableDiffusionXLPipeline",
        "img2img": "StableDiffusionXLImg2ImgPipeline",
        "inpaint": "StableDiffusionXLInpaintPipeline",
    },
    "sd15": {
        "t2i": "StableDiffusionPipeline",
        "img2img": "StableDiffusionImg2ImgPipeline",
    },
    "sd2": {
        "t2i": "StableDiffusionPipeline",
        "img2img": "StableDiffusionImg2ImgPipeline",
    },
    "sd3": {
        "t2i": "StableDiffusion3Pipeline",
        "img2img": "StableDiffusion3Img2ImgPipeline",
    },
    "flux": {"t2i": "FluxPipeline", "img2img": "FluxImg2ImgPipeline"},
    "chroma": {"t2i": "ChromaPipeline", "img2img": "ChromaImg2ImgPipeline"},
    "pixart": {"t2i": "PixArtSigmaPipeline"},
    "lumina": {"t2i": "LuminaText2ImgPipeline"},
    "lumina2": {"t2i": "Lumina2Text2ImgPipeline"},
    "hunyuan": {"t2i": "HunyuanDiTPipeline"},
    "kolors": {"t2i": "KolorsPipeline", "img2img": "KolorsImg2ImgPipeline"},
    "auraflow": {"t2i": "AuraFlowPipeline"},
    "amused": {"t2i": "AmusedPipeline", "img2img": "AmusedImg2ImgPipeline"},
    "cogview3": {"t2i": "CogView3PlusPipeline"},
    "cogview4": {"t2i": "CogView4Pipeline"},
    "wuerstchen": {"t2i": "WuerstchenCombinedPipeline"},
}

# Families whose single-file (.safetensors) checkpoints can be loaded by
# diffusers.  SanaPipeline / PixArt / Lumina / HunyuanDiT have no
# `from_single_file`, so those need a diffusers directory.
SINGLE_FILE_FAMILIES = {"sdxl", "sd15", "sd2", "sd3", "flux", "chroma"}

# Families with a LoRA trainer in this app.  `sana_trainer` is the generic one
# (SANA + diffusers layouts) and `sdxl_trainer` handles SDXL checkpoints.
TRAINABLE_FAMILIES = ("sana", "sdxl")

HIRES_FAMILIES = tuple(
    k
    for k, v in FAMILY_PROFILES.items()
    if v["caps"]["hires"]
    and v["supported"]
    and (k == "sana" or "img2img" in PIPELINE_CLASSES.get(k, {}))
)

# torch is imported lazily so `prepare_model.py --check` and the web server can
# print helpful diagnostics even when the venv is not installed yet.
torch = None
try:  # pragma: no cover - depends on the machine
    import torch as _torch

    torch = _torch
except Exception:  # noqa: BLE001
    torch = None


def norm_size(value, fallback: int = 1024) -> int:
    """SANA compresses latents 32x; sizes must be multiples of 32."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = fallback
    v = max(256, min(2048, v))
    return int(round(v / 32.0) * 32)


def gpu_info() -> dict:
    info = {"device": "cpu", "gpu": None, "vram_total": 0, "vram_used": 0, "torch": None}
    if torch is None:
        return info
    try:
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update(
                device="cuda",
                gpu=torch.cuda.get_device_name(0),
                vram_total=int(total),
                vram_used=int(total - free),
            )
            info["temp"] = _gpu_temperature()
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


# nvidia-smi is a subprocess; health polls every few seconds, so cache briefly.
_TEMP_CACHE = {"at": 0.0, "value": None}


def _gpu_temperature() -> float | None:
    now = time.time()
    if now - _TEMP_CACHE["at"] < 5.0:
        return _TEMP_CACHE["value"]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        )
        value = float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        value = None
    _TEMP_CACHE.update(at=now, value=value)
    return value


def files_ready(model_dir: Path | None = None) -> bool:
    d = Path(model_dir or MODEL_DIR)
    if not (d / "model_index.json").is_file():
        return False
    if not (d / "transformer" / "diffusion_pytorch_model.safetensors").is_file():
        return False
    if not (d / "vae" / "diffusion_pytorch_model.safetensors").is_file():
        return False
    te = d / "text_encoder"
    if not te.is_dir():
        return False
    return any(te.glob("*.safetensors")) or (te / "pytorch_model.bin").is_file()


# ── model / LoRA discovery ───────────────────────────────────────────────────
# diffusers `_class_name` → engine family.  Anything else that exists in the
# installed diffusers build is loaded as family="generic" (see
# `_dir_info`), so new architectures do not need a code change.
_DIFFUSERS_FAMILY = {
    "SanaPipeline": "sana",
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionPipeline": "sd15",
    "StableDiffusion3Pipeline": "sd3",
    "PixArtSigmaPipeline": "pixart",
    "PixArtAlphaPipeline": "pixart",
    "LuminaPipeline": "lumina",
    "LuminaText2ImgPipeline": "lumina",
    "Lumina2Pipeline": "lumina2",
    "Lumina2Text2ImgPipeline": "lumina2",
    "HunyuanDiTPipeline": "hunyuan",
    "KolorsPipeline": "kolors",
    "AuraFlowPipeline": "auraflow",
    "AmusedPipeline": "amused",
    "CogView3PlusPipeline": "cogview3",
    "CogView4Pipeline": "cogview4",
    "WuerstchenCombinedPipeline": "wuerstchen",
    "FluxPipeline": "flux",
    "ChromaPipeline": "chroma",
}

# Pipelines that are not image generators (video/audio/editing).  They are
# skipped so a folder of video models does not clutter the picker.
_NON_IMAGE_HINTS = (
    "video",
    "audio",
    "music",
    "image2video",
    "imagetovideo",
    "img2video",
    "videotovideo",
    "texttovideo",
    "skyreels",
    "animatediff",
    "cosmos",
    "ltx",
    "wan",
    "mochi",
    "hunyuanvideo",
    "easyanimate",
    "stablevideo",
    "i2vgen",
    "latte",
    "allegro",
    "cogvideox",
    "visualcloze",
    "omnigen",
    "depth",
    "intrinsics",
    "normals",
    "superresolution",
    "upscale",
    "controlnet",
    "adapter",
    "inpaint",
    "img2img",
    "imagevariation",
    "prior",
    "decoder",
    "control",
)


def _slug(path: Path, family: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:42] or "model"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{family}-{stem}-{digest}"


def _safetensors_keys(path: Path, limit: int = 400) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = []
        for key in handle.keys():
            keys.append(key)
            if len(keys) >= limit:
                break
        return keys


def _family_from_keys(keys: list[str]) -> str | None:
    """Detect the architecture from safetensors tensor names.

    The order matters: Chroma also has FLUX-style `double_blocks`, PixArt also
    has a `caption_projection` like SANA, and SD 2.x shares SD 1.5's layout.
    """
    blob = " ".join(keys).lower()
    # Chroma (FLUX.1-schnell derivative) — distilled guidance is unique to it.
    if "distilled_guidance_layer" in blob:
        return "chroma"
    # FLUX — unsupported on this GPU but identified so its LoRAs are never
    # offered to an SDXL/SANA pipeline.
    if "single_transformer_blocks" in blob or "double_blocks" in blob:
        return "flux"
    # AuraFlow
    if "double_layers" in blob or "single_layers" in blob:
        return "auraflow"
    # SD 3 / 3.5 (MMDiT + CLIP-L/CLIP-G/T5)
    if "joint_blocks" in blob or "lora_te3" in blob or "text_encoder_3" in blob:
        return "sd3"
    # Lumina / Lumina 2
    if "noise_refiner" in blob or "context_refiner" in blob or "cap_embedder" in blob:
        return "lumina"
    # SDXL markers must be checked before PixArt/SANA: SDXL transformer blocks
    # also use `ff.net.0.proj` (GEGLU), and a kohya SDXL LoRA can be recognised
    # by its text encoder 2 / label embedding / block layout.
    if "conditioner.embedders.1" in blob or "lora_te2" in blob or "text_encoder_2" in blob:
        return "sdxl"
    # Diffusers-format SDXL LoRA (what our SDXL trainer writes):
    #   unet.down_blocks.…lora_A.weight
    # Kohya SD 1.5 uses `lora_unet_down_blocks` (no dot) so it won't match.
    if any(k.lower().startswith("unet.") for k in keys) and (
        "down_blocks" in blob or "mid_block" in blob or "up_blocks" in blob
    ):
        return "sdxl"
    # Kohya SDXL LoRA: transformer_blocks live inside input/middle/output
    # blocks (SD 1.5 only has one in the middle block).  `label_emb` (the
    # size/negative-original embedding) is also SDXL-only.
    if (
        "lora_unet_input_blocks_4_1_transformer_blocks" in blob
        or "lora_unet_middle_block_1_transformer_blocks" in blob
        or "label_emb" in blob
        or ("lora_unet_output_blocks_" in blob and "transformer_blocks" in blob)
    ):
        return "sdxl"
    # SANA (DC-AE latent 32×): conv feed-forward / caption_norm / time_embed
    if (
        "ff.conv_inverted" in blob
        or "ff.conv_depth" in blob
        or "caption_norm" in blob
        or "time_embed.emb.timestep_embedder" in blob
    ):
        return "sana"
    # PixArt (also has caption_projection, but uses adaln_single + ff.net)
    if "adaln_single" in blob or "ff.net.0.proj" in blob:
        return "pixart"
    # Older heuristic kept for checkpoints that only expose the caption
    # projection (SANA LoRAs with attention adapters only).
    if "caption_projection" in blob:
        return "sana"
    if "transformer_blocks" in blob and ("transformer." in blob or "attn1." in blob):
        return "sana"
    # SD 2.x uses an OpenCLIP text encoder (`cond_stage_model.model…`), while
    # SD 1.x uses `cond_stage_model.transformer.text_model…`.
    if "cond_stage_model.model" in blob:
        return "sd2"
    if "lora_te_text_model" in blob or "cond_stage_model" in blob:
        return "sd15"
    if "lora_unet" in blob:
        return "sd15"  # ambiguous; SD1.5 is the common kohya layout
    return None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _dir_class_name(path: Path) -> str:
    return str(_read_json(path / "model_index.json").get("_class_name") or "")


def _dir_sample_size(path: Path) -> int | None:
    for sub in ("unet", "transformer"):
        value = _read_json(path / sub / "config.json").get("sample_size")
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolution_from_dir(path: Path, family: str) -> int | None:
    """Guess the native resolution from the UNet/transformer sample size."""
    ss = _dir_sample_size(path)
    if not ss:
        return None
    factor = 32 if family == "sana" else 8
    px = ss * factor
    if px >= 768:
        return 1024
    if px >= 640:
        return 768
    return 512


@functools.lru_cache(maxsize=None)
def _diffusers_has_class(name: str) -> bool:
    try:
        import diffusers

        cls = getattr(diffusers, name, None)
        return bool(cls and hasattr(cls, "from_pretrained"))
    except Exception:  # noqa: BLE001
        return False


def _dir_info(path: Path) -> tuple[str | None, str, str]:
    """Return (family, pipeline_class, reason) for a diffusers directory."""
    class_name = _dir_class_name(path)
    if not class_name:
        return None, "", "model_index.json ไม่มี _class_name"
    family = _DIFFUSERS_FAMILY.get(class_name)
    if family:
        return family, class_name, ""
    lowered = class_name.lower()
    if any(hint in lowered for hint in _NON_IMAGE_HINTS):
        return None, class_name, f"ไม่ใช่ pipeline สร้างภาพ ({class_name})"
    if not _diffusers_has_class(class_name):
        return None, class_name, f"diffusers ที่ติดตั้งไม่รู้จัก {class_name}"
    # Any other installed text-to-image pipeline runs through the generic
    # profile — that is what keeps new architectures usable without a code
    # change.  img2img/hires stays off because we cannot assume a matching
    # img2img class exists.
    return "generic", class_name, ""


def _family_from_dir(path: Path) -> str | None:
    family, _cls, _reason = _dir_info(path)
    if family == "sd15":
        # SD 2.x shares StableDiffusionPipeline but uses 768px + OpenCLIP.
        if (_dir_sample_size(path) or 0) >= 96:
            return "sd2"
    return family


def _family_from_file(path: Path) -> str | None:
    if path.suffix.lower() != ".safetensors":
        return None  # .ckpt disabled: torch.load is slow/unsafe
    try:
        return _family_from_keys(_safetensors_keys(path))
    except Exception as exc:  # noqa: BLE001
        print(f"[models] อ่าน {path.name} ไม่ได้: {exc}", flush=True)
        return None


# ── base-model variants (Pony / Illustrious / NoobAI …) ─────────────────────
# `family` is what the engine can load (sana / sdxl / sd15).  `variant` is the
# checkpoint lineage an adapter was trained on: two SDXL checkpoints can share
# no compatible LoRAs (Pony vs Illustrious), yet kohya writes
# `ss_base_model_version: sdxl_base_v1-0` for both.  So we combine filename
# hints, metadata hints and a manual override in data/variants.json.

VARIANT_LABEL = {
    "any": "ใช้ได้ทุกโมเดล",
    "sana": "SANA",
    "sdxl": "SDXL",
    "pony": "Pony",
    "illustrious": "Illustrious",
    "noobai": "NoobAI",
    "sd15": "SD 1.5",
    "sd2": "SD 2.x",
    "sd3": "SD 3 / 3.5",
    "pixart": "PixArt",
    "lumina": "Lumina",
    "lumina2": "Lumina 2",
    "hunyuan": "HunyuanDiT",
    "kolors": "Kolors",
    "auraflow": "AuraFlow",
    "amused": "Amused",
    "cogview3": "CogView3",
    "cogview4": "CogView4",
    "wuerstchen": "Würstchen",
    "generic": "Diffusers",
    "flux": "Flux",
    "chroma": "Chroma",
}
# Values the UI may set by hand.  "" clears the override (back to auto-detect).
VARIANT_CHOICES = tuple(VARIANT_LABEL)
_VARIANTS_FILE = DATA / "variants.json"
_VARIANT_LOCK = threading.Lock()
# safetensors metadata keys that sometimes name the base checkpoint
_VARIANT_META_KEYS = (
    "base_model",
    "modelspec.title",
    "ss_output_name",
    "ss_sd_model_name",
    "ss_training_comment",
    "modelspec.architecture",
    "ss_base_model_version",
    "civitai_model_name",
)


def _variant_hint(text: str) -> str | None:
    """Guess the checkpoint lineage from a filename / metadata string."""
    t = re.sub(r"[_\-./\\|:]+", " ", str(text or "")).lower()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    if re.search(r"noob\s?ai", t):
        return "noobai"
    # `_IL` / `- IL` suffixes are the common Illustrious marker; the full word
    # is matched as a substring because `myIllustriousSDXL` has no boundary.
    if re.search(r"illustrious", t) or re.search(r"\bil\b", t):
        return "illustrious"
    if re.search(r"pony", t):
        return "pony"
    if re.search(r"\bchroma", t):
        return "chroma"
    if re.search(r"\bflux", t):
        return "flux"
    return None


def _override_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def load_variant_overrides() -> dict[str, str]:
    try:
        raw = json.loads(_VARIANTS_FILE.read_text("utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        print(f"[variants] อ่าน {_VARIANTS_FILE} ไม่ได้: {exc}", flush=True)
        return {}
    paths = raw.get("paths") if isinstance(raw, dict) else None
    if not isinstance(paths, dict):
        return {}
    return {str(k): str(v) for k, v in paths.items() if str(v).strip()}


def set_variant(path: str | Path, variant: str) -> dict:
    """Persist (or clear) a manual variant override for one file."""
    key = _override_key(Path(path))
    variant = str(variant or "").strip().lower()
    if variant and variant not in VARIANT_CHOICES:
        raise ValueError(f"variant ไม่รู้จัก: {variant}")
    with _VARIANT_LOCK:
        overrides = load_variant_overrides()
        if variant:
            overrides[key] = variant
        else:
            overrides.pop(key, None)
        _VARIANTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _VARIANTS_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"paths": overrides}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(_VARIANTS_FILE)
    return {"path": key, "variant": variant, "overrides": len(overrides)}


def _variant_for(
    path: Path, family: str | None, meta: dict | None, overrides: dict[str, str]
) -> tuple[str, str]:
    """Return (variant, source); source is override / name / metadata / family."""
    key = _override_key(path)
    if key in overrides:
        return overrides[key], "override"
    found = _variant_hint(path.stem)
    if found:
        return found, "name"
    for name in _VARIANT_META_KEYS:
        value = (meta or {}).get(name)
        if not value:
            continue
        found = _variant_hint(str(value))
        if found:
            return found, "metadata"
    return (family or "any"), "family"


def _safetensors_meta(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return handle.metadata() or {}


def _model_entry(
    path: Path,
    family: str,
    kind: str,
    source: Path,
    meta: dict | None = None,
    overrides: dict[str, str] | None = None,
    pipeline_class: str = "",
    reason: str = "",
    display_name: str | None = None,
) -> dict:
    profile = FAMILY_PROFILES.get(family) or FAMILY_PROFILES["generic"]
    supported = bool(profile.get("supported", True)) and not reason
    if supported and kind == "single" and family not in SINGLE_FILE_FAMILIES:
        supported = False
        reason = (
            "ไฟล์ .safetensors เดี่ยวโหลดไม่ได้สำหรับสถาปัตยกรรมนี้ — "
            "ต้องใช้โฟลเดอร์ diffusers"
        )
    if not supported and not reason:
        reason = profile.get("reason") or "ยังไม่รองรับสถาปัตยกรรมนี้"
    defaults = dict(profile["defaults"])
    if kind == "diffusers":
        native = _resolution_from_dir(path, family)
        if native:
            defaults["width"] = native
            defaults["height"] = native
    available = files_ready(path) if family == "sana" and path == MODEL_DIR else True
    variant, variant_source = _variant_for(path, family, meta, overrides or {})
    name = display_name or (MODEL_LABEL if path == MODEL_DIR else path.stem)
    return {
        "id": MODEL_ID if path == MODEL_DIR else _slug(path, family),
        "name": name,
        "family": family,
        "family_label": FAMILY_LABEL.get(family, family),
        "variant": variant,
        "variant_label": VARIANT_LABEL.get(variant, variant),
        "variant_source": variant_source,
        "kind": kind,
        "pipeline_class": pipeline_class,
        "path": str(path),
        "source": str(source),
        "available": available,
        "supported": supported,
        "unsupported_reason": reason if not supported else "",
        "trainable": supported and family in TRAINABLE_FAMILIES,
        "defaults": defaults,
        "capabilities": profile["caps"],
    }


def _inside_diffusers(path: Path, base: Path) -> bool:
    """True when `path` sits inside a diffusers folder (has model_index.json)."""
    try:
        parent = path.parent
    except OSError:
        return False
    while True:
        if (parent / "model_index.json").is_file():
            return True
        if parent == base or parent == parent.parent:
            return False
        parent = parent.parent


# Sub-folders of models/ that never contain base checkpoints.  Without this
# the recursive scan would list every LoRA / upscaler as a broken checkpoint.
_NON_MODEL_DIRS = {
    "loras",
    "lora",
    "upscalers",
    "upscale",
    "vae",
    "embeddings",
    "controlnet",
    "datasets",
    "dataset",
    "data",
    "cache",
    "tmp",
}


def discover_models() -> list[dict]:
    overrides = load_variant_overrides()
    models: list[dict] = [
        _model_entry(MODEL_DIR, "sana", "diffusers", MODEL_DIR.parent, overrides=overrides)
    ]
    seen = {MODEL_DIR.resolve()}
    seen_files: set[tuple[str, int]] = set()
    for base in MODEL_SCAN_DIRS:
        if not base.is_dir():
            continue
        try:
            candidates = sorted(base.rglob("*"), key=lambda p: str(p).lower())
        except OSError:
            continue
        for path in candidates:
            try:
                rel_parts = path.relative_to(base).parts
            except ValueError:
                rel_parts = path.parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if rel_parts and rel_parts[0].lower() in _NON_MODEL_DIRS:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            if path.is_dir():
                if not (path / "model_index.json").is_file():
                    continue
                family, pipeline_class, reason = _dir_info(path)
                if family == "sd15" and (_dir_sample_size(path) or 0) >= 96:
                    family = "sd2"
                seen.add(resolved)
                models.append(
                    _model_entry(
                        path,
                        family or "unknown",
                        "diffusers",
                        base,
                        overrides=overrides,
                        pipeline_class=pipeline_class,
                        reason=reason,
                    )
                )
            elif path.is_file() and path.suffix.lower() in (
                ".safetensors",
                ".ckpt",
                ".pt",
                ".pth",
                ".bin",
            ):
                if _inside_diffusers(path, base):
                    continue
                try:
                    file_key = (path.name.lower(), path.stat().st_size)
                except OSError:
                    file_key = None
                if file_key and file_key in seen_files:
                    continue
                suffix = path.suffix.lower()
                if suffix == ".safetensors":
                    family = _family_from_file(path)
                    reason = (
                        ""
                        if family
                        else "ตรวจไม่พบสถาปัตยกรรมที่รองรับ (อาจไม่ใช่ checkpoint หรือไฟล์เสีย)"
                    )
                else:
                    # Loading .ckpt/.pt needs torch.load (slow + unsafe).  Show
                    # them instead of hiding the file, with a conversion hint.
                    family = None
                    reason = f"ไฟล์ {suffix} ยังไม่รองรับ — แปลงเป็น .safetensors ก่อน"
                seen.add(resolved)
                if file_key:
                    seen_files.add(file_key)
                try:
                    meta = _safetensors_meta(path) if suffix == ".safetensors" else {}
                except Exception:  # noqa: BLE001
                    meta = {}
                models.append(
                    _model_entry(
                        path,
                        family or "unknown",
                        "single",
                        base,
                        meta=meta,
                        overrides=overrides,
                        reason=reason,
                    )
                )
    # Supported models first, then by family/name so the picker is predictable.
    models[1:] = sorted(
        models[1:],
        key=lambda m: (
            0 if m.get("supported") else 1,
            m["family"],
            m["name"].lower(),
        ),
    )
    return models


def _lora_family(path: Path) -> str | None:
    try:
        return _family_from_keys(_safetensors_keys(path, limit=300))
    except Exception:  # noqa: BLE001
        return None


def discover_loras() -> list[dict]:
    items: list[dict] = []
    overrides = load_variant_overrides()
    seen: set[Path] = set()
    seen_names: set[tuple[str, int]] = set()
    for base in LORA_SCAN_DIRS:
        if not base.is_dir():
            continue
        try:
            files = sorted(base.rglob("*.safetensors"), key=lambda p: p.name.lower())
        except OSError:
            continue
        for path in files:
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            # A 0-byte / truncated safetensors is not a usable LoRA.  It used
            # to show up in the picker and fail only at generation time.
            if size < 16:
                print(
                    f"[loras] ข้ามไฟล์ว่าง/เสีย: {path.name} ({size} bytes)",
                    flush=True,
                )
                continue
            try:
                # Opening + reading one tensor catches truncated headers and
                # files whose payload is missing, not just empty files.
                from safetensors import safe_open

                meta = {}
                with safe_open(str(path), framework="pt") as handle:
                    keys = list(handle.keys())
                    if not keys:
                        raise ValueError("ไม่มี tensor")
                    handle.get_tensor(keys[0])
                    meta = handle.metadata() or {}
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[loras] ข้ามไฟล์ที่อ่านไม่ได้: {path.name} ({exc})",
                    flush=True,
                )
                continue
            name_key = (path.name.lower(), size)
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            family = _family_from_keys(keys)
            meta_family = str(meta.get("family") or "").strip().lower()
            if meta_family in ("sana", "sdxl", "sd15"):
                family = meta_family
            try:
                rank = int(meta.get("rank"))
            except (TypeError, ValueError, AttributeError):
                rank = None
            variant, variant_source = _variant_for(path, family, meta, overrides)
            items.append(
                {
                    "id": _slug(path, family or "any"),
                    "name": path.stem,
                    "path": str(path),
                    "family": family,
                    "family_label": FAMILY_LABEL.get(family or "", "ใช้ได้ทุกโมเดล"),
                    "variant": variant,
                    "variant_label": VARIANT_LABEL.get(variant, variant),
                    "variant_source": variant_source,
                    "size": size,
                    "rank": rank,
                    "source": str(base),
                }
            )
    return items


def ensure_model_files(from_pth: str | None = None, log=print) -> str:
    """Download (or convert) the SANA pipeline into MODEL_DIR.

    Returns "ready", "downloaded" or "converted".
    Safe to call repeatedly: existing files are reused / resumed.
    """
    d = MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)

    if from_pth:
        pth = Path(from_pth).expanduser().resolve()
        if not pth.is_file():
            raise FileNotFoundError(f"ไม่พบไฟล์ .pth: {pth}")
        if (d / "transformer" / "diffusion_pytorch_model.safetensors").is_file():
            log(f"มี transformer อยู่แล้ว — ข้ามการแปลง {pth.name}")
        else:
            from huggingface_hub import snapshot_download

            log("ดาวน์โหลด text encoder / VAE / tokenizer (ไม่รวม transformer)…")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(d),
                allow_patterns=[
                    "model_index.json",
                    "scheduler/*",
                    "tokenizer/*",
                    "text_encoder/*",
                    "vae/*",
                ],
                max_workers=8,
            )
            log("แปลง .pth → diffusers transformer (ใช้ RAM ประมาณ 8–10 GB)…")
            convert_pth_to_transformer(pth, d / "transformer", log=log)
            log("แปลงเสร็จแล้ว")
        if not files_ready(d):
            raise RuntimeError("ไฟล์โมเดลยังไม่ครบหลังการแปลง")
        return "converted"

    if files_ready(d):
        return "ready"

    from huggingface_hub import snapshot_download

    log(f"ดาวน์โหลด {MODEL_LABEL} ครั้งแรก (~9 GB)…")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(d),
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "vae/*",
            "transformer/*",
        ],
        max_workers=8,
    )
    if not files_ready(d):
        raise RuntimeError(
            "ดาวน์โหลดโมเดลไม่ครบ — ตรวจสอบอินเทอร์เน็ตและพื้นที่ว่างแล้วลองใหม่"
        )
    return "downloaded"


# ── optional: convert the original NVIDIA .pth ────────────────────────────────
# Trimmed from NVlabs/Sana tools/convert_scripts/convert_sana_to_diffusers.py
# (Apache-2.0). Only the SANA 1.5 1.6B 1024px layout is supported.
def convert_pth_to_transformer(pth_path, out_dir, log=print) -> None:
    if torch is None:
        raise RuntimeError("ต้องติดตั้ง PyTorch ก่อนแปลง .pth")
    from accelerate import init_empty_weights
    from diffusers import SanaTransformer2DModel
    from diffusers.models.modeling_utils import load_model_dict_into_meta

    log(f"โหลด {pth_path} (mmap)…")
    raw = torch.load(str(pth_path), map_location="cpu", mmap=True, weights_only=True)
    state = dict(raw.pop("state_dict") if "state_dict" in raw else raw)
    conv = {}

    def take(key):
        return state.pop(key)

    conv["patch_embed.proj.weight"] = take("x_embedder.proj.weight")
    conv["patch_embed.proj.bias"] = take("x_embedder.proj.bias")
    conv["caption_projection.linear_1.weight"] = take("y_embedder.y_proj.fc1.weight")
    conv["caption_projection.linear_1.bias"] = take("y_embedder.y_proj.fc1.bias")
    conv["caption_projection.linear_2.weight"] = take("y_embedder.y_proj.fc2.weight")
    conv["caption_projection.linear_2.bias"] = take("y_embedder.y_proj.fc2.bias")
    conv["time_embed.emb.timestep_embedder.linear_1.weight"] = take("t_embedder.mlp.0.weight")
    conv["time_embed.emb.timestep_embedder.linear_1.bias"] = take("t_embedder.mlp.0.bias")
    conv["time_embed.emb.timestep_embedder.linear_2.weight"] = take("t_embedder.mlp.2.weight")
    conv["time_embed.emb.timestep_embedder.linear_2.bias"] = take("t_embedder.mlp.2.bias")
    conv["time_embed.linear.weight"] = take("t_block.1.weight")
    conv["time_embed.linear.bias"] = take("t_block.1.bias")
    conv["caption_norm.weight"] = take("attention_y_norm.weight")

    for depth in range(20):
        p = f"blocks.{depth}"
        q = f"transformer_blocks.{depth}"
        conv[f"{q}.scale_shift_table"] = take(f"{p}.scale_shift_table")

        qw, kw, vw = torch.chunk(take(f"{p}.attn.qkv.weight"), 3, dim=0)
        conv[f"{q}.attn1.to_q.weight"] = qw
        conv[f"{q}.attn1.to_k.weight"] = kw
        conv[f"{q}.attn1.to_v.weight"] = vw
        conv[f"{q}.attn1.norm_q.weight"] = take(f"{p}.attn.q_norm.weight")
        conv[f"{q}.attn1.norm_k.weight"] = take(f"{p}.attn.k_norm.weight")
        conv[f"{q}.attn1.to_out.0.weight"] = take(f"{p}.attn.proj.weight")
        conv[f"{q}.attn1.to_out.0.bias"] = take(f"{p}.attn.proj.bias")

        conv[f"{q}.ff.conv_inverted.weight"] = take(f"{p}.mlp.inverted_conv.conv.weight")
        conv[f"{q}.ff.conv_inverted.bias"] = take(f"{p}.mlp.inverted_conv.conv.bias")
        conv[f"{q}.ff.conv_depth.weight"] = take(f"{p}.mlp.depth_conv.conv.weight")
        conv[f"{q}.ff.conv_depth.bias"] = take(f"{p}.mlp.depth_conv.conv.bias")
        conv[f"{q}.ff.conv_point.weight"] = take(f"{p}.mlp.point_conv.conv.weight")

        conv[f"{q}.attn2.to_q.weight"] = take(f"{p}.cross_attn.q_linear.weight")
        conv[f"{q}.attn2.to_q.bias"] = take(f"{p}.cross_attn.q_linear.bias")
        kb, vb = torch.chunk(take(f"{p}.cross_attn.kv_linear.weight"), 2, dim=0)
        kbb, vbb = torch.chunk(take(f"{p}.cross_attn.kv_linear.bias"), 2, dim=0)
        conv[f"{q}.attn2.to_k.weight"] = kb
        conv[f"{q}.attn2.to_k.bias"] = kbb
        conv[f"{q}.attn2.to_v.weight"] = vb
        conv[f"{q}.attn2.to_v.bias"] = vbb
        conv[f"{q}.attn2.norm_q.weight"] = take(f"{p}.cross_attn.q_norm.weight")
        conv[f"{q}.attn2.norm_k.weight"] = take(f"{p}.cross_attn.k_norm.weight")
        conv[f"{q}.attn2.to_out.0.weight"] = take(f"{p}.cross_attn.proj.weight")
        conv[f"{q}.attn2.to_out.0.bias"] = take(f"{p}.cross_attn.proj.bias")

    conv["proj_out.weight"] = take("final_layer.linear.weight")
    conv["proj_out.bias"] = take("final_layer.linear.bias")
    conv["scale_shift_table"] = take("final_layer.scale_shift_table")

    transformer = None
    with init_empty_weights():
        transformer = SanaTransformer2DModel(
            in_channels=32,
            out_channels=32,
            num_attention_heads=70,
            attention_head_dim=32,
            num_layers=20,
            num_cross_attention_heads=20,
            cross_attention_head_dim=112,
            cross_attention_dim=2240,
            caption_channels=2304,
            mlp_ratio=2.5,
            attention_bias=False,
            sample_size=32,
            patch_size=1,
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            interpolation_scale=None,
            timestep_scale=1.0,
            qk_norm="rms_norm_across_heads",
        )
    try:
        load_model_dict_into_meta(transformer, conv, dtype=torch.bfloat16)
    except TypeError:  # older/newer signature
        load_model_dict_into_meta(transformer, conv)
        transformer = transformer.to(torch.bfloat16)

    for leftover in ("y_embedder.y_embedding", "pos_embed", "logvar_linear.weight", "logvar_linear.bias"):
        state.pop(leftover, None)
    if state:
        log(f"คำเตือน: มีคีย์ที่ไม่ได้ใช้ {len(state)} รายการ")
    del conv, state, raw
    gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    del transformer
    gc.collect()


class Cancelled(Exception):
    pass


# ── inpainting (SDXL) ──────────────────────────────────────────────────────
def _load_pil_image(path: str) -> "PIL.Image":
    from PIL import Image
    return Image.open(path).convert("RGB")


def _prepare_inpaint_inputs(
    init_image_path: str, mask_image_path: str, width: int, height: int
) -> tuple["PIL.Image", "PIL.Image"]:
    from PIL import Image
    init = Image.open(init_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    mask = Image.open(mask_image_path).convert("L").resize((width, height), Image.NEAREST)
    return init, mask


# ── hires fix (SANA / SDXL / SD 1.5) ───────────────────────────────────────
# Classic two-pass hires fix: generate at the base size, upscale with a
# resampling filter, then run a low-strength img2img pass to restore detail.
# SDXL / SD 1.5: the img2img pipeline is loaded via ensure_ready(kind="img2img")
# which drops the t2i pipeline first (from_pipe() caused OOM on this machine).
# SANA: diffusers ships no SanaImg2ImgPipeline — ModelManager._sana_hires_refine
# implements the img2img loop manually with the SAME SanaPipeline, so SANA
# needs no pipeline reload at all.
HIRES_MAX = 2048


def _hires_params(params: dict, family: str | None, width: int, height: int) -> dict | None:
    """Return {"scale", "strength", "width", "height"} or None when disabled."""
    if not isinstance(params, dict):
        return None
    cfg = params.get("hires")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    if family not in HIRES_FAMILIES or params.get("inpaint"):
        return None
    try:
        scale = float(cfg.get("scale", 1.5))
    except (TypeError, ValueError):
        scale = 1.5
    try:
        strength = float(cfg.get("strength", 0.35))
    except (TypeError, ValueError):
        strength = 0.35
    scale = max(1.0, min(2.0, scale))
    strength = max(0.05, min(0.75, strength))
    tw, th = float(width) * scale, float(height) * scale
    if max(tw, th) > HIRES_MAX:
        factor = HIRES_MAX / max(tw, th)
        tw, th = tw * factor, th * factor
    tw, th = norm_size(tw), norm_size(th)
    if tw <= width or th <= height:
        return None
    return {"scale": round(scale, 3), "strength": round(strength, 3), "width": tw, "height": th}


def _upscale_image(image, width: int, height: int):
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resample)


def _filter_pipe_kwargs(pipe, kwargs: dict) -> dict:
    """Drop keyword arguments a pipeline's __call__ does not accept.

    Generic diffusers pipelines differ in what they take (e.g. some have no
    `negative_prompt` or no `callback_on_step_end`).  Filtering keeps newly
    detected architectures usable without a per-model branch.
    """
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[pipeline] ไม่รองรับพารามิเตอร์: {', '.join(dropped)}", flush=True)
    return {k: v for k, v in kwargs.items() if k in params}


# ── regional prompting (SDXL) ──────────────────────────────────────────────
# Each region gets its own text embedding. Inside cross-attention we route the
# image tokens of that box to its prompt, so two characters stop bleeding into
# each other. Boxes are normalised (x, y, w, h in 0..1) and applied per UNet
# resolution by inferring H/W from the latent size.
REGION_MAX = 6


def normalize_regions(raw) -> list[dict]:
    out: list[dict] = []
    for item in raw or []:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        prompt = " ".join(str(item.get("prompt") or "").split())
        if not prompt:
            continue
        try:
            weight = float(item.get("weight", 1.0))
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            w = float(item.get("w", 1.0))
            h = float(item.get("h", 1.0))
        except (TypeError, ValueError):
            continue
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.02, min(1.0 - x, w))
        h = max(0.02, min(1.0 - y, h))
        out.append(
            {
                "prompt": prompt[:2000],
                "weight": round(max(0.0, min(1.5, weight)), 3),
                "x": round(x, 5),
                "y": round(y, 5),
                "w": round(w, 5),
                "h": round(h, 5),
                "label": " ".join(str(item.get("label") or "").split())[:40],
            }
        )
        if len(out) >= REGION_MAX:
            break
    return out


class RegionalAttnProcessor:
    """Wraps a diffusers attention processor and routes queries to region prompts."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        regional=None,
        **kwargs,
    ):
        if (
            not regional
            or not getattr(attn, "is_cross_attention", False)
            or encoder_hidden_states is None
            or attention_mask is not None
        ):
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
            )
        try:
            return self._run(attn, hidden_states, encoder_hidden_states, temb, regional)
        except Exception as exc:  # noqa: BLE001
            print("regional attention warning:", exc, flush=True)
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
            )

    def _run(self, attn, hidden_states, encoder_hidden_states, temb, regional):
        batch, seq, _ = hidden_states.shape
        lat_h = int(regional.get("latent_h") or 0)
        lat_w = int(regional.get("latent_w") or 0)
        h = w = 0
        for level in range(6):
            ch, cw = lat_h >> level, lat_w >> level
            if ch > 0 and cw > 0 and ch * cw == seq:
                h, w = ch, cw
                break
        if not h:
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                temb=temb,
            )

        device = hidden_states.device
        region_id = torch.full((seq,), -1, dtype=torch.long, device=device)
        for index, region in enumerate(regional.get("regions") or []):
            x0 = int(round(float(region["x"]) * w))
            y0 = int(round(float(region["y"]) * h))
            x1 = int(round((float(region["x"]) + float(region["w"])) * w))
            y1 = int(round((float(region["y"]) + float(region["h"])) * h))
            x0, x1 = max(0, min(w - 1, x0)), max(1, min(w, x1))
            y0, y1 = max(0, min(h - 1, y0)), max(1, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = torch.zeros((h, w), dtype=torch.bool, device=device)
            mask[y0:y1, x0:x1] = True
            flat = mask.reshape(-1)
            region_id[(region_id < 0) & flat] = index

        covered = region_id >= 0
        if not bool(covered.any()):
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                temb=temb,
            )

        embeds = regional.get("embeds") or []
        cfg = bool(regional.get("cfg"))
        n = hidden_states.shape[0] // (2 if cfg else 1)
        n = max(1, n)
        if cfg:
            uncond = encoder_hidden_states[:n]
            global_cond = encoder_hidden_states[n : n * 2]
        else:
            uncond = None
            global_cond = encoder_hidden_states

        out = torch.empty_like(hidden_states)
        for index, region in enumerate(regional.get("regions") or []):
            idx = (region_id == index).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0 or index >= len(embeds):
                continue
            emb = embeds[index].to(device=device, dtype=hidden_states.dtype)
            if emb.shape[0] != n:
                emb = emb.expand(n, -1, -1)
            if cfg:
                weight = float(region.get("weight", 1.0))
                cond = global_cond + (emb - global_cond) * weight
                ehs = torch.cat([uncond, cond], dim=0)
            else:
                ehs = emb
            out[:, idx, :] = self.inner(
                attn,
                hidden_states[:, idx, :],
                encoder_hidden_states=ehs,
                attention_mask=None,
                temb=temb,
            )

        uncovered = (region_id < 0).nonzero(as_tuple=False).flatten()
        if uncovered.numel():
            out[:, uncovered, :] = self.inner(
                attn,
                hidden_states[:, uncovered, :],
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                temb=temb,
            )
        return out


def install_regional_processors(pipe) -> int:
    unet = getattr(pipe, "unet", None)
    if unet is None:
        return 0
    count = 0
    for name, module in unet.named_modules():
        if not name.endswith(("attn1", "attn2")):
            continue
        inner = getattr(module, "processor", None)
        if inner is None or isinstance(inner, RegionalAttnProcessor):
            continue
        module.set_processor(RegionalAttnProcessor(inner))
        count += 1
    return count


def encode_sdxl_regions(pipe, prompt: str, negative_prompt: str, regions: list[dict]):
    """Encode the global prompt (with CFG) and one embedding per region."""
    device = pipe._execution_device
    prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=negative_prompt or "",
        negative_prompt_2=negative_prompt or "",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=device,
    )
    prompts = [r["prompt"] for r in regions]
    res = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        negative_prompt=None,
        negative_prompt_2=None,
        do_classifier_free_guidance=False,
        num_images_per_prompt=1,
        device=device,
    )
    region_embeds = res[0]
    embeds = [region_embeds[i : i + 1] for i in range(region_embeds.shape[0])]
    return prompt_embeds, neg_embeds, pooled, neg_pooled, embeds


def save_image_file(image, job_id, index, seed, meta) -> dict:
    """Save an image as PNG (with A1111 metadata) + WebP thumbnail.

    Shared by the generation pipelines and the ESRGAN upscaler.  `meta` uses
    the generation-parameter keys; anything missing falls back to a neutral
    value so an upscale job can reuse the source parameters.  `meta["extra"]`
    is appended to the metadata line (e.g. the upscale source/scale).
    """
    from PIL import PngImagePlugin

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{str(job_id)[:8]}_{index}.png"
    path = IMAGES / name

    lora_line = ", ".join(
        f"{x.get('name') or x.get('id') or '?'}:{x.get('weight', 1)}"
        for x in meta.get("loras") or []
    )
    details = (
        f"Steps: {meta.get('steps', 0)}, Sampler: {meta.get('sampler', '')}, "
        f"CFG scale: {meta.get('guidance', 0)}, Seed: {seed}, "
        f"Size: {meta.get('width', image.width)}x{meta.get('height', image.height)}, "
        f"Model: {meta.get('model', '')}"
    )
    if lora_line:
        details += f", LoRA: {lora_line}"
    if meta.get("hires"):
        hires = meta["hires"]
        details += f", Hires: {hires['scale']}x, Hires denoise: {hires['strength']}"
    region_line = " | ".join(
        f"{r.get('label') or 'region'}: {r['prompt']} ({r['weight']})"
        for r in meta.get("regions") or []
    )
    if region_line:
        details += f", Regional: {region_line}"
    if meta.get("extra"):
        details += f", {meta['extra']}"
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "\n".join(
            [
                meta.get("prompt", ""),
                f"Negative prompt: {meta.get('negative_prompt', '')}",
                details,
            ]
        ),
    )
    info.add_text("prompt", meta.get("prompt", ""))
    info.add_text("seed", str(seed))
    info.add_text("model", meta.get("model", ""))
    if lora_line:
        info.add_text("lora", lora_line)
    image.save(path, "PNG", pnginfo=info)

    thumb_name = name[:-4] + ".webp"
    try:
        thumb = image.copy()
        thumb.thumbnail((640, 640))
        thumb.save(THUMBS / thumb_name, "WEBP", quality=86, method=3)
    except Exception as exc:  # noqa: BLE001
        print("thumbnail failed:", exc, flush=True)
        thumb_name = None

    return {
        "file": name,
        "url": f"/files/images/{name}",
        "thumb": f"/files/thumbs/{thumb_name}" if thumb_name else f"/files/images/{name}",
        "width": image.width,
        "height": image.height,
        "seed": seed,
        "elapsed": meta.get("elapsed"),
    }


class ModelManager:
    """Loads SANA lazily, keeps one image pipeline in memory, serialises jobs."""

    def __init__(self):
        self.pipe = None
        self.family = None
        self.pipe_kind = None
        self.current_id = MODEL_ID
        self.models: list[dict] = []
        self.state = "not_loaded"
        self.message = ""
        self.error = None
        self.offload = None
        self.sampler = None
        self.active_loras: list[dict] = []
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._scheduler_key = None
        self._loras_signature = None
        self.reload_models()

    # -- status ---------------------------------------------------------------
    def _set(self, state, message=""):
        with self._state_lock:
            self.state = state
            self.message = message
        print(f"[model] {state}: {message}", flush=True)

    def reload_models(self) -> list[dict]:
        self.models = discover_models()
        ids = {m["id"] for m in self.models}
        if self.current_id not in ids:
            self.current_id = self.models[0]["id"] if self.models else None
        return self.models

    def get_model(self, model_id: str | None = None) -> dict:
        if not self.models:
            self.reload_models()
        mid = model_id or self.current_id
        for entry in self.models:
            if entry["id"] == mid:
                return entry
        if mid is None and self.models:
            return self.models[0]
        raise ValueError("ไม่พบโมเดลที่เลือก — กดรีเฟรชรายการโมเดล")

    @property
    def label(self) -> str:
        try:
            return self.get_model()["name"]
        except ValueError:
            return MODEL_LABEL

    @property
    def defaults(self) -> dict:
        try:
            return self.get_model().get("defaults") or FAMILY_DEFAULTS["sana"]
        except ValueError:
            return FAMILY_DEFAULTS["sana"]

    def snapshot(self) -> dict:
        try:
            entry = self.get_model()
        except ValueError:
            entry = None
        return {
            "id": entry["id"] if entry else MODEL_ID,
            "name": entry["name"] if entry else MODEL_LABEL,
            "family": entry["family"] if entry else None,
            "family_label": entry.get("family_label") if entry else FAMILY_LABEL["sana"],
            "state": self.state,
            "message": self.message,
            "loaded": self.pipe is not None,
            "files_ready": entry.get("available", False) if entry else False,
            "offload": self.offload,
            "sampler": self.sampler,
            "model_dir": entry["path"] if entry else str(MODEL_DIR),
            "capabilities": entry.get("capabilities", {}) if entry else {},
            "defaults": entry.get("defaults", FAMILY_DEFAULTS["sana"]) if entry else FAMILY_DEFAULTS["sana"],
            "loras": list(self.active_loras),
            "models_count": len(self.models),
        }

    def list_models(self) -> dict:
        self.reload_models()
        return {"models": self.models, "current": self.current_id}

    def list_loras(self) -> dict:
        return {"loras": discover_loras(), "current": self.current_id}

    def import_lora(self, path: str) -> dict:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
        if src.suffix.lower() != ".safetensors":
            # discover_loras() only scans .safetensors, so copying a .pt/.pth
            # here would leave an orphan file that never appears in the app.
            raise ValueError("ต้องเป็นไฟล์ .safetensors")
        # Validate before copying: an unreadable / empty file would otherwise
        # be copied and then silently skipped by discover_loras().
        try:
            from safetensors import safe_open

            with safe_open(str(src), framework="pt") as handle:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError("ไม่มี tensor")
                handle.get_tensor(keys[0])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"อ่านไฟล์ LoRA ไม่ได้: {exc}") from exc
        dest_dir = MODELS_ROOT / "loras"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists() and dest.samefile(src):
            raise ValueError("LoRA นี้อยู่ใน models/loras อยู่แล้ว")
        if dest.exists():
            raise ValueError("มีไฟล์ชื่อเดียวกันอยู่ใน models/loras แล้ว")
        try:
            shutil.copyfile(src, dest)
            for lora in discover_loras():
                if Path(lora["path"]).resolve() == dest.resolve():
                    return lora
            raise ValueError("คัดลอกแล้ว แต่ระบบหา LoRA ไม่เจอ")
        except Exception:
            # Never leave a half-copied / unusable file behind.
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def delete_lora(self, lora_id: str) -> None:
        loras = discover_loras()
        found = [l for l in loras if l["id"] == lora_id]
        if not found:
            raise ValueError(f"ไม่พบ LoRA: {lora_id}")
        p = Path(found[0]["path"])
        p.unlink()

    def select_model(self, model_id: str) -> dict:
        entry = self.get_model(model_id)
        if not entry.get("supported", True):
            reason = entry.get("unsupported_reason") or "ยังไม่รองรับสถาปัตยกรรมนี้"
            raise ValueError(f"โมเดล {entry['name']} ใช้ไม่ได้: {reason}")
        with self._load_lock:
            had_pipe = self.pipe is not None
            self._drop_pipe()
            self.current_id = entry["id"]
            if had_pipe:
                msg = f"เปลี่ยนเป็น {entry['name']} — คืนหน่วยความจำโมเดลเดิมแล้ว"
            else:
                msg = f"เลือก {entry['name']} แล้ว — จะโหลดเมื่อสร้างภาพ"
            self._set("not_loaded", msg)
        return entry

    # -- loading --------------------------------------------------------------
    def ensure_ready(self, status_cb=None, kind: str = "t2i"):
        if self.pipe is not None and self.pipe_kind != kind:
            self._drop_pipe()
            self._set(
                "not_loaded",
                f"เปลี่ยนเป็นโหมด {kind} — ต้องโหลดโมเดลใหม่",
            )
        if self.pipe is not None:
            return self.pipe
        with self._load_lock:
            if self.pipe is not None:
                return self.pipe
            entry = self.get_model()
            if not entry.get("available"):
                raise RuntimeError(
                    f"โมเดล {entry['name']} ยังไม่พร้อม — ตรวจไฟล์ใน {entry['path']}"
                )
            if not entry.get("supported", True):
                reason = entry.get("unsupported_reason") or "ยังไม่รองรับสถาปัตยกรรมนี้"
                raise RuntimeError(f"โมเดล {entry['name']} ใช้ไม่ได้: {reason}")
            if torch is None:
                raise RuntimeError(
                    "ยังไม่ได้ติดตั้ง PyTorch — รัน engine/setup.sh ก่อน "
                    "(หรือตรวจว่า venv ใช้งานได้)"
                )
            family = entry["family"]
            profile = FAMILY_PROFILES.get(family) or FAMILY_PROFILES["generic"]
            model_path = Path(entry["path"])
            if family == "sana" and not files_ready(model_path):
                if model_path != MODEL_DIR:
                    raise RuntimeError(f"ไฟล์โมเดล SANA ไม่ครบ: {model_path}")
                self._set("preparing", "กำลังดาวน์โหลดโมเดลครั้งแรก (~9 GB)…")
                if status_cb:
                    status_cb("กำลังดาวน์โหลดโมเดลครั้งแรก (~9 GB) — ต้องต่ออินเทอร์เน็ต")
                ensure_model_files(log=lambda m: print("[prepare]", m, flush=True))

            loading_msg = f"กำลังโหลด {entry['name']}…"
            self._set("loading", loading_msg)
            if status_cb:
                status_cb(loading_msg)
            dtype_name = profile.get("dtype", "float16")
            dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
            pipe = None
            if family == "sana":
                from diffusers import SanaPipeline

                pipe = SanaPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                pipe.set_progress_bar_config(disable=True)
                try:
                    pipe.text_encoder.to(torch.bfloat16)
                    pipe.vae.to(torch.bfloat16)
                except Exception as exc:  # noqa: BLE001
                    print("dtype move warning:", exc, flush=True)
            else:
                # Resolve the pipeline class: known families use the table,
                # generic ones carry the class name from model_index.json.
                classes = PIPELINE_CLASSES.get(family) or {}
                class_name = classes.get(kind) or classes.get("t2i") or entry.get("pipeline_class") or ""
                if kind != "t2i" and kind not in classes:
                    raise RuntimeError(
                        f"ตระกูล {FAMILY_LABEL.get(family, family)} ไม่มี pipeline สำหรับโหมด {kind}"
                    )
                if not class_name:
                    raise RuntimeError(
                        f"ยังไม่รองรับตระกูลโมเดล: {family} (ไม่พบ pipeline class)"
                    )
                import diffusers

                Pipeline = getattr(diffusers, class_name, None)
                if Pipeline is None:
                    raise RuntimeError(
                        f"diffusers ที่ติดตั้งไม่มี {class_name} — อัปเดต diffusers ก่อน"
                    )
                if entry["kind"] == "single":
                    pipe = Pipeline.from_single_file(
                        str(model_path), torch_dtype=dtype, use_safetensors=True
                    )
                else:
                    pipe = Pipeline.from_pretrained(
                        str(model_path), torch_dtype=dtype, low_cpu_mem_usage=True
                    )
                pipe.set_progress_bar_config(disable=True)
                if dtype_name == "bfloat16":
                    # Some pipelines keep fp32 components (text encoders / VAE)
                    # after loading; match the requested dtype where possible.
                    for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                        component = getattr(pipe, attr, None)
                        if component is not None and hasattr(component, "to"):
                            try:
                                component.to(dtype)
                            except Exception:  # noqa: BLE001
                                pass

            try:
                pipe.enable_model_cpu_offload()
                self.offload = "model"
            except Exception as exc:  # noqa: BLE001
                print("model_cpu_offload unavailable:", exc, flush=True)
                pipe.enable_sequential_cpu_offload()
                self.offload = "sequential"

            self.pipe = pipe
            self.family = family
            self.pipe_kind = kind
            if CAPABILITIES.get(family, {}).get("regional"):
                try:
                    count = install_regional_processors(pipe)
                    print(f"[regional] wrapped {count} attention layers", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("regional install warning:", exc, flush=True)
            self.sampler = None
            self._scheduler_key = None
            self._loras_signature = None
            self.active_loras = []
            self.error = None
            self._set("ready", f"พร้อมใช้งาน · {entry['name']} (offload: {self.offload})")
            return pipe

    def load_async(self):
        def run():
            try:
                self.ensure_ready()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.error = str(exc)
                self._set("error", str(exc))

        t = threading.Thread(target=run, name="sana-load", daemon=True)
        t.start()
        return t

    def _drop_pipe(self):
        if self.pipe is None:
            return
        self.pipe = None
        self.family = None
        self.pipe_kind = None
        self.sampler = None
        self._scheduler_key = None
        self._loras_signature = None
        self.active_loras = []
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass

    def unload(self):
        with self._load_lock:
            had_pipe = self.pipe is not None
            self._drop_pipe()
            if had_pipe:
                self._set("not_loaded", "ปล่อยหน่วยความจำแล้ว")

    # -- sampler --------------------------------------------------------------
    def _set_sampler(self, name: str):
        if self.pipe is None:
            return
        name = (name or "dpm").lower()
        if name not in ("dpm", "euler"):
            name = "dpm"
        key = (self.family, name)
        if key == self._scheduler_key:
            self.sampler = name
            return
        if self.family == "sana":
            if name == "euler":
                from diffusers import FlowMatchEulerDiscreteScheduler

                self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
            else:
                from diffusers import DPMSolverMultistepScheduler

                self.pipe.scheduler = DPMSolverMultistepScheduler(
                    flow_shift=3.0,
                    use_flow_sigmas=True,
                    prediction_type="flow_prediction",
                )
        else:
            cfg = self.pipe.scheduler.config
            if name == "euler":
                from diffusers import EulerDiscreteScheduler

                self.pipe.scheduler = EulerDiscreteScheduler.from_config(cfg)
            else:
                from diffusers import DPMSolverMultistepScheduler

                try:
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        cfg, use_karras_sigmas=True
                    )
                except Exception:  # noqa: BLE001
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
        self._scheduler_key = key
        self.sampler = name

    # -- LoRA ----------------------------------------------------------------
    def _resolve_loras(self, refs) -> list[dict]:
        if not refs:
            return []
        available = {item["id"]: item for item in discover_loras()}
        resolved: list[dict] = []
        for ref in refs:
            if isinstance(ref, str):
                lora_id, weight = ref, 1.0
            elif isinstance(ref, dict):
                lora_id = str(ref.get("id") or "")
                weight = ref.get("weight", 1.0)
            else:
                continue
            item = available.get(lora_id)
            if item is None:
                raise ValueError(f"ไม่พบ LoRA ที่เลือก (id: {lora_id})")
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            resolved.append({**item, "weight": round(max(0.0, min(2.0, weight)), 3)})
        return resolved

    def _apply_loras(self, refs) -> list[dict]:
        resolved = self._resolve_loras(refs)
        signature = tuple((item["path"], item["weight"]) for item in resolved)
        if signature == self._loras_signature:
            return resolved
        if self.pipe is None:
            return resolved
        if self._loras_signature is not None:
            try:
                self.pipe.unload_lora_weights()
            except Exception as exc:  # noqa: BLE001
                print("lora unload warning:", exc, flush=True)
        self._loras_signature = None
        self.active_loras = []
        if not resolved:
            self._loras_signature = signature
            return resolved
        names: list[str] = []
        weights: list[float] = []
        try:
            for index, item in enumerate(resolved):
                adapter = f"user{index}"
                self.pipe.load_lora_weights(item["path"], adapter_name=adapter)
                names.append(adapter)
                weights.append(item["weight"])
            self.pipe.set_adapters(names, adapter_weights=weights)
        except Exception as exc:  # noqa: BLE001
            try:
                self.pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            self._loras_signature = None
            self.active_loras = []
            family = FAMILY_LABEL.get(self.family or "", self.family or "นี้")
            raise RuntimeError(
                f"โหลด LoRA ไม่สำเร็จ — ไฟล์อาจไม่เข้ากับโมเดล {family}: {exc}"
            ) from exc
        self._loras_signature = signature
        self.active_loras = [
            {"id": item["id"], "name": item["name"], "weight": item["weight"]}
            for item in resolved
        ]
        return resolved

    # -- generation -----------------------------------------------------------
    def generate(self, params: dict, job: dict, cancel: threading.Event):
        requested = params.get("model_id") or self.current_id
        if requested != self.current_id:
            self.select_model(requested)
        requested_family = self.get_model(requested).get("family")
        kind = (
            "inpaint"
            if params.get("inpaint")
            and CAPABILITIES.get(requested_family, {}).get("inpaint")
            else "t2i"
        )
        self.ensure_ready(
            status_cb=lambda m: job.update(
                {"phase": "preparing", "message": m, "status": "running"}
            ),
            kind=kind,
        )
        entry = self.get_model()
        defaults = entry.get("defaults") or FAMILY_DEFAULTS["sana"]
        p = params
        width = norm_size(p.get("width") or defaults["width"], fallback=defaults["width"])
        height = norm_size(p.get("height") or defaults["height"], fallback=defaults["height"])
        steps = max(2, min(60, int(p.get("steps") or defaults["steps"])))
        guidance = float(
            p.get("guidance") if p.get("guidance") is not None else defaults["guidance"]
        )
        batch = max(1, min(4, int(p.get("batch") or 1)))
        seed = int(p.get("seed") if p.get("seed") is not None else -1)
        enhance = bool(p.get("enhance", defaults.get("enhance", False))) and bool(
            CAPABILITIES.get(self.family, {}).get("enhance")
        )
        # SANA-only prompt cleanup kwarg; other pipelines ignore it.
        clean_caption = bool(p.get("clean_caption", False)) and self.family == "sana"
        max_seq = max(64, min(512, int(p.get("max_sequence_length") or 300)))
        sampler = p.get("sampler") or defaults.get("sampler") or "dpm"
        hires = _hires_params(p, self.family, width, height)
        hires_steps = max(1, int(round(steps * hires["strength"]))) if hires else 0
        total_steps = steps + hires_steps

        base_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        loras = self._apply_loras(p.get("loras") or [])
        self._set_sampler(sampler)

        regions = normalize_regions(p.get("regions")) if CAPABILITIES.get(self.family, {}).get("regional") else []
        regional = None
        if regions:
            try:
                p_emb, n_emb, pooled, n_pooled, region_embeds = encode_sdxl_regions(
                    self.pipe,
                    p.get("prompt") or "",
                    p.get("negative_prompt") or "",
                    regions,
                )
                regional = {
                    "regions": [
                        {k: r[k] for k in ("x", "y", "w", "h", "weight")}
                        for r in regions
                    ],
                    "embeds": region_embeds,
                    "cfg": guidance > 1,
                    "latent_h": height // 8,
                    "latent_w": width // 8,
                }
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"เตรียม regional prompt ไม่สำเร็จ: {exc}") from exc
        regions_meta = [
            {
                "prompt": r["prompt"],
                "weight": r["weight"],
                "label": r["label"],
                "x": r["x"],
                "y": r["y"],
                "w": r["w"],
                "h": r["h"],
            }
            for r in regions
        ]

        pipe_to_use = self.pipe

        job.update(
            {
                "phase": "generating",
                "message": "กำลังสร้างภาพ",
                "status": "running",
                "steps": total_steps,
                "batch": batch,
                "progress": 0.0,
                "step": 0,
                "image_index": 0,
                "model_id": entry["id"],
                "model_label": entry["name"],
                "family": self.family,
                "loras": [{"name": x["name"], "weight": x["weight"]} for x in loras],
                "regions": regions_meta,
            }
        )

        generated = []
        base_results = []  # (image, seed, elapsed) — kept only when hires is on
        for i in range(batch):
            if cancel.is_set():
                raise Cancelled()
            image_seed = (base_seed + i) % (2**31)
            generator = torch.Generator(device="cuda").manual_seed(image_seed)
            job["image_index"] = i
            job["step"] = 0
            job["message"] = f"กำลังสร้างภาพ {i + 1}/{batch}"

            def on_step(pipe, step_index, timestep, callback_kwargs):
                if cancel.is_set():
                    raise Cancelled()
                job["step"] = step_index + 1
                job["progress"] = round((i + (step_index + 1) / total_steps) / batch, 4)
                return callback_kwargs

            if self.family == "sana":
                kwargs = dict(
                    prompt=p.get("prompt") or "",
                    negative_prompt=p.get("negative_prompt") or "",
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                    max_sequence_length=max_seq,
                    clean_caption=clean_caption,
                    use_resolution_binning=False,
                )
                if not enhance:
                    # [] disables SANA's in-context "complex human instruction"
                    kwargs["complex_human_instruction"] = []
            elif p.get("inpaint") and CAPABILITIES.get(self.family, {}).get("inpaint"):
                init_path = p.get("init_image")
                mask_path = p.get("mask_image")
                if not init_path or not mask_path:
                    raise RuntimeError("inpaint ต้องการ init_image และ mask_image")
                init_image, mask_image = _prepare_inpaint_inputs(init_path, mask_path, width, height)
                # self.pipe is already StableDiffusionXLInpaintPipeline (ensure_ready kind="inpaint")
                kwargs = dict(
                    prompt=p.get("prompt") or "",
                    negative_prompt=p.get("negative_prompt") or "",
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    strength=float(p.get("strength") or 0.75),
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            elif regional is not None:
                kwargs = dict(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=p_emb,
                    negative_prompt_embeds=n_emb,
                    pooled_prompt_embeds=pooled,
                    negative_pooled_prompt_embeds=n_pooled,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                    cross_attention_kwargs={"regional": regional},
                )
            else:
                kwargs = dict(
                    prompt=p.get("prompt") or "",
                    negative_prompt=p.get("negative_prompt") or "",
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                )

            t0 = time.perf_counter()
            output = pipe_to_use(**_filter_pipe_kwargs(pipe_to_use, kwargs))
            elapsed = time.perf_counter() - t0
            image = output.images[0]

            if hires:
                base_results.append((image, image_seed, elapsed))
                job["progress"] = round((i + steps / total_steps) / batch, 4)
                continue

            meta = {
                "model": entry["name"],
                "model_id": entry["id"],
                "family": self.family,
                "loras": [{"name": x["name"], "weight": x["weight"]} for x in loras],
                "prompt": p.get("prompt") or "",
                "negative_prompt": p.get("negative_prompt") or "",
                "regions": regions_meta,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": image_seed,
                "sampler": sampler,
                "enhance": enhance,
                "elapsed": round(elapsed, 2),
            }
            record = self._save_image(image, job["id"], i, image_seed, meta)
            job["images"].append(record)
            job["progress"] = round((i + 1) / batch, 4)
            generated.append(record)

        if hires:
            # Release the t2i pipeline reference BEFORE the img2img reload —
            # a lingering local ref keeps the old pipeline uncollectable during
            # _drop_pipe(), putting two SDXL pipelines (~14 GB) in RAM => OOM.
            pipe_to_use = None
            generated.extend(
                self._run_hires_pass(
                    p, job, cancel,
                    base_results, hires, total_steps,
                    entry, batch, steps, guidance, sampler,
                    loras, regions_meta, regional, enhance,
                    clean_caption, max_seq,
                )
            )

        return generated

    def _run_hires_pass(
        self,
        p: dict,
        job: dict,
        cancel: threading.Event,
        base_results: list,
        hires: dict,
        total_steps: int,
        entry: dict,
        batch: int,
        steps: int,
        guidance: float,
        sampler: str,
        loras: list,
        regions_meta: list,
        regional: dict | None,
        enhance: bool,
        clean_caption: bool,
        max_seq: int,
    ) -> list:
        """Upscale base images then refine with a low-strength img2img pass.

        SDXL / SD 1.5: the t2i pipeline is dropped first (ensure_ready
        kind="img2img") — never keep two pipeline classes in memory on this
        15 GB / no-swap machine.
        SANA: reuses the SAME SanaPipeline with a manual img2img loop
        (_sana_hires_refine) — no reload at all.
        """
        def status(msg):
            job.update({"phase": "preparing", "message": msg, "status": "running"})

        regional2 = None
        if self.family == "sana":
            status(
                f"hires (SANA) — ขยายเป็น {hires['width']}×{hires['height']} (ใช้ pipeline เดิม)"
            )
            try:
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
            except Exception as exc:  # noqa: BLE001
                print("vae tiling warning:", exc, flush=True)
        else:
            status(
                f"เปลี่ยนเป็นโหมด hires ({hires['width']}×{hires['height']}) — โหลด pipeline ใหม่"
            )
            self.ensure_ready(status_cb=status, kind="img2img")
            self._set_sampler(sampler)
            loras = self._apply_loras(p.get("loras") or [])
            job["loras"] = [{"name": x["name"], "weight": x["weight"]} for x in loras]
            try:
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
            except Exception as exc:  # noqa: BLE001
                print("vae tiling warning:", exc, flush=True)
            if regional is not None:
                regional2 = {
                    **regional,
                    "latent_h": hires["height"] // 8,
                    "latent_w": hires["width"] // 8,
                }

        generated = []
        for i, (base_image, image_seed, base_elapsed) in enumerate(base_results):
            if cancel.is_set():
                raise Cancelled()
            generator = torch.Generator(device="cuda").manual_seed(image_seed)
            job["image_index"] = i
            job["step"] = steps
            job["message"] = f"เก็บรายละเอียด (hires) {i + 1}/{batch}"

            def on_hires_step(pipe, step_index, timestep, callback_kwargs):
                if cancel.is_set():
                    raise Cancelled()
                job["step"] = steps + step_index + 1
                job["progress"] = round(
                    (i + (steps + step_index + 1) / total_steps) / batch, 4
                )
                return callback_kwargs

            up_image = _upscale_image(base_image, hires["width"], hires["height"])
            t0 = time.perf_counter()
            if self.family == "sana":
                image = self._sana_hires_refine(
                    image=up_image,
                    prompt=p.get("prompt") or "",
                    negative_prompt=p.get("negative_prompt") or "",
                    guidance=guidance,
                    steps=steps,
                    strength=hires["strength"],
                    clean_caption=clean_caption,
                    max_seq=max_seq,
                    enhance=enhance,
                    generator=generator,
                    cancel=cancel,
                    on_step=lambda idx: on_hires_step(None, idx, None, None),
                )
            else:
                # note: img2img pipelines have no width/height args — the output
                # follows the (pre-upscaled) input image size exactly.
                kwargs = dict(
                    prompt=p.get("prompt") or "",
                    negative_prompt=p.get("negative_prompt") or "",
                    image=up_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    strength=hires["strength"],
                    generator=generator,
                    callback_on_step_end=on_hires_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
                if regional2 is not None:
                    kwargs.update(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        pooled_prompt_embeds=pooled,
                        negative_pooled_prompt_embeds=n_pooled,
                        cross_attention_kwargs={"regional": regional2},
                    )
                output = self.pipe(**_filter_pipe_kwargs(self.pipe, kwargs))
                image = output.images[0]
            hires_elapsed = time.perf_counter() - t0

            meta = {
                "model": entry["name"],
                "model_id": entry["id"],
                "family": self.family,
                "loras": [{"name": x["name"], "weight": x["weight"]} for x in loras],
                "prompt": p.get("prompt") or "",
                "negative_prompt": p.get("negative_prompt") or "",
                "regions": regions_meta,
                "width": hires["width"],
                "height": hires["height"],
                "steps": steps,
                "guidance": guidance,
                "seed": image_seed,
                "sampler": sampler,
                "enhance": enhance,
                "elapsed": round(base_elapsed + hires_elapsed, 2),
                "hires": hires,
            }
            record = self._save_image(image, job["id"], i, image_seed, meta)
            job["images"].append(record)
            job["progress"] = round((i + 1) / batch, 4)
            generated.append(record)

        return generated

    @torch.no_grad()
    def _sana_hires_refine(
        self,
        image,
        prompt: str,
        negative_prompt: str,
        guidance: float,
        steps: int,
        strength: float,
        clean_caption: bool,
        max_seq: int,
        enhance: bool,
        generator,
        cancel: threading.Event,
        on_step=None,
    ):
        """Manual img2img pass for SANA — diffusers ships no SanaImg2ImgPipeline.

        Mirrors the SanaPipeline denoising loop but starts from the upscaled
        image latents noised to the strength cut point of the sigma schedule:
        x_t = (1-σ)·x0 + σ·ε (identical convention for DPM-flow and Euler).
        """
        import inspect

        pipe = self.pipe
        device = pipe._execution_device
        latent_channels = pipe.transformer.config.in_channels

        # 1. pixels → VAE latents (x0 space)
        pixel_values = pipe.image_processor.preprocess(
            image, height=image.height, width=image.width
        ).to(device=device, dtype=pipe.vae.dtype)
        image_latents = pipe.vae.encode(pixel_values).latent
        image_latents = (image_latents * pipe.vae.config.scaling_factor).float()

        # 2. sigma schedule cut by strength (timesteps descend: index 0 = noisiest)
        pipe.scheduler.set_timesteps(steps, device=device)
        init_steps = max(1, min(int(steps * strength), steps - 1))
        t_start = max(steps - init_steps, 0)
        timesteps = pipe.scheduler.timesteps[t_start:]
        sigma_start = float(pipe.scheduler.sigmas[t_start])

        # 3. noise the x0 latents up to the cut point
        noise = torch.randn(
            image_latents.shape, generator=generator, device=device, dtype=torch.float32
        )
        latents = (1.0 - sigma_start) * image_latents + sigma_start * noise

        # 4. prompt embeddings (same logic as the t2i pass)
        do_cfg = guidance > 1.0
        if enhance:
            chi = inspect.signature(type(pipe).__call__).parameters[
                "complex_human_instruction"
            ].default
        else:
            chi = []
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            device=device,
            clean_caption=clean_caption,
            max_sequence_length=max_seq,
            complex_human_instruction=chi,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 5. denoise from the cut point (mirror of the SanaPipeline loop)
        transformer_dtype = pipe.transformer.dtype
        for i, t in enumerate(timesteps):
            if cancel.is_set():
                raise Cancelled()
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(latent_model_input.shape[0])
            timestep = timestep * pipe.transformer.config.timestep_scale
            noise_pred = pipe.transformer(
                latent_model_input.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (
                    noise_pred_text - noise_pred_uncond
                )
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if on_step is not None:
                on_step(i)

        # 6. decode back to PIL
        image_out = pipe.vae.decode(
            latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return pipe.image_processor.postprocess(image_out, output_type="pil")[0]

    def _save_image(self, image, job_id, index, seed, meta):
        return save_image_file(image, job_id, index, seed, meta)
