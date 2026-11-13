#!/usr/bin/env python3
"""SANA 1.5 LoRA trainer — flow-matching fine-tuning of the transformer only.

Why this design (P102 10 GB / 15 GB RAM, no swap):
- Full fine-tuning / DreamBooth is impossible on 10 GB, so only PEFT LoRA
  adapters are injected into the transformer (attn1/attn2 → q,k,v,out);
  rank 8 ≈ 5.7 M trainable params, everything else frozen.
- VAE and text encoder are loaded once each and unloaded right after use:
  VAE encodes all dataset images (latents cached on GPU), the text encoder
  embeds all captions once — RAM stays flat like a normal generation load.
- Flow-matching objective mirrors inference exactly: x_t = (1-σ)x0 + σε,
  model predicts v where x0 = x_t − σ·v (verified against the DPM-flow
  scheduler used at generation time) → loss = MSE(v_pred, ε − x0).
- lora_alpha = rank (scaling 1.0): diffusers' loader infers alpha=r when a
  file has no alpha tensors — training and inference magnitudes then agree.
- Output is written directly in the diffusers SANA LoRA format
  (`transformer.<path>.lora_A/B.weight`, metadata {"format": "pt"} plus
  provenance), so the file loads through the normal LoRA path unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import sana_engine as se

torch = se.torch

TRAIN_DIR = se.DATA / "train"
REGISTRY_FILE = se.DATA / "train_datasets.json"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_IMAGES_PER_DATASET = 200

_RESOLUTIONS = (512, 768, 1024)

# Concept types.  The training loop is the same for all of them — what changes
# is the dataset checklist, caption guidance and the default recipe.
DATASET_KINDS = ("character", "style", "outfit", "pose")
MAX_EXTRA_TRIGGERS = 5
MAX_TAGS_PER_IMAGE = 20
MAX_REPEATS = 10


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_temperature() -> float | None:
    """Current GPU temperature in °C, or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def thermal_wait(cfg: dict, job: dict, cancel: threading.Event) -> None:
    """Emergency stop: block until the GPU cools to `thermal_resume_c`.

    This is only the last line of defence (hard limit).  Normal heat is handled
    by ``thermal_governor`` which slows the card down instead of idling it.
    """
    paused_at = time.time()
    while True:
        if cancel.is_set():
            raise se.Cancelled()
        temp = gpu_temperature()
        if temp is not None and temp <= cfg["thermal_resume_c"]:
            print(f"[train] thermal resume at {temp:.0f}°C", flush=True)
            return
        waited = time.time() - paused_at
        shown = f"{temp:.0f}°C" if temp is not None else "ไม่ทราบอุณหภูมิ"
        if waited > 900:
            raise RuntimeError(
                f"GPU ร้อนเกินไป ({shown}) และไม่เย็นลงภายใน 15 นาที — "
                f"หยุดเทรนเพื่อความปลอดภัย"
            )
        job["phase"] = "paused"
        job["message"] = (
            f"พัก GPU: {shown} (เพดาน {cfg['thermal_max_c']}°C) — "
            f"รอเย็นลงถึง {cfg['thermal_resume_c']}°C ({int(waited)}s)"
        )
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


# ── thermal governor: slow down instead of stopping ─────────────────────────
_SUPPORTED_CLOCKS: dict[str, object] = {}
# True only while we have an application-clock cap applied, so the training
# exit path resets the card exactly when needed (and never touches a card we
# never slowed down).
_CLOCKS_ACTIVE = {"value": False}


def gpu_supported_clocks() -> tuple[int, list[int]] | None:
    """(memory MHz, [supported graphics MHz…]) from nvidia-smi, cached.

    The output lists one block per memory clock (e.g. 5505 MHz with the high
    core clocks, 405 MHz with the low ones).  Application clocks must use the
    top memory clock, so pick the block with the highest memory value and only
    its graphics clocks — pairing 5505 with a 405 MHz-era core clock is not a
    supported combination.
    """
    if "value" in _SUPPORTED_CLOCKS:
        return _SUPPORTED_CLOCKS["value"]  # type: ignore[return-value]
    value = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
        groups: list[tuple[int, list[int]]] = []
        for line in out.splitlines():
            match = re.search(r":\s*(\d+)\s*MHz", line)
            if not match:
                continue
            if "Memory" in line:
                groups.append((int(match.group(1)), []))
            elif "Graphics" in line and groups:
                groups[-1][1].append(int(match.group(1)))
        if groups:
            mem, graphics = max(groups, key=lambda g: g[0])
            if graphics:
                value = (mem, sorted(set(graphics)))
    except Exception:  # noqa: BLE001
        value = None
    _SUPPORTED_CLOCKS["value"] = value
    return value


def _clock_set(fraction: float) -> bool:
    """Cap the core clock to `fraction` of max (needs passwordless sudo).

    Returns False when the system doesn't allow it, so the caller can fall
    back to duty-cycling (short sleeps) instead of stopping the run.
    """
    clocks = gpu_supported_clocks()
    if clocks is None:
        return False
    mem, graphics = clocks
    target = max(graphics) * fraction
    usable = [g for g in graphics if g <= target]
    sm = usable[-1] if usable else min(graphics)
    try:
        res = subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-ac", f"{mem},{sm}"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        # nvidia-smi exits 0 even on cards that cannot lock application clocks
        # (e.g. the P102-100) — it only prints "not supported" and moves on.
        # Treat that as a failure so the governor falls back to duty-cycling.
        text = f"{res.stdout or ''}{res.stderr or ''}".lower()
        ok = res.returncode == 0 and "not supported" not in text
        if ok:
            _CLOCKS_ACTIVE["value"] = True
        return ok
    except Exception:  # noqa: BLE001
        return False


def _clock_reset() -> None:
    if not _CLOCKS_ACTIVE["value"]:
        return
    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-rac"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        pass
    finally:
        _CLOCKS_ACTIVE["value"] = False


def _set_throttle_level(cfg: dict, state: dict, level: int) -> None:
    """Apply a throttle level (0 = full speed, 1..N = slower by step_pct each)."""
    level = max(0, min(cfg["thermal_max_level"], int(level)))
    if level == state["level"] and state.get("applied"):
        return
    state["level"] = level
    state["applied"] = True
    pct = cfg["thermal_step_pct"]
    if level <= 0:
        if state.get("clocks"):
            _clock_reset()
        state["clocks"] = None
        state["sleep"] = 0.0
        return
    fraction = max(0.5, 1.0 - pct * level / 100.0)
    if state["clocks"] is not False:
        ok = _clock_set(fraction)
        if state["clocks"] is None:
            state["clocks"] = ok
            if not ok:
                print(
                    "[train] ปรับ clock ไม่ได้ (ไม่มี sudo หรือการ์ดไม่รองรับ) — "
                    "ใช้การหน่วงระหว่าง step แทน",
                    flush=True,
                )
        elif not ok:
            state["clocks"] = False
    if state["clocks"]:
        state["sleep"] = 0.0
    else:
        base = state.get("step_time") or 1.0
        state["sleep"] = min(0.8, base * pct / 100.0 * level)
    if state["clocks"]:
        how = f"clock {int(fraction * 100)}%"
    else:
        how = f"หน่วง {state['sleep'] * 1000:.0f} ms/step"
    print(
        f"[train] thermal level {level} (ช้าลง {pct * level}% · {how})",
        flush=True,
    )


def thermal_governor(
    cfg: dict, job: dict, cancel: threading.Event, state: dict
) -> None:
    """Keep the GPU cool by reducing throughput, not by stopping the run.

    - temp >= thermal_throttle_c → one level slower (5% by default, up to
      thermal_max_level).  Level drops again once it cools to
      thermal_resume_c, so the run converges to the fastest safe speed.
    - temp >= thermal_max_c → emergency pause (last resort).
    """
    temp = gpu_temperature()
    if temp is None:
        return
    state["temp"] = temp
    if temp >= cfg["thermal_max_c"]:
        print(
            f"[train] hard thermal limit at {temp:.0f}°C "
            f"(max {cfg['thermal_max_c']}°C) — pausing",
            flush=True,
        )
        job["phase"] = "paused"
        thermal_wait(cfg, job, cancel)
        job["phase"] = "generating"
        _set_throttle_level(cfg, state, 0)
        return
    level = state["level"]
    if temp >= cfg["thermal_throttle_c"]:
        level += 1
    elif temp <= cfg["thermal_resume_c"]:
        level -= 1
    _set_throttle_level(cfg, state, level)


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    import os

    os.replace(tmp, path)


# ── dataset registry ─────────────────────────────────────────────────────────
def load_registry() -> list[dict]:
    try:
        raw = json.loads(REGISTRY_FILE.read_text("utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        print("[trainer] อ่าน registry ไม่ได้:", exc, flush=True)
        return []
    return [d for d in raw if isinstance(d, dict) and d.get("id")]


def save_registry(datasets: list[dict]) -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(REGISTRY_FILE, datasets)


def get_dataset(ds_id: str) -> dict | None:
    for d in load_registry():
        if d.get("id") == ds_id:
            return d
    return None


def sanitize_text(value, limit: int, fallback: str = "") -> str:
    text = " ".join(str(value or "").split())[:limit]
    return text or fallback


def sanitize_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def sanitize_lora_name(value: str) -> str:
    """Filename-safe LoRA name ([A-Za-z0-9_-], 1..48 chars)."""
    name = "".join(c if (c.isalnum() and c.isascii()) or c in "-_" else "-" for c in str(value or ""))
    name = "-".join(part for part in name.split("-") if part)[:48]
    return name or "lora"


def sanitize_trigger(value: str) -> str:
    trigger = sanitize_text(value, 40).lower()
    return " ".join(ch for ch in trigger.split())


def sanitize_tags(value) -> list[str]:
    """Normalize a tag list/CSV string into unique, lowercase, short tags."""
    if isinstance(value, str):
        parts = re.split(r"[,;\n]", value)
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = []
    tags: list[str] = []
    for part in parts:
        tag = sanitize_text(part, 40).lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:MAX_TAGS_PER_IMAGE]


def sanitize_extra_triggers(value, primary: str = "") -> list[str]:
    """Extra trigger words a dataset can offer per image (primary excluded)."""
    primary = sanitize_trigger(primary)
    out: list[str] = []
    for trigger in sanitize_tags(value):
        if trigger != primary and trigger not in out:
            out.append(trigger)
    return out[:MAX_EXTRA_TRIGGERS]


def sanitize_repeats(value) -> int:
    try:
        repeats = int(value)
    except (TypeError, ValueError):
        repeats = 1
    return max(1, min(MAX_REPEATS, repeats))


# ── aspect-ratio bucketing ──────────────────────────────────────────────────
# Standard SDXL-style ratios, scaled to the chosen resolution.  Keeping the
# long edge <= 1344 avoids OOM on a 10 GB card (1536x640 is dropped).
_BUCKET_RATIOS = (
    (1, 1),
    (9, 7),
    (7, 9),
    (19, 13),
    (13, 19),
    (3, 2),
    (2, 3),
    (7, 4),
    (4, 7),
)
BUCKET_STEP = 64


def resolution_buckets(resolution: int) -> list[tuple[int, int]]:
    """Buckets with roughly the same pixel area as ``resolution``².

    Each ratio is converted to pixel dimensions for the requested area, then
    snapped to a multiple of 64 (the VAE's spatial stride).  The long edge is
    capped at 1344 px so a 10 GB card never sees a 1536-wide bucket.
    """
    area = max(256, int(resolution)) ** 2
    buckets: list[tuple[int, int]] = []

    def snap(value: float) -> int:
        return max(256, int(round(value / BUCKET_STEP)) * BUCKET_STEP)

    for rw, rh in _BUCKET_RATIOS:
        aspect = rw / rh
        bw = snap(math.sqrt(area * aspect))
        bh = snap(math.sqrt(area / aspect))
        long_edge = max(bw, bh)
        if long_edge > 1344:
            factor = 1344.0 / long_edge
            bw = snap(bw * factor)
            bh = snap(bh * factor)
        if (bw, bh) not in buckets:
            buckets.append((bw, bh))
    return buckets


def pick_bucket(width: int, height: int, resolution: int) -> tuple[int, int]:
    """Pick the bucket whose aspect ratio is closest to the source image."""
    try:
        ratio = float(width) / float(height)
    except (TypeError, ValueError, ZeroDivisionError):
        return (resolution, resolution)
    if ratio <= 0:
        return (resolution, resolution)
    return min(
        resolution_buckets(resolution),
        key=lambda wh: abs(math.log((wh[0] / wh[1]) / ratio)),
    )


# ── caption variants / loss helpers ─────────────────────────────────────────
def caption_variants(
    image: dict,
    ds: dict,
    shuffle: bool = False,
    keep_tokens: int = 0,
    count: int = 1,
) -> list[str]:
    """Effective captions for one image: original plus optional shuffles.

    ``keep_tokens`` protects the first N comma-separated segments (trigger +
    the most important tags) from being reordered, which is the same idea as
    kohya's ``keep_tokens``.
    """
    base = _effective_caption(image, ds)
    if not shuffle or count <= 1:
        return [base]
    segments = [part.strip() for part in base.split(",") if part.strip()]
    if len(segments) <= 1:
        return [base]
    keep = max(0, min(len(segments), int(keep_tokens)))
    head, tail = segments[:keep], segments[keep:]
    variants = [base]
    for index in range(1, int(count)):
        shuffled = list(tail)
        random.Random(f"{base}:{index}").shuffle(shuffled)
        variants.append(", ".join(head + shuffled))
    return variants


def add_noise_offset(noise, offset: float, generator=None):
    """Channel-wise noise offset (helps very dark / high-contrast datasets)."""
    if not offset:
        return noise
    channels = noise.shape[1]
    extra = torch.randn(
        (noise.shape[0], channels, 1, 1),
        generator=generator,
        device=noise.device,
        dtype=noise.dtype,
    )
    return noise + extra * float(offset)


def min_snr_weight(snr, gamma: float):
    """Min-SNR loss weight: clamp(SNR, max=gamma) / SNR."""
    if not gamma:
        return None
    return torch.clamp(snr, max=float(gamma)) / snr


# ── resume state (optimizer + scheduler + RNG + EMA) ────────────────────────
def trainstate_path(lora_path: Path) -> Path:
    return lora_path.with_suffix(".trainstate.pt")


def save_train_state(
    path: Path,
    *,
    optimizer,
    scheduler,
    step: int,
    ema_params: dict,
    rng: random.Random,
    g_rng,
    ema_loss,
    rank: int,
) -> None:
    import os

    payload = {
        "format": 1,
        "step": int(step),
        "rank": int(rank),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": {k: v.detach().cpu() for k, v in (ema_params or {}).items()},
        "rng": rng.getstate(),
        "g_rng": g_rng.get_state() if g_rng is not None else None,
        "ema_loss": float(ema_loss) if ema_loss is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def load_train_state(
    path: Path,
    *,
    optimizer,
    scheduler,
    ema_params: dict,
    rng: random.Random,
    g_rng,
    rank: int,
) -> tuple[int, object]:
    """Restore optimizer/scheduler/RNG/EMA. Returns (start_step, ema_loss)."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if int(state.get("rank") or 0) != int(rank):
        raise RuntimeError(
            f"resume state เป็น rank {state.get('rank')} แต่รอบนี้ rank {rank}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    saved_ema = state.get("ema") or {}
    if ema_params and saved_ema:
        for name, value in saved_ema.items():
            if name in ema_params:
                ema_params[name].copy_(value.to(ema_params[name].device))
    try:
        rng.setstate(state["rng"])
    except (KeyError, TypeError, ValueError):
        pass
    g_state = state.get("g_rng")
    if g_rng is not None and g_state is not None:
        try:
            g_rng.set_state(g_state)
        except (RuntimeError, TypeError, ValueError):
            pass
    return max(0, int(state.get("step") or 0)), state.get("ema_loss")


SANA_ATTN_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
SANA_FF_MODULES = ["ff.conv_inverted", "ff.conv_point"]


def create_dataset(name: str, trigger: str, kind: str, extra_triggers=None) -> dict:
    ds_id = uuid.uuid4().hex[:10]
    primary = sanitize_trigger(trigger)
    entry = {
        "id": ds_id,
        "name": sanitize_text(name, 60) or "dataset",
        "trigger": primary,
        "extra_triggers": sanitize_extra_triggers(extra_triggers, primary),
        "kind": kind if kind in DATASET_KINDS else "character",
        "created": _now(),
        "images": [],
    }
    datasets = load_registry()
    datasets.append(entry)
    save_registry(datasets)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    return entry


def delete_dataset(ds_id: str) -> None:
    datasets = [d for d in load_registry() if d.get("id") != ds_id]
    save_registry(datasets)
    shutil.rmtree(TRAIN_DIR / ds_id, ignore_errors=True)


def update_dataset(ds_id: str, name=None, trigger=None, kind=None, extra_triggers=None) -> dict:
    datasets = load_registry()
    entry = next((d for d in datasets if d.get("id") == ds_id), None)
    if entry is None:
        raise ValueError("ไม่พบ dataset นี้")
    if name is not None:
        entry["name"] = sanitize_text(name, 60) or entry["name"]
    if trigger is not None:
        entry["trigger"] = sanitize_trigger(trigger)
    if kind is not None and kind in DATASET_KINDS:
        entry["kind"] = kind
    if extra_triggers is not None:
        entry["extra_triggers"] = sanitize_extra_triggers(
            extra_triggers, entry.get("trigger") or ""
        )
    elif trigger is not None:
        # keep the list consistent when the primary trigger changes
        entry["extra_triggers"] = sanitize_extra_triggers(
            entry.get("extra_triggers"), entry.get("trigger") or ""
        )
    save_registry(datasets)
    return entry


def _dataset_dir(ds_id: str) -> Path:
    if not get_dataset(ds_id):
        raise ValueError("ไม่พบ dataset นี้")
    d = TRAIN_DIR / ds_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_image(ds_id: str, src_path: str, original_name: str = "") -> dict:
    """Copy an uploaded image into the dataset folder (uuid filename)."""
    src = Path(src_path)
    if not src.is_file():
        raise ValueError("ไม่พบไฟล์รูป")
    if src.suffix.lower() not in IMG_EXTS:
        raise ValueError("รองรับเฉพาะ .png/.jpg/.jpeg/.webp")
    datasets = load_registry()
    entry = next((d for d in datasets if d.get("id") == ds_id), None)
    if entry is None:
        raise ValueError("ไม่พบ dataset นี้")
    if len(entry.get("images") or []) >= MAX_IMAGES_PER_DATASET:
        raise ValueError(f"รูปเต็มขีดจำกัด ({MAX_IMAGES_PER_DATASET} ใบ)")
    name = " ".join(str(original_name or src.name).split())[:120]
    for image in entry["images"]:
        if image.get("name") == name:
            raise ValueError(f"มีรูปชื่อนี้อยู่แล้ว: {name}")
    directory = _dataset_dir(ds_id)
    file_id = uuid.uuid4().hex[:12] + src.suffix.lower()
    dest = directory / file_id
    shutil.copyfile(src, dest)
    image_entry = {
        "file": file_id,
        "name": name,
        "caption": "",
        "tags": [],
        "trigger": "",
        "repeats": 1,
        "multi": False,
        "added": _now(),
    }
    entry["images"].append(image_entry)
    save_registry(datasets)
    return image_entry


def remove_image(ds_id: str, file_id: str) -> None:
    datasets = load_registry()
    entry = next((d for d in datasets if d.get("id") == ds_id), None)
    if entry is None:
        raise ValueError("ไม่พบ dataset นี้")
    before = len(entry["images"])
    entry["images"] = [i for i in entry["images"] if i.get("file") != file_id]
    if len(entry["images"]) == before:
        raise ValueError("ไม่พบรูปนี้ใน dataset")
    save_registry(datasets)
    (TRAIN_DIR / ds_id / file_id).unlink(missing_ok=True)


def set_captions(ds_id: str, updates: list[dict]) -> dict:
    datasets = load_registry()
    entry = next((d for d in datasets if d.get("id") == ds_id), None)
    if entry is None:
        raise ValueError("ไม่พบ dataset นี้")
    known = {i["file"] for i in entry.get("images") or []}
    for item in updates or []:
        file_id = str(item.get("file") or "")
        if file_id not in known:
            continue
        for image in entry["images"]:
            if image["file"] == file_id:
                image["caption"] = sanitize_text(item.get("caption"), 600)
                if "multi" in item:
                    value = item.get("multi")
                    if isinstance(value, str):
                        value = value.strip().lower() in ("1", "true", "yes", "on")
                    image["multi"] = bool(value)
                if "tags" in item:
                    image["tags"] = sanitize_tags(item.get("tags"))
                if "trigger" in item:
                    image["trigger"] = sanitize_trigger(item.get("trigger") or "")
                if "repeats" in item:
                    image["repeats"] = sanitize_repeats(item.get("repeats"))
                break
    save_registry(datasets)
    return entry


def dataset_dir_path(ds_id: str) -> Path:
    """Validated absolute folder for serving images via /files/train/..."""
    if "/" in ds_id or ".." in ds_id or "\\" in ds_id:
        raise ValueError("dataset id ไม่ถูกต้อง")
    return TRAIN_DIR / ds_id


# ── dataset QC ──────────────────────────────────────────────────────────────
# A cheap pre-flight report the UI shows before a long training run.  It never
# modifies anything: duplicates are reported, not deleted, and every warning
# explains what training would do with the image as-is.
def _file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_qc(ds_id: str, resolution: int = 1024) -> dict:
    ds = get_dataset(ds_id)
    if ds is None:
        raise ValueError("ไม่พบ dataset นี้")
    from PIL import Image

    images = ds.get("images") or []
    kind = ds.get("kind") or "character"
    report: list[dict] = []
    digests: dict[str, list[str]] = {}
    buckets: dict[str, int] = {}
    solo = multi = 0
    missing_caption = 0
    small = 0
    extreme = 0
    for image in images:
        path = TRAIN_DIR / ds_id / str(image.get("file") or "")
        width = height = 0
        size_bytes = 0
        error = ""
        try:
            size_bytes = path.stat().st_size
            with Image.open(path) as handle:
                width, height = handle.size
        except Exception as exc:  # noqa: BLE001
            error = f"เปิดรูปไม่ได้: {exc}"
        aspect = (width / height) if width and height else 0.0
        bucket = pick_bucket(width, height, resolution) if width and height else (0, 0)
        bucket_key = f"{bucket[0]}×{bucket[1]}"
        if width and height:
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1
        warnings: list[str] = []
        if error:
            warnings.append(error)
        else:
            if min(width, height) < 512:
                small += 1
                warnings.append("ความละเอียดต่ำกว่า 512px — รายละเอียดจะไม่พอ")
            if aspect and (aspect < 0.4 or aspect > 2.5):
                extreme += 1
                warnings.append("สัดส่วนภาพสุดโต่ง — จะถูก crop มากตอน bucketing")
        caption = str(image.get("caption") or "").strip()
        tags = sanitize_tags(image.get("tags"))
        if not caption and not tags:
            missing_caption += 1
            warnings.append("ยังไม่มี caption/tags — โมเดลจะจำได้แค่ตัวตน")
        if image.get("multi"):
            multi += 1
        else:
            solo += 1
        try:
            digest = _file_digest(path)
            digests.setdefault(digest, []).append(str(image.get("file")))
        except Exception:  # noqa: BLE001
            digest = ""
        report.append(
            {
                "file": image.get("file"),
                "name": image.get("name") or image.get("file"),
                "width": width,
                "height": height,
                "aspect": round(aspect, 4) if aspect else 0,
                "bucket": bucket_key,
                "bytes": size_bytes,
                "multi": bool(image.get("multi")),
                "tags": tags,
                "trigger": image.get("trigger") or "",
                "repeats": sanitize_repeats(image.get("repeats")),
                "caption": caption,
                "digest": digest,
                "warnings": warnings,
            }
        )
    duplicates = [files for files in digests.values() if len(files) > 1]
    if duplicates:
        known = {f for group in duplicates for f in group}
        for item in report:
            if item["file"] in known:
                item["warnings"].append("ไฟล์ซ้ำกับรูปอื่นใน dataset")
    recommendations: list[str] = []
    if len(images) < 10:
        recommendations.append(
            f"มีแค่ {len(images)} รูป — ควรมีอย่างน้อย 10–20 รูปเพื่อให้จำตัวตนได้แม่น"
        )
    if kind == "character" and solo < 16:
        recommendations.append(
            f"รูปเดี่ยวมี {solo} รูป — เป้าหมายคือ 16–30 รูป (หลายมุม/หลายระยะ)"
        )
    if missing_caption:
        recommendations.append(
            f"{missing_caption} รูปยังไม่มี caption/tags — ใส่เพื่อแยกสิ่งที่อยากได้/ไม่อยากได้"
        )
    if duplicates:
        recommendations.append(
            f"พบไฟล์ซ้ำ {len(duplicates)} กลุ่ม — ลบออกเพื่อไม่ให้น้ำหนักภาพนั้นมากเกิน"
        )
    if small:
        recommendations.append(
            f"{small} รูปเล็กกว่า 512px — อาจทำให้รายละเอียดใบหน้า/ผิวไม่ชัด"
        )
    if extreme:
        recommendations.append(
            f"{extreme} รูปสัดส่วนสุดโต่ง — พิจารณาครอปหรือเพิ่มรูปสัดส่วนปกติ"
        )
    if not recommendations:
        recommendations.append("ข้อมูลพร้อมเทรน — ไม่พบปัญหาที่ต้องแก้ก่อน")
    return {
        "ok": True,
        "dataset": {
            "id": ds.get("id"),
            "name": ds.get("name"),
            "kind": kind,
            "trigger": ds.get("trigger") or "",
            "extra_triggers": ds.get("extra_triggers") or [],
            "count": len(images),
            "solo": solo,
            "multi": multi,
            "missing_caption": missing_caption,
            "small_images": small,
            "extreme_aspect": extreme,
            "duplicates": duplicates,
            "buckets": buckets,
            "resolution": resolution,
            "recommendations": recommendations,
        },
        "images": report,
    }


# ── trainer ──────────────────────────────────────────────────────────────────
def _effective_caption(image: dict, ds: dict) -> str:
    """Effective training caption — trigger + per-image tags + free caption.

    The trigger is the token that ties all dataset images to one identity: if
    a caption omits it, the trigger never gets associated with the character
    and the trained LoRA can't summon it at generation time.  So: prepend the
    active trigger (the image's own override, else the dataset's) whenever it
    is missing; tags come next, then the free-form caption.  Duplicates are
    removed case-insensitively so "front view" never appears twice.
    """
    caption = " ".join(str(image.get("caption") or "").split())
    trigger = sanitize_trigger(image.get("trigger") or ds.get("trigger") or "")
    tags = sanitize_tags(image.get("tags"))
    lowered = caption.lower()
    parts: list[str] = []
    seen: set[str] = set()
    if trigger and trigger not in lowered:
        parts.append(trigger)
        seen.add(trigger)
    for tag in tags:
        if tag in lowered or tag in seen:
            continue
        seen.add(tag)
        parts.append(tag)
    if caption:
        parts.append(caption)
    return ", ".join(parts) or "subject"


# Captions that hint at a second body in the frame.  Used only for the UI
# audit (the user confirms before anything is marked); never for training.
MULTI_HINTS = (
    "man",
    "men",
    "male",
    "two girls",
    "two boys",
    "duo",
    "group",
    "individuals",
    "two people",
    "another character",
    "partner",
    "couple",
)


def looks_multi_caption(caption: str) -> bool:
    text = str(caption or "").lower()
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in MULTI_HINTS)


def select_train_images(
    images: list[dict], mode: str, weight: float, kind: str = "character"
) -> tuple[list[dict], list[float] | None, str]:
    """Apply the dataset's multi-character policy and per-image repeats.

    - exclude  : train only on solo images (default — keeps supporting
                 characters out of the LoRA).
    - weighted : train on everything, but sample multi-character images less
                 often so their extra bodies contribute less.
    - include  : train on everything with equal weight.

    Non-character datasets (style / outfit / pose) always use ``include``:
    the "second character" flag only makes sense for character LoRAs.

    Per-image ``repeats`` (1..10) multiply the sampling weight, which is how
    a small dataset can show an important image more often without duplicating
    files on disk.

    Returns (images_for_training, per-image sample weights or None, mode).
    """
    mode = str(mode or "exclude").lower()
    if str(kind or "character").lower() != "character":
        mode = "include"
    if mode not in ("exclude", "weighted", "include"):
        mode = "exclude"
    images = [im for im in (images or []) if isinstance(im, dict)]
    if not images:
        raise RuntimeError("dataset ยังไม่มีรูป — เพิ่มรูปก่อนเทรน")

    def repeats(image: dict) -> int:
        return sanitize_repeats(image.get("repeats"))

    if mode == "exclude":
        solo = [im for im in images if not im.get("multi")]
        if not solo:
            raise RuntimeError(
                "ไม่มีรูปตัวละครเดี่ยวให้เทรน — ทำเครื่องหมายรูปเดี่ยวอย่างน้อย 1 รูป "
                "หรือเปลี่ยนโหมดรูปหลายตัวละครเป็น 'ลดน้ำหนัก' / 'ใช้เต็ม'"
            )
        weights = [float(repeats(im)) for im in solo]
        return solo, (weights if any(w != 1.0 for w in weights) else None), mode
    if mode == "weighted":
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.25
        w = max(0.05, min(1.0, w))
        weights = [
            (w if im.get("multi") else 1.0) * repeats(im) for im in images
        ]
        return images, weights, mode
    weights = [float(repeats(im)) for im in images]
    return images, (weights if any(w != 1.0 for w in weights) else None), mode


def _resolve_init_lora(value) -> Path | None:
    """Turn an LoRA id / filename from the UI into a file on disk."""
    raw = str(value or "").strip()
    if not raw:
        return None
    loras = se.discover_loras()
    found = next((l for l in loras if l.get("id") == raw), None)
    if found is None:
        stem = Path(raw).stem
        found = next(
            (l for l in loras if Path(str(l.get("path") or "")).stem == stem), None
        )
    if found is None:
        raise RuntimeError(f"ไม่พบ LoRA ตั้งต้น: {raw}")
    path = Path(str(found["path"]))
    if not path.is_file():
        raise RuntimeError(f"ไฟล์ LoRA ตั้งต้นหายไป: {path.name}")
    return path


def _lora_file_rank(path: Path) -> int | None:
    """Read the adapter rank from a saved LoRA (metadata first, then shapes)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
        try:
            rank = int(meta.get("rank"))
            if rank > 0:
                return rank
        except (TypeError, ValueError):
            pass
        for key in handle.keys():
            if key.endswith("lora_A.weight"):
                return int(handle.get_slice(key).get_shape()[0])
    return None


def _load_lora_weights(
    peft_model,
    path: Path,
    prefix: str = "transformer.",
    lora_b_scale: float = 1.0,
) -> tuple[int, list[str]]:
    """Load a diffusers-style LoRA file into an injected PEFT adapter.

    Files are saved with diffusers-style keys
    (``transformer.transformer_blocks.0.attn1.to_q.lora_A.weight`` for SANA,
    ``unet.down_blocks...lora_A.weight`` for SDXL) while PEFT exposes
    ``...lora_A.default.weight``.  This maps one to the other and verifies
    every tensor shape before touching the model, so a rank mismatch fails
    with a readable message instead of a cryptic load error.
    """
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    params = dict(peft_model.named_parameters())
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if prefix and name.startswith(prefix):
            name = name[len(prefix) :]
        candidates = [name]
        for part in ("lora_A", "lora_B"):
            token = f".{part}.weight"
            if name.endswith(token):
                candidates.insert(
                    0, name[: -len(token)] + f".{part}.default.weight"
                )
                break
        target_name = next((c for c in candidates if c in params), None)
        if target_name is None:
            skipped.append(str(key))
            continue
        target = params[target_name]
        if tuple(target.shape) != tuple(value.shape):
            raise RuntimeError(
                f"ขนาดน้ำหนัก LoRA ไม่ตรงกับ rank ที่ตั้งไว้ "
                f"({target_name}: {tuple(value.shape)} vs {tuple(target.shape)})"
            )
        tensor = value.to(torch.float32)
        # Files are always stored with standard LoRA scaling.  When the new
        # adapter uses rsLoRA (scale alpha/sqrt(r)) the B matrix must shrink by
        # sqrt(r) so the adapter starts from the same function.
        if lora_b_scale != 1.0 and target_name.endswith(".lora_B.default.weight"):
            tensor = tensor * float(lora_b_scale)
        mapped[target_name] = tensor
    if not mapped:
        raise RuntimeError(f"ไฟล์ LoRA ตั้งต้นไม่มีน้ำหนักที่ใช้ได้: {path.name}")
    result = set_peft_model_state_dict(
        peft_model, mapped, adapter_name="default"
    )
    unexpected = [k for k in result.unexpected_keys if k in mapped]
    if unexpected:
        raise RuntimeError(f"โหลด LoRA ตั้งต้นไม่ครบ: {unexpected[:3]}")
    return len(mapped), skipped


def _save_lora_state(
    peft_model,
    out_path: Path,
    metadata: dict,
    prefix: str = "transformer.",
    lora_b_scale: float = 1.0,
) -> None:
    """Write a diffusers-format LoRA file with clean keys + metadata.

    SANA uses ``transformer.*`` keys, SDXL uses ``unet.*``; both are what the
    matching diffusers loader expects.  The file is written to a sibling temp
    file, validated, then atomically moved into place with os.replace().  An
    interrupted / crashed save can therefore never leave a 0-byte or truncated
    .safetensors behind (which is what happened to an early build's LoRA).
    """
    from peft.utils import get_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import save_file

    raw = get_peft_model_state_dict(peft_model)
    fixed = {}
    for key, value in raw.items():
        name = str(key)
        name = name.replace("base_model.model.", "")
        name = name.replace(".default", "")
        if prefix and not name.startswith(prefix):
            name = prefix + name
        tensor = value.detach().to("cpu")
        if lora_b_scale != 1.0 and name.endswith(".lora_B.weight"):
            tensor = tensor.float() * float(lora_b_scale)
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)
        fixed[name] = tensor.contiguous()
    metadata = dict(metadata or {})
    metadata.setdefault("format", "pt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        save_file(fixed, str(tmp_path), metadata=metadata)
        with safe_open(str(tmp_path), framework="pt") as handle:
            keys = list(handle.keys())
        if not keys:
            raise RuntimeError("ไฟล์ LoRA ที่บันทึกไม่มีน้ำหนัก — ยกเลิกการเขียน")
        os.replace(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


class SanaTrainer:
    """Runs SANA 1.5 LoRA training jobs inside the normal job queue."""

    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.Lock()

    # -- public entry (called by the worker like manager.generate) -----------
    def train(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        with self.lock:
            return self._train(params, job, cancel)

    def _sanitize(self, params: dict) -> dict:
        ds = get_dataset(str(params.get("dataset_id") or ""))
        if ds is None:
            raise RuntimeError("ไม่พบ dataset ที่จะเทรน")
        if not ds.get("images"):
            raise RuntimeError("dataset ยังไม่มีรูป — เพิ่มรูปก่อนเทรน")
        rank = max(4, min(128, int(params.get("rank") or 16)))
        steps = max(10, min(20000, int(params.get("steps") or 1500)))
        lr = float(params.get("lr") or 2e-4)
        lr = max(1e-5, min(5e-4, lr))
        resolution = int(params.get("resolution") or 1024)
        resolution = resolution if resolution in _RESOLUTIONS else 1024
        lora_name = sanitize_lora_name(params.get("lora_name"))
        enhance = bool(params.get("enhance", True))
        # Quality knobs (all optional; defaults match the improved recipe).
        scheduler = str(params.get("scheduler") or "cosine").lower()
        if scheduler not in ("cosine", "constant"):
            scheduler = "cosine"
        try:
            warmup_ratio = float(params.get("warmup_ratio"))
        except (TypeError, ValueError):
            warmup_ratio = 0.03
        warmup_ratio = max(0.0, min(0.2, warmup_ratio))
        ema = bool(params.get("ema", False))
        try:
            ema_decay = float(params.get("ema_decay"))
        except (TypeError, ValueError):
            ema_decay = 0.999
        ema_decay = max(0.9, min(0.9999, ema_decay))
        sampling = str(params.get("timestep_sampling") or "logit_normal").lower()
        if sampling not in ("logit_normal", "shift"):
            sampling = "logit_normal"
        try:
            save_every = int(params.get("save_every") or 0)
        except (TypeError, ValueError):
            save_every = 0
        save_every = max(0, min(steps, save_every))
        augment_flip = bool(params.get("augment_flip", False))
        try:
            grad_accum = int(params.get("grad_accum") or 1)
        except (TypeError, ValueError):
            grad_accum = 1
        grad_accum = max(1, min(8, grad_accum))
        # Multi-character policy: by default only solo images train, so the
        # trigger doesn't absorb the supporting characters' bodies.
        multi_mode = str(params.get("multi_mode") or "exclude").lower()
        if multi_mode not in ("exclude", "weighted", "include"):
            multi_mode = "exclude"
        try:
            multi_weight = float(params.get("multi_weight"))
        except (TypeError, ValueError):
            multi_weight = 0.25
        multi_weight = max(0.05, min(1.0, multi_weight))
        # Recipe options (Phase 2). Defaults keep the previous behaviour except
        # bucketing, which is on by default because it stops square-cropping
        # portrait/landscape training images.
        bucketing = sanitize_bool(params.get("bucketing"), True)
        try:
            caption_dropout = float(params.get("caption_dropout"))
        except (TypeError, ValueError):
            caption_dropout = 0.05
        caption_dropout = max(0.0, min(0.3, caption_dropout))
        caption_shuffle = sanitize_bool(params.get("caption_shuffle"), False)
        try:
            keep_tokens = int(params.get("keep_tokens"))
        except (TypeError, ValueError):
            keep_tokens = 1
        keep_tokens = max(0, min(4, keep_tokens))
        try:
            noise_offset = float(params.get("noise_offset"))
        except (TypeError, ValueError):
            noise_offset = 0.0
        noise_offset = max(0.0, min(0.2, noise_offset))
        try:
            min_snr_gamma = float(params.get("min_snr_gamma"))
        except (TypeError, ValueError):
            min_snr_gamma = 0.0
        min_snr_gamma = max(0.0, min(20.0, min_snr_gamma))
        use_rslora = sanitize_bool(params.get("use_rslora"), False)
        target_set = str(params.get("target_modules") or "attn").lower()
        if target_set not in ("attn", "attn_ff"):
            target_set = "attn"
        resume_state = sanitize_bool(params.get("resume_state"), False)
        # Thermal guard: slow the card down (default) instead of pausing it.
        thermal_guard = bool(params.get("thermal_guard", True))
        thermal_mode = str(params.get("thermal_mode") or "throttle").lower()
        if thermal_mode not in ("throttle", "pause"):
            thermal_mode = "throttle"
        try:
            thermal_max_c = int(params.get("thermal_max_c") or 90)
        except (TypeError, ValueError):
            thermal_max_c = 90
        thermal_max_c = max(80, min(93, thermal_max_c))
        try:
            thermal_throttle_c = int(
                params.get("thermal_throttle_c") or min(87, thermal_max_c - 3)
            )
        except (TypeError, ValueError):
            thermal_throttle_c = min(87, thermal_max_c - 3)
        thermal_throttle_c = max(60, min(thermal_max_c - 1, thermal_throttle_c))
        try:
            thermal_resume_c = int(
                params.get("thermal_resume_c")
                or max(45, thermal_throttle_c - 7)
            )
        except (TypeError, ValueError):
            thermal_resume_c = max(45, thermal_throttle_c - 7)
        thermal_resume_c = max(40, min(thermal_throttle_c - 1, thermal_resume_c))
        try:
            thermal_step_pct = int(params.get("thermal_step_pct") or 5)
        except (TypeError, ValueError):
            thermal_step_pct = 5
        thermal_step_pct = max(1, min(25, thermal_step_pct))
        thermal_max_level = max(1, min(8, 25 // thermal_step_pct))
        try:
            thermal_check_every = int(params.get("thermal_check_every") or 10)
        except (TypeError, ValueError):
            thermal_check_every = 10
        thermal_check_every = max(1, min(200, thermal_check_every))
        try:
            seed = int(params.get("seed"))
        except (TypeError, ValueError):
            seed = -1
        seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        # Continue-from support: load an existing adapter before training so a
        # short run refines it instead of starting from zero.
        init_lora = _resolve_init_lora(params.get("init_lora"))
        if init_lora is not None:
            file_rank = _lora_file_rank(init_lora)
            if file_rank is not None:
                rank = max(4, min(128, int(file_rank)))
        out_path = se.MODELS_ROOT / "loras" / f"{lora_name}.safetensors"
        return {
            "ds": ds,
            "rank": rank,
            "steps": steps,
            "lr": lr,
            "resolution": resolution,
            "lora_name": lora_name,
            "out_path": out_path,
            "enhance": enhance,
            "seed": seed,
            "scheduler": scheduler,
            "warmup_ratio": warmup_ratio,
            "ema": ema,
            "ema_decay": ema_decay,
            "sampling": sampling,
            "save_every": save_every,
            "augment_flip": augment_flip,
            "grad_accum": grad_accum,
            "multi_mode": multi_mode,
            "multi_weight": multi_weight,
            "bucketing": bucketing,
            "caption_dropout": caption_dropout,
            "caption_shuffle": caption_shuffle,
            "keep_tokens": keep_tokens,
            "noise_offset": noise_offset,
            "min_snr_gamma": min_snr_gamma,
            "use_rslora": use_rslora,
            "target_set": target_set,
            "resume_state": resume_state,
            "thermal_guard": thermal_guard,
            "thermal_mode": thermal_mode,
            "thermal_max_c": thermal_max_c,
            "thermal_throttle_c": thermal_throttle_c,
            "thermal_resume_c": thermal_resume_c,
            "thermal_step_pct": thermal_step_pct,
            "thermal_max_level": thermal_max_level,
            "thermal_check_every": thermal_check_every,
            "init_lora": init_lora,
        }

    def _train(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        """Run training, then return GPU memory to the driver.

        The cleanup lives outside the training frame on purpose: while the
        frame is alive its tensors (latents, embeds, the transformer) are still
        referenced, so ``empty_cache`` would free nothing and the GPU would
        keep ~3.4 GB allocated (and stay warm) after training ends.
        """
        try:
            return self._train_inner(params, job, cancel)
        finally:
            # If the governor capped the core clock, restore full speed so the
            # card is never left slow for the next generation job.
            _clock_reset()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_inner(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        cfg = self._sanitize(params)
        ds = cfg["ds"]
        train_images, sample_weights, multi_mode = select_train_images(
            ds.get("images") or [],
            cfg["multi_mode"],
            cfg["multi_weight"],
            ds.get("kind") or "character",
        )
        solo_count = sum(1 for im in train_images if not im.get("multi"))
        multi_count = len(train_images) - solo_count

        def status(msg):
            job.update({"phase": "preparing", "message": msg, "status": "running"})
            print(f"[train] {msg}", flush=True)

        mode_label = {
            "exclude": "เฉพาะตัวละครเดี่ยว",
            "weighted": f"ลดน้ำหนักรูปหลายตัวละครเหลือ {int(cfg['multi_weight'] * 100)}%",
            "include": "ใช้ทุกรูปน้ำหนักเท่ากัน",
        }[multi_mode]
        print(
            f"[train] dataset policy: {mode_label} · ใช้ {len(train_images)}/"
            f"{len(ds['images'])} รูป (เดี่ยว {solo_count} · หลายตัวละคร {multi_count})",
            flush=True,
        )

        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("เทรนต้องใช้ GPU — ตรวจ PyTorch/CUDA ใน engine/setup.sh")

        status(f"ปล่อยหน่วยความจำโมเดลเดิมก่อนเทรน ({cfg['lora_name']})…")
        self.manager.unload()

        import inspect

        from PIL import Image, ImageOps
        from peft import LoraConfig, inject_adapter_in_model

        sana_entry = next(m for m in se.discover_models() if m["family"] == "sana")
        model_path = Path(sana_entry["path"])
        if not se.files_ready(model_path):
            raise RuntimeError("ไฟล์โมเดล SANA ไม่ครบ — ตรวจ models/")

        from diffusers import SanaPipeline

        status("กำลังโหลดโมเดล SANA สำหรับเทรน…")
        pipe = SanaPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        pipe.set_progress_bar_config(disable=True)
        device = torch.device("cuda")
        transformer = pipe.transformer

        try:
            # ── 1. VAE pass: cache latents for every image, then free the VAE ──
            status(f"เตรียม latent ({len(train_images)} รูป · {cfg['resolution']}px)…")
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception as exc:  # noqa: BLE001
                print("[train] vae tiling warning:", exc, flush=True)
            pipe.vae.to(device)
            scale = transformer.config.in_channels
            latents = []
            sizes = []
            with torch.no_grad():
                for index, image in enumerate(train_images):
                    if cancel.is_set():
                        raise se.Cancelled()
                    path = TRAIN_DIR / ds["id"] / image["file"]
                    img = Image.open(path).convert("RGB")
                    bucket = (
                        pick_bucket(img.width, img.height, cfg["resolution"])
                        if cfg["bucketing"]
                        else (cfg["resolution"], cfg["resolution"])
                    )
                    sizes.append(bucket)
                    img = ImageOps.fit(img, bucket, method=Image.LANCZOS)
                    import numpy as np

                    arr = np.asarray(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
                    pixel = tensor.unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)
                    latent = pipe.vae.encode(pixel).latent * pipe.vae.config.scaling_factor
                    latents.append(latent.float())
                    job["message"] = (
                        f"เตรียม latent {index + 1}/{len(train_images)} "
                        f"({bucket[0]}×{bucket[1]})"
                    )
                    job["progress"] = round(0.05 * (index + 1) / len(train_images), 4)
            pipe.vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── 2. text encoder pass: embed every caption once, then free ──
            variant_count = 2 if cfg["caption_shuffle"] else 1
            caption_groups = [
                caption_variants(
                    image,
                    ds,
                    cfg["caption_shuffle"],
                    cfg["keep_tokens"],
                    variant_count,
                )
                for image in train_images
            ]
            status(
                f"เตรียม embedding ของ caption ({len(caption_groups)} รูป · "
                f"{sum(len(g) for g in caption_groups)} แบบ) — ตัวอย่าง: "
                f"“{caption_groups[0][0][:70]}”"
            )
            pipe.text_encoder.to(device)
            if cfg["enhance"]:
                chi = inspect.signature(type(pipe).__call__).parameters[
                    "complex_human_instruction"
                ].default
            else:
                chi = []
            embeds, masks = [], []
            with torch.no_grad():
                for group in caption_groups:
                    group_embeds, group_masks = [], []
                    for text in group:
                        p_emb, p_mask, _, _ = pipe.encode_prompt(
                            prompt=text,
                            do_classifier_free_guidance=False,
                            negative_prompt="",
                            device=device,
                            clean_caption=False,
                            max_sequence_length=300,
                            complex_human_instruction=chi,
                        )
                        group_embeds.append(p_emb.float())
                        group_masks.append(p_mask)
                    embeds.append(group_embeds)
                    masks.append(group_masks)
                empty_emb, empty_mask, _, _ = pipe.encode_prompt(
                    prompt="",
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                    device=device,
                    clean_caption=False,
                    max_sequence_length=300,
                    complex_human_instruction=chi,
                )
                empty_emb = empty_emb.float()
            pipe.text_encoder = None
            pipe.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── 3. inject LoRA into the transformer ──
            targets = list(SANA_ATTN_MODULES)
            if cfg["target_set"] == "attn_ff":
                targets += SANA_FF_MODULES
            rs_note = " · rsLoRA" if cfg["use_rslora"] else ""
            status(
                f"ฉีด LoRA rank {cfg['rank']} "
                f"(alpha={cfg['rank']}, targets={cfg['target_set']}{rs_note})…"
            )
            lora_cfg = LoraConfig(
                r=cfg["rank"],
                lora_alpha=cfg["rank"],
                init_lora_weights="gaussian",
                target_modules=targets,
                bias="none",
                use_rslora=cfg["use_rslora"],
            )
            peft_transformer = inject_adapter_in_model(lora_cfg, transformer)
            trainable = [p for p in peft_transformer.parameters() if p.requires_grad]
            for p in trainable:
                p.data = p.data.float()
            # rsLoRA scales alpha/sqrt(r); the saved file must use standard
            # scaling, so fold sqrt(r) into lora_B on save and undo it on load.
            rs_scale = math.sqrt(cfg["rank"]) if cfg["use_rslora"] else 1.0
            if cfg["init_lora"] is not None:
                status(
                    f"โหลดน้ำหนักตั้งต้นจาก {cfg['init_lora'].name} "
                    f"(rank {cfg['rank']})…"
                )
                loaded, skipped = _load_lora_weights(
                    peft_transformer,
                    cfg["init_lora"],
                    lora_b_scale=1.0 / rs_scale if cfg["use_rslora"] else 1.0,
                )
                print(
                    f"[train] init LoRA {cfg['init_lora'].name}: "
                    f"{loaded} tensors loaded, {len(skipped)} skipped",
                    flush=True,
                )
            try:
                peft_transformer.enable_gradient_checkpointing()
            except Exception as exc:  # noqa: BLE001
                print("[train] gradient checkpointing warning:", exc, flush=True)
            peft_transformer.train()
            peft_transformer.to(device)

            opt = torch.optim.AdamW(
                trainable, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.99)
            )
            total_opt_steps = max(1, cfg["steps"])
            warmup = max(1, int(total_opt_steps * cfg["warmup_ratio"]))
            min_ratio = 0.1

            if cfg["scheduler"] == "cosine":

                def _lr_lambda(s):
                    if s < warmup:
                        return (s + 1) / warmup
                    prog = (s - warmup) / max(1, total_opt_steps - warmup)
                    return min_ratio + (1 - min_ratio) * 0.5 * (
                        1 + math.cos(math.pi * prog)
                    )

            else:

                def _lr_lambda(s):
                    return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
            rng = random.Random(cfg["seed"])
            g_rng = torch.Generator(device="cuda").manual_seed(cfg["seed"])
            shift = 3.0
            ts_scale = float(peft_transformer.config.timestep_scale)
            total = len(latents)
            ema_loss = None

            # EMA of the LoRA weights: averaging the noisy per-step updates
            # usually gives a cleaner adapter on a small dataset than the
            # literal last step. Only LoRA tensors are tracked.
            ema_params: dict[str, torch.Tensor] = {}
            if cfg["ema"]:
                for name, p in peft_transformer.named_parameters():
                    if p.requires_grad and "lora_" in name:
                        ema_params[name] = p.detach().clone().float()
            ema_decay = cfg["ema_decay"] if ema_params else 0.0

            # Resume: restore optimizer/scheduler/RNG/EMA and continue from the
            # saved step.  Only used when the caller explicitly asked for it and
            # a state file with the same rank exists next to the output file.
            start_step = 0
            state_file = trainstate_path(cfg["out_path"])
            if cfg["resume_state"] and state_file.is_file():
                status(f"กู้สถานะการเทรนจาก {state_file.name}…")
                try:
                    start_step, saved_loss = load_train_state(
                        state_file,
                        optimizer=opt,
                        scheduler=sched,
                        ema_params=ema_params,
                        rng=rng,
                        g_rng=g_rng,
                        rank=cfg["rank"],
                    )
                    if saved_loss is not None:
                        ema_loss = float(saved_loss)
                    print(
                        f"[train] resumed at step {start_step}/{cfg['steps']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[train] resume skipped: {exc}", flush=True)
                    start_step = 0

            def _write_lora(path: Path, step_no: int, final: bool) -> None:
                backup = {}
                if ema_params:
                    for name, p in peft_transformer.named_parameters():
                        if name in ema_params:
                            backup[name] = p.detach().clone()
                            p.data.copy_(ema_params[name].to(p.dtype))
                try:
                    _save_lora_state(
                        peft_transformer,
                        path,
                        {
                            "lora_name": cfg["lora_name"],
                            "trigger": ds.get("trigger") or "",
                            "family": "sana",
                            "base_model": sana_entry["name"],
                            "dataset": ds.get("name") or "",
                            "kind": ds.get("kind") or "",
                            "rank": str(cfg["rank"]),
                            "steps": str(cfg["steps"]),
                            "lr": str(cfg["lr"]),
                            "resolution": str(cfg["resolution"]),
                            "scheduler": cfg["scheduler"],
                            "timestep_sampling": cfg["sampling"],
                            "ema": str(bool(ema_params)),
                            "grad_accum": str(cfg["grad_accum"]),
                            "multi_mode": multi_mode,
                            "multi_weight": str(cfg["multi_weight"]),
                            "bucketing": str(cfg["bucketing"]),
                            "caption_dropout": str(cfg["caption_dropout"]),
                            "caption_shuffle": str(cfg["caption_shuffle"]),
                            "keep_tokens": str(cfg["keep_tokens"]),
                            "noise_offset": str(cfg["noise_offset"]),
                            "min_snr_gamma": str(cfg["min_snr_gamma"]),
                            "use_rslora": str(cfg["use_rslora"]),
                            "target_modules": cfg["target_set"],
                            "train_images": str(len(train_images)),
                            "solo_images": str(solo_count),
                            "multi_images": str(multi_count),
                            "init_from": (
                                cfg["init_lora"].name if cfg["init_lora"] else ""
                            ),
                            "checkpoint_step": str(step_no),
                            "loss": f"{ema_loss:.4f}" if ema_loss is not None else "",
                            "trained_at": _now(),
                        },
                        lora_b_scale=rs_scale,
                    )
                finally:
                    for name, p in peft_transformer.named_parameters():
                        if name in backup:
                            p.data.copy_(backup[name])

            def _save_state(lora_file: Path, step_no: int) -> None:
                """Write optimizer/scheduler/RNG state next to the checkpoint."""
                try:
                    save_train_state(
                        trainstate_path(lora_file),
                        optimizer=opt,
                        scheduler=sched,
                        step=step_no,
                        ema_params=ema_params,
                        rng=rng,
                        g_rng=g_rng,
                        ema_loss=ema_loss,
                        rank=cfg["rank"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[train] trainstate save skipped: {exc}", flush=True)

            job.update(
                {
                    "phase": "generating",
                    "message": "เริ่มเทรน LoRA",
                    "steps": cfg["steps"],
                    "step": start_step,
                    "batch": 1,
                    "progress": 0.05,
                    "status": "running",
                    "loss": None,
                    "lora_name": cfg["lora_name"],
                }
            )

            accum = cfg["grad_accum"]
            opt.zero_grad(set_to_none=True)
            thermal_state = {
                "level": 0,
                "clocks": None,
                "sleep": 0.0,
                "step_time": None,
                "applied": False,
                "temp": None,
            }
            for step in range(start_step, cfg["steps"]):
                if cancel.is_set():
                    raise se.Cancelled()
                step_started = time.time()
                if (
                    cfg["thermal_guard"]
                    and step % cfg["thermal_check_every"] == 0
                ):
                    if cfg["thermal_mode"] == "pause":
                        temp = gpu_temperature()
                        if temp is not None and temp >= cfg["thermal_max_c"]:
                            print(
                                f"[train] thermal pause at {temp:.0f}°C "
                                f"(limit {cfg['thermal_max_c']}°C)",
                                flush=True,
                            )
                            thermal_wait(cfg, job, cancel)
                            job["phase"] = "generating"
                            job["message"] = (
                                f"เทรนต่อ · step {step + 1}/{cfg['steps']}"
                            )
                    else:
                        thermal_governor(cfg, job, cancel, thermal_state)
                step_loss = 0.0
                for _micro in range(accum):
                    if sample_weights:
                        index = rng.choices(
                            range(total), weights=sample_weights, k=1
                        )[0]
                    else:
                        index = rng.randrange(total)
                    x0 = latents[index]
                    group = embeds[index]
                    variant = rng.randrange(len(group))
                    emb = group[variant]
                    mask = masks[index][variant]
                    if cfg["augment_flip"] and rng.random() < 0.5:
                        x0 = torch.flip(x0, dims=[-1])
                    if cfg["caption_dropout"] and rng.random() < cfg["caption_dropout"]:
                        emb, mask = empty_emb, empty_mask
                    if cfg["sampling"] == "logit_normal":
                        u = 1.0 / (1.0 + math.exp(-rng.gauss(0.0, 1.0)))
                    else:
                        u = rng.random()
                    sigma = shift * u / (1 + (shift - 1) * u)
                    noise = torch.randn(
                        x0.shape, generator=g_rng, device=device, dtype=x0.dtype
                    )
                    noise = add_noise_offset(noise, cfg["noise_offset"], g_rng)
                    x_t = (1.0 - sigma) * x0 + sigma * noise
                    target = noise - x0
                    t_in = torch.tensor([sigma * 1000.0], device=device) * ts_scale

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = peft_transformer(
                            x_t.to(peft_transformer.dtype),
                            encoder_hidden_states=emb.to(peft_transformer.dtype),
                            encoder_attention_mask=mask,
                            timestep=t_in.expand(x_t.shape[0]),
                            return_dict=False,
                        )[0]
                    pred = pred.float()
                    if cfg["min_snr_gamma"]:
                        snr = ((1.0 - sigma) / max(sigma, 1e-6)) ** 2
                        weight = min_snr_weight(
                            torch.tensor(
                                snr, device=device, dtype=torch.float32
                            ),
                            cfg["min_snr_gamma"],
                        )
                        per_sample = torch.nn.functional.mse_loss(
                            pred, target, reduction="none"
                        ).mean(dim=tuple(range(1, pred.dim())))
                        loss = (per_sample * weight).mean() / accum
                    else:
                        loss = torch.nn.functional.mse_loss(pred, target) / accum
                    loss.backward()
                    step_loss += float(loss.item()) * accum

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                # Thermal throttle: when clock control isn't permitted we slow
                # the loop down with a short pause per step instead of idling
                # the card for tens of seconds at a time.
                if thermal_state["sleep"] > 0:
                    time.sleep(thermal_state["sleep"])
                elapsed_step = time.time() - step_started - thermal_state["sleep"]
                previous = thermal_state.get("step_time")
                thermal_state["step_time"] = (
                    elapsed_step
                    if previous is None
                    else previous * 0.8 + elapsed_step * 0.2
                )

                if ema_params:
                    with torch.no_grad():
                        for name, p in peft_transformer.named_parameters():
                            if name in ema_params:
                                ema_params[name].mul_(ema_decay).add_(
                                    p.detach().float(), alpha=1.0 - ema_decay
                                )

                value = step_loss / accum
                ema_loss = value if ema_loss is None else ema_loss * 0.95 + value * 0.05
                job["step"] = step + 1
                job["loss"] = round(ema_loss, 6)
                job["progress"] = round(0.05 + 0.9 * (step + 1) / cfg["steps"], 4)
                throttle_note = ""
                if thermal_state["level"] > 0:
                    throttle_note = (
                        f" · ชะลอ {cfg['thermal_step_pct'] * thermal_state['level']}%"
                    )
                job["message"] = (
                    f"เทรน LoRA {cfg['lora_name']} · step {step + 1}/{cfg['steps']} "
                    f"· loss {ema_loss:.4f}{throttle_note}"
                )
                if (step + 1) % 25 == 0:
                    print(
                        f"[train] step {step + 1}/{cfg['steps']} loss {ema_loss:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                if (
                    cfg["save_every"]
                    and (step + 1) % cfg["save_every"] == 0
                    and (step + 1) < cfg["steps"]
                ):
                    ckpt = cfg["out_path"].with_name(
                        f"{cfg['out_path'].stem}-step{step + 1}.safetensors"
                    )
                    if not ckpt.exists():
                        print(f"[train] checkpoint step {step + 1} → {ckpt.name}", flush=True)
                        job["message"] = (
                            f"บันทึก checkpoint step {step + 1}/{cfg['steps']}…"
                        )
                        _write_lora(ckpt, step + 1, final=False)
                        _save_state(ckpt, step + 1)

            if cancel.is_set():
                raise se.Cancelled()

            # ── 4. package the final LoRA file (EMA weights if enabled) ──
            status("บันทึกไฟล์ LoRA…")
            out_path = cfg["out_path"]
            if out_path.exists():
                raise RuntimeError(f"มีไฟล์ชื่อ {out_path.name} อยู่แล้ว — ตั้งชื่อใหม่")
            se.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            (se.MODELS_ROOT / "loras").mkdir(parents=True, exist_ok=True)
            _write_lora(out_path, cfg["steps"], final=True)
            _save_state(out_path, cfg["steps"])

            for lora in se.discover_loras():
                if Path(lora["path"]).resolve() == out_path.resolve():
                    job["progress"] = 1.0
                    status(f"เทรนเสร็จ — บันทึก {out_path.name} แล้ว")
                    return {"lora": lora}
            raise RuntimeError("บันทึกแล้วแต่หา LoRA ไม่เจอในรายการ")
        finally:
            try:
                del pipe
            except Exception:  # noqa: BLE001
                pass
