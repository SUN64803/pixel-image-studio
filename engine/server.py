#!/usr/bin/env python3
"""
PixelSmith Image Studio — local engine HTTP API.

Binds loopback only. Serves:
  /                 the program window shell (host/index.html)
  /studio/*         the image generator page (studio/)
  /files/*          generated PNGs + WebP thumbnails (data/)
  /api/*            JSON API used by the page

Jobs run one at a time on a background worker thread; the page polls
/api/jobs/<id> for progress. Nothing ever leaves the machine.
"""
from __future__ import annotations

import getpass
import json
import os
import queue
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sana_engine as se
import sana_trainer as strn
import sdxl_trainer as sdtr
import prompt_presets as ppres
import assistant as ast
import assistant_llm as astllm
import assistant_vision as astvis
import upscaler as upr

ROOT = se.ROOT
DATA = se.DATA
IMAGES = se.IMAGES
THUMBS = se.THUMBS
HISTORY_FILE = DATA / "history.json"
HOST = os.environ.get("IMGS_HOST", "127.0.0.1")
PORT = int(os.environ.get("IMGS_PORT", "47824"))
VERSION = "1.5.2"
MAX_ACTIVE_JOBS = 8
# Per-image upload cap for train datasets (photos are ~1–5 MB).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# ── GPU power modes ─────────────────────────────────────────────────────────
# host/pixelsmith-power-setup installs a tiny root helper (one pkexec prompt)
# and a sudoers rule.  The helper stores the chosen wattage in
# /var/lib/pixelsmith/power.  The app owns the limit while it is running:
#   * startup applies the saved mode
#   * a small in-engine watchdog re-applies it if the driver resets it
#   * shutdown restores the card's default (the saved choice is kept)
# There is deliberately no systemd timer fighting the UI.
POWER_HELPER = "/usr/local/bin/pixelsmith-power"
POWER_SETUP = ROOT / "host" / "pixelsmith-power-setup"
# Presets are fractions of the card's *default* power limit, which makes them
# portable: 60/80/100 % of 250 W is exactly 150/200/250 W on the P102-100.
POWER_PRESETS = (
    ("low", 0.60),
    ("balanced", 0.80),
    ("performance", 1.00),
)
POWER_MIN_FALLBACK = 100
POWER_MAX_FALLBACK = 350
POWER_SUPPORTED = sys.platform.startswith("linux")
POWER_WATCH_INTERVAL = float(
    os.environ.get("PIXELSMITH_POWER_WATCH_INTERVAL", "15")
)
POWER_LOCK = threading.Lock()
_POWER_READY_CACHE = {"at": 0.0, "value": False}
# Last mode chosen in the app (mirrors /var/lib/pixelsmith/power).  Kept in
# memory so the UI can show the saved mode even when the actual limit differs.
_POWER_DESIRED: Optional[int] = None
_POWER_ENFORCE = False
_POWER_WATCH_STOP = threading.Event()
_POWER_WATCH_THREAD: Optional[threading.Thread] = None


@asynccontextmanager
async def _lifespan(_app):
    # The app owns the GPU power limit while it runs: apply the saved mode on
    # startup and restore the card default when the engine stops.
    _power_startup()
    try:
        yield
    finally:
        _power_shutdown()


app = FastAPI(title="PixelSmith Image Studio", version=VERSION, lifespan=_lifespan)

# Everything the pages need is same-origin (scripts, styles, /files/ images,
# /api fetches, the /studio iframe).  The only exception is the data: SVG
# background in studio/style.css.  Strict CSP also silences Electron's
# "Insecure Content-Security-Policy" warning in the dev console.
_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-src 'self'; "
    "frame-ancestors 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

# The engine is a private, loopback-only, same-origin app — no CORS is
# required, and allowing cross-origin requests (the old CORSMiddleware with
# allow_origins="*") would let any website open in the user's browser call
# the API (generate, delete history/LoRA, …).  Extra hostnames (e.g. the
# test client) can be trusted via IMGS_TRUSTED_HOSTS (comma-separated).
_LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
_LOCAL_HOSTNAMES |= {
    h.strip().lower()
    for h in os.environ.get("IMGS_TRUSTED_HOSTS", "").split(",")
    if h.strip()
}


def _hostname(value: str) -> str:
    """Lowercase hostname (no port / brackets) or "" for empty input."""
    host = (value or "").strip()
    if not host:
        return ""
    if ":" in host:
        host = host.rsplit(":", 1)[0]
    return host.strip("[]").lower()


@app.middleware("http")
async def _security_headers(request, call_next):
    # DNS-rebinding / drive-by guard: anything that doesn't claim a local
    # Host, or POSTs/DELETEs with a foreign Origin, is not from our pages.
    if _hostname(request.headers.get("host")) not in _LOCAL_HOSTNAMES | {""}:
        return PlainTextResponse("Forbidden", status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and _hostname(urlparse(origin).hostname) not in _LOCAL_HOSTNAMES:
            return PlainTextResponse("Forbidden", status_code=403)
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response

STATE_LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
HISTORY: list[dict] = []
ACTIVE_JOB: Optional[str] = None
WORK_Q: queue.Queue[str] = queue.Queue()
manager = se.ModelManager()
trainer = strn.SanaTrainer(manager)
sdxl_trainer = sdtr.SdxlTrainer(manager)


# ── helpers ──────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_history() -> list[dict]:
    try:
        raw = json.loads(HISTORY_FILE.read_text("utf-8"))
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict) and r.get("images")]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        print("history load failed:", exc, flush=True)
    return []


HISTORY = load_history()


def persist_history() -> None:
    with STATE_LOCK:
        rows = HISTORY[:800]
    try:
        tmp = HISTORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, HISTORY_FILE)
    except Exception as exc:  # noqa: BLE001
        print("history save failed:", exc, flush=True)


def public_job(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "message": job.get("message"),
        "progress": job.get("progress", 0.0),
        "step": job.get("step", 0),
        "steps": job.get("steps", 0),
        "image_index": job.get("image_index", 0),
        "batch": job.get("batch", 1),
        "loss": job.get("loss"),
        "model_id": job.get("model_id"),
        "model_label": job.get("model_label"),
        "family": job.get("family"),
        "loras": job.get("loras") or [],
        "regions": job.get("regions") or job.get("params", {}).get("regions") or [],
        "images": [dict(x) for x in job.get("images", [])],
        "params": dict(job.get("params", {})),
        "result": job.get("result"),
        "created": job.get("created"),
        "started": job.get("started"),
        "completed": job.get("completed"),
        "elapsed": job.get("elapsed"),
        "error": job.get("error"),
    }


def history_record(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job.get("status"),
        "model": job.get("model_label") or manager.label,
        "model_id": job.get("model_id") or job.get("params", {}).get("model_id"),
        "family": job.get("family"),
        "loras": job.get("loras") or [],
        "params": dict(job.get("params", {})),
        "images": [dict(x) for x in job.get("images", [])],
        "created": job.get("created"),
        "started": job.get("started"),
        "completed": job.get("completed"),
        "elapsed": job.get("elapsed"),
        "error": job.get("error"),
    }


def run_upscale_job(params: dict, job: dict, cancel: threading.Event) -> dict:
    """Upscale one existing image with Real-ESRGAN (runs on the job worker).

    The source image's generation parameters are copied into the new history
    record so the lightbox still shows the original prompt/settings, with an
    extra note pointing at the source file.
    """
    name = str(params.get("file") or "")
    src_path = IMAGES / name
    if not src_path.is_file():
        raise RuntimeError("ไม่พบภาพต้นฉบับ")
    # The worker is serial, so the generation pipeline is idle here.  Free it
    # before loading ESRGAN — this machine has 10 GB VRAM / no swap and must
    # never hold two pipelines at once.
    try:
        manager.unload()
    except Exception:  # noqa: BLE001
        pass
    started = time.perf_counter()
    image = se._load_pil_image(str(src_path))
    scale = 2 if int(params.get("scale") or 4) == 2 else 4
    tile = upr.pick_tile(*image.size, scale=scale, requested=params.get("tile"))

    with STATE_LOCK:
        job["phase"] = "downloading" if not upr.weights_ready() else "upscaling"
        job["message"] = (
            "กำลังดาวน์โหลดโมเดล upscale (~64 MB)…"
            if not upr.weights_ready()
            else "กำลังโหลดโมเดล upscale…"
        )
        job["progress"] = 0.02
        job["steps"] = 1
        job["step"] = 0

    def on_progress(done: int, total: int) -> None:
        with STATE_LOCK:
            job["phase"] = "upscaling"
            job["progress"] = round(0.05 + 0.9 * done / max(1, total), 3)
            job["step"] = done
            job["steps"] = total
            job["message"] = f"กำลังขยาย {done}/{total}"

    canceled = False
    try:
        result = upr.UPSCALER.upscale(
            image, scale=scale, tile=tile, on_progress=on_progress, cancel=cancel
        )
    except upr.UpscaleCancelled:
        # Do not chain the exception: its traceback would keep the tile
        # buffers (up to ~800 MB at 8192 px) alive while unload() runs, so
        # empty_cache() could not return them to the driver.
        canceled = True
    finally:
        # Release the tile buffers back to the driver; the next job reloads
        # the 64 MB model in under a second.
        upr.UPSCALER.unload()
    if canceled:
        raise se.Cancelled()

    meta = dict(params)
    meta.update(
        width=result.width,
        height=result.height,
        model=upr.MODEL_LABEL,
        hires=None,
        elapsed=round(time.perf_counter() - started, 1),
        extra=f"Upscale: {scale}x {upr.MODEL_LABEL} from {name}",
    )
    seed = meta.get("seed")
    entry = se.save_image_file(
        result, job["id"], 0, -1 if seed is None else seed, meta
    )
    with STATE_LOCK:
        job["images"] = [entry]
        job["progress"] = 1.0
    return {
        "file": entry["file"],
        "scale": scale,
        "width": result.width,
        "height": result.height,
    }


def worker_loop() -> None:
    global ACTIVE_JOB
    while True:
        job_id = WORK_Q.get()
        job = None
        started_at = time.perf_counter()
        try:
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    continue
                if job.get("status") == "canceled":
                    continue
                job.update(
                    status="running",
                    phase="starting",
                    message="กำลังเริ่ม…",
                    started=now_iso(),
                )
                cancel = job["_cancel"]
                params = dict(job["params"])
                ACTIVE_JOB = job_id

            try:
                if params.get("job_type") == "train":
                    if params.get("family") == "sdxl":
                        result = sdxl_trainer.train(params, job, cancel)
                    else:
                        result = trainer.train(params, job, cancel)
                    with STATE_LOCK:
                        job["result"] = result
                elif params.get("job_type") == "upscale":
                    result = run_upscale_job(params, job, cancel)
                    with STATE_LOCK:
                        job["result"] = result
                else:
                    manager.generate(params, job, cancel)
                with STATE_LOCK:
                    canceled = cancel.is_set()
                    job["status"] = "canceled" if canceled else "done"
                    job["phase"] = "canceled" if canceled else "done"
                    job["message"] = "ยกเลิกแล้ว" if canceled else "เสร็จแล้ว"
                    if not canceled:
                        job["progress"] = 1.0
            except se.Cancelled:
                with STATE_LOCK:
                    job["status"] = "canceled"
                    job["phase"] = "canceled"
                    job["message"] = "ยกเลิกแล้ว"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                with STATE_LOCK:
                    job["status"] = "error"
                    job["phase"] = "error"
                    job["error"] = str(exc)
                    job["message"] = str(exc)
            finally:
                with STATE_LOCK:
                    job["completed"] = now_iso()
                    job["elapsed"] = round(time.perf_counter() - started_at, 2)
                    job.pop("_cancel", None)
                    if job.get("images"):
                        HISTORY.insert(0, history_record(job))
                        HISTORY[:] = HISTORY[:800]
                if job.get("images"):
                    persist_history()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            with STATE_LOCK:
                ACTIVE_JOB = None
            WORK_Q.task_done()


threading.Thread(target=worker_loop, name="sana-worker", daemon=True).start()


# ── request models ───────────────────────────────────────────────────────────
class LoraSpec(BaseModel):
    id: str = Field(default="", max_length=200)
    weight: float = 1.0


class RegionSpec(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    weight: float = 1.0
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    label: str = Field(default="", max_length=40)


class HiresSpec(BaseModel):
    enabled: bool = False
    scale: float = 1.5
    strength: float = 0.35


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 4.5
    seed: int = -1
    batch: int = 1
    enhance: bool = True
    clean_caption: bool = False
    sampler: str = "dpm"
    max_sequence_length: int = 300
    model_id: Optional[str] = None
    loras: list[LoraSpec] = Field(default_factory=list)
    regions: list[RegionSpec] = Field(default_factory=list)
    inpaint: bool = False
    init_image: Optional[str] = None
    mask_image: Optional[str] = None
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    hires: HiresSpec = Field(default_factory=HiresSpec)


class UpscaleRequest(BaseModel):
    file: str = Field(..., max_length=255)
    scale: int = 4
    # None = pick the tile size from the output size (see upscaler.pick_tile).
    tile: int | None = None


class SelectRequest(BaseModel):
    id: str


class ClearRequest(BaseModel):
    delete_files: bool = True


class LoraImportRequest(BaseModel):
    path: str = Field(default="", max_length=2000)


class LoraDeleteRequest(BaseModel):
    id: str


# ── API ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def api_health():
    with STATE_LOCK:
        active = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
        busy = ACTIVE_JOB is not None
    model = manager.snapshot()
    return {
        "ok": True,
        "version": VERSION,
        "model": model,
        "gpu": se.gpu_info(),
        "queue": active,
        "busy": busy,
        "data_dir": str(DATA),
        "model_dir": model["model_dir"],
    }


@app.get("/api/models")
def api_models():
    return manager.list_models()


@app.post("/api/models/select")
def api_models_select(req: SelectRequest):
    with STATE_LOCK:
        busy = ACTIVE_JOB is not None
    if busy:
        raise HTTPException(409, "มีงานกำลังสร้างภาพอยู่ — ยกเลิกก่อนเปลี่ยนโมเดล")
    if manager.state in ("loading", "preparing"):
        raise HTTPException(409, "กำลังโหลดโมเดลอยู่ — รอสักครู่แล้วลองใหม่")
    try:
        entry = manager.select_model(req.id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "model": manager.snapshot(), "entry": entry}


@app.get("/api/loras")
def api_loras():
    data = manager.list_loras()
    snapshot = manager.snapshot()
    return {"loras": data["loras"], "current": data["current"], "family": snapshot["family"]}


@app.post("/api/loras/import")
def api_loras_import(req: LoraImportRequest):
    path = (req.path or "").strip()
    if not path:
        raise HTTPException(400, "ต้องระบุ path ของ LoRA")
    try:
        lora = manager.import_lora(path)
        return {"ok": True, "lora": lora}
    except Exception as exc:
        raise HTTPException(400, f"นำ LoRA เข้าไม่ได้: {exc}") from exc


@app.post("/api/loras/delete")
def api_loras_delete(req: LoraDeleteRequest):
    lora_id = (req.id or "").strip()
    if not lora_id:
        raise HTTPException(400, "ต้องระบุ LoRA ID")
    try:
        manager.delete_lora(lora_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(400, f"ลบ LoRA ไม่ได้: {exc}") from exc


# ── prompt presets (บันทึก/ใช้/นำเข้า prompt + settings) ────────────────────
class PresetUpsertRequest(BaseModel):
    name: str = Field(default="", max_length=80)
    prompt: str = Field(default="", max_length=8000)
    negative: str = Field(default="", max_length=4000)
    settings: dict = Field(default_factory=dict)


class PresetImportRequest(BaseModel):
    # one of: text (A1111 blob or ### collection format) / from_file (basename
    # of a .txt inside data/) — `name` labels a single A1111 blob import
    text: str = Field(default="", max_length=200_000)
    from_file: str = Field(default="", max_length=120)
    name: str = Field(default="", max_length=80)
    replace_names: bool = False


@app.get("/api/presets")
def api_presets_list():
    return {"presets": ppres.load_presets()}


@app.post("/api/presets")
def api_presets_create(req: PresetUpsertRequest):
    if not (req.name or "").strip():
        raise HTTPException(400, "ตั้งชื่อพรีเซ็ตก่อน")
    if not (req.prompt or "").strip() and not (req.negative or "").strip():
        raise HTTPException(400, "พรีเซ็ตต้องมี prompt หรือ negative อย่างน้อยหนึ่งอย่าง")
    entry = ppres.add_preset(req.name, req.prompt, req.negative, req.settings or {})
    return {"ok": True, "preset": entry}


@app.put("/api/presets/{preset_id}")
def api_presets_update(preset_id: str, req: PresetUpsertRequest):
    updated = ppres.update_preset(
        preset_id,
        name=req.name or None,
        prompt=req.prompt,
        negative=req.negative,
        settings=req.settings or None,
    )
    if updated is None:
        raise HTTPException(404, "ไม่พบพรีเซ็ตนี้")
    return {"ok": True, "preset": updated}


@app.delete("/api/presets/{preset_id}")
def api_presets_delete(preset_id: str):
    if not ppres.delete_preset(preset_id):
        raise HTTPException(404, "ไม่พบพรีเซ็ตนี้")
    return {"ok": True}


@app.post("/api/presets/import")
def api_presets_import(req: PresetImportRequest):
    text = (req.text or "").strip()
    from_file = (req.from_file or "").strip()
    entries: list[dict] = []
    if from_file:
        try:
            text = ppres.read_collection_file(from_file)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        entries.extend(ppres.parse_collection(text))
    elif text:
        if "###" in text:
            entries.extend(ppres.parse_collection(text))
        else:
            blob = ppres.parse_a1111(text)
            if not (blob["prompt"] or blob["negative"] or blob["settings_str"]):
                raise HTTPException(400, "ไม่พบ prompt/parameters ที่อ่านได้ในข้อความ")
            settings = ppres.settings_from_a1111(blob["settings_str"], blob["prompt"])
            cleaned = settings.pop("cleaned_prompt", None)
            entries.append({
                "name": (req.name or "").strip() or "A1111 import",
                "prompt": cleaned or blob["prompt"],
                "negative": blob["negative"],
                "settings": settings,
            })
    if not entries:
        raise HTTPException(400, "ไม่พบพรีเซ็ตที่นำเข้าได้จากข้อความ")
    counts = ppres.import_presets(entries, replace_names=req.replace_names)
    return {"ok": True, **counts, "presets": ppres.load_presets()}


def _assist_free_vram() -> None:
    """Unload the local assistant LLM so SANA gets the whole 10 GB GPU.

    Both fit badly together (Qwen 4B q8 ≈ 4.7 GB + SANA ≈ 5–6 GB), so the
    assistant model is evicted the moment a generation/training job starts.
    """
    try:
        astllm.unload()
    except Exception:  # noqa: BLE001 — never block generation on the assistant
        pass


@app.post("/api/model/load")
def api_model_load():
    if manager.pipe is not None:
        return {"ok": True, "model": manager.snapshot()}
    # Loading a generation pipeline while a job is queued/running would put
    # two pipelines on the 10 GB GPU (the worker loads its own model).
    if _active_count() > 0:
        raise HTTPException(
            409, "มีงานอยู่ในคิว/กำลังทำงาน — รอให้เสร็จก่อนโหลดโมเดล"
        )
    _assist_free_vram()
    manager.load_async()
    return {"ok": True, "model": manager.snapshot(), "started": True}


@app.post("/api/model/unload")
def api_model_unload():
    with STATE_LOCK:
        busy = ACTIVE_JOB is not None
    if busy:
        raise HTTPException(409, "มีงานกำลังสร้างภาพอยู่ — ยกเลิกก่อนแล้วค่อยคืนหน่วยความจำ")
    manager.unload()
    return {"ok": True, "model": manager.snapshot()}


@app.get("/api/upscale")
def api_upscale_info():
    """Availability of the Real-ESRGAN model (downloaded on first use)."""
    return {"ok": True, **upr.UPSCALER.info()}


@app.post("/api/upscale")
def api_upscale(req: UpscaleRequest):
    name = Path(req.file or "").name
    if name != req.file or not (IMAGES / name).is_file():
        raise HTTPException(404, "ไม่พบภาพต้นฉบับ")
    with STATE_LOCK:
        active = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            429, f"คิวเต็ม (สูงสุด {MAX_ACTIVE_JOBS} งาน) — รองานเดิมเสร็จก่อน"
        )
    _assist_free_vram()
    scale = 2 if int(req.scale) == 2 else 4
    # Copy the source record's parameters so the new history entry keeps the
    # original prompt/settings in the lightbox.
    with STATE_LOCK:
        source = next(
            (
                r
                for r in HISTORY
                if any(i.get("file") == name for i in r.get("images", []))
            ),
            None,
        )
    params = dict((source or {}).get("params") or {})
    params.pop("tile", None)
    params.update(
        job_type="upscale",
        file=name,
        scale=scale,
        upscale={"from": name, "scale": scale, "model": upr.MODEL_LABEL},
    )
    if req.tile:
        params["tile"] = max(128, min(1024, int(req.tile)))
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "phase": "queued",
        "message": "อยู่ในคิว",
        "progress": 0.0,
        "step": 0,
        "steps": 1,
        "image_index": 0,
        "batch": 1,
        "images": [],
        "params": params,
        "model_id": upr.MODEL_ID,
        "model_label": upr.MODEL_LABEL,
        "family": "upscale",
        "loras": [],
        "created": now_iso(),
        "started": None,
        "completed": None,
        "elapsed": None,
        "error": None,
        "_cancel": threading.Event(),
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    WORK_Q.put(job_id)
    return {"ok": True, "job_id": job_id, "job": public_job(job)}


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "กรุณาใส่พรอมป์ตก่อนสร้างภาพ")
    with STATE_LOCK:
        active = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            429, f"คิวเต็ม (สูงสุด {MAX_ACTIVE_JOBS} งาน) — รองานเดิมเสร็จก่อน"
        )
    _assist_free_vram()
    try:
        entry = manager.get_model(req.model_id or manager.current_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not entry.get("available"):
        raise HTTPException(400, f"โมเดล {entry['name']} ยังไม่พร้อมใช้งาน")
    loras: list[dict] = []
    for item in req.loras:
        lora_id = (item.id or "").strip()
        if not lora_id:
            continue
        try:
            weight = float(item.weight)
        except (TypeError, ValueError):
            weight = 1.0
        loras.append({"id": lora_id, "weight": max(0.0, min(2.0, weight))})
    params = req.model_dump()
    family = entry.get("family")
    params.update(
        prompt=prompt,
        width=se.norm_size(req.width),
        height=se.norm_size(req.height),
        steps=max(2, min(60, int(req.steps))),
        guidance=max(1.0, min(15.0, float(req.guidance))),
        batch=max(1, min(4, int(req.batch))),
        sampler=(req.sampler or "dpm").lower(),
        max_sequence_length=max(64, min(512, int(req.max_sequence_length))),
        model_id=entry["id"],
        loras=loras,
        regions=se.normalize_regions([r.model_dump() for r in req.regions]) if entry.get("family") == "sdxl" else [],
        inpaint=req.inpaint and entry.get("family") == "sdxl",
        init_image=req.init_image if req.inpaint else None,
        mask_image=req.mask_image if req.inpaint else None,
        strength=max(0.0, min(1.0, float(req.strength))),
        hires={
            "enabled": bool(req.hires.enabled)
            and family in se.HIRES_FAMILIES
            and not (req.inpaint and family == "sdxl"),
            "scale": max(1.0, min(2.0, float(req.hires.scale))),
            "strength": max(0.05, min(0.75, float(req.hires.strength))),
        },
    )
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "phase": "queued",
        "message": "อยู่ในคิว",
        "progress": 0.0,
        "step": 0,
        "steps": params["steps"],
        "image_index": 0,
        "batch": params["batch"],
        "images": [],
        "params": params,
        "created": now_iso(),
        "started": None,
        "completed": None,
        "elapsed": None,
        "error": None,
        "_cancel": threading.Event(),
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    WORK_Q.put(job_id)
    return {"ok": True, "job_id": job_id, "job": public_job(job)}


def ordered_active_jobs() -> list[dict]:
    """Active jobs in true execution order: running first, then queue order."""
    with WORK_Q.mutex:
        queued_ids = list(WORK_Q.queue)
    with STATE_LOCK:
        jobs = {j["id"]: j for j in JOBS.values() if j.get("status") in ("queued", "running")}
    ordered = [j for j in jobs.values() if j.get("status") == "running"]
    seen = {j["id"] for j in ordered}
    for jid in queued_ids:
        if jid in seen:
            continue
        job = jobs.get(jid)
        if job is not None:
            ordered.append(job)
            seen.add(jid)
    for jid, job in jobs.items():
        if jid not in seen:
            ordered.append(job)
            seen.add(jid)
    return ordered


@app.get("/api/jobs")
def api_jobs(active: int = 0, limit: int = 20):
    if active:
        jobs = ordered_active_jobs()
    else:
        with STATE_LOCK:
            jobs = list(JOBS.values())
        jobs = sorted(jobs, key=lambda j: j.get("created") or "", reverse=True)
    return {"jobs": [public_job(j) for j in jobs[: max(1, min(100, limit))]]}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "ไม่พบงานนี้")
        return public_job(job)


@app.post("/api/jobs/{job_id}/move")
def api_move(job_id: str, direction: str = "up"):
    direction = (direction or "up").lower()
    if direction not in ("up", "down"):
        raise HTTPException(400, "direction ต้องเป็น up หรือ down")
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "ไม่พบงานนี้")
        if job.get("status") != "queued":
            raise HTTPException(409, "จัดลำดับได้เฉพาะงานที่ยังอยู่ในคิว")
    with WORK_Q.mutex:
        ids = list(WORK_Q.queue)
        if job_id not in ids:
            raise HTTPException(409, "งานนี้ไม่ได้อยู่ในคิวแล้ว")
        idx = ids.index(job_id)
        new_idx = idx - 1 if direction == "up" else idx + 1
        if not (0 <= new_idx < len(ids)):
            raise HTTPException(409, "อยู่ปลายคิวแล้ว")
        ids.pop(idx)
        ids.insert(new_idx, job_id)
        WORK_Q.queue.clear()
        WORK_Q.queue.extend(ids)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "ไม่พบงานนี้")
        event = job.get("_cancel")
        if event is not None:
            event.set()
        if job.get("status") == "queued":
            job["status"] = "canceled"
            job["phase"] = "canceled"
            job["message"] = "ยกเลิกแล้ว"
            job["completed"] = now_iso()
    # a canceled queued job no longer needs its slot in the queue
    if job.get("status") == "canceled":
        with WORK_Q.mutex:
            try:
                WORK_Q.queue.remove(job_id)
            except ValueError:
                pass
    return {"ok": True, "job": public_job(job)}


@app.get("/api/history")
def api_history(offset: int = 0, limit: int = 60):
    offset = max(0, offset)
    with STATE_LOCK:
        total = len(HISTORY)
        items = HISTORY[offset : offset + max(1, min(200, limit))]
        items = [dict(x) for x in items]
    return {"items": items, "total": total, "offset": offset}


@app.delete("/api/history/{job_id}")
def api_history_delete(job_id: str):
    with STATE_LOCK:
        record = next((r for r in HISTORY if r.get("id") == job_id), None)
        if record is None:
            raise HTTPException(404, "ไม่พบภาพชุดนี้")
        HISTORY.remove(record)
    remove_files(record.get("images", []))
    persist_history()
    return {"ok": True}


@app.post("/api/history/clear")
def api_history_clear(req: ClearRequest):
    # Clearing deletes every file in data/images — doing that while a job is
    # writing new images would leave the job with broken thumbnails/records.
    if _active_count() > 0:
        raise HTTPException(409, "มีงานอยู่ในคิว/กำลังทำงาน — รอให้เสร็จก่อนล้างประวัติ")
    with STATE_LOCK:
        count = len(HISTORY)
        HISTORY.clear()
    if req.delete_files:
        for path in list(IMAGES.glob("*")):
            try:
                path.unlink()
            except OSError:
                pass
        for path in list(THUMBS.glob("*")):
            try:
                path.unlink()
            except OSError:
                pass
    persist_history()
    return {"ok": True, "removed": count}


def remove_files(images: list[dict]) -> None:
    for image in images:
        name = image.get("file")
        if not name:
            continue
        for path in (IMAGES / name, THUMBS / (Path(name).stem + ".webp")):
            try:
                path.unlink()
            except OSError:
                pass


@app.post("/api/open-folder")
def api_open_folder(which: str = "images"):
    target = {
        "images": IMAGES,
        "thumbs": THUMBS,
        "data": DATA,
        "models": se.MODELS_ROOT,
        "loras": se.MODELS_ROOT / "loras",
        "logs": DATA,
        "backups": DATA / "backups",
    }.get(which, IMAGES)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(
                ["xdg-open", str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"เปิดโฟลเดอร์ไม่ได้: {exc}") from exc
    return {"ok": True, "path": str(target)}


# ── LoRA trainer (SANA) ─────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    lora_name: str = Field(default="", max_length=60)
    # Base model to train against. Empty = currently selected model. The
    # family (sana / sdxl) is derived from this entry, never trusted from the
    # client, and selects the matching trainer.
    model_id: str = Field(default="", max_length=200)
    rank: int = 16
    steps: int = 1500
    lr: float = 2e-4
    resolution: int = 1024
    enhance: bool = True
    seed: int = -1
    # quality knobs
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    ema: bool = False
    ema_decay: float = 0.999
    timestep_sampling: str = "logit_normal"
    save_every: int = 0
    augment_flip: bool = False
    grad_accum: int = 1
    # multi-character policy (exclude | weighted | include)
    multi_mode: str = "exclude"
    multi_weight: float = 0.25
    # recipe options (Phase 2)
    bucketing: bool = True
    caption_dropout: float = 0.05
    caption_shuffle: bool = False
    keep_tokens: int = 1
    noise_offset: float = 0.0
    min_snr_gamma: float = 0.0
    use_rslora: bool = False
    target_modules: str = "attn"  # attn | attn_ff
    resume_state: bool = False
    # continue training from an existing adapter (id/name from /api/loras)
    init_lora: str = Field(default="", max_length=200)
    # thermal guard: slow the GPU down (default) instead of pausing it
    thermal_guard: bool = True
    thermal_mode: str = "throttle"  # throttle | pause
    thermal_max_c: int = 90
    thermal_throttle_c: int = 87
    thermal_resume_c: int = 80
    thermal_step_pct: int = 5
    thermal_check_every: int = 10


class SampleRequest(BaseModel):
    """Post-training sample: a normal generation job using the fresh LoRA."""

    lora_id: str = Field(default="", max_length=200)
    model_id: str = Field(default="", max_length=200)
    prompt: str = Field(default="", max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 4.5
    seed: int = -1
    batch: int = 4


def _active_count() -> int:
    with STATE_LOCK:
        return sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))


def _enqueue_job(job_id: str, job: dict) -> None:
    with STATE_LOCK:
        JOBS[job_id] = job
    WORK_Q.put(job_id)


def _check_capacity() -> None:
    if _active_count() >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            429, f"คิวเต็ม (สูงสุด {MAX_ACTIVE_JOBS} งาน) — รองานเดิมเสร็จก่อน"
        )


def _new_job(job_id: str, params: dict, steps: int, message: str) -> dict:
    return {
        "id": job_id,
        "status": "queued",
        "phase": "queued",
        "message": message,
        "progress": 0.0,
        "step": 0,
        "steps": steps,
        "image_index": 0,
        "batch": 1,
        "images": [],
        "params": params,
        "created": now_iso(),
        "started": None,
        "completed": None,
        "elapsed": None,
        "error": None,
        "_cancel": threading.Event(),
    }


@app.get("/api/train/datasets")
def api_train_datasets():
    return {"datasets": strn.load_registry(), "resolution_options": strn._RESOLUTIONS}


@app.post("/api/train/datasets")
def api_train_create_dataset(req: dict):
    name = (req.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "ตั้งชื่อ dataset ก่อน")
    entry = strn.create_dataset(
        name,
        str(req.get("trigger") or ""),
        str(req.get("kind") or "character"),
        req.get("extra_triggers"),
    )
    return {"ok": True, "dataset": entry}


@app.patch("/api/train/datasets/{ds_id}")
def api_train_update_dataset(ds_id: str, req: dict):
    try:
        entry = strn.update_dataset(
            ds_id,
            name=req.get("name"),
            trigger=req.get("trigger"),
            kind=req.get("kind"),
            extra_triggers=req.get("extra_triggers"),
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "dataset": entry}


@app.delete("/api/train/datasets/{ds_id}")
def api_train_delete_dataset(ds_id: str):
    if strn.get_dataset(ds_id) is None:
        raise HTTPException(404, "ไม่พบ dataset นี้")
    strn.delete_dataset(ds_id)
    return {"ok": True}


@app.post("/api/train/datasets/{ds_id}/images")
async def api_train_add_images(ds_id: str, files: list[UploadFile] = File(...)):
    added, errors = [], []
    for upload in files[: strn.MAX_IMAGES_PER_DATASET]:
        # Reject oversized uploads before reading — upload.read() pulls the
        # whole file into RAM and this machine has 15 GB and no swap.
        size = getattr(upload, "size", None)
        if size and size > MAX_UPLOAD_BYTES:
            errors.append(
                f"{upload.filename or 'image'}: ไฟล์ใหญ่เกิน "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.filename or "img").suffix) as tmp:
                tmp.file.write(await upload.read())
                tmp_path = tmp.name
            entry = strn.add_image(ds_id, tmp_path, upload.filename or "")
            added.append(entry)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:  # noqa: BLE001
                    pass
    if not added and errors:
        raise HTTPException(400, errors[0])
    return {"ok": True, "added": added, "errors": errors}


@app.delete("/api/train/datasets/{ds_id}/images/{file_id}")
def api_train_remove_image(ds_id: str, file_id: str):
    try:
        strn.remove_image(ds_id, file_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@app.put("/api/train/datasets/{ds_id}/captions")
def api_train_set_captions(ds_id: str, req: dict):
    updates = req.get("updates")
    if updates is None:
        updates = []
    if not isinstance(updates, list):
        raise HTTPException(400, "updates ต้องเป็น list")
    updates = [item for item in updates if isinstance(item, dict)]
    try:
        entry = strn.set_captions(ds_id, updates)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "dataset": entry}


@app.get("/api/train/datasets/{ds_id}/qc")
def api_train_dataset_qc(ds_id: str, resolution: int = 1024):
    try:
        return strn.dataset_qc(ds_id, max(256, min(1536, int(resolution))))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/train/datasets/{ds_id}/sample")
def api_train_sample(ds_id: str, req: SampleRequest):
    ds = strn.get_dataset(ds_id)
    if ds is None:
        raise HTTPException(404, "ไม่พบ dataset นี้")
    prompt = (req.prompt or "").strip()
    if not prompt:
        prompt = ", ".join(
            part
            for part in [
                ds.get("trigger") or "",
                *[str(t) for t in (ds.get("extra_triggers") or [])],
            ]
            if part
        )
    if not prompt:
        raise HTTPException(400, "ใส่พรอมป์ตหรือ trigger ก่อนสร้างตัวอย่าง")
    gen = GenerateRequest(
        prompt=prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance=req.guidance,
        seed=req.seed,
        batch=req.batch,
        model_id=req.model_id or None,
        loras=[LoraSpec(id=req.lora_id, weight=1.0)] if req.lora_id else [],
    )
    return api_generate(gen)


@app.post("/api/train/datasets/{ds_id}/train")
def api_train_start(ds_id: str, req: TrainRequest):
    _check_capacity()
    _assist_free_vram()
    if strn.get_dataset(ds_id) is None:
        raise HTTPException(404, "ไม่พบ dataset นี้")
    try:
        base_model = manager.get_model(req.model_id or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    family = base_model.get("family")
    if family not in ("sana", "sdxl"):
        raise HTTPException(
            400,
            "ยังไม่รองรับการเทรนโมเดลตระกูล "
            f"{base_model.get('family_label') or family}",
        )
    if not (req.lora_name or "").strip():
        raise HTTPException(400, "ตั้งชื่อ LoRA ก่อนเทรน")
    if strn.sanitize_lora_name(req.lora_name) == "lora":
        raise HTTPException(400, "ชื่อ LoRA ต้องมีตัวอักษร/ตัวเลข")
    out_path = se.MODELS_ROOT / "loras" / f"{strn.sanitize_lora_name(req.lora_name)}.safetensors"
    if out_path.exists():
        raise HTTPException(409, f"มี LoRA ชื่อ {out_path.stem} อยู่แล้ว — ตั้งชื่อใหม่")
    # Two queued jobs with the same output name would collide at save time
    # (the second one fails only after training for an hour).
    with STATE_LOCK:
        duplicate = next(
            (
                j
                for j in JOBS.values()
                if j.get("status") in ("queued", "running")
                and (j.get("params") or {}).get("job_type") == "train"
                and (j.get("params") or {}).get("lora_name") == out_path.stem
            ),
            None,
        )
    if duplicate is not None:
        raise HTTPException(409, f"มีงานเทรนชื่อ {out_path.stem} อยู่ในคิวแล้ว")
    ds = strn.get_dataset(ds_id)
    if not ds.get("images"):
        raise HTTPException(400, "dataset ยังไม่มีรูป — เพิ่มรูปก่อนเทรน")
    # Multi-character policy: fail now (not after loading the model) when the
    # selected policy leaves no images to train on.
    try:
        strn.select_train_images(
            ds.get("images") or [],
            req.multi_mode,
            req.multi_weight,
            ds.get("kind") or "character",
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Continue-from: resolve early so a typo fails now, not after loading the model.
    try:
        init_path = strn._resolve_init_lora(req.init_lora)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    if init_path is not None and init_path.resolve() == out_path.resolve():
        raise HTTPException(400, "LoRA ตั้งต้นชื่อเดียวกับผลลัพธ์ — ตั้งชื่อใหม่")
    if init_path is not None:
        lora_entry = next(
            (
                l
                for l in se.discover_loras()
                if Path(str(l.get("path") or "")).resolve() == init_path.resolve()
            ),
            None,
        )
        if lora_entry and lora_entry.get("family") not in (None, family):
            raise HTTPException(
                400,
                f"LoRA ตั้งต้นเป็นตระกูล {lora_entry.get('family_label')} "
                f"ไม่ตรงกับโมเดลฐาน ({base_model.get('family_label')})",
            )
    params = {
        "job_type": "train",
        "family": family,
        "model_id": base_model["id"],
        "dataset_id": ds_id,
        "lora_name": strn.sanitize_lora_name(req.lora_name),
        "rank": req.rank,
        "steps": req.steps,
        "lr": req.lr,
        "resolution": req.resolution,
        "enhance": req.enhance,
        "seed": req.seed,
        "scheduler": req.scheduler,
        "warmup_ratio": req.warmup_ratio,
        "ema": req.ema,
        "ema_decay": req.ema_decay,
        "timestep_sampling": req.timestep_sampling,
        "save_every": req.save_every,
        "augment_flip": req.augment_flip,
        "grad_accum": req.grad_accum,
        "multi_mode": req.multi_mode,
        "multi_weight": req.multi_weight,
        "bucketing": req.bucketing,
        "caption_dropout": req.caption_dropout,
        "caption_shuffle": req.caption_shuffle,
        "keep_tokens": req.keep_tokens,
        "noise_offset": req.noise_offset,
        "min_snr_gamma": req.min_snr_gamma,
        "use_rslora": req.use_rslora,
        "target_modules": req.target_modules,
        "resume_state": req.resume_state,
        "thermal_guard": req.thermal_guard,
        "thermal_mode": req.thermal_mode,
        "thermal_max_c": req.thermal_max_c,
        "thermal_throttle_c": req.thermal_throttle_c,
        "thermal_resume_c": req.thermal_resume_c,
        "thermal_step_pct": req.thermal_step_pct,
        "thermal_check_every": req.thermal_check_every,
        "init_lora": req.init_lora,
    }
    job_id = uuid.uuid4().hex
    job = _new_job(job_id, params, params["steps"], "อยู่ในคิว (เทรน LoRA)")
    _enqueue_job(job_id, job)
    return {"ok": True, "job_id": job_id, "job": public_job(job)}


@app.get("/api/log")
def api_log(tail: int = 200):
    path = DATA / "engine.log"
    try:
        lines = path.read_text("utf-8", errors="replace").splitlines()
        return {"path": str(path), "lines": lines[-max(1, min(2000, tail)) :]}
    except FileNotFoundError:
        return {"path": str(path), "lines": []}


# ── in-app assistant (P1: rule-based brain + tool registry) ─────────────────
ASSIST_PENDING = ast.ActionStore(limit=32, ttl=600)
ASSIST_UNDO = ast.ActionStore(limit=50, ttl=3600)


def _assist_dataset(ds_id=None) -> dict:
    if ds_id:
        ds = strn.get_dataset(str(ds_id))
        if ds is None:
            raise ValueError("ไม่พบ dataset นี้")
        return ds
    datasets = strn.load_registry()
    if not datasets:
        raise ValueError("ยังไม่มี dataset — สร้างชุดเทรนก่อน")
    return datasets[0]


def _assist_error(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc) or exc.__class__.__name__


def _assist_dataset_summary(args: dict) -> dict:
    ds = _assist_dataset(args.get("dataset_id"))
    images = [i for i in (ds.get("images") or []) if isinstance(i, dict)]
    solo_images = [i for i in images if not i.get("multi")]
    multi = len(images) - len(solo_images)
    missing = [i for i in solo_images if not ast.has_angle_tag(i.get("caption"))]
    fillable = [i for i in missing if ast.missing_angle_tags(i.get("caption"))]
    manual = [i for i in missing if not ast.missing_angle_tags(i.get("caption"))]
    angles: list[str] = []
    for image in images:
        text = str(image.get("caption") or "").lower()
        for tag in ast.ANGLE_TAGS:
            if tag in text and tag not in angles:
                angles.append(tag)
    likely = [i for i in solo_images if strn.looks_multi_caption(i.get("caption"))]
    notes: list[str] = []
    if len(solo_images) < 16:
        notes.append(f"ต้องมีรูปเดี่ยวอย่างน้อย 16 รูป (มี {len(solo_images)})")
    if len(angles) < 4:
        notes.append(f"ต้องมีอย่างน้อย 4 มุมที่ต่างกัน (มี {len(angles)})")
    if missing:
        notes.append(f"อีก {len(missing)} รูปยังไม่มี tag มุม (เติมอัตโนมัติได้ {len(fillable)})")
    if likely:
        notes.append(f"{len(likely)} รูปเดี่ยว caption บ่งชี้ว่ามีคนที่สอง — ตรวจว่า mark multi ครบ")
    return {
        "id": ds.get("id"),
        "name": ds.get("name"),
        "trigger": ds.get("trigger"),
        "total": len(images),
        "solo": len(solo_images),
        "multi": multi,
        "missing_tags": len(missing),
        "fillable": len(fillable),
        "manual": len(manual),
        "manual_files": [i.get("file") for i in manual],
        "angles": angles,
        "ready": len(solo_images) >= 16 and len(angles) >= 4 and not missing,
        "notes": notes,
    }


def _assist_list_datasets(args: dict) -> dict:
    rows = []
    for ds in strn.load_registry():
        images = [i for i in (ds.get("images") or []) if isinstance(i, dict)]
        solo = sum(1 for i in images if not i.get("multi"))
        rows.append(
            {
                "id": ds.get("id"),
                "name": ds.get("name"),
                "trigger": ds.get("trigger"),
                "total": len(images),
                "solo": solo,
                "multi": len(images) - solo,
            }
        )
    return {"datasets": rows}


def _assist_list_loras(args: dict) -> dict:
    data = manager.list_loras()
    rows = [
        {
            "id": l.get("id"),
            "name": l.get("name"),
            "family": l.get("family"),
            "family_label": l.get("family_label"),
            "size_mb": round(float(l.get("size") or 0) / 1024 / 1024, 1),
        }
        for l in data.get("loras") or []
    ]
    return {"loras": rows}


def _assist_list_models(args: dict) -> dict:
    data = manager.list_models()
    rows = [
        {
            "id": m.get("id"),
            "name": m.get("name"),
            "family": m.get("family"),
            "family_label": m.get("family_label"),
            "available": bool(m.get("available")),
        }
        for m in data.get("models") or []
    ]
    return {"models": rows, "current": data.get("current")}


def _assist_gpu_status(args: dict) -> dict:
    info = se.gpu_info()
    snapshot = manager.snapshot()
    with STATE_LOCK:
        queued = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
        busy = ACTIVE_JOB is not None
    return {
        "gpu": info.get("gpu") or info.get("device"),
        "temp": info.get("temp"),
        "vram_used_mb": round(float(info.get("vram_used") or 0) / 1024 / 1024),
        "vram_total_mb": round(float(info.get("vram_total") or 0) / 1024 / 1024),
        "model_state": snapshot.get("state"),
        "model_name": snapshot.get("name"),
        "queue": queued,
        "busy": busy,
    }


def _assist_history_stats(args: dict) -> dict:
    with STATE_LOCK:
        rows = [dict(r) for r in HISTORY]
    images = sum(len(r.get("images") or []) for r in rows)
    latest = rows[0] if rows else None
    return {
        "total": len(rows),
        "images": images,
        "latest": {
            "id": latest.get("id"),
            "created": latest.get("created"),
            "model": latest.get("model"),
        }
        if latest
        else None,
    }


def _assist_list_jobs(args: dict) -> dict:
    rows = [
        {
            "id": j.get("id"),
            "status": j.get("status"),
            "message": j.get("message"),
            "progress": j.get("progress"),
        }
        for j in ordered_active_jobs()
    ]
    return {"jobs": rows}


def _assist_read_images(args: dict) -> tuple[dict, object]:
    """Start a local vision pass over dataset images (one at a time)."""
    if astvis.RUNNER.running():
        raise ValueError("กำลังอ่านรูปชุดก่อนอยู่ — รอให้เสร็จก่อน")
    ds = _assist_dataset(args.get("dataset_id"))
    with STATE_LOCK:
        busy = ACTIVE_JOB is not None
        active = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
    if busy or active or manager.state in ("loading", "preparing"):
        raise ValueError(
            "GPU กำลังทำงาน (สร้างภาพ/โหลดโมเดล) — รอให้เสร็จก่อนแล้วสั่งอ่านรูปใหม่"
        )
    if not astvis.available():
        raise ValueError(
            f"โมเดล {astllm.model_name()} ไม่รองรับ vision หรือ Ollama ไม่พร้อม"
        )
    images = [i for i in (ds.get("images") or []) if isinstance(i, dict)]
    if args.get("files"):
        wanted = {str(x) for x in args["files"] if str(x).strip()}
        images = [i for i in images if i.get("file") in wanted]
    if not images:
        raise ValueError("ไม่มีรูปให้อ่านใน dataset นี้")
    if len(images) > astvis.MAX_IMAGES:
        images = images[: astvis.MAX_IMAGES]
    # Vision uses the local LLM: free SANA so only one model sits in VRAM.
    if manager.pipe is not None:
        try:
            manager.unload()
        except Exception:  # noqa: BLE001 — start anyway; Ollama manages its own memory
            pass
    payload = dict(ds)
    payload["dir"] = str(strn.dataset_dir_path(ds.get("id")))
    job = astvis.RUNNER.start(payload, images, astllm.model_name())
    return {
        "job_id": job["id"],
        "dataset_id": ds.get("id"),
        "dataset_name": ds.get("name"),
        "total": job["total"],
    }, None


def _assist_fill_tags(args: dict) -> tuple[dict, object]:
    ds = _assist_dataset(args.get("dataset_id"))
    updates, previous, filled = [], [], []
    for image in ds.get("images") or []:
        if image.get("multi"):
            continue
        add = ast.missing_angle_tags(image.get("caption"))
        if not add:
            continue
        previous.append({"file": image.get("file"), "caption": image.get("caption") or ""})
        updates.append(
            {
                "file": image.get("file"),
                "caption": ast.with_angle_tags(image.get("caption"), add),
            }
        )
        filled.append(f"{image.get('file')} → {', '.join(add)}")
    if not updates:
        return {
            "dataset_id": ds.get("id"),
            "filled": 0,
            "files": [],
            "remaining": len(
                [
                    i
                    for i in (ds.get("images") or [])
                    if not i.get("multi") and not ast.has_angle_tag(i.get("caption"))
                ]
            ),
        }, None
    strn.set_captions(ds.get("id"), updates)

    def undo():
        strn.set_captions(ds.get("id"), previous)

    remaining = len(
        [
            i
            for i in (ds.get("images") or [])
            if not i.get("multi") and not ast.has_angle_tag(i.get("caption"))
        ]
    ) - len(updates)
    return {
        "dataset_id": ds.get("id"),
        "filled": len(updates),
        "files": filled,
        "remaining": max(0, remaining),
    }, undo


def _assist_set_trigger(args: dict) -> tuple[dict, object]:
    ds = _assist_dataset(args.get("dataset_id"))
    value = str(args.get("trigger") or "").strip()
    if not value:
        raise ValueError("ต้องระบุ trigger ใหม่")
    new = strn.sanitize_trigger(value)
    if not new:
        raise ValueError("trigger ต้องมีตัวอักษรหรือตัวเลข")
    old = ds.get("trigger") or ""
    strn.update_dataset(ds.get("id"), trigger=new)

    def undo():
        strn.update_dataset(ds.get("id"), trigger=old)

    return {"dataset_id": ds.get("id"), "old": old, "new": new}, undo


def _assist_set_caption(args: dict) -> tuple[dict, object]:
    ds = _assist_dataset(args.get("dataset_id"))
    file_id = str(args.get("file") or "").strip()
    caption = str(args.get("caption") or "").strip()
    if not file_id:
        raise ValueError("ต้องระบุชื่อไฟล์")
    image = next(
        (
            i
            for i in (ds.get("images") or [])
            if i.get("file") == file_id or Path(str(i.get("file") or "")).stem == file_id
        ),
        None,
    )
    if image is None:
        raise ValueError(f"ไม่พบไฟล์ {file_id} ใน dataset นี้")
    old = image.get("caption") or ""
    strn.set_captions(ds.get("id"), [{"file": image.get("file"), "caption": caption}])

    def undo():
        strn.set_captions(ds.get("id"), [{"file": image.get("file"), "caption": old}])

    return {"dataset_id": ds.get("id"), "file": image.get("file"), "old": old, "new": caption}, undo


def _assist_select_model(args: dict) -> tuple[dict, object]:
    target = str(args.get("target") or "").strip().lower()
    if not target:
        raise ValueError("ต้องระบุชื่อหรือ id ของโมเดล")
    models = manager.list_models().get("models") or []
    found = next((m for m in models if str(m.get("id")).lower() == target), None)
    if found is None:
        found = next((m for m in models if target in str(m.get("name") or "").lower()), None)
    if found is None:
        found = next((m for m in models if target in str(m.get("id") or "").lower()), None)
    if found is None:
        raise ValueError(f"ไม่พบโมเดล “{args.get('target')}”")
    old = manager.current_id
    manager.select_model(found.get("id"))

    def undo():
        if old:
            manager.select_model(old)

    return {
        "old": old,
        "new": found.get("id"),
        "name": found.get("name"),
    }, undo


def _assist_generate(args: dict) -> tuple[dict, object]:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("ต้องระบุพรอมป์ต")
    res = api_generate(GenerateRequest(prompt=prompt))
    return {"job_id": res.get("job_id"), "prompt": prompt}, None


def _assist_delete_history(args: dict) -> tuple[dict, object]:
    job_id = str(args.get("job_id") or "").strip()
    with STATE_LOCK:
        if job_id:
            record = next((r for r in HISTORY if r.get("id") == job_id), None)
        else:
            record = HISTORY[0] if HISTORY else None
        if record is None:
            raise ValueError("ไม่พบภาพที่จะลบ")
        HISTORY.remove(record)
    remove_files(record.get("images") or [])
    persist_history()
    return {
        "job_id": record.get("id"),
        "removed": 1,
        "images": len(record.get("images") or []),
    }, None


def _assist_clear_history(args: dict) -> tuple[dict, object]:
    if _active_count() > 0:
        raise ValueError("มีงานในคิว/กำลังทำงาน — รอให้เสร็จก่อนล้างประวัติ")
    with STATE_LOCK:
        removed = len(HISTORY)
        HISTORY.clear()
    for path in list(IMAGES.glob("*")):
        try:
            path.unlink()
        except OSError:
            pass
    for path in list(THUMBS.glob("*")):
        try:
            path.unlink()
        except OSError:
            pass
    persist_history()
    return {"removed": removed}, None


def _assist_delete_lora(args: dict) -> tuple[dict, object]:
    target = str(args.get("id") or "").strip()
    if not target:
        raise ValueError("ต้องระบุชื่อหรือ id ของ LoRA")
    loras = manager.list_loras().get("loras") or []
    found = next((l for l in loras if str(l.get("id")) == target), None)
    if found is None:
        found = next(
            (l for l in loras if Path(str(l.get("path") or "")).stem == Path(target).stem),
            None,
        )
    if found is None:
        found = next(
            (l for l in loras if target.lower() in str(l.get("name") or "").lower()), None
        )
    if found is None:
        raise ValueError(f"ไม่พบ LoRA “{target}”")
    manager.delete_lora(found.get("id"))
    return {"id": found.get("id"), "name": found.get("name")}, None


def _assist_start_training(args: dict) -> tuple[dict, object]:
    ds = _assist_dataset(args.get("dataset_id"))
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("ต้องระบุชื่อ LoRA")
    req = TrainRequest(
        lora_name=name,
        steps=int(args.get("steps") or 1500),
        rank=int(args.get("rank") or 32),
    )
    res = api_train_start(ds.get("id"), req)
    return {
        "dataset_id": ds.get("id"),
        "job_id": res.get("job_id"),
        "lora_name": strn.sanitize_lora_name(name),
        "steps": req.steps,
        "rank": req.rank,
    }, None


_ASSIST_EXECUTORS = {
    "dataset_summary": _assist_dataset_summary,
    "list_datasets": _assist_list_datasets,
    "list_loras": _assist_list_loras,
    "list_models": _assist_list_models,
    "gpu_status": _assist_gpu_status,
    "history_stats": _assist_history_stats,
    "list_jobs": _assist_list_jobs,
    "read_images": _assist_read_images,
    "fill_angle_tags": _assist_fill_tags,
    "set_trigger": _assist_set_trigger,
    "set_caption": _assist_set_caption,
    "select_model": _assist_select_model,
    "generate_image": _assist_generate,
    "delete_history": _assist_delete_history,
    "clear_history": _assist_clear_history,
    "delete_lora": _assist_delete_lora,
    "start_training": _assist_start_training,
}


def _assist_execute(tool: str, args: dict, lang: str = "th", brain=None):
    if tool == "help":
        return {"text": ast.help_text(brain, lang)}, None
    fn = _ASSIST_EXECUTORS.get(tool)
    if fn is None:
        raise ValueError(f"ไม่รู้จักเครื่องมือ {tool}")
    out = fn(args)
    if isinstance(out, tuple):
        return out
    return out, None


def _assist_pending_summary(tool: str, args: dict) -> str:
    if tool == "clear_history":
        return "จะลบทุกภาพและประวัติทั้งหมด — ย้อนกลับไม่ได้"
    if tool == "delete_history":
        target = args.get("job_id") or "ภาพล่าสุด"
        return f"จะลบ {target} พร้อมไฟล์บนดิสก์ — ย้อนกลับไม่ได้"
    if tool == "delete_lora":
        return f"จะลบ LoRA “{args.get('id') or '—'}” ออกจากเครื่อง — ย้อนกลับไม่ได้"
    if tool == "start_training":
        return (
            f"จะเทรน LoRA “{args.get('name')}” · dataset "
            f"{args.get('dataset_id') or 'ที่เลือกอยู่'} · {args.get('steps') or 1500} steps"
            " — ใช้เวลานานและการ์ดจะร้อน"
        )
    return ast.TOOL_INDEX.get(tool, {}).get("title", tool)


def _assist_action(tool: str, result: dict, undo_id=None) -> dict:
    spec = ast.TOOL_INDEX.get(tool) or {}
    return {
        "id": uuid.uuid4().hex[:12],
        "tool": tool,
        "tier": spec.get("tier", "read"),
        "title": spec.get("title", tool),
        "status": "done",
        "result": result,
        "undo_id": undo_id,
    }


class AssistantChatRequest(BaseModel):
    message: str = Field(default="", max_length=2000)
    dataset_id: Optional[str] = None
    # UI language — the LLM brain writes its clarify/reply text to match
    lang: str = Field(default="", max_length=8)


class AssistantIdRequest(BaseModel):
    id: str = ""


class VisionStartRequest(BaseModel):
    dataset_id: Optional[str] = None
    files: Optional[list[str]] = None


class VisionApplyRequest(BaseModel):
    files: Optional[list[str]] = None


class AssistantModelRequest(BaseModel):
    model: str = ""


def _assist_brain_mode() -> str:
    """Pick the brain for one message without ever running two pipelines.

    The 10 GB card cannot hold SANA (≈9 GB when warm) and Qwen 4B q8
    (≈4.7 GB) at the same time. Rules:
      • GPU busy (queued/running job or model loading) → rule-based brain
        (deterministic, zero VRAM, no risk of OOM mid-job)
      • SANA loaded but idle → unload it, then use the LLM
      • otherwise → LLM as configured
    """
    mode = astllm.brain_mode()
    if mode == "rules":
        return "rules"
    with STATE_LOCK:
        busy = ACTIVE_JOB is not None
        active = sum(1 for j in JOBS.values() if j.get("status") in ("queued", "running"))
    if busy or active or manager.state in ("loading", "preparing"):
        return "rules"
    if manager.pipe is not None:
        try:
            manager.unload()
        except Exception:  # noqa: BLE001 — fall through to the LLM anyway
            pass
    return mode


@app.get("/api/assistant/tools")
def api_assistant_tools(lang: str = ""):
    return {
        "tools": ast.TOOLS,
        "suggestions": ast.suggestions_for(lang),
        "tiers": list(ast.TIERS),
        "brain": astllm.status(),
        "vision": astvis.status(),
    }


@app.get("/api/assistant/state")
def api_assistant_state():
    return {
        "pending": ASSIST_PENDING.count(),
        "undo": ASSIST_UNDO.count(),
        "brain": astllm.status(),
        "vision": astvis.status(),
    }


@app.get("/api/assistant/models")
def api_assistant_models():
    astllm.invalidate_models()
    return {
        "models": astllm.model_info(),
        "current": astllm.model_name(),
        "default": astllm.default_model(),
        "brain": astllm.status(),
        "vision": astvis.status(),
    }


@app.post("/api/assistant/model")
def api_assistant_model(req: AssistantModelRequest):
    value = (req.model or "").strip()
    if (os.environ.get("IMGS_ASSIST_MODEL") or "").strip():
        raise HTTPException(
            409,
            "โมเดลถูกล็อกด้วย env IMGS_ASSIST_MODEL — แก้ที่ environment ก่อน",
        )
    if value:
        names = astllm.installed_models()
        if not names:
            raise HTTPException(400, "ต่อ Ollama ไม่ได้ — ตรวจว่า Ollama ทำงานอยู่")
        if not any(name == value or name.startswith(value + ":") for name in names):
            raise HTTPException(400, f"ไม่พบโมเดล “{value}” ใน Ollama")
        if astvis.RUNNER.running():
            raise HTTPException(409, "กำลังอ่านรูปอยู่ — รอให้เสร็จก่อนเปลี่ยนโมเดล")
    try:
        astllm.set_model(value)
    except astllm.LLMError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        "model": astllm.model_name(),
        "brain": astllm.status(),
        "vision": astvis.status(),
    }


@app.post("/api/assistant/chat")
def api_assistant_chat(req: AssistantChatRequest):
    intent = ast.parse_intent(
        req.message, brain=_assist_brain_mode(), lang=(req.lang or "").strip()
    )
    tool = intent.get("tool")
    args = dict(intent.get("args") or {})
    if req.dataset_id:
        args["dataset_id"] = req.dataset_id
    base = {
        "ok": True,
        "actions": [],
        "pending": [],
        "suggestions": ast.suggestions_for(req.lang),
        "brain": intent.get("brain") or "rules",
    }
    if intent.get("clarify"):
        return {**base, "reply": intent["clarify"]}
    if not tool:
        return {**base, "reply": ast.help_text(intent.get("brain"), req.lang)}
    spec = ast.TOOL_INDEX.get(tool)
    if spec is None:
        return {**base, "reply": f"ไม่รู้จักคำสั่ง “{tool}”"}

    if tool == "help":
        return {**base, "reply": ast.help_text(intent.get("brain"), req.lang)}

    if spec["tier"] == "destructive":
        summary = _assist_pending_summary(tool, args)
        pid = ASSIST_PENDING.add({"tool": tool, "args": args, "created": now_iso()})
        return {
            **base,
            "reply": f"ต้องยืนยันก่อน: {spec['title']}\n{summary}",
            "pending": [
                {
                    "id": pid,
                    "tool": tool,
                    "tier": "destructive",
                    "title": spec["title"],
                    "summary": summary,
                    "args": args,
                    "confirm_label": "ยืนยัน",
                }
            ],
        }

    try:
        result, undo = _assist_execute(tool, args, lang=req.lang, brain=intent.get("brain"))
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not a crash
        return {**base, "ok": False, "reply": f"ทำไม่ได้: {_assist_error(exc)}"}

    undo_id = None
    if undo is not None:
        undo_id = ASSIST_UNDO.add({"tool": tool, "label": spec["title"], "undo": undo})
    return {
        **base,
        "reply": ast.summarize_result(tool, result, intent.get("brain")),
        "actions": [_assist_action(tool, result, undo_id)],
    }


@app.post("/api/assistant/confirm")
def api_assistant_confirm(req: AssistantIdRequest):
    payload = ASSIST_PENDING.get(req.id)
    if payload is None:
        raise HTTPException(404, "การ์ดนี้หมดอายุแล้ว — ส่งคำสั่งใหม่")
    tool = payload.get("tool")
    args = payload.get("args") or {}
    try:
        result, undo = _assist_execute(tool, args)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, _assist_error(exc)) from exc
    ASSIST_PENDING.pop(req.id)
    spec = ast.TOOL_INDEX.get(tool) or {}
    undo_id = None
    if undo is not None:
        undo_id = ASSIST_UNDO.add({"tool": tool, "label": spec.get("title", tool), "undo": undo})
    return {
        "ok": True,
        "reply": ast.summarize_result(tool, result),
        "action": _assist_action(tool, result, undo_id),
    }


@app.post("/api/assistant/cancel")
def api_assistant_cancel(req: AssistantIdRequest):
    removed = ASSIST_PENDING.pop(req.id)
    return {"ok": True, "cancelled": removed is not None}


@app.post("/api/assistant/undo")
def api_assistant_undo(req: AssistantIdRequest):
    payload = ASSIST_UNDO.pop(req.id)
    if payload is None:
        raise HTTPException(404, "Undo หมดอายุแล้ว")
    undo = payload.get("undo")
    try:
        if undo is not None:
            undo()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, _assist_error(exc)) from exc
    return {"ok": True, "reply": f"ย้อนกลับแล้ว: {payload.get('label')}"}


# ── assistant vision (P3: local VL reads dataset images) ────────────────────
@app.post("/api/assistant/vision")
def api_assistant_vision_start(req: VisionStartRequest):
    try:
        result, _ = _assist_read_images(
            {"dataset_id": req.dataset_id, "files": req.files}
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, _assist_error(exc)) from exc
    return {"ok": True, **result}


@app.get("/api/assistant/vision/{job_id}")
def api_assistant_vision_get(job_id: str):
    job = astvis.RUNNER.get(job_id)
    if job is None:
        raise HTTPException(404, "ไม่พบงานอ่านรูปนี้ (อาจหมดอายุแล้ว)")
    return {"ok": True, "job": job}


@app.post("/api/assistant/vision/{job_id}/cancel")
def api_assistant_vision_cancel(job_id: str):
    cancelled = astvis.RUNNER.cancel(job_id)
    job = astvis.RUNNER.get(job_id)
    if job is None:
        raise HTTPException(404, "ไม่พบงานอ่านรูปนี้ (อาจหมดอายุแล้ว)")
    return {"ok": True, "cancelled": cancelled, "job": job}


@app.post("/api/assistant/vision/{job_id}/apply")
def api_assistant_vision_apply(job_id: str, req: VisionApplyRequest):
    job = astvis.RUNNER.get(job_id)
    if job is None:
        raise HTTPException(404, "ไม่พบงานอ่านรูปนี้ (อาจหมดอายุแล้ว)")
    if job.get("status") in ("running", "cancelling"):
        raise HTTPException(409, "ยังอ่านไม่เสร็จ — รอให้จบก่อนแล้วค่อยใช้ tag")
    selected = {str(x) for x in (req.files or []) if str(x).strip()}
    updates, previous, applied = [], [], []
    for row in job.get("results") or []:
        if not row.get("angle_added"):
            continue
        if selected and row.get("file") not in selected:
            continue
        old = str(row.get("old_caption") or "")
        new = str(row.get("suggested_caption") or "")
        if not new or new == old:
            continue
        previous.append({"file": row.get("file"), "caption": old})
        updates.append({"file": row.get("file"), "caption": new})
        applied.append(f"{row.get('file')} → {row.get('angle_added')}")
    if not updates:
        return {"ok": True, "updated": 0, "files": [], "undo_id": None}
    ds_id = job.get("dataset_id")
    try:
        strn.set_captions(ds_id, updates)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, _assist_error(exc)) from exc

    def undo():
        strn.set_captions(ds_id, previous)

    undo_id = ASSIST_UNDO.add(
        {"tool": "read_images", "label": "ใช้ tag มุมจาก vision", "undo": undo}
    )
    return {
        "ok": True,
        "updated": len(updates),
        "files": applied,
        "undo_id": undo_id,
    }


# ── GPU power modes ─────────────────────────────────────────────────────────
class PowerRequest(BaseModel):
    watts: Optional[int] = Field(default=None, ge=1, le=2000)
    mode: Optional[str] = None


def _run_power(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a power helper / sudo command (kept separate so tests can stub it)."""
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _power_readings() -> Optional[dict]:
    """Read power/temperature fields from nvidia-smi (no root required)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.limit,power.draw,temperature.gpu,"
                "power.min_limit,power.max_limit,power.default_limit,"
                "persistence_mode",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
    except Exception:  # noqa: BLE001 — no card / no driver
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None

    def num(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "limit": num(parts[0]),
        "draw": num(parts[1]),
        "temp": num(parts[2]),
        "min": num(parts[3]),
        "max": num(parts[4]),
        "default": num(parts[5]),
        "persistence": parts[6] or "unknown",
    }


def _power_modes(readings: Optional[dict]) -> list[dict]:
    """Resolve the three presets against the current card's limits."""
    base = (readings or {}).get("default") or (readings or {}).get("max") or 250.0
    lo = int((readings or {}).get("min") or POWER_MIN_FALLBACK)
    hi = int((readings or {}).get("max") or POWER_MAX_FALLBACK)
    modes = []
    for mode_id, ratio in POWER_PRESETS:
        watts = max(lo, min(hi, int(round(float(base) * ratio))))
        modes.append({"id": mode_id, "watts": watts})
    return modes


def _power_mode_for(limit: Optional[float], modes: list[dict]) -> Optional[str]:
    if limit is None:
        return None
    best = min(modes, key=lambda m: abs(m["watts"] - float(limit)))
    return best["id"] if abs(best["watts"] - float(limit)) <= 5 else None


def _power_helper_installed() -> bool:
    return os.path.isfile(POWER_HELPER) and os.access(POWER_HELPER, os.X_OK)


def _power_json(text: str) -> Optional[dict]:
    """Parse one helper JSON line, ignoring anything non-JSON."""
    try:
        data = json.loads((text or "").strip())
    except Exception:  # noqa: BLE001 — plain text / empty output
        return None
    return data if isinstance(data, dict) else None


def _power_configured(data: Optional[dict]) -> Optional[int]:
    if not data or data.get("configured") is None:
        return None
    try:
        return int(round(float(data["configured"])))
    except (TypeError, ValueError):
        return None


def _power_helper_ready(force: bool = False) -> bool:
    """True when the helper exists *and* the sudoers rule works.

    Cached briefly: /api/power is polled by the settings card, and running
    `sudo -n` every few seconds is unnecessary.
    """
    global _POWER_DESIRED
    now = time.time()
    if not force and now - _POWER_READY_CACHE["at"] < 15.0:
        return _POWER_READY_CACHE["value"]
    ok = False
    if _power_helper_installed():
        try:
            proc = _run_power(
                ["sudo", "-n", POWER_HELPER, "status"], timeout=5
            )
            ok = proc.returncode == 0
            if ok:
                configured = _power_configured(_power_json(proc.stdout))
                if configured is not None:
                    _POWER_DESIRED = configured
        except Exception:  # noqa: BLE001 — sudo missing, timeout, …
            ok = False
    _POWER_READY_CACHE.update(at=now, value=ok)
    return ok


def _power_status_payload() -> dict:
    readings = _power_readings()
    modes = _power_modes(readings)
    available = readings is not None
    ready = _power_helper_ready() if available and POWER_SUPPORTED else False
    limit = readings["limit"] if readings else None
    saved = _POWER_DESIRED
    return {
        "ok": True,
        "available": available,
        "supported": POWER_SUPPORTED,
        "installed": ready,
        "setup_required": available and POWER_SUPPORTED and not ready,
        "modes": modes,
        # mode = what the card is doing right now, saved_mode = the app's choice.
        "mode": _power_mode_for(limit, modes),
        "saved": saved,
        "saved_mode": _power_mode_for(saved, modes) if saved is not None else None,
        "limit": limit,
        "draw": readings["draw"] if readings else None,
        "temp": readings["temp"] if readings else None,
        "min": readings["min"] if readings else None,
        "max": readings["max"] if readings else None,
        "default": readings["default"] if readings else None,
        "persistence": readings["persistence"] if readings else None,
    }


def _power_error(proc: subprocess.CompletedProcess) -> str:
    for stream in (proc.stdout, proc.stderr):
        try:
            data = json.loads((stream or "").strip())
        except Exception:  # noqa: BLE001 — plain text, not JSON
            continue
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    text = (proc.stderr or proc.stdout or "").strip()
    return text.splitlines()[-1] if text else ""


def _power_apply_saved() -> Optional[dict]:
    """Apply the remembered wattage (called once when the engine starts)."""
    global _POWER_DESIRED, _POWER_ENFORCE
    if not POWER_SUPPORTED or not _power_helper_installed():
        return None
    try:
        with POWER_LOCK:
            proc = _run_power(["sudo", "-n", POWER_HELPER, "apply"], timeout=90)
    except Exception:  # noqa: BLE001 — sudo missing, timeout, …
        return None
    data = _power_json(proc.stdout)
    configured = _power_configured(data)
    if configured is not None:
        _POWER_DESIRED = configured
        # Even if the driver refused the limit right now, remember the choice
        # so the watchdog can retry once the GPU is free.
        _POWER_ENFORCE = True
        _POWER_READY_CACHE.update(at=time.time(), value=True)
    return data


def _power_reset_now() -> Optional[dict]:
    """Restore the card default without changing the saved choice."""
    global _POWER_DESIRED, _POWER_ENFORCE
    if not POWER_SUPPORTED or not _power_helper_installed():
        return None
    with POWER_LOCK:
        _POWER_ENFORCE = False
        try:
            proc = _run_power(["sudo", "-n", POWER_HELPER, "reset"], timeout=20)
        except Exception:  # noqa: BLE001
            return None
    data = _power_json(proc.stdout)
    configured = _power_configured(data)
    if configured is not None:
        _POWER_DESIRED = configured
    if data is not None and data.get("ok"):
        _POWER_READY_CACHE.update(at=time.time(), value=True)
    return data


def _power_watchdog_tick() -> bool:
    """Re-apply the saved mode when the driver has drifted away from it.

    Returns True when a re-apply was attempted.  Cheap when nothing is wrong:
    only reads nvidia-smi, never sudo.
    """
    global _POWER_DESIRED
    if not _POWER_ENFORCE or _POWER_DESIRED is None:
        return False
    readings = _power_readings()
    limit = readings.get("limit") if readings else None
    if limit is None:
        return False
    if abs(float(limit) - float(_POWER_DESIRED)) <= 2.0:
        return False
    with POWER_LOCK:
        if not _POWER_ENFORCE or _POWER_DESIRED is None:
            return False
        try:
            proc = _run_power(["sudo", "-n", POWER_HELPER, "apply"], timeout=90)
        except Exception:  # noqa: BLE001
            return True
        data = _power_json(proc.stdout)
        configured = _power_configured(data)
        if configured is not None:
            _POWER_DESIRED = configured
        return True


def _power_watch_loop() -> None:
    while not _POWER_WATCH_STOP.wait(POWER_WATCH_INTERVAL):
        try:
            _power_watchdog_tick()
        except Exception:  # noqa: BLE001 — never kill the watchdog thread
            pass


def _power_startup() -> None:
    """Apply the saved mode and start the drift watchdog."""
    global _POWER_WATCH_THREAD
    if os.environ.get("IMGS_POWER_AUTOSTART", "1") == "0":
        return
    _power_apply_saved()
    if _POWER_WATCH_THREAD is None or not _POWER_WATCH_THREAD.is_alive():
        _POWER_WATCH_STOP.clear()
        _POWER_WATCH_THREAD = threading.Thread(
            target=_power_watch_loop, name="power-watchdog", daemon=True
        )
        _POWER_WATCH_THREAD.start()


def _power_shutdown() -> None:
    """Stop enforcing and restore the card default before the process exits."""
    _POWER_WATCH_STOP.set()
    if os.environ.get("IMGS_POWER_AUTOSTART", "1") == "0":
        return
    _power_reset_now()


@app.get("/api/power")
def api_power():
    return _power_status_payload()


@app.post("/api/power")
def api_power_set(req: PowerRequest):
    global _POWER_DESIRED, _POWER_ENFORCE
    readings = _power_readings()
    if readings is None:
        raise HTTPException(503, "ไม่พบการ์ด NVIDIA")
    if not POWER_SUPPORTED:
        raise HTTPException(400, "โหมดพลังงานใช้ได้เฉพาะ Linux")
    if not _power_helper_ready(force=True):
        raise HTTPException(
            409, "ยังไม่ได้เปิดใช้งานโหมดพลังงาน — กดปุ่มเปิดใช้งานก่อน"
        )
    modes = _power_modes(readings)
    if req.mode:
        match = next((m for m in modes if m["id"] == req.mode), None)
        if match is None:
            raise HTTPException(400, f"ไม่รู้จักโหมด: {req.mode}")
        watts = match["watts"]
    elif req.watts is not None:
        watts = int(req.watts)
    else:
        raise HTTPException(400, "ต้องระบุ watts หรือ mode")
    lo = int(readings["min"] or POWER_MIN_FALLBACK)
    hi = int(readings["max"] or POWER_MAX_FALLBACK)
    if not lo <= watts <= hi:
        raise HTTPException(400, f"power limit ต้องอยู่ระหว่าง {lo}–{hi} W")

    with POWER_LOCK:
        try:
            proc = _run_power(["sudo", "-n", POWER_HELPER, "set", str(watts)])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"เรียกตัวช่วยไม่สำเร็จ: {exc}") from exc
        if proc.returncode != 0:
            detail = _power_error(proc)
            lowered = detail.lower()
            if "password is required" in lowered or "not allowed" in lowered:
                _POWER_READY_CACHE.update(at=0.0, value=False)
                raise HTTPException(
                    403, "sudoers ยังไม่ได้ตั้งค่า — กดปุ่มเปิดใช้งานโหมดพลังงานอีกครั้ง"
                )
            # The helper saves the choice before applying it.  If only the
            # driver refused for a moment (GPU busy), keep the choice and let
            # the watchdog retry instead of rolling back to the old mode.
            configured = _power_configured(_power_json(proc.stdout))
            if configured is not None:
                _POWER_DESIRED = configured
                _POWER_ENFORCE = True
                _POWER_READY_CACHE.update(at=time.time(), value=True)
                return _power_status_payload()
            raise HTTPException(500, detail or "ตั้งค่า power limit ไม่สำเร็จ")
        _POWER_DESIRED = watts
        _POWER_ENFORCE = True

    _POWER_READY_CACHE.update(at=time.time(), value=True)
    return _power_status_payload()


@app.post("/api/power/reset")
def api_power_reset():
    """Restore the card default; the saved choice stays for the next launch."""
    global _POWER_DESIRED, _POWER_ENFORCE
    if not POWER_SUPPORTED:
        raise HTTPException(400, "โหมดพลังงานใช้ได้เฉพาะ Linux")
    if not _power_helper_ready(force=True):
        raise HTTPException(
            409, "ยังไม่ได้เปิดใช้งานโหมดพลังงาน — กดปุ่มเปิดใช้งานก่อน"
        )
    with POWER_LOCK:
        _POWER_ENFORCE = False
        try:
            proc = _run_power(["sudo", "-n", POWER_HELPER, "reset"], timeout=20)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"เรียกตัวช่วยไม่สำเร็จ: {exc}") from exc
    data = _power_json(proc.stdout)
    configured = _power_configured(data)
    if configured is not None:
        _POWER_DESIRED = configured
    if proc.returncode != 0 or (data is not None and data.get("ok") is False):
        detail = _power_error(proc)
        raise HTTPException(500, detail or "คืนค่า power limit ไม่สำเร็จ")

    _POWER_READY_CACHE.update(at=time.time(), value=True)
    return _power_status_payload()


@app.post("/api/power/setup")
def api_power_setup():
    if not POWER_SUPPORTED:
        raise HTTPException(400, "โหมดพลังงานใช้ได้เฉพาะ Linux")
    if not POWER_SETUP.is_file():
        raise HTTPException(500, "ไม่พบไฟล์ติดตั้ง (host/pixelsmith-power-setup)")
    pkexec = shutil.which("pkexec")
    if not pkexec:
        raise HTTPException(
            400, "ไม่พบ pkexec — ติดตั้ง polkit หรือรันสคริปต์ติดตั้งด้วย sudo"
        )
    user = getpass.getuser()
    if not user or user == "root":
        raise HTTPException(400, "ไม่สามารถระบุผู้ใช้เดสก์ท็อปสำหรับตั้งค่า sudoers")

    with POWER_LOCK:
        try:
            proc = _run_power(
                [pkexec, "/bin/bash", str(POWER_SETUP), user], timeout=180
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                504, "ติดตั้งไม่เสร็จภายในเวลา — ลองใหม่แล้วป้อนรหัสให้เร็วขึ้น"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"เปิดตัวติดตั้งไม่สำเร็จ: {exc}") from exc
    if proc.returncode in (126, 127):
        raise HTTPException(403, "การยืนยันถูกยกเลิกหรือไม่ได้รับอนุญาต")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(500, detail or "ติดตั้งไม่สำเร็จ")

    _POWER_READY_CACHE.update(at=0.0, value=True)
    return _power_status_payload()


# ── backup / restore ─────────────────────────────────────────────────────────
# One ZIP with the user's own data: history, images, thumbnails, presets and
# training datasets. Models and LoRAs are NOT included (they are large and live
# in models/); the UI says so. Restore is additive: it never deletes anything.
BACKUP_FORMAT = 1
BACKUP_TOP_FILES = (
    "history.json",
    "prompt_presets.json",
    "train_datasets.json",
    "prompt-collection.txt",
)
BACKUP_TREES = ("images", "thumbs", "train")
MAX_BACKUP_UPLOAD = 4 * 1024 ** 3   # 4 GB compressed
MAX_BACKUP_FILES = 20000
MAX_BACKUP_UNCOMPRESSED = 12 * 1024 ** 3  # 12 GB after extraction


def _backup_member_ok(name: str) -> bool:
    """Reject absolute paths, traversal, and anything outside the allowlist."""
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        return False
    parts = Path(name).parts
    if any(p in ("", ".", "..") for p in parts):
        return False
    if len(parts) == 1:
        return name in BACKUP_TOP_FILES or name == "manifest.json"
    return parts[0] in BACKUP_TREES


@app.post("/api/backup/export")
def api_backup_export():
    backup_dir = DATA / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"pixelsmith-backup-{stamp}.zip"
    manifest = {
        "format": BACKUP_FORMAT,
        "app_version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "history, images, thumbs, prompt presets and training datasets",
    }
    files = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name in BACKUP_TOP_FILES:
            src = DATA / name
            if src.is_file():
                zf.write(src, name)
                files += 1
        for tree in BACKUP_TREES:
            base = DATA / tree
            if not base.is_dir():
                continue
            for src in sorted(base.rglob("*")):
                if src.is_file():
                    zf.write(src, f"{tree}/{src.relative_to(base).as_posix()}")
                    files += 1
    return {
        "ok": True,
        "path": str(target),
        "size": target.stat().st_size,
        "files": files,
        "created_at": manifest["created_at"],
    }


def _merge_json_list(path: Path, incoming, key: str) -> int:
    """Add incoming records whose key is not already present. Returns added."""
    if not isinstance(incoming, list):
        return 0
    try:
        current = json.loads(path.read_text("utf-8")) if path.is_file() else []
    except Exception:  # noqa: BLE001
        current = []
    if not isinstance(current, list):
        current = []
    seen = {row.get(key) for row in current if isinstance(row, dict)}
    added = [row for row in incoming if isinstance(row, dict) and row.get(key) not in seen]
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(current + added, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    return len(added)


@app.post("/api/backup/restore")
async def api_backup_restore(file: UploadFile = File(...)):
    if ordered_active_jobs():
        raise HTTPException(409, "มีงานกำลังรันหรือรอคิวอยู่ — รอให้เสร็จก่อนกู้คืน")
    backup_dir = DATA / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tmp = backup_dir / f".restore-{uuid.uuid4().hex}.zip"
    size = 0
    try:
        with tmp.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BACKUP_UPLOAD:
                    raise HTTPException(413, "ไฟล์สำรองใหญ่เกินไป (สูงสุด 4 GB)")
                fh.write(chunk)
        try:
            with zipfile.ZipFile(tmp) as zf:
                infos = zf.infolist()
                if len(infos) > MAX_BACKUP_FILES:
                    raise HTTPException(400, "ไฟล์สำรองมีไฟล์มากเกินไป")
                if sum(i.file_size for i in infos) > MAX_BACKUP_UNCOMPRESSED:
                    raise HTTPException(400, "ข้อมูลในไฟล์สำรองใหญ่เกินไป")
                names = [i.filename for i in infos]
                if "manifest.json" not in names:
                    raise HTTPException(400, "ไม่ใช่ไฟล์สำรองของ PixelSmith (ไม่มี manifest)")
                try:
                    manifest = json.loads(zf.read("manifest.json"))
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(400, f"manifest เสียหาย: {exc}") from exc
                if int(manifest.get("format") or 0) != BACKUP_FORMAT:
                    raise HTTPException(400, "ไฟล์สำรองเวอร์ชันไม่รองรับ")
                for name in names:
                    if name.endswith("/"):
                        continue
                    if not _backup_member_ok(name):
                        raise HTTPException(400, f"path ในไฟล์สำรองไม่ปลอดภัย: {name}")

                counts = {"images": 0, "thumbs": 0, "train": 0}
                payloads: dict[str, bytes] = {}
                for info in infos:
                    name = info.filename
                    if name.endswith("/") or name == "manifest.json":
                        continue
                    if "/" not in name:
                        payloads[name] = zf.read(name)
                        continue
                    tree, _, rel = name.partition("/")
                    dest = DATA / tree / rel
                    if dest.exists():
                        continue  # additive restore: keep local files
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    counts[tree] += 1
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "ไฟล์ ZIP เสียหายหรือไม่สมบูรณ์") from exc

        history_added = _merge_json_list(HISTORY_FILE, _json_from(payloads.get("history.json")), "id")
        if history_added:
            with STATE_LOCK:
                known = {row.get("id") for row in HISTORY if isinstance(row, dict)}
                for row in _json_from(payloads.get("history.json")) or []:
                    if isinstance(row, dict) and row.get("id") not in known:
                        HISTORY.append(row)
                        known.add(row.get("id"))
            persist_history()
        presets_added = _merge_json_list(
            ppres.PRESETS_FILE, _json_from(payloads.get("prompt_presets.json")), "name"
        )
        datasets_added = _merge_json_list(
            strn.REGISTRY_FILE, _json_from(payloads.get("train_datasets.json")), "id"
        )
        collection = payloads.get("prompt-collection.txt")
        collection_added = 0
        if collection:
            dest = DATA / "prompt-collection.txt"
            if not dest.exists():
                dest.write_bytes(collection)
                collection_added = 1
        return {
            "ok": True,
            "restored": {
                **counts,
                "history": history_added,
                "presets": presets_added,
                "datasets": datasets_added,
                "prompt_collection": collection_added,
            },
            "manifest": manifest,
        }
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _json_from(raw: bytes | None):
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ── static pages ─────────────────────────────────────────────────────────────
@app.get("/")
def page_index():
    return FileResponse(str(ROOT / "host" / "index.html"), media_type="text/html")


@app.get("/setup")
def page_setup():
    return FileResponse(str(ROOT / "host" / "setup.html"), media_type="text/html")


@app.get("/favicon.ico")
def favicon():
    icon = ROOT / "electron" / "icon.png"
    if icon.is_file():
        return FileResponse(str(icon), media_type="image/png")
    raise HTTPException(404)


app.mount("/studio", StaticFiles(directory=str(ROOT / "studio"), html=True), name="studio")
app.mount("/host", StaticFiles(directory=str(ROOT / "host"), html=True), name="host")
app.mount("/files", StaticFiles(directory=str(DATA)), name="files")


if __name__ == "__main__":
    import uvicorn

    print(f"[image-studio] http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("IMGS_LOG", "info"))
