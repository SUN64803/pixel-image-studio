"""Local vision reading for the assistant (P3) — Ollama + qwen3.5 vision.

Reads dataset images **on this machine** and suggests the standard angle tag
(+ a short English caption and a multi-character warning). Nothing is uploaded
anywhere, which is required for this dataset.

A single background thread analyses images one at a time. The caller polls the
job; applying suggestions goes through `strn.set_captions` (atomic) and is
registered in the assistant Undo store by `server.py`.
"""
from __future__ import annotations

import base64
import io
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import assistant as ast
import assistant_llm as llm

MAX_SIDE = 768
MAX_IMAGES = 40
KEEP_JOBS = 5

ANGLE_CHOICES = ast.ANGLE_TAGS + ["none"]

PROMPT = """You are labelling images for a LoRA training dataset. Reply with JSON only:
{"angle": "front view|side view|three-quarter view|back view|close-up|high angle|low angle|none",
 "multi": true|false,
 "caption": "one short English sentence describing pose, framing and setting; do not guess a name"}
Rules:
- angle = camera angle of the main subject; use "none" if unclear.
- multi = true only if another person/character is clearly visible.
- caption: max 30 words, comma separated tags, no identity words."""


class VisionError(RuntimeError):
    """Image could not be read or the model reply was unusable."""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def encode_image(path: Path) -> str:
    """Downscale to 768 px and JPEG-encode — big originals are slow to upload."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a hard dep
        raise VisionError("ไม่มี Pillow ในเครื่อง") from exc
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
    except Exception as exc:  # noqa: BLE001
        raise VisionError(f"เปิดรูปไม่ได้: {exc}") from exc
    return base64.b64encode(buf.getvalue()).decode("ascii")


def capabilities() -> list[str]:
    """Vision capability of the currently selected assistant model."""
    return llm.model_capabilities()


def available() -> bool:
    return "vision" in capabilities()


def status() -> dict:
    caps = capabilities()
    ok = "vision" in caps
    return {
        "model": llm.model_name(),
        "available": ok,
        "label": f"{llm.model_name()} · vision" if ok else "ไม่มีโมเดล vision",
    }


def analyze(path: Path) -> dict:
    """One image → {angle, multi, caption}. Raises VisionError."""
    if not available():
        raise VisionError(f"โมเดล {llm.model_name()} ไม่รองรับ vision")
    image = encode_image(path)
    data = llm._request(
        "/api/chat",
        {
            "model": llm.model_name(),
            "messages": [{"role": "user", "content": PROMPT, "images": [image]}],
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 180},
        },
        timeout=180.0,
    )
    raw = ((data.get("message") or {}).get("content")) or ""
    try:
        parsed = llm._extract_json(raw)
    except llm.LLMError as exc:
        raise VisionError(str(exc)) from exc
    angle = str(parsed.get("angle") or "").strip().lower()
    if angle not in ANGLE_CHOICES:
        angle = "none"
    caption = " ".join(str(parsed.get("caption") or "").split())[:300]
    multi = parsed.get("multi")
    if isinstance(multi, str):
        multi = multi.strip().lower() in ("1", "true", "yes", "on")
    return {"angle": angle, "multi": bool(multi), "caption": caption}


def suggest(image: dict, reading: dict) -> dict:
    """Turn one vision reading into an apply-able suggestion."""
    old = str(image.get("caption") or "")
    angle = reading.get("angle") or "none"
    add_angle = angle != "none" and not ast.has_angle_tag(old)
    return {
        "file": image.get("file"),
        "name": image.get("name") or image.get("file"),
        "old_caption": old,
        "multi_old": bool(image.get("multi")),
        "angle": angle,
        "angle_added": angle if add_angle else None,
        "multi": bool(reading.get("multi")),
        "multi_changed": bool(reading.get("multi")) != bool(image.get("multi")),
        "caption": reading.get("caption") or "",
        "suggested_caption": ast.with_angle_tags(old, [angle]) if add_angle else old,
    }


class VisionRunner:
    """Bounded store of vision jobs + one worker thread at a time."""

    def __init__(self, keep: int = KEEP_JOBS):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._keep = max(1, int(keep))
        self._thread: threading.Thread | None = None
        self._cancel: threading.Event | None = None

    # ── store ───────────────────────────────────────────────────────────────
    def _trim(self) -> None:
        while len(self._order) > self._keep:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def get(self, job_id: str):
        with self._lock:
            job = self._jobs.get(str(job_id or ""))
            return dict(job) if job else None

    def latest(self):
        with self._lock:
            if not self._order:
                return None
            return dict(self._jobs.get(self._order[-1]) or {})

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self, dataset: dict, images: list[dict], model: str) -> dict:
        with self._lock:
            if self.running():
                raise VisionError("กำลังอ่านรูปชุดก่อนอยู่ — รอให้เสร็จก่อน")
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "status": "running",
                "dataset_id": dataset.get("id"),
                "dataset_name": dataset.get("name"),
                "model": model,
                "total": len(images),
                "done": 0,
                "current": None,
                "results": [],
                "error": None,
                "created": now_iso(),
                "finished": None,
            }
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._trim()
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(job_id, dataset, list(images), self._cancel),
                daemon=True, name=f"vision-{job_id}",
            )
            self._thread.start()
            return dict(job)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(str(job_id or ""))
            if job is None or job["status"] != "running":
                return False
            if self._cancel is not None:
                self._cancel.set()
            job["status"] = "cancelling"
            return True

    def _run(self, job_id: str, dataset: dict, images: list[dict], cancel) -> None:
        base = Path(str(dataset.get("dir") or "")) if dataset.get("dir") else None
        for index, image in enumerate(images):
            if cancel.is_set():
                break
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["current"] = image.get("file")
            path = None
            if base is not None:
                path = base / str(image.get("file") or "")
            try:
                if path is None or not path.is_file():
                    raise VisionError("ไม่พบไฟล์รูป")
                reading = analyze(path)
                result = suggest(image, reading)
            except Exception as exc:  # noqa: BLE001 — one bad image must not stop the run
                result = {
                    "file": image.get("file"),
                    "name": image.get("name") or image.get("file"),
                    "old_caption": image.get("caption") or "",
                    "multi_old": bool(image.get("multi")),
                    "angle": "none",
                    "angle_added": None,
                    "multi": bool(image.get("multi")),
                    "multi_changed": False,
                    "caption": "",
                    "suggested_caption": image.get("caption") or "",
                    "error": str(exc),
                }
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["results"].append(result)
                job["done"] = index + 1
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["current"] = None
            job["finished"] = now_iso()
            job["status"] = "cancelled" if cancel.is_set() else "done"


RUNNER = VisionRunner()
