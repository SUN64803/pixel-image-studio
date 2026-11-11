"""Local LLM brain for the assistant (P2) — talks to Ollama on loopback.

Default model: `qwen3.5:4b-q8_0` (already installed on this machine).
Everything stays local: no cloud calls, no API keys.

The LLM is only used to *choose a tool*; execution still goes through the
same validated executors in `server.py`, so the three permission tiers are
unchanged and a bad model reply can never bypass confirmation.

If Ollama is missing / the model is not installed / a request fails, the
caller falls back to the rule-based parser in `assistant.py`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3.5:4b-q8_0"
DEFAULT_URL = "http://127.0.0.1:11434"

BRAINS = ("auto", "llm", "rules")

_TAGS_TTL = 30.0
_CAPS_TTL = 60.0
_status_lock = threading.Lock()
_status_cache: dict = {"stamp": 0.0, "models": [], "error": None}
_caps_lock = threading.Lock()
_caps_cache: dict[str, tuple[float, list[str]]] = {}



class LLMError(RuntimeError):
    """Ollama unreachable, model missing, or reply was not usable JSON."""


def _config_path() -> Path:
    """Where the UI model choice is persisted (env override wins over this)."""
    data = os.environ.get("IMGS_DATA")
    base = Path(data) if data else Path(__file__).resolve().parent.parent / "data"
    return base / "assistant_config.json"


def _saved_model() -> str:
    try:
        data = json.loads(_config_path().read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("model") or "").strip() if isinstance(data, dict) else ""


def default_model() -> str:
    return DEFAULT_MODEL


def model_name() -> str:
    """Env override → saved UI choice → built-in default."""
    env = (os.environ.get("IMGS_ASSIST_MODEL") or "").strip()
    if env:
        return env
    return _saved_model() or DEFAULT_MODEL


def _invalidate_caps() -> None:
    with _caps_lock:
        _caps_cache.clear()


def set_model(name: str) -> str:
    """Persist the assistant model choice. Empty string resets to default."""
    value = str(name or "").strip()
    path = _config_path()
    if not value:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise LLMError(f"ลบการตั้งค่าโมเดลไม่ได้: {exc}") from exc
        _invalidate_caps()
        return model_name()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"model": value}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:
        raise LLMError(f"บันทึกการตั้งค่าโมเดลไม่ได้: {exc}") from exc
    _invalidate_caps()
    return value


def model_capabilities(name: str | None = None) -> list[str]:
    """Ollama capabilities for one model (`completion`, `vision`, ...)."""
    target = str(name or model_name())
    now = time.monotonic()
    with _caps_lock:
        hit = _caps_cache.get(target)
        if hit and now - hit[0] < _CAPS_TTL:
            return list(hit[1])
    try:
        data = _request("/api/show", {"model": target}, timeout=3.0)
        caps = [str(x) for x in data.get("capabilities") or []]
    except LLMError:
        caps = []
    with _caps_lock:
        _caps_cache[target] = (now, caps)
    return caps


def model_info() -> list[dict]:
    """Installed Ollama models with capability flags for the picker."""
    try:
        data = _request("/api/tags", timeout=3.0)
    except LLMError:
        return []
    rows = []
    for item in data.get("models") or []:
        name = str(item.get("name") or item.get("model") or "")
        if not name:
            continue
        caps = model_capabilities(name)
        details = item.get("details") or {}
        rows.append(
            {
                "name": name,
                "size": item.get("size"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "capabilities": caps,
                "completion": "completion" in caps,
                "vision": "vision" in caps,
                "selected": name == model_name(),
            }
        )
    rows.sort(key=lambda row: (not row["selected"], row["name"]))
    return rows



def base_url() -> str:
    return (os.environ.get("IMGS_OLLAMA_URL") or DEFAULT_URL).rstrip("/")


def brain_mode() -> str:
    mode = (os.environ.get("IMGS_ASSIST_BRAIN") or "auto").strip().lower()
    return mode if mode in BRAINS else "auto"


def _timeout() -> float:
    try:
        return max(5.0, float(os.environ.get("IMGS_ASSIST_TIMEOUT") or 90))
    except (TypeError, ValueError):
        return 90.0


def _request(path: str, payload=None, timeout: float = 5.0) -> dict:
    url = base_url() + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"ต่อ Ollama ไม่ได้: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise LLMError("Ollama ตอบไม่ใช่ JSON") from exc


def installed_models() -> list[str]:
    """Cached list of model names (30 s) — keeps health checks cheap."""
    now = time.monotonic()
    with _status_lock:
        if now - _status_cache["stamp"] < _TAGS_TTL:
            return list(_status_cache["models"])
    try:
        data = _request("/api/tags", timeout=2.0)
        names = [str(m.get("name") or m.get("model") or "") for m in data.get("models") or []]
        error = None
    except LLMError as exc:
        names, error = [], str(exc)
    with _status_lock:
        _status_cache.update(stamp=now, models=names, error=error)
    return names


def invalidate_models() -> None:
    """Force the next status/model-list call to re-scan Ollama."""
    with _status_lock:
        _status_cache.update(stamp=0.0, models=[], error=None)
    _invalidate_caps()


def available() -> bool:
    target = model_name()
    return any(name == target or name.startswith(target + ":") for name in installed_models())


def status() -> dict:
    mode = brain_mode()
    names = installed_models()
    with _status_lock:
        error = _status_cache.get("error")
    model = model_name()
    ok = any(name == model or name.startswith(model + ":") for name in names)
    if mode == "rules":
        label = "กฎ (rule-based)"
    elif ok:
        label = f"{model} · local"
    else:
        label = "กฎ (ไม่พบโมเดล LLM)"
    return {
        "mode": mode,
        "model": model,
        "default": DEFAULT_MODEL,
        "saved": _saved_model(),
        "env_override": bool((os.environ.get("IMGS_ASSIST_MODEL") or "").strip()),
        "available": ok,
        "label": label,
        "url": base_url(),
        "error": error,
    }


def loaded() -> bool:
    """True when Ollama currently holds the model (likely in VRAM)."""
    try:
        data = _request("/api/ps", timeout=1.0)
    except LLMError:
        return False
    target = model_name()
    return any(
        str(m.get("name") or m.get("model") or "") == target for m in data.get("models") or []
    )


def unload() -> bool:
    """Free VRAM before SANA loads. Never loads the model just to unload it."""
    if not loaded():
        return False
    try:
        _request(
            "/api/generate",
            {"model": model_name(), "keep_alive": 0, "stream": False},
            timeout=5.0,
        )
        return True
    except LLMError:
        return False


# ── intent prompt ────────────────────────────────────────────────────────────
def _tool_lines() -> str:
    import assistant as ast

    lines = []
    for tool in ast.TOOLS:
        required = ast.REQUIRED_ARGS.get(tool["name"]) or []
        args = f" args: {', '.join(required)}" if required else ""
        lines.append(f"- {tool['name']} — {tool['title']}{args} · เช่น “{tool['examples'][0]}”")
    return "\n".join(lines)


def system_prompt() -> str:
    return (
        "คุณแปลงคำสั่งของผู้ใช้ (ไทยหรืออังกฤษ) เป็น JSON สำหรับแอป Image Studio\n"
        "tool ที่ใช้ได้:\n"
        f"{_tool_lines()}\n\n"
        "กติกา:\n"
        "1) ตอบ JSON ก้อนเดียวเท่านั้น ห้ามมีข้อความอื่นหรือ markdown\n"
        '2) รูปแบบ: {"tool": "<ชื่อ tool>", "args": {…}, "clarify": "<คำถามกลับ>"}\n'
        "3) ใส่ args เฉพาะค่าที่ผู้ใช้ระบุจริง ห้ามแต่งชื่อ LoRA/trigger/caption ขึ้นเอง\n"
        "4) ถ้าคำสั่งตรงกับ tool แต่ค่าจำเป็นยังขาด ให้ tool เป็นชื่อนั้นและ clarify ถามสั้น ๆ\n"
        "5) ถ้าไม่ตรง tool ใดเลย ให้ tool = null และ clarify ถามสั้น ๆ ว่าให้ช่วยอะไรในแอป\n"
        "6) งานลบ/เทรน ระบบจะขอผู้ใช้ยืนยันเอง ไม่ต้องถามใน clarify"
    )


_EXAMPLES = [
    ("ตรวจ dataset ให้หน่อย", {"tool": "dataset_summary", "args": {}, "clarify": ""}),
    ("มี LoRA อะไรบ้าง", {"tool": "list_loras", "args": {}, "clarify": ""}),
    (
        "เริ่มเทรนชื่อ my-lora-v7 steps 800",
        {"tool": "start_training", "args": {"name": "my-lora-v7", "steps": 800}, "clarify": ""},
    ),
    ("ลบภาพล่าสุด", {"tool": "delete_history", "args": {}, "clarify": ""}),
    (
        "ช่วยเขียนกลอนให้หน่อย",
        {"tool": None, "args": {}, "clarify": "ผมช่วยเรื่องภาพและ LoRA ในแอปได้ครับ"},
    ),
]


def _extract_json(text: str) -> dict:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise LLMError("LLM ตอบไม่ใช่ JSON") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("LLM ตอบ JSON เสีย") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM ตอบไม่ใช่ object")
    return data


def raw_intent(message: str, lang: str = "th") -> dict:
    """Ask the model for one tool call. Raises LLMError on any problem."""
    if not available():
        raise LLMError(f"ไม่พบโมเดล {model_name()} ใน Ollama")
    system = system_prompt()
    if (lang or "").strip().lower() == "en":
        system += (
            '\n7) The app UI language is English — write the "clarify" text '
            "in English.\n"
        )
    messages = [{"role": "system", "content": system}]
    for user, reply in _EXAMPLES:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
    messages.append({"role": "user", "content": str(message)[:2000]})
    data = _request(
        "/api/chat",
        {
            "model": model_name(),
            "messages": messages,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 220},
        },
        timeout=_timeout(),
    )
    content = ((data.get("message") or {}).get("content")) or ""
    return _extract_json(content)
