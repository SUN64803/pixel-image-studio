"""Assistant core — tool registry, rule-based intent parsing, angle helpers.

P1 deliberately ships **without an LLM**: the "brain" is a deterministic
parser that maps Thai/English phrasing onto a small tool registry.  The tool
executors live in `server.py` because they need the live job queue, model
manager and history; this module stays import-light so it can be unit-tested
without torch.

Tiers (agreed with the user):
  read        — safe, run immediately
  write       — data change, run immediately + Undo
  destructive — delete/train, must be confirmed by the user first
"""
from __future__ import annotations

import re
import threading
import time
import uuid

TIERS = ("read", "write", "destructive")

# ── angle tags (kept byte-for-byte in sync with studio/app.js) ───────────────
ANGLE_TAGS = [
    "front view",
    "side view",
    "three-quarter view",
    "back view",
    "close-up",
    "high angle",
    "low angle",
]

ANGLE_PATTERNS = [
    ("front view", r"\b(from the front|front view|facing (the )?(viewer|camera|forward))\b"),
    ("side view", r"\b(from the side|side view|side profile|in profile|profile view)\b"),
    ("three-quarter view", r"\b(three[- ]quarter|3/4|three quarter view)\b"),
    ("back view", r"\b(from behind|from the back|back view|rear view|seen from behind)\b"),
    ("close-up", r"\b(close[- ]?up|headshot|portrait shot)\b"),
    ("high angle", r"\b(high[- ]angle|looking down (at|from)|from above)\b"),
    ("low angle", r"\b(low[- ]angle|looking up (at|from)|from below)\b"),
]

_ANGLE_RES = [(tag, re.compile(pat, re.I)) for tag, pat in ANGLE_PATTERNS]


def detect_angle_tags(caption) -> list[str]:
    text = str(caption or "")
    return [tag for tag, rx in _ANGLE_RES if rx.search(text)]


def has_angle_tag(caption) -> bool:
    text = str(caption or "").lower()
    return any(tag in text for tag in ANGLE_TAGS)


def missing_angle_tags(caption) -> list[str]:
    text = str(caption or "").lower()
    return [tag for tag in detect_angle_tags(caption) if tag not in text]


def with_angle_tags(caption, tags) -> str:
    base = str(caption or "").strip()
    tags = list(tags or [])
    if not tags:
        return base
    return f"{', '.join(tags)}, {base}" if base else ", ".join(tags)


# ── tool registry (executors are wired in server.py) ────────────────────────
TOOLS = [
    {
        "name": "dataset_summary",
        "title": "ตรวจความพร้อม dataset",
        "tier": "read",
        "description": "สรุปจำนวนรูปเดี่ยว/หลายตัวละคร, tag มุมที่ขาด และสถานะพร้อมเทรน",
        "examples": ["ตรวจ dataset", "audit"],
    },
    {
        "name": "list_datasets",
        "title": "ดูรายการ dataset",
        "tier": "read",
        "description": "ลิสต์ชุดเทรนทั้งหมดพร้อมจำนวนรูปและ trigger",
        "examples": ["มี dataset อะไรบ้าง", "list datasets"],
    },
    {
        "name": "list_loras",
        "title": "ดูรายการ LoRA",
        "tier": "read",
        "description": "ลิสต์ LoRA ที่ติดตั้งไว้พร้อมขนาด/ตระกูล",
        "examples": ["มี LoRA อะไรบ้าง", "list loras"],
    },
    {
        "name": "list_models",
        "title": "ดูรายการโมเดล",
        "tier": "read",
        "description": "ลิสต์โมเดลที่สแกนพบและโมเดลที่เลือกอยู่",
        "examples": ["มีโมเดลอะไรบ้าง", "models"],
    },
    {
        "name": "gpu_status",
        "title": "สถานะ GPU / คิว",
        "tier": "read",
        "description": "อุณหภูมิ, VRAM, งานที่กำลังทำและคิว",
        "examples": ["สถานะ GPU", "การ์ดจอร้อนไหม"],
    },
    {
        "name": "history_stats",
        "title": "สรุปคลังภาพ",
        "tier": "read",
        "description": "จำนวนภาพในประวัติและภาพล่าสุด",
        "examples": ["มีกี่ภาพแล้ว", "history"],
    },
    {
        "name": "list_jobs",
        "title": "ดูคิวงาน",
        "tier": "read",
        "description": "งานที่กำลังทำ/รออยู่ในคิว",
        "examples": ["คิวมีอะไร", "jobs"],
    },
    {
        "name": "read_images",
        "title": "อ่านรูปด้วย vision",
        "tier": "read",
        "description": "ให้โมเดล vision ในเครื่องดูรูปใน dataset แล้วเสนอ tag มุม/คำบรรยาย/คำเตือนหลายตัวละคร (ไม่ส่งข้อมูลออก)",
        "examples": ["อ่านรูปใน dataset", "วิเคราะห์รูป", "read images"],
    },
    {
        "name": "fill_angle_tags",
        "title": "เติม tag มุมอัตโนมัติ",
        "tier": "write",
        "description": "เติม tag มุมมาตรฐานให้รูปเดี่ยวที่ caption บอกมุมอยู่แล้ว (atomic + Undo)",
        "examples": ["เติม tag มุม", "auto tag"],
    },
    {
        "name": "set_trigger",
        "title": "เปลี่ยน trigger word",
        "tier": "write",
        "description": "แก้ trigger ของ dataset (Undo ได้)",
        "examples": ["เปลี่ยน trigger เป็น my character"],
    },
    {
        "name": "set_caption",
        "title": "แก้ caption รายรูป",
        "tier": "write",
        "description": "แก้ caption ของไฟล์ที่ระบุ (Undo ได้)",
        "examples": ["caption 2fc60a9fdc9c เป็น back view, standing"],
    },
    {
        "name": "select_model",
        "title": "เลือกโมเดล",
        "tier": "write",
        "description": "สลับโมเดลที่ใช้สร้างภาพ (Undo ได้)",
        "examples": ["เลือกโมเดล sana"],
    },
    {
        "name": "generate_image",
        "title": "สร้างภาพ",
        "tier": "write",
        "description": "ส่งงานสร้างภาพเข้าคิว (ย้อนกลับไม่ได้)",
        "examples": ["สร้างภาพ my character standing in a garden"],
    },
    {
        "name": "delete_history",
        "title": "ลบภาพจากประวัติ",
        "tier": "destructive",
        "description": "ลบภาพล่าสุดหรือชุดที่ระบุ พร้อมไฟล์บนดิสก์ (ย้อนกลับไม่ได้)",
        "examples": ["ลบภาพล่าสุด"],
    },
    {
        "name": "clear_history",
        "title": "ล้างประวัติทั้งหมด",
        "tier": "destructive",
        "description": "ลบทุกภาพและประวัติ (ย้อนกลับไม่ได้)",
        "examples": ["ล้างประวัติ"],
    },
    {
        "name": "delete_lora",
        "title": "ลบ LoRA",
        "tier": "destructive",
        "description": "ลบไฟล์ LoRA ออกจากเครื่อง (ย้อนกลับไม่ได้)",
        "examples": ["ลบ lora old-test"],
    },
    {
        "name": "start_training",
        "title": "เริ่มเทรน LoRA",
        "tier": "destructive",
        "description": "เข้าคิวเทรน LoRA จาก dataset (ใช้เวลานาน/การ์ดร้อน)",
        "examples": ["เริ่มเทรนชื่อ my-lora-v7 steps 800"],
    },
    {
        "name": "help",
        "title": "ช่วยเหลือ",
        "tier": "read",
        "description": "แสดงความสามารถและตัวอย่างคำสั่ง",
        "examples": ["ช่วยอะไรได้บ้าง"],
    },
]

TOOL_INDEX = {tool["name"]: tool for tool in TOOLS}

# args each tool accepts / must have — also used to validate LLM output
ARG_KEYS = {
    "dataset_summary": [],
    "list_datasets": [],
    "list_loras": [],
    "list_models": [],
    "gpu_status": [],
    "history_stats": [],
    "list_jobs": [],
    "read_images": [],
    "fill_angle_tags": [],
    "set_trigger": ["trigger"],
    "set_caption": ["file", "caption"],
    "select_model": ["target"],
    "generate_image": ["prompt"],
    "delete_history": ["job_id"],
    "clear_history": [],
    "delete_lora": ["id"],
    "start_training": ["name", "steps", "rank"],
    "help": [],
}

REQUIRED_ARGS = {
    "set_trigger": ["trigger"],
    "set_caption": ["file", "caption"],
    "select_model": ["target"],
    "generate_image": ["prompt"],
    "delete_lora": ["id"],
    "start_training": ["name"],
}

SUGGESTIONS = [
    "ตรวจ dataset",
    "อ่านรูปใน dataset",
    "เติม tag มุม",
    "สถานะ GPU",
    "มี LoRA อะไรบ้าง",
    "ลบภาพล่าสุด",
]


# English quick commands — every one parses through the rule brain too
# (the rule regexes accept English keywords) so the LLM brain is optional.
SUGGESTIONS_EN = [
    "audit dataset",
    "read dataset images",
    "fill angle tags",
    "gpu status",
    "list loras",
    "delete latest image",
]


def suggestions_for(lang=None) -> list:
    """Quick-command labels for the UI language (they are sent as commands)."""
    return SUGGESTIONS_EN if (lang or "").strip().lower() == "en" else SUGGESTIONS


def tier_of(name: str) -> str:
    tool = TOOL_INDEX.get(name)
    return tool["tier"] if tool else "read"


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _clean_arg(text: str) -> str:
    value = str(text or "").strip().strip("\"'“”‘’ ")
    return re.sub(r"\s+", " ", value)


def _after(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.I)
    return _clean_arg(match.group(1)) if match else ""


_RESERVED_NAMES = {"lora", "model", "โมเดล", "ใหม่", "ต่อ", "sample"}


def _parse_rules(message) -> dict:
    """Rule-based parser — the deterministic fallback / offline brain."""
    t = _norm(message)
    if not t:
        return {"tool": None, "args": {}, "clarify": "พิมพ์คำสั่งหรือคำถามมาก่อนครับ"}

    # destructive first — "ลบประวัติ" must not be mistaken for a history read
    if re.search(r"ล้าง(ประวัติ|ทั้งหมด|ทุกอย่าง)|ลบ(ทั้งหมด|ทุกอย่าง)|clear history", t):
        return {"tool": "clear_history", "args": {}, "clarify": None}

    if re.search(r"ลบ.{0,12}(lora|โลร่า)", t):
        target = _after(t, r"ลบ.{0,12}(?:lora|โลร่า)\s*(?:ชื่อ)?\s*[:=]?\s*(.+)$")
        return {"tool": "delete_lora", "args": {"id": target}, "clarify": None}

    if re.search(r"ลบ.{0,12}(รูป|ภาพ|ประวัติ|history)|delete.{0,12}(image|history)", t):
        job_id = _after(t, r"\b([0-9a-f]{12,32})\b")
        return {"tool": "delete_history", "args": {"job_id": job_id}, "clarify": None}

    if re.search(
        r"(เริ่ม|สั่ง|ขอ).{0,8}เทรน|start.{0,8}train|เทรน.{0,8}lora|train.{0,8}lora", t
    ):
        name = _after(t, r"(?:ชื่อ|name)\s*[:=]?\s*([A-Za-z0-9._-]{2,60})")
        if not name:
            name = _after(t, r"lora\s*[:=]\s*([A-Za-z0-9._-]{2,60})")
        if name.lower() in _RESERVED_NAMES:
            name = ""
        steps = _after(t, r"steps?\s*[:=]?\s*(\d{1,5})")
        rank = _after(t, r"rank\s*[:=]?\s*(\d{1,4})")
        args = {"name": name}
        if steps:
            args["steps"] = int(steps)
        if rank:
            args["rank"] = int(rank)
        clarify = None if name else "ต้องระบุชื่อ LoRA ด้วยครับ เช่น “เริ่มเทรนชื่อ my-lora-v7”"
        return {"tool": "start_training", "args": args, "clarify": clarify}

    if re.search(r"(อ่าน|วิเคราะห์|สแกน|read|scan).{0,12}(รูป|ภาพ|image)", t):
        return {"tool": "read_images", "args": {}, "clarify": None}

    if re.search(r"(เติม|ใส่|เพิ่ม|fill|auto).{0,12}(tag|แท็ก|มุม)|tag.{0,6}มุม", t):
        return {"tool": "fill_angle_tags", "args": {}, "clarify": None}

    if re.search(r"(เปลี่ยน|ตั้ง|แก้|set).{0,10}trigger", t):
        value = _after(t, r"trigger\s*(?:ใหม่)?\s*(?:เป็น|=|:|to)\s*(.+)$")
        if not value:
            value = _after(t, r"trigger\s+(.+)$")
        return {
            "tool": "set_trigger",
            "args": {"trigger": value},
            "clarify": None if value else "ต้องระบุ trigger ใหม่ด้วยครับ เช่น “เปลี่ยน trigger เป็น my character”",
        }

    if re.search(
        r"(แก้|ตั้ง|เปลี่ยน|set).{0,10}caption|^caption\b|caption\s+[\w.-]+\s*(?:เป็น|=|:|->)", t
    ):
        match = re.search(
            r"caption\s*(?:ของ|of)?\s*([\w.-]+)\s*(?:เป็น|=|:|->)\s*(.+)$", t, re.I
        )
        if not match:
            return {
                "tool": "set_caption",
                "args": {},
                "clarify": "รูปแบบ: “caption <ชื่อไฟล์> เป็น <ข้อความใหม่>”",
            }
        return {
            "tool": "set_caption",
            "args": {"file": match.group(1), "caption": _clean_arg(match.group(2))},
            "clarify": None,
        }

    if re.search(r"เลือก.{0,10}(โมเดล|model)|switch model|use model", t):
        target = _after(t, r"(?:โมเดล|model)\s*[:=]?\s*(.+)$")
        return {
            "tool": "select_model",
            "args": {"target": target},
            "clarify": None if target else "ต้องระบุชื่อหรือ id ของโมเดลด้วยครับ",
        }

    if re.search(r"(สร้าง|เจน|gen|generate).{0,8}(ภาพ|รูป|image)", t):
        prompt = _after(t, r"(?:ภาพ|รูป|image)\s*(?:ว่า|ว่าเป็น|:|=)?\s*(.+)$")
        return {
            "tool": "generate_image",
            "args": {"prompt": prompt},
            "clarify": None if prompt else "ต้องระบุพรอมป์ตด้วยครับ เช่น “สร้างภาพ my character standing”",
        }

    if re.search(
        r"(ตรวจ|เช็ค|สรุป|ดู|สถานะ|พร้อม).{0,12}(dataset|ชุดเทรน|data ?set)|audit|พร้อมเทรน", t
    ):
        return {"tool": "dataset_summary", "args": {}, "clarify": None}

    if re.search(r"(ลิสต์|รายการ|มี|ดู).{0,10}(dataset|ชุดเทรน)|datasets", t):
        return {"tool": "list_datasets", "args": {}, "clarify": None}

    if re.search(r"(ลิสต์|รายการ|มี|ดู).{0,10}(lora|โลร่า)|loras", t):
        return {"tool": "list_loras", "args": {}, "clarify": None}

    if re.search(r"(ลิสต์|รายการ|มี|ดู).{0,10}(โมเดล|model)|models", t):
        return {"tool": "list_models", "args": {}, "clarify": None}

    if re.search(r"gpu|การ์ดจอ|vram|อุณหภูมิ|temp|สถานะเครื่อง|\bram\b", t):
        return {"tool": "gpu_status", "args": {}, "clarify": None}

    if re.search(r"ประวัติ|history|คลังภาพ|กี่ภาพ|กี่รูป|แกลเลอรี|gallery", t):
        return {"tool": "history_stats", "args": {}, "clarify": None}

    if re.search(r"คิว|queue|งานค้าง|\bjobs?\b|กำลังทำ", t):
        return {"tool": "list_jobs", "args": {}, "clarify": None}

    if re.search(r"\bhelp\b|ช่วยอะไรได้|ทำอะไรได้|คำสั่ง|ความสามารถ|ใช้งานยังไง", t):
        return {"tool": "help", "args": {}, "clarify": None}

    return {
        "tool": None,
        "args": {},
        "clarify": "ยังไม่เข้าใจคำสั่งนี้ — พิมพ์ “ช่วยอะไรได้บ้าง” เพื่อดูตัวอย่าง",
    }


def parse_intent_rules(message) -> dict:
    intent = _parse_rules(message)
    intent["brain"] = "rules"
    return intent


def _coerce_args(tool: str, args) -> dict:
    """Keep only args the tool understands; drop empty/unknown values."""
    allowed = set(ARG_KEYS.get(tool, [])) | {"dataset_id"}
    out: dict = {}
    if not isinstance(args, dict):
        return out
    for key, value in args.items():
        key = str(key or "").strip()
        if key not in allowed or value is None:
            continue
        if key in ("steps", "rank"):
            try:
                number = int(float(value))
            except (TypeError, ValueError):
                continue
            if number > 0:
                out[key] = number
            continue
        text = _clean_arg(value)
        if text and text.lower() not in ("null", "none"):
            out[key] = text
    return out


def validate_intent(data) -> dict:
    """Turn an LLM reply into a safe intent dict.

    The model only *chooses* a tool; args are filtered to the tool's schema and
    missing required args become a Thai clarify question. A non-empty clarify is
    ignored when the tool call is complete (small models like to add chatter).
    """
    fallback = "ยังไม่เข้าใจคำสั่งนี้ — พิมพ์ “ช่วยอะไรได้บ้าง” เพื่อดูตัวอย่าง"
    if not isinstance(data, dict):
        return {"tool": None, "args": {}, "clarify": fallback, "brain": "llm"}
    tool = str(data.get("tool") or "").strip().lower()
    if tool in ("", "none", "null"):
        tool = None
    if tool is not None and tool not in TOOL_INDEX:
        return {"tool": None, "args": {}, "clarify": fallback, "brain": "llm"}
    args = _coerce_args(tool, data.get("args") or {}) if tool else {}
    missing = [key for key in REQUIRED_ARGS.get(tool, []) if not args.get(key)]
    if missing:
        clarify = _clean_arg(data.get("clarify") or "")
        return {
            "tool": tool,
            "args": args,
            "clarify": clarify or f"ต้องระบุ {', '.join(missing)} ด้วยครับ",
            "brain": "llm",
        }
    clarify = None if tool else (_clean_arg(data.get("clarify") or "") or fallback)
    return {"tool": tool, "args": args, "clarify": clarify, "brain": "llm"}


def parse_intent(message, brain=None, lang="th") -> dict:
    """Map one user message onto a tool call.

    `brain`: "auto" (LLM when available, rules as fallback) · "llm" · "rules".
    `lang`: UI language — the LLM writes its reply/clarify text to match.
    Nothing is executed here.
    """
    import assistant_llm as llm

    mode = (brain or llm.brain_mode() or "auto").strip().lower()
    if mode not in ("auto", "llm", "rules"):
        mode = "auto"
    if mode != "rules":
        try:
            return validate_intent(llm.raw_intent(message, lang=lang))
        except Exception:  # noqa: BLE001 — any LLM problem → deterministic fallback
            pass
    return parse_intent_rules(message)


# ── reply summaries (Thai) ──────────────────────────────────────────────────
def _join(items, empty="—") -> str:
    items = [str(x) for x in (items or []) if str(x).strip()]
    return " · ".join(items) if items else empty


def summarize_result(tool: str, result: dict, brain=None) -> str:
    result = result or {}
    if tool == "dataset_summary":
        notes = result.get("notes") or []
        head = (
            f"dataset {result.get('name')} ({result.get('id')}) · trigger "
            f"“{result.get('trigger') or '—'}”\n"
            f"รูปทั้งหมด {result.get('total', 0)} — เดี่ยว {result.get('solo', 0)} · "
            f"หลายตัวละคร {result.get('multi', 0)}\n"
            f"tag มุมขาด {result.get('missing_tags', 0)} รูป "
            f"(เติมอัตโนมัติได้ {result.get('fillable', 0)} · ต้องเติมมือ {result.get('manual', 0)})\n"
            f"มุมที่มี: {_join(result.get('angles'))}\n"
            f"สถานะ: {'พร้อมเทรน ✅' if result.get('ready') else 'ยังไม่พร้อม ⚠️'}"
        )
        return head + ("\n• " + "\n• ".join(notes) if notes else "")

    if tool == "read_images":
        if not result.get("job_id"):
            return result.get("message") or "ไม่ได้เริ่มอ่านรูป"
        return (
            f"เริ่มอ่านรูปด้วย vision แล้ว {result.get('total', 0)} รูป"
            f" (job {result.get('job_id')})\n"
            "กำลังดูทีละรูปในเครื่อง — ระหว่างนี้การ์ดจอถูกใช้โดยโมเดล vision "
            "จะสร้างภาพ/เทรนพร้อมกันไม่ได้"
        )

    if tool == "fill_angle_tags":
        if not result.get("filled"):
            return "ไม่มี caption ที่เติม tag มุมได้อัตโนมัติ (รูปที่เหลือต้องเติมมือ)"
        text = f"เติม tag มุมแล้ว {result['filled']} รูป"
        if result.get("files"):
            text += "\n• " + "\n• ".join(result["files"])
        if result.get("remaining"):
            text += f"\nยังต้องเติมมืออีก {result['remaining']} รูป"
        return text

    if tool == "list_datasets":
        rows = result.get("datasets") or []
        if not rows:
            return "ยังไม่มี dataset"
        return "dataset ที่มี:\n• " + "\n• ".join(
            f"{r['name']} ({r['id']}) — {r['total']} รูป (เดี่ยว {r['solo']} · multi {r['multi']})"
            f" · trigger “{r['trigger'] or '—'}”"
            for r in rows
        )

    if tool == "list_loras":
        rows = result.get("loras") or []
        if not rows:
            return "ยังไม่มี LoRA"
        return "LoRA ที่ติดตั้ง:\n• " + "\n• ".join(
            f"{r['name']} — {r.get('family_label') or r.get('family') or 'any'} · {r.get('size_mb', 0)} MB"
            for r in rows
        )

    if tool == "list_models":
        rows = result.get("models") or []
        current = result.get("current")
        if not rows:
            return "ไม่พบโมเดล"
        body = "\n• ".join(
            f"{'▸ ' if r.get('id') == current else ''}{r.get('name')} ({r.get('id')})"
            f" · {r.get('family_label') or r.get('family') or '—'}"
            f"{'' if r.get('available') else ' · ไฟล์ยังไม่ครบ'}"
            for r in rows
        )
        return "โมเดลที่สแกนพบ:\n• " + body

    if tool == "gpu_status":
        return (
            f"GPU: {result.get('gpu') or '—'} · {result.get('temp', '—')}°C\n"
            f"VRAM: {result.get('vram_used_mb', 0)} / {result.get('vram_total_mb', 0)} MB\n"
            f"โมเดล: {result.get('model_state')} · คิว: {result.get('queue', 0)} งาน"
            f" · {'กำลังทำงาน' if result.get('busy') else 'ว่าง'}"
        )

    if tool == "history_stats":
        latest = result.get("latest")
        text = (
            f"ประวัติ {result.get('total', 0)} ชุด · ภาพรวม {result.get('images', 0)} ภาพ"
        )
        if latest:
            text += f"\nล่าสุด: {latest.get('created') or '—'} · {latest.get('model') or '—'}"
        return text

    if tool == "list_jobs":
        rows = result.get("jobs") or []
        if not rows:
            return "ไม่มีงานในคิว"
        return "งานในคิว:\n• " + "\n• ".join(
            f"{r.get('status')} · {r.get('message') or r.get('id')}"
            f" ({round(float(r.get('progress') or 0) * 100)}%)"
            for r in rows
        )

    if tool == "set_trigger":
        return f"เปลี่ยน trigger: “{result.get('old') or '—'}” → “{result.get('new') or '—'}”"
    if tool == "set_caption":
        return f"แก้ caption ของ {result.get('file')} แล้ว"
    if tool == "select_model":
        return f"เลือกโมเดล: {result.get('old') or '—'} → {result.get('new') or '—'}"
    if tool == "generate_image":
        return (
            f"ส่งงานสร้างภาพเข้าคิวแล้ว (job {result.get('job_id')})\n"
            f"พรอมป์ต: {result.get('prompt')}"
        )
    if tool == "delete_history":
        return f"ลบแล้ว {result.get('removed', 0)} ชุด ({result.get('images', 0)} ภาพ)"
    if tool == "clear_history":
        return f"ล้างประวัติแล้ว {result.get('removed', 0)} ชุด"
    if tool == "delete_lora":
        return f"ลบ LoRA “{result.get('name') or result.get('id')}” แล้ว"
    if tool == "start_training":
        return (
            f"เข้าคิวเทรน LoRA “{result.get('lora_name')}” แล้ว (job {result.get('job_id')})\n"
            f"dataset {result.get('dataset_id')} · {result.get('steps')} steps"
        )
    if tool == "help":
        return help_text(brain)
    return result.get("message") or "เรียบร้อย"


# English display names + example commands for the help listing
TOOL_TITLES_EN = {
    "dataset_summary": "Audit dataset readiness",
    "list_datasets": "List datasets",
    "list_loras": "List LoRAs",
    "list_models": "List models",
    "gpu_status": "GPU / queue status",
    "history_stats": "Gallery summary",
    "list_jobs": "Show job queue",
    "read_images": "Read images with vision",
    "fill_angle_tags": "Auto-fill angle tags",
    "set_trigger": "Change trigger word",
    "set_caption": "Edit image caption",
    "select_model": "Select model",
    "generate_image": "Generate image",
    "delete_history": "Delete image from history",
    "clear_history": "Clear all history",
    "delete_lora": "Delete LoRA",
    "start_training": "Start LoRA training",
}

EXAMPLES_EN = {
    "dataset_summary": "audit dataset",
    "list_datasets": "list datasets",
    "list_loras": "list loras",
    "list_models": "list models",
    "gpu_status": "gpu status",
    "history_stats": "history stats",
    "list_jobs": "show queue",
    "read_images": "read dataset images",
    "fill_angle_tags": "fill angle tags",
    "set_trigger": "set trigger to my character",
    "set_caption": "caption 2fc60a9fdc9c: back view, standing",
    "select_model": "use model sana",
    "generate_image": "generate image my character standing",
    "delete_history": "delete latest image",
    "clear_history": "clear history",
    "delete_lora": "delete lora old-test",
    "start_training": "start training name: my-lora-v7",
}


def help_text(brain=None, lang="th") -> str:
    en = (lang or "").strip().lower() == "en"
    groups = {
        "read": "Read data (runs right away)" if en else "อ่านข้อมูล (ทำเลย)",
        "write": "Data edits (run + Undo)" if en else "แก้ข้อมูล (ทำเลย + Undo)",
        "destructive": "Delete/Train (needs confirmation)"
        if en
        else "ลบ/เทรน (ต้องยืนยัน)",
    }
    if en:
        head = (
            "Brain: local LLM (nothing leaves the machine) — understands commands "
            "more flexibly. Examples:"
            if brain == "llm"
            else "Brain: rules (rule-based) — it understands commands like these examples:"
        )
    else:
        head = "สมอง: LLM โลคัลในเครื่อง (ไม่ส่งข้อมูลออก) — เข้าใจคำสั่งยืดหยุ่นขึ้น ตัวอย่าง:" if brain == "llm" else "สมอง: กฎ (rule-based) — เข้าใจคำสั่งตามตัวอย่างนี้:"
    lines = [head]
    for tier in TIERS:
        rows = [tool for tool in TOOLS if tool["tier"] == tier and tool["name"] != "help"]
        if not rows:
            continue
        lines.append(f"\n{groups[tier]}")
        for tool in rows:
            if en:
                lines.append(
                    f"• {TOOL_TITLES_EN.get(tool['name'], tool['title'])}"
                    f" — e.g. “{EXAMPLES_EN.get(tool['name'], tool['examples'][0])}”"
                )
            else:
                lines.append(f"• {tool['title']} — เช่น “{tool['examples'][0]}”")
    return "\n".join(lines)
    groups = {
        "read": "อ่านข้อมูล (ทำเลย)",
        "write": "แก้ข้อมูล (ทำเลย + Undo)",
        "destructive": "ลบ/เทรน (ต้องยืนยัน)",
    }
    if brain == "llm":
        head = "สมอง: LLM โลคัลในเครื่อง (ไม่ส่งข้อมูลออก) — เข้าใจคำสั่งยืดหยุ่นขึ้น ตัวอย่าง:"
    else:
        head = "สมอง: กฎ (rule-based) — เข้าใจคำสั่งตามตัวอย่างนี้:"
    lines = [head]
    for tier in TIERS:
        rows = [tool for tool in TOOLS if tool["tier"] == tier and tool["name"] != "help"]
        if not rows:
            continue
        lines.append(f"\n{groups[tier]}")
        lines.extend(f"• {tool['title']} — เช่น “{tool['examples'][0]}”" for tool in rows)
    return "\n".join(lines)


# ── in-memory pending / undo store ─────────────────────────────────────────
class ActionStore:
    """Bounded, TTL'd store for pending confirmations and undo closures.

    In-memory on purpose: restarting the engine clears pending actions and
    undo history, which is the safe default for destructive operations.
    """

    def __init__(self, limit: int = 50, ttl: float = 600.0):
        self._lock = threading.RLock()
        self._items: dict[str, tuple[float, dict]] = {}
        self._order: list[str] = []
        self._limit = max(1, int(limit))
        self._ttl = max(1.0, float(ttl))

    def _expired(self, stamp: float) -> bool:
        return (time.monotonic() - stamp) > self._ttl

    def _trim(self) -> None:
        while len(self._order) > self._limit:
            old = self._order.pop(0)
            self._items.pop(old, None)

    def add(self, payload: dict) -> str:
        self.prune()
        item_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._items[item_id] = (time.monotonic(), payload)
            self._order.append(item_id)
            self._trim()
        return item_id

    def get(self, item_id: str):
        with self._lock:
            row = self._items.get(str(item_id or ""))
            if row is None:
                return None
            if self._expired(row[0]):
                self._items.pop(item_id, None)
                return None
            return row[1]

    def pop(self, item_id: str):
        with self._lock:
            row = self._items.pop(str(item_id or ""), None)
            if row is not None:
                try:
                    self._order.remove(item_id)
                except ValueError:
                    pass
            if row is None or self._expired(row[0]):
                return None
            return row[1]

    def prune(self) -> int:
        with self._lock:
            dead = [key for key, (stamp, _) in self._items.items() if self._expired(stamp)]
            for key in dead:
                self._items.pop(key, None)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
            return len(dead)

    def count(self) -> int:
        self.prune()
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._order.clear()
