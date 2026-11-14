#!/usr/bin/env python3
"""Prompt presets — save/load generation prompt + settings bundles.

Storage: data/prompt_presets.json (a flat list of preset dicts).  Each preset:

    {
      "id": "a1b2c3d4e5f6",
      "name": "My Preset",
      "prompt": "…",
      "negative": "…",
      "settings": {            # subset the studio form understands
        "width": 768, "height": 1152, "steps": 31, "guidance": 4,
        "sampler": "dpm", "seed": 3873388, "model": "base-model-xl",
        "loras": [{"name": "style_il", "weight": 0.8}]
      },
      "created": "<iso>", "updated": "<iso>", "source": "ui" | "import"
    }

Import helpers understand two text formats:
* A1111/Forge "parameters" blob (prompt \n Negative prompt: … \n Steps: …)
* Our own prompt-collection.txt format:
      ### <name>
      Prompt: …
      Negative: …
      Settings: Steps: 31, Sampler: DPM++ 2M, …
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import sana_engine as se

PRESETS_FILE = se.DATA / "prompt_presets.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_presets() -> list[dict]:
    try:
        data = json.loads(PRESETS_FILE.read_text("utf-8"))
        presets = data.get("presets") if isinstance(data, dict) else data
        return [p for p in (presets or []) if isinstance(p, dict)]
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 — corrupt file must not kill the API
        print(f"[presets] อ่าน {PRESETS_FILE.name} ไม่ได้: {exc}", flush=True)
        return []


def save_presets(presets: list[dict]) -> None:
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PRESETS_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "presets": presets}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    tmp.replace(PRESETS_FILE)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def add_preset(name: str, prompt: str, negative: str, settings: dict | None,
               source: str = "ui") -> dict:
    presets = load_presets()
    name = " ".join(str(name or "").split())[:80] or "preset"
    entry = {
        "id": _new_id(),
        "name": name,
        "prompt": str(prompt or "").strip(),
        "negative": str(negative or "").strip(),
        "settings": settings or {},
        "created": _now(),
        "updated": _now(),
        "source": source,
    }
    presets.insert(0, entry)
    save_presets(presets)
    return entry


def update_preset(preset_id: str, **fields) -> dict | None:
    presets = load_presets()
    for p in presets:
        if p.get("id") == preset_id:
            for key in ("name", "prompt", "negative", "settings"):
                if key in fields and fields[key] is not None:
                    p[key] = fields[key]
            p["updated"] = _now()
            save_presets(presets)
            return p
    return None


def delete_preset(preset_id: str) -> bool:
    presets = load_presets()
    keep = [p for p in presets if p.get("id") != preset_id]
    if len(keep) == len(presets):
        return False
    save_presets(keep)
    return True


# ── A1111 / Forge "parameters" parsing ──────────────────────────────────────
def parse_a1111(text: str) -> dict:
    """Parse an A1111/Forge parameters blob into {prompt, negative, settings_str}."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return {"prompt": "", "negative": "", "settings_str": ""}
    neg = ""
    if "\nNegative prompt:" in text:
        prompt, rest = text.split("\nNegative prompt:", 1)
        lines = rest.split("\n")
        neg = lines[0].strip()
        settings_str = "\n".join(lines[1:]).strip()
    else:
        prompt = text
        settings_str = ""
        # settings live on the last line starting with "Steps:"
        lines = text.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if re.match(r"\s*Steps\s*:", lines[i]):
                settings_str = "\n".join(lines[i:]).strip()
                prompt = "\n".join(lines[:i]).strip()
                break
    return {"prompt": prompt.strip(), "negative": neg, "settings_str": settings_str}


# settings line keys → studio form fields
_SAMPLER_MAP = {
    "dpm++ 2m": "dpm",
    "dpm++ 2m karras": "dpm",
    "dpm++ sde": "dpm",
    "dpm++ 3m sde": "dpm",
    "euler": "euler",
    "euler a": "euler",
    "euler ancestral": "euler",
}


def _parse_kv(settings_str: str) -> dict:
    """'Steps: 31, Sampler: DPM++ 2M, CFG scale: 4, …' → {steps: 31, …}.

    Values may contain commas inside quotes (Lora hashes: "a: 1, b: 2").
    """
    out: dict = {}
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9 _/()-]*):\s*(\"(?:[^\"\\]|\\.)*\"|[^,]*)(?:,|$)")
    for m in pattern.finditer(settings_str or ""):
        key = m.group(1).strip().lower()
        val = m.group(2).strip().strip('"').strip()
        if key:
            out[key] = val
    return out


# A1111/Forge keys the studio has no field for.  Reported to the UI so an
# import never drops half of the source settings without telling the user.
_IGNORED_SETTINGS = {
    "hires upscale": "Hires upscale",
    "hires steps": "Hires steps",
    "hires upscaler": "Hires upscaler",
    "denoising strength": "Denoising strength",
    "clip skip": "Clip skip",
    "vae": "VAE",
    "vae hash": "VAE hash",
    "ensd": "ENSD",
    "version": "Version",
    "lora hashes": "LoRA hashes",
    "ti hashes": "TI hashes",
    "schedule type": "Schedule type",
    "distilled cfg scale": "Distilled CFG",
    "ngms": "NGMS",
}


def normalize_prompt_syntax(text: str) -> tuple[str, list[str]]:
    """Strip A1111 weight syntax that diffusers cannot use.

    The studio does not run Compel, so `(word:1.2)`, `[word]`, `{word}` and
    `BREAK` would reach the tokenizer as literal characters.  Keeping the
    words and dropping the markers preserves the meaning of the prompt.
    """
    notes: list[str] = []
    s = str(text or "")
    if not s.strip():
        return "", notes
    if re.search(r"\bBREAK\b", s):
        s = re.sub(r"\bBREAK\b", ", ", s)
        notes.append("แทน BREAK ด้วย ','")
    changed = False
    for _ in range(4):  # nested emphasis up to 4 levels
        new = re.sub(r"\(([^()]*?):\s*-?[0-9]*\.?[0-9]+\s*\)", r"\1", s)
        new = re.sub(r"\(([^()]*)\)", r"\1", new)
        new = re.sub(r"\[([^\[\]]*)\]", r"\1", new)
        new = re.sub(r"\{([^{}]*)\}", r"\1", new)
        if new == s:
            break
        s = new
        changed = True
    if changed:
        notes.append("ลบวงเล็บ/น้ำหนักคำ (A1111 syntax) — engine ไม่รองรับ")
    # BREAK → "," plus bracket removal can leave ", ," runs behind.
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,")
    return s, notes


def settings_from_a1111(settings_str: str, prompt: str = "", negative: str = "") -> dict:
    """Map an A1111 settings line onto the studio form fields.

    Only fields that were really present in the source are returned: a missing
    key must not overwrite the form with a default (that used to set steps=2
    and CFG=1 on every preset without settings).  `notes` explains clamps and
    prompt-syntax normalisation, `ignored` lists A1111 keys the studio cannot
    use, so the UI can report them instead of dropping them silently.
    """
    kv = _parse_kv(settings_str)
    settings: dict = {}
    notes: list[str] = []

    try:
        steps = int(float(kv.get("steps", 0)) or 0)
    except (TypeError, ValueError):
        steps = 0
    if steps > 0:
        clamped = max(2, min(60, steps))
        if clamped != steps:
            notes.append(f"steps {steps} → {clamped} (รองรับ 2–60)")
        settings["steps"] = clamped

    try:
        guidance = float(kv.get("cfg scale", 0) or 0)
    except (TypeError, ValueError):
        guidance = 0.0
    if guidance > 0:
        clamped_g = max(1.0, min(15.0, guidance))
        if clamped_g != guidance:
            notes.append(f"CFG {guidance:g} → {clamped_g:g} (รองรับ 1–15)")
        settings["guidance"] = clamped_g

    size = kv.get("size", "")
    m = re.match(r"(\d+)\s*[x×]\s*(\d+)", size)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        cw = max(256, min(2048, w - w % 32))
        ch = max(256, min(2048, h - h % 32))
        if (cw, ch) != (w, h):
            notes.append(f"ขนาด {w}×{h} → {cw}×{ch} (คูณ 32 · 256–2048)")
        settings["width"], settings["height"] = cw, ch

    try:
        seed = int(kv.get("seed", ""))
    except (TypeError, ValueError):
        seed = None
    if seed is not None and seed >= 0:
        settings["seed"] = seed

    sampler = (kv.get("sampler", "") or "").strip().lower()
    if sampler:
        settings["sampler"] = _SAMPLER_MAP.get(
            sampler,
            "dpm" if "dpm" in sampler else "euler" if "euler" in sampler else "dpm",
        )
        if sampler not in _SAMPLER_MAP:
            notes.append(f"sampler '{kv.get('sampler')}' → {settings['sampler']}")

    model = (kv.get("model", "") or "").strip()
    if model:
        settings["model"] = model

    ignored = [label for key, label in _IGNORED_SETTINGS.items() if key in kv]
    if ignored:
        settings["ignored"] = ignored

    # LoRA weights come from <lora:name:weight> tags in the prompt
    cleaned_prompt, loras = loras_from_prompt(prompt)
    if loras:
        settings["loras"] = loras
    cleaned_prompt, prompt_notes = normalize_prompt_syntax(cleaned_prompt)
    cleaned_negative, neg_notes = normalize_prompt_syntax(negative)
    notes.extend(prompt_notes)
    notes.extend(neg_notes)
    settings["cleaned_prompt"] = cleaned_prompt
    settings["cleaned_negative"] = cleaned_negative
    settings["model_hash"] = (kv.get("model hash", "") or "").strip()
    if notes:
        settings["notes"] = notes
    return settings


def loras_from_prompt(prompt: str) -> tuple[str, list[dict]]:
    """Strip <lora:name:weight> tags from a prompt; return (clean prompt, loras).

    Forge prompts embed adapter weights as tags; the studio applies LoRAs
    through its own list, so the tags must move out of the prompt text.
    """
    loras: list[dict] = []
    seen: set[str] = set()

    def _sub(m: re.Match) -> str:
        name = m.group(1).strip()
        try:
            weight = max(0.0, min(2.0, float(m.group(2))))
        except (TypeError, ValueError):
            weight = 1.0
        if name and name.lower() not in seen:
            seen.add(name.lower())
            loras.append({"name": name, "weight": weight})
        return " "

    cleaned = re.sub(r"<lora:([^:>]+):([0-9.]+)>", _sub, prompt or "", flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    return cleaned, loras


# ── prompt-collection.txt format ─────────────────────────────────────────────
def parse_collection(text: str) -> list[dict]:
    """Parse our '### name / Prompt: / Negative: / Settings:' text format.

    Values may span multiple lines (Forge prompts often carry ADDCOMM/ADDCOL
    regional markers and <lora:…> tags on their own lines), so each field runs
    until the next field header / next ### block / end of text.
    """
    text = (text or "").replace("\r\n", "\n")
    blocks = re.split(r"\n(?=###\s?)", "\n" + text)
    presets: list[dict] = []
    for block in blocks:
        head = re.match(r"###[ \t]*(.+?)[ \t]*$", block, re.M)
        if not head:
            continue
        name = head.group(1).strip()
        body = block[head.end():]
        pm = re.search(r"^Prompt:[ \t]*(.*?)(?=^Negative:|^Settings:|^###|\Z)",
                       body, re.M | re.S)
        nm = re.search(r"^Negative:[ \t]*(.*?)(?=^Settings:|^###|\Z)",
                       body, re.M | re.S)
        sm = re.search(r"^Settings:[ \t]*(.*?)(?=^###|\Z)", body, re.M | re.S)
        prompt = (pm.group(1) if pm else "").strip()
        neg = (nm.group(1) if nm else "").strip()
        settings_str = " ".join(
            line.strip()
            for line in (sm.group(1) if sm else "").splitlines()
            if line.strip()
        )
        # a prompt field may itself embed a full A1111 blob — parse as fallback
        p2 = parse_a1111(prompt)
        if p2["negative"] and not neg:
            neg = p2["negative"]
        if p2["settings_str"] and not settings_str:
            settings_str = p2["settings_str"]
        if p2["prompt"]:
            prompt = p2["prompt"]
        settings = settings_from_a1111(settings_str, prompt, neg)
        cleaned = settings.pop("cleaned_prompt", None)
        if cleaned:
            prompt = cleaned
        cleaned_neg = settings.pop("cleaned_negative", None)
        if cleaned_neg:
            neg = cleaned_neg
        if not (prompt or neg or settings):
            continue
        presets.append({
            "name": name[:80],
            "prompt": prompt,
            "negative": neg,
            "settings": settings,
            "source": "import",
        })
    return presets


def import_presets(entries: list[dict], replace_names: bool = False) -> dict:
    """Merge imported entries into storage; skip entries whose name already exists."""
    presets = load_presets()
    existing = {p.get("name", "").strip().lower(): p for p in presets}
    added = skipped = updated = 0
    for e in entries:
        name = (e.get("name") or "").strip()
        if not name:
            skipped += 1
            continue
        if name.lower() in existing:
            if replace_names:
                target = existing[name.lower()]
                target.update({
                    "prompt": e.get("prompt", target.get("prompt", "")),
                    "negative": e.get("negative", target.get("negative", "")),
                    "settings": e.get("settings", target.get("settings", {})),
                    "updated": _now(),
                    "source": "import",
                })
                updated += 1
            else:
                skipped += 1
            continue
        entry = {
            "id": _new_id(),
            "name": name[:80],
            "prompt": e.get("prompt", ""),
            "negative": e.get("negative", ""),
            "settings": e.get("settings") or {},
            "created": _now(),
            "updated": _now(),
            "source": "import",
        }
        presets.insert(0, entry)
        existing[name.lower()] = entry
        added += 1
    if added or updated or (replace_names and entries):
        save_presets(presets)
    return {"added": added, "updated": updated, "skipped": skipped, "total": len(presets)}


def read_collection_file(filename: str) -> str:
    """Read a text file from the data dir by basename (no traversal)."""
    base = Path(filename).name
    if not base.lower().endswith(".txt"):
        raise ValueError("รองรับเฉพาะไฟล์ .txt")
    path = se.DATA / base
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ {base} ในโฟลเดอร์ data/")
    return path.read_text(encoding="utf-8", errors="replace")
