"""Prompt presets — parser, settings mapping and CRUD/import API tests."""
import pytest
from fastapi.testclient import TestClient

import prompt_presets as pp
import server as srv


@pytest.fixture()
def client():
    with TestClient(srv.app) as c:
        yield c


A1111_BLOB = (
    "masterpiece, best quality, <lora:style_il:0.8>\n"
    "Negative prompt: bad anatomy, low quality\n"
    "Steps: 31, Sampler: DPM++ 2M, CFG scale: 4, Seed: 3873388, "
    "Size: 768x1152, Model: base-model-xl"
)

COLLECTION = (
    "### Duo preset\n"
    "Prompt:2girls, Duo, Score_9, <lora:training_x:0.8>\n"
    "ADDCOMM\n"
    "white fur, detailed, <lora:tag_two:0.6>\n"
    "Negative:bad anatomy, ((Text))\n"
    "Settings:Steps: 31, Sampler: DPM++ 2M, CFG scale: 4, Seed: 3432053035, "
    "Size: 1152x768, Model: base-model-xl\n"
    "\n"
    "### Second preset\n"
    "Prompt:score_9, <lora:pony_style:1>\n"
    "Negative:worst quality\n"
)


def test_parse_a1111_blob():
    parsed = pp.parse_a1111(A1111_BLOB)
    assert parsed["prompt"].startswith("masterpiece")
    assert parsed["negative"] == "bad anatomy, low quality"
    assert parsed["settings_str"].startswith("Steps: 31")


def test_settings_from_a1111_maps_form_fields():
    settings = pp.settings_from_a1111(
        "Steps: 31, Sampler: DPM++ 2M, CFG scale: 4, Seed: 3873388, "
        "Size: 768x1152, Model: base-model-xl",
        A1111_BLOB.split("\n")[0],
    )
    assert settings["steps"] == 31
    assert settings["guidance"] == 4.0
    assert settings["width"] == 768 and settings["height"] == 1152
    assert settings["seed"] == 3873388
    assert settings["sampler"] == "dpm"
    assert settings["model"] == "base-model-xl"
    # lora tags are lifted out of the prompt into the loras list
    assert settings["loras"] == [{"name": "style_il", "weight": 0.8}]
    assert "<lora:" not in settings["cleaned_prompt"]


def test_parse_collection_keeps_multiline_and_all_loras():
    entries = pp.parse_collection(COLLECTION)
    assert [e["name"] for e in entries] == ["Duo preset", "Second preset"]
    first = entries[0]
    # multi-line regional prompt must survive intact
    assert "ADDCOMM" in first["prompt"] and "white fur, detailed" in first["prompt"]
    names = [l["name"] for l in first["settings"]["loras"]]
    assert names == ["training_x", "tag_two"]
    assert first["settings"]["width"] == 1152 and first["settings"]["height"] == 768
    second = entries[1]
    assert second["settings"]["loras"] == [{"name": "pony_style", "weight": 1.0}]
    assert "<lora:" not in second["prompt"]


def test_preset_crud_roundtrip():
    entry = pp.add_preset("  Test  Preset ", "hello", "bad", {"steps": 20})
    try:
        assert entry["name"] == "Test Preset"
        all_presets = pp.load_presets()
        assert any(p["id"] == entry["id"] for p in all_presets)
        updated = pp.update_preset(entry["id"], prompt="hello v2")
        assert updated["prompt"] == "hello v2"
        assert pp.delete_preset(entry["id"]) is True
        assert pp.delete_preset(entry["id"]) is False
    finally:
        pp.delete_preset(entry["id"])


def test_import_presets_skips_duplicate_names():
    first = pp.import_presets([{"name": "Dup Test", "prompt": "a", "negative": "b"}])
    assert first["added"] == 1
    second = pp.import_presets([{"name": "Dup Test", "prompt": "c", "negative": "d"}])
    assert second["added"] == 0 and second["skipped"] == 1
    forced = pp.import_presets(
        [{"name": "Dup Test", "prompt": "c", "negative": "d"}], replace_names=True
    )
    assert forced["updated"] == 1
    entry = next(p for p in pp.load_presets() if p["name"] == "Dup Test")
    assert entry["prompt"] == "c"
    pp.delete_preset(entry["id"])


def test_import_rejects_missing_collection_file(client):
    res = client.post(
        "/api/presets/import", json={"from_file": "does-not-exist.txt"}
    )
    assert res.status_code == 404
    res = client.post(
        "/api/presets/import", json={"from_file": "../../etc/passwd"}
    )
    assert res.status_code == 400


def test_preset_api_crud_and_import(client):
    res = client.post(
        "/api/presets",
        json={"name": "API Preset", "prompt": "p1", "negative": "n1", "settings": {"steps": 20}},
    )
    assert res.status_code == 200
    entry = res.json()["preset"]
    try:
        listed = client.get("/api/presets").json()["presets"]
        assert any(p["id"] == entry["id"] for p in listed)

        res = client.put(
            f"/api/presets/{entry['id']}", json={"prompt": "p2"}
        )
        assert res.status_code == 200
        assert res.json()["preset"]["prompt"] == "p2"

        # import a collection text with 2 sets; one shares the API Preset name
        coll = (
            "### API Preset\nPrompt:imported\nNegative:imported-neg\n\n"
            "### Imported Set\nPrompt:imported2\nNegative:neg2\n"
        )
        res = client.post("/api/presets/import", json={"text": coll})
        assert res.status_code == 200
        counts = res.json()
        assert counts["added"] == 1 and counts["skipped"] == 1
        names = [p["name"] for p in counts["presets"]]
        assert "Imported Set" in names
    finally:
        srv_none = None  # cleanup regardless
        for p in pp.load_presets():
            if p["name"] in ("API Preset", "Imported Set"):
                pp.delete_preset(p["id"])


def test_import_single_a1111_blob_with_name(client):
    res = client.post(
        "/api/presets/import", json={"text": A1111_BLOB, "name": "Blob Test"}
    )
    assert res.status_code == 200
    assert res.json()["added"] == 1
    entry = next(p for p in pp.load_presets() if p["name"] == "Blob Test")
    assert entry["settings"]["steps"] == 31
    assert entry["settings"]["loras"] == [{"name": "style_il", "weight": 0.8}]
    pp.delete_preset(entry["id"])


def test_presets_file_ignored_shape():
    # corrupt file → empty list instead of a crash
    path = pp.PRESETS_FILE
    backup = path.read_text() if path.exists() else None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json")
        assert pp.load_presets() == []
    finally:
        if backup is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(backup)


def test_settings_from_a1111_invents_no_defaults():
    # A preset without a Settings: line must not reset the form to steps=2/CFG=1.
    settings = pp.settings_from_a1111("", "a prompt")
    for key in ("steps", "guidance", "sampler", "width", "height", "seed", "model"):
        assert key not in settings
    assert settings["cleaned_prompt"] == "a prompt"
    assert settings["cleaned_negative"] == ""


def test_settings_from_a1111_reports_ignored_and_clamped():
    settings = pp.settings_from_a1111(
        "Steps: 150, CFG scale: 30, Size: 3000x700, Sampler: LMS, "
        "Clip skip: 2, Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+, "
        "Denoising strength: 0.45, VAE: sdxl_vae.safetensors",
        "a prompt",
    )
    assert settings["steps"] == 60
    assert settings["guidance"] == 15.0
    assert (settings["width"], settings["height"]) == (2048, 672)
    assert settings["sampler"] == "dpm"
    assert set(settings["ignored"]) == {
        "Hires upscale",
        "Hires upscaler",
        "Denoising strength",
        "Clip skip",
        "VAE",
    }
    notes = " | ".join(settings["notes"])
    assert "steps 150 → 60" in notes
    assert "CFG 30 → 15" in notes
    assert "3000×700 → 2048×672" in notes
    assert "LMS" in notes


def test_settings_from_a1111_keeps_valid_values_without_notes():
    settings = pp.settings_from_a1111(
        "Steps: 28, CFG scale: 4.5, Size: 832x1216, Sampler: Euler a, Seed: 42",
        "a prompt",
    )
    assert settings["steps"] == 28
    assert settings["guidance"] == 4.5
    assert (settings["width"], settings["height"]) == (832, 1216)
    assert settings["sampler"] == "euler"
    assert settings["seed"] == 42
    assert "notes" not in settings and "ignored" not in settings


def test_normalize_prompt_syntax_strips_weights_and_break():
    cleaned, notes = pp.normalize_prompt_syntax(
        "masterpiece, (best quality:1.3), [low quality], {glow} BREAK bad"
    )
    assert cleaned == "masterpiece, best quality, low quality, glow, bad"
    assert notes  # explains what was rewritten


def test_normalize_prompt_syntax_plain_text_is_untouched():
    cleaned, notes = pp.normalize_prompt_syntax("simple prompt, no syntax")
    assert cleaned == "simple prompt, no syntax"
    assert notes == []
    assert pp.normalize_prompt_syntax("") == ("", [])


def test_collection_normalizes_negative_emphasis():
    entries = pp.parse_collection(COLLECTION)
    first = entries[0]
    assert first["negative"] == "bad anatomy, Text"
    assert "((" not in first["negative"]
