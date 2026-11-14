"""Base-model variants (Pony / Illustrious / …) — detection and overrides."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sana_engine as se
import server as srv


@pytest.fixture()
def client():
    with TestClient(srv.app) as c:
        yield c


@pytest.fixture()
def clean_variants():
    """Keep the real override file untouched; restore it afterwards."""
    path = se._VARIANTS_FILE
    backup = path.read_text(encoding="utf-8") if path.exists() else None
    try:
        yield path
    finally:
        if backup is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(backup, encoding="utf-8")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("some-style-v2-PonyXL", "pony"),
        ("some-guy-PONYv1", "pony"),
        ("style-a_IL", "illustrious"),
        ("style-b_IL", "illustrious"),
        ("nightstone style - Illustrious", "illustrious"),
        ("myIllustriousSDXL_v130", "illustrious"),
        ("noobaiXLNAIXL_vPred10Version", "noobai"),
        ("flux1-dev", "flux"),
        ("Chroma1-HD", "chroma"),
        # generic SDXL names must stay undecided so the family fallback applies
        ("mystery-xl", None),
        ("add-detail-xl", None),
        ("mystery-style-v1", None),
        ("", None),
    ],
)
def test_variant_hint(name, expected):
    assert se._variant_hint(name) == expected


def test_variant_hint_does_not_match_inside_words():
    # "will" / "smile" contain "il" but are not an Illustrious marker
    assert se._variant_hint("smile style") is None
    assert se._variant_hint("willing") is None


def test_variant_for_precedence(tmp_path):
    path = tmp_path / "style-a_IL.safetensors"
    key = se._override_key(path)
    # name wins over metadata
    assert se._variant_for(path, "sdxl", {"base_model": "pony"}, {}) == (
        "illustrious",
        "name",
    )
    # override wins over name
    assert se._variant_for(path, "sdxl", {}, {key: "pony"}) == ("pony", "override")
    # metadata is used when the name is generic
    generic = tmp_path / "mystery.safetensors"
    meta = {"ss_base_model_version": "ponyDiffusionV6XL"}
    assert se._variant_for(generic, "sdxl", meta, {}) == ("pony", "metadata")
    # nothing matches → the engine family is the best guess
    assert se._variant_for(generic, "sdxl", {}, {}) == ("sdxl", "family")
    assert se._variant_for(generic, None, {}, {}) == ("any", "family")


def test_set_variant_roundtrip_and_clear(tmp_path, clean_variants):
    path = tmp_path / "mystery-xl.safetensors"
    key = se._override_key(path)

    result = se.set_variant(path, "pony")
    assert result["variant"] == "pony" and result["path"] == key
    assert se.load_variant_overrides()[key] == "pony"

    # explicit "any" is stored, not treated as a clear
    se.set_variant(path, "any")
    assert se.load_variant_overrides()[key] == "any"

    se.set_variant(path, "")
    assert key not in se.load_variant_overrides()


def test_set_variant_rejects_unknown_value(tmp_path, clean_variants):
    with pytest.raises(ValueError):
        se.set_variant(tmp_path / "x.safetensors", "midjourney")


def test_load_variant_overrides_ignores_corrupt_json(clean_variants):
    clean_variants.parent.mkdir(parents=True, exist_ok=True)
    clean_variants.write_text("not json", encoding="utf-8")
    assert se.load_variant_overrides() == {}

    # wrong shape is ignored too
    clean_variants.write_text(json.dumps(["pony"]), encoding="utf-8")
    assert se.load_variant_overrides() == {}


def test_model_entry_exposes_variant(tmp_path):
    path = tmp_path / "myIllustriousSDXL_v130.safetensors"
    entry = se._model_entry(path, "sdxl", "single", tmp_path)
    assert entry["variant"] == "illustrious"
    assert entry["variant_label"] == "Illustrious"
    assert entry["variant_source"] == "name"

    with_override = se._model_entry(
        path, "sdxl", "single", tmp_path, overrides={se._override_key(path): "pony"}
    )
    assert with_override["variant"] == "pony"
    assert with_override["variant_source"] == "override"


def test_variants_api_roundtrip(client, tmp_path, clean_variants):
    path = str(tmp_path / "style-a_IL.safetensors")
    res = client.post("/api/variants", json={"path": path, "variant": "pony"})
    assert res.status_code == 200
    assert res.json()["variant"] == "pony"

    data = client.get("/api/variants").json()
    assert data["overrides"][se._override_key(Path(path))] == "pony"
    assert "illustrious" in data["choices"]
    assert data["labels"]["pony"] == "Pony"

    res = client.post("/api/variants", json={"path": path, "variant": ""})
    assert res.status_code == 200
    assert se._override_key(Path(path)) not in se.load_variant_overrides()


def test_variants_api_rejects_bad_input(client, clean_variants):
    res = client.post("/api/variants", json={"path": "", "variant": "pony"})
    assert res.status_code == 400
    res = client.post("/api/variants", json={"path": "/tmp/x.safetensors", "variant": "nope"})
    assert res.status_code == 400
