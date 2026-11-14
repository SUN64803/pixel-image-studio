"""Tests for LoRA discovery and family detection."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import sana_engine as se


def _write_st(tmp_path, name, keys):
    path = tmp_path / name
    save_file({k: torch.zeros(1) for k in keys}, str(path))
    return path


def test_lora_family_detection(tmp_path):
    sdxl = _write_st(tmp_path, "xl-style.safetensors", ["lora_te2.foo"])
    assert se._lora_family(sdxl) == "sdxl"

    sd15 = _write_st(tmp_path, "old-lora.safetensors", ["lora_te_text_model.x"])
    assert se._lora_family(sd15) == "sd15"

    unknown = _write_st(tmp_path, "mystery.safetensors", ["whatever.key"])
    assert se._lora_family(unknown) is None


def test_lora_family_invalid_file(tmp_path):
    bad = tmp_path / "broken.safetensors"
    bad.write_bytes(b"garbage")
    assert se._lora_family(bad) is None


def test_discover_loras_scans_and_labels(tmp_path, monkeypatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_st(dir_a, "mylora.safetensors", ["caption_projection.x"])
    # duplicate (same name + size) in another dir must be de-duplicated
    _write_st(dir_b, "mylora.safetensors", ["caption_projection.x"])
    _write_st(dir_a, "universal.safetensors", ["no.idea.what"])

    monkeypatch.setattr(se, "LORA_SCAN_DIRS", [dir_a, dir_b])
    loras = se.discover_loras()

    names = [item["name"] for item in loras]
    assert names.count("mylora") == 1
    by_name = {item["name"]: item for item in loras}
    assert by_name["mylora"]["family"] == "sana"
    assert by_name["mylora"]["family_label"] == "SANA"
    assert by_name["universal"]["family"] is None
    assert by_name["universal"]["family_label"] == "ใช้ได้ทุกโมเดล"
    assert by_name["universal"]["id"].startswith("any-")
    for item in loras:
        assert set(item) == {
            "id",
            "name",
            "path",
            "family",
            "family_label",
            "variant",
            "variant_label",
            "variant_source",
            "size",
            "rank",
            "source",
        }
        assert Path(item["path"]).is_file()


def test_discover_loras_skips_empty_and_corrupt(tmp_path, monkeypatch):
    _write_st(tmp_path, "good.safetensors", ["caption_projection.x"])
    (tmp_path / "empty.safetensors").write_bytes(b"")
    (tmp_path / "junk.safetensors").write_bytes(b"not a safetensors file")
    monkeypatch.setattr(se, "LORA_SCAN_DIRS", [tmp_path])
    names = [item["name"] for item in se.discover_loras()]
    assert names == ["good"]


def test_import_lora_rejects_unsupported_extension(tmp_path):
    src = tmp_path / "legacy.pt"
    src.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="safetensors"):
        se.ModelManager().import_lora(str(src))
    assert not (se.MODELS_ROOT / "loras" / "legacy.pt").exists()


def test_import_lora_rejects_corrupt_safetensors(tmp_path):
    src = tmp_path / "corrupt.safetensors"
    src.write_bytes(b"definitely not safetensors")
    with pytest.raises(ValueError, match="อ่านไฟล์"):
        se.ModelManager().import_lora(str(src))
    assert not (se.MODELS_ROOT / "loras" / "corrupt.safetensors").exists()


def test_lora_id_format(tmp_path):
    path = _write_st(tmp_path, "Sana Test LoRA.safetensors", ["caption_projection.x"])
    monkey_loras = se.discover_loras.__wrapped__ if hasattr(se.discover_loras, "__wrapped__") else None
    items = [{"id": se._slug(path, "sana"), "name": path.stem}]
    lora_id = items[0]["id"]
    # documented format: {family}-{name}-{hash}
    assert lora_id.startswith("sana-sana-test-lora-")
    assert len(lora_id.rsplit("-", 1)[1]) == 8
