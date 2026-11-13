"""Backup/restore API tests.

The endpoints work on the engine's data dir; each test points srv.DATA (and the
presets/dataset files) at a temp directory, so the real data is never touched.
"""
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import prompt_presets as ppres
import sana_trainer as strn
import server as srv


@pytest.fixture()
def client():
    with TestClient(srv.app) as c:
        yield c


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for sub in ("images", "thumbs", "train"):
        (data / sub).mkdir(parents=True)
    monkeypatch.setattr(srv, "DATA", data)
    monkeypatch.setattr(srv, "HISTORY_FILE", data / "history.json")
    monkeypatch.setattr(srv, "HISTORY", [])
    monkeypatch.setattr(ppres, "PRESETS_FILE", data / "prompt_presets.json")
    monkeypatch.setattr(strn, "TRAIN_DIR", data / "train")
    monkeypatch.setattr(strn, "REGISTRY_FILE", data / "train_datasets.json")
    return data


def _seed(data: Path) -> None:
    (data / "images" / "a.png").write_bytes(b"PNG-A")
    (data / "thumbs" / "a.webp").write_bytes(b"WEBP-A")
    (data / "train" / "ds1").mkdir()
    (data / "train" / "ds1" / "1.png").write_bytes(b"TRAIN-1")
    (data / "history.json").write_text(
        json.dumps([{"id": "rec-1", "images": [{"file": "a.png"}]}]), encoding="utf-8"
    )
    (data / "prompt_presets.json").write_text(
        json.dumps([{"name": "preset-a", "prompt": "x"}]), encoding="utf-8"
    )
    (data / "train_datasets.json").write_text(
        json.dumps([{"id": "ds1", "name": "Dataset 1", "images": []}]), encoding="utf-8"
    )


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return buf.getvalue()


def _manifest() -> bytes:
    return json.dumps({"format": 1, "app_version": "test"}).encode()


def test_export_contains_user_data(client, data_dir):
    _seed(data_dir)
    res = client.post("/api/backup/export")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["files"] >= 6
    target = Path(body["path"])
    assert target.is_file() and target.stat().st_size == body["size"]
    with zipfile.ZipFile(target) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == srv.BACKUP_FORMAT
    assert {
        "history.json",
        "prompt_presets.json",
        "train_datasets.json",
        "images/a.png",
        "thumbs/a.webp",
        "train/ds1/1.png",
    } <= names


def test_restore_is_additive_and_keeps_local_files(client, data_dir):
    _seed(data_dir)
    backup = _zip_bytes(
        {
            "manifest.json": _manifest(),
            "history.json": json.dumps(
                [
                    {"id": "rec-1", "images": [{"file": "a.png"}]},
                    {"id": "rec-2", "images": [{"file": "b.png"}]},
                ]
            ).encode(),
            "prompt_presets.json": json.dumps(
                [
                    {"name": "preset-a", "prompt": "local"},
                    {"name": "preset-b", "prompt": "new"},
                ]
            ).encode(),
            "train_datasets.json": json.dumps(
                [
                    {"id": "ds1", "name": "Dataset 1", "images": []},
                    {"id": "ds2", "name": "Dataset 2", "images": []},
                ]
            ).encode(),
            "images/b.png": b"PNG-B",
            "images/a.png": b"SHOULD-NOT-OVERWRITE",
            "thumbs/b.webp": b"WEBP-B",
            "train/ds2/1.png": b"TRAIN-2",
        }
    )
    res = client.post(
        "/api/backup/restore",
        files={"file": ("backup.zip", backup, "application/zip")},
    )
    assert res.status_code == 200, res.text
    restored = res.json()["restored"]
    assert restored["history"] == 1
    assert restored["presets"] == 1
    assert restored["datasets"] == 1
    assert restored["images"] == 1
    assert restored["thumbs"] == 1
    assert restored["train"] == 1

    # existing data is untouched, new data lands next to it
    assert (data_dir / "images" / "a.png").read_bytes() == b"PNG-A"
    assert (data_dir / "images" / "b.png").read_bytes() == b"PNG-B"
    assert (data_dir / "train" / "ds2" / "1.png").read_bytes() == b"TRAIN-2"
    history = json.loads((data_dir / "history.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in history] == ["rec-1", "rec-2"]
    presets = json.loads((data_dir / "prompt_presets.json").read_text(encoding="utf-8"))
    assert presets == [
        {"name": "preset-a", "prompt": "x"},
        {"name": "preset-b", "prompt": "new"},
    ]
    datasets = json.loads((data_dir / "train_datasets.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in datasets] == ["ds1", "ds2"]


def test_restore_rejects_plain_zip(client, data_dir):
    backup = _zip_bytes({"history.json": b"[]"})
    res = client.post(
        "/api/backup/restore",
        files={"file": ("backup.zip", backup, "application/zip")},
    )
    assert res.status_code == 400
    assert "manifest" in res.json()["detail"]


def test_restore_rejects_unknown_format(client, data_dir):
    backup = _zip_bytes({"manifest.json": json.dumps({"format": 99}).encode()})
    res = client.post(
        "/api/backup/restore",
        files={"file": ("backup.zip", backup, "application/zip")},
    )
    assert res.status_code == 400
    assert "เวอร์ชัน" in res.json()["detail"]


def test_restore_rejects_path_traversal(client, data_dir):
    backup = _zip_bytes(
        {
            "manifest.json": _manifest(),
            "../evil.txt": b"nope",
        }
    )
    res = client.post(
        "/api/backup/restore",
        files={"file": ("backup.zip", backup, "application/zip")},
    )
    assert res.status_code == 400
    assert "ไม่ปลอดภัย" in res.json()["detail"]
    assert not (data_dir.parent / "evil.txt").exists()


def test_restore_refuses_while_jobs_run(client, data_dir, monkeypatch):
    monkeypatch.setattr(srv, "ordered_active_jobs", lambda: [{"id": "job-1"}])
    backup = _zip_bytes({"manifest.json": _manifest()})
    res = client.post(
        "/api/backup/restore",
        files={"file": ("backup.zip", backup, "application/zip")},
    )
    assert res.status_code == 409
