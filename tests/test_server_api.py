"""HTTP API tests — queue, validation and hires sanitisation.

`ModelManager.generate` is stubbed with a gate-controlled fake, so no model
ever loads. The gate is OPEN by default (jobs complete instantly); tests that
need to hold the worker clear it and set it again to release.
"""
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import sana_engine as se
import server as srv
import upscaler as upr


@pytest.fixture()
def client(monkeypatch):
    gate = threading.Event()
    gate.set()  # open → jobs drain immediately

    # Tests must not depend on model weights being installed.  Serve a fake
    # SANA + SDXL registry so generate/select/hires paths are exercised even
    # in a clean checkout.
    from pathlib import Path

    root = Path("/tmp/fake-models")
    fake_models = [
        se._model_entry(root / "sana", "sana", "diffusers", root),
        se._model_entry(root / "sdxl.safetensors", "sdxl", "single", root),
    ]
    monkeypatch.setattr(se, "discover_models", lambda: list(fake_models))
    srv.manager.reload_models()

    def fake_generate(self, params, job, cancel):
        job["images"] = []
        gate.wait(timeout=15)
        return []

    monkeypatch.setattr(se.ModelManager, "generate", fake_generate)
    with TestClient(srv.app) as c:
        ns = SimpleNamespace(http=c, gate=gate)
        yield ns
    gate.set()  # never leave the worker blocked between tests
    wait_until(lambda: active(ns) == [], timeout=12)


def wait_until(fn, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return fn()


def submit(c, prompt):
    res = c.http.post("/api/generate", json={"prompt": prompt, "width": 512, "height": 512})
    assert res.status_code == 200, res.text
    return res.json()["job_id"]


def active(c):
    return c.http.get("/api/jobs?active=1").json()["jobs"]


# ── validation ───────────────────────────────────────────────────────────────
def test_generate_requires_prompt(client):
    res = client.http.post("/api/generate", json={"prompt": "   "})
    assert res.status_code == 400


def test_generate_sanitises_values(client):
    res = client.http.post(
        "/api/generate",
        json={"prompt": "x", "steps": 999, "batch": 99, "guidance": -5},
    )
    assert res.status_code == 200
    params = res.json()["job"]["params"]
    assert params["steps"] == 60
    assert params["batch"] == 4
    assert params["guidance"] == 1.0


def test_job_shape(client):
    job_id = submit(client, "shape test")
    assert wait_until(
        lambda: client.http.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    )
    job = client.http.get(f"/api/jobs/{job_id}").json()
    for key in ("id", "status", "phase", "progress", "params", "images", "created", "elapsed"):
        assert key in job


# ── hires sanitisation ──────────────────────────────────────────────────────
def test_hires_passes_through_for_sana(client):
    # SANA supports hires (manual img2img loop) — must NOT be stripped
    entry = srv.manager.get_model(srv.manager.current_id)
    assert entry["family"] == "sana"
    res = client.http.post(
        "/api/generate",
        json={"prompt": "h", "hires": {"enabled": True, "scale": 1.5, "strength": 0.35}},
    )
    job_id = res.json()["job_id"]
    assert res.json()["job"]["params"]["hires"]["enabled"] is True
    assert wait_until(lambda: client.http.get(f"/api/jobs/{job_id}").json()["status"] == "done")


def test_hires_disabled_when_inpainting(client, monkeypatch):
    original = srv.manager.current_id
    try:
        sdxl = next(
            m for m in srv.manager.list_models()["models"] if m["family"] == "sdxl"
        )
        res = client.http.post("/api/models/select", json={"id": sdxl["id"]})
        assert res.status_code == 200
        res = client.http.post(
            "/api/generate",
            json={
                "prompt": "h",
                "inpaint": True,
                "init_image": "/tmp/x.png",
                "mask_image": "/tmp/m.png",
                "hires": {"enabled": True, "scale": 1.5, "strength": 0.35},
            },
        )
        params = res.json()["job"]["params"]
        assert params["inpaint"] is True
        assert params["hires"]["enabled"] is False  # inpaint wins
        job_id = res.json()["job_id"]
        assert wait_until(
            lambda: client.http.get(f"/api/jobs/{job_id}").json()["status"] == "done"
        )
    finally:
        client.http.post("/api/models/select", json={"id": original})


def test_hires_enabled_for_sdxl(client):
    original = srv.manager.current_id
    try:
        sdxl = next(
            m for m in srv.manager.list_models()["models"] if m["family"] == "sdxl"
        )
        res = client.http.post("/api/models/select", json={"id": sdxl["id"]})
        assert res.status_code == 200
        res = client.http.post(
            "/api/generate",
            json={"prompt": "h", "hires": {"enabled": True, "scale": 9, "strength": 2}},
        )
        params = res.json()["job"]["params"]
        assert params["hires"]["enabled"] is True
        assert params["hires"]["scale"] == 2.0
        assert params["hires"]["strength"] == 0.75
        job_id = res.json()["job_id"]
        assert wait_until(
            lambda: client.http.get(f"/api/jobs/{job_id}").json()["status"] == "done"
        )
    finally:
        client.http.post("/api/models/select", json={"id": original})


def test_history_negative_offset_is_clamped(client):
    res = client.http.get("/api/history?offset=-5")
    assert res.status_code == 200
    assert res.json()["offset"] == 0


def test_model_load_rejected_while_job_active(client):
    client.gate.clear()
    jid = submit(client, "load guard")
    assert wait_until(
        lambda: any(j["id"] == jid and j["status"] == "running" for j in active(client))
    )
    try:
        res = client.http.post("/api/model/load")
        assert res.status_code == 409
    finally:
        client.gate.set()
    assert wait_until(lambda: client.http.get(f"/api/jobs/{jid}").json()["status"] == "done")


def test_train_captions_rejects_non_list(client):
    res = client.http.post("/api/train/datasets", json={"name": "cap guard", "trigger": "cap"})
    ds = res.json()["dataset"]
    try:
        bad = client.http.put(
            f"/api/train/datasets/{ds['id']}/captions", json={"updates": "not a list"}
        )
        assert bad.status_code == 400
        ok = client.http.put(
            f"/api/train/datasets/{ds['id']}/captions", json={"updates": [{"file": "x"}]}
        )
        assert ok.status_code == 200
    finally:
        client.http.delete(f"/api/train/datasets/{ds['id']}")


def test_clear_history_rejected_while_job_active(client):
    client.gate.clear()
    jid = submit(client, "clear guard")
    assert wait_until(
        lambda: any(j["id"] == jid and j["status"] == "running" for j in active(client))
    )
    try:
        res = client.http.post("/api/history/clear", json={"delete_files": False})
        assert res.status_code == 409
    finally:
        client.gate.set()
    assert wait_until(lambda: client.http.get(f"/api/jobs/{jid}").json()["status"] == "done")


# ── queue: ordering, move, cancel, cap ──────────────────────────────────────
def test_queue_flow(client):
    client.gate.clear()  # hold the worker so A runs and B/C stay queued
    a = submit(client, "flow A")
    assert wait_until(lambda: any(j["id"] == a and j["status"] == "running" for j in active(client)))

    b = submit(client, "flow B")
    c = submit(client, "flow C")
    ids = [j["id"] for j in active(client)]
    assert ids == [a, b, c]

    # move C up → [A, C, B]
    assert client.http.post(f"/api/jobs/{c}/move?direction=up").status_code == 200
    assert [j["id"] for j in active(client)] == [a, c, b]

    # moving a running job is rejected
    assert client.http.post(f"/api/jobs/{a}/move?direction=up").status_code == 409

    # cancel B → removed from the queue entirely
    assert client.http.post(f"/api/jobs/{b}/cancel").status_code == 200
    assert [j["id"] for j in active(client)] == [a, c]
    assert client.http.get(f"/api/jobs/{b}").json()["status"] == "canceled"

    # moving a canceled (out-of-queue) job is rejected
    assert client.http.post(f"/api/jobs/{b}/move?direction=up").status_code == 409

    # release the worker: A then C complete in order
    client.gate.set()
    assert wait_until(
        lambda: client.http.get(f"/api/jobs/{a}").json()["status"] == "done"
        and client.http.get(f"/api/jobs/{c}").json()["status"] == "done"
    )
    assert wait_until(lambda: active(client) == [])


def test_queue_cap(client):
    client.gate.clear()
    limit = srv.MAX_ACTIVE_JOBS
    ids = [submit(client, f"cap {i}") for i in range(limit)]
    assert len(active(client)) == limit
    res = client.http.post("/api/generate", json={"prompt": "over the cap"})
    assert res.status_code == 429
    client.gate.set()
    for jid in ids:
        client.http.post(f"/api/jobs/{jid}/cancel")
    assert wait_until(lambda: active(client) == [])


def test_train_rejects_no_solo_images(client):
    import sana_trainer as st

    res = client.http.post(
        "/api/train/datasets", json={"name": "multi guard", "trigger": "mg"}
    )
    ds = res.json()["dataset"]
    try:
        datasets = st.load_registry()
        for entry in datasets:
            if entry.get("id") == ds["id"]:
                entry["images"] = [
                    {"file": "a.png", "name": "a", "caption": "", "multi": True}
                ]
        st.save_registry(datasets)

        bad = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={
                "lora_name": "multi-guard-test",
                "multi_mode": "exclude",
            },
        )
        assert bad.status_code == 400
        assert "ตัวละครเดี่ยว" in bad.json()["detail"]
    finally:
        client.http.delete(f"/api/train/datasets/{ds['id']}")


def test_train_rejects_unknown_init_lora(client):
    import sana_trainer as st

    res = client.http.post(
        "/api/train/datasets", json={"name": "resume guard", "trigger": "rg"}
    )
    ds = res.json()["dataset"]
    try:
        datasets = st.load_registry()
        for entry in datasets:
            if entry.get("id") == ds["id"]:
                entry["images"] = [{"file": "a.png", "name": "a", "caption": ""}]
        st.save_registry(datasets)

        bad = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={"lora_name": "resume-test", "init_lora": "does-not-exist"},
        )
        assert bad.status_code == 400
        assert "ไม่พบ LoRA" in bad.json()["detail"]
    finally:
        client.http.delete(f"/api/train/datasets/{ds['id']}")


# ── SDXL trainer routing / validation ───────────────────────────────────────
def _train_dataset_with_image(client, name):
    import sana_trainer as st

    res = client.http.post(
        "/api/train/datasets", json={"name": name, "trigger": "trig"}
    )
    ds = res.json()["dataset"]
    datasets = st.load_registry()
    for entry in datasets:
        if entry.get("id") == ds["id"]:
            entry["images"] = [{"file": "a.png", "name": "a", "caption": ""}]
    st.save_registry(datasets)
    return ds


def test_train_rejects_unknown_model_id(client):
    import sana_trainer as st

    ds = _train_dataset_with_image(client, "model guard")
    try:
        bad = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={"lora_name": "model-guard-test", "model_id": "does-not-exist"},
        )
        assert bad.status_code == 400
        assert "ไม่พบโมเดล" in bad.json()["detail"]
    finally:
        st.delete_dataset(ds["id"])


def test_train_rejects_non_trainable_family(client, monkeypatch):
    import sana_trainer as st
    import server as srv

    ds = _train_dataset_with_image(client, "family guard")
    try:
        monkeypatch.setattr(
            srv.manager,
            "get_model",
            lambda model_id=None: {
                "id": "sd15-x",
                "name": "SD 1.5",
                "family": "sd15",
                "family_label": "SD 1.5",
                "path": "/tmp/x.safetensors",
                "kind": "single",
            },
        )
        bad = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={"lora_name": "family-guard-test", "model_id": "sd15-x"},
        )
        assert bad.status_code == 400
        assert "ยังไม่รองรับ" in bad.json()["detail"]
    finally:
        st.delete_dataset(ds["id"])


def test_train_routes_sdxl_jobs(client, monkeypatch):
    import uuid

    import sana_trainer as st
    import server as srv

    ds = _train_dataset_with_image(client, "sdxl route")
    calls = {}

    def fake_train(params, job, cancel):
        calls.update(params)
        return {"lora": None}

    monkeypatch.setattr(srv.sdxl_trainer, "train", fake_train)
    monkeypatch.setattr(
        srv.manager,
        "get_model",
        lambda model_id=None: {
            "id": "sdxl-x",
            "name": "SDXL Test",
            "family": "sdxl",
            "family_label": "SDXL",
            "path": "/tmp/x.safetensors",
            "kind": "single",
        },
    )
    name = "sdxl-route-" + uuid.uuid4().hex[:8]
    try:
        res = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={"lora_name": name, "model_id": "sdxl-x", "steps": 10},
        )
        assert res.status_code == 200, res.text
        jid = res.json()["job_id"]
        assert wait_until(
            lambda: client.http.get(f"/api/jobs/{jid}").json()["status"]
            in ("done", "error"),
            timeout=8,
        )
        assert calls.get("family") == "sdxl"
        assert calls.get("model_id") == "sdxl-x"
        assert calls.get("lora_name") == name
    finally:
        st.delete_dataset(ds["id"])


def test_train_rejects_init_lora_from_other_family(client, monkeypatch, tmp_path):
    import sana_trainer as st
    import server as srv

    ds = _train_dataset_with_image(client, "init family guard")
    lora_file = tmp_path / "Other.safetensors"
    lora_file.write_bytes(b"x")
    entry = {
        "id": "sdxl-other-abc",
        "name": "Other",
        "path": str(lora_file),
        "family": "sdxl",
        "family_label": "SDXL",
    }
    monkeypatch.setattr(st.se, "discover_loras", lambda: [entry])
    monkeypatch.setattr(
        srv.manager,
        "get_model",
        lambda model_id=None: {
            "id": "sana-x",
            "name": "SANA Test",
            "family": "sana",
            "family_label": "SANA",
            "path": "/tmp/x",
            "kind": "diffusers",
        },
    )
    try:
        bad = client.http.post(
            f"/api/train/datasets/{ds['id']}/train",
            json={
                "lora_name": "init-family-guard",
                "model_id": "sana-x",
                "init_lora": "sdxl-other-abc",
            },
        )
        assert bad.status_code == 400
        assert "ไม่ตรงกับโมเดลฐาน" in bad.json()["detail"]
    finally:
        st.delete_dataset(ds["id"])


# ── Host / Origin guard (loopback-only app) ──────────────────────────────────
def test_guard_rejects_foreign_host(client):
    res = client.http.get("/api/health", headers={"Host": "evil.example.com"})
    assert res.status_code == 403


def test_guard_allows_local_host(client):
    res = client.http.get(
        "/api/health", headers={"Host": "127.0.0.1:47824"}
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_guard_rejects_foreign_origin_post(client):
    res = client.http.post(
        "/api/generate",
        json={"prompt": "hi"},
        headers={"Origin": "https://evil.example.com"},
    )
    assert res.status_code == 403
    res = client.http.post(
        "/api/generate",
        json={"prompt": "guard-origin-local"},
        headers={"Origin": "http://127.0.0.1:47824"},
    )
    assert res.status_code == 200
    assert wait_until(lambda: active(client) == [])


# ── ESRGAN upscale ──────────────────────────────────────────────────────────
def _fake_upscale(self, image, scale=4, tile=None, on_progress=None, cancel=None):
    if on_progress:
        on_progress(1, 1)
    return image.resize((image.width * scale, image.height * scale), Image.LANCZOS)


def _make_source(name="upscale-src.png", size=(8, 6)):
    se.IMAGES.mkdir(parents=True, exist_ok=True)
    path = se.IMAGES / name
    Image.new("RGB", size, (20, 40, 60)).save(path)
    return path


def test_upscale_info(client):
    res = client.http.get("/api/upscale")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["scale"] == 4


def test_upscale_missing_source(client):
    res = client.http.post("/api/upscale", json={"file": "nope.png"})
    assert res.status_code == 404


def test_upscale_rejects_path_traversal(client):
    res = client.http.post("/api/upscale", json={"file": "../history.json"})
    assert res.status_code == 404


def test_upscale_job_saves_new_image(client, monkeypatch):
    src = _make_source()
    monkeypatch.setattr(upr.RealESRGANUpscaler, "upscale", _fake_upscale)
    res = client.http.post("/api/upscale", json={"file": src.name, "scale": 4})
    assert res.status_code == 200, res.text
    jid = res.json()["job_id"]
    assert wait_until(
        lambda: client.http.get(f"/api/jobs/{jid}").json()["status"]
        in ("done", "error")
    )
    job = client.http.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "done", job
    assert job["family"] == "upscale"
    assert job["images"][0]["width"] == 32
    assert job["images"][0]["height"] == 24
    hist = client.http.get("/api/history?offset=0&limit=20").json()
    assert any(r["id"] == jid for r in hist["items"])


def test_upscale_tile_defaults_to_auto(client, monkeypatch):
    src = _make_source("upscale-auto.png")
    monkeypatch.setattr(upr.RealESRGANUpscaler, "upscale", _fake_upscale)
    res = client.http.post("/api/upscale", json={"file": src.name, "scale": 4})
    assert res.status_code == 200, res.text
    jid = res.json()["job_id"]
    assert wait_until(
        lambda: client.http.get(f"/api/jobs/{jid}").json()["status"]
        in ("done", "error")
    )
    job = client.http.get(f"/api/jobs/{jid}").json()
    assert "tile" not in job["params"]  # left to pick_tile()


def test_upscale_explicit_tile_is_clamped(client, monkeypatch):
    src = _make_source("upscale-tile.png")
    monkeypatch.setattr(upr.RealESRGANUpscaler, "upscale", _fake_upscale)
    res = client.http.post(
        "/api/upscale", json={"file": src.name, "scale": 4, "tile": 64}
    )
    assert res.status_code == 200, res.text
    jid = res.json()["job_id"]
    assert wait_until(
        lambda: client.http.get(f"/api/jobs/{jid}").json()["status"]
        in ("done", "error")
    )
    job = client.http.get(f"/api/jobs/{jid}").json()
    assert job["params"]["tile"] == 128
