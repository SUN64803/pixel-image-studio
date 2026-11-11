"""Assistant P1 tests — rule-based parser, angle helpers, tools/confirm/undo.

No model ever loads: `api_generate` and the model/LoRA manager methods are
stubbed where a test needs them. Dataset files live in the isolated
IMGS_DATA temp dir created by conftest.
"""
import time

import pytest
from fastapi.testclient import TestClient

import assistant as ast
import assistant_vision as astvis
import sana_trainer as st
import server as srv


@pytest.fixture(autouse=True)
def _fresh_stores(monkeypatch):
    # Tests must never call the real Ollama — the rule-based brain is the
    # deterministic path they assert on. Vision capabilities are stubbed too
    # (individual vision tests opt in by patching them again).
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "rules")
    monkeypatch.setattr(astvis, "capabilities", lambda: [])
    srv.ASSIST_PENDING.clear()
    srv.ASSIST_UNDO.clear()
    yield
    srv.ASSIST_PENDING.clear()
    srv.ASSIST_UNDO.clear()


@pytest.fixture()
def client():
    with TestClient(srv.app) as c:
        yield c


@pytest.fixture()
def dataset():
    created = st.create_dataset("assistant-test", "my character", "character")
    ds_id = created["id"]
    yield ds_id
    st.delete_dataset(ds_id)


def _set_images(ds_id, images):
    datasets = st.load_registry()
    entry = next(d for d in datasets if d["id"] == ds_id)
    entry["images"] = images
    st.save_registry(datasets)
    return entry


def _images():
    return [
        {"file": "aaaaaaaaaaaa.png", "caption": "A view from behind the character.", "multi": False},
        {"file": "bbbbbbbbbbbb.png", "caption": "turns around to flash a peace sign.", "multi": False},
        {"file": "cccccccccccc.png", "caption": "with a man in the room", "multi": True},
    ]


# ── parser ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "message,tool",
    [
        ("ตรวจ dataset", "dataset_summary"),
        ("audit หน่อย", "dataset_summary"),
        ("มี dataset อะไรบ้าง", "list_datasets"),
        ("มี LoRA อะไรบ้าง", "list_loras"),
        ("มีโมเดลอะไรบ้าง", "list_models"),
        ("สถานะ GPU", "gpu_status"),
        ("มีกี่ภาพแล้ว", "history_stats"),
        ("คิวมีอะไร", "list_jobs"),
        ("เติม tag มุม", "fill_angle_tags"),
        ("ล้างประวัติ", "clear_history"),
        ("ลบภาพล่าสุด", "delete_history"),
        ("ลบ lora old-test", "delete_lora"),
        ("เริ่มเทรนชื่อ my-lora-v7", "start_training"),
        ("เลือกโมเดล sana", "select_model"),
        ("สร้างภาพ my character standing", "generate_image"),
        ("ช่วยอะไรได้บ้าง", "help"),
    ],
)
def test_parse_intent_maps_tools(message, tool):
    assert ast.parse_intent(message)["tool"] == tool


def test_parse_intent_extracts_args():
    intent = ast.parse_intent("เริ่มเทรนชื่อ my-lora-v7 steps 800 rank 16")
    assert intent["tool"] == "start_training"
    assert intent["args"]["name"] == "my-lora-v7"
    assert intent["args"]["steps"] == 800
    assert intent["args"]["rank"] == 16

    intent = ast.parse_intent("เปลี่ยน trigger เป็น my character")
    assert intent["tool"] == "set_trigger"
    assert intent["args"]["trigger"] == "my character"

    intent = ast.parse_intent("caption aaaaaaaaaaaa เป็น back view, standing")
    assert intent["tool"] == "set_caption"
    assert intent["args"] == {"file": "aaaaaaaaaaaa", "caption": "back view, standing"}

    intent = ast.parse_intent("สร้างภาพ a cat on a sofa")
    assert intent["args"]["prompt"] == "a cat on a sofa"


def test_parse_intent_clarifies_missing_args():
    assert ast.parse_intent("เริ่มเทรน")["clarify"]
    assert ast.parse_intent("แก้ caption")["clarify"]
    assert ast.parse_intent("")["clarify"]
    unknown = ast.parse_intent("สวัสดีตอนเช้า")
    assert unknown["tool"] is None and unknown["clarify"]


# ── angle helpers ────────────────────────────────────────────────────────────
def test_angle_helpers_are_strict():
    loose = "A view from behind the character."
    assert ast.has_angle_tag(loose) is False
    assert ast.missing_angle_tags(loose) == ["back view"]

    tagged = "back view, A view from behind the character."
    assert ast.has_angle_tag(tagged) is True
    assert ast.missing_angle_tags(tagged) == []

    assert ast.with_angle_tags("sitting", ["front view"]) == "front view, sitting"
    assert ast.with_angle_tags("", ["front view"]) == "front view"
    assert ast.with_angle_tags("sitting", []) == "sitting"


# ── API: read tools ──────────────────────────────────────────────────────────
def test_tools_endpoint_lists_registry(client):
    res = client.get("/api/assistant/tools")
    assert res.status_code == 200
    body = res.json()
    names = {tool["name"] for tool in body["tools"]}
    assert {"dataset_summary", "fill_angle_tags", "start_training", "help"} <= names
    assert set(body["tiers"]) == {"read", "write", "destructive"}
    assert body["suggestions"]


def test_chat_help(client):
    res = client.post("/api/assistant/chat", json={"message": "ช่วยอะไรได้บ้าง"})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "rule-based" in body["reply"]
    assert body["actions"] == [] and body["pending"] == []


def test_chat_dataset_summary(client, dataset):
    _set_images(dataset, _images())
    res = client.post(
        "/api/assistant/chat",
        json={"message": "ตรวจ dataset", "dataset_id": dataset},
    )
    body = res.json()
    assert body["ok"] is True
    result = body["actions"][0]["result"]
    assert result["total"] == 3 and result["solo"] == 2 and result["multi"] == 1
    assert result["missing_tags"] == 2
    assert result["fillable"] == 1
    assert result["manual"] == 1
    assert result["ready"] is False
    assert "tag มุมขาด 2" in body["reply"]


def test_chat_unknown_message_returns_clarify(client):
    body = client.post("/api/assistant/chat", json={"message": "สวัสดี"}).json()
    assert body["ok"] is True
    assert body["actions"] == []
    assert "ไม่เข้าใจ" in body["reply"]


# ── API: write tools + undo ──────────────────────────────────────────────────
def test_fill_angle_tags_and_undo(client, dataset):
    _set_images(dataset, _images())
    body = client.post(
        "/api/assistant/chat",
        json={"message": "เติม tag มุม", "dataset_id": dataset},
    ).json()
    action = body["actions"][0]
    assert action["tool"] == "fill_angle_tags"
    assert action["result"]["filled"] == 1
    assert action["undo_id"]

    entry = st.get_dataset(dataset)
    captions = {i["file"]: i["caption"] for i in entry["images"]}
    assert captions["aaaaaaaaaaaa.png"].startswith("back view,")
    assert captions["bbbbbbbbbbbb.png"] == "turns around to flash a peace sign."
    assert captions["cccccccccccc.png"] == "with a man in the room"  # multi untouched

    undo = client.post("/api/assistant/undo", json={"id": action["undo_id"]})
    assert undo.status_code == 200
    entry = st.get_dataset(dataset)
    captions = {i["file"]: i["caption"] for i in entry["images"]}
    assert captions["aaaaaaaaaaaa.png"] == "A view from behind the character."


def test_fill_angle_tags_noop(client, dataset):
    _set_images(dataset, [{"file": "dddddddddddd.png", "caption": "no angle here", "multi": False}])
    body = client.post(
        "/api/assistant/chat",
        json={"message": "เติม tag มุม", "dataset_id": dataset},
    ).json()
    assert body["actions"][0]["result"]["filled"] == 0
    assert body["actions"][0]["undo_id"] is None


def test_set_trigger_and_undo(client, dataset):
    body = client.post(
        "/api/assistant/chat",
        json={"message": "เปลี่ยน trigger เป็น mytrigger7", "dataset_id": dataset},
    ).json()
    assert body["actions"][0]["result"]["new"] == "mytrigger7"
    assert st.get_dataset(dataset)["trigger"] == "mytrigger7"
    client.post("/api/assistant/undo", json={"id": body["actions"][0]["undo_id"]})
    assert st.get_dataset(dataset)["trigger"] == "my character"


def test_set_caption_and_undo(client, dataset):
    _set_images(dataset, _images())
    body = client.post(
        "/api/assistant/chat",
        json={
            "message": "caption aaaaaaaaaaaa เป็น front view, standing",
            "dataset_id": dataset,
        },
    ).json()
    assert body["actions"][0]["result"]["file"] == "aaaaaaaaaaaa.png"
    entry = st.get_dataset(dataset)
    assert {i["file"]: i["caption"] for i in entry["images"]}["aaaaaaaaaaaa.png"] == (
        "front view, standing"
    )
    client.post("/api/assistant/undo", json={"id": body["actions"][0]["undo_id"]})
    entry = st.get_dataset(dataset)
    assert {i["file"]: i["caption"] for i in entry["images"]}["aaaaaaaaaaaa.png"] == (
        "A view from behind the character."
    )


def test_select_model_and_undo(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        srv.manager, "list_models", lambda: {"models": [{"id": "sana-a", "name": "SANA A"}], "current": "sana-a"}
    )
    monkeypatch.setattr(srv.manager, "select_model", lambda mid: calls.append(mid) or {"id": mid})
    monkeypatch.setattr(srv.manager, "current_id", "sana-a")
    body = client.post(
        "/api/assistant/chat", json={"message": "เลือกโมเดล sana"}
    ).json()
    assert body["actions"][0]["result"]["new"] == "sana-a"
    assert calls == ["sana-a"]
    client.post("/api/assistant/undo", json={"id": body["actions"][0]["undo_id"]})
    assert calls == ["sana-a", "sana-a"]


def test_generate_image_queues_job(client, monkeypatch):
    seen = {}

    def fake_generate(req):
        seen["prompt"] = req.prompt
        return {"ok": True, "job_id": "job123", "job": {}}

    monkeypatch.setattr(srv, "api_generate", fake_generate)
    body = client.post(
        "/api/assistant/chat", json={"message": "สร้างภาพ my character in a garden"}
    ).json()
    assert seen["prompt"] == "my character in a garden"
    assert body["actions"][0]["result"]["job_id"] == "job123"
    assert body["actions"][0]["undo_id"] is None


# ── API: destructive tools need confirmation ────────────────────────────────
def test_delete_history_requires_confirmation(client):
    record = {"id": "deadbeef", "images": [], "created": "now", "model": "test"}
    srv.HISTORY.append(record)
    try:
        body = client.post("/api/assistant/chat", json={"message": "ลบภาพล่าสุด"}).json()
        assert body["actions"] == []
        assert len(body["pending"]) == 1
        card = body["pending"][0]
        assert card["tier"] == "destructive"
        assert card["tool"] == "delete_history"
        assert any(r.get("id") == "deadbeef" for r in srv.HISTORY)

        confirmed = client.post("/api/assistant/confirm", json={"id": card["id"]})
        assert confirmed.status_code == 200
        assert confirmed.json()["action"]["result"]["removed"] == 1
        assert not any(r.get("id") == "deadbeef" for r in srv.HISTORY)
    finally:
        srv.HISTORY[:] = [r for r in srv.HISTORY if r.get("id") != "deadbeef"]


def test_cancel_pending_does_nothing(client):
    record = {"id": "cafebabe", "images": [], "created": "now", "model": "test"}
    srv.HISTORY.append(record)
    try:
        card = client.post("/api/assistant/chat", json={"message": "ลบภาพล่าสุด"}).json()[
            "pending"
        ][0]
        res = client.post("/api/assistant/cancel", json={"id": card["id"]})
        assert res.json()["cancelled"] is True
        assert any(r.get("id") == "cafebabe" for r in srv.HISTORY)
        # the card is gone: confirming it now fails
        assert client.post("/api/assistant/confirm", json={"id": card["id"]}).status_code == 404
    finally:
        srv.HISTORY[:] = [r for r in srv.HISTORY if r.get("id") != "cafebabe"]


def test_start_training_requires_name_then_confirmation(client, dataset):
    body = client.post(
        "/api/assistant/chat", json={"message": "เริ่มเทรน", "dataset_id": dataset}
    ).json()
    assert body["pending"] == []
    assert "ชื่อ LoRA" in body["reply"]

    body = client.post(
        "/api/assistant/chat",
        json={"message": "เริ่มเทรนชื่อ my-lora-v7 steps 800", "dataset_id": dataset},
    ).json()
    assert len(body["pending"]) == 1
    card = body["pending"][0]
    assert card["tool"] == "start_training"
    assert "my-lora-v7" in card["summary"]
    client.post("/api/assistant/cancel", json={"id": card["id"]})


def test_delete_lora_confirm_uses_manager(client, monkeypatch):
    monkeypatch.setattr(
        srv.manager,
        "list_loras",
        lambda: {"loras": [{"id": "old-test", "name": "old-test", "path": "/tmp/old-test.safetensors"}], "current": None},
    )
    deleted = []
    monkeypatch.setattr(srv.manager, "delete_lora", lambda lid: deleted.append(lid))
    body = client.post("/api/assistant/chat", json={"message": "ลบ lora old-test"}).json()
    card = body["pending"][0]
    res = client.post("/api/assistant/confirm", json={"id": card["id"]})
    assert res.status_code == 200
    assert deleted == ["old-test"]


# ── API: store edge cases ────────────────────────────────────────────────────
def test_confirm_and_undo_expire(client):
    assert client.post("/api/assistant/confirm", json={"id": "nope"}).status_code == 404
    assert client.post("/api/assistant/undo", json={"id": "nope"}).status_code == 404
    state = client.get("/api/assistant/state").json()
    assert state["pending"] == 0 and state["undo"] == 0
    assert state["brain"]["mode"] == "rules"


# ── P2: local LLM brain (Ollama) ─────────────────────────────────────────────
def test_validate_intent_filters_unknown_args():
    intent = ast.validate_intent(
        {"tool": "start_training", "args": {"name": "my-lora-v7", "steps": "800", "bogus": "x"}}
    )
    assert intent == {
        "tool": "start_training",
        "args": {"name": "my-lora-v7", "steps": 800},
        "clarify": None,
        "brain": "llm",
    }


def test_validate_intent_requires_args():
    intent = ast.validate_intent({"tool": "set_trigger", "args": {}, "clarify": ""})
    assert intent["tool"] == "set_trigger"
    assert intent["clarify"]

    intent = ast.validate_intent({"tool": "set_caption", "args": {"file": "abc"}, "clarify": ""})
    assert intent["clarify"]

    # complete call: the model's chatter is ignored
    intent = ast.validate_intent(
        {"tool": "fill_angle_tags", "args": {}, "clarify": "กรุณาระบุ caption"}
    )
    assert intent["clarify"] is None


def test_validate_intent_unknown_tool_and_null():
    assert ast.validate_intent({"tool": "rm -rf", "args": {}})["tool"] is None
    unknown = ast.validate_intent({"tool": None, "args": {}, "clarify": "ช่วยอะไรดี"})
    assert unknown["tool"] is None and unknown["clarify"] == "ช่วยอะไรดี"
    assert ast.validate_intent("not a dict")["tool"] is None


def test_parse_intent_uses_llm_when_enabled(monkeypatch):
    import assistant_llm as llm

    monkeypatch.setattr(
        llm,
        "raw_intent",
        lambda message, lang="th": {"tool": "list_loras", "args": {}, "clarify": ""},
    )
    intent = ast.parse_intent("ขอดูโลร่าหน่อย", brain="auto")
    assert intent["tool"] == "list_loras"
    assert intent["brain"] == "llm"
    # UI language is forwarded to the LLM
    seen = {}
    monkeypatch.setattr(
        llm,
        "raw_intent",
        lambda message, lang="th": seen.setdefault("lang", lang)
        or {"tool": "list_loras", "args": {}, "clarify": ""},
    )
    ast.parse_intent("list loras", brain="auto", lang="en")
    assert seen["lang"] == "en"


def test_parse_intent_falls_back_to_rules_on_llm_error(monkeypatch):
    import assistant_llm as llm

    def boom(message):
        raise llm.LLMError("offline")

    monkeypatch.setattr(llm, "raw_intent", boom)
    intent = ast.parse_intent("ตรวจ dataset", brain="auto")
    assert intent["tool"] == "dataset_summary"
    assert intent["brain"] == "rules"


def test_parse_intent_rules_mode_never_calls_llm(monkeypatch):
    import assistant_llm as llm

    calls = []
    monkeypatch.setattr(llm, "raw_intent", lambda message: calls.append(message) or {})
    assert ast.parse_intent("มี LoRA อะไรบ้าง", brain="rules")["tool"] == "list_loras"
    assert calls == []


def test_llm_status_and_model_name(monkeypatch):
    import assistant_llm as llm

    monkeypatch.setenv("IMGS_ASSIST_MODEL", "qwen3.5:4b-q8_0")
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "auto")
    monkeypatch.setattr(llm, "installed_models", lambda: ["qwen3.5:4b-q8_0", "other:1b"])
    status = llm.status()
    assert status["available"] is True
    assert status["model"] == "qwen3.5:4b-q8_0"
    assert "local" in status["label"]

    monkeypatch.setattr(llm, "installed_models", lambda: [])
    assert llm.status()["available"] is False
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "rules")
    assert llm.status()["label"].startswith("กฎ")


def test_llm_extract_json_variants():
    import assistant_llm as llm

    assert llm._extract_json('{"tool": "help"}')["tool"] == "help"
    assert llm._extract_json('```json\n{"tool": "help"}\n```')["tool"] == "help"
    assert llm._extract_json('ครับ {"tool": "help"} จบ')["tool"] == "help"
    with pytest.raises(llm.LLMError):
        llm._extract_json("no json here")


def test_chat_reports_rules_brain(client):
    body = client.post("/api/assistant/chat", json={"message": "ตรวจ dataset"}).json()
    assert body["brain"] == "rules"


def test_tools_endpoint_includes_brain(client, monkeypatch):
    import assistant_llm as llm

    monkeypatch.setattr(llm, "installed_models", lambda: ["qwen3.5:4b-q8_0"])
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "auto")
    body = client.get("/api/assistant/tools").json()
    assert body["brain"]["model"] == "qwen3.5:4b-q8_0"
    assert body["brain"]["available"] is True


# ── P2: one pipeline at a time (10 GB card) ─────────────────────────────────
def test_brain_mode_uses_rules_while_gpu_busy(monkeypatch):
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "auto")
    monkeypatch.setattr(srv.manager, "state", "ready")
    monkeypatch.setattr(srv.manager, "pipe", object(), raising=False)
    monkeypatch.setattr(srv, "ACTIVE_JOB", {"id": "running"})
    assert srv._assist_brain_mode() == "rules"


def test_brain_mode_unloads_sana_when_idle(monkeypatch):
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "auto")
    calls = []
    monkeypatch.setattr(srv.manager, "state", "ready")
    monkeypatch.setattr(srv.manager, "pipe", object(), raising=False)
    monkeypatch.setattr(srv.manager, "unload", lambda: calls.append("unload"))
    monkeypatch.setattr(srv, "ACTIVE_JOB", None)
    assert srv._assist_brain_mode() == "auto"
    assert calls == ["unload"]


def test_brain_mode_respects_rules_setting(monkeypatch):
    monkeypatch.setenv("IMGS_ASSIST_BRAIN", "rules")
    monkeypatch.setattr(srv.manager, "unload", lambda: pytest.fail("must not unload"))
    assert srv._assist_brain_mode() == "rules"


def test_free_vram_never_raises(monkeypatch):
    import assistant_llm as llm

    calls = []
    monkeypatch.setattr(llm, "unload", lambda: calls.append("x"))
    srv._assist_free_vram()
    assert calls == ["x"]

    def boom():
        raise RuntimeError("ollama down")

    monkeypatch.setattr(llm, "unload", boom)
    srv._assist_free_vram()  # swallowed on purpose


# ── P3: local vision reading ─────────────────────────────────────────────────
def _write_images(ds_id, names):
    folder = st.dataset_dir_path(ds_id)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"not-a-real-image")


def _fake_analyze(angle="back view", multi=False):
    def analyze(path):
        return {"angle": angle, "multi": multi, "caption": "a test caption"}

    return analyze


def _run_vision(client, dataset, monkeypatch, angle="back view"):
    images = _images()
    _set_images(dataset, images)
    _write_images(dataset, [i["file"] for i in images])
    monkeypatch.setattr(astvis, "capabilities", lambda: ["vision"])
    monkeypatch.setattr(astvis, "analyze", _fake_analyze(angle))
    res = client.post("/api/assistant/vision", json={"dataset_id": dataset})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    job = None
    for _ in range(250):
        job = client.get(f"/api/assistant/vision/{job_id}").json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    for _ in range(250):
        if not astvis.RUNNER.running():
            break
        time.sleep(0.02)
    assert job and job["status"] == "done"
    return job_id, job


def test_read_images_tool_registered_and_parsed():
    tool = ast.TOOL_INDEX["read_images"]
    assert tool["tier"] == "read"
    assert ast.parse_intent("อ่านรูปใน dataset")["tool"] == "read_images"
    assert ast.parse_intent("วิเคราะห์รูปหน่อย")["tool"] == "read_images"
    assert ast.parse_intent("read images")["tool"] == "read_images"


def test_vision_start_poll_done(client, dataset, monkeypatch):
    job_id, job = _run_vision(client, dataset, monkeypatch)
    assert job["id"] == job_id
    assert job["total"] == 3 and job["done"] == 3
    first = job["results"][0]
    assert first["angle"] == "back view"
    assert first["angle_added"] == "back view"
    assert first["suggested_caption"].startswith("back view, ")
    assert job["results"][2]["multi_old"] is True
    assert job["results"][2]["multi_changed"] is True
    assert job["results"][2]["multi"] is False  # advisory only, never applied


def test_vision_apply_selected_files_then_undo(client, dataset, monkeypatch):
    job_id, job = _run_vision(client, dataset, monkeypatch)
    files = [r["file"] for r in job["results"] if r["angle_added"]]
    assert len(files) == 3
    res = client.post(
        f"/api/assistant/vision/{job_id}/apply", json={"files": files[:1]}
    )
    body = res.json()
    assert body["updated"] == 1 and body["undo_id"]
    ds = st.get_dataset(dataset)
    target = next(i for i in ds["images"] if i["file"] == files[0])
    assert target["caption"].startswith("back view, ")
    untouched = next(i for i in ds["images"] if i["file"] == files[1])
    assert not ast.has_angle_tag(untouched["caption"])

    undo = client.post("/api/assistant/undo", json={"id": body["undo_id"]})
    assert undo.status_code == 200
    ds = st.get_dataset(dataset)
    target = next(i for i in ds["images"] if i["file"] == files[0])
    assert not ast.has_angle_tag(target["caption"])


def test_vision_apply_skips_existing_angle_tags(client, dataset, monkeypatch):
    images = _images()
    images[0]["caption"] = "back view, A view from behind the character."
    _set_images(dataset, images)
    _write_images(dataset, [i["file"] for i in images])
    monkeypatch.setattr(astvis, "capabilities", lambda: ["vision"])
    monkeypatch.setattr(astvis, "analyze", _fake_analyze())
    job_id = client.post("/api/assistant/vision", json={"dataset_id": dataset}).json()["job_id"]
    for _ in range(250):
        job = client.get(f"/api/assistant/vision/{job_id}").json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["results"][0]["angle_added"] is None
    body = client.post(f"/api/assistant/vision/{job_id}/apply", json={}).json()
    assert body["updated"] == 2


def test_vision_cancel(client, dataset, monkeypatch):
    names = [f"{i:012x}.png" for i in range(6)]
    _set_images(dataset, [{"file": n, "caption": "", "multi": False} for n in names])
    _write_images(dataset, names)
    monkeypatch.setattr(astvis, "capabilities", lambda: ["vision"])

    def slow(path):
        time.sleep(0.05)
        return {"angle": "front view", "multi": False, "caption": "x"}

    monkeypatch.setattr(astvis, "analyze", slow)
    res = client.post("/api/assistant/vision", json={"dataset_id": dataset})
    job_id = res.json()["job_id"]
    cancel = client.post(f"/api/assistant/vision/{job_id}/cancel")
    assert cancel.status_code == 200 and cancel.json()["cancelled"] is True
    job = None
    for _ in range(250):
        job = client.get(f"/api/assistant/vision/{job_id}").json()["job"]
        if job["status"] in ("cancelled", "done"):
            break
        time.sleep(0.02)
    for _ in range(250):
        if not astvis.RUNNER.running():
            break
        time.sleep(0.02)
    assert job["status"] == "cancelled"
    assert job["done"] < job["total"]


def test_vision_refuses_when_gpu_busy(client, dataset, monkeypatch):
    _set_images(dataset, _images())
    monkeypatch.setattr(astvis, "capabilities", lambda: ["vision"])
    monkeypatch.setattr(astvis.RUNNER, "running", lambda: False)
    monkeypatch.setattr(srv, "ACTIVE_JOB", {"id": "busy"})
    res = client.post("/api/assistant/vision", json={"dataset_id": dataset})
    assert res.status_code == 400
    assert "GPU" in res.json()["detail"]


def test_vision_refuses_without_vision_model(client, dataset, monkeypatch):
    _set_images(dataset, _images())
    monkeypatch.setattr(astvis, "capabilities", lambda: ["completion"])
    res = client.post("/api/assistant/vision", json={"dataset_id": dataset})
    assert res.status_code == 400
    assert "vision" in res.json()["detail"]


def test_vision_unknown_job(client):
    assert client.get("/api/assistant/vision/nope").status_code == 404
    assert client.post("/api/assistant/vision/nope/apply", json={}).status_code == 404
    assert client.post("/api/assistant/vision/nope/cancel").status_code == 404


def test_chat_read_images_starts_job(client, dataset, monkeypatch):
    _set_images(dataset, _images())
    _write_images(dataset, [i["file"] for i in _images()])
    monkeypatch.setattr(astvis, "capabilities", lambda: ["vision"])
    monkeypatch.setattr(astvis, "analyze", _fake_analyze())
    body = client.post(
        "/api/assistant/chat",
        json={"message": "อ่านรูปใน dataset", "dataset_id": dataset},
    ).json()
    assert body["brain"] == "rules"
    action = body["actions"][0]
    assert action["tool"] == "read_images"
    assert action["result"]["job_id"]
    assert action["result"]["total"] == 3
    assert "vision" in body["reply"]
    job_id = action["result"]["job_id"]
    for _ in range(250):
        job = client.get(f"/api/assistant/vision/{job_id}").json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    for _ in range(250):
        if not astvis.RUNNER.running():
            break
        time.sleep(0.02)


# ── assistant model picker ──────────────────────────────────────────────────
import assistant_llm as astllm  # noqa: E402


@pytest.fixture()
def model_config(tmp_path, monkeypatch):
    """Isolated config file for the assistant model choice."""
    monkeypatch.setenv("IMGS_DATA", str(tmp_path))
    astllm._invalidate_caps()
    yield tmp_path / "assistant_config.json"
    astllm._invalidate_caps()
    astllm.invalidate_models()


def test_model_choice_persists_and_resets(model_config, monkeypatch):
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    assert astllm.model_name() == astllm.default_model()
    assert astllm.set_model("qwen3.5:4b-bf16") == "qwen3.5:4b-bf16"
    assert model_config.exists()
    assert astllm.model_name() == "qwen3.5:4b-bf16"
    assert astllm._saved_model() == "qwen3.5:4b-bf16"
    assert astllm.set_model("") == astllm.default_model()
    assert not model_config.exists()
    assert astllm.model_name() == astllm.default_model()


def test_model_env_override_wins(model_config, monkeypatch):
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    astllm.set_model("qwen3.5:4b-bf16")
    monkeypatch.setenv("IMGS_ASSIST_MODEL", "some-env-model")
    assert astllm.model_name() == "some-env-model"
    assert astllm._saved_model() == "qwen3.5:4b-bf16"


def test_model_status_reports_default_saved_env(model_config, monkeypatch):
    monkeypatch.setattr(astllm, "installed_models", lambda: ["qwen3.5:4b-q8_0"])
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    astllm.set_model("qwen3.5:4b-bf16")
    info = astllm.status()
    assert info["default"] == astllm.DEFAULT_MODEL
    assert info["saved"] == "qwen3.5:4b-bf16"
    assert info["model"] == "qwen3.5:4b-bf16"
    assert info["env_override"] is False
    monkeypatch.setenv("IMGS_ASSIST_MODEL", "qwen3.5:4b-q8_0")
    info = astllm.status()
    assert info["env_override"] is True
    assert info["model"] == "qwen3.5:4b-q8_0"
    assert info["available"] is True


def test_model_info_flags_and_sorting(model_config, monkeypatch):
    def fake_request(path, payload=None, timeout=5.0):
        if path == "/api/tags":
            return {
                "models": [
                    {"name": "b:latest", "size": 20, "details": {"parameter_size": "4B"}},
                    {"name": "a:latest", "size": 10, "details": {"quantization_level": "Q8_0"}},
                ]
            }
        if path == "/api/show":
            caps = ["completion", "vision"] if payload["model"].startswith("a") else ["completion"]
            return {"capabilities": caps}
        raise astllm.LLMError("unexpected")

    monkeypatch.setattr(astllm, "_request", fake_request)
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    astllm.invalidate_models()
    astllm._invalidate_caps()
    rows = astllm.model_info()
    assert [row["name"] for row in rows] == ["a:latest", "b:latest"]
    assert rows[0]["vision"] is True and rows[0]["completion"] is True
    assert rows[1]["vision"] is False and rows[1]["completion"] is True
    assert all(row["selected"] is False for row in rows)

    monkeypatch.setenv("IMGS_ASSIST_MODEL", "b:latest")
    astllm._invalidate_caps()
    rows = astllm.model_info()
    assert rows[0]["name"] == "b:latest" and rows[0]["selected"] is True


def _stub_ollama(monkeypatch, names=("qwen3.5:4b-q8_0", "qwen3.5:4b-bf16")):
    monkeypatch.setattr(astllm, "installed_models", lambda: list(names))
    monkeypatch.setattr(astllm, "status", lambda: {
        "mode": "rules",
        "model": astllm.model_name(),
        "default": astllm.DEFAULT_MODEL,
        "saved": astllm._saved_model(),
        "env_override": False,
        "available": True,
        "label": "test",
        "url": "http://127.0.0.1:11434",
        "error": None,
    })


def test_models_endpoint_lists_installed(client, model_config, monkeypatch):
    rows = [
        {"name": "qwen3.5:4b-bf16", "completion": True, "vision": True, "selected": False},
        {"name": "qwen3.5:4b-q8_0", "completion": True, "vision": True, "selected": True},
    ]
    monkeypatch.setattr(astllm, "model_info", lambda: rows)
    _stub_ollama(monkeypatch)
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    body = client.get("/api/assistant/models").json()
    assert body["current"] == astllm.DEFAULT_MODEL
    assert body["default"] == astllm.DEFAULT_MODEL
    assert [m["name"] for m in body["models"]] == [m["name"] for m in rows]
    assert "brain" in body and "vision" in body


def test_model_endpoint_switches_and_resets(client, model_config, monkeypatch):
    _stub_ollama(monkeypatch)
    monkeypatch.delenv("IMGS_ASSIST_MODEL", raising=False)
    res = client.post("/api/assistant/model", json={"model": "qwen3.5:4b-bf16"})
    assert res.status_code == 200
    assert res.json()["model"] == "qwen3.5:4b-bf16"
    assert astllm._saved_model() == "qwen3.5:4b-bf16"

    res = client.post("/api/assistant/model", json={"model": ""})
    assert res.status_code == 200
    assert res.json()["model"] == astllm.DEFAULT_MODEL
    assert astllm._saved_model() == ""


def test_model_endpoint_rejects_unknown(client, model_config, monkeypatch):
    _stub_ollama(monkeypatch)
    res = client.post("/api/assistant/model", json={"model": "nope:1b"})
    assert res.status_code == 400
    assert "nope:1b" in res.json()["detail"]
    assert astllm._saved_model() == ""


def test_model_endpoint_requires_ollama(client, model_config, monkeypatch):
    _stub_ollama(monkeypatch, names=())
    res = client.post("/api/assistant/model", json={"model": "qwen3.5:4b-bf16"})
    assert res.status_code == 400
    assert "Ollama" in res.json()["detail"]


def test_model_endpoint_blocked_while_vision_running(client, model_config, monkeypatch):
    _stub_ollama(monkeypatch)
    monkeypatch.setattr(astvis.RUNNER, "running", lambda: True)
    res = client.post("/api/assistant/model", json={"model": "qwen3.5:4b-bf16"})
    assert res.status_code == 409
    assert astllm._saved_model() == ""


def test_model_endpoint_blocked_by_env_override(client, model_config, monkeypatch):
    _stub_ollama(monkeypatch)
    monkeypatch.setenv("IMGS_ASSIST_MODEL", "qwen3.5:4b-q8_0")
    res = client.post("/api/assistant/model", json={"model": "qwen3.5:4b-bf16"})
    assert res.status_code == 409
    assert "IMGS_ASSIST_MODEL" in res.json()["detail"]
    assert astllm._saved_model() == ""


# ── English UI language: suggestions / help / rule-brain EN commands ────────
def test_suggestions_for_en_are_ascii():
    en = ast.suggestions_for("en")
    assert len(en) == len(ast.SUGGESTIONS)
    assert all(s.isascii() for s in en)
    # Thai is the default
    assert ast.suggestions_for(None) == ast.SUGGESTIONS
    assert ast.suggestions_for("th") == ast.SUGGESTIONS


def test_help_text_en_has_no_thai():
    th = ast.help_text("rules")
    assert "สมอง:" in th
    en = ast.help_text("rules", lang="en")
    assert "Brain: rules" in en
    assert "Brain: local LLM" in ast.help_text("llm", lang="en")
    import re

    assert not re.search(r"[\u0E00-\u0E7F]", en), en
    # every tool row shows an English example command
    assert "e.g. “audit dataset”" in en


def test_rule_brain_parses_english_quick_commands():
    assert ast.parse_intent_rules("audit dataset")["tool"] == "dataset_summary"
    assert ast.parse_intent_rules("list loras")["tool"] == "list_loras"
    assert ast.parse_intent_rules("gpu status")["tool"] == "gpu_status"
    assert ast.parse_intent_rules("read dataset images")["tool"] == "read_images"
    assert ast.parse_intent_rules("fill angle tags")["tool"] == "fill_angle_tags"
    assert ast.parse_intent_rules("delete latest image")["tool"] == "delete_history"


def test_rule_brain_parses_english_examples_with_args():
    intent = ast.parse_intent_rules("start training name: my-lora-v7")
    assert intent["tool"] == "start_training"
    assert intent["args"]["name"] == "my-lora-v7"

    intent = ast.parse_intent_rules("set trigger to my character")
    assert intent["tool"] == "set_trigger"
    assert intent["args"]["trigger"] == "my character"

    intent = ast.parse_intent_rules("caption 2fc60a9fdc9c: back view, standing")
    assert intent["tool"] == "set_caption"
    assert intent["args"]["file"] == "2fc60a9fdc9c"
    assert intent["args"]["caption"] == "back view, standing"
