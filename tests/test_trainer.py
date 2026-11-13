"""Tests for dataset registry helpers and LoRA output packaging."""
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open

import sana_trainer as st


def _sample_images(n):
    return [{"file": f"img{i}.png", "name": f"img{i}", "caption": "", "added": ""} for i in range(n)]


def test_sanitize_lora_name():
    assert st.sanitize_lora_name("My Cool LoRA!") == "My-Cool-LoRA"
    # non-ASCII (Thai) disappears → falls back to "lora"
    assert st.sanitize_lora_name("  แปลก ๆ ") == "lora"
    assert st.sanitize_lora_name("") == "lora"
    long = "x" * 100
    assert len(st.sanitize_lora_name(long)) == 48


def test_sanitize_text_and_trigger():
    assert st.sanitize_text("  a  b  ", 10) == "a b"
    assert st.sanitize_trigger("  FoXY ") == "foxy"
    assert st.sanitize_trigger("") == ""


def test_dataset_crud_cycle():
    created = st.create_dataset("Crud Test", "crudtest", "character")
    ds_id = created["id"]
    try:
        assert st.get_dataset(ds_id)["name"] == "Crud Test"
        updated = st.update_dataset(ds_id, name="Renamed", trigger=" ren trigger ", kind="style")
        assert updated["name"] == "Renamed"
        assert updated["trigger"] == "ren trigger"
        assert updated["kind"] == "style"

        # invalid kind is ignored (keeps old)
        updated2 = st.update_dataset(ds_id, kind="banana")
        assert updated2["kind"] == "style"
    finally:
        st.delete_dataset(ds_id)
    assert st.get_dataset(ds_id) is None


def test_dataset_delete_removes_folder():
    created = st.create_dataset("Folder Test", "foldertest", "character")
    ds_id = created["id"]
    d = st.TRAIN_DIR / ds_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "dummy.txt").write_text("x")
    st.delete_dataset(ds_id)
    assert not d.exists()


def test_add_remove_image_validates():
    import tempfile
    from PIL import Image

    created = st.create_dataset("Img Test", "imgtest", "character")
    ds_id = created["id"]
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.new("RGB", (8, 8)).save(tmp.name)
            src = tmp.name
        entry = st.add_image(ds_id, src, "photo one.png")
        assert entry["file"].endswith(".png")
        assert len(st.get_dataset(ds_id)["images"]) == 1
        # duplicate name rejected
        try:
            st.add_image(ds_id, src, "photo one.png")
            raise AssertionError("duplicate should fail")
        except ValueError:
            pass
        # bad extension rejected
        bad = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        bad.close()
        try:
            st.add_image(ds_id, bad.name, "x.gif")
            raise AssertionError("gif should fail")
        except ValueError:
            pass
        # captions
        st.set_captions(ds_id, [{"file": entry["file"], "caption": "a furry fox"}])
        ds = st.get_dataset(ds_id)
        assert ds["images"][0]["caption"] == "a furry fox"
        assert ds["images"][0]["multi"] is False
        # multi-character flag is saved with the caption
        st.set_captions(
            ds_id,
            [{"file": entry["file"], "caption": "a furry fox with a man", "multi": True}],
        )
        assert st.get_dataset(ds_id)["images"][0]["multi"] is True
        st.set_captions(ds_id, [{"file": entry["file"], "multi": False}])
        assert st.get_dataset(ds_id)["images"][0]["multi"] is False
        # remove
        st.remove_image(ds_id, entry["file"])
        assert st.get_dataset(ds_id)["images"] == []
    finally:
        st.delete_dataset(ds_id)


def test_multi_caption_hint():
    assert st.looks_multi_caption("The character is having sex with a man.")
    assert st.looks_multi_caption("several individuals, two people")
    assert not st.looks_multi_caption("woman standing, side view")
    assert not st.looks_multi_caption("")


def test_select_train_images_policies():
    images = [
        {"file": "a"},
        {"file": "b", "multi": True},
        {"file": "c"},
    ]
    solo, weights, mode = st.select_train_images(images, "exclude", 0.25)
    assert [im["file"] for im in solo] == ["a", "c"]
    assert weights is None and mode == "exclude"

    all_images, weights, mode = st.select_train_images(images, "weighted", 0.25)
    assert [im["file"] for im in all_images] == ["a", "b", "c"]
    assert weights == [1.0, 0.25, 1.0]
    assert mode == "weighted"

    all_images, weights, mode = st.select_train_images(images, "include", 0.25)
    assert len(all_images) == 3 and weights is None and mode == "include"

    # unknown mode falls back to the safe default (solo only)
    solo, weights, mode = st.select_train_images(images, "banana", 0.25)
    assert [im["file"] for im in solo] == ["a", "c"] and mode == "exclude"

    # weights are clamped to 5%..100%
    _, low, _ = st.select_train_images(images, "weighted", 0)
    assert low[1] == 0.05
    _, high, _ = st.select_train_images(images, "weighted", 9)
    assert high[1] == 1.0

    # only multi images + exclude → clear error instead of training on them
    try:
        st.select_train_images([{"file": "b", "multi": True}], "exclude", 0.25)
        raise AssertionError("no solo images should fail")
    except RuntimeError as exc:
        assert "ตัวละครเดี่ยว" in str(exc)

    try:
        st.select_train_images([], "exclude", 0.25)
        raise AssertionError("empty dataset should fail")
    except RuntimeError:
        pass


def test_effective_caption_anchors_trigger():
    ds = {"trigger": "foxy", "name": "x"}
    # empty caption → trigger alone
    assert st._effective_caption({"caption": ""}, ds) == "foxy"
    # caption WITHOUT the trigger → trigger auto-prepended (the classic LoRA
    # mistake: captions never mention the trigger → trigger learns nothing)
    assert st._effective_caption({"caption": "a fox in snow"}, ds) == "foxy, a fox in snow"
    # caption already containing the trigger (case-insensitive) → untouched
    assert st._effective_caption({"caption": "Foxy posing at night"}, ds) == "Foxy posing at night"
    # no trigger configured → caption as-is
    assert st._effective_caption({"caption": "a fox", "name": "y"}, {"trigger": "", "name": "x"}) == "a fox"


def test_save_lora_state_format(tmp_path):
    """Inject tiny LoRA → save → file must match diffusers SANA LoRA format."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["0", "2"], bias="none")
    peft_model = inject_adapter_in_model(cfg, model)
    out = tmp_path / "out.safetensors"
    st._save_lora_state(peft_model, out, {"lora_name": "t", "family": "sana"})
    with safe_open(str(out), framework="pt") as f:
        keys = list(f.keys())
        assert keys, "file empty"
        assert all(k.startswith("transformer.") for k in keys), keys[:3]
        assert all("base_model" not in k and ".default" not in k for k in keys)
        meta = f.metadata() or {}
        assert meta.get("format") == "pt"
        assert meta.get("lora_name") == "t"
        assert meta.get("family") == "sana"


def test_save_lora_state_is_atomic(tmp_path, monkeypatch):
    """A crash during save must never damage an existing LoRA file."""
    import peft.utils
    import safetensors.torch

    out = tmp_path / "keep.safetensors"
    out.write_bytes(b"original")
    monkeypatch.setattr(
        peft.utils, "get_peft_model_state_dict", lambda model: {"foo": torch.zeros(1)}
    )

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(safetensors.torch, "save_file", boom)
    try:
        st._save_lora_state(object(), out, {})
        raise AssertionError("save should have failed")
    except RuntimeError:
        pass
    assert out.read_bytes() == b"original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_train_param_validation():
    created = st.create_dataset("Param Test", "paramtest", "character")
    ds_id = created["id"]
    trainer = st.SanaTrainer(manager=None)
    try:
        # dataset ยังไม่มีรูป → ห้ามเทรน (เช็คก่อน clamping)
        try:
            trainer._sanitize({"dataset_id": ds_id, "lora_name": "x"})
            raise AssertionError("empty dataset should be rejected")
        except RuntimeError:
            pass
        # ใส่รายการรูปจำลองลง registry (ไม่ต้องมีไฟล์จริงสำหรับ _sanitize)
        datasets = st.load_registry()
        datasets[0]["images"] = _sample_images(2)
        st.save_registry(datasets)

        cfg = trainer._sanitize(
            {
                "dataset_id": ds_id,
                "lora_name": "x",
                "rank": 999,
                "steps": 99999,
                "lr": 9,
                "resolution": 42,
                "scheduler": "banana",
                "warmup_ratio": 5,
                "ema_decay": 99,
                "timestep_sampling": "banana",
                "save_every": 999999,
                "grad_accum": 99,
                "multi_mode": "banana",
                "multi_weight": 9,
                "thermal_max_c": 999,
                "thermal_resume_c": 999,
                "thermal_check_every": 999,
            }
        )
        assert cfg["rank"] == 128
        assert cfg["steps"] == 20000
        assert cfg["lr"] == 5e-4
        assert cfg["resolution"] == 1024  # invalid → default
        assert cfg["seed"] >= 0  # -1 → random seed
        # new quality knobs fall back / clamp to safe values
        assert cfg["scheduler"] == "cosine"
        assert cfg["warmup_ratio"] == 0.2
        assert cfg["ema_decay"] == 0.9999
        assert cfg["sampling"] == "logit_normal"
        assert cfg["save_every"] == 20000  # clamped to steps
        assert cfg["grad_accum"] == 8
        assert cfg["ema"] is False
        assert cfg["multi_mode"] == "exclude"
        assert cfg["multi_weight"] == 1.0
        assert cfg["augment_flip"] is False
        # phase-2 recipe defaults / clamps
        assert cfg["bucketing"] is True
        assert cfg["caption_dropout"] == 0.05
        assert cfg["caption_shuffle"] is False
        assert cfg["keep_tokens"] == 1
        assert cfg["noise_offset"] == 0.0
        assert cfg["min_snr_gamma"] == 0.0
        assert cfg["use_rslora"] is False
        assert cfg["target_set"] == "attn"
        assert cfg["resume_state"] is False
        # thermal guard clamps to a safe band; resume stays below the throttle
        assert cfg["thermal_guard"] is True
        assert cfg["thermal_mode"] == "throttle"
        assert cfg["thermal_max_c"] == 93
        assert cfg["thermal_throttle_c"] == 87
        assert cfg["thermal_resume_c"] == 86
        assert cfg["thermal_step_pct"] == 5
        assert cfg["thermal_max_level"] == 5
        assert cfg["thermal_check_every"] == 200

        # explicit values are preserved
        cfg2 = trainer._sanitize(
            {
                "dataset_id": ds_id,
                "lora_name": "x",
                "rank": 32,
                "steps": 3000,
                "lr": 1.5e-4,
                "resolution": 1024,
                "scheduler": "constant",
                "ema": False,
                "timestep_sampling": "shift",
                "save_every": 500,
                "grad_accum": 2,
                "multi_mode": "weighted",
                "multi_weight": 0.4,
                "thermal_guard": False,
                "thermal_mode": "pause",
                "thermal_throttle_c": 88,
                "thermal_resume_c": 81,
                "thermal_step_pct": 4,
            }
        )
        assert cfg2["rank"] == 32
        assert cfg2["steps"] == 3000
        assert cfg2["scheduler"] == "constant"
        assert cfg2["ema"] is False
        assert cfg2["sampling"] == "shift"
        assert cfg2["save_every"] == 500
        assert cfg2["grad_accum"] == 2
        assert cfg2["thermal_guard"] is False
        assert cfg2["multi_mode"] == "weighted"
        assert cfg2["multi_weight"] == 0.4
        assert cfg2["thermal_mode"] == "pause"
        assert cfg2["thermal_max_c"] == 90
        assert cfg2["thermal_throttle_c"] == 88
        assert cfg2["thermal_resume_c"] == 81
        assert cfg2["thermal_step_pct"] == 4
        assert cfg2["thermal_max_level"] == 6

        # phase-2 explicit values are preserved and clamped
        cfg_phase2 = trainer._sanitize(
            {
                "dataset_id": ds_id,
                "lora_name": "x",
                "bucketing": False,
                "caption_dropout": 9,
                "caption_shuffle": True,
                "keep_tokens": 9,
                "noise_offset": 9,
                "min_snr_gamma": 99,
                "use_rslora": True,
                "target_modules": "attn_ff",
                "resume_state": True,
            }
        )
        assert cfg_phase2["bucketing"] is False
        assert cfg_phase2["caption_dropout"] == 0.3
        assert cfg_phase2["caption_shuffle"] is True
        assert cfg_phase2["keep_tokens"] == 4
        assert cfg_phase2["noise_offset"] == 0.2
        assert cfg_phase2["min_snr_gamma"] == 20.0
        assert cfg_phase2["use_rslora"] is True
        assert cfg_phase2["target_set"] == "attn_ff"
        assert cfg_phase2["resume_state"] is True

        # EMA is opt-in: explicit true is still honoured
        cfg3 = trainer._sanitize({"dataset_id": ds_id, "lora_name": "x", "ema": True})
        assert cfg3["ema"] is True
    finally:
        st.delete_dataset(ds_id)

def test_thermal_wait_resumes_when_cool(monkeypatch):
    temps = iter([90.0, 88.0, 74.0])
    monkeypatch.setattr(st, "gpu_temperature", lambda: next(temps))
    monkeypatch.setattr(st.time, "sleep", lambda _s: None)
    job = {"phase": "generating", "message": ""}
    cfg = {"thermal_max_c": 82, "thermal_resume_c": 75}
    st.thermal_wait(cfg, job, __import__("threading").Event())
    assert job["phase"] == "paused"  # it did pause before resuming
    assert "พัก GPU" in job["message"]


def test_thermal_wait_aborts_after_15_min(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 95.0)
    monkeypatch.setattr(st.time, "sleep", lambda _s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr(st.time, "time", lambda: clock.__setitem__("t", clock["t"] + 120) or clock["t"])
    job = {"phase": "generating", "message": ""}
    cfg = {"thermal_max_c": 82, "thermal_resume_c": 75}
    try:
        st.thermal_wait(cfg, job, __import__("threading").Event())
        raise AssertionError("should abort when the GPU never cools")
    except RuntimeError as exc:
        assert "15 นาที" in str(exc)


def test_thermal_wait_honours_cancel(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 95.0)
    monkeypatch.setattr(st.time, "sleep", lambda _s: None)
    cancel = __import__("threading").Event()
    cancel.set()
    job = {"phase": "generating", "message": ""}
    cfg = {"thermal_max_c": 82, "thermal_resume_c": 75}
    try:
        st.thermal_wait(cfg, job, cancel)
        raise AssertionError("cancel should stop the wait")
    except st.se.Cancelled:
        pass


# ── throttle governor (slow down instead of pausing) ────────────────────────
def _governor_cfg(**overrides):
    cfg = {
        "thermal_max_c": 90,
        "thermal_throttle_c": 87,
        "thermal_resume_c": 80,
        "thermal_step_pct": 5,
        "thermal_max_level": 5,
    }
    cfg.update(overrides)
    return cfg


def _governor_state(**overrides):
    state = {
        "level": 0,
        "clocks": None,
        "sleep": 0.0,
        "step_time": 1.0,
        "applied": False,
        "temp": None,
    }
    state.update(overrides)
    return state


def test_governor_throttles_one_level_when_hot(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 88.0)
    monkeypatch.setattr(st, "_clock_set", lambda _f: False)
    state = _governor_state()
    st.thermal_governor(_governor_cfg(), {}, __import__("threading").Event(), state)
    assert state["level"] == 1
    # no clock control → short duty-cycle sleep proportional to step time
    assert state["sleep"] == pytest.approx(0.05)
    assert state["temp"] == 88.0


def test_governor_speeds_back_up_when_cool(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 79.0)
    state = _governor_state(level=2, clocks=False, applied=True)
    st.thermal_governor(_governor_cfg(), {}, __import__("threading").Event(), state)
    assert state["level"] == 1


def test_governor_never_goes_below_zero(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 70.0)
    state = _governor_state(level=0, clocks=False, applied=True)
    st.thermal_governor(_governor_cfg(), {}, __import__("threading").Event(), state)
    assert state["level"] == 0
    assert state["sleep"] == 0.0


def test_governor_caps_level(monkeypatch):
    monkeypatch.setattr(st, "gpu_temperature", lambda: 89.0)
    monkeypatch.setattr(st, "_clock_set", lambda _f: False)
    state = _governor_state(level=5, clocks=False, applied=True)
    st.thermal_governor(_governor_cfg(), {}, __import__("threading").Event(), state)
    assert state["level"] == 5  # max_level is 5 → no further slowdown


def test_governor_uses_clock_control_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "gpu_temperature", lambda: 88.0)
    monkeypatch.setattr(st, "_clock_set", lambda f: calls.append(f) or True)
    state = _governor_state()
    st.thermal_governor(_governor_cfg(), {}, __import__("threading").Event(), state)
    assert state["clocks"] is True
    assert state["sleep"] == 0.0
    assert calls == [pytest.approx(0.95)]


def test_governor_hard_limit_pauses(monkeypatch):
    paused = []
    monkeypatch.setattr(st, "gpu_temperature", lambda: 91.0)
    monkeypatch.setattr(
        st, "thermal_wait", lambda cfg, job, cancel: paused.append(cfg)
    )
    state = _governor_state(level=3, clocks=True, applied=True)
    job = {"phase": "generating", "message": ""}
    st.thermal_governor(_governor_cfg(), job, __import__("threading").Event(), state)
    assert paused  # emergency pause invoked
    assert state["level"] == 0  # full speed again after cooling
    assert job["phase"] == "generating"


def test_set_throttle_level_resets_clocks_at_zero(monkeypatch):
    reset = []
    monkeypatch.setattr(st, "_clock_reset", lambda: reset.append(True))
    state = _governor_state(level=2, clocks=True, sleep=0.1, applied=True)
    st._set_throttle_level(_governor_cfg(), state, 0)
    assert reset == [True]
    assert state["level"] == 0
    assert state["sleep"] == 0.0
    assert state["clocks"] is None


def test_supported_clocks_uses_top_memory_block(monkeypatch):
    """nvidia-smi lists one block per memory clock — use the highest one."""
    output = (
        "    Supported Clocks\n"
        "        Memory                                         : 5505 MHz\n"
        "            Graphics                                   : 1911 MHz\n"
        "            Graphics                                   : 1898 MHz\n"
        "        Memory                                         : 405 MHz\n"
        "            Graphics                                   : 607 MHz\n"
        "            Graphics                                   : 139 MHz\n"
    )
    monkeypatch.setattr(st.subprocess, "check_output", lambda *a, **k: output)
    monkeypatch.setattr(st, "_SUPPORTED_CLOCKS", {})
    assert st.gpu_supported_clocks() == (5505, [1898, 1911])


def test_clock_set_treats_unsupported_as_failure(monkeypatch):
    """The P102-100 exits 0 but prints "not supported" — must fall back."""
    from types import SimpleNamespace

    monkeypatch.setattr(st, "gpu_supported_clocks", lambda: (5505, [139, 1911]))
    monkeypatch.setattr(
        st.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="Setting applications clocks is not supported for GPU 0.\n",
            stderr="",
        ),
    )
    monkeypatch.setitem(st._CLOCKS_ACTIVE, "value", False)
    assert st._clock_set(0.95) is False
    assert st._CLOCKS_ACTIVE["value"] is False


def test_clock_set_accepts_real_success(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(st, "gpu_supported_clocks", lambda: (5505, [139, 1911]))
    monkeypatch.setattr(
        st.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="All done.\n", stderr=""
        ),
    )
    monkeypatch.setitem(st._CLOCKS_ACTIVE, "value", False)
    assert st._clock_set(0.95) is True
    assert st._CLOCKS_ACTIVE["value"] is True
    monkeypatch.setitem(st._CLOCKS_ACTIVE, "value", False)


def test_clock_reset_skips_when_no_cap_applied(monkeypatch):
    calls = []
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setitem(st._CLOCKS_ACTIVE, "value", False)
    st._clock_reset()
    assert calls == []


# ── continue-from (resume) support ──────────────────────────────────────────
class _TinyAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_k = torch.nn.Linear(8, 8)
        self.to_v = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


class _TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _TinyAttn()


class _TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_TinyBlock()])


def _tiny_lora(rank=4):
    model = _TinyTransformer()
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    return inject_adapter_in_model(cfg, model)


def test_lora_roundtrip_and_resume(tmp_path):
    src = _tiny_lora(4)
    path = tmp_path / "tiny.safetensors"
    st._save_lora_state(src, path, {"rank": "4", "steps": "10"})

    assert st._lora_file_rank(path) == 4

    dst = _tiny_lora(4)
    loaded, skipped = st._load_lora_weights(dst, path)
    assert loaded == 8 and skipped == []
    a = dict(src.named_parameters())
    b = dict(dst.named_parameters())
    for name, value in a.items():
        if "lora_" in name:
            # the file is saved as bf16, so compare at bf16 precision
            assert torch.allclose(
                value.to(torch.bfloat16).float(), b[name], atol=1e-2
            ), name


def test_lora_rank_from_shapes_without_metadata(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "no-meta.safetensors"
    save_file(
        {"transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8)},
        str(path),
    )
    assert st._lora_file_rank(path) == 4


def test_load_lora_rejects_rank_mismatch(tmp_path):
    path = tmp_path / "r4.safetensors"
    st._save_lora_state(_tiny_lora(4), path, {"rank": "4"})
    bigger = _tiny_lora(8)
    try:
        st._load_lora_weights(bigger, path)
        raise AssertionError("rank mismatch should raise")
    except RuntimeError as exc:
        assert "rank" in str(exc)


def test_resolve_init_lora_by_id_and_stem(tmp_path, monkeypatch):
    path = tmp_path / "MyAdapter.safetensors"
    st._save_lora_state(_tiny_lora(4), path, {"rank": "4"})
    entry = {"id": "sana-myadapter-abc", "name": "MyAdapter", "path": str(path)}
    monkeypatch.setattr(st.se, "discover_loras", lambda: [entry])

    assert st._resolve_init_lora("sana-myadapter-abc") == path
    assert st._resolve_init_lora("MyAdapter.safetensors") == path
    assert st._resolve_init_lora("") is None
    try:
        st._resolve_init_lora("nope")
        raise AssertionError("unknown lora should raise")
    except RuntimeError as exc:
        assert "ไม่พบ LoRA" in str(exc)


def test_sanitize_adopts_init_rank(tmp_path, monkeypatch):
    path = tmp_path / "rank32.safetensors"
    st._save_lora_state(_tiny_lora(4), path, {"rank": "32"})
    monkeypatch.setattr(
        st.se,
        "discover_loras",
        lambda: [{"id": "init-1", "name": "init", "path": str(path)}],
    )
    created = st.create_dataset("Resume Test", "resumetest", "character")
    ds_id = created["id"]
    try:
        datasets = st.load_registry()
        datasets[0]["images"] = _sample_images(2)
        st.save_registry(datasets)
        cfg = st.SanaTrainer(manager=None)._sanitize(
            {"dataset_id": ds_id, "lora_name": "resumed", "rank": 16, "init_lora": "init-1"}
        )
        assert cfg["init_lora"] == path
        assert cfg["rank"] == 32
    finally:
        st.delete_dataset(ds_id)


# ── SDXL trainer ────────────────────────────────────────────────────────────
def test_save_sdxl_lora_uses_unet_prefix(tmp_path):
    src = _tiny_lora(4)
    path = tmp_path / "sdxl.safetensors"
    st._save_lora_state(src, path, {"family": "sdxl", "rank": "4"}, prefix="unet.")
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        assert keys and all(k.startswith("unet.") for k in keys), keys[:3]
        assert (f.metadata() or {}).get("family") == "sdxl"
    # roundtrip into a fresh adapter
    dst = _tiny_lora(4)
    loaded, skipped = st._load_lora_weights(dst, path, prefix="unet.")
    assert loaded == 8 and skipped == []


def test_sdxl_family_detection_from_diffusers_keys():
    import sana_engine as se

    keys = [
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight",
        "unet.mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight",
        "unet.up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.weight",
    ]
    assert se._family_from_keys(keys) == "sdxl"
    # kohya SD1.5 layout must not be mistaken for SDXL
    assert (
        se._family_from_keys(["lora_unet_down_blocks_0_attn1_to_q.lora_down.weight"])
        == "sd15"
    )


class _FakeManager:
    def __init__(self, entry):
        self.entry = entry

    def get_model(self, model_id=None):
        if model_id and model_id != self.entry["id"]:
            raise ValueError("ไม่พบโมเดลที่เลือก")
        return self.entry


def test_sdxl_sanitize_defaults_and_clamps(tmp_path):
    import sdxl_trainer as sdtr

    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"x")
    entry = {
        "id": "sdxl-test",
        "name": "Test SDXL",
        "family": "sdxl",
        "path": str(model_file),
        "kind": "single",
    }
    created = st.create_dataset("SDXL Param Test", "sdxlparam", "character")
    ds_id = created["id"]
    try:
        datasets = st.load_registry()
        for d in datasets:
            if d.get("id") == ds_id:
                d["images"] = _sample_images(2)
        st.save_registry(datasets)
        trainer = sdtr.SdxlTrainer(manager=_FakeManager(entry))
        cfg = trainer._sanitize({"dataset_id": ds_id, "lora_name": "x"})
        assert cfg["rank"] == 16
        assert cfg["steps"] == 1000
        assert cfg["lr"] == 1e-4
        assert cfg["resolution"] == 768
        assert cfg["sampling"] == "logit_normal"
        assert cfg["ema"] is False
        assert cfg["out_path"].parent == st.se.MODELS_ROOT / "loras"
        assert cfg["bucketing"] is True
        assert cfg["caption_dropout"] == 0.05
        assert cfg["keep_tokens"] == 1
        assert cfg["noise_offset"] == 0.0
        assert cfg["min_snr_gamma"] == 0.0
        assert cfg["use_rslora"] is False
        assert cfg["target_set"] == "attn"
        assert cfg["resume_state"] is False

        cfg2 = trainer._sanitize(
            {
                "dataset_id": ds_id,
                "lora_name": "x",
                "rank": 999,
                "steps": 99999,
                "lr": 9,
                "resolution": 42,
                "scheduler": "banana",
                "timestep_sampling": "banana",
                "ema_decay": 99,
                "grad_accum": 99,
            }
        )
        assert cfg2["rank"] == 128
        assert cfg2["steps"] == 20000
        assert cfg2["lr"] == 5e-4
        assert cfg2["resolution"] == 768
        assert cfg2["scheduler"] == "cosine"
        assert cfg2["sampling"] == "logit_normal"
        assert cfg2["ema_decay"] == 0.9999
        assert cfg2["grad_accum"] == 8
    finally:
        st.delete_dataset(ds_id)


def test_sdxl_sanitize_rejects_non_sdxl_model(tmp_path):
    import sdxl_trainer as sdtr

    created = st.create_dataset("SDXL Guard Test", "sdxlguard", "character")
    ds_id = created["id"]
    try:
        datasets = st.load_registry()
        for d in datasets:
            if d.get("id") == ds_id:
                d["images"] = _sample_images(1)
        st.save_registry(datasets)
        entry = {
            "id": "sana-x",
            "name": "SANA",
            "family": "sana",
            "path": str(tmp_path / "m"),
            "kind": "diffusers",
        }
        try:
            sdtr.SdxlTrainer(manager=_FakeManager(entry))._sanitize(
                {"dataset_id": ds_id, "lora_name": "x"}
            )
            raise AssertionError("non-SDXL model should be rejected")
        except RuntimeError as exc:
            assert "SDXL" in str(exc)
    finally:
        st.delete_dataset(ds_id)


def test_dataset_kinds_and_extra_triggers():
    created = st.create_dataset(
        "Kind Test", "lola", "outfit", extra_triggers="armor form, casual, armor form"
    )
    ds_id = created["id"]
    try:
        assert created["kind"] == "outfit"
        assert created["extra_triggers"] == ["armor form", "casual"]
        # invalid kind falls back to character
        bad = st.create_dataset("Bad Kind", "x", "banana")
        try:
            assert bad["kind"] == "character"
        finally:
            st.delete_dataset(bad["id"])
        # changing the primary trigger drops it from the extra list
        updated = st.update_dataset(ds_id, trigger="Armor Form")
        assert updated["trigger"] == "armor form"
        assert "armor form" not in updated["extra_triggers"]
        # pose is a first-class kind now
        assert st.update_dataset(ds_id, kind="pose")["kind"] == "pose"
        assert st.update_dataset(ds_id, kind="banana")["kind"] == "pose"
    finally:
        st.delete_dataset(ds_id)


def test_sanitize_tags_triggers_repeats():
    assert st.sanitize_tags("Front View, smiling; front view") == [
        "front view",
        "smiling",
    ]
    assert st.sanitize_tags(["a", "A", " b "]) == ["a", "b"]
    assert st.sanitize_tags("") == []
    assert st.sanitize_extra_triggers("a, b, a", "a") == ["b"]
    assert st.sanitize_repeats("7") == 7
    assert st.sanitize_repeats(0) == 1
    assert st.sanitize_repeats(999) == st.MAX_REPEATS
    assert st.sanitize_repeats("nope") == 1


def test_effective_caption_tags_and_trigger_override():
    ds = {"trigger": "foxy", "name": "x"}
    # tags are inserted after the trigger, before the free caption
    assert (
        st._effective_caption(
            {"caption": "standing in snow", "tags": ["front view", "smiling"]}, ds
        )
        == "foxy, front view, smiling, standing in snow"
    )
    # a per-image trigger override replaces the dataset trigger
    assert (
        st._effective_caption(
            {"caption": "battle pose", "trigger": "armor form"}, ds
        )
        == "armor form, battle pose"
    )
    # tags already present in the caption are not duplicated
    assert (
        st._effective_caption(
            {"caption": "front view, standing", "tags": ["front view"]}, ds
        )
        == "foxy, front view, standing"
    )
    # duplicate tags (case-insensitive) are collapsed
    assert (
        st._effective_caption(
            {"caption": "", "tags": ["Front View", "front view"]}, ds
        )
        == "foxy, front view"
    )
    # no trigger/tags/caption still yields a usable caption
    assert st._effective_caption({"caption": ""}, {"trigger": ""}) == "subject"


def test_select_train_images_repeats_and_non_character():
    images = [
        {"file": "a", "repeats": 3},
        {"file": "b", "multi": True, "repeats": 2},
        {"file": "c"},
    ]
    solo, weights, mode = st.select_train_images(images, "exclude", 0.25)
    assert [im["file"] for im in solo] == ["a", "c"]
    assert weights == [3.0, 1.0]

    included, weights, mode = st.select_train_images(images, "include", 0.25)
    assert [im["file"] for im in included] == ["a", "b", "c"]
    assert weights == [3.0, 2.0, 1.0]

    # weighted combines the multi factor with repeats
    _, weights, _ = st.select_train_images(images, "weighted", 0.5)
    assert weights == [3.0, 1.0, 1.0]

    # style/outfit/pose datasets never drop multi-flagged images
    included, weights, mode = st.select_train_images(
        images, "exclude", 0.25, kind="style"
    )
    assert mode == "include"
    assert [im["file"] for im in included] == ["a", "b", "c"]
    assert weights == [3.0, 2.0, 1.0]

    # repeats are clamped to 1..10
    _, weights, _ = st.select_train_images(
        [{"file": "a", "repeats": 99}, {"file": "b", "repeats": 0}], "include", 0.25
    )
    assert weights == [10.0, 1.0]


def test_set_captions_saves_tags_trigger_repeats():
    created = st.create_dataset("Caption Fields", "capfield", "character")
    ds_id = created["id"]
    try:
        datasets = st.load_registry()
        for d in datasets:
            if d.get("id") == ds_id:
                d["images"] = _sample_images(1)
        st.save_registry(datasets)
        file_id = st.get_dataset(ds_id)["images"][0]["file"]
        st.set_captions(
            ds_id,
            [
                {
                    "file": file_id,
                    "caption": "posing",
                    "tags": "Front View, smiling",
                    "trigger": " Armor Form ",
                    "repeats": 4,
                }
            ],
        )
        image = st.get_dataset(ds_id)["images"][0]
        assert image["tags"] == ["front view", "smiling"]
        assert image["trigger"] == "armor form"
        assert image["repeats"] == 4
        # omitted fields keep their previous values
        st.set_captions(ds_id, [{"file": file_id, "caption": "new caption"}])
        image = st.get_dataset(ds_id)["images"][0]
        assert image["tags"] == ["front view", "smiling"]
        assert image["trigger"] == "armor form"
        assert image["repeats"] == 4
    finally:
        st.delete_dataset(ds_id)


def test_resolution_buckets_and_pick():
    buckets = st.resolution_buckets(1024)
    assert (1024, 1024) in buckets
    assert (1216, 832) in buckets or (1152, 896) in buckets
    # all sides are multiples of 64 and >= 256
    for width, height in buckets:
        assert width % st.BUCKET_STEP == 0
        assert height % st.BUCKET_STEP == 0
        assert width >= 256 and height >= 256
    # portrait / landscape / square pick matching orientation
    assert st.pick_bucket(1000, 2000, 1024)[1] > st.pick_bucket(1000, 2000, 1024)[0]
    assert st.pick_bucket(2000, 1000, 1024)[0] > st.pick_bucket(2000, 1000, 1024)[1]
    assert st.pick_bucket(1024, 1024, 1024) == (1024, 1024)
    # invalid input falls back to square
    assert st.pick_bucket(0, 0, 768) == (768, 768)


def test_caption_variants_shuffle_and_keep_tokens():
    ds = {"trigger": "lola"}
    image = {"caption": "smiling, park, sunny", "tags": ["front view"], "trigger": ""}
    base = st._effective_caption(image, ds)
    assert st.caption_variants(image, ds) == [base]
    variants = st.caption_variants(image, ds, shuffle=True, keep_tokens=1, count=4)
    assert len(variants) == 4
    assert variants[0] == base
    # first segment (trigger) is never moved
    for variant in variants[1:]:
        assert variant.split(",")[0].strip() == "lola"
    # deterministic for the same caption
    assert variants == st.caption_variants(image, ds, shuffle=True, keep_tokens=1, count=4)
    # a caption with only one content segment can still be reordered with the
    # trigger, but the variant count is honoured
    solo = {"caption": "only", "tags": []}
    solo_variants = st.caption_variants(solo, ds, shuffle=True, count=3)
    assert len(solo_variants) == 3
    assert solo_variants[0] == st._effective_caption(solo, ds)


def test_noise_offset_and_min_snr_weight():
    torch.manual_seed(0)
    noise = torch.zeros(2, 4, 8, 8)
    plain = st.add_noise_offset(noise, 0.0)
    assert torch.equal(plain, noise)
    shifted = st.add_noise_offset(noise, 0.1)
    assert shifted.shape == noise.shape
    assert shifted.abs().sum() > 0
    # same channel shares one offset across spatial positions
    assert torch.allclose(shifted[0, 0, 0, 0], shifted[0, 0, 3, 5])
    weight = st.min_snr_weight(torch.tensor([0.1, 1.0, 10.0]), 5.0)
    assert weight[0] == pytest.approx(1.0)
    assert weight[1] == pytest.approx(1.0)
    assert weight[2] == pytest.approx(0.5)
    assert st.min_snr_weight(torch.tensor([1.0]), 0) is None


def test_train_state_roundtrip(tmp_path):
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    rng = __import__("random").Random(7)
    g_rng = torch.Generator(device="cpu").manual_seed(7)
    ema = {"x.lora_A.weight": torch.tensor([0.5])}
    param.grad = torch.tensor([0.1, 0.2])
    opt.step()
    sched.step()
    path = st.trainstate_path(tmp_path / "adapter.safetensors")
    st.save_train_state(
        path,
        optimizer=opt,
        scheduler=sched,
        step=42,
        ema_params=ema,
        rng=rng,
        g_rng=g_rng,
        ema_loss=0.123,
        rank=16,
    )
    assert path.exists()
    param2 = torch.nn.Parameter(torch.tensor([9.0, 9.0]))
    opt2 = torch.optim.AdamW([param2], lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _s: 1.0)
    ema2 = {"x.lora_A.weight": torch.tensor([0.0])}
    step, loss = st.load_train_state(
        path,
        optimizer=opt2,
        scheduler=sched2,
        ema_params=ema2,
        rng=__import__("random").Random(0),
        g_rng=torch.Generator(device="cpu"),
        rank=16,
    )
    assert step == 42
    assert loss == pytest.approx(0.123)
    assert torch.allclose(ema2["x.lora_A.weight"], torch.tensor([0.5]))
    # rank mismatch is rejected
    with pytest.raises(RuntimeError):
        st.load_train_state(
            path,
            optimizer=opt2,
            scheduler=sched2,
            ema_params=ema2,
            rng=__import__("random").Random(0),
            g_rng=None,
            rank=8,
        )


def test_rslora_scaling_roundtrip(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    base = Tiny()
    cfg = LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["proj"], bias="none", use_rslora=True,
    )
    model = inject_adapter_in_model(cfg, base)
    path = tmp_path / "rs.safetensors"
    st._save_lora_state(
        model, path, {"rank": "4"}, prefix="", lora_b_scale=2.0
    )
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        b_key = next(k for k in keys if k.endswith("lora_B.weight"))
        saved_b = handle.get_tensor(b_key).float()
    with torch.no_grad():
        raw_b = model.proj.lora_B.default.weight.detach().float()
    assert torch.allclose(saved_b, raw_b * 2.0, atol=1e-2)
    # loading back with 1/scale restores the original magnitude
    fresh = inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                   target_modules=["proj"], bias="none", use_rslora=True),
        Tiny(),
    )
    loaded, skipped = st._load_lora_weights(
        fresh, path, prefix="", lora_b_scale=0.5
    )
    assert loaded >= 2 and not skipped
    with torch.no_grad():
        restored = fresh.proj.lora_B.default.weight.detach().float()
    assert torch.allclose(restored, raw_b, atol=1e-2)


def test_dataset_qc_reports_duplicates_and_buckets():
    import tempfile
    from PIL import Image

    created = st.create_dataset("QC Test", "qctrigger", "character")
    ds_id = created["id"]
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.new("RGB", (640, 480), (10, 20, 30)).save(tmp.name)
            src = tmp.name
        first = st.add_image(ds_id, src, "qc-one.png")
        second = st.add_image(ds_id, src, "qc-two.png")
        st.set_captions(ds_id, [{"file": first["file"], "caption": "park"}])
        report = st.dataset_qc(ds_id, 1024)
        assert report["dataset"]["count"] == 2
        assert report["dataset"]["solo"] == 2
        assert report["dataset"]["missing_caption"] == 1
        assert len(report["dataset"]["duplicates"]) == 1
        assert report["dataset"]["duplicates"][0] == sorted(
            [first["file"], second["file"]]
        ) or set(report["dataset"]["duplicates"][0]) == {
            first["file"], second["file"]
        }
        rows = {row["file"]: row for row in report["images"]}
        assert rows[first["file"]]["width"] == 640
        assert rows[first["file"]]["height"] == 480
        assert rows[second["file"]]["bucket"].count("×") == 1
        assert any("caption" in w for w in rows[second["file"]]["warnings"])
        assert report["dataset"]["recommendations"]
    finally:
        st.delete_dataset(ds_id)
