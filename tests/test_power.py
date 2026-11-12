"""GPU power-mode API + helper script tests.

No real sudo, nvidia-smi or systemd is touched: the API layer is stubbed and
the helper is exercised with a fake `nvidia-smi` and a temp state directory.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server as srv

READINGS = {
    "limit": 150.0,
    "draw": 8.5,
    "temp": 37.0,
    "min": 125.0,
    "max": 300.0,
    "default": 250.0,
    "persistence": "Disabled",
}

HELPER = Path(__file__).resolve().parent.parent / "host" / "pixelsmith-power"


@pytest.fixture()
def client():
    with TestClient(srv.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_power_globals(monkeypatch):
    monkeypatch.setattr(srv, "_POWER_DESIRED", None)
    monkeypatch.setattr(srv, "_POWER_ENFORCE", False)
    srv._POWER_READY_CACHE.update(at=0.0, value=False)
    yield


@pytest.fixture()
def ready(monkeypatch):
    monkeypatch.setattr(srv, "_power_helper_ready", lambda force=False: True)
    monkeypatch.setattr(srv, "_power_readings", lambda: dict(READINGS))
    monkeypatch.setattr(srv, "_POWER_DESIRED", 200)


@pytest.fixture()
def not_ready(monkeypatch):
    monkeypatch.setattr(srv, "_power_helper_ready", lambda force=False: False)
    monkeypatch.setattr(srv, "_power_readings", lambda: dict(READINGS))


def test_power_status_reports_modes(client, ready):
    res = client.get("/api/power")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["available"] is True
    assert body["supported"] is True
    assert body["installed"] is True
    assert body["setup_required"] is False
    # mode = actual card state, saved_mode = the app's remembered choice.
    assert body["mode"] == "low"
    assert body["saved"] == 200
    assert body["saved_mode"] == "balanced"
    assert {m["id"]: m["watts"] for m in body["modes"]} == {
        "low": 150,
        "balanced": 200,
        "performance": 250,
    }


def test_power_modes_scale_with_card_default(monkeypatch):
    readings = dict(READINGS, default=175.0, max=175.0, min=60.0)
    assert {m["id"]: m["watts"] for m in srv._power_modes(readings)} == {
        "low": 105,
        "balanced": 140,
        "performance": 175,
    }


def test_power_modes_clamp_to_card_minimum():
    readings = dict(READINGS, default=120.0, max=120.0, min=110.0)
    assert {m["id"]: m["watts"] for m in srv._power_modes(readings)} == {
        "low": 110,
        "balanced": 110,
        "performance": 120,
    }


def test_power_status_setup_required(client, not_ready):
    body = client.get("/api/power").json()
    assert body["setup_required"] is True
    assert body["installed"] is False
    res = client.post("/api/power", json={"mode": "balanced"})
    assert res.status_code == 409


def test_power_unsupported_platform(client, monkeypatch):
    monkeypatch.setattr(srv, "POWER_SUPPORTED", False)
    monkeypatch.setattr(srv, "_power_readings", lambda: dict(READINGS))
    body = client.get("/api/power").json()
    assert body["supported"] is False
    assert body["setup_required"] is False
    res = client.post("/api/power", json={"mode": "low"})
    assert res.status_code == 400


def test_power_set_mode_calls_helper(client, ready, monkeypatch):
    calls = []

    def fake_run(args, timeout=60):
        calls.append((args, timeout))
        return subprocess.CompletedProcess(args, 0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power", json={"mode": "performance"})
    assert res.status_code == 200, res.text
    args, _ = calls[0]
    assert args[:3] == ["sudo", "-n", srv.POWER_HELPER]
    assert args[3:] == ["set", "250"]


def test_power_set_watts(client, ready, monkeypatch):
    calls = []

    def fake_run(args, timeout=60):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power", json={"watts": 180})
    assert res.status_code == 200
    assert calls[0][3:] == ["set", "180"]


def test_power_set_keeps_choice_when_driver_is_busy(client, ready, monkeypatch):
    """A transient driver refusal must not roll the saved mode back."""

    def fake_run(args, timeout=60):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout='{"ok":false,"error":"could not set power limit","configured":180}',
            stderr="",
        )

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power", json={"watts": 180})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["saved"] == 180
    assert srv._POWER_DESIRED == 180
    assert srv._POWER_ENFORCE is True


def test_power_reset_restores_default_but_keeps_saved(client, ready, monkeypatch):
    calls = []

    def fake_run(args, timeout=60):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"ok":true,"configured":200,"limit":250.0}',
            stderr="",
        )

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power/reset")
    assert res.status_code == 200, res.text
    assert calls[0][3:] == ["reset"]
    body = res.json()
    assert body["saved"] == 200
    assert body["saved_mode"] == "balanced"
    assert srv._POWER_DESIRED == 200
    assert srv._POWER_ENFORCE is False


def test_power_reset_reports_failure(client, ready, monkeypatch):
    def fake_run(args, timeout=60):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"ok":false,"error":"could not reset power limit","configured":200}',
            stderr="",
        )

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power/reset")
    assert res.status_code == 500
    assert "could not reset" in res.json()["detail"]
    assert srv._POWER_DESIRED == 200
    assert srv._POWER_ENFORCE is False


def test_power_startup_applies_saved_mode(monkeypatch):
    calls = []

    def fake_run(args, timeout=60):
        calls.append((args, timeout))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"ok":true,"configured":200,"limit":200.0}',
            stderr="",
        )

    monkeypatch.setattr(srv, "_power_helper_installed", lambda: True)
    monkeypatch.setattr(srv, "_run_power", fake_run)
    monkeypatch.setattr(srv, "_POWER_DESIRED", None)
    monkeypatch.setattr(srv, "_POWER_ENFORCE", False)

    data = srv._power_apply_saved()
    assert calls[0][0][3:] == ["apply"]
    assert data["configured"] == 200
    assert srv._POWER_DESIRED == 200
    assert srv._POWER_ENFORCE is True


def test_power_watchdog_reapplies_after_drift(monkeypatch):
    calls = []

    def fake_run(args, timeout=60):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"ok":true,"configured":200,"limit":200.0}',
            stderr="",
        )

    monkeypatch.setattr(srv, "_POWER_DESIRED", 200)
    monkeypatch.setattr(srv, "_POWER_ENFORCE", True)
    monkeypatch.setattr(srv, "_power_readings", lambda: dict(READINGS, limit=150.0))
    monkeypatch.setattr(srv, "_run_power", fake_run)

    assert srv._power_watchdog_tick() is True
    assert calls[0][3:] == ["apply"]


def test_power_watchdog_leaves_matching_limit_alone(monkeypatch):
    monkeypatch.setattr(srv, "_POWER_DESIRED", 200)
    monkeypatch.setattr(srv, "_POWER_ENFORCE", True)
    monkeypatch.setattr(srv, "_power_readings", lambda: dict(READINGS, limit=200.0))
    monkeypatch.setattr(
        srv, "_run_power", lambda *a, **k: pytest.fail("helper must not run")
    )

    assert srv._power_watchdog_tick() is False


def test_power_rejects_out_of_range(client, ready, monkeypatch):
    monkeypatch.setattr(
        srv, "_run_power", lambda *a, **k: pytest.fail("helper must not run")
    )
    res = client.post("/api/power", json={"watts": 500})
    assert res.status_code == 400
    assert "125" in res.json()["detail"] and "300" in res.json()["detail"]


def test_power_rejects_unknown_mode(client, ready):
    res = client.post("/api/power", json={"mode": "turbo"})
    assert res.status_code == 400


def test_power_requires_a_target(client, ready):
    res = client.post("/api/power", json={})
    assert res.status_code == 400


def test_power_helper_failure_is_reported(client, ready, monkeypatch):
    def fake_run(args, timeout=60):
        return subprocess.CompletedProcess(
            args, 1, stdout='{"ok":false,"error":"nope"}', stderr=""
        )

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power", json={"watts": 150})
    assert res.status_code == 500
    assert "nope" in res.json()["detail"]


def test_power_password_required_marks_setup(client, ready, monkeypatch):
    def fake_run(args, timeout=60):
        return subprocess.CompletedProcess(
            args, 1, stdout="", stderr="sudo: a password is required"
        )

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power", json={"watts": 150})
    assert res.status_code == 403
    assert srv._POWER_READY_CACHE["value"] is False


def test_power_setup_without_pkexec(client, monkeypatch):
    monkeypatch.setattr(srv.shutil, "which", lambda name: None)
    res = client.post("/api/power/setup")
    assert res.status_code == 400


def test_power_setup_cancelled(client, monkeypatch):
    monkeypatch.setattr(srv.shutil, "which", lambda name: "/usr/bin/pkexec")
    monkeypatch.setattr(srv.getpass, "getuser", lambda: "tester")

    def fake_run(args, timeout=60):
        return subprocess.CompletedProcess(args, 126, stdout="", stderr="dismissed")

    monkeypatch.setattr(srv, "_run_power", fake_run)
    res = client.post("/api/power/setup")
    assert res.status_code == 403


# ── helper script ──────────────────────────────────────────────────────────
def _fake_smi(tmp_path: Path, initial: int = 250, fail_pl: bool = False):
    limit_file = tmp_path / "limit"
    limit_file.write_text(str(initial))
    smi = tmp_path / "nvidia-smi"
    pl_line = (
        "  -pl) exit 1 ;;"
        if fail_pl
        else '  -pl) echo "$2" > "$limit_file"; exit 0 ;;'
    )
    smi.write_text(
        f"""#!/usr/bin/env bash
limit_file={limit_file}
case "$1" in
  -pm) exit 0 ;;
{pl_line}
esac
case "$*" in
  *power.min_limit*) echo 125.00 ;;
  *power.max_limit*) echo 300.00 ;;
  *power.default_limit*) echo 250.00 ;;
  *power.limit*) cat "$limit_file" ;;
  *power.draw*) echo 8.50 ;;
  *temperature.gpu*) echo 37 ;;
  *persistence_mode*) echo Enabled ;;
  *name*) echo "Fake GPU" ;;
  *) echo "" ;;
esac
"""
    )
    smi.chmod(0o755)
    return smi, limit_file


def _run_helper(smi: Path, state_dir: Path, *args, **extra_env):
    env = os.environ.copy()
    env["PIXELSMITH_SMI"] = str(smi)
    env["PIXELSMITH_POWER_STATE_DIR"] = str(state_dir)
    env["PIXELSMITH_POWER_RETRIES"] = "1"
    env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.run(
        ["bash", str(HELPER), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_helper_set_remembers_value(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    state_dir = tmp_path / "state"
    res = _run_helper(smi, state_dir, "set", "200")
    assert res.returncode == 0, res.stderr
    assert (state_dir / "power").read_text().strip() == "200"
    assert limit_file.read_text().strip() == "200"
    body = json.loads(res.stdout)
    assert body["ok"] is True
    assert body["configured"] == 200
    assert body["limit"] == 200.0


def test_helper_rejects_out_of_range(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    res = _run_helper(smi, tmp_path / "state", "set", "999")
    assert res.returncode == 2
    assert not (tmp_path / "state" / "power").exists()
    assert limit_file.read_text().strip() == "250"


def test_helper_apply_uses_saved_value(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "power").write_text("180\n")
    res = _run_helper(smi, state_dir, "apply")
    assert res.returncode == 0, res.stderr
    assert limit_file.read_text().strip() == "180"
    assert json.loads(res.stdout)["limit"] == 180.0


def test_helper_apply_clamps_saved_value(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "power").write_text("999\n")
    res = _run_helper(smi, state_dir, "apply")
    assert res.returncode == 0
    assert limit_file.read_text().strip() == "300"
    # apply enforces the clamped value but never rewrites the user's choice.
    assert (state_dir / "power").read_text().strip() == "999"


def test_helper_status_defaults_to_card_default(tmp_path):
    smi, _ = _fake_smi(tmp_path)
    res = _run_helper(smi, tmp_path / "state", "status")
    assert res.returncode == 0
    body = json.loads(res.stdout)
    assert body["configured"] == 250
    assert body["min"] == 125.0
    assert body["max"] == 300.0


def test_helper_set_keeps_saved_value_on_failure(tmp_path):
    smi, limit_file = _fake_smi(tmp_path, fail_pl=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "power").write_text("150\n")
    res = _run_helper(smi, state_dir, "set", "200")
    assert res.returncode == 1
    # The app owns the retry: the chosen mode stays saved even when the
    # driver refuses it for a moment.
    assert (state_dir / "power").read_text().strip() == "200"
    assert limit_file.read_text().strip() == "250"
    body = json.loads(res.stdout)
    assert body["ok"] is False
    assert body["configured"] == 200


def test_helper_reset_restores_card_default(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "power").write_text("200\n")
    res = _run_helper(smi, state_dir, "reset")
    assert res.returncode == 0, res.stderr
    assert limit_file.read_text().strip() == "250"
    # reset never forgets the user's last choice.
    assert (state_dir / "power").read_text().strip() == "200"
    body = json.loads(res.stdout)
    assert body["ok"] is True
    assert body["configured"] == 200
    assert body["limit"] == 250.0


def test_helper_reset_without_saved_mode(tmp_path):
    smi, limit_file = _fake_smi(tmp_path)
    res = _run_helper(smi, tmp_path / "state", "reset")
    assert res.returncode == 0
    assert limit_file.read_text().strip() == "250"
    assert json.loads(res.stdout)["configured"] == 250


def test_helper_apply_failure_keeps_saved_value(tmp_path):
    smi, limit_file = _fake_smi(tmp_path, fail_pl=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "power").write_text("200\n")
    res = _run_helper(smi, state_dir, "apply")
    assert res.returncode == 0
    body = json.loads(res.stdout)
    assert body["ok"] is False
    assert body["configured"] == 200
    assert (state_dir / "power").read_text().strip() == "200"
    assert limit_file.read_text().strip() == "250"
