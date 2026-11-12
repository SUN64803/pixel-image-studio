// ─────────────────────────────────────────────────────────────────────────────
// PixelSmith Image Studio — the desktop program (PART A: the window).
//
//   * This file is the window AND the process supervisor. It starts the Python
//     engine (engine/server.py) on loopback, waits until it answers, then points
//     a Chromium view at the page that engine serves. The image generator itself
//     is completely separate: replace studio/ + engine/ and this file still works.
//
//   * The engine serves everything (host page, studio page, generated files, API)
//     on one fixed loopback origin, so the page's fetch() calls are same-origin
//     and nothing needs a custom protocol or file:// URL.
//
//   * Nothing leaves the machine: the server binds 127.0.0.1 on a fixed port.
//     The port must stay fixed because browser storage is keyed by origin.
// ─────────────────────────────────────────────────────────────────────────────

const { app, BrowserWindow, Menu, shell, ipcMain, dialog, session } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const path = require("path");

const isDev = !app.isPackaged;
// dev: ../engine, ../studio, ../host next to this folder. packaged: copied into
// Contents/Resources by package.json's extraResources.
const APP_ROOT = isDev ? path.join(__dirname, "..") : process.resourcesPath;
const PORT = 47824;
const HOST = "127.0.0.1";
const BASE_URL = `http://${HOST}:${PORT}/`;
// Writable app home. In dev this is the project folder itself; in packaged
// builds (AppImage) the bundle is read-only, so data/models/venv live in the
// user's data home instead. Override with IMGS_HOME (testing / portable use).
const HOME_DIR = isDev
  ? APP_ROOT
  : process.env.IMGS_HOME ||
    path.join(
      process.env.XDG_DATA_HOME ||
        path.join(app.getPath("home"), ".local", "share"),
      "PixelSmithImageStudio"
    );
const DATA = isDev ? path.join(APP_ROOT, "data") : path.join(HOME_DIR, "data");
const LOG_FILE = path.join(DATA, "engine.log");
const ENGINE_DIR = path.join(APP_ROOT, "engine");
// venv: engine/.venv in dev; a writable home dir when packaged
const VENV_DIR = isDev ? path.join(ENGINE_DIR, ".venv") : path.join(HOME_DIR, "venv");
const MODELS_DIR = isDev ? path.join(APP_ROOT, "models") : path.join(HOME_DIR, "models");
const ROOT = APP_ROOT; // legacy alias: read-only program root (dialogs/labels)

let win = null;
let engineProc = null;
let engineStarting = false;
let engineOwned = false;
let quitting = false;
let powerResetStarted = false;
let downloadHookInstalled = false;

// Stable WM class so Linux panels/taskbars use the program's icon + name.
app.commandLine.appendSwitch("class", "pixelsmith-image");
if (process.platform === "win32") app.setAppUserModelId("org.pixelsmith.image");

// ── engine supervision ──────────────────────────────────────────────────────
function enginePython() {
  return process.platform === "win32"
    ? path.join(VENV_DIR, "Scripts", "python.exe")
    : path.join(VENV_DIR, "bin", "python");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ping(timeout = 1200) {
  return new Promise((resolve) => {
    const req = http.get(
      { host: HOST, port: PORT, path: "/api/health", timeout },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            const data = JSON.parse(body);
            resolve(!!(data && data.ok === true));
          } catch (e) {
            resolve(false);
          }
        });
      }
    );
    req.on("timeout", () => { req.destroy(); resolve(false); });
    req.on("error", () => resolve(false));
  });
}

function postJson(pathname, body, timeout = 8000) {
  return new Promise((resolve) => {
    const payload = JSON.stringify(body || {});
    const req = http.request(
      {
        host: HOST,
        port: PORT,
        path: pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
        timeout,
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            body: data,
          });
        });
      }
    );
    req.on("timeout", () => { req.destroy(); resolve({ ok: false, timeout: true }); });
    req.on("error", (err) => resolve({ ok: false, error: String(err && err.message || err) }));
    req.end(payload);
  });
}

// Keep the engine log from growing without bound: once it passes 5 MB it is
// renamed to engine.log.1 (replacing the previous one) before the engine
// reopens it.
function rotateLog(file) {
  try {
    if (fs.statSync(file).size < 5 * 1024 * 1024) return;
    const old = file + ".1";
    try { fs.unlinkSync(old); } catch (e) { /* no previous log */ }
    fs.renameSync(file, old);
  } catch (e) {
    /* first run: nothing to rotate */
  }
}

function startEngineProcess() {
  const py = enginePython();
  if (!fs.existsSync(py)) {
    return { ok: false, reason: "missing-venv", py };
  }
  try {
    fs.mkdirSync(DATA, { recursive: true });
    rotateLog(LOG_FILE);
    const out = fs.openSync(LOG_FILE, "a");
    fs.writeSync(out, `\n\n──── ${new Date().toISOString()} starting engine ────\n`);
    engineProc = spawn(py, [path.join(ENGINE_DIR, "server.py")], {
      cwd: ENGINE_DIR,
      env: {
        ...process.env,
        IMGS_HOST: HOST,
        IMGS_PORT: String(PORT),
        IMGS_DATA: DATA,
        IMGS_MODELS: MODELS_DIR,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", out, out],
      windowsHide: true,
    });
    engineProc.on("error", (err) => {
      console.error("[engine] spawn failed:", err);
    });
    engineProc.on("exit", (code, signal) => {
      if (!quitting) console.log(`[engine] exited code=${code} signal=${signal}`);
      engineProc = null;
    });
    engineOwned = true;
    return { ok: true, proc: engineProc };
  } catch (err) {
    return { ok: false, reason: "spawn", error: String(err && err.message || err) };
  }
}

async function waitForEngine(proc, timeoutMs = 180000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (quitting) return { ok: false, reason: "quitting" };
    if (proc && proc.exitCode !== null) {
      // It may have exited because another copy of the engine already owns the
      // port — in that case a healthy engine is still there to use.
      if (await ping()) return { ok: true, reused: true };
      return { ok: false, reason: "exited", code: proc.exitCode };
    }
    if (await ping()) return { ok: true };
    await sleep(600);
  }
  return { ok: false, reason: "timeout" };
}

async function ensureEngine() {
  if (await ping()) return { ok: true, reused: true };
  const started = startEngineProcess();
  if (!started.ok) return started;
  return waitForEngine(started.proc);
}

function stopEngine() {
  if (!engineProc) return;
  const proc = engineProc;
  engineProc = null;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true });
    } else {
      proc.kill("SIGTERM");
      setTimeout(() => {
        try { if (proc.exitCode === null) proc.kill("SIGKILL"); } catch (e) {}
      }, 4000).unref();
    }
  } catch (e) {
    /* already gone */
  }
}

// Power mode is owned by the app: on quit, ask the engine to restore the
// card's default limit (the saved choice stays for the next launch).
function resetPowerOnQuit() {
  if (process.platform !== "linux" || !engineOwned) {
    return Promise.resolve({ ok: true, skipped: true });
  }
  return postJson("/api/power/reset", {}, 8000).catch(() => ({ ok: false }));
}

function spawnPowerResetFallback() {
  if (process.platform !== "linux" || !engineOwned) return;
  try {
    // Fixed command: the helper's sudoers rule allows `reset` without a
    // password.  Detached so it survives this process exiting.
    const child = spawn(
      "sh",
      ["-c", "sleep 4; exec sudo -n /usr/local/bin/pixelsmith-power reset"],
      { detached: true, stdio: "ignore" }
    );
    child.unref();
  } catch (e) {
    /* helper not installed — nothing else to do */
  }
}

// ── tiny local pages shown while the engine boots / when it cannot start ────
function htmlDataUrl(html) {
  return "data:text/html;charset=utf-8," + encodeURIComponent(html);
}

const PAGE_CSS = `
  html,body{margin:0;height:100%;background:#0b0f14;color:#e6edf3;overflow:hidden;
    font:13px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Thai",sans-serif}
  body{display:grid;place-items:center;
    background:radial-gradient(760px 420px at 50% 30%,rgba(139,92,246,.13),transparent 65%),#0b0f14}
  .wrap{max-width:660px;padding:34px;text-align:center}
  .mark{display:grid;place-items:center;width:52px;height:52px;margin:0 auto 18px;border-radius:15px;
    background:linear-gradient(135deg,rgba(139,92,246,.28),rgba(34,211,238,.22));
    border:1px solid rgba(139,92,246,.45);color:#c4b5fd}
  h1{margin:0 0 8px;font-size:18px;letter-spacing:.2px}
  p{margin:8px 0;color:#93a3b5}
  .spin{width:30px;height:30px;margin:0 auto 18px;border-radius:50%;
    border:2.5px solid rgba(139,92,246,.25);border-top-color:#8b5cf6;animation:s .8s linear infinite}
  @keyframes s{to{transform:rotate(360deg)}}
  code,pre{font-family:ui-monospace,"JetBrains Mono",monospace}
  code{background:#0a1017;border:1px solid #1c2734;border-radius:6px;padding:1px 6px;font-size:12px;color:#a5e9f5}
  pre{text-align:left;background:#0a1017;border:1px solid #1c2734;border-radius:10px;padding:12px 14px;
    overflow:auto;font-size:12px;color:#b7f0fb;white-space:pre-wrap;word-break:break-word;margin:14px 0}
  .note{color:#fbbf24;font-size:12px}
  .muted{color:#6f7f91;font-size:11.5px}
`;

function loadingPage() {
  return `<!doctype html><html lang="th"><head><meta charset="utf-8"><title>PixelSmith Image Studio</title>
  <style>${PAGE_CSS}</style></head><body><div class="wrap">
  <div class="spin"></div>
  <hr style="display:none">
  <h1>กำลังเปิด Image Studio…</h1>
  <p>กำลังเริ่ม engine และเตรียมพื้นที่ทำงาน</p>
  <p class="muted">ครั้งแรกอาจใช้เวลาสักครู่ (Python + torch)</p>
  </div></body></html>`;
}

function escapeHtml(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function errorPage(result) {
  const reason = result && result.reason;
  let headline = "เปิด engine ไม่สำเร็จ";
  let body = "";
  if (reason === "missing-venv") {
    headline = "ยังไม่ได้ติดตั้ง engine";
    body = `
      <p>ไม่พบ Python virtual environment ที่ <code>engine/.venv</code></p>
      <p>เปิด terminal แล้วรันสคริปต์ติดตั้ง (ดาวน์โหลด torch + โมเดล ครั้งแรกครั้งเดียว):</p>
      <pre>cd "${escapeHtml(ROOT)}"
bash engine/setup.sh</pre>
      <p class="note">ถ้ามีไฟล์ .pth อยู่แล้ว ใช้ <code>bash engine/setup.sh --from-pth "/path/to/SANA1.5_1.6B_1024px.pth"</code></p>`;
  } else if (reason === "timeout") {
    headline = "engine ใช้เวลานานเกินไป";
    body = `<p>engine ยังไม่ตอบสนองภายในเวลาที่กำหนด — ดู log เพื่อหาสาเหตุ</p>`;
  } else if (reason === "exited") {
    headline = "engine ปิดตัวเองทันที";
    body = `<p>engine ออกด้วยรหัส <code>${escapeHtml(result.code)}</code> — มักเกิดจาก dependency ไม่ครบหรือพอร์ตถูกใช้</p>`;
  } else {
    body = `<p>${escapeHtml((result && (result.error || result.reason)) || "ไม่ทราบสาเหตุ")}</p>`;
  }
  return `<!doctype html><html lang="th"><head><meta charset="utf-8"><title>PixelSmith Image Studio</title>
  <style>${PAGE_CSS}</style></head><body><div class="wrap">
  <h1>${escapeHtml(headline)}</h1>
  ${body}
  <p><b>log:</b> <code>${escapeHtml(LOG_FILE)}</code></p>
  <p class="muted">แก้ไขแล้วปิดหน้าต่างนี้แล้วเปิดโปรแกรมใหม่ (หรือใช้เมนู File → ลองเริ่ม engine ใหม่)</p>
  </div></body></html>`;
}

// ── first-run installer (packaged builds) ───────────────────────────────────
// On a clean machine the AppImage bundle is read-only and has no Python env.
// This pipeline creates the venv, installs the engine packages (torch) and
// downloads the SANA model into the writable home dir — with live progress.
let installRunning = false;

function installerLang() {
  const locales = process.env.LANG || app.getLocale() || "en";
  return String(locales).toLowerCase().startsWith("th") ? "th" : "en";
}

function installerSteps(lang) {
  const th = lang === "th";
  return [
    {
      id: "setup",
      label: th
        ? "สร้าง Python environment + ติดตั้งแพ็กเกจ (torch ~5 GB — ครั้งแรก)"
        : "Create the Python environment and install packages (torch ~5 GB, first run)",
    },
    {
      id: "model",
      label: th
        ? "โหลดโมเดลเริ่มต้น SANA 1.5 · 1.6B · 1024px (~9 GB)"
        : "Download the default SANA 1.5 · 1.6B · 1024px model (~9 GB)",
    },
    { id: "engine", label: th ? "เริ่ม engine" : "Start the engine" },
  ];
}

function sendInstall(win, payload) {
  try {
    if (win && !win.isDestroyed()) win.webContents.send("install:update", payload);
  } catch (e) {
    /* window gone mid-install */
  }
}

function runSpawn(cmd, args, env, onLine) {
  return new Promise((resolve) => {
    const proc = spawn(cmd, args, {
      cwd: ENGINE_DIR,
      env: {
        ...process.env,
        ...env,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let buf = "";
    const pump = (chunk) => {
      buf += chunk.toString();
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).replace(/\r.*/g, "").trimEnd();
        buf = buf.slice(idx + 1);
        if (line.trim()) onLine(line);
      }
    };
    proc.stdout.on("data", pump);
    proc.stderr.on("data", pump);
    proc.on("error", (err) => resolve({ ok: false, error: String(err) }));
    proc.on("exit", (code) => resolve({ ok: code === 0, code }));
  });
}

async function runInstaller(win) {
  if (installRunning) return { ok: false, busy: true };
  installRunning = true;
  const lang = installerLang();
  const th = lang === "th";
  const steps = installerSteps(lang);
  const send = (payload) => sendInstall(win, payload);
  send({ type: "start", steps, home: HOME_DIR, log: LOG_FILE, lang });
  const result = { ok: false };
  try {
    fs.mkdirSync(DATA, { recursive: true });
    fs.mkdirSync(MODELS_DIR, { recursive: true });

    // ── step 1: venv + packages (setup.sh bootstraps uv and installs torch) ──
    send({ type: "step", id: "setup", status: "running" });
    const r1 = await runSpawn(
      "bash",
      [path.join(ENGINE_DIR, "setup.sh"), "--no-model"],
      { IMGS_VENV: VENV_DIR, IMGS_DATA: DATA, IMGS_MODELS: MODELS_DIR },
      (line) => send({ type: "log", id: "setup", line })
    );
    if (!r1.ok) {
      send({
        type: "step",
        id: "setup",
        status: "error",
        error: th
          ? `ติดตั้งแพ็กเกจไม่สำเร็จ (exit ${r1.code ?? "?"}) ${r1.error || ""}`
          : `Package install failed (exit ${r1.code ?? "?"}) ${r1.error || ""}`,
      });
      return result;
    }
    send({ type: "step", id: "setup", status: "done" });

    // ── step 2: download the default SANA model ────────────────────────────
    if (process.platform !== "win32") {
      send({ type: "step", id: "model", status: "running" });
      const r2 = await runSpawn(
        enginePython(),
        [path.join(ENGINE_DIR, "prepare_model.py")],
        { IMGS_DATA: DATA, IMGS_MODELS: MODELS_DIR },
        (line) => send({ type: "log", id: "model", line })
      );
      if (!r2.ok) {
        send({
          type: "step",
          id: "model",
          status: "error",
          error: th
            ? `โหลดโมเดลไม่สำเร็จ (exit ${r2.code ?? "?"}) ${r2.error || ""}`
            : `Model download failed (exit ${r2.code ?? "?"}) ${r2.error || ""}`,
        });
        return result;
      }
      send({ type: "step", id: "model", status: "done" });
    }

    result.ok = true;
    try {
      fs.writeFileSync(
        path.join(HOME_DIR, ".installed"),
        new Date().toISOString() + "\n"
      );
    } catch (e) {
      /* non-fatal — next launch just re-checks */
    }
    return result;
  } finally {
    installRunning = false;
  }
}

function installerPage() {
  const th = installerLang() === "th";
  const T = th
    ? {
        title: "ติดตั้ง PixelSmith Image Studio",
        sub: "กำลังติดตั้งทุกอย่างให้อัตโนมัติ — ต้องต่ออินเทอร์เน็ตครั้งแรกเท่านั้น",
        done: "ติดตั้งเสร็จ — กำลังเปิดโปรแกรม…",
        failed: "ติดตั้งไม่สำเร็จ",
        retryHint:
          "แก้ไขแล้วลองใหม่จากเมนู File → ลองเริ่ม engine ใหม่ (ติดตั้งต่อจากจุดที่ค้างอัตโนมัติ)",
        logLabel: "log ล่าสุด",
        home: "โฟลเดอร์ข้อมูล",
      }
    : {
        title: "Installing PixelSmith Image Studio",
        sub: "Setting everything up automatically — internet needed on the first run only",
        done: "Install finished — starting the app…",
        failed: "Install failed",
        retryHint:
          "Once fixed, retry from File → Restart engine (it resumes where it left off)",
        logLabel: "Recent log",
        home: "Data folder",
      };
  return `<!doctype html><html lang="${th ? "th" : "en"}"><head><meta charset="utf-8">
  <title>${T.title}</title>
  <style>${PAGE_CSS}
    .steps{max-width:660px;margin:0 auto;text-align:left}
    .step{display:flex;align-items:center;gap:10px;padding:9px 12px;border:1px solid #1c2734;
      border-radius:10px;margin:8px 0;background:#0d141c}
    .step .ico{width:22px;height:22px;display:grid;place-items:center;border-radius:50%;
      border:2px solid #2a3644;color:#8b98a8;font-size:12px;flex:0 0 auto}
    .step.running .ico{border-top-color:#8b5cf6;animation:s .8s linear infinite;color:transparent}
    .step.done .ico{border-color:#34d399;color:#34d399}
    .step.error .ico{border-color:#f87171;color:#f87171}
    .step.done{border-color:rgba(52,211,153,.4)}
    .step.error{border-color:rgba(248,113,113,.5)}
    #log{max-height:170px;text-align:left;font-size:11.5px}
    .err{color:#fca5a5;font-size:12.5px;margin:10px 0 0;text-align:center}
    .wrap{max-width:760px}
  </style></head><body><div class="wrap">
  <div class="spin" id="spin"></div>
  <h1>${T.title}</h1>
  <p>${T.sub}</p>
  <div class="steps" id="steps"></div>
  <pre id="log"></pre>
  <p class="err" id="err" style="display:none"></p>
  <p class="muted">${T.home}: <span id="homeDir"></span></p>
  <p class="muted" id="hint" style="display:none">${T.retryHint}</p>
  </div>
  <script>
  (function () {
    "use strict";
    const HOME = ${JSON.stringify(HOME_DIR)};
    const ICONS = { pending: "·", running: "", done: "✓", error: "✕" };
    const stepsEl = document.getElementById("steps");
    const logEl = document.getElementById("log");
    const errEl = document.getElementById("err");
    const spin = document.getElementById("spin");
    const hint = document.getElementById("hint");
    document.getElementById("homeDir").textContent = HOME;
    let order = [];
    function renderSteps(steps) {
      order = steps;
      stepsEl.innerHTML = steps
        .map(
          (s) =>
            '<div class="step pending" data-id="' + s.id + '">' +
            '<span class="ico"></span><span>' + s.label + "</span></div>"
        )
        .join("");
    }
    function setStep(id, status, text) {
      const el = stepsEl.querySelector('[data-id="' + id + '"]');
      if (!el) return;
      el.className = "step " + status;
      el.querySelector(".ico").textContent = ICONS[status] || "·";
      if (status === "error" && text) {
        errEl.textContent = text;
        errEl.style.display = "block";
        spin.style.display = "none";
        hint.style.display = "block";
      }
    }
    function pushLog(line) {
      logEl.textContent = (logEl.textContent.split("\\n").slice(-7).join("\\n") +
        "\\n" + line).split("\\n").slice(-8).join("\\n");
      logEl.scrollTop = logEl.scrollHeight;
    }
    if (window.pixelApp && window.pixelApp.onInstallUpdate) {
      window.pixelApp.onInstallUpdate((p) => {
        if (p.type === "start") renderSteps(p.steps);
        else if (p.type === "log") pushLog(p.line);
        else if (p.type === "step") setStep(p.id, p.status, p.error);
      });
    }
  })();
  </script>
  </body></html>`;
}

const INSTALLER_URL = htmlDataUrl(installerPage());
const LOADING_URL = htmlDataUrl(loadingPage());

async function launch() {
  if (!win || win.isDestroyed()) return;
  if (
    app.isPackaged &&
    process.platform !== "win32" &&
    (!fs.existsSync(enginePython()) ||
      !fs.existsSync(path.join(HOME_DIR, ".installed")))
  ) {
    // First run (or an interrupted install): auto-install venv + packages +
    // model. The installer is idempotent and resumes where it left off.
    await win.loadURL(INSTALLER_URL).catch(() => {});
    const ok = await runInstaller(win);
    if (!win || win.isDestroyed()) return;
    if (!ok) {
      try {
        await win.webContents.executeJavaScript(
          `document.getElementById("spin").style.display="none";
           document.getElementById("hint").style.display="block";`
        );
      } catch (e) {
        /* page may be closed */
      }
      return;
    }
  }
  await win.loadURL(LOADING_URL).catch(() => {});
  const result = await ensureEngine();
  if (!win || win.isDestroyed()) return;
  if (result.ok) {
    await win.loadURL(BASE_URL).catch((err) => {
      dialog.showErrorBox(
        "เปิดหน้า Image Studio ไม่ได้",
        `โหลด ${BASE_URL} ไม่สำเร็จ (${(err && err.message) || err})`
      );
    });
  } else {
    await win.loadURL(htmlDataUrl(errorPage(result))).catch(() => {});
  }
}

async function restartEngine() {
  if (engineStarting) return;
  engineStarting = true;
  try {
    stopEngine();
    await sleep(600);
    await launch();
  } finally {
    engineStarting = false;
  }
}

// Native dialogs can leave Chromium's focus/caret state stale on Linux.
// Re-focus the window and its web contents once they close.
function refocusWindow() {
  if (!win || win.isDestroyed()) return;
  try {
    win.focus();
    win.webContents.focus();
  } catch (e) {
    /* window is going away */
  }
}

// In-app message boxes. The host page renders the payload and replies with the
// index of the clicked button, so Chromium never loses focus/caret state.
// While the window shows the loading or first-run installer page the host is
// not there to render it, so fall back to a native box (no caret to break).
let appDialogSeq = 0;
const pendingAppDialogs = new Map();

function showAppDialog(options = {}) {
  const opts = {
    title: options.title || "",
    message: options.message || "",
    detail: options.detail || "",
    buttons: Array.isArray(options.buttons) && options.buttons.length ? options.buttons : ["OK"],
    defaultId: Number.isInteger(options.defaultId) ? options.defaultId : 0,
    cancelId: Number.isInteger(options.cancelId) ? options.cancelId : 0,
  };
  const hostReady =
    win && !win.isDestroyed() &&
    String(win.webContents.getURL() || "").startsWith(BASE_URL);
  if (!hostReady) {
    const optsNative = {
      type: "info",
      title: opts.title,
      message: opts.message,
      detail: opts.detail,
      buttons: opts.buttons,
      defaultId: opts.defaultId,
      cancelId: opts.cancelId,
    };
    const box = win && !win.isDestroyed()
      ? dialog.showMessageBox(win, optsNative)
      : dialog.showMessageBox(optsNative);
    return box
      .then((r) => { refocusWindow(); return r.response; })
      .catch(() => { refocusWindow(); return opts.cancelId; });
  }
  return new Promise((resolve) => {
    const id = ++appDialogSeq;
    pendingAppDialogs.set(id, { resolve, cancelId: opts.cancelId });
    win.webContents.send("app-dialog", { id, ...opts });
  });
}

ipcMain.on("app-dialog-result", (_ev, msg) => {
  if (!msg || !pendingAppDialogs.has(msg.id)) return;
  const pending = pendingAppDialogs.get(msg.id);
  pendingAppDialogs.delete(msg.id);
  pending.resolve(Number.isInteger(msg.response) ? msg.response : 0);
  refocusWindow();
});

function resolvePendingAppDialogs() {
  for (const [id, pending] of pendingAppDialogs) {
    pendingAppDialogs.delete(id);
    pending.resolve(pending.cancelId);
  }
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1000,
    minHeight: 620,
    backgroundColor: "#0b0f14",
    title: "PixelSmith Image Studio",
    show: false,
    autoHideMenuBar: false,
    icon: path.join(__dirname, "icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });

  win.once("ready-to-show", () => win.show());

  if (!downloadHookInstalled) {
    downloadHookInstalled = true;
    session.defaultSession.on("will-download", (event, item) => {
      item.once("done", (e, state) => {
        if (state !== "completed") return;
        const target = item.getSavePath();
        if (!target) return;
        showAppDialog({
          title: "ดาวน์โหลดเสร็จแล้ว",
          message: "บันทึก " + path.basename(target),
          detail: target,
          buttons: ["เปิดโฟลเดอร์", "ปิด"],
          defaultId: 0,
          cancelId: 1,
        }).then((response) => {
          if (response === 0) shell.showItemInFolder(target);
        });
      });
    });
  }

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });

  win.on("closed", () => { win = null; resolvePendingAppDialogs(); });
  await launch();
}

// ── menu ────────────────────────────────────────────────────────────────────
function buildMenu() {
  const template = [
    {
      label: "File",
      submenu: [
        { label: "Reload page", accelerator: "CmdOrCtrl+R", click: () => win && win.webContents.reload() },
        { label: "ลองเริ่ม engine ใหม่", click: restartEngine },
        { type: "separator" },
        { label: "เปิดโฟลเดอร์ภาพ", click: () => shell.openPath(path.join(DATA, "images")) },
        { label: "เปิดโฟลเดอร์ข้อมูล", click: () => shell.openPath(DATA) },
        { label: "เปิดโฟลเดอร์โปรแกรม", click: () => shell.openPath(ROOT) },
        { type: "separator" },
        { role: "quit", label: "Quit" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" }, { role: "redo" }, { type: "separator" },
        { role: "cut" }, { role: "copy" }, { role: "paste" }, { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        { label: "Fullscreen", accelerator: "F11", click: () => win && win.setFullScreen(!win.isFullScreen()) },
        { type: "separator" },
        { role: "zoomIn" }, { role: "zoomOut" }, { role: "resetZoom" },
        { type: "separator" },
        { role: "toggleDevTools" },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "คู่มือติดตั้ง",
          click: () => showAppDialog({
            title: "ติดตั้ง Image Studio",
            message: "ติดตั้งครั้งแรกครั้งเดียว",
            detail:
              "เปิด terminal แล้วรัน:\n\n" +
              `  cd "${ROOT}"\n` +
              "  bash engine/setup.sh\n\n" +
              "จากนั้นเปิดโปรแกรมด้วย:\n\n" +
              "  bash electron/linux-setup.sh\n\n" +
              `โฟลเดอร์ข้อมูล: ${DATA}`,
            buttons: ["เปิดโฟลเดอร์โปรแกรม", "ปิด"],
            defaultId: 0,
            cancelId: 1,
          }).then((response) => {
            if (response === 0) shell.openPath(ROOT);
          }),
        },
        {
          label: "About PixelSmith Image Studio",
          click: () => showAppDialog({
            title: "About",
            message: "PixelSmith Image Studio",
            detail:
              `เวอร์ชัน ${app.getVersion()} — สร้างภาพด้วย SANA 1.5 (1.6B · 1024px) บนเครื่องนี้\n` +
              "ไม่ต้องต่ออินเทอร์เน็ตหลังติดตั้ง ไม่มีบัญชี ไม่มีเซิร์ฟเวอร์\n\n" +
              `Electron ${process.versions.electron} · Chromium ${process.versions.chrome} · ${process.platform} ${process.arch}\n` +
              `Engine: ${BASE_URL}\n` +
              `Program folder: ${ROOT}`,
            buttons: ["OK"],
          }),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── IPC bridge (small, for the host page's convenience buttons) ─────────────
ipcMain.handle("open-folder", (_ev, which) => {
  const allowed = {
    root: HOME_DIR,
    images: path.join(DATA, "images"),
    thumbs: path.join(DATA, "thumbs"),
    data: DATA,
    models: MODELS_DIR,
    logs: DATA,
  };
  return shell.openPath(allowed[String(which || "root")] || HOME_DIR);
});
ipcMain.handle("app-info", () => ({
  platform: process.platform,
  version: app.getVersion(),
  root: ROOT,
  data: DATA,
  engine: BASE_URL,
}));

// ── lifecycle ───────────────────────────────────────────────────────────────
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(async () => {
    app.setName("PixelSmith Image Studio");
    try {
      await session.defaultSession.clearCache();
    } catch (e) {
      console.error("clearCache failed:", e);
    }
    await createWindow();
    buildMenu();
    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("before-quit", (event) => {
    if (powerResetStarted) {
      quitting = true;
      return;
    }
    powerResetStarted = true;
    event.preventDefault();
    quitting = true;
    resetPowerOnQuit().finally(() => {
      stopEngine();
      // The engine may still be releasing the GPU when the request above ran;
      // retry once through the helper after it has had a moment to exit.
      spawnPowerResetFallback();
      setTimeout(() => app.quit(), 150);
    });
  });
  app.on("will-quit", () => stopEngine());
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });
}
