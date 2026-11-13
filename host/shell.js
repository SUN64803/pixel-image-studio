// host shell logic — extracted from index.html so CSP can stay strict (script-src 'self')
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const frame = $("#frame");
  const connecting = $("#connecting");
  const modelPill = $("#modelPill");
  const modelText = $("#modelText");
  const vram = $("#vram");
  const banner = $("#banner");
  const bannerText = $("#bannerText");
  const bannerAction = $("#bannerAction");
  let bannerActionState = null;
  let bannerSeq = 0;
  const loadBtn = $("#loadModel");
  const unloadBtn = $("#unloadModel");

  const STATE = {
    ready: { cls: "ok", text: "พร้อมใช้งาน" },
    loading: { cls: "busy", text: "กำลังโหลดโมเดล…" },
    preparing: { cls: "busy", text: "กำลังดาวน์โหลดโมเดล…" },
    not_loaded: { cls: "warn", text: "ยังไม่โหลดโมเดล" },
    error: { cls: "err", text: "engine ผิดพลาด" },
  };

  function gb(bytes) { return bytes ? (bytes / 1073741824).toFixed(1) : "0"; }

  const blocking = $("#blocking");
  let blockingDepth = 0;   // operations started by the host bar
  let externalBusy = false; // the studio iframe asked for the overlay
  let blockingText = "กำลังเปลี่ยนค่า กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น";

  function updateBlocking() {
    if (!blocking) return;
    const text = $("#blockingText");
    if (text) text.textContent = blockingText;
    blocking.classList.toggle("hidden", !(blockingDepth > 0 || externalBusy));
  }
  function showBlocking(message) {
    blockingDepth += 1;
    if (message) blockingText = message;
    updateBlocking();
  }
  function hideBlocking() {
    blockingDepth = Math.max(0, blockingDepth - 1);
    updateBlocking();
  }

  // The studio runs in an iframe and cannot cover the host bar. It asks the
  // host to show this full-window overlay while it applies a slow setting, so
  // the bar buttons are not clickable either.
  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const msg = event.data;
    if (!msg || msg.pixelsmithBusy !== true) return;
    externalBusy = !!msg.on;
    if (msg.on && msg.text) blockingText = msg.text;
    updateBlocking();
  });

  function showBanner(message, action) {
    bannerSeq += 1;
    if (!message) { banner.classList.remove("show"); bannerActionState = null; return; }
    bannerText.textContent = message;
    bannerActionState = action && typeof action.run === "function" ? action : null;
    if (bannerAction) {
      bannerAction.hidden = !bannerActionState;
      bannerAction.textContent = bannerActionState ? bannerActionState.label || "ลองใหม่" : "";
    }
    banner.classList.add("show");
  }

  function applyHealth(h) {
    const m = (h && h.model) || {};
    const info = STATE[m.state] || STATE.error;
    modelPill.className = "pill " + info.cls;
    modelText.textContent = info.text;
    modelPill.title = m.message || "";
    loadBtn.disabled = m.state === "loading" || m.state === "preparing";
    unloadBtn.disabled = !m.loaded;
    if (h && h.gpu && h.gpu.gpu) {
      vram.textContent = `${h.gpu.gpu} · ${gb(h.gpu.vram_used)}/${gb(h.gpu.vram_total)} GB`;
    } else if (h && h.gpu && h.gpu.torch) {
      vram.textContent = `CPU · torch ${h.gpu.torch}`;
    } else {
      vram.textContent = "";
    }
    if (m.state === "error") showBanner("engine แจ้งข้อผิดพลาด: " + (m.message || "ไม่ทราบสาเหตุ") + " — ดูรายละเอียดที่ปุ่ม log");
    else if (m.state === "not_loaded") showBanner("");
  }

  async function refresh() {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const h = await res.json();
      applyHealth(h);
      connecting.classList.add("hide");
      return h;
    } catch (err) {
      modelPill.className = "pill err";
      modelText.textContent = "เชื่อมต่อ engine ไม่ได้";
      modelPill.title = String(err && err.message || err);
      vram.textContent = "";
      return null;
    }
  }

  loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    showBanner("");
    showBlocking("กำลังโหลดโมเดล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    let poll = null;
    const stopBlocking = () => {
      if (poll) clearInterval(poll);
      poll = null;
      hideBlocking();
    };
    try {
      const res = await fetch("/api/model/load", { method: "POST" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      modelPill.className = "pill busy";
      modelText.textContent = "กำลังโหลดโมเดล…";
      poll = setInterval(async () => {
        const h = await refresh();
        if (!h || !["loading", "preparing"].includes(h.model.state)) stopBlocking();
      }, 900);
    } catch (err) {
      stopBlocking();
      showBanner("โหลดโมเดลไม่สำเร็จ: " + (err.message || err));
      loadBtn.disabled = false;
    }
  });

  unloadBtn.addEventListener("click", async () => {
    showBlocking("กำลังคืนหน่วยความจำโมเดล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      const res = await fetch("/api/model/unload", { method: "POST" });
      if (!res.ok) {
        let msg = "HTTP " + res.status;
        try { const j = await res.json(); msg = j.detail || j.message || msg; } catch (e) {}
        throw new Error(msg);
      }
      showBanner("");
      await refresh();
    } catch (err) {
      showBanner(err.message || String(err));
    } finally {
      hideBlocking();
    }
  });

  $("#openImages").addEventListener("click", async () => {
    try {
      const res = await fetch("/api/open-folder?which=images", { method: "POST" });
      if (!res.ok) throw new Error("HTTP " + res.status);
    } catch (err) {
      showBanner("เปิดโฟลเดอร์ไม่ได้: " + (err.message || err));
    }
  });

  $("#reload").addEventListener("click", () => {
    try { frame.contentWindow.location.reload(); }
    catch (e) { frame.src = "/studio/"; }
  });

  frame.addEventListener("load", () => connecting.classList.add("hide"));

  $("#bannerClose").addEventListener("click", () => showBanner(""));
  if (bannerAction) {
    bannerAction.addEventListener("click", () => {
      const action = bannerActionState;
      if (action) action.run();
    });
  }

  // ── engine crash recovery banner (main process → host) ──────────────────
  // The supervisor auto-restarts a crashed engine; these messages keep the
  // user informed and offer a manual retry after the last attempt fails.
  if (window.pixelApp && typeof pixelApp.onEngineStatus === "function") {
    pixelApp.onEngineStatus((s) => {
      if (!s || !s.state) return;
      if (s.state === "restarting") {
        showBanner(
          `engine หยุดทำงาน — กำลังเริ่มใหม่ (ครั้งที่ ${s.attempt || 1}/${s.max || 3})…`
        );
      } else if (s.state === "recovered") {
        showBanner("engine กลับมาทำงานแล้ว");
        const seq = bannerSeq;
        setTimeout(() => {
          if (seq === bannerSeq) showBanner("");
        }, 6000);
      } else if (s.state === "failed") {
        showBanner(
          `engine หยุดทำงาน: ${s.message || "ไม่ทราบสาเหตุ"} — ดู log หรือลองเริ่มใหม่`,
          {
            label: "ลองใหม่",
            run: () => {
              showBanner("กำลังเริ่ม engine ใหม่…");
              if (window.pixelApp) pixelApp.restartEngine();
            },
          }
        );
      }
    });
  }

  const logModal = $("#logModal");
  async function loadLog() {
    $("#logBody").textContent = "กำลังโหลด…";
    try {
      const res = await fetch("/api/log?tail=400", { cache: "no-store" });
      const data = await res.json();
      $("#logPath").textContent = data.path || "";
      const lines = data.lines || [];
      $("#logBody").textContent = lines.length ? lines.join("\n") : "(log ว่าง — engine ยังไม่ได้เขียนอะไร)";
      $("#logBody").scrollTop = $("#logBody").scrollHeight;
    } catch (err) {
      $("#logBody").textContent = "โหลด log ไม่ได้: " + (err.message || err);
    }
  }
  $("#logs").addEventListener("click", () => { logModal.hidden = false; loadLog(); });
  $("#logClose").addEventListener("click", () => { logModal.hidden = true; });
  $("#logRefresh").addEventListener("click", loadLog);
  logModal.addEventListener("click", (e) => { if (e.target === logModal) logModal.hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") logModal.hidden = true; });

  // ── in-app dialog (replaces native message boxes) ───────────────────────
  // The main process asks for About / install guide / download-finished
  // messages through IPC; rendering them here keeps Chromium's focus and text
  // caret intact (native dialogs can break them on Linux).
  const appDialog = $("#appDialog");
  const appDialogTitle = $("#appDialogTitle");
  const appDialogMessage = $("#appDialogMessage");
  const appDialogDetail = $("#appDialogDetail");
  const appDialogActions = $("#appDialogActions");
  let appDialogId = null;
  let appDialogCancel = 0;

  function closeAppDialog(response) {
    if (!appDialog || appDialog.hidden) return;
    appDialog.hidden = true;
    const id = appDialogId;
    const cancel = appDialogCancel;
    appDialogId = null;
    appDialogActions.textContent = "";
    if (id != null && window.pixelApp && window.pixelApp.dialogResult) {
      window.pixelApp.dialogResult(id, response == null ? cancel : response);
    }
  }

  function showAppDialog(payload) {
    if (!appDialog) return;
    appDialogId = payload.id;
    appDialogCancel = Number.isInteger(payload.cancelId) ? payload.cancelId : 0;
    appDialogTitle.textContent = payload.title || "";
    appDialogMessage.textContent = payload.message || "";
    appDialogDetail.textContent = payload.detail || "";
    appDialogDetail.hidden = !payload.detail;
    appDialogActions.textContent = "";
    const labels = Array.isArray(payload.buttons) && payload.buttons.length
      ? payload.buttons
      : ["OK"];
    const defaultId = Number.isInteger(payload.defaultId) ? payload.defaultId : 0;
    labels.forEach((label, index) => {
      const btn = document.createElement("button");
      btn.type = "button";
      if (index === defaultId) btn.className = "primary";
      btn.textContent = label;
      btn.addEventListener("click", () => closeAppDialog(index));
      appDialogActions.appendChild(btn);
    });
    appDialog.hidden = false;
    const first = appDialogActions.querySelector(".primary") || appDialogActions.firstElementChild;
    if (first) first.focus();
  }

  if (window.pixelApp && window.pixelApp.onAppDialog) {
    window.pixelApp.onAppDialog((payload) => { if (payload) showAppDialog(payload); });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !appDialog || appDialog.hidden) return;
    e.preventDefault();
    closeAppDialog(null);
  });

  refresh();
  setInterval(refresh, 3000);
})();
