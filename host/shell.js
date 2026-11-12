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
  let blockingDepth = 0;
  function showBlocking(message) {
    if (!blocking) return;
    blockingDepth += 1;
    const text = $("#blockingText");
    if (text && message) text.textContent = message;
    blocking.classList.remove("hidden");
  }
  function hideBlocking() {
    blockingDepth = Math.max(0, blockingDepth - 1);
    if (blockingDepth > 0) return;
    if (blocking) blocking.classList.add("hidden");
  }

  function showBanner(message) {
    if (!message) { banner.classList.remove("show"); return; }
    bannerText.textContent = message;
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

  refresh();
  setInterval(refresh, 3000);
})();
