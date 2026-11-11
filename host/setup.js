// setup page logic — extracted so CSP can stay strict (script-src 'self')
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
      const old = btn.textContent;
      btn.textContent = "คัดลอกแล้ว ✓";
      setTimeout(() => { btn.textContent = old; }, 1400);
    } catch (err) {
      btn.textContent = "คัดลอกไม่ได้";
      setTimeout(() => { btn.textContent = "คัดลอก"; }, 1400);
    }
  });

  const enginePill = $("#enginePill"), engineText = $("#engineText");
  const modelPill = $("#modelPill"), modelText = $("#modelText");
  const gpuPill = $("#gpuPill"), gpuText = $("#gpuText");
  const stateLine = $("#stateLine");

  function setPill(el, textEl, cls, text) {
    el.className = "pill " + cls;
    textEl.textContent = text;
  }

  async function check() {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const h = await res.json();
      setPill(enginePill, engineText, "ok", "engine ทำงานอยู่");
      const m = h.model || {};
      if (m.files_ready) setPill(modelPill, modelText, "ok", "ไฟล์โมเดลครบ");
      else if (m.state === "preparing") setPill(modelPill, modelText, "warn", "กำลังดาวน์โหลดโมเดล");
      else setPill(modelPill, modelText, "warn", "ยังไม่มีไฟล์โมเดล");
      const g = h.gpu || {};
      if (g.gpu) setPill(gpuPill, gpuText, "ok", g.gpu);
      else setPill(gpuPill, gpuText, "err", "ไม่พบ GPU / torch");
      stateLine.textContent = m.state === "ready"
        ? "พร้อมใช้งานแล้ว — กลับไปที่หน้า Image Studio ได้เลย"
        : "engine พร้อม แต่โมเดลจะถูกโหลด/ดาวน์โหลดอัตโนมัติเมื่อกดสร้างภาพครั้งแรก";
    } catch (err) {
      setPill(enginePill, engineText, "err", "engine ไม่ทำงาน");
      setPill(modelPill, modelText, "err", "ตรวจไม่ได้");
      setPill(gpuPill, gpuText, "err", "ตรวจไม่ได้");
      stateLine.textContent = "ยังไม่ได้เปิด engine — ติดตั้งตามขั้นตอนด้านบนแล้วเปิดโปรแกรม";
    }
    try {
      const res = await fetch("/api/log?tail=120", { cache: "no-store" });
      const data = await res.json();
      const lines = data.lines || [];
      $("#logBox").textContent = lines.length ? lines.join("\n") : "—";
      $("#logBox").scrollTop = $("#logBox").scrollHeight;
    } catch (e) {
      $("#logBox").textContent = "โหลด log ไม่ได้ (engine ยังไม่ทำงาน)";
    }
  }

  $("#retry").addEventListener("click", () => {
    if (window.pixelApp && window.pixelApp.openProgramFolder) window.pixelApp.openProgramFolder();
    check();
  });

  check();
  setInterval(check, 4000);
})();
