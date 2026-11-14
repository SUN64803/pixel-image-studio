/* PixelSmith Image Studio — image generator page logic (vanilla, offline). */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  // translate a composed UI string (confirm() dialogs etc.) — the dictionary
  // lives in i18n.js; Thai source text falls through unchanged
  const tr = (s) => (window.I18N && I18N.tr(s)) || s;

  const DEFAULTS = {
    prompt: "",
    negative_prompt: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 4.5,
    seed: -1,
    batch: 1,
    enhance: true,
    sampler: "dpm",
    model_id: "",
    loras: [],
    regions_enabled: false,
    region_layout: "2",
    regions: [],
    inpaint: false,
    init_image: "",
    mask_image: "",
    strength: 0.75,
    hires_enabled: false,
    hires_scale: 1.5,
    hires_strength: 0.35,
  };

  const REGION_LAYOUTS = {
    "2": { labels: ["ซ้าย", "ขวา"], boxes: [[0, 0, 0.5, 1], [0.5, 0, 0.5, 1]] },
    "3": {
      labels: ["ซ้าย", "กลาง", "ขวา"],
      boxes: [[0, 0, 1 / 3, 1], [1 / 3, 0, 1 / 3, 1], [2 / 3, 0, 1 / 3, 1]],
    },
    "4": {
      labels: ["บนซ้าย", "บนขวา", "ล่างซ้าย", "ล่างขวา"],
      boxes: [[0, 0, 0.5, 0.5], [0.5, 0, 0.5, 0.5], [0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
    },
  };

  const RATIOS = [
    { label: "1:1", w: 1024, h: 1024 },
    { label: "4:3", w: 1152, h: 864 },
    { label: "3:4", w: 864, h: 1152 },
    { label: "3:2", w: 1152, h: 768 },
    { label: "2:3", w: 768, h: 1152 },
    { label: "16:9", w: 1280, h: 736 },
    { label: "9:16", w: 736, h: 1280 },
  ];

  const STYLES = [
    { label: "ภาพถ่าย", add: "photorealistic, natural lighting, 50mm lens, ultra detailed, 8k" },
    { label: "อนิเมะ", add: "anime style, clean line art, vibrant colors, detailed illustration" },
    { label: "ภาพวาด", add: "digital painting, expressive brush strokes, dramatic lighting, concept art" },
    { label: "3D", add: "3d render, octane render, soft studio lighting, highly detailed" },
    { label: "ไซเบอร์พังค์", add: "cyberpunk, neon lights, futuristic city, cinematic atmosphere" },
    { label: "แฟนตาซี", add: "fantasy art, epic composition, volumetric light, intricate detail" },
    { label: "ภาพยนตร์", add: "cinematic still, anamorphic lens, film grain, moody lighting" },
    { label: "มินิมอล", add: "minimalist, clean composition, pastel palette, generous negative space" },
  ];

  const IDEAS = [
    "a serene Japanese garden in autumn, koi pond, soft morning mist, photorealistic",
    "a cozy reading nook by a rainy window, warm lamp light, books, cinematic",
    "a neon cyberpunk street market at night, rain, reflections, ultra detailed",
    "a majestic white tiger in a snowy forest, golden hour, wildlife photography",
    "an astronaut floating above a colorful nebula, cinematic sci-fi, 8k",
    "a floating island with waterfalls, fantasy landscape, epic lighting",
    "a delicious bowl of ramen, steam, soft light, food photography, macro",
    "a geometric fox made of glowing crystals, dark background, 3d render",
    "a vintage train station in the mist, moody atmosphere, cinematic still",
    "a portrait of an elderly fisherman, weathered face, dramatic light, photorealistic",
  ];

  const MAX_QUEUE = 8;

  const state = {
    settings: { ...DEFAULTS },
    history: [],
    total: 0,
    jobId: null,
    runningJobId: null,
    liveJobId: null,
    knownActive: new Set(),
    activeCount: 0,
    poll: null,
    pollStart: 0,
    liveJob: null,
    lb: null,
    model: null,
    power: null,
    registry: [],
    currentId: null,
    currentModel: null,
    loras: [],
    healthTimer: null,
    busy: false,
    loraManagerList: [],
    datasets: [],
    activeDs: null,
  };

  // ── tiny helpers ─────────────────────────────────────────────────────────
  const esc = (value) =>
    String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const normalizeSize = (v) => {
    const n = clamp(Math.round((Number(v) || 1024) / 32) * 32, 256, 2048);
    return n;
  };
  const gb = (bytes) => {
    if (!bytes) return "0";
    return (bytes / 1024 ** 3).toFixed(1);
  };
  const fmtClock = (sec) => {
    sec = Math.max(0, Math.round(sec));
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  };
  const fmtDate = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleString("th-TH", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  async function api(path, options = {}) {
    const opts = { ...options };
    opts.headers = { ...(options.headers || {}) };
    // let the engine know the UI language for messages it composes
    if (window.I18N) opts.headers["x-lang"] = I18N.getLang();
    if (!(opts.body instanceof FormData)) {
      opts.headers["content-type"] = "application/json";
    }
    const res = await fetch(path, opts);
    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      /* empty body */
    }
    if (!res.ok) {
      const msg =
        (data && (data.detail || data.message)) || `คำขอถูกปฏิเสธ (HTTP ${res.status})`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  let toastSeq = 0;
  function toast(message, kind = "info", ms = 4200) {
    const box = $("#toasts");
    if (!box) return;
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.dataset.id = String(++toastSeq);
    el.innerHTML = `<span>${esc(tr(message))}</span>`;
    box.appendChild(el);
    setTimeout(() => {
      el.classList.add("out");
      setTimeout(() => el.remove(), 240);
    }, ms);
  }

  // ── in-app confirm dialog ───────────────────────────────────────────────
  // Never use the native confirm()/alert(): in Electron the text caret and
  // keyboard focus can break after the native dialog closes, and only a full
  // app restart brings them back. This async dialog keeps focus in the page.
  const confirmState = { resolve: null, opener: null };

  function uiConfirm(message, options = {}) {
    return new Promise((resolve) => {
      const modal = $("#confirmModal");
      if (!modal) {
        // last-resort fallback (should not happen)
        resolve(false);
        return;
      }
      // A second confirm while one is open cancels the first one.
      if (confirmState.resolve) {
        const prev = confirmState.resolve;
        confirmState.resolve = null;
        prev(false);
      }
      confirmState.resolve = resolve;
      confirmState.opener =
        document.activeElement instanceof HTMLElement ? document.activeElement : null;

      const title = $("#confirmTitle");
      if (title) title.textContent = tr(options.title || "ยืนยันการทำรายการ");
      const text = $("#confirmText");
      if (text) text.textContent = String(message == null ? "" : message);
      const ok = $("#confirmOk");
      if (ok) {
        ok.textContent = tr(options.confirmLabel || "ยืนยัน");
        ok.classList.toggle("danger", options.danger !== false);
      }
      const cancel = $("#confirmCancel");
      if (cancel) cancel.textContent = tr(options.cancelLabel || "ยกเลิก");

      modal.classList.remove("hidden");
      // Safe default: focus Cancel, so Enter never deletes by accident.
      setTimeout(() => cancel && cancel.focus(), 0);
    });
  }

  function closeConfirm(result) {
    const modal = $("#confirmModal");
    if (!modal || modal.classList.contains("hidden")) return;
    modal.classList.add("hidden");
    const resolve = confirmState.resolve;
    const opener = confirmState.opener;
    confirmState.resolve = null;
    confirmState.opener = null;
    if (resolve) resolve(!!result);
    // Put the caret back where the user left it.
    if (opener && document.body.contains(opener) && typeof opener.focus === "function") {
      try {
        opener.focus({ preventScroll: true });
      } catch (e) {
        try {
          opener.focus();
        } catch (e2) {
          /* element gone */
        }
      }
    }
  }

  function bindConfirmDialog() {
    const modal = $("#confirmModal");
    if (!modal) return;
    $("#confirmOk")?.addEventListener("click", () => closeConfirm(true));
    $("#confirmCancel")?.addEventListener("click", () => closeConfirm(false));
    modal.addEventListener("click", (event) => {
      if (event.target.closest("[data-confirm-cancel]")) closeConfirm(false);
    });
    // Keep the dialog's keys away from the app-level shortcuts (Ctrl+Enter,
    // Escape closing the assistant/lightbox, …) and close on Escape.
    modal.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Escape") {
        event.preventDefault();
        closeConfirm(false);
      }
    });
    // Simple focus trap: Tab cycles between the two buttons.
    modal.addEventListener("keydown", (event) => {
      if (event.key !== "Tab") return;
      const focusables = [$("#confirmCancel"), $("#confirmOk")].filter(Boolean);
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
  }

  // ── in-app training manual (bilingual) ──────────────────────────────────
  // The content lives in training-guide.js (Thai + English for every block).
  // Rendering from data keeps the manual independent of the i18n dictionary
  // and lets the language switch re-render the open modal instantly.
  const GUIDE = window.TRAINING_GUIDE;
  const guideState = { opener: null, section: "overview", search: "" };

  function guideLang() {
    return window.I18N ? I18N.getLang() : "th";
  }

  function guideUi() {
    const lang = guideLang();
    return (GUIDE && GUIDE.ui && (GUIDE.ui[lang] || GUIDE.ui.th)) || {};
  }

  function guideText(value) {
    if (!value) return "";
    if (typeof value === "string") return value;
    return value[guideLang()] || value.th || value.en || "";
  }

  function guideBlockText(block) {
    const parts = [];
    if (block.p) parts.push(block.p.th, block.p.en);
    if (block.list) block.list.forEach((item) => parts.push(item.th, item.en));
    if (block.steps) block.steps.forEach((item) => parts.push(item.th, item.en));
    if (block.note) parts.push(block.note.text.th, block.note.text.en);
    if (block.code) parts.push(block.code);
    if (block.table) {
      block.table.head.forEach((cell) => parts.push(cell.th, cell.en));
      block.table.rows.forEach((row) =>
        row.forEach((cell) => parts.push(cell.th, cell.en))
      );
    }
    return parts.join(" ");
  }

  function guideSectionMatches(section, query) {
    if (!query) return true;
    const haystack = [
      section.title.th,
      section.title.en,
      ...section.blocks.map(guideBlockText),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  }

  function renderGuideBlock(block) {
    if (block.p) return `<p>${esc(guideText(block.p))}</p>`;
    if (block.list) {
      return `<ul>${block.list
        .map((item) => `<li>${esc(guideText(item))}</li>`)
        .join("")}</ul>`;
    }
    if (block.steps) {
      return `<ol>${block.steps
        .map((item) => `<li>${esc(guideText(item))}</li>`)
        .join("")}</ol>`;
    }
    if (block.note) {
      const tone = ["info", "warn", "ok"].includes(block.note.tone)
        ? block.note.tone
        : "info";
      return `<div class="guide-note ${tone}">${esc(
        guideText(block.note.text)
      )}</div>`;
    }
    if (block.code) {
      return `<pre class="guide-code">${esc(block.code)}</pre>`;
    }
    if (block.table) {
      const head = block.table.head
        .map((cell) => `<th>${esc(guideText(cell))}</th>`)
        .join("");
      const rows = block.table.rows
        .map(
          (row) =>
            `<tr>${row
              .map((cell) => `<td>${esc(guideText(cell))}</td>`)
              .join("")}</tr>`
        )
        .join("");
      return `<div class="guide-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    return "";
  }

  function renderGuide(options = {}) {
    if (!GUIDE || !Array.isArray(GUIDE.sections)) return;
    const modal = $("#guideModal");
    if (!modal) return;
    const ui = guideUi();
    const lang = guideLang();
    const query = guideState.search.trim().toLowerCase();

    const title = $("#guideTitle");
    if (title) title.textContent = ui.title || "คู่มือการเทรน LoRA";
    const sub = $("#guideSub");
    if (sub) sub.textContent = ui.subtitle || "";
    const foot = $("#guideFoot");
    if (foot) foot.textContent = ui.footer || "";
    const search = $("#guideSearch");
    if (search) {
      search.placeholder = ui.search || "";
      if (document.activeElement !== search) search.value = guideState.search;
    }
    const toc = $("#guideToc");
    if (toc) toc.setAttribute("aria-label", ui.contents || "สารบัญ");
    const close = $("#guideClose");
    if (close) close.setAttribute("aria-label", ui.close || "ปิด");

    const visible = GUIDE.sections.filter((section) =>
      guideSectionMatches(section, query)
    );
    if (
      guideState.section &&
      !visible.some((section) => section.id === guideState.section) &&
      visible.length
    ) {
      guideState.section = visible[0].id;
    }

    if (toc) {
      toc.innerHTML = visible.length
        ? visible
            .map(
              (section) => `
                <button type="button" class="guide-toc-item${
                  section.id === guideState.section ? " active" : ""
                }" data-guide-target="${esc(section.id)}">
                  <span aria-hidden="true">${esc(section.icon || "•")}</span>
                  <span>${esc(guideText(section.title))}</span>
                </button>`
            )
            .join("")
        : "";
    }

    const content = $("#guideContent");
    if (content) {
      content.innerHTML = visible.length
        ? visible
            .map(
              (section) => `
                <section class="guide-section" id="guide-sec-${esc(section.id)}">
                  <h4><span aria-hidden="true">${esc(section.icon || "")}</span> ${esc(
                guideText(section.title)
              )}</h4>
                  ${section.blocks.map(renderGuideBlock).join("")}
                </section>`
            )
            .join("")
        : `<p class="guide-empty">${esc(ui.noResults || "")}</p>`;
      if (options.scrollTo) {
        const target = content.querySelector(
          `#guide-sec-${options.scrollTo}`
        );
        if (target) {
          content.scrollTop = Math.max(0, target.offsetTop - 8);
        }
      }
    }
    // Language tag in the footer so it is obvious both are available.
    if (foot && foot.textContent) {
      foot.textContent = `${foot.textContent} · ${lang === "en" ? "EN" : "TH"}`;
    }
  }

  function openGuide(sectionId) {
    if (!GUIDE) return;
    const modal = $("#guideModal");
    if (!modal) return;
    guideState.opener =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    if (sectionId) guideState.section = sectionId;
    guideState.search = "";
    modal.classList.remove("hidden");
    renderGuide({ scrollTo: guideState.section });
    const search = $("#guideSearch");
    setTimeout(() => search && search.focus(), 0);
  }

  function closeGuide() {
    const modal = $("#guideModal");
    if (!modal || modal.classList.contains("hidden")) return;
    modal.classList.add("hidden");
    const opener = guideState.opener;
    guideState.opener = null;
    if (opener && document.body.contains(opener) && typeof opener.focus === "function") {
      try {
        opener.focus({ preventScroll: true });
      } catch (e) {
        try {
          opener.focus();
        } catch (e2) {
          /* element gone */
        }
      }
    }
  }

  function bindGuide() {
    const modal = $("#guideModal");
    if (!modal || !GUIDE) return;
    $("#guideOpen")?.addEventListener("click", () => openGuide("overview"));
    $("#guideOpenTop")?.addEventListener("click", () => openGuide("overview"));
    $("#guideClose")?.addEventListener("click", closeGuide);
    modal.addEventListener("click", (event) => {
      if (event.target.closest("[data-guide-close]")) closeGuide();
    });
    $("#guideSearch")?.addEventListener("input", () => {
      guideState.search = $("#guideSearch").value || "";
      renderGuide();
    });
    $("#guideToc")?.addEventListener("click", (event) => {
      const button = event.target.closest("[data-guide-target]");
      if (!button) return;
      guideState.section = button.dataset.guideTarget;
      renderGuide({ scrollTo: guideState.section });
    });
    // Keep the manual's keys away from app-level shortcuts; Escape closes.
    modal.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Escape") {
        event.preventDefault();
        closeGuide();
        return;
      }
      if (event.key !== "Tab") return;
      const focusables = Array.from(
        modal.querySelectorAll('button, input, [tabindex="0"]')
      ).filter((el) => el.offsetParent !== null);
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
    // Re-render the open manual in the new language (content is data-driven).
    document.addEventListener("langchange", () => {
      if (!modal.classList.contains("hidden")) renderGuide();
    });
  }

  // ── blocking status for slow operations ─────────────────────────────────
  const BUSY_TEXT = "กำลังเปลี่ยนค่า กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น";
  let busyDepth = 0;

  // The studio usually runs in an iframe inside the host shell. The host's
  // overlay covers the whole window (including the host top bar), so ask it to
  // show that instead of drawing a local overlay that leaves the bar clickable.
  function hostShell() {
    try {
      if (
        window.parent &&
        window.parent !== window &&
        window.parent.document.getElementById("blocking")
      ) {
        return window.parent;
      }
    } catch (e) {
      /* cross-origin parent */
    }
    return null;
  }

  function showBlocking(message) {
    const overlay = $("#busyOverlay");
    busyDepth += 1;
    const text = tr(message || BUSY_TEXT);
    const label = $("#busyText");
    if (label) label.textContent = text;
    const host = hostShell();
    if (host) {
      try {
        host.postMessage(
          { pixelsmithBusy: true, on: true, text },
          window.location.origin
        );
      } catch (e) {
        /* fall back to the local overlay */
      }
    }
    if (!overlay) return;
    if (host) overlay.classList.add("hidden");
    else overlay.classList.remove("hidden");
  }

  function hideBlocking() {
    busyDepth = Math.max(0, busyDepth - 1);
    if (busyDepth > 0) return;
    const host = hostShell();
    if (host) {
      try {
        host.postMessage(
          { pixelsmithBusy: true, on: false },
          window.location.origin
        );
      } catch (e) {
        /* nothing to do */
      }
    }
    const overlay = $("#busyOverlay");
    if (overlay) overlay.classList.add("hidden");
  }

  // File.path was removed in Electron 32 — resolve native paths through the
  // preload bridge instead (falls back to File.path for browsers/old builds).
  function filePath(file) {
    if (!file) return "";
    // The studio normally runs in a same-origin iframe inside the host shell;
    // the preload bridge lives on the top window, so fall back to parent.
    let bridge = window.pixelApp || null;
    if (!bridge) {
      try {
        if (window.parent && window.parent !== window) bridge = window.parent.pixelApp || null;
      } catch (e) {
        /* cross-origin parent */
      }
    }
    try {
      if (bridge && typeof bridge.getPathForFile === "function") {
        return bridge.getPathForFile(file) || "";
      }
    } catch (e) {
      /* not a native file */
    }
    return typeof file.path === "string" ? file.path : "";
  }

  // ── settings persistence ────────────────────────────────────────────────
  function loadSettings() {
    try {
      const raw = JSON.parse(localStorage.getItem("imgs:settings:v1") || "{}");
      return { ...DEFAULTS, ...raw };
    } catch (e) {
      return { ...DEFAULTS };
    }
  }
  function saveSettings() {
    try {
      localStorage.setItem("imgs:settings:v1", JSON.stringify(state.settings));
    } catch (e) {
      /* storage blocked — app still works */
    }
  }
  let saveTimer = null;
  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      state.settings = readForm();
      saveSettings();
    }, 260);
  }

  function readLoras() {
    const list = $("#loraList");
    if (!list) return state.settings.loras || [];
    const seen = new Set();
    const out = $$("#loraList .lora-item")
      .map((item) => {
        seen.add(item.dataset.id);
        if (!item.querySelector("input[type=checkbox]").checked) return null;
        const raw = Number(item.querySelector(".lora-weight").value);
        return {
          id: item.dataset.id,
          weight: clamp(Number.isFinite(raw) ? raw : 1, 0, 2),
        };
      })
      .filter(Boolean);
    // Selected LoRAs hidden by the variant filter must not be dropped when the
    // form re-reads the list.  Adapters that the new family cannot load are
    // dropped: keeping them would make the next generation fail.
    const fam = state.currentModel && state.currentModel.family;
    for (const l of state.settings.loras || []) {
      if (!l || !l.id || seen.has(l.id)) continue;
      const known = (state.loras || []).find((x) => x.id === l.id);
      if (known && !loraLoadable(known, fam)) continue;
      out.push(l);
    }
    return out;
  }

  function regionCapable() {
    const model = state.currentModel;
    return !!(model && model.capabilities && model.capabilities.regional);
  }

  function inpaintCapable() {
    const model = state.currentModel;
    return !!(model && model.capabilities && model.capabilities.inpaint);
  }

  function hiresCapable() {
    const model = state.currentModel;
    return !!(model && model.capabilities && model.capabilities.hires);
  }

  function enhanceCapable() {
    const model = state.currentModel;
    return !!(model && model.capabilities && model.capabilities.enhance);
  }

  function readRegionRows() {
    const list = $("#regionList");
    if (!list) return [];
    const key = ($("#regionLayout") && $("#regionLayout").value) || "2";
    const cfg = REGION_LAYOUTS[key] || REGION_LAYOUTS["2"];
    return $$("#regionList .region-item").map((item, i) => {
      const box = cfg.boxes[i] || cfg.boxes[cfg.boxes.length - 1];
      const rawWeight = Number(item.querySelector(".region-weight").value);
      return {
        prompt: item.querySelector(".region-prompt").value.trim(),
        weight: clamp(Number.isFinite(rawWeight) ? rawWeight : 1, 0, 1.5),
        x: box[0],
        y: box[1],
        w: box[2],
        h: box[3],
        label: cfg.labels[i] || `โซน ${i + 1}`,
      };
    });
  }

  function setRegionDisabled(disabled) {
    const box = $("#regionBox");
    if (!box) return;
    $$("#regionList input").forEach((el) => {
      el.disabled = disabled;
    });
    $("#regionLayout").disabled = disabled;
    $("#regionEnabled").disabled = disabled && state.busy;
    box.classList.toggle("is-off", disabled);
  }

  function renderRegions(layout, saved, enabled) {
    const box = $("#regionBox");
    if (!box) return;
    const capable = regionCapable();
    box.classList.toggle("hidden", !capable);
    if (!capable) return;
    const key = REGION_LAYOUTS[String(layout)] ? String(layout) : "2";
    $("#regionLayout").value = key;
    const cfg = REGION_LAYOUTS[key];
    const prev = Array.isArray(saved) ? saved : [];
    $("#regionList").innerHTML = cfg.labels
      .map((label, i) => {
        const r = prev[i] || {};
        return `<div class="region-item" data-i="${i}">
          <div class="region-head">
            <span class="region-label">${esc(label)}</span>
            <input type="number" class="region-weight" min="0" max="1.5" step="0.05"
              value="${esc(r.weight == null ? 1 : r.weight)}" title="น้ำหนักโซน">
          </div>
          <input class="region-prompt" type="text" placeholder="ตัวละคร/วัตถุในโซนนี้…"
            value="${esc(r.prompt || "")}">
        </div>`;
      })
      .join("");
    $("#regionEnabled").checked = !!enabled;
    setRegionDisabled(!enabled || state.busy);
  }

  function readForm() {
    return {
      prompt: $("#prompt").value,
      negative_prompt: $("#negative").value,
      width: normalizeSize($("#width").value),
      height: normalizeSize($("#height").value),
      steps: clamp(Number($("#steps").value) || 20, 2, 50),
      guidance: clamp(Number($("#guidance").value) || 4.5, 1, 10),
      seed: Number.isFinite(Number($("#seed").value)) ? Math.trunc(Number($("#seed").value)) : -1,
      batch: clamp(Number($("#batch").value) || 1, 1, 4),
      enhance: $("#enhance").checked,
      sampler: $("#sampler").value || "dpm",
      model_id: ($("#modelSelect") && $("#modelSelect").value) || state.settings.model_id || "",
      loras: readLoras(),
      regions_enabled: !!($("#regionEnabled") && $("#regionEnabled").checked),
      region_layout: ($("#regionLayout") && $("#regionLayout").value) || "2",
      regions: readRegionRows(),
      inpaint: inpaintCapable()
        && !!filePath($("#initImage")?.files?.[0])
        && !!filePath($("#maskImage")?.files?.[0]),
      init_image: filePath($("#initImage")?.files?.[0]),
      mask_image: filePath($("#maskImage")?.files?.[0]),
      strength: clamp(Number($("#inpaintStrength")?.value) || 0.75, 0, 1),
      hires_enabled: !!($("#hiresEnabled")?.checked),
      hires_scale: clamp(Number($("#hiresScale")?.value) || 1.5, 1, 2),
      hires_strength: clamp(Number($("#hiresStrength")?.value) || 0.35, 0.05, 0.75),
    };
  }

  function applyForm(settings) {
    const s = { ...DEFAULTS, ...(settings || {}) };
    $("#prompt").value = s.prompt || "";
    $("#negative").value = s.negative_prompt || "";
    $("#width").value = normalizeSize(s.width);
    $("#height").value = normalizeSize(s.height);
    $("#steps").value = clamp(Number(s.steps) || 20, 2, 50);
    $("#guidance").value = clamp(Number(s.guidance) || 4.5, 1, 10);
    $("#seed").value = Number.isFinite(Number(s.seed)) ? Number(s.seed) : -1;
    $("#batch").value = String(clamp(Number(s.batch) || 1, 1, 4));
    $("#sampler").value = s.sampler === "euler" ? "euler" : "dpm";
    $("#enhance").checked = !!s.enhance;
    $("#stepsVal").textContent = $("#steps").value;
    $("#cfgVal").textContent = Number($("#guidance").value).toFixed(1);
    const layout = REGION_LAYOUTS[String(s.region_layout)] ? String(s.region_layout) : "2";
    renderRegions(layout, Array.isArray(s.regions) ? s.regions : [], s.regions_enabled);
    if (s.model_id && state.registry.some((m) => m.id === s.model_id)) {
      $("#modelSelect").value = s.model_id;
      state.currentModel = state.registry.find((m) => m.id === s.model_id) || state.currentModel;
      updateModelMeta(state.currentModel);
    }
    updatePromptCount();
    updateSizeHint();
    markRatio();
    markChips();
    state.settings = readForm();
    if (s.model_id) state.settings.model_id = s.model_id;
    if (Array.isArray(s.loras)) state.settings.loras = s.loras;
    state.settings.regions_enabled = !!s.regions_enabled;
    state.settings.region_layout = layout;
    state.settings.regions = Array.isArray(s.regions) ? s.regions : [];
    state.settings.inpaint = !!s.inpaint;
    state.settings.init_image = s.init_image || "";
    state.settings.mask_image = s.mask_image || "";
    state.settings.strength = clamp(Number(s.strength) || 0.75, 0, 1);
    state.settings.hires_enabled = !!s.hires_enabled;
    state.settings.hires_scale = clamp(Number(s.hires_scale) || 1.5, 1, 2);
    state.settings.hires_strength = clamp(Number(s.hires_strength) || 0.35, 0.05, 0.75);
    if ($("#hiresEnabled")) {
      $("#hiresEnabled").checked = !!s.hires_enabled;
      $("#hiresScale").value = state.settings.hires_scale;
      $("#hiresStrength").value = state.settings.hires_strength;
    }
  }

  // ── UI builders ─────────────────────────────────────────────────────────
  function buildRatios() {
    $("#ratios").innerHTML = RATIOS.map(
      (r, i) => `<button type="button" class="ratio" data-i="${i}" title="${r.w}×${r.h}">${r.label}</button>`
    ).join("");
    $("#ratios").addEventListener("click", (e) => {
      const btn = e.target.closest(".ratio");
      if (!btn) return;
      const r = RATIOS[Number(btn.dataset.i)];
      $("#width").value = r.w;
      $("#height").value = r.h;
      updateSizeHint();
      markRatio();
      scheduleSave();
    });
  }

  function markRatio() {
    const w = normalizeSize($("#width").value);
    const h = normalizeSize($("#height").value);
    $$(".ratio").forEach((btn, i) => {
      btn.classList.toggle("active", RATIOS[i].w === w && RATIOS[i].h === h);
    });
  }

  function buildChips() {
    $("#styleChips").innerHTML = STYLES.map(
      (s, i) => `<button type="button" class="chip" data-i="${i}">${esc(s.label)}</button>`
    ).join("");
    $("#styleChips").addEventListener("click", (e) => {
      const btn = e.target.closest(".chip");
      if (!btn) return;
      toggleStyle(STYLES[Number(btn.dataset.i)]);
    });
  }

  function toggleStyle(style) {
    const add = style.add;
    const parts = $("#prompt")
      .value.split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const idx = parts.indexOf(add);
    if (idx >= 0) parts.splice(idx, 1);
    else parts.push(add);
    $("#prompt").value = parts.join(", ");
    updatePromptCount();
    markChips();
    scheduleSave();
  }

  function markChips() {
    const text = $("#prompt").value;
    $$("#styleChips .chip").forEach((btn, i) => {
      btn.classList.toggle("active", text.includes(STYLES[i].add));
    });
  }

  function buildExamples() {
    $("#examples").innerHTML = IDEAS.slice(0, 4)
      .map(
        (p) =>
          `<button type="button" class="chip" data-prompt="${esc(p)}">${esc(
            p.length > 62 ? p.slice(0, 60) + "…" : p
          )}</button>`
      )
      .join("");
    $("#examples").addEventListener("click", (e) => {
      const btn = e.target.closest(".chip");
      if (!btn) return;
      $("#prompt").value = btn.dataset.prompt;
      updatePromptCount();
      markChips();
      scheduleSave();
      $("#prompt").focus();
      $("#panel .panel-inner").scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  function updatePromptCount() {
    $("#promptCount").textContent = `${$("#prompt").value.length} ตัวอักษร`;
  }

  function updateSizeHint() {
    const w = normalizeSize($("#width").value);
    const h = normalizeSize($("#height").value);
    $("#sizeHint").textContent = `${((w * h) / 1e6).toFixed(2)} MP · ${w}×${h}`;
    updateAdvancedSummary();
  }

  function updateAdvancedSummary() {
    const el = $("#advSummary");
    if (!el) return;
    const w = normalizeSize($("#width").value);
    const h = normalizeSize($("#height").value);
    const steps = $("#steps").value;
    const cfg = Number($("#guidance").value).toFixed(1);
    const seed = Number($("#seed").value);
    el.textContent = `${w}×${h} · ${steps} steps · CFG ${cfg} · seed ${seed < 0 ? "สุ่ม" : seed}`;
  }

  // ── models & LoRA ───────────────────────────────────────────────────────
  function updateModelMeta(entry) {
    if (!entry) return;
    $("#modelSub").textContent = entry.name || "Image Studio";
    const fam = entry.family_label || (entry.family || "").toUpperCase();
    const kind = entry.kind === "single" ? "checkpoint" : "diffusers";
    const warn =
      entry.supported === false
        ? ` · ⚠ ${tr(entry.unsupported_reason || "ยังไม่รองรับ")}`
        : "";
    $("#modelFamily").textContent = fam ? `${fam} · ${kind}${warn}` : "—";
    $("#modelFamily").title = warn ? tr(entry.unsupported_reason || "") : "";
    const canEnhance = !!(entry.capabilities && entry.capabilities.enhance);
    $("#enhance").disabled = !canEnhance;
    if (!canEnhance) $("#enhance").checked = false;
    const sw = $("#enhance").closest(".switch");
    if (sw) sw.title = canEnhance ? "" : "ตัวเลือกนี้ใช้กับโมเดลที่รองรับเท่านั้น";
    if ($("#regionBox")) {
      const rows = readRegionRows();
      const saved = rows.length ? rows : state.settings.regions || [];
      const enabled = rows.length
        ? $("#regionEnabled").checked
        : !!state.settings.regions_enabled;
      renderRegions(
        $("#regionLayout").value || state.settings.region_layout || "2",
        saved,
        enabled
      );
    }
    if ($("#inpaintBox")) {
      const box = $("#inpaintBox");
      box.classList.toggle("hidden", !inpaintCapable());
      box.style.display = !inpaintCapable() ? "none" : "";
    }
    if ($("#hiresBox")) {
      const box = $("#hiresBox");
      const capable = hiresCapable();
      box.classList.toggle("hidden", !capable);
      box.style.display = !capable ? "none" : "";
      if (capable && $("#hiresEnabled") && $("#hiresEnabled").checked && !state.busy) {
        box.open = true;
      }
    }
  }

  function applyModelDefaults(entry) {
    const d = (entry && entry.defaults) || {};
    $("#width").value = normalizeSize(d.width || 1024);
    $("#height").value = normalizeSize(d.height || 1024);
    $("#steps").value = clamp(Number(d.steps) || 20, 2, 50);
    $("#guidance").value = clamp(Number(d.guidance) || 4.5, 1, 10);
    $("#sampler").value = d.sampler === "euler" ? "euler" : "dpm";
    $("#enhance").checked =
      entry && entry.capabilities && entry.capabilities.enhance ? !!d.enhance : false;
    $("#stepsVal").textContent = $("#steps").value;
    $("#cfgVal").textContent = Number($("#guidance").value).toFixed(1);
    updateSizeHint();
    markRatio();
    scheduleSave();
  }

  async function selectModel(id, options = {}) {
    const silent = !!options.silent;
    const applyDefaults = options.applyDefaults !== false;
    const entry = (state.registry || []).find((m) => m.id === id);
    if (entry && entry.supported === false) {
      toast(
        tr(`โมเดลนี้ใช้ไม่ได้: ${entry.unsupported_reason || "ยังไม่รองรับ"}`),
        "warn",
        7000
      );
      if ($("#modelSelect")) $("#modelSelect").value = state.currentId || "";
      return false;
    }
    try {
      const res = await api("/api/models/select", {
        method: "POST",
        body: JSON.stringify({ id }),
      });
      state.currentId = id;
      state.currentModel =
        state.registry.find((m) => m.id === id) || res.entry || state.currentModel;
      updateModelMeta(state.currentModel);
      if (applyDefaults) applyModelDefaults(state.currentModel);
      state.settings.model_id = id;
      // keep the dropdown in sync with the state — programmatic switches
      // (assistant, presets) otherwise leave a stale option selected
      if ($("#modelSelect")) $("#modelSelect").value = id;
      saveSettings();
      if (!silent) toast(`เลือกโมเดล ${state.currentModel.name} แล้ว`, "ok", 2600);
      return true;
    } catch (err) {
      toast(err.message, "err", 7000);
      if ($("#modelSelect")) $("#modelSelect").value = state.currentId || "";
      return false;
    }
  }

  async function refreshModels(preferredId) {
    try {
      const data = await api("/api/models");
      state.registry = data.models || [];
      const sel = $("#modelSelect");
      if (!state.registry.length) {
        sel.innerHTML = "";
        return;
      }
      sel.innerHTML = state.registry
        .map((m) => {
          const suffix =
            m.supported === false
              ? ` (${tr("ไม่รองรับ")})`
              : m.available === false
                ? ` (${tr("ยังไม่พร้อม")})`
                : "";
          return `<option value="${esc(m.id)}"${m.supported === false ? " disabled" : ""}>${esc(
            m.name
          )} — ${esc(m.family_label || m.family || "")}${suffix}</option>`;
        })
        .join("");
      const wanted = preferredId || state.settings.model_id;
      const usable = state.registry.filter((m) => m.supported !== false);
      if (!usable.length) {
        sel.insertAdjacentHTML(
          "afterbegin",
          `<option value="" selected disabled>${esc(tr("ไม่มีโมเดลที่ใช้งานได้"))}</option>`
        );
        state.currentModel = null;
        state.currentId = "";
        updateModelMeta({
          name: "Image Studio",
          family: "",
          supported: false,
          unsupported_reason: tr("ไม่พบโมเดลที่ใช้งานได้ — เพิ่มโมเดลหรือกดสแกนใหม่"),
        });
        toast(tr("ไม่พบโมเดลที่ใช้งานได้ — เพิ่มโมเดลหรือกดสแกนใหม่"), "warn", 8000);
        renderModelVariants();
        return;
      }
      const target =
        usable.find((m) => m.id === wanted) ||
        usable.find((m) => m.id === data.current) ||
        usable[0];
      sel.value = target.id;
      state.currentModel = target;
      state.currentId = data.current || target.id;
      updateModelMeta(target);
      if (target.id !== state.currentId) {
        await selectModel(target.id, { silent: true, applyDefaults: false });
      } else if (!wanted) {
        applyModelDefaults(target);
      }
      state.settings.model_id = target.id;
      saveSettings();
      renderModelVariants();
    } catch (err) {
      toast(`โหลดรายการโมเดลไม่ได้: ${err.message}`, "err", 6000);
    }
  }

  // ── base-model variants (Pony / Illustrious / …) ─────────────────────────
  const VARIANT_LABELS = {
    any: "ใช้ได้ทุกโมเดล",
    sana: "SANA",
    sdxl: "SDXL",
    pony: "Pony",
    illustrious: "Illustrious",
    noobai: "NoobAI",
    sd15: "SD 1.5",
    sd2: "SD 2.x",
    sd3: "SD 3 / 3.5",
    pixart: "PixArt",
    lumina: "Lumina",
    lumina2: "Lumina 2",
    hunyuan: "HunyuanDiT",
    kolors: "Kolors",
    auraflow: "AuraFlow",
    amused: "Amused",
    cogview3: "CogView3",
    cogview4: "CogView4",
    wuerstchen: "Würstchen",
    generic: "Diffusers",
    flux: "Flux",
    chroma: "Chroma",
  };
  const VARIANT_CHOICES = Object.keys(VARIANT_LABELS);

  function variantLabel(value, fallback) {
    const raw = VARIANT_LABELS[value] || value || fallback || "";
    return raw ? tr(raw) : "";
  }

  function modelVariant() {
    const m = state.currentModel || {};
    return m.variant || m.family || "any";
  }

  // Adapters that the current engine family can actually load.
  function loraLoadable(l, fam) {
    if (!l) return true;
    if (l.variant === "flux" || l.variant === "chroma") return false;
    return !fam || !l.family || l.family === "any" || l.family === fam;
  }

  // A LoRA tagged "any" works everywhere.  A LoRA whose variant was only
  // guessed from the engine family (plain "SDXL", source=family) is unverified
  // but usually fine — show it.  Only a positively detected different lineage
  // (name/metadata/override) is a real mismatch.
  function loraFitsModel(l, fam, variant) {
    const lv = l.variant || "any";
    // Flux/Chroma adapters can never load into SANA/SDXL/SD1.5 pipelines.
    if (lv === "flux" || lv === "chroma") return false;
    if (!fam || !l.family || l.family === "any") return true;
    if (l.family !== fam) return false;
    if (lv === "any") return true;
    if (l.variant_source === "family") return true;
    return lv === variant;
  }

  function renderLoras(selected) {
    const box = $("#loraBox");
    const list = $("#loraList");
    if (!box || !list) return;
    const fam = state.currentModel && state.currentModel.family;
    const variant = modelVariant();
    const showAll = !!($("#loraShowAll") && $("#loraShowAll").checked);
    const all = (state.loras || []).filter(
      (l) => !fam || !l.family || l.family === "any" || l.family === fam
    );
    if (!all.length) {
      box.classList.add("hidden");
      list.innerHTML = "";
      return;
    }
    const fitting = all.filter((l) => loraFitsModel(l, fam, variant));
    const mismatched = all.filter((l) => !loraFitsModel(l, fam, variant));
    const items = showAll ? fitting.concat(mismatched) : fitting;
    const source = Array.isArray(selected) ? selected : state.settings.loras || [];
    const chosen = new Map(
      source.map((x) => {
        const w = Number(x.weight);
        return [x.id, Number.isFinite(w) ? w : 1];
      })
    );
    box.classList.remove("hidden");
    if (!items.length) {
      list.innerHTML = `<p class="micro">${esc(
        tr('ไม่มี LoRA ที่ตรงกับ variant นี้ — เปิด "แสดง LoRA ที่ไม่ตรง variant" เพื่อดูรายการ')
      )}</p>`;
      return;
    }
    list.innerHTML = items
      .map((l) => {
        const on = chosen.has(l.id);
        const weight = on ? chosen.get(l.id) : 1;
        const mismatch = !loraFitsModel(l, fam, variant);
        const tag = variantLabel(l.variant, l.family_label || l.family);
        return `<div class="lora-item${mismatch ? " lora-mismatch" : ""}" data-id="${esc(l.id)}">
          <label class="lora-check">
            <input type="checkbox" value="${esc(l.id)}" ${on ? "checked" : ""}>
            <span>${esc(l.name)}</span>
          </label>
          <span class="lora-tag">${mismatch ? "⚠ " : ""}${esc(tag)}</span>
          <input type="number" class="lora-weight" min="0" max="2" step="0.05"
            value="${esc(weight)}" title="น้ำหนัก LoRA" ${on ? "" : "disabled"}>
        </div>`;
      })
      .join("");
  }

  async function refreshLoras(saved) {
    try {
      const data = await api("/api/loras");
      state.loras = data.loras || [];
      renderLoras(Array.isArray(saved) ? saved : state.settings.loras || []);
      renderLoraManager();
      renderInitLoras();
      renderSampleLoras();
    } catch (err) {
      console.warn("loras:", err);
    }
  }

  // ── prompt presets (บันทึก/ใช้/นำเข้า prompt + settings) ──────────────────
  const presetState = { items: [] };

  async function refreshPresets() {
    try {
      const data = await api("/api/presets");
      presetState.items = data.presets || [];
      renderPresets();
    } catch (err) {
      console.warn("presets:", err);
    }
  }

  function renderPresets() {
    const list = $("#presetList");
    if (!list) return;
    const items = presetState.items || [];
    if (!items.length) {
      list.innerHTML =
        '<p class="micro">ยังไม่มีพรีเซ็ต — บันทึกค่าปัจจุบันหรือนำเข้าจากไฟล์ด้านล่าง</p>';
      return;
    }
    list.innerHTML = items
      .map((p) => {
        const s = p.settings || {};
        const bits = [];
        if (s.width && s.height) bits.push(`${s.width}×${s.height}`);
        if (Number.isFinite(Number(s.steps))) bits.push(`${s.steps} steps`);
        const loraCount = (s.loras || []).length;
        if (loraCount) bits.push(`${loraCount} LoRA`);
        const issues = [...(s.ignored || []), ...(s.notes || [])];
        return `<div class="preset-item" data-id="${esc(p.id)}">
          <button type="button" class="preset-name" title="คลิกเพื่อใช้พรีเซ็ต">${esc(p.name)}</button>
          <span class="preset-meta">${esc(bits.join(" · "))}</span>
          ${issues.length ? `<span class="lora-tag" title="${esc(issues.join(" · "))}">⚠ ${issues.length}</span>` : ""}
          <button type="button" class="ghost-icon preset-del" data-id="${esc(p.id)}" title="ลบพรีเซ็ต">✕</button>
        </div>`;
      })
      .join("");
    $$("#presetList .preset-name").forEach((btn) => {
      btn.addEventListener("click", () => {
        const item = btn.closest(".preset-item");
        const p = presetState.items.find((x) => x.id === item.dataset.id);
        if (p) applyPreset(p);
      });
    });
    $$("#presetList .preset-del").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const p = presetState.items.find((x) => x.id === btn.dataset.id);
        if (!p) return;
        if (!(await uiConfirm(tr(`ลบพรีเซ็ต "${p.name}"?`)))) return;
        try {
          await api(`/api/presets/${p.id}`, { method: "DELETE" });
          await refreshPresets();
          toast("ลบพรีเซ็ตแล้ว", "ok", 1800);
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
    });
  }

  function normLoraName(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/\.safetensors$/, "")
      .replace(/\\/g, "/")
      .split("/")
      .pop()
      .replace(/[\s_\-]+/g, " ")
      .trim();
  }

  // A1111/Forge tags often keep sub-folders (`<lora:styles/foo:0.8>`) or lose
  // the exact spacing of the local file name, so match a bit more loosely.
  function resolveLora(name) {
    const key = normLoraName(name);
    if (!key) return null;
    const list = state.loras || [];
    const exact = list.find((x) => normLoraName(x.name) === key);
    if (exact) return exact;
    const tail = String(name || "")
      .replace(/\.safetensors$/i, "")
      .replace(/\\/g, "/")
      .toLowerCase();
    const byPath = list.find((x) =>
      String(x.path || "")
        .toLowerCase()
        .replace(/\.safetensors$/, "")
        .endsWith("/" + tail)
    );
    if (byPath) return byPath;
    const cands = list.filter((x) => {
      const n = normLoraName(x.name);
      return n && (n.includes(key) || key.includes(n));
    });
    return cands.length === 1 ? cands[0] : null;
  }

  async function applyPreset(p) {
    const s = p.settings || {};
    const missing = [];
    const warnings = [];
    const overrides = { prompt: p.prompt || "", negative_prompt: p.negative || "" };
    const sel = (s.loras || [])
      .map((l) => {
        const found = resolveLora(l.name);
        if (!found) {
          missing.push(l.name);
          return null;
        }
        return { id: found.id, weight: Number(l.weight) || 1 };
      })
      .filter(Boolean);
    // switch model first (async) so the LoRA list re-renders for the right
    // family before we tick the preset's adapters
    if (s.model) {
      const found = (state.registry || []).find(
        (x) => String(x.name).toLowerCase() === String(s.model).toLowerCase()
      );
      if (found && found.id !== state.currentId) {
        await selectModel(found.id, { silent: true, applyDefaults: false });
      }
      if (found) {
        // pass the id through so applyForm can't revert it from a stale select
        overrides.model_id = found.id;
      } else {
        warnings.push(tr(`โมเดล "${s.model}" ไม่มีในระบบ`));
      }
    }
    if (Number.isFinite(Number(s.width))) overrides.width = s.width;
    if (Number.isFinite(Number(s.height))) overrides.height = s.height;
    if (Number.isFinite(Number(s.steps))) overrides.steps = s.steps;
    if (Number.isFinite(Number(s.guidance))) overrides.guidance = s.guidance;
    if (s.sampler) overrides.sampler = s.sampler;
    if (Number(s.seed) >= 0) overrides.seed = Number(s.seed);
    const fam = state.currentModel && state.currentModel.family;
    const variant = modelVariant();
    // Adapters from another family cannot load at all; keep them out of the
    // form (they are still reported below).  Same-family variant mismatches
    // stay selected so the user can decide.
    const usable = sel.filter((x) => {
      const l = (state.loras || []).find((y) => y.id === x.id);
      return loraLoadable(l, fam);
    });
    applyForm({ ...readForm(), ...overrides });
    // A preset without LoRAs must clear the current ones instead of keeping
    // whatever happened to be ticked before.
    state.settings.loras = usable;
    renderLoras(usable);
    toast(tr(`ใช้พรีเซ็ตแล้ว: ${p.name}`), "ok", 2600);
    if (missing.length) warnings.push(tr(`LoRA ไม่มีในระบบ: ${missing.join(", ")}`));
    const offVariant = sel
      .map((x) => (state.loras || []).find((l) => l.id === x.id))
      .filter((l) => l && !loraFitsModel(l, fam, variant))
      .map((l) => l.name);
    if (offVariant.length)
      warnings.push(tr(`LoRA ไม่ตรง variant: ${offVariant.join(", ")}`));
    if (s.ignored && s.ignored.length)
      warnings.push(tr(`นำเข้าไม่ได้: ${s.ignored.join(", ")}`));
    if (s.notes && s.notes.length)
      warnings.push(tr(`ปรับค่า: ${s.notes.join(" · ")}`));
    if (warnings.length) toast(warnings.join(" · "), "warn", 9000);
  }

  async function saveCurrentPreset() {
    const name = ($("#presetName")?.value || "").trim();
    if (!name) {
      toast("ตั้งชื่อพรีเซ็ตก่อน", "warn");
      $("#presetName")?.focus();
      return;
    }
    const f = readForm();
    const loras = (f.loras || []).map((l) => {
      const found = (state.loras || []).find((x) => x.id === l.id);
      return { name: found ? found.name : String(l.id), weight: Number(l.weight) || 1 };
    });
    try {
      await api("/api/presets", {
        method: "POST",
        body: JSON.stringify({
          name,
          prompt: f.prompt,
          negative: f.negative_prompt,
          settings: {
            width: f.width,
            height: f.height,
            steps: f.steps,
            guidance: f.guidance,
            sampler: f.sampler,
            seed: f.seed,
            loras,
            model: state.currentModel ? state.currentModel.name : "",
          },
        }),
      });
      $("#presetName").value = "";
      await refreshPresets();
      toast(tr(`บันทึกพรีเซ็ตแล้ว: ${name}`), "ok", 2600);
    } catch (err) {
      toast(err.message, "err", 6000);
    }
  }

  function importToast(res) {
    toast(
      tr(`นำเข้าพรีเซ็ตแล้ว: เพิ่ม ${res.added} · อัปเดต ${res.updated} · ข้าม ${res.skipped}`),
      "ok",
      4000
    );
  }

  async function importPresetsFromCollection() {
    try {
      const res = await api("/api/presets/import", {
        method: "POST",
        body: JSON.stringify({ from_file: "prompt-collection.txt" }),
      });
      await refreshPresets();
      importToast(res);
    } catch (err) {
      toast(err.message, "err", 6000);
    }
  }

  // Picker for "continue training from an existing adapter".
  function renderInitLoras() {
    const sel = $("#trainInitLora");
    if (!sel) return;
    const prev = sel.value;
    const fam = (currentTrainerModel() || {}).family;
    const loras = (state.loras || [])
      .filter((l) => !fam || !l.family || l.family === "any" || l.family === fam)
      .slice()
      .sort((a, b) => String(a.name).localeCompare(String(b.name)));
    sel.innerHTML =
      '<option value="">— เริ่มจากศูนย์ —</option>' +
      loras
        .map(
          (l) =>
            `<option value="${esc(l.id)}">${esc(l.name)}${
              l.rank ? ` · r${l.rank}` : ""
            }</option>`
        )
        .join("");
    if (prev && loras.some((l) => l.id === prev)) sel.value = prev;
  }

  // Picker for the post-training sample button.
  function renderSampleLoras() {
    const sel = $("#sampleLora");
    if (!sel) return;
    const prev = sel.value;
    const loras = (state.loras || [])
      .slice()
      .sort((a, b) => String(a.name).localeCompare(String(b.name)));
    sel.innerHTML =
      '<option value="">— เลือก LoRA —</option>' +
      loras
        .map(
          (l) =>
            `<option value="${esc(l.id)}">${esc(l.name)}${
              l.rank ? ` · r${l.rank}` : ""
            }</option>`
        )
        .join("");
    if (prev && loras.some((l) => l.id === prev)) sel.value = prev;
  }

  function renderDatasetQc(res) {
    const box = $("#dsQcResult");
    if (!box) return;
    const ds = res.dataset || {};
    const recs = (ds.recommendations || [])
      .map((text) => `<li>${esc(text)}</li>`)
      .join("");
    const buckets = Object.entries(ds.buckets || {})
      .map(([size, count]) => `<span class="qc-chip">${esc(size)} × ${count}</span>`)
      .join(" ");
    const rows = (res.images || [])
      .filter((img) => (img.warnings || []).length)
      .map(
        (img) => `
          <li><b>${esc(img.name)}</b> · ${img.width}×${img.height}
            <span class="micro">(${esc(img.bucket)})</span>
            <ul>${img.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>
          </li>`
      )
      .join("");
    box.innerHTML = `
      <div class="qc-head">
        <b>${esc(ds.name || "dataset")}</b>
        <span class="micro">${ds.count} รูป · เดี่ยว ${ds.solo} · หลายตัว ${ds.multi}
          · ไม่มี caption ${ds.missing_caption} · ไฟล์ซ้ำ ${(ds.duplicates || []).length} กลุ่ม</span>
      </div>
      <div class="qc-buckets">${buckets || '<span class="micro">ไม่มีข้อมูล bucket</span>'}</div>
      <ul class="qc-recs">${recs}</ul>
      ${rows ? `<details><summary>รูปที่มีคำเตือน</summary><ul class="qc-images">${rows}</ul></details>` : ""}
    `;
  }

  // ── LoRA trainer ────────────────────────────────────────────────────────
  function trainerModels() {
    return (state.registry || []).filter((m) => m.trainable && m.supported !== false);
  }

  function currentTrainerModel() {
    const sel = $("#trainerModel");
    if (sel && sel.value) {
      return (state.registry || []).find((m) => m.id === sel.value) || null;
    }
    const cur = state.currentModel;
    if (cur && cur.trainable && cur.supported !== false) return cur;
    return trainerModels()[0] || null;
  }

  function updateTrainerModelNote() {
    const note = $("#trainerModelNote");
    if (!note) return;
    const m = currentTrainerModel();
    if (!m) {
      note.textContent = "ยังไม่มีโมเดลที่เทรนได้ (SANA / SDXL)";
      return;
    }
    note.textContent =
      m.family === "sdxl"
        ? "SDXL ช้ากว่า SANA มากบนการ์ดนี้ — แนะนำ 768px · rank 16 · LR 1e-4 · 500–1000 steps"
        : "SANA เร็วที่สุด — แนะนำ 1024px · rank 32–64 · LR 2e-4 · 1500+ steps";
  }

  function renderTrainerModels() {
    const sel = $("#trainerModel");
    if (!sel) return;
    const prev = sel.value;
    const models = trainerModels();
    sel.innerHTML =
      '<option value="">— ใช้โมเดลที่เลือกอยู่ —</option>' +
      models
        .map(
          (m) =>
            `<option value="${esc(m.id)}">${esc(m.name)} — ${esc(
              m.family_label || m.family || ""
            )}${m.available === false ? " (ยังไม่พร้อม)" : ""}</option>`
        )
        .join("");
    if (prev && models.some((m) => m.id === prev)) sel.value = prev;
    const cur = state.currentModel;
    if (!sel.value && (!cur || !cur.trainable)) {
      const first = models[0];
      if (first) sel.value = first.id;
    }
    updateTrainerModelNote();
  }

  async function refreshTrainer() {
    try {
      const data = await api("/api/train/datasets");
      state.datasets = data.datasets || [];
      renderTrainer();
    } catch (err) {
      console.warn("trainer:", err);
    }
  }

  function activeDataset() {
    return (state.datasets || []).find((d) => d.id === state.activeDs) || null;
  }

  const MULTI_HINTS =
    /\b(man|men|male|individuals|two people|two girls|two boys|duo|group|another character|partner|couple)\b/i;

  // tag มุมมาตรฐาน — เรียงตามลำดับที่ใช้วางหน้าคำอธิบาย
  const ANGLE_TAGS = [
    "front view",
    "side view",
    "three-quarter view",
    "back view",
    "close-up",
    "high angle",
    "low angle",
  ];
  // วลีในประโยคอิสระ → tag มาตรฐาน
  const ANGLE_PATTERNS = [
    ["front view", /\b(from the front|front view|facing (the )?(viewer|camera|forward))\b/i],
    ["side view", /\b(from the side|side view|side profile|in profile|profile view)\b/i],
    ["three-quarter view", /\b(three[- ]quarter|3\/4|three quarter view)\b/i],
    ["back view", /\b(from behind|from the back|back view|rear view|seen from behind)\b/i],
    ["close-up", /\b(close[- ]?up|headshot|portrait shot)\b/i],
    ["high angle", /\b(high[- ]angle|looking down (at|from)|from above)\b/i],
    ["low angle", /\b(low[- ]angle|looking up (at|from)|from below)\b/i],
  ];

  const KIND_LABELS = {
    character: "ตัวละคร",
    style: "สไตล์ภาพ",
    outfit: "ชุด/ของตกแต่ง",
    pose: "ท่า/คอนเซปต์",
  };

  const KIND_HINTS = {
    character:
      "ตัวละคร: ควรมีรูปเดี่ยว ≥ 16 รูป หลายมุม และ caption บรรยายฉาก/ท่า/ชุด ไม่บรรยายลักษณะถาวร",
    style:
      "สไตล์ภาพ: ควรมี 20–30 รูปขึ้นไป หลากหลายเนื้อหา และ caption บรรยายเนื้อหา ไม่บรรยายสไตล์",
    outfit:
      "ชุด/ของตกแต่ง: 10–20 รูป เห็นชิ้นงานชัดทั้งระยะใกล้/ไกล และ caption บรรยายรายละเอียดที่คงที่",
    pose:
      "ท่า/คอนเซปต์: 20–40 รูป มุมกล้องหลากหลาย และ caption ระบุชื่อท่า/คอนเซปต์",
  };

  const KIND_TARGETS = { character: [16, 30], style: [20, 30], outfit: [10, 20], pose: [20, 40] };

  const TRAIN_HINTS = {
    character:
      "💡 character: rank 32 · LR 1e-4 · 1024px · cosine · caption บรรยายฉาก/ท่า/ชุด ไม่บรรยายลักษณะถาวร",
    style:
      "💡 style: rank 32 · LR 1.5e-4 · 1024px · caption บรรยายเนื้อหาให้ละเอียด แต่ไม่บรรยายสไตล์",
    outfit:
      "💡 outfit: rank 16 · LR 1e-4 · 1024px · caption เน้นรายละเอียดชิ้นงานที่คงที่",
    pose:
      "💡 pose: rank 16–32 · LR 1e-4 · caption ระบุชื่อท่า/คอนเซปต์",
  };

  const TRAIN_PRESETS = {
    character: {
      rank: 32, steps: 1200, lr: 0.0001, resolution: 1024,
      scheduler: "cosine", sampling: "logit_normal", saveEvery: 200,
      multiMode: "exclude", ema: false, flip: false,
      bucketing: true, captionDropout: 0.05, captionShuffle: false,
      keepTokens: 1, noiseOffset: 0, minSnr: 0, rsLora: false, targetModules: "attn",
    },
    style: {
      rank: 32, steps: 2000, lr: 0.00015, resolution: 1024,
      scheduler: "cosine", sampling: "logit_normal", saveEvery: 250,
      multiMode: "include", ema: false, flip: true,
      bucketing: true, captionDropout: 0.05, captionShuffle: true,
      keepTokens: 1, noiseOffset: 0.02, minSnr: 5, rsLora: false, targetModules: "attn",
    },
    outfit: {
      rank: 16, steps: 900, lr: 0.0001, resolution: 1024,
      scheduler: "cosine", sampling: "logit_normal", saveEvery: 150,
      multiMode: "include", ema: false, flip: false,
      bucketing: true, captionDropout: 0.05, captionShuffle: false,
      keepTokens: 1, noiseOffset: 0, minSnr: 0, rsLora: false, targetModules: "attn",
    },
    fast: {
      rank: 16, steps: 300, lr: 0.00015, resolution: 768,
      scheduler: "constant", sampling: "logit_normal", saveEvery: 100,
      multiMode: "include", ema: false, flip: false,
      bucketing: true, captionDropout: 0.05, captionShuffle: false,
      keepTokens: 1, noiseOffset: 0, minSnr: 0, rsLora: false, targetModules: "attn",
    },
  };

  function trainerTriggers(ds) {
    const out = [];
    const push = (value) => {
      const trigger = String(value || "").trim().toLowerCase();
      if (trigger && !out.includes(trigger)) out.push(trigger);
    };
    push(ds.trigger);
    (ds.extra_triggers || []).forEach(push);
    (ds.images || []).forEach((img) => push(img.trigger));
    return out;
  }

  function looksMultiCaption(caption) {
    return MULTI_HINTS.test(String(caption || ""));
  }

  function detectAngleTags(caption) {
    const text = String(caption || "");
    return ANGLE_PATTERNS.filter(([, re]) => re.test(text)).map(([tag]) => tag);
  }

  // เข้มงวด: ต้องมี tag มาตรฐานจริง ๆ ไม่นับแค่คำว่า "view" ลอย ๆ
  function hasAngleTag(caption) {
    const text = String(caption || "").toLowerCase();
    return ANGLE_TAGS.some((tag) => text.includes(tag));
  }

  function missingAngleTags(caption) {
    const text = String(caption || "").toLowerCase();
    return detectAngleTags(caption).filter((tag) => !text.includes(tag));
  }

  function withAngleTags(caption, tags) {
    const base = String(caption || "").trim();
    if (!tags || !tags.length) return base;
    return base ? `${tags.join(", ")}, ${base}` : tags.join(", ");
  }

  function renderAudit(ds) {
    const box = $("#dsAudit");
    if (!box) return;
    const images = ds.images || [];
    const kind = ds.kind || "character";
    const [targetLow, targetHigh] = KIND_TARGETS[kind] || KIND_TARGETS.character;
    const low = images.length < targetLow;
    const tagged = images.filter(
      (img) => (img.tags || []).length || String(img.caption || "").trim()
    ).length;

    if (kind !== "character") {
      box.innerHTML = `
        <div class="audit-pills">
          <span class="audit-pill">ทั้งหมด <b>${images.length}</b></span>
          <span class="audit-pill ${low ? "warn" : "ok"}">เป้าหมาย <b>${targetLow}–${targetHigh}</b></span>
          <span class="audit-pill ${tagged === images.length && images.length ? "ok" : ""}">มี caption/tag <b>${tagged}/${images.length}</b></span>
        </div>
        <span class="micro">${KIND_HINTS[kind] || KIND_HINTS.character}</span>
      `;
      return;
    }

    const soloImages = images.filter((img) => !img.multi);
    const solo = soloImages.length;
    const multi = images.length - solo;
    const noAngle = soloImages.filter((img) => !hasAngleTag(img.caption)).length;
    const fillable = soloImages.filter(
      (img) => !hasAngleTag(img.caption) && missingAngleTags(img.caption).length
    ).length;
    const manual = noAngle - fillable;
    const likely = images.filter(
      (img) => !img.multi && looksMultiCaption(img.caption)
    ).length;
    box.innerHTML = `
      <div class="audit-pills">
        <span class="audit-pill">ทั้งหมด <b>${images.length}</b></span>
        <span class="audit-pill ok">เดี่ยว <b>${solo}</b></span>
        <span class="audit-pill ${multi ? "warn" : ""}">หลายตัวละคร <b>${multi}</b></span>
        <span class="audit-pill ${noAngle ? "warn" : "ok"}">tag มุม (เดี่ยว) <b>${noAngle}</b></span>
        <span class="audit-pill ${low ? "warn" : "ok"}">เป้าหมาย <b>${targetLow}–${targetHigh}</b></span>
      </div>
      ${
        likely
          ? `<button type="button" class="ghost-btn" id="dsMarkMulti">ทำเครื่องหมาย ${likely} รูปที่น่ามีคนที่สอง</button>`
          : ""
      }
      ${
        fillable
          ? `<button type="button" class="ghost-btn" id="dsAutoTag">เติม tag มุมอัตโนมัติ (${fillable} รูปเดี่ยว)</button>`
          : ""
      }
      ${
        manual
          ? `<span class="micro warn">อีก ${manual} รูปเดี่ยว caption ไม่ได้บอกมุม — เติมเองจากแถบ tag ด้านล่าง</span>`
          : ""
      }
      <span class="micro">${KIND_HINTS.character}</span>
    `;
  }

  function renderTrainer() {
    renderTrainerModels();
    const sel = $("#trainerDs");
    if (!sel) return;
    const datasets = state.datasets || [];
    const current = state.activeDs;
    sel.innerHTML =
      `<option value="">— เลือกชุดเทรน (${datasets.length}) —</option>` +
      datasets
        .map(
          (d) =>
            `<option value="${esc(d.id)}">${esc(d.name)} (${(d.images || []).length} รูป)</option>`
        )
        .join("");
    sel.value = current && datasets.some((d) => d.id === current) ? current : "";
    if (!sel.value) state.activeDs = null;
    const ds = activeDataset();
    $("#trainerActive").classList.toggle("hidden", !ds);
    $("#trainerNewBox").classList.toggle("hidden", !!ds);
    if (!ds) return;
    $("#dsTitle").textContent = ds.name + (ds.trigger ? ` · trigger: ${ds.trigger}` : "");
    $("#dsMeta").textContent = KIND_LABELS[ds.kind] || ds.kind || "ตัวละคร";
    const kindSel = $("#dsKindEdit");
    if (kindSel && document.activeElement !== kindSel) {
      kindSel.value = ds.kind || "character";
    }
    const triggerInput = $("#dsTriggerEdit");
    if (triggerInput && document.activeElement !== triggerInput) {
      triggerInput.value = ds.trigger || "";
    }
    const extraInput = $("#dsExtraTriggers");
    if (extraInput && document.activeElement !== extraInput) {
      extraInput.value = (ds.extra_triggers || []).join(", ");
    }
    const hint = $("#trainHint");
    if (hint) hint.textContent = TRAIN_HINTS[ds.kind] || TRAIN_HINTS.character;
    const triggers = trainerTriggers(ds);
    const list = $("#captionList");
    list.innerHTML = (ds.images || [])
      .map((img) => {
        const options = [
          `<option value="">— ใช้ trigger หลัก —</option>`,
          ...triggers.map(
            (t) =>
              `<option value="${esc(t)}"${(img.trigger || "") === t ? " selected" : ""}>${esc(t)}</option>`
          ),
        ].join("");
        return `<div class="caption-row" data-file="${esc(img.file)}">
          <img class="caption-thumb" loading="lazy" src="/files/train/${esc(ds.id)}/${esc(img.file)}" alt="">
          <div class="caption-main">
            <textarea class="caption-input" rows="2" maxlength="600"
              placeholder="caption — trigger word ถูกเติมให้อัตโนมัติ">${esc(img.caption || "")}</textarea>
            <div class="caption-extra">
              <select class="caption-trigger" title="trigger ของรูปนี้">${options}</select>
              <input class="caption-tags" type="text" maxlength="240"
                placeholder="tags (คั่นด้วย ,)" value="${esc((img.tags || []).join(", "))}">
              <span class="caption-repeats-label">×</span>
              <input class="caption-repeats" type="number" min="1" max="10" step="1"
                value="${Number(img.repeats) || 1}" title="จำนวนรอบต่อ epoch">
            </div>
            <span class="q-sub">${esc(img.name || img.file)}</span>
          </div>
          <label class="caption-role" title="รูปนี้มีตัวละครอื่นอยู่ในภาพหรือไม่" style="${ds.kind === "character" ? "" : "display:none"}">
            <select class="caption-multi">
              <option value="solo"${img.multi ? "" : " selected"}>เดี่ยว</option>
              <option value="multi"${img.multi ? " selected" : ""}>หลายตัวละคร</option>
            </select>
          </label>
          <button type="button" class="ghost-btn danger-hover caption-del" title="ลบรูปนี้">✕</button>
        </div>`;
      })
      .join("");
    renderAudit(ds);
    updateCalcSteps();
  }

  function collectCaptions() {
    const ds = activeDataset();
    if (!ds) return [];
    const isCharacter = (ds.kind || "character") === "character";
    return $$("#captionList .caption-row").map((row) => ({
      file: row.dataset.file,
      caption: row.querySelector(".caption-input").value.trim(),
      // Non-character datasets hide the selector; omitting the field keeps
      // the stored flag intact if the user switches the kind back later.
      ...(isCharacter
        ? { multi: row.querySelector(".caption-multi").value === "multi" }
        : {}),
      tags: row
        .querySelector(".caption-tags")
        .value.split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
      trigger: row.querySelector(".caption-trigger").value,
      repeats: Number(row.querySelector(".caption-repeats").value) || 1,
    }));
  }

  async function saveCaptions() {
    const ds = activeDataset();
    if (!ds) return;
    try {
      await api(`/api/train/datasets/${ds.id}/captions`, {
        method: "PUT",
        body: JSON.stringify({ updates: collectCaptions() }),
      });
      toast("บันทึก caption แล้ว", "ok", 2000);
      await refreshTrainer();
    } catch (err) {
      toast(err.message, "err", 6000);
    }
  }

  function clampNumber(value, min, max, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, n));
  }

  function updateCalcSteps() {
    const label = $("#calcSteps");
    if (!label) return 0;
    const rows = $$("#captionList .caption-row");
    const totalRepeats = rows.length
      ? rows.reduce(
          (sum, row) =>
            sum + clampNumber(row.querySelector(".caption-repeats")?.value, 1, 10, 1),
          0
        )
      : (activeDataset()?.images || []).reduce(
          (sum, img) => sum + clampNumber(img.repeats, 1, 10, 1),
          0
        );
    const epochs = clampNumber($("#calcEpochs")?.value, 1, 100, 10);
    const accum = clampNumber($("#calcAccum")?.value, 1, 8, 1);
    const steps = Math.max(1, Math.round((totalRepeats * epochs) / accum));
    label.textContent = `≈ ${steps.toLocaleString()} steps · ${totalRepeats} samples/epoch`;
    return steps;
  }

  function applyTrainPreset(name) {
    const preset = TRAIN_PRESETS[name];
    if (!preset) return;
    const model = currentTrainerModel();
    const isSdxl = model && model.family === "sdxl";
    const resolution = isSdxl ? Math.min(preset.resolution, 768) : preset.resolution;
    const steps = isSdxl ? Math.round(preset.steps * 1.5) : preset.steps;
    if ($("#trainRank")) $("#trainRank").value = String(preset.rank);
    if ($("#trainSteps")) $("#trainSteps").value = String(steps);
    if ($("#trainLr")) $("#trainLr").value = String(preset.lr);
    if ($("#trainRes")) $("#trainRes").value = String(resolution);
    if ($("#trainScheduler")) $("#trainScheduler").value = preset.scheduler;
    if ($("#trainSampling")) $("#trainSampling").value = preset.sampling;
    if ($("#trainSaveEvery")) $("#trainSaveEvery").value = String(preset.saveEvery);
    if ($("#trainMultiMode")) $("#trainMultiMode").value = preset.multiMode;
    if ($("#trainMultiWeight")) {
      $("#trainMultiWeight").disabled = preset.multiMode !== "weighted";
    }
    if ($("#trainEma")) $("#trainEma").checked = preset.ema;
    if ($("#trainFlip")) $("#trainFlip").checked = preset.flip;
    if ($("#trainBucketing")) $("#trainBucketing").checked = !!preset.bucketing;
    if ($("#trainCaptionDropout")) {
      $("#trainCaptionDropout").value = String(preset.captionDropout ?? 0.05);
    }
    if ($("#trainCaptionShuffle")) {
      $("#trainCaptionShuffle").checked = !!preset.captionShuffle;
    }
    if ($("#trainKeepTokens")) $("#trainKeepTokens").value = String(preset.keepTokens ?? 1);
    if ($("#trainNoiseOffset")) $("#trainNoiseOffset").value = String(preset.noiseOffset ?? 0);
    if ($("#trainMinSnr")) $("#trainMinSnr").value = String(preset.minSnr ?? 0);
    if ($("#trainRsLora")) $("#trainRsLora").checked = !!preset.rsLora;
    if ($("#trainTargetModules")) {
      $("#trainTargetModules").value = preset.targetModules || "attn";
    }
    updateCalcSteps();
    toast(
      `ใช้ preset ${name} แล้ว${isSdxl ? " (ปรับเป็น SDXL 768px)" : ""}`,
      "ok"
    );
  }

  function readTrainConfig() {
    return {
      format: "pixelsmith-train-config",
      version: 1,
      kind: activeDataset()?.kind || "character",
      rank: Number($("#trainRank")?.value) || 32,
      steps: Number($("#trainSteps")?.value) || 1500,
      lr: Number($("#trainLr")?.value) || 0.0002,
      resolution: Number($("#trainRes")?.value) || 1024,
      seed: Number($("#trainSeed")?.value ?? -1),
      scheduler: $("#trainScheduler")?.value || "cosine",
      timestep_sampling: $("#trainSampling")?.value || "logit_normal",
      save_every: Number($("#trainSaveEvery")?.value) || 0,
      grad_accum: Number($("#trainGradAccum")?.value) || 1,
      multi_mode: $("#trainMultiMode")?.value || "exclude",
      multi_weight: (Number($("#trainMultiWeight")?.value) || 25) / 100,
      ema: !!$("#trainEma")?.checked,
      augment_flip: !!$("#trainFlip")?.checked,
      bucketing: $("#trainBucketing") ? !!$("#trainBucketing").checked : true,
      caption_dropout: Number($("#trainCaptionDropout")?.value ?? 0.05),
      caption_shuffle: !!$("#trainCaptionShuffle")?.checked,
      keep_tokens: Number($("#trainKeepTokens")?.value ?? 1),
      noise_offset: Number($("#trainNoiseOffset")?.value ?? 0),
      min_snr_gamma: Number($("#trainMinSnr")?.value ?? 0),
      use_rslora: !!$("#trainRsLora")?.checked,
      target_modules: $("#trainTargetModules")?.value || "attn",
    };
  }

  function applyTrainConfig(cfg) {
    if (!cfg || cfg.format !== "pixelsmith-train-config") {
      throw new Error("ไฟล์ config ไม่ถูกต้อง");
    }
    const setSelect = (selector, value) => {
      const el = $(selector);
      if (el && Array.from(el.options).some((opt) => opt.value === String(value))) {
        el.value = String(value);
      }
    };
    const rankOptions = Array.from($("#trainRank")?.options || []).map((o) => Number(o.value));
    const rank = clampNumber(cfg.rank, 4, 128, 32);
    if (rankOptions.includes(rank) && $("#trainRank")) {
      $("#trainRank").value = String(rank);
    }
    const resOptions = Array.from($("#trainRes")?.options || []).map((o) => Number(o.value));
    const resolution = clampNumber(cfg.resolution, 512, 1024, 1024);
    if ($("#trainRes")) {
      $("#trainRes").value = String(
        resOptions.includes(resolution) ? resolution : resOptions.includes(768) ? 768 : 1024
      );
    }
    if ($("#trainSteps")) {
      $("#trainSteps").value = String(clampNumber(cfg.steps, 10, 20000, 1500));
    }
    if ($("#trainLr")) {
      $("#trainLr").value = String(clampNumber(cfg.lr, 0.00001, 0.0005, 0.0002));
    }
    if ($("#trainSeed")) {
      $("#trainSeed").value = String(clampNumber(cfg.seed, -1, 2147483647, -1));
    }
    setSelect("#trainScheduler", cfg.scheduler || "cosine");
    setSelect("#trainSampling", cfg.timestep_sampling || "logit_normal");
    if ($("#trainSaveEvery")) {
      $("#trainSaveEvery").value = String(clampNumber(cfg.save_every, 0, 20000, 0));
    }
    if ($("#trainGradAccum")) {
      $("#trainGradAccum").value = String(clampNumber(cfg.grad_accum, 1, 8, 1));
    }
    setSelect("#trainMultiMode", cfg.multi_mode || "exclude");
    if ($("#trainMultiWeight")) {
      $("#trainMultiWeight").value = String(
        Math.round(clampNumber(cfg.multi_weight, 0.05, 1, 0.25) * 100)
      );
      $("#trainMultiWeight").disabled = $("#trainMultiMode")?.value !== "weighted";
    }
    if ($("#trainEma")) $("#trainEma").checked = !!cfg.ema;
    if ($("#trainFlip")) $("#trainFlip").checked = !!cfg.augment_flip;
    if ($("#trainBucketing")) $("#trainBucketing").checked = cfg.bucketing !== false;
    if ($("#trainCaptionDropout")) {
      $("#trainCaptionDropout").value = String(
        clampNumber(cfg.caption_dropout, 0, 0.3, 0.05)
      );
    }
    if ($("#trainCaptionShuffle")) {
      $("#trainCaptionShuffle").checked = !!cfg.caption_shuffle;
    }
    if ($("#trainKeepTokens")) {
      $("#trainKeepTokens").value = String(clampNumber(cfg.keep_tokens, 0, 4, 1));
    }
    if ($("#trainNoiseOffset")) {
      $("#trainNoiseOffset").value = String(clampNumber(cfg.noise_offset, 0, 0.2, 0));
    }
    if ($("#trainMinSnr")) {
      $("#trainMinSnr").value = String(clampNumber(cfg.min_snr_gamma, 0, 20, 0));
    }
    if ($("#trainRsLora")) $("#trainRsLora").checked = !!cfg.use_rslora;
    setSelect("#trainTargetModules", cfg.target_modules || "attn");
    updateCalcSteps();
  }

  function exportTrainConfig() {
    const cfg = readTrainConfig();
    const blob = new Blob([JSON.stringify(cfg, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `pixelsmith-train-${cfg.kind}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("ส่งออก config แล้ว", "ok");
  }

  async function importTrainConfig(file) {
    try {
      applyTrainConfig(JSON.parse(await file.text()));
      toast("นำเข้า config แล้ว", "ok");
    } catch (err) {
      toast(`นำเข้า config ไม่สำเร็จ: ${err.message}`, "err", 6000);
    }
  }

  function variantOptions(item) {
    const auto = `${tr("อัตโนมัติ")} (${variantLabel(item.variant, item.family_label || item.family)})`;
    const opts = [
      `<option value="" ${item.variant_source === "override" ? "" : "selected"}>${esc(auto)}</option>`,
    ];
    for (const v of VARIANT_CHOICES) {
      const sel = item.variant_source === "override" && item.variant === v ? "selected" : "";
      opts.push(`<option value="${v}" ${sel}>${esc(variantLabel(v))}</option>`);
    }
    return opts.join("");
  }

  function bindVariantSelects(root, after) {
    $$(`${root} .lora-variant`).forEach((sel) => {
      sel.addEventListener("change", async () => {
        try {
          await api("/api/variants", {
            method: "POST",
            body: JSON.stringify({ path: sel.dataset.path, variant: sel.value }),
          });
          await after();
          toast(tr("อัปเดต variant แล้ว"), "ok", 1800);
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
    });
  }

  function renderLoraManager() {
    const box = $("#loraManagerBox");
    const list = $("#loraManagerList");
    if (!box || !list) return;
    const loras = state.loras || [];
    box.classList.remove("hidden");
    list.innerHTML = loras.length
      ? loras
          .map((l) => {
            const tag = l.family ? l.family_label || String(l.family).toUpperCase() : "any";
            return `<div class="lora-item" data-id="${esc(l.id)}">
              <span class="lora-name">${esc(l.name)}</span>
              <span class="lora-tag">${esc(tag)}</span>
              <span class="lora-size">${esc(((l.size || 0) / 1024 / 1024).toFixed(1))} MB</span>
              <select class="lora-variant" data-path="${esc(l.path || "")}" title="แก้ variant ของ LoRA">
                ${variantOptions(l)}
              </select>
              <button type="button" class="ghost-icon" data-id="${esc(l.id)}" title="ลบ LoRA">
                <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M9 3h6a1 1 0 0 1 1 1v2h4v2h-1v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6H3V4h4V3Zm1 2v2h4V5H10Zm-3 3v12h10V7H7Zm3 3a1 1 0 0 1 1 1v6a1 1 0 0 1-2 0v-6a1 1 0 0 1 1-1Zm4 0a1 1 0 0 1 1 1v6a1 1 0 0 1-2 0v-6a1 1 0 0 1 1-1Z"/></svg>
              </button>
            </div>`;
          })
          .join("")
      : `<p class="micro">ยังไม่มี LoRA — นำเข้าไฟล์ .safetensors ด้านบน</p>`;
    bindVariantSelects("#loraManagerList", async () => {
      await refreshLoras();
    });
    $$("#loraManagerList .lora-item button[data-id]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!(await uiConfirm(tr(`ลบ LoRA "${btn.closest('.lora-item').querySelector('.lora-name').textContent}" ใช่หรือไม่?`)))) return;
        try {
          await api("/api/loras/delete", { method: "POST", body: JSON.stringify({ id }) });
          await refreshLoras();
          toast("ลบแล้ว", "ok");
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
    });
  }

  function renderModelVariants() {
    const list = $("#modelVariantList");
    if (!list) return;
    const models = state.registry || [];
    list.innerHTML = models.length
      ? models
          .map(
            (m) => `<div class="lora-item" data-id="${esc(m.id)}">
              <span class="lora-name">${esc(m.name)}</span>
              <span class="lora-tag">${esc(m.family_label || m.family || "")}</span>
              <select class="lora-variant" data-path="${esc(m.path || "")}" title="แก้ variant ของโมเดล">
                ${variantOptions(m)}
              </select>
            </div>`
          )
          .join("")
      : `<p class="micro">—</p>`;
    bindVariantSelects("#modelVariantList", async () => {
      await refreshModels(state.currentId || undefined);
      await refreshLoras();
    });
  }

  async function onModelChange(event) {
    if (state.busy) {
      event.target.value = state.currentId || "";
      return;
    }
    const keep = readLoras();
    const ok = await selectModel(event.target.value, { applyDefaults: true });
    if (ok) {
      renderLoras(keep);
      // Drop adapters from the previous family; keep same-family ones that the
      // variant filter may be hiding.
      state.settings.loras = readLoras();
      saveSettings();
    }
  }

  function setLoraDisabled(disabled) {
    $$("#loraList .lora-item").forEach((item) => {
      const checkbox = item.querySelector("input[type=checkbox]");
      const weight = item.querySelector(".lora-weight");
      checkbox.disabled = disabled;
      weight.disabled = disabled || !checkbox.checked;
    });
  }

  // ── GPU power modes ─────────────────────────────────────────────────────
  function powerLabel(modeId) {
    return { low: "Low temp", balanced: "Balanced", performance: "Performance" }[
      modeId
    ] || modeId;
  }

  function renderPower(p) {
    state.power = p || null;
    const modes = p && Array.isArray(p.modes) ? p.modes : [];
    const byId = Object.fromEntries(modes.map((m) => [m.id, m]));
    $$("#powerModes .power-mode").forEach((btn) => {
      const m = byId[btn.dataset.mode];
      const watts = btn.querySelector(".power-watts");
      if (watts) watts.textContent = m ? `${m.watts} W` : "—";
      btn.disabled =
        !p || !p.available || p.supported === false || !!p.setup_required;
      // Highlight the mode the app has saved, not just what the card is doing
      // right now: the watchdog may still be re-applying it after a reset.
      const chosen = p && (p.saved_mode || p.mode);
      btn.classList.toggle("active", !!chosen && chosen === btn.dataset.mode);
    });
    const stateEl = $("#powerState");
    if (stateEl) {
      if (!p || !p.available) {
        stateEl.textContent = "ไม่พบการ์ด NVIDIA";
      } else if (p.supported === false) {
        stateEl.textContent = "ใช้ได้เฉพาะ Linux";
      } else {
        const bits = [`${Math.round(p.limit)} W`];
        if (Number.isFinite(p.temp)) bits.push(`${Math.round(p.temp)}°C`);
        if (Number.isFinite(p.draw)) bits.push(`${p.draw.toFixed(1)} W`);
        stateEl.textContent = bits.join(" · ");
      }
    }
    const hint = $("#powerHint");
    if (hint) {
      if (!p || !p.available) {
        hint.textContent = "โหมดพลังงานใช้ได้เฉพาะเครื่องที่มีการ์ด NVIDIA";
      } else if (p.supported === false) {
        hint.textContent = "โหมดพลังงานใช้ได้เฉพาะ Linux ที่มีการ์ด NVIDIA";
      } else if (p.setup_required) {
        hint.textContent =
          "เปิดใช้งานครั้งเดียว (ต้องใส่รหัสผ่านระบบ) — หลังจากนั้นสลับโหมดได้ทันทีโดยไม่ต้องใส่รหัส";
      } else {
        hint.textContent =
          "ใช้โหมดที่เลือกตลอดเวลาแอปเปิด — เมื่อปิดแอปจะกลับไปใช้ค่าเริ่มต้นของการ์ด";
      }
    }
    const setupBtn = $("#powerSetup");
    if (setupBtn) {
      setupBtn.classList.toggle(
        "hidden",
        !p || p.supported === false || !p.setup_required
      );
    }
  }

  async function refreshPower() {
    try {
      renderPower(await api("/api/power"));
    } catch (err) {
      renderPower({ available: false });
    }
  }

  async function setPowerMode(mode) {
    const btn = document.querySelector(
      `#powerModes .power-mode[data-mode="${mode}"]`
    );
    if (btn) btn.disabled = true;
    showBlocking();
    try {
      const p = await api("/api/power", {
        method: "POST",
        body: JSON.stringify({ mode }),
      });
      renderPower(p);
      const m = (p.modes || []).find((x) => x.id === mode);
      toast(
        `เปลี่ยนโหมดพลังงานเป็น ${powerLabel(mode)}${m ? ` (${m.watts} W)` : ""} แล้ว`,
        "ok"
      );
    } catch (err) {
      toast(err.message, "err");
      await refreshPower();
    } finally {
      hideBlocking();
    }
  }

  async function setupPower() {
    const btn = $("#powerSetup");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "กำลังเปิดใช้งาน…";
    }
    showBlocking("กำลังเปิดใช้งานโหมดพลังงาน กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      const p = await api("/api/power/setup", { method: "POST" });
      renderPower(p);
      toast("เปิดใช้งานโหมดพลังงานแล้ว", "ok");
    } catch (err) {
      toast(err.message, "err", 6000);
    } finally {
      hideBlocking();
      if (btn) {
        btn.disabled = false;
        btn.textContent = "เปิดใช้งานโหมดพลังงาน";
      }
    }
  }

  // ── backup / restore ────────────────────────────────────────────────────
  function mbLabel(bytes) {
    const mb = (Number(bytes) || 0) / 1048576;
    return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
  }

  async function exportBackup() {
    const btn = $("#backupExport");
    if (btn) btn.disabled = true;
    showBlocking("กำลังสำรองข้อมูล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      const res = await api("/api/backup/export", { method: "POST" });
      const state = $("#backupState");
      if (state) state.textContent = `${res.files} ไฟล์ · ${mbLabel(res.size)}`;
      toast(`สำรองข้อมูลแล้ว: ${res.files} ไฟล์ · ${mbLabel(res.size)}`, "ok", 8000);
    } catch (err) {
      toast(err.message, "err", 7000);
    } finally {
      hideBlocking();
      if (btn) btn.disabled = false;
    }
  }

  async function importBackup(file) {
    if (!file) return;
    const ok = await uiConfirm(
      tr("กู้คืนข้อมูลจากไฟล์สำรอง? ข้อมูลที่มีอยู่จะไม่ถูกลบ — ระบบจะเพิ่มเฉพาะรายการที่ยังไม่มี"),
      { title: "กู้คืนจาก ZIP", confirmLabel: "กู้คืน", cancelLabel: "ยกเลิก", danger: false }
    );
    if (!ok) return;
    const btn = $("#backupImport");
    if (btn) btn.disabled = true;
    showBlocking("กำลังกู้คืนข้อมูล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      const form = new FormData();
      form.append("file", file, file.name || "backup.zip");
      const res = await api("/api/backup/restore", { method: "POST", body: form });
      const r = res.restored || {};
      const total = Object.values(r).reduce((sum, n) => sum + (Number(n) || 0), 0);
      const reload = await uiConfirm(
        tr(
          `กู้คืนแล้ว ${total} รายการ (ประวัติ ${r.history || 0} · ภาพ ${r.images || 0} · presets ${r.presets || 0} · dataset ${r.datasets || 0}) — โหลดหน้าใหม่เพื่อเห็นข้อมูลทั้งหมด?`
        ),
        { title: "กู้คืนสำเร็จ", confirmLabel: "โหลดใหม่", cancelLabel: "ไว้ก่อน", danger: false }
      );
      if (reload) location.reload();
    } catch (err) {
      toast(err.message, "err", 9000);
    } finally {
      hideBlocking();
      if (btn) btn.disabled = false;
      const input = $("#backupFile");
      if (input) input.value = "";
    }
  }

  // ── model status ────────────────────────────────────────────────────────
  function updateModelBadge(model) {
    const badge = $("#modelBadge");
    const text = $("#modelBadgeText");
    if (!badge) return;
    const stateName = (model && model.state) || "error";
    const map = {
      ready: { cls: "ok", label: "พร้อม", title: model?.message || "โมเดลพร้อมใช้งาน" },
      loading: { cls: "busy", label: "กำลังโหลด", title: model?.message || "กำลังโหลดโมเดล…" },
      preparing: { cls: "busy", label: "กำลังดาวน์โหลด", title: model?.message || "กำลังดาวน์โหลดโมเดล…" },
      not_loaded: { cls: "warn", label: "ยังไม่โหลด", title: "จะโหลดอัตโนมัติเมื่อกดสร้างภาพ" },
      error: { cls: "err", label: "ผิดพลาด", title: model?.message || "engine ผิดพลาด" },
    };
    const info = map[stateName] || map.error;
    badge.className = `model-badge ${info.cls}`;
    text.textContent = info.label;
    badge.title = info.title;
  }

  async function refreshHealth() {
    state.healthTicks = (state.healthTicks || 0) + 1;
    if (state.healthTicks % 8 === 0) {
      const pane = document.querySelector('#panel .pane[data-pane="settings"]');
      if (pane && pane.classList.contains("active")) refreshPower();
    }
    try {
      const h = await api("/api/health");
      state.model = h.model;
      updateModelBadge(h.model);
      if (h.gpu && h.gpu.gpu) {
        const temp = Number(h.gpu.temp);
        const hasTemp = Number.isFinite(temp) && temp > 0;
        $("#stageSub").textContent = `${h.gpu.gpu} · VRAM ${gb(h.gpu.vram_used)}/${gb(
          h.gpu.vram_total
        )} GB${hasTemp ? ` · ${Math.round(temp)}°C` : ""} · ${
          h.model.loaded ? "โมเดลพร้อม" : "ยังไม่โหลดโมเดล"
        }`;
        const sub = $("#stageSub");
        sub.classList.toggle("hot", hasTemp && temp >= 82);
        sub.classList.toggle("very-hot", hasTemp && temp >= 88);
      } else {
        $("#stageSub").textContent = "ภาพที่สร้างจะถูกเก็บไว้ในเครื่องนี้";
      }
    } catch (err) {
      updateModelBadge({ state: "error", message: err.message });
      $("#stageSub").textContent = "เชื่อมต่อ engine ไม่ได้ — ลองรีโหลด";
    }
  }

  // ── generation ──────────────────────────────────────────────────────────
  // busy = a job is actually running; queued jobs never disable the form
  // (submitting more jobs to the queue is the whole point).
  function setBusy(running) {
    state.busy = running;
    $("#cancel").classList.toggle("hidden", !running);
    $("#progress").classList.toggle("hidden", !running);
    if ($("#modelSelect")) $("#modelSelect").disabled = running;
    setLoraDisabled(running);
    const capable = regionCapable();
    const regionOn = $("#regionEnabled") && $("#regionEnabled").checked;
    setRegionDisabled(running || !capable || !regionOn);
    if ($("#regionEnabled")) $("#regionEnabled").disabled = running || !capable;
  }

  async function onGenerate(event) {
    event.preventDefault();
    if (state.activeCount >= MAX_QUEUE) {
      toast(`คิวเต็ม (สูงสุด ${MAX_QUEUE} งาน) — รองานเดิมเสร็จก่อน`, "warn");
      return;
    }
    const settings = readForm();
    if (!settings.prompt.trim()) {
      toast("ใส่พรอมป์ตก่อนสร้างภาพครับ", "warn");
      $("#prompt").focus();
      return;
    }
    const params = {
      ...settings,
      regions:
        settings.regions_enabled && regionCapable()
          ? settings.regions.filter((r) => r.prompt)
          : [],
      inpaint: settings.inpaint && inpaintCapable(),
      hires: {
        enabled: !!(settings.hires_enabled && hiresCapable() && !settings.inpaint),
        scale: settings.hires_scale,
        strength: settings.hires_strength,
      },
    };
    state.settings = settings;
    saveSettings();
    const queuedBehind = state.activeCount > 0;
    try {
      const res = await api("/api/generate", {
        method: "POST",
        body: JSON.stringify(params),
      });
      state.knownActive.add(res.job_id);
      startQueuePolling();
      await tickQueue();
      if (queuedBehind) {
        toast(`เพิ่มเข้าคิวแล้ว (ลำดับ ${state.activeCount})`, "ok", 2600);
      }
    } catch (err) {
      toast(err.message, "err", 7000);
    }
  }

  // ── multi-job queue polling ─────────────────────────────────────────────
  async function tickQueue() {
    let data;
    try {
      data = await api("/api/jobs?active=1&limit=24");
    } catch (err) {
      return; // engine hiccup — retry on the next tick
    }
    try {
      await runTickBody(data);
    } catch (err) {
      console.error("tickQueue:", err); // never let one bad tick kill the poll loop
    }
  }

  async function runTickBody(data) {
    const jobs = data.jobs || [];
    state.activeCount = jobs.length;
    const current = new Set(jobs.map((j) => j.id));

    // resolve jobs that left the active list (done / error / canceled)
    for (const id of Array.from(state.knownActive)) {
      if (current.has(id)) continue;
      state.knownActive.delete(id);
      resolveFinished(id);
    }
    jobs.forEach((j) => state.knownActive.add(j.id));

    const running = jobs.find((j) => j.status === "running") || null;
    const queued = jobs.filter((j) => j.status === "queued");
    state.runningJobId = running ? running.id : null;
    setBusy(!!running);

    if (running) {
      $("#live").classList.remove("hidden");
      updateLive(running);
    } else {
      state.runningJobId = null;
    }
    renderQueue(queued);
    $("#queuePanel").classList.toggle("hidden", queued.length === 0);
    if (!jobs.length) stopQueuePolling();
  }

  function startQueuePolling() {
    if (state.poll) return;
    state.pollStart = Date.now();
    state.poll = setInterval(tickQueue, 900);
  }

  function stopQueuePolling() {
    if (!state.poll) return;
    clearInterval(state.poll);
    state.poll = null;
    state.activeCount = 0;
    state.runningJobId = null;
  }

  async function resolveFinished(id) {
    try {
      const job = await api(`/api/jobs/${id}`);
      const isTrain = (job.params || {}).job_type === "train";
      if (job.status === "done") {
        if (isTrain) {
          $("#live").classList.remove("hidden");
          updateLive(job); // repaint the stale "training…" panel as finished
          const lora = (job.result || {}).lora || {};
          toast(`เทรนเสร็จ — LoRA "${lora.name || job.params.lora_name}" พร้อมใช้แล้ว`, "ok", 6000);
          await Promise.all([refreshLoras(), refreshTrainer()]);
        } else {
          $("#live").classList.remove("hidden");
          updateLive(job); // show the final result in the live area
          toast(
            job.images.length > 1 ? `สร้างเสร็จ ${job.images.length} ภาพ` : "สร้างภาพเสร็จแล้ว",
            "ok"
          );
          await refreshHistory(true);
        }
      } else if (job.status === "error") {
        $("#live").classList.remove("hidden");
        updateLive(job);
        toast(`${isTrain ? "เทรนไม่สำเร็จ" : "สร้างไม่สำเร็จ"}: ${job.error || job.message}`, "err", 9000);
      } else if (job.status === "canceled") {
        $("#live").classList.remove("hidden");
        updateLive(job);
        toast(isTrain ? "ยกเลิกการเทรนแล้ว" : "ยกเลิกแล้ว", "warn");
        if (job.images.length) await refreshHistory(true);
      }
      refreshHealth();
    } catch (e) {
      /* job already purged */
    }
  }

  function renderQueue(queued) {
    const list = $("#queueList");
    if (!list) return;
    $("#queueMeta").textContent = queued.length ? `${queued.length} งานรออยู่` : "";
    list.innerHTML = queued
      .map((job, i) => {
        const p = job.params || {};
        const sub = p.job_type === "train"
          ? [
              `${(p.steps || 0)} steps`,
              `rank ${p.rank || 32}`,
              `${p.resolution || 1024}px`,
            ].join(" · ")
          : [
              job.model_label || "",
              `${p.width || "?"}×${p.height || "?"}`,
              Number(p.batch) > 1 ? `${p.batch} ภาพ` : "",
              p.hires && p.hires.enabled ? "hires" : "",
            ]
              .filter(Boolean)
              .join(" · ");
        return `<div class="queue-item" data-id="${esc(job.id)}">
          <span class="q-pos">${i + 1}</span>
          <div class="q-info">
            <span class="q-prompt">${esc(
              p.job_type === "train"
                ? `เทรน LoRA: ${p.lora_name || ""}`
                : (p.prompt || "").slice(0, 90)
            )}</span>
            <span class="q-sub">${esc(sub)}</span>
          </div>
          <div class="q-actions">
            <button type="button" data-act="up" title="เลื่อนขึ้น" ${i === 0 ? "disabled" : ""}>▲</button>
            <button type="button" data-act="down" title="เลื่อนลง" ${i === queued.length - 1 ? "disabled" : ""}>▼</button>
            <button type="button" class="danger-hover" data-act="cancel" title="ยกเลิกงานนี้">✕</button>
          </div>
        </div>`;
      })
      .join("");
  }

  function updateLive(job) {
    state.liveJob = job;
    const live = $("#live");
    if (state.liveJobId !== job.id) {
      // switching to a different job — clear stale result images
      state.liveJobId = job.id;
      state.pollStart = Date.now();
      const box = $("#liveImages");
      if (box) box.innerHTML = "";
      $("#fill").style.width = "0%";
      $("#liveFill").style.width = "0%";
    }
    live.classList.remove("hidden", "done", "err");
    if (job.status === "done") live.classList.add("done");
    if (job.status === "error" || job.status === "canceled") live.classList.add("err");

    let title = "กำลังสร้างภาพ…";
    const isTrainJob = (job.params || {}).job_type === "train";
    if (isTrainJob) title = "กำลังเทรน LoRA…";
    if (job.status === "queued") title = isTrainJob ? "เทรนอยู่ในคิว…" : "อยู่ในคิว…";
    else if (job.status === "done") title = isTrainJob ? "เทรนเสร็จแล้ว" : "ผลลัพธ์ล่าสุด";
    else if (job.status === "error") title = isTrainJob ? "เทรนไม่สำเร็จ" : "เกิดข้อผิดพลาด";
    else if (job.status === "canceled") title = isTrainJob ? "ยกเลิกการเทรนแล้ว" : "ยกเลิกแล้ว";
    else if (job.phase === "preparing") title = "กำลังเตรียมโมเดล…";
    else if (job.phase === "loading") title = "กำลังโหลดโมเดล…";
    else if (job.phase === "paused") title = "พัก GPU ให้เย็นลง…";
    $("#liveTitle").textContent = title;

    const elapsed =
      job.elapsed != null ? job.elapsed : (Date.now() - state.pollStart) / 1000;
    const stepText = job.steps ? `${job.step || 0}/${job.steps}` : "";
    const batchText = job.batch > 1 ? `ภาพ ${(job.image_index || 0) + 1}/${job.batch}` : "";
    const lossText = job.loss != null ? `loss ${job.loss}` : "";
    $("#liveMeta").textContent = [stepText, batchText, lossText, fmtClock(elapsed)].filter(Boolean).join(" · ");

    const pct = Math.round(clamp(Number(job.progress) || 0, 0, 1) * 100);
    $("#fill").style.width = `${pct}%`;
    $("#liveFill").style.width = `${pct}%`;
    $("#progText").textContent = job.message || title;
    $("#progTime").textContent = `${pct}% · ${fmtClock(elapsed)}`;

    const box = $("#liveImages");
    const existing = new Set($$(".shot", box).map((el) => el.dataset.file));
    (job.images || []).forEach((img) => {
      if (existing.has(img.file)) return;
      existing.add(img.file);
      const shot = document.createElement("div");
      shot.className = "shot";
      shot.dataset.file = img.file;
      shot.innerHTML = `<img loading="lazy" src="${esc(img.thumb)}" alt=""><span class="seed">seed ${esc(
        img.seed
      )}</span>`;
      shot.addEventListener("click", () => {
        const record = {
          ...job,
          model: job.model_label || (state.currentModel && state.currentModel.name) || "โมเดล",
        };
        openLightbox(record, (job.images || []).indexOf(img));
      });
      box.appendChild(shot);
    });
  }

  async function cancelJob() {
    const id = state.runningJobId || state.jobId;
    if (!id) return;
    try {
      await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      toast("กำลังยกเลิก…", "warn");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  // ── history ─────────────────────────────────────────────────────────────
  async function refreshHistory(reset) {
    try {
      const data = await api("/api/history?offset=0&limit=60");
      state.total = data.total;
      state.history = data.items || [];
      renderGallery();
    } catch (err) {
      if (reset) toast(`โหลดประวัติไม่ได้: ${err.message}`, "err");
    }
  }

  async function loadMore() {
    try {
      const data = await api(`/api/history?offset=${state.history.length}&limit=60`);
      if (!data.items || !data.items.length) {
        $("#loadMore").classList.add("hidden");
        return;
      }
      state.history.push(...data.items);
      state.total = data.total;
      appendCards(data.items);
    } catch (err) {
      toast(err.message, "err");
    }
  }

  function renderGallery() {
    const gallery = $("#gallery");
    gallery.innerHTML = "";
    const items = state.history || [];
    $("#empty").classList.toggle("hidden", items.length > 0);
    if (!items.length) {
      $("#loadMore").classList.add("hidden");
      return;
    }
    const frag = document.createDocumentFragment();
    items.forEach((rec) => frag.appendChild(historyCard(rec)));
    gallery.appendChild(frag);
    $("#loadMore").classList.toggle("hidden", state.history.length >= state.total);
  }

  function appendCards(items) {
    const gallery = $("#gallery");
    const frag = document.createDocumentFragment();
    items.forEach((rec) => frag.appendChild(historyCard(rec)));
    gallery.appendChild(frag);
    $("#empty").classList.add("hidden");
    $("#loadMore").classList.toggle("hidden", state.history.length >= state.total);
  }

  function historyCard(rec) {
    const cover = (rec.images || [])[0] || {};
    const params = rec.params || {};
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.id = rec.id;
    card.innerHTML = `
      <div class="cover">
        <img loading="lazy" src="${esc(cover.thumb || cover.url || "")}" alt="">
        ${(rec.images || []).length > 1 ? `<span class="count">×${rec.images.length}</span>` : ""}
        <div class="overlay">
          <button type="button" class="icon-btn" data-act="reuse" title="ใช้การตั้งค่านี้">✦</button>
          <button type="button" class="icon-btn" data-act="download" title="ดาวน์โหลด">⇩</button>
          <button type="button" class="icon-btn" data-act="delete" title="ลบชุดนี้">🗑</button>
        </div>
      </div>
      <div class="info">
        <div class="p">${esc(params.prompt || "(ไม่มีพรอมป์ต)")}</div>
        <div class="row">
          <span class="pill">${esc(cover.width || "?")}×${esc(cover.height || "?")}</span>
          <span class="pill">seed ${esc(cover.seed ?? "?")}</span>
          <span class="pill">${esc(params.steps || "?")} steps</span>
        </div>
      </div>`;
    card.addEventListener("click", (event) => {
      const action = event.target.closest("[data-act]");
      if (action) {
        event.stopPropagation();
        handleCardAction(action.dataset.act, rec, 0);
        return;
      }
      openLightbox(rec, 0);
    });
    return card;
  }

  async function handleCardAction(action, rec, index) {
    if (action === "reuse") {
      reuseRecord(rec);
    } else if (action === "download") {
      downloadImage((rec.images || [])[index] || (rec.images || [])[0]);
    } else if (action === "delete") {
      await deleteRecord(rec);
    }
  }

  function reuseRecord(rec) {
    const params = rec.params || {};
    applyForm({ ...readForm(), ...params });
    if (params.model_id) {
      const entry = state.registry.find((m) => m.id === params.model_id);
      if (entry) {
        $("#modelSelect").value = entry.id;
        state.currentId = entry.id;
        state.currentModel = entry;
        updateModelMeta(entry);
      }
    }
    const wanted = Array.isArray(params.loras) ? params.loras : [];
    const missing = wanted.filter(
      (x) => !(state.loras || []).some((l) => l.id === x.id)
    );
    renderLoras(wanted);
    closeLightbox();
    toast("ใช้การตั้งค่าเดิมแล้ว — กดสร้างภาพได้เลย", "ok");
    if (missing.length) {
      toast(
        `LoRA ที่เคยใช้ไม่อยู่แล้ว: ${missing.map((x) => x.name || x.id).join(", ")}`,
        "warn",
        6000
      );
    }
    $("#panel .panel-inner").scrollTo({ top: 0, behavior: "smooth" });
    $("#prompt").focus();
  }

  async function deleteRecord(rec) {
    if (!rec || !state.history.some((r) => r.id === rec.id)) {
      toast("ภาพนี้ยังไม่ถูกบันทึกในประวัติ", "warn");
      return;
    }
    if (!(await uiConfirm(tr("ลบภาพชุดนี้และไฟล์ทั้งหมดออกจากเครื่อง?")))) return;
    try {
      await api(`/api/history/${encodeURIComponent(rec.id)}`, { method: "DELETE" });
      state.history = state.history.filter((r) => r.id !== rec.id);
      state.total = Math.max(0, state.total - 1);
      if (state.lb && state.lb.record && state.lb.record.id === rec.id) closeLightbox();
      renderGallery();
      toast("ลบแล้ว", "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  async function clearHistory() {
    if (
      !(await uiConfirm(
        tr(
          "ล้างประวัติทั้งหมดและลบไฟล์ภาพทุกไฟล์ในโฟลเดอร์ data/images ?\nการกระทำนี้ย้อนกลับไม่ได้"
        )
      ))
    )
      return;
    showBlocking("กำลังล้างประวัติ กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      await api("/api/history/clear", {
        method: "POST",
        body: JSON.stringify({ delete_files: true }),
      });
      state.history = [];
      state.total = 0;
      $("#live").classList.add("hidden");
      renderGallery();
      toast("ล้างประวัติแล้ว", "ok");
    } catch (err) {
      toast(err.message, "err");
    } finally {
      hideBlocking();
    }
  }

  function downloadImage(img) {
    if (!img || !img.url) return;
    const a = document.createElement("a");
    a.href = img.url;
    a.download = img.file || "image.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ── lightbox ────────────────────────────────────────────────────────────
  function openLightbox(record, index) {
    if (!record || !record.images || !record.images.length) return;
    state.lb = { record, index: clamp(index || 0, 0, record.images.length - 1) };
    renderLightbox();
    $("#lightbox").classList.remove("hidden");
    document.removeEventListener("keydown", lightboxKeys);
    document.addEventListener("keydown", lightboxKeys);
  }

  function renderLightbox() {
    if (!state.lb) return;
    const { record, index } = state.lb;
    const images = record.images || [];
    const img = images[index];
    if (!img) return closeLightbox();
    const params = record.params || {};
    const canDelete = state.history.some((r) => r.id === record.id);

    $("#lbImage").src = img.url;
    $("#lbImage").alt = params.prompt || "generated image";
    const scale = Number($("#lbUpscaleScale") && $("#lbUpscaleScale").value) === 2 ? 2 : 4;
    const srcW = img.width || params.width || 0;
    const srcH = img.height || params.height || 0;
    const canUpscale = !srcW || srcW <= 2048;
    $("#lbUpscale").disabled = !canUpscale;
    $("#lbUpscale").textContent = tr(
      scale === 2 ? "ขยาย 2× (ESRGAN)" : "ขยาย 4× (ESRGAN)"
    );
    $("#lbUpscale").title = canUpscale
      ? "ขยายด้วย Real-ESRGAN (คุณภาพสูงกว่า LANCZOS)"
      : "ภาพใหญ่เกินไปสำหรับขยาย 4× (สูงสุด 8192px)";
    $("#lbUpscaleSize").textContent = srcW
      ? `${srcW}×${srcH || srcW} → ${srcW * scale}×${(srcH || srcW) * scale}`
      : "";
    $("#lbCounter").textContent =
      images.length > 1 ? `${index + 1} / ${images.length}` : "";
    $("#lbPrev").disabled = images.length <= 1;
    $("#lbNext").disabled = images.length <= 1;
    $("#lbDate").textContent = fmtDate(record.created);
    $("#lbPrompt").textContent = params.prompt || "(ไม่มีข้อมูลพรอมป์ต)";
    $("#lbDelete").classList.toggle("hidden", !canDelete);

    const negative = params.negative_prompt || "";
    $("#lbNegBox").classList.toggle("hidden", !negative);
    $("#lbNeg").textContent = negative;

    const meta = [
      ["โมเดล", record.model || (state.currentModel && state.currentModel.name) || "—"],
      ["ขนาด", `${img.width || params.width || "?"}×${img.height || params.height || "?"}`],
      ["ขั้นตอน", params.steps ?? "?"],
      ["CFG", params.guidance ?? "?"],
      ["Sampler", params.sampler === "euler" ? "Euler" : "DPM-Solver"],
      ["Seed", img.seed ?? "?"],
      ["เพิ่มพรอมป์ต", params.enhance ? "ใช้ enhance" : "ไม่ใช้"],
      ["เวลา/ภาพ", img.elapsed != null ? `${img.elapsed} วิ` : "—"],
      ["สร้างเมื่อ", fmtDate(record.created)],
    ];
    const loras = (record.loras && record.loras.length ? record.loras : params.loras) || [];
    if (loras.length) {
      meta.splice(6, 0, [
        "LoRA",
        loras
          .map((x) => `${x.name || x.id || "?"} ×${x.weight == null ? 1 : x.weight}`)
          .join(", "),
      ]);
    }
    const regions = params.regions || [];
    if (regions.length) {
      meta.push([
        "Regional",
        regions
          .map((r) => `${r.label || "โซน"}: ${r.prompt}`)
          .join(" | "),
      ]);
    }
    $("#lbMeta").innerHTML = meta
      .map(
        ([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v == null ? "—" : v)}</dd>`
      )
      .join("");
  }

  function closeLightbox() {
    state.lb = null;
    $("#lightbox").classList.add("hidden");
    $("#lbImage").removeAttribute("src");
    document.removeEventListener("keydown", lightboxKeys);
  }

  function stepLightbox(delta) {
    if (!state.lb) return;
    const n = state.lb.record.images.length;
    if (n <= 1) return;
    state.lb.index = (state.lb.index + delta + n) % n;
    renderLightbox();
  }

  function lightboxKeys(event) {
    if (event.key === "Escape") closeLightbox();
    else if (event.key === "ArrowLeft") stepLightbox(-1);
    else if (event.key === "ArrowRight") stepLightbox(1);
  }

  // ── ESRGAN upscale (queued like a generation job) ───────────────────────
  async function upscaleCurrent() {
    if (!state.lb) return;
    const img = state.lb.record.images[state.lb.index];
    if (!img || !img.file) return;
    const btn = $("#lbUpscale");
    if (btn.disabled) return;
    const label = btn.textContent;
    const scale =
      Number($("#lbUpscaleScale") && $("#lbUpscaleScale").value) === 2 ? 2 : 4;
    const tileRaw = Number($("#lbUpscaleTile") && $("#lbUpscaleTile").value);
    const body = { file: img.file, scale };
    if (Number.isFinite(tileRaw) && tileRaw > 0) body.tile = tileRaw;
    btn.disabled = true;
    btn.textContent = "กำลังขยาย…";
    try {
      const res = await api("/api/upscale", {
        method: "POST",
        body: JSON.stringify(body),
      });
      toast(tr(`กำลังขยายภาพด้วย ESRGAN ${scale}×…`), "ok", 3000);
      const job = await waitForJob(res.job_id);
      if (job.status === "done") toast("ขยายภาพเสร็จแล้ว", "ok", 2500);
      else if (job.status === "canceled") toast("ยกเลิกการขยายแล้ว", "warn", 2500);
    } catch (err) {
      toast(err.message, "err", 7000);
    } finally {
      btn.textContent = label;
      // The job may have opened the new 4096² record: let renderLightbox()
      // decide the disabled state instead of always re-enabling.
      if (state.lb) renderLightbox();
      else btn.disabled = false;
    }
  }

  async function waitForJob(jobId) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const job = await api(`/api/jobs/${jobId}`);
      if (
        job.status === "done" ||
        job.status === "error" ||
        job.status === "canceled"
      ) {
        await refreshHistory(true);
        const rec = state.history.find((r) => r.id === jobId);
        if (rec) openLightbox(rec, 0);
        if (job.status === "error") {
          toast(job.error || "ขยายภาพไม่สำเร็จ", "err", 7000);
        }
        return job;
      }
    }
  }

  // ── panel tabs + assistant drawer ───────────────────────────────────────
  const PANEL_TAB_KEY = "imgs.panelTab:v1";
  const PANEL_TABS = ["create", "train", "settings"];

  function setPanelTab(name, options = {}) {
    const tab = PANEL_TABS.includes(name) ? name : "create";
    $$("#panelTabs .panel-tab").forEach((btn) => {
      const on = btn.dataset.pane === tab;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    });
    $$("#panel .pane").forEach((pane) => {
      pane.classList.toggle("active", pane.dataset.pane === tab);
    });
    const panel = $("#panel");
    if (panel) panel.classList.toggle("wide", tab === "train");
    if (options.persist !== false) {
      try {
        localStorage.setItem(PANEL_TAB_KEY, tab);
      } catch (e) {
        /* storage blocked — ไม่เป็นไร */
      }
    }
    const inner = $("#panel .panel-inner");
    if (inner) inner.scrollTo({ top: 0 });
    return tab;
  }

  function setAssistantOpen(open) {
    const drawer = $("#assistantDrawer");
    if (!drawer) return;
    drawer.classList.toggle("open", !!open);
    const btn = $("#assistantToggle");
    if (btn) {
      btn.classList.toggle("active", !!open);
      btn.setAttribute("aria-pressed", open ? "true" : "false");
    }
    if (open) {
      loadAssistantQuick();
      const sel = $("#assistantModelSelect");
      if (sel && !sel.options.length) loadAssistantModels();
      assistantScroll();
    }
  }

  function assistantIsOpen() {
    const drawer = $("#assistantDrawer");
    return !!drawer && drawer.classList.contains("open");
  }

  // ── assistant chat (P1: rule-based brain lives on the engine) ───────────
  // quick commands are sent as chat text, so they follow the UI language
  const ASSIST_FALLBACK_SUGGESTIONS = {
    th: [
      "ตรวจ dataset",
      "อ่านรูปใน dataset",
      "เติม tag มุม",
      "สถานะ GPU",
      "มี LoRA อะไรบ้าง",
      "ลบภาพล่าสุด",
    ],
    en: [
      "audit dataset",
      "read dataset images",
      "fill angle tags",
      "gpu status",
      "list loras",
      "delete latest image",
    ],
  };
  const ASSIST_TIER_PILL = {
    read: ["ok", "อ่าน"],
    write: ["warn", "แก้ข้อมูล"],
    destructive: ["err", "ลบ/เทรน"],
  };
  const ASSIST_TRAIN_TOOLS = ["fill_angle_tags", "set_trigger", "set_caption", "start_training"];
  const ASSIST_VISION_POLL_MS = 1200;
  const assistantState = { busy: 0, suggestions: [] };

  function setAssistantBrainPill(brain) {
    const pill = $("#assistantBrain");
    if (!pill || !brain) return;
    const mode = brain.mode || "auto";
    const ok = mode !== "rules" && brain.available;
    pill.textContent = brain.label || mode;
    pill.className = `pill ${ok ? "ok" : "warn"}`;
    pill.title = brain.error ? `${brain.label} — ${brain.error}` : brain.label || "";
  }

  function setAssistantModelNote(text) {
    const note = $("#assistantModelNote");
    if (note) note.textContent = text || "";
  }

  async function loadAssistantModels() {
    const sel = $("#assistantModelSelect");
    if (!sel) return;
    const refresh = $("#assistantModelRefresh");
    if (refresh) refresh.disabled = true;
    try {
      const data = await api("/api/assistant/models");
      const models = data.models || [];
      const current = data.current || "";
      const locked = !!(data.brain && data.brain.env_override);
      sel.innerHTML = "";
      const def = document.createElement("option");
      def.value = "";
      def.textContent = `ค่าเริ่มต้น (${data.default || "—"})`;
      sel.appendChild(def);
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        const label = [m.name];
        if (m.vision) label.push("vision");
        if (m.completion === false) label.push("ไม่รองรับแชท");
        opt.textContent = label.join(" · ");
        opt.disabled = m.completion === false;
        sel.appendChild(opt);
      });
      const installed = models.some((m) => m.name === current);
      sel.value = current && current !== data.default && installed ? current : "";
      sel.dataset.current = current;
      if (locked) {
        sel.disabled = true;
        setAssistantModelNote(`โมเดลถูกล็อกด้วย env IMGS_ASSIST_MODEL=${current}`);
      } else if (!models.length) {
        sel.disabled = true;
        setAssistantModelNote(
          "ไม่พบโมเดลใน Ollama — เปิด Ollama แล้วติดตั้งโมเดลก่อน (เช่น ollama pull qwen3.5:4b-q8_0)"
        );
      } else {
        sel.disabled = false;
        const active = models.find((m) => m.name === current);
        if (!active) {
          setAssistantModelNote(`ยังไม่ติดตั้งโมเดล ${current} — เลือกจากรายการหรือติดตั้งใน Ollama`);
        } else if (!active.vision) {
          setAssistantModelNote("โมเดลนี้ใช้แชทได้ แต่ไม่มี vision — ฟังก์ชันอ่านรูปจะใช้ไม่ได้");
        } else {
          setAssistantModelNote("");
        }
      }
      setAssistantBrainPill(data.brain);
    } catch (err) {
      sel.innerHTML = "";
      sel.disabled = true;
      setAssistantModelNote(`สแกนโมเดลไม่ได้: ${err.message}`);
    } finally {
      if (refresh) refresh.disabled = false;
    }
  }

  async function assistantChooseModel(name) {
    const sel = $("#assistantModelSelect");
    if (sel) sel.disabled = true;
    showBlocking("กำลังเปลี่ยนโมเดลผู้ช่วย กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น");
    try {
      const data = await api("/api/assistant/model", {
        method: "POST",
        body: JSON.stringify({ model: name }),
      });
      if (sel) {
        sel.dataset.current = data.model || "";
        sel.disabled = !!(data.brain && data.brain.env_override);
      }
      setAssistantBrainPill(data.brain);
      if (data.vision && data.vision.available) setAssistantModelNote("");
      else setAssistantModelNote("โมเดลนี้ไม่มี vision — ฟังก์ชันอ่านรูปจะใช้ไม่ได้");
      toast(`ใช้โมเดล ${data.model}`, "ok", 2500);
    } catch (err) {
      toast(`เปลี่ยนโมเดลไม่ได้: ${err.message}`, "err", 5000);
      await loadAssistantModels();
    } finally {
      hideBlocking();
    }
  }

  function assistantScroll() {
    const body = $("#assistantBody");
    if (body) body.scrollTop = body.scrollHeight;
  }

  function assistantBubble(role, text, kind) {
    const log = $("#assistantLog");
    if (!log) return null;
    const wrap = document.createElement("div");
    wrap.className = `assistant-msg ${role}${kind ? ` ${kind}` : ""}`;
    const bubble = document.createElement("div");
    bubble.className = "assistant-bubble";
    bubble.textContent = String(text == null ? "" : text);
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    assistantScroll();
    return wrap;
  }

  function assistantSetBusy(on) {
    assistantState.busy = Math.max(0, assistantState.busy + (on ? 1 : -1));
    const busy = assistantState.busy > 0;
    ["#assistantInput", "#assistantSend"].forEach((sel) => {
      const el = $(sel);
      if (el) el.disabled = busy;
    });
    $$("#assistantQuick .assistant-chip").forEach((chip) => {
      chip.disabled = busy;
    });
  }

  function assistantResultRows(tool, result) {
    const r = result || {};
    if (tool === "fill_angle_tags") {
      const rows = [["เติมแล้ว", `${r.filled || 0} รูป`]];
      if (r.remaining != null) rows.push(["ยังต้องเติมมือ", `${r.remaining} รูป`]);
      return rows;
    }
    if (tool === "set_trigger") {
      return [
        ["เดิม", r.old || "—"],
        ["ใหม่", r.new || "—"],
      ];
    }
    if (tool === "set_caption") return [["ไฟล์", r.file || "—"]];
    if (tool === "select_model") {
      return [
        ["เดิม", r.old || "—"],
        ["ใหม่", r.new || r.name || "—"],
      ];
    }
    if (tool === "generate_image") {
      return [
        ["job", r.job_id || "—"],
        ["พรอมป์ต", r.prompt || "—"],
      ];
    }
    if (tool === "delete_history") {
      return [["ลบแล้ว", `${r.removed || 0} ชุด · ${r.images || 0} ภาพ`]];
    }
    if (tool === "clear_history") return [["ลบแล้ว", `${r.removed || 0} ชุด`]];
    if (tool === "delete_lora") return [["LoRA", r.name || r.id || "—"]];
    if (tool === "start_training") {
      return [
        ["LoRA", r.lora_name || "—"],
        ["job", r.job_id || "—"],
        ["steps", String(r.steps || "—")],
      ];
    }
    return [];
  }

  function assistantActionCard(action) {
    const log = $("#assistantLog");
    if (!log || !action) return;
    const rows = assistantResultRows(action.tool, action.result);
    const wrap = document.createElement("div");
    wrap.className = "assistant-actions";
    const card = document.createElement("div");
    card.className = "assistant-result";
    const head = document.createElement("div");
    head.className = "assistant-result-head";
    const title = document.createElement("b");
    title.textContent = action.title || action.tool || "ผลลัพธ์";
    const [tierClass, tierLabel] = ASSIST_TIER_PILL[action.tier] || ["", action.tier || ""];
    const pill = document.createElement("span");
    pill.className = `pill ${tierClass}`.trim();
    pill.textContent = tierLabel;
    head.append(title, pill);
    card.appendChild(head);
    if (rows.length) {
      const dl = document.createElement("dl");
      dl.className = "assistant-kv";
      rows.forEach(([key, value]) => {
        const dt = document.createElement("dt");
        dt.textContent = key;
        const dd = document.createElement("dd");
        dd.textContent = value;
        dl.append(dt, dd);
      });
      card.appendChild(dl);
    }
    if (action.undo_id) {
      const undo = document.createElement("button");
      undo.type = "button";
      undo.className = "btn assistant-undo";
      undo.textContent = "↶ ย้อนกลับ";
      undo.addEventListener("click", () =>
        assistantUndo(action.undo_id, undo, action.tool)
      );
      card.appendChild(undo);
    }
    wrap.appendChild(card);
    log.appendChild(wrap);
    assistantScroll();
  }

  function assistantVisionProgress(card, job) {
    const total = Math.max(1, Number(job.total) || 0);
    const done = Number(job.done) || 0;
    const pct = Math.min(100, Math.round((done / total) * 100));
    const bar = card.querySelector(".assistant-progress-bar");
    if (bar) bar.style.width = `${pct}%`;
    const status = card.querySelector(".assistant-vision-status");
    if (!status) return;
    if (job.status === "running") {
      status.textContent = `กำลังอ่าน ${done}/${total} รูป${job.current ? ` · ${job.current}` : ""}`;
    } else if (job.status === "cancelling") {
      status.textContent = `กำลังหยุด… (${done}/${total})`;
    } else if (job.status === "cancelled") {
      status.textContent = `หยุดแล้ว — อ่านไป ${done}/${total} รูป`;
    } else {
      status.textContent = `อ่านครบ ${done}/${total} รูป`;
    }
  }

  function assistantVisionRows(card, job) {
    const box = card.querySelector(".assistant-vision-rows");
    if (!box) return;
    box.innerHTML = "";
    (job.results || []).forEach((row) => {
      const item = document.createElement("label");
      item.className = `assistant-vision-row${row.error ? " error" : ""}`;
      const check = document.createElement("input");
      check.type = "checkbox";
      check.dataset.file = row.file || "";
      check.checked = !!row.angle_added;
      check.disabled = !row.angle_added;
      const body = document.createElement("span");
      body.className = "assistant-vision-body";
      const top = document.createElement("span");
      top.className = "assistant-vision-top";
      const name = document.createElement("span");
      name.className = "assistant-vision-name";
      name.textContent = row.name || row.file || "—";
      top.appendChild(name);
      if (row.angle && row.angle !== "none") {
        const angle = document.createElement("span");
        angle.className = row.angle_added ? "pill ok" : "pill";
        angle.textContent = row.angle_added ? `+ ${row.angle}` : row.angle;
        top.appendChild(angle);
      }
      body.appendChild(top);
      if (row.multi_changed) {
        const warn = document.createElement("span");
        warn.className = "assistant-vision-warn";
        warn.textContent = row.multi
          ? "⚠ vision เห็นหลายตัวละคร — ตรวจก่อนแก้ flag"
          : "ℹ vision มองว่าเป็นรูปเดี่ยว — ตรวจก่อนแก้ flag";
        body.appendChild(warn);
      }
      if (row.error) {
        const err = document.createElement("span");
        err.className = "assistant-vision-warn";
        err.textContent = row.error;
        body.appendChild(err);
      } else if (row.caption) {
        const caption = document.createElement("span");
        caption.className = "assistant-vision-caption";
        caption.textContent = row.caption;
        caption.title = row.caption;
        body.appendChild(caption);
      }
      item.append(check, body);
      box.appendChild(item);
    });
    if (!(job.results || []).length) {
      const empty = document.createElement("div");
      empty.className = "assistant-note";
      empty.textContent = "ยังไม่มีผลอ่านรูป";
      box.appendChild(empty);
    }
  }

  function assistantVisionCard(action) {
    const log = $("#assistantLog");
    const jobId = action && action.result && action.result.job_id;
    if (!log || !jobId) return;
    const wrap = document.createElement("div");
    wrap.className = "assistant-actions";
    const card = document.createElement("div");
    card.className = "assistant-card vision";
    card.dataset.jobId = jobId;
    const head = document.createElement("div");
    head.className = "assistant-result-head";
    const title = document.createElement("b");
    title.textContent = action.title || "อ่านรูปด้วย vision";
    const pill = document.createElement("span");
    pill.className = "pill ok";
    pill.textContent = "อ่าน";
    head.append(title, pill);
    const status = document.createElement("div");
    status.className = "assistant-vision-status";
    status.textContent = `กำลังเริ่มอ่าน ${(action.result && action.result.total) || 0} รูป…`;
    const progress = document.createElement("div");
    progress.className = "assistant-progress";
    const bar = document.createElement("div");
    bar.className = "assistant-progress-bar";
    progress.appendChild(bar);
    const rows = document.createElement("div");
    rows.className = "assistant-vision-rows";
    const actions = document.createElement("div");
    actions.className = "assistant-card-actions";
    const apply = document.createElement("button");
    apply.type = "button";
    apply.className = "btn primary";
    apply.textContent = "ใช้ tag มุมที่เลือก";
    apply.disabled = true;
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn";
    cancel.textContent = "หยุดอ่าน";
    actions.append(apply, cancel);
    card.append(head, status, progress, rows, actions);
    wrap.appendChild(card);
    log.appendChild(wrap);
    assistantScroll();

    let timer = null;
    let stopped = false;

    const finish = (job) => {
      stopped = true;
      if (timer) clearTimeout(timer);
      assistantVisionProgress(card, job);
      assistantVisionRows(card, job);
      apply.disabled = !(job.results || []).some((row) => row.angle_added);
      cancel.remove();
      assistantScroll();
      if (job.status === "done" && job.done) toast(`อ่านรูปครบ ${job.done} รูป`, "ok", 2500);
    };

    const poll = async () => {
      if (stopped) return;
      try {
        const data = await api(`/api/assistant/vision/${jobId}`);
        const job = data.job || {};
        assistantVisionProgress(card, job);
        if (job.status === "running" || job.status === "cancelling") {
          timer = setTimeout(poll, ASSIST_VISION_POLL_MS);
        } else {
          finish(job);
        }
      } catch (err) {
        stopped = true;
        if (timer) clearTimeout(timer);
        status.textContent = `อ่านรูปไม่สำเร็จ: ${err.message}`;
        cancel.remove();
      }
    };

    cancel.addEventListener("click", async () => {
      cancel.disabled = true;
      try {
        await api(`/api/assistant/vision/${jobId}/cancel`, { method: "POST" });
        status.textContent = "กำลังหยุด…";
      } catch (err) {
        cancel.disabled = false;
        status.textContent = `หยุดไม่สำเร็จ: ${err.message}`;
      }
    });

    apply.addEventListener("click", async () => {
      const files = $$("input[type=checkbox]:checked", card)
        .map((el) => el.dataset.file)
        .filter(Boolean);
      if (!files.length) return;
      apply.disabled = true;
      try {
        const data = await api(`/api/assistant/vision/${jobId}/apply`, {
          method: "POST",
          body: JSON.stringify({ files }),
        });
        const note = document.createElement("div");
        note.className = "assistant-note";
        note.textContent = data.updated
          ? `ใช้ tag มุมแล้ว ${data.updated} รูป`
          : "ไม่มี tag ใหม่ให้ใช้";
        actions.replaceChildren(note);
        if (data.undo_id) {
          const undo = document.createElement("button");
          undo.type = "button";
          undo.className = "btn assistant-undo";
          undo.textContent = "↶ ย้อนกลับ";
          undo.addEventListener("click", () =>
            assistantUndo(data.undo_id, undo, "read_images")
          );
          actions.appendChild(undo);
        }
        refreshTrainer();
        assistantScroll();
        toast(`ใช้ tag แล้ว ${data.updated || 0} รูป`, "ok", 2500);
      } catch (err) {
        apply.disabled = false;
        assistantBubble("bot", `ใช้ tag ไม่สำเร็จ: ${err.message}`, "error");
      }
    });

    poll();
    return card;
  }

  function assistantPendingCard(pending) {
    const log = $("#assistantLog");
    if (!log || !pending) return;
    const card = document.createElement("div");
    card.className = "assistant-card pending";
    card.dataset.pendingId = pending.id;
    const title = document.createElement("b");
    title.textContent = pending.title || "ยืนยันคำสั่ง";
    const summary = document.createElement("div");
    summary.textContent = pending.summary || "";
    const actions = document.createElement("div");
    actions.className = "assistant-card-actions";
    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "btn danger";
    confirm.textContent = pending.confirm_label || "ยืนยัน";
    confirm.addEventListener("click", () => assistantConfirm(pending.id, card));
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn";
    cancel.textContent = "ยกเลิก";
    cancel.addEventListener("click", () => assistantCancel(pending.id, card));
    actions.append(confirm, cancel);
    card.append(title, summary, actions);
    log.appendChild(card);
    assistantScroll();
  }

  function assistantAfter(action) {
    if (!action) return;
    const tool = action.tool;
    if (ASSIST_TRAIN_TOOLS.includes(tool)) refreshTrainer();
    if (tool === "delete_history" || tool === "clear_history") refreshHistory();
    if (tool === "select_model") refreshModels();
    if (tool === "delete_lora") refreshLoras();
    if (tool === "generate_image") resumeActive();
  }

  async function assistantConfirm(id, card) {
    if (card) $$("button", card).forEach((btn) => (btn.disabled = true));
    assistantSetBusy(true);
    try {
      const data = await api("/api/assistant/confirm", {
        method: "POST",
        body: JSON.stringify({ id }),
      });
      if (card) card.remove();
      if (data.reply) assistantBubble("bot", data.reply);
      if (data.action) {
        if (data.action.tool === "read_images") assistantVisionCard(data.action);
        else assistantActionCard(data.action);
        assistantAfter(data.action);
      }
      toast("ยืนยันแล้ว", "ok", 2000);
    } catch (err) {
      if (card) $$("button", card).forEach((btn) => (btn.disabled = false));
      assistantBubble("bot", `ทำไม่ได้: ${err.message}`, "error");
    } finally {
      assistantSetBusy(false);
    }
  }

  async function assistantCancel(id, card) {
    if (card) $$("button", card).forEach((btn) => (btn.disabled = true));
    try {
      await api("/api/assistant/cancel", { method: "POST", body: JSON.stringify({ id }) });
      if (card) {
        card.classList.remove("pending");
        const actions = card.querySelector(".assistant-card-actions");
        if (actions) actions.remove();
        const note = document.createElement("div");
        note.className = "assistant-note";
        note.textContent = "ยกเลิกแล้ว — ไม่มีการเปลี่ยนแปลง";
        card.appendChild(note);
      }
    } catch (err) {
      if (card) $$("button", card).forEach((btn) => (btn.disabled = false));
      assistantBubble("bot", `ยกเลิกไม่สำเร็จ: ${err.message}`, "error");
    }
  }

  async function assistantUndo(undoId, btn, tool) {
    if (btn) btn.disabled = true;
    try {
      const data = await api("/api/assistant/undo", {
        method: "POST",
        body: JSON.stringify({ id: undoId }),
      });
      if (btn) {
        const note = document.createElement("div");
        note.className = "assistant-note";
        note.textContent = data.reply || "ย้อนกลับแล้ว";
        btn.replaceWith(note);
      }
      assistantAfter({ tool });
      toast("ย้อนกลับแล้ว", "ok", 2000);
    } catch (err) {
      if (btn) btn.disabled = false;
      assistantBubble("bot", `Undo ไม่สำเร็จ: ${err.message}`, "error");
    }
  }

  function renderAssistantQuick(items) {
    const box = $("#assistantQuick");
    if (!box) return;
    const list = items && items.length ? items : assistantState.suggestions;
    if (items && items.length) assistantState.suggestions = items.slice();
    box.innerHTML = "";
    list.forEach((label) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "assistant-chip";
      chip.textContent = label;
      chip.disabled = assistantState.busy > 0;
      chip.addEventListener("click", () => assistantSend(label));
      box.appendChild(chip);
    });
  }

  async function loadAssistantQuick() {
    const lang = window.I18N ? I18N.getLang() : "th";
    try {
      const data = await api(`/api/assistant/tools?lang=${lang}`);
      const suggestions = data.suggestions || [];
      renderAssistantQuick(
        suggestions.length
          ? suggestions
          : ASSIST_FALLBACK_SUGGESTIONS[lang] || ASSIST_FALLBACK_SUGGESTIONS.th
      );
      setAssistantBrainPill(data.brain);
    } catch (err) {
      renderAssistantQuick(
        ASSIST_FALLBACK_SUGGESTIONS[lang] || ASSIST_FALLBACK_SUGGESTIONS.th
      );
      setAssistantBrainPill({ mode: "rules", available: false, label: "กฎ (ออฟไลน์)" });
    }
  }

  async function assistantSend(text) {
    const input = $("#assistantInput");
    const message = String(text == null ? (input ? input.value : "") : text).trim();
    if (!message || assistantState.busy > 0) return;
    if (text == null && input) input.value = "";
    assistantBubble("user", message);
    assistantSetBusy(true);
    try {
      const dataset = activeDataset();
      const data = await api("/api/assistant/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          dataset_id: dataset ? dataset.id : null,
          lang: window.I18N ? I18N.getLang() : "th",
        }),
      });
      const brainPill = $("#assistantBrain");
      if (data.brain === "rules" && brainPill && brainPill.classList.contains("ok")) {
        setAssistantBrainPill({ mode: "rules", available: false, label: "กฎ (สำรอง)" });
      }
      if (data.reply) assistantBubble("bot", data.reply, data.ok === false ? "error" : "");
      (data.actions || []).forEach((action) => {
        if (action.tool === "read_images") assistantVisionCard(action);
        else if (action.tier !== "read" || action.undo_id) assistantActionCard(action);
        assistantAfter(action);
      });
      (data.pending || []).forEach(assistantPendingCard);
      if (Array.isArray(data.suggestions) && data.suggestions.length) {
        renderAssistantQuick(data.suggestions);
      }
    } catch (err) {
      assistantBubble("bot", `เชื่อมต่อผู้ช่วยไม่ได้: ${err.message}`, "error");
    } finally {
      assistantSetBusy(false);
    }
  }

  // ── wiring ──────────────────────────────────────────────────────────────
  function wire() {
    bindConfirmDialog();
    bindGuide();
    // Safety net for Chromium's occasional lost caret after a native window
    // dialog (file picker, message box): rebuild the caret on the focused
    // editable element when the window regains focus.
    window.addEventListener("focus", () => {
      const el = document.activeElement;
      if (!el) return;
      const tag = el.tagName;
      if (tag !== "TEXTAREA" && tag !== "INPUT") return;
      if (el.type === "checkbox" || el.type === "radio" || el.type === "file") return;
      let start = null;
      let end = null;
      try {
        start = el.selectionStart;
        end = el.selectionEnd;
      } catch (e) {
        /* number/email inputs do not expose a selection */
      }
      try {
        el.blur();
        el.focus({ preventScroll: true });
        if (typeof start === "number" && typeof end === "number") {
          el.setSelectionRange(start, end);
        }
      } catch (e) {
        /* element does not support selection */
      }
    });
    $$("#panelTabs .panel-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        setPanelTab(btn.dataset.pane);
        if (btn.dataset.pane === "settings") refreshPower();
      });
    });
    let savedTab = "create";
    try {
      savedTab = localStorage.getItem(PANEL_TAB_KEY) || "create";
    } catch (e) {
      /* ข้าม */
    }
    setPanelTab(savedTab, { persist: false });

    $$("#powerModes .power-mode").forEach((btn) => {
      btn.addEventListener("click", () => setPowerMode(btn.dataset.mode));
    });
    if ($("#powerSetup")) {
      $("#powerSetup").addEventListener("click", setupPower);
    }

    if ($("#backupExport")) {
      $("#backupExport").addEventListener("click", exportBackup);
    }
    if ($("#backupImport") && $("#backupFile")) {
      $("#backupImport").addEventListener("click", () => $("#backupFile").click());
      $("#backupFile").addEventListener("change", (e) => {
        const file = e.target.files && e.target.files[0];
        importBackup(file);
      });
    }

    if ($("#assistantToggle")) {
      $("#assistantToggle").addEventListener("click", () =>
        setAssistantOpen(!assistantIsOpen())
      );
    }
    if ($("#assistantClose")) {
      $("#assistantClose").addEventListener("click", () => setAssistantOpen(false));
    }
    if ($("#assistantSend")) {
      $("#assistantSend").addEventListener("click", () => assistantSend());
    }
    if ($("#assistantModelSelect")) {
      $("#assistantModelSelect").addEventListener("change", (event) => {
        assistantChooseModel(event.target.value);
      });
    }
    if ($("#assistantModelRefresh")) {
      $("#assistantModelRefresh").addEventListener("click", () => loadAssistantModels());
    }
    if ($("#assistantInput")) {
      $("#assistantInput").addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          assistantSend();
        }
      });
    }
    renderAssistantQuick(ASSIST_FALLBACK_SUGGESTIONS);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && assistantIsOpen()) setAssistantOpen(false);
    });

    $("#form").addEventListener("submit", onGenerate);
    $("#modelSelect").addEventListener("change", onModelChange);
    $("#loraList").addEventListener("change", (event) => {
      const item = event.target.closest(".lora-item");
      if (!item) return;
      if (event.target.classList.contains("lora-weight")) {
        const raw = Number(event.target.value);
        event.target.value = clamp(Number.isFinite(raw) ? raw : 1, 0, 2);
      }
      const checkbox = item.querySelector("input[type=checkbox]");
      const weight = item.querySelector(".lora-weight");
      weight.disabled = !checkbox.checked;
      scheduleSave();
    });
    $("#loraList").addEventListener("input", (event) => {
      if (event.target.classList.contains("lora-weight")) scheduleSave();
    });
    if ($("#loraShowAll")) {
      $("#loraShowAll").addEventListener("change", () => renderLoras(readLoras()));
    }
    if ($("#loraImportBtn")) {
      $("#loraImportBtn").addEventListener("click", async () => {
        const input = $("#loraImportFile");
        if (!input.files?.[0]) {
          toast("เลือกไฟล์ก่อนนำเข้า", "warn");
          input.focus();
          return;
        }
        const file = input.files[0];
        try {
          const res = await api("/api/loras/import", {
            method: "POST",
            body: JSON.stringify({ path: filePath(file) }),
          });
          await refreshLoras();
          input.value = "";
          toast("นำ LoRA เข้าแล้ว: " + res.lora.name, "ok");
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
    }
    if ($("#presetSave")) {
      $("#presetSave").addEventListener("click", saveCurrentPreset);
    }
    if ($("#presetName")) {
      // Enter in the preset name should save the preset — not submit the
      // generate form (the input lives inside #form).
      $("#presetName").addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.isComposing) return;
        event.preventDefault();
        saveCurrentPreset();
      });
    }
    if ($("#presetImportCollection")) {
      $("#presetImportCollection").addEventListener("click", importPresetsFromCollection);
    }
    if ($("#presetImportFile")) {
      $("#presetImportFile").addEventListener("change", async () => {
        const input = $("#presetImportFile");
        const file = input.files?.[0];
        if (!file) return;
        try {
          const text = await file.text();
          const res = await api("/api/presets/import", {
            method: "POST",
            body: JSON.stringify({
              text,
              name: file.name.replace(/\.txt$/i, ""),
            }),
          });
          await refreshPresets();
          importToast(res);
        } catch (err) {
          toast(err.message, "err", 6000);
        }
        input.value = "";
      });
    }
    if ($("#langSelect")) {
      $("#langSelect").value = window.I18N ? I18N.getLang() : "th";
      $("#langSelect").addEventListener("change", () => {
        const next = $("#langSelect").value;
        if (!window.I18N || next === I18N.getLang()) return;
        I18N.setLang(next);
        toast(next === "en" ? "Language: English" : "ภาษา: ไทย", "ok", 2200);
      });
      // quick commands are sent as chat text — refresh them with the language
      document.addEventListener("langchange", () => {
        if (assistantIsOpen()) loadAssistantQuick();
      });
    }
    if ($("#regionEnabled")) {
      $("#regionEnabled").addEventListener("change", () => {
        const on = $("#regionEnabled").checked;
        setRegionDisabled(!on || state.busy);
        $("#regionEnabled").disabled = state.busy || !regionCapable();
        scheduleSave();
      });
    }
    if ($("#regionLayout")) {
      $("#regionLayout").addEventListener("change", () => {
        renderRegions(
          $("#regionLayout").value,
          readRegionRows(),
          $("#regionEnabled").checked
        );
        scheduleSave();
      });
    }
    if ($("#regionList")) {
      $("#regionList").addEventListener("change", (event) => {
        if (event.target.classList.contains("region-weight")) {
          const raw = Number(event.target.value);
          event.target.value = clamp(Number.isFinite(raw) ? raw : 1, 0, 1.5);
        }
        scheduleSave();
      });
    }
    if ($("#inpaintBox") && inpaintCapable()) {
      const toggle = () => {
        const box = $("#inpaintBox");
        box.classList.toggle("hidden", !inpaintCapable());
        if (box.style.display !== "none" && !state.busy) {
          box.open = true;
        }
      };
      toggle();
      form.addEventListener("modelchanged", toggle);
      scheduleSave();
    }
    if ($("#queueList")) {
      $("#queueList").addEventListener("click", async (event) => {
        const btn = event.target.closest("button[data-act]");
        if (!btn || btn.disabled) return;
        const item = btn.closest(".queue-item");
        if (!item) return;
        const id = item.dataset.id;
        const act = btn.dataset.act;
        try {
          if (act === "cancel") {
            await api(`/api/jobs/${id}/cancel`, { method: "POST" });
            toast("ยกเลิกงานในคิวแล้ว", "warn", 2400);
          } else {
            await api(`/api/jobs/${id}/move?direction=${act}`, { method: "POST" });
          }
          await tickQueue();
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
    }
    if ($("#trainerDs")) {
      $("#trainerDs").addEventListener("change", () => {
        state.activeDs = $("#trainerDs").value || null;
        renderTrainer();
      });
      $("#dsCreate").addEventListener("click", async () => {
        const name = $("#dsName").value.trim();
        if (!name) {
          toast("ตั้งชื่อชุดเทรนก่อน", "warn");
          $("#dsName").focus();
          return;
        }
        try {
          const res = await api("/api/train/datasets", {
            method: "POST",
            body: JSON.stringify({
              name,
              trigger: $("#dsTrigger").value.trim(),
              kind: $("#dsKind") ? $("#dsKind").value : "character",
            }),
          });
          state.activeDs = res.dataset.id;
          $("#dsName").value = "";
          $("#dsTrigger").value = "";
          await refreshTrainer();
          toast("สร้าง dataset แล้ว — นำเข้ารูปได้เลย", "ok");
        } catch (err) {
          toast(err.message, "err", 6000);
        }
      });
      $("#dsUpload").addEventListener("change", async () => {
        const ds = activeDataset();
        const input = $("#dsUpload");
        if (!ds) {
          input.value = "";
          return;
        }
        if (!input.files?.length) return;
        const form = new FormData();
        for (const file of input.files) form.append("files", file, file.name);
        try {
          const res = await api(`/api/train/datasets/${ds.id}/images`, {
            method: "POST",
            body: form,
          });
          const added = (res.added || []).length;
          const failed = (res.errors || []).length;
          toast(
            failed ? `นำเข้า ${added} รูป · ข้าม ${failed} (ซ้ำ/ไม่รองรับ)` : `นำเข้า ${added} รูปแล้ว`,
            failed ? "warn" : "ok"
          );
          await refreshTrainer();
        } catch (err) {
          toast(err.message, "err", 6000);
        } finally {
          input.value = "";
        }
      });
      let lastCaptionInput = null;
      $("#captionList").addEventListener("focusin", (event) => {
        if (event.target.classList.contains("caption-input")) {
          lastCaptionInput = event.target;
        }
      });
      const CUSTOM_TAGS_KEY = "imgs.customTags";
      const readCustomTags = () => {
        try {
          const raw = JSON.parse(localStorage.getItem(CUSTOM_TAGS_KEY) || "[]");
          return Array.isArray(raw)
            ? raw.filter((t) => typeof t === "string" && t.trim()).slice(0, 12)
            : [];
        } catch {
          return [];
        }
      };
      const renderCustomTags = () => {
        const box = $("#customTagList");
        if (!box) return;
        box.innerHTML = readCustomTags()
          .map(
            (tag) =>
              `<button type="button" class="tag-btn tag-custom" data-tag="${esc(tag)}" title="คลิกเพื่อแทรก · Alt+คลิกเพื่อลบ">${esc(tag)}</button>`
          )
          .join("");
      };
      const rememberCustomTag = (tag) => {
        const clean = String(tag || "").trim().replace(/\s+/g, " ").slice(0, 40);
        if (!clean) return;
        const list = readCustomTags();
        if (!list.some((t) => t.toLowerCase() === clean.toLowerCase())) {
          list.unshift(clean);
          try {
            localStorage.setItem(CUSTOM_TAGS_KEY, JSON.stringify(list.slice(0, 12)));
          } catch {
            /* storage เต็ม/ปิด — ข้าม */
          }
        }
        renderCustomTags();
      };
      const insertTag = (tag) => {
        if (!lastCaptionInput || !document.body.contains(lastCaptionInput)) {
          toast("คลิกในช่อง caption ก่อน แล้วค่อยกด tag", "warn", 3000);
          return false;
        }
        const clean = String(tag || "").trim();
        if (!clean) return false;
        const current = lastCaptionInput.value.trim();
        if (current.toLowerCase().includes(clean.toLowerCase())) {
          toast("มี tag นี้ใน caption แล้ว", "warn", 1800);
          lastCaptionInput.focus();
          return false;
        }
        const base = current.replace(/[.,;:\s]+$/, "");
        lastCaptionInput.value = base ? `${base}, ${clean}` : clean;
        lastCaptionInput.focus();
        lastCaptionInput.dispatchEvent(new Event("input"));
        return true;
      };
      $("#captionTagbar").addEventListener("click", (event) => {
        const btn = event.target.closest(".tag-btn");
        if (!btn || !btn.dataset.tag) return;
        const tag = btn.dataset.tag;
        if (btn.classList.contains("tag-custom") && event.altKey) {
          try {
            localStorage.setItem(
              CUSTOM_TAGS_KEY,
              JSON.stringify(readCustomTags().filter((t) => t !== tag))
            );
          } catch {
            /* ข้าม */
          }
          renderCustomTags();
          toast("ลบ tag ออกจากรายการแล้ว", "ok", 1800);
          return;
        }
        insertTag(tag);
      });
      const customTagInput = $("#customTagInput");
      const addCustomTag = () => {
        if (!customTagInput) return;
        const value = customTagInput.value;
        if (!value.trim()) return;
        if (insertTag(value)) rememberCustomTag(value);
        customTagInput.value = "";
      };
      if ($("#customTagAdd")) {
        $("#customTagAdd").addEventListener("click", addCustomTag);
      }
      if (customTagInput) {
        customTagInput.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            addCustomTag();
          }
        });
      }
      renderCustomTags();
      $("#dsAudit").addEventListener("click", async (event) => {
        if (event.target.closest("#dsAutoTag")) {
          const ds = activeDataset();
          if (!ds) return;
          const plan = [];
          for (const item of collectCaptions()) {
            if (item.multi) continue;
            const add = missingAngleTags(item.caption);
            if (add.length) plan.push({ item, add });
          }
          if (!plan.length) {
            toast("ไม่มี caption ที่เติม tag มุมได้อัตโนมัติ", "warn", 3000);
            return;
          }
          const preview = plan
            .slice(0, 6)
            .map((p) => `• ${p.add.join(", ")} — ${p.item.file}`)
            .join("\n");
          const more = plan.length > 6 ? `\n…อีก ${plan.length - 6} รูป` : "";
          if (!(await uiConfirm(tr(`จะเติม tag มุมให้ ${plan.length} รูป:\n${preview}${more}\n\nยืนยัน?`), { danger: false }))) {
            return;
          }
          try {
            await api(`/api/train/datasets/${ds.id}/captions`, {
              method: "PUT",
              body: JSON.stringify({
                updates: plan.map((p) => ({
                  ...p.item,
                  caption: withAngleTags(p.item.caption, p.add),
                })),
              }),
            });
            await refreshTrainer();
            toast(`เติม tag มุมแล้ว ${plan.length} รูป`, "ok", 3500);
          } catch (err) {
            toast(err.message, "err", 6000);
          }
          return;
        }
        const btn = event.target.closest("#dsMarkMulti");
        if (!btn) return;
        const ds = activeDataset();
        if (!ds) return;
        if (!(await uiConfirm(tr("ทำเครื่องหมายรูปที่ caption บ่งบอกว่ามีคนที่สองเป็น “หลายตัวละคร”?"), { danger: false }))) return;
        const updates = collectCaptions().map((item) => ({
          ...item,
          multi: item.multi || looksMultiCaption(item.caption),
        }));
        try {
          await api(`/api/train/datasets/${ds.id}/captions`, {
            method: "PUT",
            body: JSON.stringify({ updates }),
          });
          await refreshTrainer();
          toast("ทำเครื่องหมายแล้ว — ตรวจสอบอีกครั้งก่อนเทรน", "ok", 3500);
        } catch (err) {
          toast(err.message, "err", 6000);
        }
      });
      const syncMultiWeight = () => {
        const mode = $("#trainMultiMode") ? $("#trainMultiMode").value : "exclude";
        if ($("#trainMultiWeight")) $("#trainMultiWeight").disabled = mode !== "weighted";
      };
      if ($("#trainMultiMode")) {
        $("#trainMultiMode").addEventListener("change", syncMultiWeight);
        syncMultiWeight();
      }
      $("#captionsSave").addEventListener("click", saveCaptions);
      $("#captionList").addEventListener("click", async (event) => {
        const btn = event.target.closest(".caption-del");
        if (!btn) return;
        const ds = activeDataset();
        const row = btn.closest(".caption-row");
        if (!ds || !row) return;
        try {
          await api(`/api/train/datasets/${ds.id}/images/${row.dataset.file}`, {
            method: "DELETE",
          });
          await refreshTrainer();
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
      $("#dsTriggerSave").addEventListener("click", async () => {
        const ds = activeDataset();
        if (!ds) return;
        try {
          await api(`/api/train/datasets/${ds.id}`, {
            method: "PATCH",
            body: JSON.stringify({
              trigger: $("#dsTriggerEdit").value.trim(),
              kind: $("#dsKindEdit") ? $("#dsKindEdit").value : ds.kind,
              extra_triggers: $("#dsExtraTriggers")
                ? $("#dsExtraTriggers").value
                : [],
            }),
          });
          await refreshTrainer();
          toast(
            "บันทึกประเภท/trigger แล้ว — caption ที่ไม่มี trigger จะถูกเติมอัตโนมัติตอนเทรน",
            "ok",
            4200
          );
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
      $("#dsDelete").addEventListener("click", async () => {
        const ds = activeDataset();
        if (!ds) return;
        if (!(await uiConfirm(tr(`ลบ dataset "${ds.name}" และรูปทั้งหมด?`)))) return;
        try {
          await api(`/api/train/datasets/${ds.id}`, { method: "DELETE" });
          state.activeDs = null;
          await refreshTrainer();
          toast("ลบ dataset แล้ว", "ok", 2200);
        } catch (err) {
          toast(err.message, "err", 5000);
        }
      });
      if ($("#trainPresetApply")) {
        $("#trainPresetApply").addEventListener("click", () => {
          const name = $("#trainPreset").value;
          if (!name) {
            toast("เลือก preset ก่อน", "warn");
            return;
          }
          applyTrainPreset(name);
        });
      }
      if ($("#trainCfgExport")) {
        $("#trainCfgExport").addEventListener("click", exportTrainConfig);
      }
      if ($("#trainCfgImport") && $("#trainCfgFile")) {
        $("#trainCfgImport").addEventListener("click", () =>
          $("#trainCfgFile").click()
        );
        $("#trainCfgFile").addEventListener("change", async () => {
          const input = $("#trainCfgFile");
          const file = input.files && input.files[0];
          input.value = "";
          if (file) await importTrainConfig(file);
        });
      }
      ["#calcEpochs", "#calcAccum"].forEach((selector) => {
        if ($(selector)) $(selector).addEventListener("input", updateCalcSteps);
      });
      if ($("#calcApply")) {
        $("#calcApply").addEventListener("click", () => {
          const steps = updateCalcSteps();
          if ($("#trainSteps")) $("#trainSteps").value = String(steps);
          toast(`ตั้ง steps = ${steps.toLocaleString()} แล้ว`, "ok");
        });
      }
      if ($("#captionList")) {
        $("#captionList").addEventListener("input", (event) => {
          if (event.target.classList.contains("caption-repeats")) updateCalcSteps();
        });
      }
      if ($("#dsKind")) {
        $("#dsKind").addEventListener("change", () => {
          const hint = $("#dsKindHint");
          if (hint) {
            hint.textContent = KIND_HINTS[$("#dsKind").value] || KIND_HINTS.character;
          }
        });
      }
      $("#trainStart").addEventListener("click", async () => {
        const ds = activeDataset();
        if (!ds) return;
        const name = $("#loraNameOut").value.trim();
        if (!name) {
          toast("ตั้งชื่อ LoRA ผลลัพธ์ก่อน", "warn");
          $("#loraNameOut").focus();
          return;
        }
        try {
          // Persist on-screen caption/tags/repeats first so the engine trains
          // on exactly what the user sees (repeats live in the dataset).
          await api(`/api/train/datasets/${ds.id}/captions`, {
            method: "PUT",
            body: JSON.stringify({ updates: collectCaptions() }),
          });
          const res = await api(`/api/train/datasets/${ds.id}/train`, {
            method: "POST",
            body: JSON.stringify({
              lora_name: name,
              model_id: (currentTrainerModel() || {}).id || "",
              init_lora: $("#trainInitLora") ? $("#trainInitLora").value : "",
              rank: Number($("#trainRank").value) || 32,
              steps: Number($("#trainSteps").value) || 1500,
              lr: Number($("#trainLr").value) || 0.0002,
              resolution: Number($("#trainRes").value) || 1024,
              enhance: $("#trainEnhance").checked,
              seed: Number($("#trainSeed").value) || -1,
              scheduler: $("#trainScheduler") ? $("#trainScheduler").value : "cosine",
              ema: $("#trainEma") ? $("#trainEma").checked : true,
              timestep_sampling: $("#trainSampling") ? $("#trainSampling").value : "logit_normal",
              save_every: $("#trainSaveEvery") ? Number($("#trainSaveEvery").value) || 0 : 0,
              augment_flip: $("#trainFlip") ? $("#trainFlip").checked : false,
              grad_accum: $("#trainGradAccum") ? Number($("#trainGradAccum").value) || 1 : 1,
              multi_mode: $("#trainMultiMode") ? $("#trainMultiMode").value : "exclude",
              multi_weight: ($("#trainMultiWeight") ? Number($("#trainMultiWeight").value) || 25 : 25) / 100,
              bucketing: $("#trainBucketing") ? $("#trainBucketing").checked : true,
              caption_dropout: $("#trainCaptionDropout") ? Number($("#trainCaptionDropout").value) || 0 : 0.05,
              caption_shuffle: $("#trainCaptionShuffle") ? $("#trainCaptionShuffle").checked : false,
              keep_tokens: $("#trainKeepTokens") ? Number($("#trainKeepTokens").value) || 0 : 1,
              noise_offset: $("#trainNoiseOffset") ? Number($("#trainNoiseOffset").value) || 0 : 0,
              min_snr_gamma: $("#trainMinSnr") ? Number($("#trainMinSnr").value) || 0 : 0,
              use_rslora: $("#trainRsLora") ? $("#trainRsLora").checked : false,
              target_modules: $("#trainTargetModules") ? $("#trainTargetModules").value : "attn",
              resume_state: $("#trainResumeState") ? $("#trainResumeState").checked : false,
              thermal_guard: $("#trainThermalGuard") ? $("#trainThermalGuard").checked : true,
              thermal_mode: $("#trainThermalMode") ? $("#trainThermalMode").value : "throttle",
              thermal_throttle_c: $("#trainThermalThrottle") ? Number($("#trainThermalThrottle").value) || 87 : 87,
              thermal_resume_c: $("#trainThermalResume") ? Number($("#trainThermalResume").value) || 80 : 80,
              thermal_max_c: $("#trainThermalMax") ? Number($("#trainThermalMax").value) || 90 : 90,
              thermal_step_pct: $("#trainThermalStep") ? Number($("#trainThermalStep").value) || 5 : 5,
            }),
          });
          state.knownActive.add(res.job_id);
          startQueuePolling();
          await tickQueue();
          toast("เทรนเข้าคิวแล้ว — ดูความคืบหน้าในแผงคิว", "ok");
        } catch (err) {
          toast(err.message, "err", 7000);
        }
      });
      $("#dsQcRun").addEventListener("click", async () => {
        const ds = activeDataset();
        if (!ds) return;
        const box = $("#dsQcResult");
        const note = $("#dsQcNote");
        if (box) {
          box.classList.remove("hidden");
          box.textContent = "กำลังตรวจ dataset…";
        }
        try {
          const res = await api(
            `/api/train/datasets/${ds.id}/qc?resolution=${
              Number($("#trainRes")?.value) || 1024
            }`
          );
          renderDatasetQc(res);
          if (note) note.textContent = "ตรวจล่าสุด: " + (res.dataset?.count || 0) + " รูป";
        } catch (err) {
          if (box) box.textContent = err.message;
          toast(err.message, "err", 7000);
        }
      });
      if ($("#sampleRun")) {
        $("#sampleRun").addEventListener("click", async () => {
          const ds = activeDataset();
          const loraId = $("#sampleLora") ? $("#sampleLora").value : "";
          if (!ds) {
            toast("เลือกชุดเทรนก่อน", "warn");
            return;
          }
          if (!loraId) {
            toast("เลือก LoRA ที่จะทดสอบก่อน", "warn");
            return;
          }
          try {
            const res = await api(`/api/train/datasets/${ds.id}/sample`, {
              method: "POST",
              body: JSON.stringify({
                lora_id: loraId,
                model_id: (currentTrainerModel() || {}).id || "",
                prompt: $("#samplePrompt") ? $("#samplePrompt").value : "",
                batch: 4,
                seed: 12345,
              }),
            });
            state.knownActive.add(res.job_id);
            startQueuePolling();
            await tickQueue();
            toast("สร้างภาพทดสอบเข้าคิวแล้ว", "ok");
          } catch (err) {
            toast(err.message, "err", 7000);
          }
        });
      }
      $("#trainerModel").addEventListener("change", () => {
        const m = currentTrainerModel();
        updateTrainerModelNote();
        renderInitLoras();
        if (!m) return;
        // Sensible per-family defaults (the user can still change every field).
        if (m.family === "sdxl") {
          $("#trainRank").value = "16";
          $("#trainSteps").value = "1000";
          $("#trainLr").value = "0.0001";
          $("#trainRes").value = "768";
        } else {
          $("#trainRank").value = "32";
          $("#trainSteps").value = "1500";
          $("#trainLr").value = "0.0002";
          $("#trainRes").value = "1024";
        }
      });
      $("#trainInitLora").addEventListener("change", () => {
        const lora = (state.loras || []).find(
          (l) => l.id === $("#trainInitLora").value
        );
        if (!lora || !lora.rank) return;
        const rankSel = $("#trainRank");
        if (
          !Array.from(rankSel.options).some(
            (o) => Number(o.value) === Number(lora.rank)
          )
        ) {
          const opt = document.createElement("option");
          opt.value = String(lora.rank);
          opt.textContent = String(lora.rank);
          rankSel.appendChild(opt);
        }
        rankSel.value = String(lora.rank);
      });
    }
    $("#cancel").addEventListener("click", cancelJob);
    $("#dice").addEventListener("click", () => {
      $("#seed").value = Math.floor(Math.random() * 2147483647);
      updateAdvancedSummary();
      scheduleSave();
    });
    $("#idea").addEventListener("click", () => {
      $("#prompt").value = IDEAS[Math.floor(Math.random() * IDEAS.length)];
      updatePromptCount();
      markChips();
      scheduleSave();
      toast("สุ่มไอเดียแล้ว", "info", 1800);
    });

    const form = $("#form");
    form.addEventListener("input", (event) => {
      const id = event.target.id;
      if (id === "prompt") updatePromptCount();
      if (id === "steps") $("#stepsVal").textContent = $("#steps").value;
      if (id === "guidance") $("#cfgVal").textContent = Number($("#guidance").value).toFixed(1);
      if (id === "width" || id === "height") updateSizeHint();
      updateAdvancedSummary();
      scheduleSave();
    });
    form.addEventListener("change", (event) => {
      const id = event.target.id;
      if (id === "width" || id === "height") {
        event.target.value = normalizeSize(event.target.value);
        updateSizeHint();
        markRatio();
      }
      if (id === "prompt") markChips();
      updateAdvancedSummary();
      scheduleSave();
    });
    document.addEventListener("keydown", (event) => {
      if (!(event.ctrlKey || event.metaKey) || event.key !== "Enter") return;
      if (!$("#lightbox").classList.contains("hidden")) return;
      // Ctrl+Enter inside the assistant drawer is not a generate shortcut.
      if (
        assistantIsOpen() &&
        event.target &&
        typeof event.target.closest === "function" &&
        event.target.closest("#assistantDrawer")
      )
        return;
      event.preventDefault();
      if (form.requestSubmit) form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true }));
    });

    $("#refresh").addEventListener("click", async () => {
      await refreshHistory(true);
      await refreshHealth();
      toast("โหลดข้อมูลใหม่แล้ว", "ok", 1600);
    });
    if ($("#modelRescan")) {
      $("#modelRescan").addEventListener("click", async () => {
        if (state.busy) {
          toast(tr("มีงานกำลังทำงานอยู่ — รอให้เสร็จก่อน"), "warn", 4000);
          return;
        }
        const before = (state.registry || []).length;
        await refreshModels(state.currentId || undefined);
        await refreshLoras();
        const after = (state.registry || []).length;
        toast(
          tr(`สแกนโมเดลแล้ว: พบ ${after} รายการ${after > before ? ` (+${after - before})` : ""}`),
          "ok",
          4000
        );
      });
    }
    $("#openFolder").addEventListener("click", async () => {
      try {
        await api("/api/open-folder?which=images", { method: "POST" });
      } catch (err) {
        toast(err.message, "err");
      }
    });
    $("#clearHistory").addEventListener("click", clearHistory);
    $("#loadMore").addEventListener("click", loadMore);

    $("#lightbox").addEventListener("click", (event) => {
      if (event.target.closest("[data-close]")) closeLightbox();
    });
    $("#lbPrev").addEventListener("click", () => stepLightbox(-1));
    $("#lbNext").addEventListener("click", () => stepLightbox(1));
    $("#lbUpscale").addEventListener("click", upscaleCurrent);
    $("#lbUpscaleScale").addEventListener("change", () => {
      if (state.lb) renderLightbox();
    });
    $("#lbDownload").addEventListener("click", () => {
      if (state.lb) downloadImage(state.lb.record.images[state.lb.index]);
    });
    $("#lbReuse").addEventListener("click", () => {
      if (state.lb) reuseRecord(state.lb.record);
    });
    $("#lbDelete").addEventListener("click", async () => {
      if (state.lb) await deleteRecord(state.lb.record);
    });
    $("#lbPrompt").addEventListener("click", async () => {
      if (!state.lb) return;
      const text = (state.lb.record.params || {}).prompt || "";
      try {
        await navigator.clipboard.writeText(text);
        toast("คัดลอกพรอมป์ตแล้ว", "ok", 1600);
      } catch (e) {
        toast("คัดลอกไม่สำเร็จ", "warn", 2000);
      }
    });
    $("#lbPrompt").title = "คลิกเพื่อคัดลอกพรอมป์ต";
    $("#lbCopyParams").addEventListener("click", async () => {
      if (!state.lb) return;
      const p = state.lb.record.params || {};
      const sampler = p.sampler === "euler" ? "Euler" : "DPM++ 2M";
      const bits = [
        `Steps: ${p.steps ?? 20}`,
        `Sampler: ${sampler}`,
        `CFG scale: ${p.guidance ?? 4.5}`,
        `Seed: ${p.seed ?? -1}`,
        `Size: ${p.width ?? "?"}x${p.height ?? "?"}`,
      ];
      const model = (state.currentModel && state.currentModel.name) || "";
      if (model) bits.push(`Model: ${model}`);
      const blob = [
        p.prompt || "",
        p.negative_prompt ? `Negative prompt: ${p.negative_prompt}` : "",
        bits.join(", "),
      ]
        .filter((l) => l.trim())
        .join("\n");
      try {
        await navigator.clipboard.writeText(blob);
        toast("คัดลอก parameters แล้ว", "ok", 2000);
      } catch (e) {
        toast("คัดลอกไม่สำเร็จ", "warn", 2000);
      }
    });

    window.addEventListener("beforeunload", () => {
      if (state.poll) clearInterval(state.poll);
    });
  }

  async function resumeActive() {
    try {
      const data = await api("/api/jobs?active=1&limit=24");
      const jobs = data.jobs || [];
      if (jobs.length) {
        jobs.forEach((j) => state.knownActive.add(j.id));
        state.activeCount = jobs.length;
        startQueuePolling();
        await tickQueue();
      }
    } catch (e) {
      /* engine may still be starting */
    }
  }

  async function init() {
    buildRatios();
    buildChips();
    buildExamples();
    const saved = loadSettings();
    applyForm(saved);
    wire();
    await refreshModels(saved.model_id);
    await refreshLoras(saved.loras);
    await refreshPresets();
    await refreshTrainer();
    await refreshHealth();
    await refreshPower();
    await refreshHistory(true);
    await resumeActive();
    state.healthTimer = setInterval(refreshHealth, 4000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
