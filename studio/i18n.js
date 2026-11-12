// ─────────────────────────────────────────────────────────────────────────────
// i18n.js — UI translation layer for PixelSmith Image Studio
//
// Thai is the source language baked into the markup and the JS strings.
// This layer renders the SAME DOM in English (and, later, other languages)
// without touching any app logic:
//
//   * `I18N.tr(text)`            → translate one string (JS side, confirm() etc.)
//   * `I18N.translateNode(text)` → translate DOM text/attribute values
//   * `I18N.apply(root)`         → translate an existing DOM subtree
//   * `I18N.setLang(l)` / `getLang()` / `onChange` event "langchange"
//
// How it works: a MutationObserver watches the whole document and swaps any
// Thai text/attribute that has an entry in the dictionary (exact match, or a
// regex pattern for messages with interpolated values).  Originals are kept in
// WeakMaps so switching back to Thai restores them.  When the language is the
// source language the observer exits immediately — zero overhead.
//
// Keys are the whitespace-normalized Thai source strings.  When adding UI
// text, copy the string EXACTLY (including “”, ·, —, °C …) as the key.
// ─────────────────────────────────────────────────────────────────────────────
(function () {
  "use strict";

  const LS_KEY = "studio_lang";
  const THAI = /[\u0E00-\u0E7F]/;

  let lang = null;
  try {
    const saved = localStorage.getItem(LS_KEY);
    if (saved === "th" || saved === "en") lang = saved;
  } catch (e) {
    /* storage blocked — detect from the OS locale instead */
  }
  if (!lang) {
    // first run: follow the system locale (Thai UI stays Thai, everyone else
    // gets English; the user can override it in ตั้งค่า/Settings)
    const loc = (navigator.language || "th").toLowerCase();
    lang = loc.startsWith("th") ? "th" : "en";
  }

  function norm(s) {
    return String(s).replace(/\s+/g, " ").trim();
  }

  // ── static dictionary: Thai source → English ──────────────────────────────
  const EN = {
    // tabs / brand / status
    "กำลังเชื่อมต่อ": "Connecting",
    "สร้างภาพ": "Generate",
    "เทรน LoRA": "Train LoRA",
    "ตั้งค่า": "Settings",
    "หมวดการใช้งาน": "usage sections",
    "สถานะโมเดล": "Model status",

    // create pane
    "โมเดล": "Model",
    "พรอมป์ต": "Prompt",
    "อธิบายภาพที่อยากได้… (ภาษาอังกฤษให้ผลดีที่สุด)":
      "Describe the image you want… (English works best)",
    "สุ่มไอเดีย": "Random idea",
    "SANA เข้าใจภาษาอังกฤษได้ดีที่สุด · ใช้ “เพิ่มรายละเอียด” ช่วยขยายความ":
      "SANA understands English best · use “Auto-enhance prompt” to expand it",
    "(สิ่งที่ไม่ต้องการ)": "(things to avoid)",
    "(แยกพรอมป์ตตามโซน · SDXL)": "(split prompt by zone · SDXL)",
    "เปิดใช้งาน regional": "Enable regional",
    "เลย์เอาต์": "Layout",
    "ซ้าย / ขวา": "Left / Right",
    "ซ้าย / กลาง / ขวา": "Left / Center / Right",
    "พรอมป์ตหลัก = ฉาก/พื้นหลัง · พรอมป์ตโซน = ตัวละคร/วัตถุในโซนนั้น":
      "Main prompt = scene/background · zone prompts = character/object in that zone",
    "เพิ่มรายละเอียดพรอมป์ตอัตโนมัติ": "Auto-enhance prompt",
    "(สไตล์ / ตัวละคร)": "(style / character)",
    "เลือก LoRA ที่เข้ากับโมเดลปัจจุบัน · น้ำหนัก 0.5–1.0 คือค่าที่ใช้บ่อย":
      "Pick LoRAs that fit the current model · 0.5–1.0 is the usual weight range",
    "(แก้ไขส่วนที่เลือก · SDXL)": "(edit selected area · SDXL)",
    "ภาพต้นฉบับ": "Source image",
    "มาส์ก (white = แก้ไข)": "Mask (white = edit)",
    "Strength (ความแรง)": "Strength",
    "ใช้กับ SDXL เท่านั้น — รูปสีขาวในมาส์กจะถูกแก้ไข":
      "SDXL only — white areas of the mask will be edited",
    "(upscale + เก็บรายละเอียด)": "(upscale + keep detail)",
    "เปิดใช้งาน hires fix": "Enable hires fix",
    "Scale (เท่า)": "Scale (×)",
    "Denoise (ความแรง)": "Denoise",
    "สร้างภาพตามขนาดที่ตั้ง แล้วขยาย + ลงลึกรายละเอียดอีกรอบ · SANA/SDXL/SD1.5 · 1.25–1.5× ปลอดภัยกับ VRAM 10 GB · ใช้ร่วมกับ inpaint ไม่ได้":
      "Generates at the set size, then upscales + refines detail · SANA/SDXL/SD1.5 · 1.25–1.5× is safe on 10 GB VRAM · cannot be combined with inpaint",
    "ตั้งค่าขั้นสูง": "Advanced",
    "ขนาดภาพ": "Image size",
    "กว้าง": "Width",
    "สูง": "Height",
    "จำนวนภาพ": "Batch",
    "1 ภาพ": "1 image",
    "2 ภาพ": "2 images",
    "3 ภาพ": "3 images",
    "4 ภาพ": "4 images",
    "-1 = สุ่ม": "-1 = random",
    "สุ่ม seed": "Random seed",
    "🎲 สุ่ม": "🎲 Random",
    "กำลังเตรียม…": "Preparing…",
    "กำลังโหลด…": "Loading…",
    "ใช้ enhance": "On",
    "ไม่ใช้": "Off",
    "ยกเลิกการสร้าง": "Cancel generation",

    // train pane
    "(เทรนเอง · SANA / SDXL)": "(train your own · SANA / SDXL)",
    "โมเดลฐานสำหรับเทรน": "Base model for training",
    "— ใช้โมเดลที่เลือกอยู่ —": "— Use selected model —",
    "ชุดเทรนที่มีอยู่": "Existing datasets",
    "— เลือกชุดเทรน —": "— Select a dataset —",
    "ชื่อชุดเทรน": "Dataset name",
    "ชื่อ dataset": "dataset name",
    "เช่น foxy": "e.g. foxy",
    "สร้าง dataset": "Create dataset",
    "Trigger word (แก้ได้)": "Trigger word (editable)",
    "เช่น mytrigger7": "e.g. mytrigger7",
    "บันทึก trigger": "Save trigger",
    "นำเข้ารูป (เลือกได้หลายใบ)": "Import images (multiple allowed)",
    "ลบ dataset นี้": "Delete this dataset",
    "แทรก tag มุม/ท่า:": "Insert angle/pose tag:",
    "tag อื่น ๆ…": "other tags…",
    "เพิ่ม tag ที่พิมพ์ (Enter)": "Add typed tag (Enter)",
    "บันทึก caption & ประเภท": "Save captions & types",
    "ชื่อ LoRA ผลลัพธ์": "Output LoRA name",
    "เทรนต่อจาก LoRA เดิม": "Continue from LoRA",
    "— เริ่มจากศูนย์ —": "— Start from scratch —",
    "ใช้ prompt enhance ตอนเทรน": "Use prompt enhance while training",
    "(ตรงกับ generate ปกติ)": "(matches normal generation)",
    "ตัวเลือกขั้นสูง": "Advanced options",
    "(คุณภาพต่อ step)": "(per-step quality)",
    "Shift (แบบเดิม)": "Shift (legacy)",
    "Checkpoint ทุก (steps)": "Checkpoint every (steps)",
    "รูปหลายตัวละคร": "Multi-character images",
    "ไม่ใช้ (เฉพาะเดี่ยว)": "Exclude (solo only)",
    "ใช้แบบลดน้ำหนัก": "Use with lower weight",
    "ใช้เต็ม": "Include fully",
    "น้ำหนักรูปหลายตัวละคร (%)": "Multi-character weight (%)",
    "โหมดระบายความร้อน": "Thermal mode",
    "ชะลอ GPU (แนะนำ)": "Throttle GPU (recommended)",
    "พัก GPU": "Pause GPU",
    "เริ่มชะลอเมื่อ (°C)": "Throttle at (°C)",
    "กลับมาเต็ม speed เมื่อ (°C)": "Full speed again at (°C)",
    "หยุดฉุกเฉินที่ (°C)": "Emergency stop at (°C)",
    "ลดความเร็วต่อระดับ (%)": "Slowdown per level (%)",
    "(เฉลี่ยน้ำหนักให้เนียนขึ้น — ทำให้ LoRA อ่อนลง/‖BA‖ ต่ำ ไม่แนะนำสำหรับ run สั้น)":
      "(averages weights — weakens the LoRA / low ‖BA‖; not recommended for short runs)",
    "สุ่มพลิกภาพซ้าย-ขวา": "Random left-right flip",
    "(ชะลอ GPU อัตโนมัติเมื่อร้อนเกิน — หยุดเฉพาะกรณีฉุกเฉิน)":
      "(auto-throttle when hot — pauses only in emergencies)",
    "Cosine + EMA + logit-normal ให้ผลดีขึ้นในจำนวน step ที่น้อยลง · checkpoint ช่วยเลือก step ที่ดีที่สุด · grad accumulation >1 จะช้าลงตามจำนวนเท่า · โหมดชะลอจะลด clock ลงทีละ 5% สูงสุด 25% เมื่อถึง 87°C และกลับมาเร็วขึ้นเมื่อเย็นถึง 80°C (ถ้าไม่มีสิทธิ์ sudo จะหน่วงระหว่าง step แทน)":
      "Cosine + EMA + logit-normal give better results in fewer steps · checkpoints help pick the best step · grad accumulation >1 slows each step proportionally · throttle mode lowers the clock 5% per level (max 25%) at 87°C and speeds back up at 80°C (without sudo it sleeps between steps instead)",
    "🚀 เริ่มเทรน (เข้าคิว)": "🚀 Start training (queue)",
    "เทรน = ปล่อยโมเดล generate ก่อนเสมอ — งานอื่นในคิวจะรอจนเทรนจบ · caption ว่าง = ใช้ trigger word · ผลลัพธ์โผล่ในหน้า LoRA ทันที":
      "Training always frees the generation model first — other queued jobs wait until it finishes · empty caption = use the trigger word · the result appears in the LoRA list right away",
    "💡 ค่าแนะนำ character LoRA: rank 32 · steps 3000–4000 · LR 0.00015 · 1024px · cosine + EMA · ใช้ trigger ที่ไม่ซ้ำ (เช่น mytrigger7) และให้ caption บรรยายฉาก/ท่า/ชุดที่ต่างกัน โดยไม่บรรยายลักษณะตัวละครที่เหมือนกันทุกภาพ":
      "💡 Character LoRA recipe: rank 32 · 3000–4000 steps · LR 0.00015 · 1024px · cosine + EMA · use a unique trigger (e.g. mytrigger7) and let captions describe varied scenes/poses/outfits — never describe character traits shared by every image",

    // settings pane
    "(จัดการ LoRA)": "(manage LoRAs)",
    "นำเข้า LoRA": "Import LoRA",
    "นำเข้า": "Import",
    "นำ LoRA จากโฟลเดอร์อื่นเข้ามาใช้งาน": "Bring LoRAs from other folders into the app",
    "ที่เก็บไฟล์": "File locations",
    "โมเดล: models/ · LoRA: models/loras/ · ภาพ: data/images/ · dataset: data/train/":
      "models: models/ · LoRA: models/loras/ · images: data/images/ · datasets: data/train/",
    "ผู้ช่วยทำงานในเครื่อง 100% — อ่าน/แก้ข้อมูลให้อัตโนมัติ (เติม tag, จัดการ caption, สั่งเจนภาพ) และอ่านรูปได้ด้วยโมเดล vision ในเครื่อง":
      "The assistant runs 100% locally — reads/edits data automatically (tag filling, captions, queued generation) and can read images with a local vision model",
    "ภาษา / Language": "Language",
    "ทำงานในเครื่อง 100%": "Runs 100% locally",
    "ผู้ช่วย AI": "AI assistant",

    // GPU power modes
    "โหมดพลังงาน GPU": "GPU power mode",
    "ไม่พบการ์ด NVIDIA": "No NVIDIA GPU found",
    "โหมดพลังงานใช้ได้เฉพาะเครื่องที่มีการ์ด NVIDIA":
      "Power modes require an NVIDIA GPU",
    "ใช้ได้เฉพาะ Linux": "Linux only",
    "โหมดพลังงานใช้ได้เฉพาะ Linux ที่มีการ์ด NVIDIA":
      "Power modes are available on Linux with an NVIDIA GPU",
    "โหมดพลังงานใช้ได้เฉพาะ Linux": "Power modes are available on Linux only",
    "เปิดใช้งานครั้งเดียว (ต้องใส่รหัสผ่านระบบ) — หลังจากนั้นสลับโหมดได้ทันทีโดยไม่ต้องใส่รหัส":
      "One-time setup (system password required) — after that you can switch modes instantly",
    "ใช้โหมดที่เลือกตลอดเวลาแอปเปิด — เมื่อปิดแอปจะกลับไปใช้ค่าเริ่มต้นของการ์ด":
      "The selected mode stays active while the app is open — closing the app restores the card's default limit",
    "เปิดใช้งานโหมดพลังงาน": "Enable power control",
    "กำลังเปิดใช้งาน…": "Enabling…",
    "เปิดใช้งานโหมดพลังงานแล้ว": "Power control enabled",
    "ยังไม่ได้เปิดใช้งานโหมดพลังงาน — กดปุ่มเปิดใช้งานก่อน":
      "Power control is not enabled yet — click Enable power control first",
    "sudoers ยังไม่ได้ตั้งค่า — กดปุ่มเปิดใช้งานโหมดพลังงานอีกครั้ง":
      "sudoers is not configured — click Enable power control again",
    "ไม่พบ pkexec — ติดตั้ง polkit หรือรันสคริปต์ติดตั้งด้วย sudo":
      "pkexec not found — install polkit or run the setup script with sudo",
    "การยืนยันถูกยกเลิกหรือไม่ได้รับอนุญาต": "Authorization was cancelled or denied",
    "ติดตั้งไม่สำเร็จ": "Setup failed",
    "ต้องระบุ watts หรือ mode": "Provide either watts or mode",
    "ไม่พบไฟล์ติดตั้ง (host/pixelsmith-power-setup)":
      "Setup file not found (host/pixelsmith-power-setup)",
    "ไม่สามารถระบุผู้ใช้เดสก์ท็อปสำหรับตั้งค่า sudoers":
      "Could not determine the desktop user for the sudoers rule",
    "ติดตั้งไม่เสร็จภายในเวลา — ลองใหม่แล้วป้อนรหัสให้เร็วขึ้น":
      "Setup timed out — try again and enter the password promptly",

    // stage
    "ผลลัพธ์": "Results",
    "ภาพที่สร้างจะถูกเก็บไว้ในเครื่องนี้": "Generated images are stored on this machine",
    "✦ ผู้ช่วย": "✦ Assistant",
    "เปิด/ปิดผู้ช่วย AI": "Toggle AI assistant",
    "โหลดประวัติใหม่": "Reload history",
    "📁 โฟลเดอร์": "📁 Folder",
    "เปิดโฟลเดอร์ภาพ": "Open images folder",
    "ล้างประวัติ": "Clear history",
    "ล้างประวัติและลบไฟล์ภาพ": "Clear history and delete image files",
    "กำลังสร้างภาพ…": "Generating…",
    "คิวงาน": "Job queue",
    "ยังไม่มีภาพ": "No images yet",
    "เขียนพรอมป์ตแล้วกด": "Write a prompt and press",
    "— ผลลัพธ์จะถูกเก็บไว้ในเครื่องและแสดงที่นี่":
      "— results are stored locally and shown here",
    "โหลดเพิ่ม": "Load more",

    // lightbox
    "ปิด": "Close",
    "ก่อนหน้า": "Previous",
    "ถัดไป": "Next",
    "รายละเอียดภาพ": "Image details",
    "✦ ใช้การตั้งค่านี้": "✦ Reuse settings",
    "ขยาย 4× (ESRGAN)": "Upscale 4× (ESRGAN)",
    "ขยายด้วย Real-ESRGAN (คุณภาพสูงกว่า LANCZOS)":
      "Upscale with Real-ESRGAN (better than LANCZOS)",
    "ภาพใหญ่เกินไปสำหรับขยาย 4× (สูงสุด 8192px)":
      "Image too large for 4× upscale (max 8192px)",
    "กำลังขยาย…": "Upscaling…",
    "กำลังขยายภาพด้วย ESRGAN…": "Upscaling with ESRGAN…",
    "ขยายภาพเสร็จแล้ว": "Upscale complete",
    "ขยายภาพไม่สำเร็จ": "Upscale failed",
    "ยกเลิกการขยายแล้ว": "Upscale canceled",
    "กำลังดาวน์โหลดโมเดล upscale (~64 MB)…":
      "Downloading the upscale model (~64 MB)…",
    "กำลังโหลดโมเดล upscale…": "Loading the upscale model…",
    "ไม่พบภาพต้นฉบับ": "Source image not found",
    "ดาวน์โหลด": "Download",
    "ลบ": "Delete",
    "(ไม่มีพรอมป์ต)": "(no prompt)",
    "(ไม่มีข้อมูลพรอมป์ต)": "(no prompt data)",
    "ขนาด": "Size",
    "ขั้นตอน": "Steps",
    "เพิ่มพรอมป์ต": "Enhance",
    "เวลา/ภาพ": "Time/image",
    "สร้างเมื่อ": "Created",
    "คลิกเพื่อคัดลอกพรอมป์ต": "Click to copy prompt",

    // assistant drawer
    "สมองผู้ช่วย": "Assistant brain",
    "โมเดลผู้ช่วย": "Assistant model",
    "เลือกโมเดล Ollama ที่ติดตั้งในเครื่อง": "Pick an Ollama model installed on this machine",
    "สแกนโมเดลใหม่": "Rescan models",
    "สวัสดีครับ — ผู้ช่วยทำงานในเครื่อง 100% (ไม่ส่งข้อมูลออก) พิมพ์คำสั่งหรือเลือกจากปุ่มด้านล่างได้เลย":
      "Hello! The assistant runs 100% locally (nothing leaves the machine). Type a command or pick a button below.",
    "สิทธิ์:": "Permissions:",
    "ทำเลย": "run",
    "ทำเลย ·": "run ·",
    "ทำเลย + Undo ·": "run + Undo ·",
    "ทำเลย + Undo": "run + Undo",
    "ต้องยืนยัน": "needs confirmation",
    "พิมพ์คำสั่ง…": "Type a command…",
    "ส่ง": "Send",
    "อ่าน": "Read",
    "แก้ข้อมูล": "Data edits",
    "ลบ/เทรน": "Delete/Train",
    "ยืนยันคำสั่ง": "Confirm command",
    "ยืนยัน": "Confirm",
    "ยกเลิก": "Cancel",
    "ยืนยันแล้ว": "Confirmed",
    "ยกเลิกแล้ว — ไม่มีการเปลี่ยนแปลง": "Canceled — nothing changed",
    "ย้อนกลับแล้ว": "Undone",
    "↶ ย้อนกลับ": "↶ Undo",
    "กฎ (ออฟไลน์)": "Rules (offline)",
    "กฎ (สำรอง)": "Rules (fallback)",
    "ตรวจ dataset": "Audit dataset",
    "อ่านรูปใน dataset": "Read dataset images",
    "เติม tag มุม": "Fill angle tags",
    "สถานะ GPU": "GPU status",
    "มี LoRA อะไรบ้าง": "List LoRAs",
    "ลบภาพล่าสุด": "Delete latest image",
    "ยังไม่มีผลอ่านรูป": "No vision results yet",
    "อ่านรูปด้วย vision": "Read images with vision",
    "กำลังเริ่มอ่าน": "Starting to read",
    "ใช้ tag มุมที่เลือก": "Apply selected angle tags",
    "หยุดอ่าน": "Stop reading",
    "⚠ vision เห็นหลายตัวละคร — ตรวจก่อนแก้ flag":
      "⚠ vision sees multiple characters — check before changing the flag",
    "ℹ vision มองว่าเป็นรูปเดี่ยว — ตรวจก่อนแก้ flag":
      "ℹ vision sees a solo image — check before changing the flag",
    "ไม่มี tag ใหม่ให้ใช้": "No new tags to apply",
    "เติมแล้ว": "Filled",
    "ยังต้องเติมมือ": "Still needs manual tags",
    "เดิม": "Old",
    "ใหม่": "New",
    "ไฟล์": "File",
    "ค่าเริ่มต้น": "Default",
    "ไม่รองรับแชท": "no chat",
    "ไม่พบโมเดลใน Ollama — เปิด Ollama แล้วติดตั้งโมเดลก่อน (เช่น ollama pull qwen3.5:4b-q8_0)":
      "No models found in Ollama — start Ollama and install one first (e.g. ollama pull qwen3.5:4b-q8_0)",
    "โมเดลนี้ใช้แชทได้ แต่ไม่มี vision — ฟังก์ชันอ่านรูปจะใช้ไม่ได้":
      "This model can chat but has no vision — image reading won't work",
    "โมเดลนี้ไม่มี vision — ฟังก์ชันอ่านรูปจะใช้ไม่ได้":
      "This model has no vision — image reading won't work",

    // dataset / trainer (app.js)
    "ทั้งหมด": "All",
    "เดี่ยว": "Solo",
    "หลายตัวละคร": "Multi-char",
    "tag มุม (เดี่ยว)": "Angle tags (solo)",
    "สไตล์ภาพ": "Image style",
    "ตัวละคร/วัตถุ": "Character/object",
    "caption — trigger word ถูกเติมให้อัตโนมัติ":
      "caption — trigger word is added automatically",
    "รูปนี้มีตัวละครอื่นอยู่ในภาพหรือไม่":
      "Does this image contain another character?",
    "ลบรูปนี้": "Delete image",
    "บันทึก caption แล้ว": "Captions saved",
    "ยังไม่มี LoRA — นำเข้าไฟล์ .safetensors ด้านบน":
      "No LoRAs yet — import .safetensors files above",
    "ลบ LoRA": "Delete LoRA",
    "น้ำหนัก LoRA": "LoRA weight",
    "น้ำหนักโซน": "Zone weight",
    "ตัวละคร/วัตถุในโซนนี้…": "character/object in this zone…",
    "ยังไม่มีโมเดลที่เทรนได้ (SANA / SDXL)": "No trainable models (SANA / SDXL)",
    "SDXL ช้ากว่า SANA มากบนการ์ดนี้ — แนะนำ 768px · rank 16 · LR 1e-4 · 500–1000 steps":
      "SDXL is much slower than SANA on this GPU — suggested: 768px · rank 16 · LR 1e-4 · 500–1000 steps",
    "SANA เร็วที่สุด — แนะนำ 1024px · rank 32–64 · LR 2e-4 · 1500+ steps":
      "SANA is the fastest — suggested: 1024px · rank 32–64 · LR 2e-4 · 1500+ steps",
    "ค่าเริ่มต้นเทรนเฉพาะรูป “เดี่ยว” — รูปหลายตัวละครจะถูกข้ามเพื่อไม่ให้ trigger จำตัวละครประกอบ · ถ้าเลือกโหมดลดน้ำหนัก ควรใส่ caption เช่น “with another character” ให้รูปเหล่านั้นด้วย":
      "Training uses “solo” images only — multi-character images are skipped so the trigger never learns a composite character · with the low-weight mode, caption those images e.g. “with another character”",
    "พร้อม": "Ready",
    "กำลังโหลด": "Loading",
    "กำลังดาวน์โหลด": "Downloading",
    "ยังไม่โหลด": "Not loaded",
    "ผิดพลาด": "Error",
    "โมเดลพร้อมใช้งาน": "Model ready",
    "กำลังโหลดโมเดล…": "Loading model…",
    "กำลังดาวน์โหลดโมเดล…": "Downloading model…",
    "จะโหลดอัตโนมัติเมื่อกดสร้างภาพ": "Loads automatically when you generate",
    "engine ผิดพลาด": "Engine error",
    "โมเดลพร้อม": "Model ready",
    "ยังไม่โหลดโมเดล": "Model not loaded",
    "เชื่อมต่อ engine ไม่ได้ — ลองรีโหลด": "Can't reach the engine — try reloading",
    "ใส่พรอมป์ตก่อนสร้างภาพครับ": "Enter a prompt first",
    "สร้างภาพเสร็จแล้ว": "Image generated",
    "ยกเลิกการเทรนแล้ว": "Training canceled",
    "ยกเลิกแล้ว": "Canceled",
    "กำลังยกเลิก…": "Canceling…",
    "ใช้การตั้งค่านี้": "Reuse settings",
    "ลบชุดนี้": "Delete this set",
    "ใช้การตั้งค่าเดิมแล้ว — กดสร้างภาพได้เลย": "Settings reused — ready to generate",
    "ภาพนี้ยังไม่ถูกบันทึกในประวัติ": "This image isn't in history",
    "ลบภาพชุดนี้และไฟล์ทั้งหมดออกจากเครื่อง?":
      "Delete this image set and all its files from disk?",
    "ล้างประวัติทั้งหมดและลบไฟล์ภาพทุกไฟล์ในโฟลเดอร์ data/images ?\nการกระทำนี้ย้อนกลับไม่ได้":
      "Clear ALL history and delete every image file in data/images?\nThis cannot be undone.",
    "ล้างประวัติแล้ว": "History cleared",
    "เลื่อนขึ้น": "Move up",
    "ยกเลิกงานนี้": "Cancel this job",
    "เทรนอยู่ในคิว…": "Training queued…",
    "อยู่ในคิว…": "Queued…",
    "เทรนเสร็จแล้ว": "Training done",
    "ผลลัพธ์ล่าสุด": "Latest result",
    "เทรนไม่สำเร็จ": "Training failed",
    "สร้างไม่สำเร็จ": "Generation failed",
    "เกิดข้อผิดพลาด": "Something went wrong",
    "กำลังเตรียมโมเดล…": "Preparing model…",
    "พัก GPU ให้เย็นลง…": "Cooling down GPU…",
    "กำลังเทรน LoRA…": "Training LoRA…",
    "โซน": "Zone",
    "เลือกไฟล์ก่อนนำเข้า": "Pick a file first",
    "ยกเลิกงานในคิวแล้ว": "Queued job canceled",
    "ตั้งชื่อชุดเทรนก่อน": "Name the dataset first",
    "สร้าง dataset แล้ว — นำเข้ารูปได้เลย": "Dataset created — import images now",
    "คลิกในช่อง caption ก่อน แล้วค่อยกด tag":
      "Click inside a caption field first, then press a tag",
    "มี tag นี้ใน caption แล้ว": "Caption already has this tag",
    "ลบ tag ออกจากรายการแล้ว": "Tag removed from the list",
    "คลิกเพื่อแทรก · Alt+คลิกเพื่อลบ": "Click to insert · Alt+click to remove",
    "ไม่มี caption ที่เติม tag มุมได้อัตโนมัติ":
      "No captions can be auto-tagged",
    "ทำเครื่องหมายรูปที่ caption บ่งบอกว่ามีคนที่สองเป็น “หลายตัวละคร”?":
      "Mark images whose captions mention a second character as “multi-character”?",
    "ทำเครื่องหมายแล้ว — ตรวจสอบอีกครั้งก่อนเทรน":
      "Marked — double-check before training",
    "บันทึก trigger แล้ว — caption ที่ไม่มี trigger จะถูกเติมอัตโนมัติตอนเทรน":
      "Trigger saved — captions without it get it prepended automatically during training",
    "ลบ dataset แล้ว": "Dataset deleted",
    "ตั้งชื่อ LoRA ผลลัพธ์ก่อน": "Set the output LoRA name first",
    "เทรนเข้าคิวแล้ว — ดูความคืบหน้าในแผงคิว":
      "Training queued — watch progress in the queue panel",
    "สุ่มไอเดียแล้ว": "Idea randomized",
    "โหลดข้อมูลใหม่แล้ว": "Data refreshed",
    "คัดลอกพรอมป์ตแล้ว": "Prompt copied",
    "คัดลอกไม่สำเร็จ": "Couldn't copy",
    "ลบแล้ว": "Deleted",

    // server messages surfaced in toasts / queue
    "มีงานกำลังสร้างภาพอยู่ — ยกเลิกก่อนเปลี่ยนโมเดล":
      "A generation job is running — cancel it before switching models",
    "กำลังโหลดโมเดลอยู่ — รอสักครู่แล้วลองใหม่":
      "Model is loading — wait a moment and retry",
    "ต้องระบุ path ของ LoRA": "LoRA path is required",
    "ต้องระบุ LoRA ID": "LoRA ID is required",
    "มีงานกำลังสร้างภาพอยู่ — ยกเลิกก่อนแล้วค่อยคืนหน่วยความจำ":
      "A generation job is running — cancel it before freeing memory",
    "กรุณาใส่พรอมป์ตก่อนสร้างภาพ": "Enter a prompt first",
    "ไม่พบงานนี้": "Job not found",
    "direction ต้องเป็น up หรือ down": "direction must be up or down",
    "จัดลำดับได้เฉพาะงานที่ยังอยู่ในคิว": "Only queued jobs can be reordered",
    "งานนี้ไม่ได้อยู่ในคิวแล้ว": "This job is no longer in the queue",
    "อยู่ปลายคิวแล้ว": "Already at the end of the queue",
    "ไม่พบภาพชุดนี้": "Image set not found",
    "มีงานอยู่ในคิว/กำลังทำงาน — รอให้เสร็จก่อนล้างประวัติ":
      "Jobs are queued/running — wait for them before clearing history",
    "updates ต้องเป็น list": "updates must be a list",
    "ตั้งชื่อ LoRA ก่อนเทรน": "Name the output LoRA first",
    "ชื่อ LoRA ต้องมีตัวอักษร/ตัวเลข": "LoRA name must contain letters/numbers",
    "dataset ยังไม่มีรูป — เพิ่มรูปก่อนเทรน":
      "Dataset has no images — add some first",
    "LoRA ตั้งต้นชื่อเดียวกับผลลัพธ์ — ตั้งชื่อใหม่":
      "Init LoRA has the same name as the output — pick another name",
    "ต่อ Ollama ไม่ได้ — ตรวจว่า Ollama ทำงานอยู่":
      "Can't reach Ollama — check that it's running",
    "กำลังอ่านรูปอยู่ — รอให้เสร็จก่อนเปลี่ยนโมเดล":
      "Reading images — wait for it to finish before switching models",
    "การ์ดนี้หมดอายุแล้ว — ส่งคำสั่งใหม่":
      "This card expired — send the command again",
    "Undo หมดอายุแล้ว": "Undo expired",
    "ไม่พบงานอ่านรูปนี้ (อาจหมดอายุแล้ว)":
      "Vision job not found (it may have expired)",
    "ยังอ่านไม่เสร็จ — รอให้จบก่อนแล้วค่อยใช้ tag":
      "Reading isn't finished — wait for it before applying tags",
    "ไม่มีรูปตัวละครเดี่ยวให้เทรน — ทำเครื่องหมายรูปเดี่ยวอย่างน้อย 1 รูป หรือเปลี่ยนโหมดรูปหลายตัวละครเป็น 'ลดน้ำหนัก' / 'ใช้เต็ม'":
      "No solo images to train on — mark at least one image as solo, or switch the multi-character mode to 'lower weight' / 'include'",
    "เทรนต้องใช้ GPU — ตรวจ PyTorch/CUDA ใน engine/setup.sh":
      "Training needs a GPU — check PyTorch/CUDA in engine/setup.sh",
    "ไฟล์โมเดล SANA ไม่ครบ — ตรวจ models/":
      "SANA model files incomplete — check models/",
    "เทรนด้วย SDXL trainer ต้องเลือกโมเดลฐานตระกูล SDXL":
      "The SDXL trainer requires an SDXL base model",
    "ไม่พบโมเดลที่เลือก": "Selected model not found",
    "ไม่พบโมเดลที่เลือก — กดรีเฟรชรายการโมเดล":
      "Selected model not found — refresh the model list",
    "ต้องเป็นไฟล์ .safetensors": "Must be a .safetensors file",
    "LoRA นี้อยู่ใน models/loras อยู่แล้ว":
      "This LoRA is already in models/loras",
    "มีไฟล์ชื่อเดียวกันอยู่ใน models/loras แล้ว":
      "A file with the same name already exists in models/loras",
    "คัดลอกแล้ว แต่ระบบหา LoRA ไม่เจอ":
      "Copied, but the LoRA didn't show up in the list",
    "ดาวน์โหลดโมเดลไม่ครบ — ตรวจสอบอินเทอร์เน็ตและพื้นที่ว่างแล้วลองใหม่":
      "Model download incomplete — check your connection and free space, then retry",
    "ไฟล์โมเดลยังไม่ครบหลังการแปลง":
      "Model files still incomplete after conversion",
    "แปลง .pth → diffusers transformer (ใช้ RAM ประมาณ 8–10 GB)…":
      "Converting .pth → diffusers transformer (uses ~8–10 GB RAM)…",
    "ดาวน์โหลด text encoder / VAE / tokenizer (ไม่รวม transformer)…":
      "Downloading text encoder / VAE / tokenizer (transformer excluded)…",
    "แปลงเสร็จแล้ว": "Conversion finished",
    "ต้องติดตั้ง PyTorch ก่อนแปลง .pth":
      "Install PyTorch before converting .pth",
    "ไม่มี tensor": "No tensors",
    "ใช้ได้ทุกโมเดล": "Works with every model",
    "dataset id ไม่ถูกต้อง": "Invalid dataset id",
    "ไฟล์ LoRA ที่บันทึกไม่มีน้ำหนัก — ยกเลิกการเขียน":
      "Saved LoRA file has no weights — aborting write",
    "ไม่พบ dataset ที่จะเทรน": "Training dataset not found",
    "หยุดเทรนเพื่อความปลอดภัย": "Stopped training for safety",
    "เริ่มเทรน LoRA": "Starting LoRA training",
    "บันทึกไฟล์ LoRA…": "Saving LoRA file…",
    "กำลังโหลดโมเดล SANA สำหรับเทรน…": "Loading the SANA model for training…",
    "เฉพาะตัวละครเดี่ยว": "Solo images only",
    "ใช้ทุกรูปน้ำหนักเท่ากัน": "All images weighted equally",

    // host shell
    "SANA / SDXL / SD 1.5 · ทำงานในเครื่องนี้": "SANA / SDXL / SD 1.5 · runs on this machine",
    "โหลดโมเดล": "Load model",
    "โหลดโมเดลขึ้นหน่วยความจำล่วงหน้า":
      "Load the model into memory ahead of time",
    "คืน VRAM": "Free VRAM",
    "คืนหน่วยความจำ GPU": "Release GPU memory",
    "📁 ภาพ": "📁 Images",
    "เปิดโฟลเดอร์ภาพที่สร้าง": "Open the generated-images folder",
    "ดู log ล่าสุดของ engine": "View the recent engine log",
    "โหลดหน้าใหม่": "Reload page",
    "กำลังเชื่อมต่อ…": "Connecting…",
    "กำลังเปิดหน้า Image Studio…": "Opening Image Studio…",
    "↻ โหลดใหม่": "↻ Reload",
    "(log ว่าง — engine ยังไม่ได้เขียนอะไร)":
      "(log is empty — the engine hasn't written anything)",
    "พร้อมใช้งาน": "Ready",
    "เชื่อมต่อ engine ไม่ได้": "Can't reach the engine",
    "ไม่ทราบสาเหตุ": "Unknown cause",
    " — ดูรายละเอียดที่ปุ่ม log": " — see the log button for details",
    "engine แจ้งข้อผิดพลาด: ": "Engine reported an error: ",

    // setup page
    "ติดตั้ง PixelSmith Image Studio": "Install PixelSmith Image Studio",
    "ติดตั้ง Image Studio": "Install Image Studio",
    "ถ้ามีไฟล์": "If a",
    "ในเครื่องอยู่แล้ว ใช้การแปลงแทนการโหลด transformer ได้:":
      "file already exists on this machine, you can convert it instead of downloading the transformer:",
    "ทุกอย่างทำงานในเครื่องนี้ ไม่มีข้อมูลส่งออกอินเทอร์เน็ต · ภาพและประวัติเก็บในโฟลเดอร์":
      "Everything runs on this machine — nothing is sent to the internet · images and history are kept in the",
    "log ล่าสุด": "Recent log",
    "SANA 1.5 · 1.6B · 1024px — ติดตั้งครั้งเดียว ใช้ได้ตลอด":
      "SANA 1.5 · 1.6B · 1024px — install once, use forever",
    "สถานะเครื่องนี้": "This machine",
    "กำลังตรวจ…": "Checking…",
    "1) ติดตั้ง engine + โมเดล": "1) Install the engine + model",
    "เปิด terminal แล้วรัน (ดาวน์โหลด torch ~2.3 GB + โมเดล ~9 GB ครั้งแรกครั้งเดียว):":
      "Open a terminal and run (first time downloads torch ~2.3 GB + model ~9 GB):",
    "คัดลอก": "Copy",
    "2) เปิดโปรแกรม": "2) Launch the app",
    "ครั้งแรกจะติดตั้ง Electron (~100 MB) แล้วเปิดหน้าต่างโปรแกรมให้อัตโนมัติ":
      "The first run installs Electron (~100 MB), then opens the app window automatically",
    "ตรวจใหม่ / เปิดโปรแกรม": "Re-check / launch",
    "ไปที่หน้า Image Studio →": "Go to Image Studio →",
    "คัดลอกแล้ว ✓": "Copied ✓",
    "คัดลอกไม่ได้": "Couldn't copy",
    "engine ทำงานอยู่": "Engine running",
    "ไฟล์โมเดลครบ": "Model files complete",
    "กำลังดาวน์โหลดโมเดล": "Downloading model",
    "ยังไม่มีไฟล์โมเดล": "Model files missing",
    "ไม่พบ GPU / torch": "No GPU / torch",
    "พร้อมใช้งานแล้ว — กลับไปที่หน้า Image Studio ได้เลย":
      "Ready — you can go back to Image Studio now",
    "engine พร้อม แต่โมเดลจะถูกโหลด/ดาวน์โหลดอัตโนมัติเมื่อกดสร้างภาพครั้งแรก":
      "Engine is ready; the model loads/downloads automatically the first time you generate",
    "engine ไม่ทำงาน": "Engine not running",
    "ตรวจไม่ได้": "Can't check",
    "ยังไม่ได้เปิด engine — ติดตั้งตามขั้นตอนด้านบนแล้วเปิดโปรแกรม":
      "The engine isn't running — follow the steps above, then launch the app",
    "โหลด log ไม่ได้ (engine ยังไม่ทำงาน)":
      "Couldn't load the log (engine isn't running)",

    // region labels (app.js)
    "ซ้าย": "Left",
    "ขวา": "Right",
    "กลาง": "Center",
    "บนซ้าย": "Top left",
    "บนขวา": "Top right",
    "ล่างซ้าย": "Bottom left",
    "ล่างขวา": "Bottom right",

    // style chips (app.js)
    "ภาพถ่าย": "Photo",
    "อนิเมะ": "Anime",
    "ภาพวาด": "Painting",
    "ไซเบอร์พังค์": "Cyberpunk",
    "แฟนตาซี": "Fantasy",
    "ภาพยนตร์": "Cinematic",
    "มินิมอล": "Minimal",

    // assistant tool titles (engine/assistant.py, shown in results)
    "ตรวจความพร้อม dataset": "Audit dataset readiness",
    "ดูรายการ dataset": "List datasets",
    "ดูรายการ LoRA": "List LoRAs",
    "ดูรายการโมเดล": "List models",
    "สถานะ GPU / คิว": "GPU / queue status",
    "สรุปคลังภาพ": "Gallery summary",
    "ดูคิวงาน": "Show job queue",
    "เติม tag มุมอัตโนมัติ": "Auto-fill angle tags",
    "เปลี่ยน trigger word": "Change trigger word",
    "แก้ caption รายรูป": "Edit image caption",
    "เลือกโมเดล": "Select model",
    "ลบภาพจากประวัติ": "Delete image from history",
    "ล้างประวัติทั้งหมด": "Clear all history",
    "เริ่มเทรน LoRA": "Start LoRA training",
    "ช่วยเหลือ": "Help",

    // job lifecycle (server.py worker + queue)
    "เสร็จแล้ว": "Finished",
    "อยู่ในคิว": "In queue",
    "กำลังสร้างภาพ": "Generating",

    // engine generation / loading messages
    "กำลังดาวน์โหลดโมเดลครั้งแรก (~9 GB)…":
      "Downloading the model for the first time (~9 GB)…",
    "กำลังดาวน์โหลดโมเดลครั้งแรก (~9 GB) — ต้องต่ออินเทอร์เน็ต":
      "Downloading the model for the first time (~9 GB) — an internet connection is required",
    "inpaint ต้องการ init_image และ mask_image":
      "Inpaint needs both init_image and mask_image",
    "ไม่มี Pillow ในเครื่อง": "Pillow isn't installed",
    "ไม่มีโมเดล vision": "No vision model",
    "ไม่พบไฟล์รูป": "Image file not found",
    "ไม่มีรูปให้อ่านใน dataset นี้": "No images to read in this dataset",
    "กำลังอ่านรูปชุดก่อนอยู่ — รอให้เสร็จก่อน":
      "Already reading images — wait for it to finish",
    "GPU กำลังทำงาน (สร้างภาพ/โหลดโมเดล) — รอให้เสร็จก่อนแล้วสั่งอ่านรูปใหม่":
      "The GPU is busy (generating / loading a model) — wait for it, then ask again",
    "ใช้ tag มุมจาก vision": "Apply angle tags from vision",

    // prompt presets
    "(บันทึก/ใช้/นำเข้า)": "(save/use/import)",
    "ชื่อพรีเซ็ต": "Preset name",
    "ชื่อพรีเซ็ตใหม่": "new preset name",
    "บันทึกปัจจุบัน": "Save current",
    "บันทึกค่าในฟอร์มปัจจุบันเป็นพรีเซ็ต":
      "Save the current form values as a preset",
    "นำเข้าจาก collection": "Import from collection",
    "อ่านจาก data/prompt-collection.txt": "Read from data/prompt-collection.txt",
    "ไฟล์ .txt (A1111/collection)": ".txt file (A1111/collection)",
    "ใช้พรีเซ็ต = เติม prompt/negative/ขนาด/steps/CFG/sampler + LoRA ที่มีในระบบ · แก้ค่าได้หลังใช้เสมอ":
      "Applying a preset fills prompt/negative/size/steps/CFG/sampler + LoRAs present in the system · values stay editable afterwards",
    "ยังไม่มีพรีเซ็ต — บันทึกค่าปัจจุบันหรือนำเข้าจากไฟล์ด้านล่าง":
      "No presets yet — save the current values or import from a file below",
    "คลิกเพื่อใช้พรีเซ็ต": "Click to apply this preset",
    "ลบพรีเซ็ต": "Delete preset",
    "ตั้งชื่อพรีเซ็ตก่อน": "Name the preset first",
    "ลบพรีเซ็ตแล้ว": "Preset deleted",
    "คัดลอก parameters": "Copy parameters",
    "คัดลอกพรอมป์ต + negative + ค่าตั้งต้น แบบ A1111":
      "Copy prompt + negative + generation values in A1111 format",
    "คัดลอก parameters แล้ว": "Parameters copied",
    "ภาพล่าสุด": "the latest image",
    "ต้องยืนยันก่อน:": "Confirmation required:",
    "จะลบทุกภาพและประวัติทั้งหมด — ย้อนกลับไม่ได้":
      "This deletes every image and the whole history — it cannot be undone",

    // assistant rule-brain replies / confirmations
    "พิมพ์คำสั่งหรือคำถามมาก่อนครับ": "Type a command or question first",
    "ต้องระบุชื่อ LoRA ด้วยครับ เช่น “เริ่มเทรนชื่อ my-lora-v7”":
      "Please specify a LoRA name, e.g. “start training named my-lora-v7”",
    "ต้องระบุ trigger ใหม่ด้วยครับ เช่น “เปลี่ยน trigger เป็น my character”":
      "Please specify the new trigger, e.g. “change trigger to my character”",
    "รูปแบบ: “caption <ชื่อไฟล์> เป็น <ข้อความใหม่>”":
      "Format: “caption <file> as <new text>”",
    "ต้องระบุชื่อหรือ id ของโมเดลด้วยครับ":
      "Please specify the model name or id",
    "ต้องระบุพรอมป์ตด้วยครับ เช่น “สร้างภาพ my character standing”":
      "Please specify a prompt, e.g. “generate image my character standing”",
    "ยังไม่เข้าใจคำสั่งนี้ — พิมพ์ “ช่วยอะไรได้บ้าง” เพื่อดูตัวอย่าง":
      "I don't understand that command — type “help” for examples",
    "ไม่ได้เริ่มอ่านรูป": "Couldn't start reading images",
    "กำลังดูทีละรูปในเครื่อง — ระหว่างนี้การ์ดจอถูกใช้โดยโมเดล vision จะสร้างภาพ/เทรนพร้อมกันไม่ได้":
      "Reading images one by one locally — the GPU is held by the vision model, so generation/training can't run at the same time",
    "ไม่มี caption ที่เติม tag มุมได้อัตโนมัติ (รูปที่เหลือต้องเติมมือ)":
      "No captions can be auto-tagged (the rest need manual tags)",
    "ยังไม่มี dataset": "No datasets yet",
    "ยังไม่มี LoRA": "No LoRAs yet",
    "ไม่พบโมเดล": "No models found",
    "ไม่มีงานในคิว": "Queue is empty",
    "เรียบร้อย": "Done",
    "อ่านข้อมูล (ทำเลย)": "Read data (runs right away)",
    "แก้ข้อมูล (ทำเลย + Undo)": "Data edits (run + Undo)",
    "ลบ/เทรน (ต้องยืนยัน)": "Delete/Train (needs confirmation)",
    "สมอง: LLM โลคัลในเครื่อง (ไม่ส่งข้อมูลออก) — เข้าใจคำสั่งยืดหยุ่นขึ้น ตัวอย่าง:":
      "Brain: local LLM (nothing leaves the machine) — understands commands more flexibly. Examples:",
    "สมอง: กฎ (rule-based) — เข้าใจคำสั่งตามตัวอย่างนี้:":
      "Brain: rules (rule-based) — it understands commands like these examples:",
    "ไฟล์ยังไม่ครบ": "files incomplete",

    // in-app confirm dialog + blocking status
    "ยืนยันการทำรายการ": "Confirm action",
    "กำลังเปลี่ยนค่า กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Applying changes. Please don't switch windows or do anything else.",
    "กำลังเปิดใช้งานโหมดพลังงาน กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Enabling power modes. Please don't switch windows or do anything else.",
    "กำลังเปลี่ยนโมเดลผู้ช่วย กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Switching the assistant model. Please don't switch windows or do anything else.",
    "กำลังล้างประวัติ กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Clearing history. Please don't switch windows or do anything else.",
    "กำลังโหลดโมเดล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Loading the model. Please don't switch windows or do anything else.",
    "กำลังคืนหน่วยความจำโมเดล กรุณาอย่าสลับหน้าต่างหรือทำอย่างอื่น":
      "Freeing model memory. Please don't switch windows or do anything else.",
    "กำลังเปิดใช้งาน…": "Enabling…",
  };

  // ── dynamic messages: regex on the normalized Thai text ───────────────────
  const PATTERNS = [
    [/^เปลี่ยนโหมดพลังงานเป็น (.+) \((\d+) W\) แล้ว$/,
      (m) => `Power mode set to ${m[1]} (${m[2]} W)`],
    [/^ไม่รู้จักโหมด: (.+)$/, (m) => `Unknown mode: ${m[1]}`],
    [/^power limit ต้องอยู่ระหว่าง (\d+)–(\d+) W$/,
      (m) => `Power limit must be between ${m[1]}–${m[2]} W`],
    [/^เรียกตัวช่วยไม่สำเร็จ: (.+)$/, (m) => `Could not run the helper: ${m[1]}`],
    [/^เปิดตัวติดตั้งไม่สำเร็จ: (.+)$/, (m) => `Could not start setup: ${m[1]}`],
    [/^กำลังขยาย (\d+)\/(\d+)$/, (m) => `Upscaling ${m[1]}/${m[2]}`],
    [/^ภาพใหญ่เกินไป: (\d+)×(\d+) ×4 = (\d+)×(\d+) \(สูงสุด (\d+)px ต่อด้าน\)$/,
      (m) => `Image too large: ${m[1]}×${m[2]} ×4 = ${m[3]}×${m[4]} (max ${m[5]}px per side)`],
    [/^(\d+) ตัวอักษร$/, (m) => `${m[1]} chars`],
    [/^โซน (\d+)$/, (m) => `Zone ${m[1]}`],
    [/^(\d+×\d+) · (\d+) steps · CFG ([\d.]+) · seed (สุ่ม|\d+)$/,
      (m) => `${m[1]} · ${m[2]} steps · CFG ${m[3]} · seed ${m[4] === "สุ่ม" ? "random" : m[4]}`],
    [/^(.+) \(ยังไม่พร้อม\)$/, (m) => `${m[1]} (not ready)`],
    [/^— เลือกชุดเทรน \((\d+)\) —$/, (m) => `— Select a dataset (${m[1]}) —`],
    [/^(.+) \((\d+) รูป\)$/, (m) => `${m[1]} (${m[2]} images)`],
    [/^ทำเครื่องหมาย (\d+) รูปที่น่ามีคนที่สอง$/,
      (m) => `Mark ${m[1]} images that likely have a second character`],
    [/^เติม tag มุมอัตโนมัติ \((\d+) รูปเดี่ยว\)$/,
      (m) => `Auto-fill angle tags (${m[1]} solo images)`],
    [/^อีก (\d+) รูปเดี่ยว caption ไม่ได้บอกมุม(.*)$/,
      (m) => `${m[1]} more solo captions lack angle tags — fill them from the tag bar below`],
    [/^คิวเต็ม \(สูงสุด (\d+) งาน\) — รองานเดิมเสร็จก่อน$/,
      (m) => `Queue full (max ${m[1]}) — wait for running jobs`],
    [/^เพิ่มเข้าคิวแล้ว \(ลำดับ (\d+)\)$/, (m) => `Queued (position ${m[1]})`],
    [/^เทรนเสร็จ — LoRA "(.+)" พร้อมใช้แล้ว$/,
      (m) => `Training finished — LoRA "${m[1]}" is ready`],
    [/^สร้างเสร็จ (\d+) ภาพ$/, (m) => `Generated ${m[1]} images`],
    [/^(\d+) งานรออยู่$/, (m) => `${m[1]} job(s) waiting`],
    [/^(\d+) ภาพ$/, (m) => `${m[1]} images`],
    [/^(\d+) รูป$/, (m) => `${m[1]} images`],
    [/^(\d+) ชุด · (\d+) ภาพ$/, (m) => `${m[1]} sets · ${m[2]} images`],
    [/^(\d+) ชุด$/, (m) => `${m[1]} sets`],
    [/^(\d+) วิ$/, (m) => `${m[1]} s`],
    [/^เทรน LoRA: (.+)$/, (m) => `Train LoRA: ${m[1]}`],
    [/^ภาพ (\d+)\/(\d+)$/, (m) => `Image ${m[1]}/${m[2]}`],
    [/^LoRA ที่เคยใช้ไม่อยู่แล้ว: (.+)$/,
      (m) => `Previously used LoRAs no longer present: ${m[1]}`],
    [/^ใช้โมเดล (.+)$/, (m) => `Using model ${m[1]}`],
    [/^เลือกโมเดล (.+) แล้ว$/, (m) => `Model ${m[1]} selected`],
    [/^ใช้ tag มุมแล้ว (\d+) รูป$/, (m) => `Applied angle tags to ${m[1]} images`],
    [/^ใช้ tag แล้ว (\d+) รูป$/, (m) => `Applied tags to ${m[1]} images`],
    [/^อ่านรูปครบ (\d+) รูป$/, (m) => `Read ${m[1]} images`],
    [/^กำลังอ่าน (\d+)\/(\d+) รูป(.*)$/,
      (m) => `Reading ${m[1]}/${m[2]}${m[3] || ""}`],
    [/^กำลังหยุด… \((\d+)\/(\d+)\)$/, (m) => `Stopping… (${m[1]}/${m[2]})`],
    [/^หยุดแล้ว — อ่านไป (\d+)\/(\d+) รูป$/,
      (m) => `Stopped — read ${m[1]}/${m[2]} images`],
    [/^อ่านครบ (\d+)\/(\d+) รูป$/,
      (m) => `Read all ${m[1]}/${m[2]} images`],
    [/^กำลังเริ่มอ่าน (\d+) รูป…$/, (m) => `Starting to read ${m[1]} images…`],
    [/^จะเติม tag มุมให้ (\d+) รูป:([\s\S]*)$/,
      (m) => `Fill angle tags for ${m[1]} images:${m[2]}`],
    [/^เติม tag มุมแล้ว (\d+) รูป$/,
      (m) => `Filled angle tags for ${m[1]} images`],
    [/^…อีก (\d+) รูป$/, (m) => `…and ${m[1]} more`],
    [/^ลบ dataset "(.+)" และรูปทั้งหมด\?$/,
      (m) => `Delete dataset "${m[1]}" and all its images?`],
    [/^ลบ LoRA "(.+)" ใช่หรือไม่\?$/, (m) => `Delete LoRA "${m[1]}"?`],
    [/^นำเข้า (\d+) รูป · ข้าม (\d+) \(ซ้ำ\/ไม่รองรับ\)$/,
      (m) => `Imported ${m[1]} images · skipped ${m[2]} (duplicate/unsupported)`],
    [/^นำเข้า (\d+) รูปแล้ว$/, (m) => `Imported ${m[1]} images`],
    [/^ค่าเริ่มต้น \((.+)\)$/, (m) => `Default (${m[1]})`],
    [/^โมเดลถูกล็อกด้วย env IMGS_ASSIST_MODEL=(.+)$/,
      (m) => `Model locked via env IMGS_ASSIST_MODEL=${m[1]}`],
    [/^ยังไม่ติดตั้งโมเดล (.+) — เลือกจากรายการหรือติดตั้งใน Ollama$/,
      (m) => `Model ${m[1]} isn't installed — pick another or install it in Ollama`],
    [/^ไม่พบโมเดล “(.+)” ใน Ollama$/,
      (m) => `Model "${m[1]}" not found in Ollama`],
    [/^โมเดล (.+) ยังไม่พร้อมใช้งาน$/,
      (m) => `Model ${m[1]} isn't ready`],
    [/^โมเดล (.+) ยังไม่พร้อม — ตรวจไฟล์ใน (.+)$/,
      (m) => `Model ${m[1]} isn't ready — check files in ${m[2]}`],
    [/^เปลี่ยนเป็น (.+) — คืนหน่วยความจำโมเดลเดิมแล้ว$/,
      (m) => `Switched to ${m[1]} — previous model freed`],
    [/^เลือก (.+) แล้ว — จะโหลดเมื่อสร้างภาพ$/,
      (m) => `Selected ${m[1]} — it loads when you generate`],
    [/^เปลี่ยนเป็นโหมด (.+) — ต้องโหลดโมเดลใหม่$/,
      (m) => `Switched to ${m[1]} mode — the model reloads`],
    [/^ดาวน์โหลด (.+) ครั้งแรก \(~9 GB\)…$/,
      (m) => `Downloading ${m[1]} (~9 GB, first time)…`],
    [/^โหลด (.+) \(mmap\)…$/, (m) => `Loading ${m[1]} (mmap)…`],
    [/^ไม่พบไฟล์ \.pth: (.+)$/, (m) => `.pth file not found: ${m[1]}`],
    [/^มี transformer อยู่แล้ว — ข้ามการแปลง (.+)$/,
      (m) => `Transformer already exists — skipping conversion of ${m[1]}`],
    [/^มี LoRA ชื่อ (.+) อยู่แล้ว — ตั้งชื่อใหม่$/,
      (m) => `A LoRA named ${m[1]} already exists — pick another name`],
    [/^มีงานเทรนชื่อ (.+) อยู่ในคิวแล้ว$/,
      (m) => `A training job named ${m[1]} is already queued`],
    [/^มีไฟล์ชื่อ (.+) อยู่แล้ว — ตั้งชื่อใหม่$/,
      (m) => `A file named ${m[1]} already exists — pick another name`],
    [/^มีรูปชื่อนี้อยู่แล้ว: (.+)$/,
      (m) => `An image with this name already exists: ${m[1]}`],
    [/^รูปเต็มขีดจำกัด \((\d+) ใบ\)$/,
      (m) => `Image limit reached (${m[1]})`],
    [/^ไม่พบ LoRA ตั้งต้น: (.+)$/, (m) => `Init LoRA not found: ${m[1]}`],
    [/^ไฟล์ LoRA ตั้งต้นหายไป: (.+)$/,
      (m) => `Init LoRA file is missing: ${m[1]}`],
    [/^ไฟล์ LoRA ตั้งต้นไม่มีน้ำหนักที่ใช้ได้: (.+)$/,
      (m) => `Init LoRA file has no usable weights: ${m[1]}`],
    [/^โหลด LoRA ตั้งต้นไม่ครบ: (.+)$/,
      (m) => `Init LoRA didn't fully load: ${m[1]}`],
    [/^ปล่อยหน่วยความจำโมเดลเดิมก่อนเทรน \((.+)\)…$/,
      (m) => `Freeing the previous model before training (${m[1]})…`],
    [/^กำลังโหลดโมเดล SDXL สำหรับเทรน \((.+)\)…$/,
      (m) => `Loading the SDXL model for training (${m[1]})…`],
    [/^เตรียม latent \((\d+) รูป · (\d+)px\)…$/,
      (m) => `Preparing latents (${m[1]} images · ${m[2]}px)…`],
    [/^เตรียม latent (\d+)\/(\d+)$/, (m) => `Preparing latents ${m[1]}/${m[2]}`],
    [/^เตรียม embedding ของ caption \((\d+) รูป\) — ตัวอย่าง: “(.*)”$/,
      (m) => `Preparing caption embeddings (${m[1]} images) — sample: "${m[2]}"`],
    [/^ฉีด LoRA rank (\d+) \(alpha=\d+, scale 1\.0\)…$/,
      (m) => `Injecting LoRA rank ${m[1]} (alpha=${m[1]}, scale 1.0)…`],
    [/^โหลดน้ำหนักตั้งต้นจาก (.+) \(rank (\d+)\)…$/,
      (m) => `Loading init weights from ${m[1]} (rank ${m[2]})…`],
    [/^เทรนต่อ · step (\d+)\/(\d+)$/,
      (m) => `Resuming training · step ${m[1]}/${m[2]}`],
    [/^เทรน LoRA (SDXL )?(.+) · step (\d+)\/(\d+) · loss ([\d.]+)((?: · ชะลอ \d+%)?)$/,
      (m) => {
        const slow = m[6] ? ` · throttled ${m[6].match(/\d+/)[0]}%` : "";
        return `Training LoRA ${m[1] ? "SDXL " : ""}${m[2]} · step ${m[3]}/${m[4]} · loss ${m[5]}${slow}`;
      }],
    [/^บันทึก checkpoint step (\d+)\/(\d+)…$/,
      (m) => `Saving checkpoint step ${m[1]}/${m[2]}…`],
    [/^เทรนเสร็จ — บันทึก (.+) แล้ว$/,
      (m) => `Training finished — saved ${m[1]}`],
    [/^พัก GPU: (.+) \(เพดาน (\d+)°C\) — รอเย็นลงถึง (\d+)°C \((\d+)s\)$/,
      (m) => `Pausing GPU: ${m[1]} (ceiling ${m[2]}°C) — waiting to cool to ${m[3]}°C (${m[4]}s)`],
    [/^GPU ร้อนเกินไป \((.+)\) และไม่เย็นลงภายใน 15 นาที — หยุดเทรนเพื่อความปลอดภัย$/,
      (m) => `GPU too hot (${m[1]}) and didn't cool within 15 min — stopping training for safety`],
    [/^ลดน้ำหนักรูปหลายตัวละครเหลือ (\d+)%$/,
      (m) => `Multi-character images downweighted to ${m[1]}%`],
    [/^ไม่พบไฟล์โมเดล SDXL: (.+)$/,
      (m) => `SDXL model file not found: ${m[1]}`],
    [/^ยังไม่รองรับการเทรนโมเดลตระกูล (.+)$/,
      (m) => `Training isn't supported for the ${m[1]} family yet`],
    [/^LoRA ตั้งต้นเป็นตระกูล (.+) ไม่ตรงกับโมเดลฐาน \((.+)\)$/,
      (m) => `Init LoRA is family ${m[1]}, which doesn't match the base model (${m[2]})`],
    [/^engine แจ้งข้อผิดพลาด: (.+)$/,
      (m) => `Engine reported an error: ${m[1]} — see the log button for details`],
    [/^([\s\S]+) · (ยังไม่โหลดโมเดล|โมเดลพร้อม)$/,
      (m) => `${m[1]} · ${m[2] === "โมเดลพร้อม" ? "model ready" : "model not loaded"}`],

    // engine job messages
    [/^กำลังโหลด (.+)…$/, (m) => `Loading ${m[1]}…`],
    [/^กำลังสร้างภาพ (\d+)\/(\d+)$/, (m) => `Generating ${m[1]}/${m[2]}`],
    [/^เก็บรายละเอียด \(hires\) (\d+)\/(\d+)$/,
      (m) => `Refining detail (hires) ${m[1]}/${m[2]}`],
    [/^hires \(SANA\) — ขยายเป็น (\d+)×(\d+) \(ใช้ pipeline เดิม\)$/,
      (m) => `hires (SANA) — upscaling to ${m[1]}×${m[2]} (reusing the pipeline)`],
    [/^เปลี่ยนเป็นโหมด hires \((\d+)×(\d+)\) — โหลด pipeline ใหม่$/,
      (m) => `Switching to hires mode (${m[1]}×${m[2]}) — reloading the pipeline`],
    [/^พร้อมใช้งาน · (.+) \(offload: (.+)\)$/,
      (m) => `Ready · ${m[1]} (offload: ${m[2]})`],
    [/^ไฟล์โมเดล SANA ไม่ครบ: (.+)$/,
      (m) => `SANA model files incomplete: ${m[1]}`],
    [/^ยังไม่รองรับตระกูลโมเดล: (.+)$/,
      (m) => `The ${m[1]} family isn't supported yet`],
    [/^ไม่พบ LoRA ที่เลือก \(id: (.+)\)$/,
      (m) => `Selected LoRA not found (id: ${m[1]})`],
    [/^โมเดล (.+) ไม่รองรับ vision( หรือ Ollama ไม่พร้อม)?$/,
      (m) => `Model ${m[1]} doesn't support vision${m[2] || ""}`],

    // assistant rule-brain replies (assistant.py)
    [/^dataset (.+) \(([a-z0-9]+)\) · trigger “(.*)”\nรูปทั้งหมด (\d+) — เดี่ยว (\d+) · หลายตัวละคร (\d+)\ntag มุมขาด (\d+) รูป \(เติมอัตโนมัติได้ (\d+) · ต้องเติมมือ (\d+)\)\nมุมที่มี: (.*)\nสถานะ: (พร้อมเทรน ✅|ยังไม่พร้อม ⚠️)([\s\S]*)$/,
      (m) => `dataset ${m[1]} (${m[2]}) · trigger “${m[3]}”\nTotal ${m[4]} images — solo ${m[5]} · multi-char ${m[6]}\nImages missing angle tags: ${m[7]} (${m[8]} auto-fillable · ${m[9]} need manual tags)\nAngles present: ${m[10]}\nStatus: ${m[11] === "พร้อมเทรน ✅" ? "ready to train ✅" : "not ready ⚠️"}${m[12].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^เริ่มอ่านรูปด้วย vision แล้ว (\d+) รูป \(job ([a-f0-9]+)\)\nกำลังดูทีละรูปในเครื่อง — ระหว่างนี้การ์ดจอถูกใช้โดยโมเดล vision จะสร้างภาพ\/เทรนพร้อมกันไม่ได้$/,
      (m) => `Started reading ${m[1]} images with vision (job ${m[2]})\nReading locally one by one — the GPU is held by the vision model, so generation/training can't run at the same time`],
    [/^เริ่มอ่านรูปด้วย vision แล้ว (\d+) รูป$/,
      (m) => `Started reading ${m[1]} images with vision`],
    [/^เติม tag มุมแล้ว (\d+) รูป([\s\S]*)$/,
      (m) => {
        const tail = m[2].split("\n").map((l) => translateOne(l) ?? l).join("\n");
        return `Filled angle tags for ${m[1]} images${tail}`;
      }],
    [/^ยังต้องเติมมืออีก (\d+) รูป$/,
      (m) => `${m[1]} images still need manual tags`],
    [/^dataset ที่มี:\n([\s\S]+)$/,
      (m) => `Datasets:\n${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^(.+) \(([a-z0-9]+)\) — (\d+) รูป \(เดี่ยว (\d+) · multi (\d+)\)(?: · trigger “(.*)”)?$/,
      (m) => `${m[1]} (${m[2]}) — ${m[3]} images (solo ${m[4]} · multi ${m[5]})${m[6] != null ? ` · trigger "${m[6]}"` : ""}`],
    [/^LoRA ที่ติดตั้ง:\n([\s\S]+)$/,
      (m) => `Installed LoRAs:\n${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^โมเดลที่สแกนพบ:\n([\s\S]+)$/,
      (m) => `Models found:\n${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^([\s\S]+) · ไฟล์ยังไม่ครบ$/, (m) => `${m[1]} · files incomplete`],
    [/^GPU: (.+)\nVRAM: (.+)\nโมเดล: (.+) · คิว: (\d+) งาน · (กำลังทำงาน|ว่าง)$/,
      (m) => `GPU: ${m[1]}\nVRAM: ${m[2]}\nModel: ${m[3]} · Queue: ${m[4]} job(s) · ${m[5] === "กำลังทำงาน" ? "working" : "idle"}`],
    [/^ประวัติ (\d+) ชุด · ภาพรวม (\d+) ภาพ([\s\S]*)$/,
      (m) => `History: ${m[1]} sets · ${m[2]} images total${m[3].replace("ล่าสุด:", "Latest:")}`],
    [/^งานในคิว:\n([\s\S]+)$/,
      (m) => `Queued jobs:\n${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^(• )?(queued|running|done|error|canceled) · ([\s\S]+) \((\d+)%\)$/,
      (m) => `${m[1] || ""}${m[2]} · ${translateOne(m[3]) ?? m[3]} (${m[4]}%)`],
    [/^เปลี่ยน trigger: “(.*)” → “(.*)”$/,
      (m) => `Trigger changed: “${m[1]}” → “${m[2]}”`],
    [/^แก้ caption ของ (.+) แล้ว$/, (m) => `Caption updated for ${m[1]}`],
    [/^เลือกโมเดล: (.+) → (.+)$/, (m) => `Model selected: ${m[1]} → ${m[2]}`],
    [/^ส่งงานสร้างภาพเข้าคิวแล้ว \(job ([a-f0-9]+)\)\nพรอมป์ต: ([\s\S]+)$/,
      (m) => `Generation queued (job ${m[1]})\nPrompt: ${m[2]}`],
    [/^ลบแล้ว (\d+) ชุด \((\d+) ภาพ\)$/,
      (m) => `Deleted ${m[1]} sets (${m[2]} images)`],
    [/^ล้างประวัติแล้ว (\d+) ชุด$/, (m) => `History cleared (${m[1]} sets)`],
    [/^ลบ LoRA “(.+)” แล้ว$/, (m) => `Deleted LoRA “${m[1]}”`],
    [/^เข้าคิวเทรน LoRA “(.+)” แล้ว \(job ([a-f0-9]+)\)\ndataset (.+) · (\d+) steps$/,
      (m) => `LoRA training queued “${m[1]}” (job ${m[2]})\ndataset ${m[3]} · ${m[4]} steps`],
    [/^ต้องระบุ ([\w.,\s]+) ด้วยครับ$/, (m) => `Please specify ${m[1]}`],
    [/^• (.+) — เช่น “(.*)”$/,
      (m) => `• ${translateOne(m[1]) ?? m[1]} — e.g. “${m[2]}”`],
    [/^ต้องยืนยันก่อน: ([\s\S]+)$/,
      (m) => `Confirmation required:\n${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^ไม่รู้จักคำสั่ง “(.+)”$/, (m) => `Unknown command: ${m[1]}`],

    // prompt presets toasts / confirms
    [/^ใช้พรีเซ็ตแล้ว: (.+)$/, (m) => `Preset applied: ${m[1]}`],
    [/^บันทึกพรีเซ็ตแล้ว: (.+)$/, (m) => `Preset saved: ${m[1]}`],
    [/^ลบพรีเซ็ต "(.+)"\?$/, (m) => `Delete preset "${m[1]}"?`],
    [/^LoRA ในพรีเซ็ตไม่มีในระบบ: (.+)$/,
      (m) => `Preset LoRAs missing from the system: ${m[1]}`],
    [/^นำเข้าพรีเซ็ตแล้ว: เพิ่ม (\d+) · อัปเดต (\d+) · ข้าม (\d+)$/,
      (m) => `Presets imported: ${m[1]} added · ${m[2]} updated · ${m[3]} skipped`],
    [/^จะลบ (.+) พร้อมไฟล์บนดิสก์ — ย้อนกลับไม่ได้$/,
      (m) => `This deletes ${translateOne(m[1]) ?? m[1]} and its files on disk — it cannot be undone`],
    [/^จะลบ LoRA “(.+)” ออกจากเครื่อง — ย้อนกลับไม่ได้$/,
      (m) => `This deletes the LoRA “${m[1]}” from this machine — it cannot be undone`],
    [/^จะเทรน LoRA “(.+)” · dataset ([\w-]+) · (\d+) steps — ใช้เวลานานและการ์ดจะร้อน$/,
      (m) => `This trains the LoRA “${m[1]}” · dataset ${m[2]} · ${m[3]} steps — it takes a while and heats up the GPU`],

    // dataset audit notes (server.py _assist_dataset_summary)
    [/^ต้องมีรูปเดี่ยวอย่างน้อย 16 รูป \(มี (\d+)\)$/,
      (m) => `Needs at least 16 solo images (has ${m[1]})`],
    [/^ต้องมีอย่างน้อย 4 มุมที่ต่างกัน \(มี (\d+)\)$/,
      (m) => `Needs at least 4 distinct angles (has ${m[1]})`],
    [/^อีก (\d+) รูปยังไม่มี tag มุม \(เติมอัตโนมัติได้ (\d+)\)$/,
      (m) => `${m[1]} more images have no angle tag (${m[2]} auto-fillable)`],
    [/^(\d+) รูปเดี่ยว caption บ่งชี้ว่ามีคนที่สอง — ตรวจว่า mark multi ครบ$/,
      (m) => `${m[1]} solo captions hint at a second character — check the multi marks`],

    // assistant help_text (multi-line; translate the body line by line)
    [/^สมอง: LLM โลคัลในเครื่อง \(ไม่ส่งข้อมูลออก\) — เข้าใจคำสั่งยืดหยุ่นขึ้น ตัวอย่าง:([\s\S]+)$/,
      (m) => `Brain: local LLM (nothing leaves the machine) — understands commands more flexibly. Examples:${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
    [/^สมอง: กฎ \(rule-based\) — เข้าใจคำสั่งตามตัวอย่างนี้:([\s\S]+)$/,
      (m) => `Brain: rules (rule-based) — it understands commands like these examples:${m[1].split("\n").map((l) => translateOne(l) ?? l).join("\n")}`],
  ].filter((p) => p && p[0] instanceof RegExp && typeof p[1] === "function");

  // Composed toasts/errors: "ไทย prefix: rest" — translate prefix + recurse.
  const ERR_PREFIX = {
    "โหลดรายการโมเดลไม่ได้": "Couldn't load the model list",
    "โหลดประวัติไม่ได้": "Couldn't load history",
    "โหลดโมเดลไม่สำเร็จ": "Couldn't load the model",
    "โหลด log ไม่ได้": "Couldn't load the log",
    "เปิดโฟลเดอร์ไม่ได้": "Couldn't open the folder",
    "เทรนไม่สำเร็จ": "Training failed",
    "สร้างไม่สำเร็จ": "Generation failed",
    "อ่านรูปไม่สำเร็จ": "Reading images failed",
    "หยุดไม่สำเร็จ": "Couldn't stop",
    "ใช้ tag ไม่สำเร็จ": "Couldn't apply tags",
    "ทำไม่ได้": "Couldn't do it",
    "ยกเลิกไม่สำเร็จ": "Couldn't cancel",
    "Undo ไม่สำเร็จ": "Undo failed",
    "เชื่อมต่อผู้ช่วยไม่ได้": "Assistant unreachable",
    "สแกนโมเดลไม่ได้": "Couldn't scan models",
    "เปลี่ยนโมเดลไม่ได้": "Couldn't switch model",
    "โหลด LoRA ไม่สำเร็จ": "Couldn't load the LoRA",
    "เตรียม regional prompt ไม่สำเร็จ": "Couldn't prepare the regional prompt",
    "เปิดรูปไม่ได้": "Couldn't open the image",
  };

  function translateOne(text) {
    if (lang !== "en" || !THAI.test(text)) return null;
    const raw = String(text).trim();
    if (!raw) return null;
    // try the raw text first (preserves newlines for multi-line messages),
    // then the whitespace-collapsed variant
    for (const attempt of [raw, norm(raw)]) {
      if (EN[attempt]) return EN[attempt];
      for (const [re, fn] of PATTERNS) {
        const m = re.exec(attempt);
        if (m) return fn(m);
      }
    }
    const key = norm(raw);
    const idx = key.indexOf(": ");
    if (idx > 0) {
      const prefix = key.slice(0, idx);
      if (ERR_PREFIX[prefix]) {
        const rest = key.slice(idx + 2);
        const restT = translateOne(rest);
        return `${ERR_PREFIX[prefix]}: ${restT == null ? rest : restT}`;
      }
    }
    return null;
  }

  // originals kept so switching back to Thai restores them
  const ORIG_TEXT = new WeakMap();
  const ORIG_ATTR = new WeakMap();

  function applyText(node) {
    if (!node || node.nodeType !== 3) return;
    if (lang === "en") {
      if (node.parentElement && node.parentElement.closest("[data-no-i18n], .assistant-msg.user")) return;
      const t = translateOne(node.nodeValue);
      if (t != null && t !== node.nodeValue) {
        if (!ORIG_TEXT.has(node)) ORIG_TEXT.set(node, node.nodeValue);
        node.nodeValue = t;
      }
    } else if (ORIG_TEXT.has(node)) {
      node.nodeValue = ORIG_TEXT.get(node);
      ORIG_TEXT.delete(node);
    }
  }

  function applyAttrs(el) {
    if (!el || !el.getAttribute) return;
    if (el.closest && el.closest("[data-no-i18n], .assistant-msg.user")) return;
    for (const attr of ["placeholder", "title", "aria-label"]) {
      if (lang === "en") {
        const val = el.getAttribute(attr);
        if (val && THAI.test(val)) {
          const t = translateOne(val);
          if (t != null && t !== val) {
            let rec = ORIG_ATTR.get(el);
            if (!rec) ORIG_ATTR.set(el, (rec = {}));
            if (!(attr in rec)) rec[attr] = val;
            el.setAttribute(attr, t);
          }
        }
      } else {
        const rec = ORIG_ATTR.get(el);
        if (rec && attr in rec) {
          el.setAttribute(attr, rec[attr]);
          delete rec[attr];
        }
      }
    }
  }

  function applyNode(node) {
    if (!node) return;
    if (node.nodeType === 3) return applyText(node);
    if (node.nodeType !== 1) return;
    applyAttrs(node);
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    let n;
    const texts = [];
    while ((n = walker.nextNode())) texts.push(n);
    texts.forEach(applyText);
    if (lang === "en") {
      // children added with the subtree may carry attributes too
      const els = node.querySelectorAll ? node.querySelectorAll("[placeholder],[title],[aria-label]") : [];
      els.forEach(applyAttrs);
    }
  }

  function apply(root) {
    if (!root) return;
    if (root.nodeType === 3) return applyText(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const texts = [];
    let n;
    while ((n = walker.nextNode())) texts.push(n);
    texts.forEach(applyText);
    if (lang === "en") {
      const els = root.querySelectorAll
        ? root.querySelectorAll("[placeholder],[title],[aria-label]")
        : [];
      els.forEach(applyAttrs);
      if (root.getAttribute && (root.getAttribute("placeholder") || root.getAttribute("title") || root.getAttribute("aria-label"))) applyAttrs(root);
    }
  }

  function setLang(next) {
    if (next !== "th" && next !== "en") return;
    lang = next;
    try {
      localStorage.setItem(LS_KEY, lang);
    } catch (e) {
      /* storage blocked */
    }
    document.documentElement.lang = lang;
    apply(document);
    document.dispatchEvent(new CustomEvent("langchange", { detail: { lang } }));
  }

  const obs = new MutationObserver((records) => {
    if (lang === "th" && ORIG_TEXT.size === 0 && ORIG_ATTR.size === 0) return;
    for (const rec of records) {
      if (rec.type === "childList") {
        rec.addedNodes.forEach(applyNode);
      } else if (rec.type === "characterData") {
        applyText(rec.target);
      } else if (rec.type === "attributes") {
        applyAttrs(rec.target);
      }
    }
  });
  obs.observe(document.documentElement, {
    childList: true,
    characterData: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["placeholder", "title", "aria-label"],
  });

  if (lang !== "th") {
    const boot = () => {
      document.documentElement.lang = lang;
      apply(document);
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
      boot();
    }
  }

  // Public translator: unlike the internal translateOne() (which returns null
  // to mean "no translation"), this always returns usable text — the source
  // string falls through unchanged. Callers assign the result straight into
  // textContent, so returning null would blank the UI.
  function tr(text) {
    const out = translateOne(text);
    if (out != null) return out;
    return text == null ? "" : String(text);
  }

  window.I18N = {
    tr,
    apply,
    setLang,
    getLang: () => lang,
    EN,
  };
})();
