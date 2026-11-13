/* PixelSmith Image Studio — in-app training manual (Thai + English).
 *
 * Data only: studio/app.js renders it inside #guideModal and re-renders on
 * "langchange".  Every block carries both languages, so the manual never
 * depends on the i18n dictionary (which only covers short UI strings).
 *
 * Block shapes:
 *   { p: {th, en} }                       paragraph
 *   { list: [{th, en}, ...] }             bullet list
 *   { steps: [{th, en}, ...] }            numbered list
 *   { table: { head: [...], rows: [[...]] } }
 *   { note: { tone: "info"|"warn"|"ok", text: {th, en} } }
 *   { code: "..." }                       preformatted, language-neutral
 */
(() => {
  "use strict";

  const T = (th, en) => ({ th, en });

  window.TRAINING_GUIDE = {
    ui: {
      th: {
        title: "คู่มือการเทรน LoRA",
        subtitle:
          "อธิบายทุกฟังก์ชัน: ทำอะไร · ผลกระทบ · ข้อดี/ข้อเสีย · ค่าแนะนำ (SANA และ SDXL)",
        search: "ค้นหาหัวข้อหรือค่า เช่น bucketing, rank, 0.05…",
        noResults: "ไม่พบหัวข้อที่ค้นหา — ลองใช้คำอื่น",
        close: "ปิดคู่มือ",
        contents: "สารบัญ",
        footer:
          "ค่าตัวเลขเป็นคำแนะนำจากงานทดลองบนการ์ด 10 GB (P102-100) — ปรับตาม dataset จริงได้",
        open: "📘 คู่มือการเทรน",
        openTitle: "เปิดคู่มือการเทรน (ไทย/อังกฤษ)",
      },
      en: {
        title: "LoRA training manual",
        subtitle:
          "Every option explained: what it does · effect · pros/cons · recommended values (SANA and SDXL)",
        search: "Search a topic or value, e.g. bucketing, rank, 0.05…",
        noResults: "No matching topic — try another word",
        close: "Close manual",
        contents: "Contents",
        footer:
          "Values are recommendations from runs on a 10 GB card (P102-100) — adjust for your dataset",
        open: "📘 Training manual",
        openTitle: "Open the training manual (Thai/English)",
      },
    },

    sections: [
      {
        id: "overview",
        icon: "🧭",
        title: T("เริ่มต้นอย่างเร็ว", "Quick start"),
        blocks: [
          {
            steps: [
              T(
                "เลือกโมเดลฐาน (SANA หรือ SDXL) — LoRA ต้องอยู่ตระกูลเดียวกับโมเดลฐาน",
                "Choose the base model (SANA or SDXL) — a LoRA must match its base family"
              ),
              T(
                "สร้าง dataset ตั้งชื่อ + trigger + ประเภทคอนเซปต์",
                "Create a dataset: name, trigger word and concept kind"
              ),
              T(
                "อัปโหลดรูป แล้วใส่ tag/trigger/repeats ต่อรูป",
                "Import images, then set tags / trigger / repeats per image"
              ),
              T(
                "เขียน caption เฉพาะสิ่งที่เปลี่ยนในแต่ละรูป (ฉาก ท่า ชุด แสง)",
                "Caption only what changes per image (scene, pose, outfit, light)"
              ),
              T(
                "กดตรวจ QC แล้วแก้คำเตือน (ไฟล์ซ้ำ รูปเล็ก caption ขาด)",
                "Run dataset QC and fix the warnings (duplicates, tiny images, missing captions)"
              ),
              T(
                "เลือก preset หรือตั้งค่าเอง — เปิด bucketing ไว้",
                "Pick a preset or set values yourself — keep bucketing on"
              ),
              T(
                "กดเริ่มเทรน งานจะเข้าคิวและปล่อยโมเดล generate ก่อนอัตโนมัติ",
                "Start training; it joins the queue and unloads the generation model first"
              ),
              T(
                "สร้างภาพทดสอบด้วย seed คงที่ แล้วเลือก checkpoint ที่ดีที่สุด",
                "Generate samples with a fixed seed and keep the best checkpoint"
              ),
            ],
          },
          {
            note: {
              tone: "info",
              text: T(
                "คุณภาพ LoRA มาจาก dataset + caption มากกว่าการจูน optimizer — ถ้ามีเวลาแก้อย่างเดียว ให้แก้รูปและ caption ก่อน",
                "LoRA quality comes from the dataset and captions more than optimizer tuning — if you only fix one thing, fix images and captions first"
              ),
            },
          },
        ],
      },

      {
        id: "kind",
        icon: "🏷️",
        title: T("ประเภทคอนเซปต์ (Dataset kind)", "Concept kind"),
        blocks: [
          {
            p: T(
              "ประเภทไม่ได้เปลี่ยนอัลกอริทึมเทรน แต่เปลี่ยนเช็กลิสต์รูป คำแนะนำ caption ค่า preset และนโยบายรูปหลายตัวละคร",
              "The kind does not change the training algorithm. It changes the image checklist, caption guidance, preset values and the multi-character policy."
            ),
          },
          {
            table: {
              head: [
                T("ประเภท", "Kind"),
                T("ใช้เมื่อ", "Use when"),
                T("จำนวนรูป", "Images"),
                T("caption ควรเน้น", "Caption focus"),
              ],
              rows: [
                [
                  T("Character", "Character"),
                  T("อยากให้จำใบหน้า/ตัวตน", "You want the face/identity"),
                  T("16–30 รูปเดี่ยว หลายมุม", "16–30 solo, several angles"),
                  T(
                    "ฉาก/ท่า/ชุด/แสง — ไม่บรรยายลักษณะถาวร",
                    "Scene/pose/outfit/light — never describe permanent traits"
                  ),
                ],
                [
                  T("Style", "Style"),
                  T("อยากได้สไตล์สี/เส้น/แสง", "You want colors/lines/lighting"),
                  T("20–30+ รูป หลากหลายเนื้อหา", "20–30+ varied subjects"),
                  T(
                    "บรรยายเนื้อหาให้ละเอียด — ไม่บรรยายสไตล์",
                    "Describe the subject in detail — never describe the style"
                  ),
                ],
                [
                  T("Outfit / object", "Outfit / object"),
                  T("อยากให้จำชุด/ไอเทม", "You want the outfit/item"),
                  T("10–20 รูป เห็นชิ้นงานชัด", "10–20 clear item shots"),
                  T(
                    "รายละเอียดชิ้นงานที่คงที่ (สี โลโก้ วัสดุ)",
                    "Fixed item details (color, logo, material)"
                  ),
                ],
                [
                  T("Pose / concept", "Pose / concept"),
                  T("อยากให้จำท่าหรือมุม", "You want a pose or camera concept"),
                  T("20–40 รูป หลายมุมกล้อง", "20–40 varied camera angles"),
                  T(
                    "ชื่อท่า/คอนเซปต์ + มุมกล้อง",
                    "Pose/concept name + camera angle"
                  ),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "style/outfit/pose บังคับใช้รูปหลายตัวละครแบบ include (น้ำหนักเท่ากัน) เพราะเนื้อหาหลักไม่ใช่ตัวละครเดียว",
                "style/outfit/pose force the multi-character policy to include (equal weight) because the concept is not one character"
              ),
              T(
                "character ใช้ exclude เป็นค่าเริ่มต้น เพื่อไม่ให้ trigger ไปจำตัวละครประกอบ",
                "character defaults to exclude so the trigger never absorbs background characters"
              ),
              T(
                "เปลี่ยนประเภทได้ตลอด แต่ควรเปลี่ยนก่อนเริ่มเทรนรอบจริง",
                "You can change the kind anytime, but do it before a real run"
              ),
            ],
          },
        ],
      },

      {
        id: "trigger",
        icon: "🔑",
        title: T("Trigger word และ Extra triggers", "Trigger word and extra triggers"),
        blocks: [
          {
            p: T(
              "Trigger คือคำพิเศษที่ผูกทุกภาพเข้าด้วยกัน (เช่น lola, mytrigger7) ตอน generate ใส่คำนี้ในพรอมป์ตเพื่อเรียกคอนเซปต์",
              "The trigger is a rare token that ties every image together (e.g. lola, mytrigger7). Put it in the prompt to summon the concept."
            ),
          },
          {
            table: {
              head: [T("ช่อง", "Field"), T("ทำอะไร", "What it does"), T("แนะนำ", "Recommendation")],
              rows: [
                [
                  T("Trigger หลัก", "Main trigger"),
                  T("คำเดียวที่แทนคอนเซปต์ทั้งหมด", "One token for the whole concept"),
                  T(
                    "คำแปลก ตัวพิมพ์เล็ก ไม่ซ้ำคำทั่วไป เช่น lola7",
                    "Rare lowercase token, not a common word, e.g. lola7"
                  ),
                ],
                [
                  T("Extra triggers (สูงสุด 5)", "Extra triggers (max 5)"),
                  T("คำเรียกสำรอง/ชื่อชุด/ชื่อท่า", "Aliases, outfit or pose names"),
                  T("armor form, casual outfit", "armor form, casual outfit"),
                ],
                [
                  T("Trigger ต่อรูป", "Per-image trigger"),
                  T("ใช้เมื่อบางรูปต้องใช้คำอื่น", "Use when a few images need another token"),
                  T("เฉพาะรูปชุดเกราะ → armor form", "Only the armor images → armor form"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "ระบบเติม trigger ให้อัตโนมัติถ้า caption ไม่มี และไม่เติมซ้ำถ้ามีอยู่แล้ว",
                "The trigger is added automatically when missing, and never duplicated"
              ),
              T(
                "ข้อดี: เรียกคอนเซปต์แม่น แยกหลายรูปแบบได้",
                "Pro: precise recall and several variants from one dataset"
              ),
              T(
                "ข้อเสีย: trigger ที่ซ้ำคำธรรมดา (girl, red) จะปนกับของเดิมในโมเดล",
                "Con: a trigger that is a common word (girl, red) blends with what the model already knows"
              ),
            ],
          },
        ],
      },

      {
        id: "captions",
        icon: "✍️",
        title: T("Caption, Tags และ Repeats", "Captions, tags and repeats"),
        blocks: [
          {
            p: T(
              "Caption ที่ใช้เทรนถูกประกอบเป็น: trigger, tag1, tag2, … , caption อิสระ (ตัดคำซ้ำแบบไม่สนตัวพิมพ์)",
              "The training caption is composed as: trigger, tag1, tag2, … , free caption (case-insensitive duplicates removed)."
            ),
          },
          {
            table: {
              head: [T("ช่อง", "Field"), T("ทำอะไร", "What it does"), T("คำแนะนำ", "Guidance")],
              rows: [
                [
                  T("Caption", "Caption"),
                  T("บรรยายสิ่งที่เปลี่ยนในภาพ", "Describe what changes in the image"),
                  T(
                    "ฉาก ท่า มุมกล้อง ชุด แสง อารมณ์ — ภาษาอังกฤษได้ผลดีที่สุด",
                    "Scene, pose, angle, outfit, light, mood — English works best"
                  ),
                ],
                [
                  T("Tags", "Tags"),
                  T("ป้ายสั้น สูงสุด 20 ต่อรูป", "Short labels, up to 20 per image"),
                  T(
                    "ใช้แยกมุม/ท่าออกจากตัวตน (front view, sitting)",
                    "Separate angles/poses from identity (front view, sitting)"
                  ),
                ],
                [
                  T("Repeats (1–10)", "Repeats (1–10)"),
                  T("จำนวนรอบที่รูปถูกใช้ต่อ epoch", "Times the image is used per epoch"),
                  T(
                    "รูปสำคัญ/มีน้อย → 2–3 · รูปซ้ำ/คุณภาพต่ำ → 1 หรือลบ",
                    "Rare/important images → 2–3 · duplicates/low quality → 1 or remove"
                  ),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "ควรเขียน: สิ่งที่แตกต่างในแต่ละรูป เพื่อให้ trigger เก็บเฉพาะสิ่งที่เป็นคอนเซปต์",
                "Do write: what differs per image, so the trigger learns only the concept"
              ),
              T(
                "ไม่ควรเขียน: ลักษณะถาวร (ใบหน้า สีผม หุ่น) เพราะจะทำให้ trigger เรียนได้ไม่เต็มที่",
                "Do not write: permanent traits (face, hair color, body) — they weaken what the trigger learns"
              ),
              T(
                "ถ้า tag ซ้ำกับ caption ระบบจะไม่เพิ่มซ้ำให้",
                "If a tag already appears in the caption, it is not added twice"
              ),
            ],
          },
        ],
      },

      {
        id: "multi",
        icon: "👥",
        title: T("นโยบายรูปหลายตัวละคร", "Multi-character policy"),
        blocks: [
          {
            table: {
              head: [
                T("โหมด", "Mode"),
                T("ใช้รูป", "Uses"),
                T("น้ำหนัก", "Weight"),
                T("เหมาะกับ", "Best for"),
              ],
              rows: [
                [
                  T("Exclude", "Exclude"),
                  T("เฉพาะรูปเดี่ยว", "Solo images only"),
                  T("1.0", "1.0"),
                  T("Character LoRA (ค่าเริ่มต้น)", "Character LoRA (default)"),
                ],
                [
                  T("Weighted", "Weighted"),
                  T("ทุกรูป", "All images"),
                  T("รูป multi = 5–100%", "multi = 5–100%"),
                  T(
                    "character ที่อยากให้อยู่ร่วมฉากได้",
                    "character that should share scenes"
                  ),
                ],
                [
                  T("Include", "Include"),
                  T("ทุกรูป", "All images"),
                  T("1.0", "1.0"),
                  T("style/outfit/pose (บังคับอัตโนมัติ)", "style/outfit/pose (forced)"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "Weighted weight แนะนำ 20–30% (ค่าเริ่มต้น 25%)",
                "Weighted weight: 20–30% recommended (default 25%)"
              ),
              T(
                "ถ้าใช้ weighted/include ควรใส่ caption “with another character” ให้รูป multi",
                "When using weighted/include, caption multi images with “with another character”"
              ),
              T(
                "รูป multi ไม่ควรมี trigger เดียวกับรูปเดี่ยวถ้าตัวละครที่สองจะถูกดูดเข้าไปใน LoRA",
                "A multi image should not share the trigger if the second character must not leak into the LoRA"
              ),
            ],
          },
        ],
      },

      {
        id: "qc",
        icon: "🔍",
        title: T("ตรวจคุณภาพ dataset (QC)", "Dataset QC"),
        blocks: [
          {
            p: T(
              "ปุ่ม QC ตรวจข้อมูลก่อนเทรนและไม่แก้อะไรให้อัตโนมัติ — ปลอดภัยกับข้อมูล ใช้ก่อนทุกครั้งที่แก้ dataset",
              "The QC button inspects the dataset without changing anything — safe to run after every edit."
            ),
          },
          {
            list: [
              T("จำนวนรูปทั้งหมด / เดี่ยว / หลายตัวละคร", "Total / solo / multi-character images"),
              T("ขนาด (กว้าง×สูง) และ bucket ที่จะใช้", "Size (w×h) and the bucket that will be used"),
              T("ไฟล์ซ้ำ (SHA-1 ตรงกัน)", "Duplicate files (identical SHA-1)"),
              T("รูปที่ไม่มี caption/tags", "Images without caption or tags"),
              T("รูปเล็กกว่า 512px และสัดส่วนสุดโต่ง", "Images below 512px and extreme aspect ratios"),
              T("คำแนะนำสรุปพร้อมเกณฑ์เป้าหมาย", "A summary with target thresholds"),
            ],
          },
          {
            note: {
              tone: "ok",
              text: T(
                "เกณฑ์ที่ควรผ่าน: ไม่มีไฟล์ซ้ำ · รูปเดี่ยว ≥16 (character) · ทุกรูปมี caption/tags · ด้านสั้น ≥512px",
                "Target: no duplicates · ≥16 solo images (character) · every image captioned/tagged · short side ≥512px"
              ),
            },
          },
        ],
      },

      {
        id: "bucketing",
        icon: "📐",
        title: T("Aspect-ratio bucketing", "Aspect-ratio bucketing"),
        blocks: [
          {
            p: T(
              "เลือกขนาดกล่อง (bucket) ที่ใกล้สัดส่วนรูปจริงที่สุด แล้ว resize/ครอปให้น้อยที่สุด แทนการบังคับทุกภาพเป็นจัตุรัส",
              "Picks the bucket closest to each image's real aspect ratio, minimizing crop instead of forcing every image to a square."
            ),
          },
          {
            p: T(
              "ตัวอย่าง bucket ที่ resolution 1024: 1024×1024, 1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344",
              "Example buckets at 1024: 1024×1024, 1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344"
            ),
          },
          {
            list: [
              T(
                "ข้อดี: เก็บองค์ประกอบครบ ฝึกแนวตั้ง/แนวนอนได้ตรงสัดส่วน คุณภาพดีขึ้นชัดกับ dataset หลายสัดส่วน",
                "Pro: keeps composition, trains portrait/landscape at their real ratio, clearly better on mixed-aspect datasets"
              ),
              T(
                "ข้อเสีย: ใช้ VRAM มากกว่าเล็กน้อย และโมเดลเห็นขนาดหลากหลายขึ้น (ปกติไม่เป็นปัญหา)",
                "Con: slightly more VRAM and the model sees more sizes (rarely a problem)"
              ),
              T(
                "แนะนำ: เปิด (ค่าเริ่มต้น) ทุกกรณี · ปิดเมื่อต้องการเทียบกับพฤติกรรมเดิม",
                "Recommendation: on (default) always · turn off only to compare against the old square behavior"
              ),
            ],
          },
          {
            note: {
              tone: "info",
              text: T(
                "SANA รองรับ latent ขนาดต่างกันได้เอง · SDXL ต้องส่ง time_ids ต่อรูป ระบบจัดการให้อัตโนมัติ",
                "SANA handles variable latent sizes natively · SDXL needs per-image time_ids, which the engine sets automatically"
              ),
            },
          },
        ],
      },

      {
        id: "dropout",
        icon: "🎲",
        title: T("Caption dropout", "Caption dropout"),
        blocks: [
          {
            p: T(
              "สุ่มทิ้ง caption ทั้งหมดของ step นั้น (ใช้ embedding ว่างแทน) ตามความน่าจะเป็นที่ตั้ง",
              "Randomly drops the whole caption for a step (empty embedding) with the configured probability."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: ฝึก unconditional ให้แข็งขึ้น ลดการพึ่ง caption มากเกินไป ช่วย composition",
                "Pro: stronger unconditional branch, less caption dependence, better composition"
              ),
              T(
                "ข้อเสีย: มากเกินไปทำให้ความสัมพันธ์ trigger↔ภาพอ่อนลง (LoRA ตอบ trigger ไม่แม่น)",
                "Con: too much weakens trigger↔image association (the LoRA responds less reliably)"
              ),
            ],
          },
          {
            table: {
              head: [T("ค่า", "Value"), T("เหมาะกับ", "Use for")],
              rows: [
                [T("0", "0"), T("ต้องการ trigger แม่นที่สุด", "Maximum trigger fidelity")],
                [T("0.02–0.05", "0.02–0.05"), T("style / งานทั่วไป (ค่าเริ่มต้น 0.05)", "style / general (default 0.05)")],
                [T("0.05", "0.05"), T("character / outfit", "character / outfit")],
                [T("สูงสุด 0.15", "max 0.15"), T("dataset เล็ก (<20 รูป) — ห้ามเกิน", "small datasets (<20 images) — never exceed")],
              ],
            },
          },
        ],
      },

      {
        id: "shuffle",
        icon: "🔀",
        title: T("Caption shuffle + Keep tokens", "Caption shuffle + keep tokens"),
        blocks: [
          {
            p: T(
              "สร้าง caption หลายลำดับคำ (สลับเฉพาะส่วนหลัง) แล้วสุ่มใช้แต่ละ step",
              "Builds several word orders (only the tail is shuffled) and picks one per step."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: กันโมเดลยึดลำดับคำ ทำให้แต่ละแท็กถูกเรียนแยกกัน ช่วย generalize",
                "Pro: stops the model relying on word order, so each tag is learned separately and generalizes better"
              ),
              T(
                "ข้อเสีย: caption สั้นมากแทบไม่มีผล · keep_tokens=0 อาจสลับ trigger ไปกลางประโยค",
                "Con: almost no effect on very short captions · keep_tokens=0 can move the trigger mid-sentence"
              ),
              T(
                "Keep tokens (0–4): จำนวนส่วนแรก (คั่นด้วย comma) ที่ไม่ถูกสลับ — กัน trigger/คำสำคัญหลุดตำแหน่ง",
                "Keep tokens (0–4): leading comma-separated segments never shuffled — keeps the trigger/key words in place"
              ),
            ],
          },
          {
            note: {
              tone: "ok",
              text: T(
                "แนะนำ: เปิดเฉพาะ style และ caption ที่มีหลายแท็ก · keep_tokens 1 (กัน trigger) หรือ 2 · character ปิดไว้ก่อน",
                "Recommendation: on for style and multi-tag captions · keep_tokens 1 (trigger) or 2 · off for character"
              ),
            },
          },
        ],
      },

      {
        id: "noise",
        icon: "🌓",
        title: T("Noise offset", "Noise offset"),
        blocks: [
          {
            p: T(
              "บวกค่า offset แบบสุ่มแยกตาม channel ลงใน noise ตอนเทรน ช่วยเรื่องแสงเงาของ dataset ที่มืดหรือคอนทราสต์สูง",
              "Adds a random per-channel offset to the training noise; helps lighting on dark or high-contrast datasets."
            ),
          },
          {
            table: {
              head: [T("ค่า", "Value"), T("เหมาะกับ", "Use for")],
              rows: [
                [T("0", "0"), T("งานทั่วไป (ค่าเริ่มต้น)", "General (default)")],
                [T("0.02–0.05", "0.02–0.05"), T("dataset มืด/กลางคืน/backlight", "dark/night/backlit datasets")],
                [T("0.1", "0.1"), T("กรณีสุดโต่งเท่านั้น", "Extreme cases only")],
              ],
            },
          },
          {
            list: [
              T(
                "สูงเกินไปทำให้ภาพออกเทา/contrast เพี้ยน",
                "Too high washes out contrast and makes images gray"
              ),
              T(
                "ไม่เกี่ยวกับการแก้ภาพที่ generate แล้ว — เป็นการปรับการเรียนเท่านั้น",
                "It does not post-process generated images — it only changes training"
              ),
            ],
          },
        ],
      },

      {
        id: "minsnr",
        icon: "⚖️",
        title: T("Min-SNR gamma", "Min-SNR gamma"),
        blocks: [
          {
            p: T(
              "ถ่วงน้ำหนัก loss ตาม SNR ของ timestep: weight = clamp(SNR, max=gamma) / SNR",
              "Weights the loss by timestep SNR: weight = clamp(SNR, max=gamma) / SNR."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: ลดอิทธิพลของ step ที่ noisy มาก/น้อยเกินไป เรียนโครงสร้างเร็วขึ้น เสถียรกับ LR สูง",
                "Pro: down-weights extreme noise levels, learns structure faster, more stable at higher LR"
              ),
              T(
                "ข้อเสีย: gamma สูงเกินไปอาจลดรายละเอียดเล็ก ๆ (ผิว/ลายผ้า)",
                "Con: too high can reduce fine detail (skin, fabric patterns)"
              ),
              T(
                "SANA ใช้ SNR = ((1-σ)/σ)² · SDXL ใช้ SNR = α/(1-α)",
                "SANA uses SNR = ((1-σ)/σ)² · SDXL uses SNR = α/(1-α)"
              ),
            ],
          },
          {
            table: {
              head: [T("ค่า", "Value"), T("เหมาะกับ", "Use for")],
              rows: [
                [T("0 (ปิด)", "0 (off)"), T("character — ต้องการ likeness สูง (ค่าเริ่มต้น)", "character — maximum likeness (default)")],
                [T("5", "5"), T("style", "style")],
                [T("5–10", "5–10"), T("LR สูง หรือ step น้อย", "high LR or few steps")],
              ],
            },
          },
        ],
      },

      {
        id: "rslora",
        icon: "🧮",
        title: T("rsLoRA", "rsLoRA"),
        blocks: [
          {
            p: T(
              "เปลี่ยน scaling ของ LoRA จาก alpha/r เป็น alpha/√r ทำให้ rank สูงเรียนได้เสถียรขึ้น",
              "Changes LoRA scaling from alpha/r to alpha/√r, which keeps high ranks stable."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: gradient ไม่ระเบิด/หายง่ายเมื่อ rank 64–128",
                "Pro: gradients stay healthy at rank 64–128"
              ),
              T(
                "ข้อเสีย: ไฟล์ที่ได้ต้องแปลง scaling กลับเป็นแบบมาตรฐานตอนบันทึก — ระบบทำให้อัตโนมัติ",
                "Con: the saved file must use standard scaling — the engine does this automatically"
              ),
              T(
                "ความเข้ากันได้: lora_B ถูกคูณ √r ก่อนเซฟ จึงโหลดกับ diffusers ปกติได้",
                "Compatibility: lora_B is multiplied by √r before saving, so normal diffusers loaders work"
              ),
              T(
                "แนะนำ: เปิดเมื่อ rank 64 ขึ้นไป · ปิดสำหรับ rank 4–32",
                "Recommendation: on for rank 64+ · off for rank 4–32"
              ),
            ],
          },
        ],
      },

      {
        id: "targets",
        icon: "🎯",
        title: T("Target modules", "Target modules"),
        blocks: [
          {
            table: {
              head: [
                T("ค่า", "Value"),
                T("เรียนอะไร", "Trains"),
                T("ข้อดี", "Pros"),
                T("ข้อเสีย", "Cons"),
              ],
              rows: [
                [
                  T("attn (ค่าเริ่มต้น)", "attn (default)"),
                  T("q, k, v, out ของ attention", "attention q, k, v, out"),
                  T("เบา เร็ว ไฟล์เล็ก ปลอดภัย", "Light, fast, small file, safe"),
                  T("texture/โครงสร้างอาจไม่ลึก", "May not capture deep texture/structure"),
                ],
                [
                  T("attn_ff", "attn_ff"),
                  T("attention + feed-forward", "attention + feed-forward"),
                  T("จำรายละเอียด/โครงสร้างเก่งขึ้น", "Better detail and structure recall"),
                  T("ช้า ~30–50% ไฟล์ใหญ่ overfit ง่ายขึ้น", "~30–50% slower, larger file, easier to overfit"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "SANA FF: ff.conv_inverted, ff.conv_point · SDXL FF: ff.net.0.proj, ff.net.2",
                "SANA FF: ff.conv_inverted, ff.conv_point · SDXL FF: ff.net.0.proj, ff.net.2"
              ),
              T(
                "แนะนำ: attn สำหรับ character/outfit · attn_ff เมื่อรู้สึกว่า LoRA จำรายละเอียดไม่พอและมี step เหลือ",
                "Recommendation: attn for character/outfit · attn_ff when detail recall is insufficient and you have steps to spare"
              ),
            ],
          },
        ],
      },

      {
        id: "rank",
        icon: "📊",
        title: T("Rank (ขนาด LoRA)", "Rank (LoRA size)"),
        blocks: [
          {
            p: T(
              "จำนวนมิติของเมทริกซ์ LoRA — ยิ่งสูงยิ่งจุรายละเอียดได้มาก แต่ไฟล์ใหญ่และ overfit ง่ายขึ้น",
              "The LoRA matrix dimension — higher holds more detail, but makes bigger files that overfit sooner."
            ),
          },
          {
            table: {
              head: [T("ประเภท", "Kind"), T("Rank แนะนำ", "Recommended rank"), T("ช่วง", "Range")],
              rows: [
                [T("Character", "Character"), T("32", "32"), T("16–64", "16–64")],
                [T("Style", "Style"), T("32", "32"), T("16–64", "16–64")],
                [T("Outfit / object", "Outfit / object"), T("16", "16"), T("8–32", "8–32")],
                [T("Pose / concept", "Pose / concept"), T("16–32", "16–32"), T("8–32", "8–32")],
                [T("ทดลองเร็ว ๆ", "Quick test"), T("16", "16"), T("8–16", "8–16")],
              ],
            },
          },
          {
            list: [
              T(
                "rank สูง: จำรายละเอียด/หลายรูปแบบได้ดี แต่ต้องใช้ step มากขึ้นและช้ากว่าเล็กน้อย",
                "High rank: better detail and multiple variants, but needs more steps and is slightly slower"
              ),
              T(
                "เทรนต่อจาก LoRA เดิม: ระบบใช้ rank ของไฟล์เดิมอัตโนมัติเพื่อโหลดน้ำหนักได้",
                "Continuing from an existing LoRA: the engine reuses the file's rank so weights load"
              ),
            ],
          },
        ],
      },

      {
        id: "steps",
        icon: "🔢",
        title: T("Steps, Epochs และเครื่องคิดเลข", "Steps, epochs and the calculator"),
        blocks: [
          {
            p: T(
              "Steps = จำนวนรอบอัปเดต optimizer ทั้งหมด · Epochs = จำนวนรอบที่วนครบ dataset · grad_accum = จำนวน micro-step ต่อการอัปเดต 1 ครั้ง",
              "Steps = total optimizer updates · epochs = passes over the dataset · grad_accum = micro-steps per update."
            ),
          },
          {
            code: "steps ≈ (ผลรวม repeats ของทุกรูป × epochs) / grad_accum\nsteps ≈ (sum of repeats × epochs) / grad_accum",
          },
          {
            table: {
              head: [T("ประเภท", "Kind"), T("Steps แนะนำ", "Recommended steps")],
              rows: [
                [T("Character", "Character"), T("1,000–1,500", "1,000–1,500")],
                [T("Style", "Style"), T("1,500–2,500", "1,500–2,500")],
                [T("Outfit", "Outfit"), T("800–1,200", "800–1,200")],
                [T("Pose", "Pose"), T("1,200–2,000", "1,200–2,000")],
                [T("SANA official", "SANA official"), T("500 (rank 4 — เบามาก)", "500 (rank 4 — very light)")],
              ],
            },
          },
          {
            list: [
              T(
                "steps มากเกิน: overfit (สีเพี้ยน จำภาพต้นฉบับ ตอบพรอมป์ตอื่นไม่ได้)",
                "Too many steps: overfit (wrong colors, copies training images, ignores other prompts)"
              ),
              T(
                "บันทึก checkpoint ทุก 100–250 step แล้วเลือก step ที่ดีที่สุด — ไม่จำเป็นต้องใช้ step สุดท้าย",
                "Save a checkpoint every 100–250 steps and pick the best one — the last step is not always best"
              ),
            ],
          },
        ],
      },

      {
        id: "lr",
        icon: "📉",
        title: T("Learning rate (LR)", "Learning rate (LR)"),
        blocks: [
          {
            table: {
              head: [T("ตระกูล", "Family"), T("LR แนะนำ", "Recommended LR"), T("ช่วงที่ใช้ได้", "Usable range")],
              rows: [
                [T("SANA", "SANA"), T("1e-4 – 2e-4", "1e-4 – 2e-4"), T("5e-5 – 3e-4", "5e-5 – 3e-4")],
                [T("SDXL", "SDXL"), T("1e-4", "1e-4"), T("5e-5 – 2e-4", "5e-5 – 2e-4")],
              ],
            },
          },
          {
            list: [
              T(
                "สูงเกิน: loss แกว่ง ภาพเพี้ยน overfit เร็ว",
                "Too high: loss oscillates, artifacts, fast overfit"
              ),
              T(
                "ต่ำเกิน: เรียนช้า ต้องเพิ่ม step อาจเรียนไม่ทัน",
                "Too low: slow learning, needs more steps, may never converge"
              ),
              T(
                "rank สูง/attn_ff: ใช้ช่วงล่างของ LR",
                "High rank / attn_ff: use the lower end of the LR range"
              ),
              T(
                "ระบบจำกัด LR ไว้ที่ 1e-5 – 5e-4",
                "The engine clamps LR to 1e-5 – 5e-4"
              ),
            ],
          },
        ],
      },

      {
        id: "schedule",
        icon: "📈",
        title: T("LR schedule + Warmup", "LR schedule + warmup"),
        blocks: [
          {
            table: {
              head: [T("ค่า", "Value"), T("พฤติกรรม", "Behavior"), T("เหมาะกับ", "Best for")],
              rows: [
                [
                  T("Cosine + warmup", "Cosine + warmup"),
                  T("ค่อย ๆ ขึ้นแล้วลดลงแบบ cosine", "Ramps up, then decays as a cosine"),
                  T("งานจริงเกือบทั้งหมด", "Almost all real runs"),
                ],
                [
                  T("Constant", "Constant"),
                  T("LR คงที่ตลอด", "Constant LR throughout"),
                  T("run สั้น ทดลอง SANA official", "Short runs, experiments, SANA official"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "Warmup ratio (0–0.2, ค่าเริ่มต้น 0.03) = สัดส่วน step แรกที่ค่อย ๆ เพิ่ม LR",
                "Warmup ratio (0–0.2, default 0.03) = share of the first steps used to ramp LR up"
              ),
              T(
                "dataset เล็ก/LR สูง → เพิ่ม warmup เป็น 0.05–0.1 เพื่อกัน loss ระเบิด",
                "Small dataset / high LR → raise warmup to 0.05–0.1 to avoid an early loss spike"
              ),
            ],
          },
        ],
      },

      {
        id: "sampling",
        icon: "🎚️",
        title: T("Timestep sampling", "Timestep sampling"),
        blocks: [
          {
            table: {
              head: [T("ค่า", "Value"), T("ทำอะไร", "What it does"), T("เหมาะกับ", "Best for")],
              rows: [
                [
                  T("Logit-normal (ค่าเริ่มต้น)", "Logit-normal (default)"),
                  T("สุ่ม timestep เน้นช่วงกลาง", "Samples timesteps weighted toward the middle"),
                  T("งานจริง เรียนโครงสร้าง+รายละเอียดสมดุล", "Real runs; balances structure and detail"),
                ],
                [
                  T("Shift / Uniform", "Shift / Uniform"),
                  T("สุ่มกระจายเท่ากัน", "Uniform spread"),
                  T("ทดลอง เทรนสั้น อ้างอิงสูตรเดิม", "Experiments, short runs, reference recipes"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "Logit-normal ให้ผลดีกว่าในจำนวน step ที่น้อย",
                "Logit-normal gives better results in fewer steps"
              ),
              T(
                "ถ้าผลออกมาแข็ง/แบน ลองสลับเป็น uniform เพื่อดูความต่าง",
                "If results look stiff/flat, try uniform to compare"
              ),
            ],
          },
        ],
      },

      {
        id: "ema",
        icon: "🧊",
        title: T("EMA weights", "EMA weights"),
        blocks: [
          {
            p: T(
              "เก็บค่าเฉลี่ยของน้ำหนัก LoRA ตลอดการเทรน แล้วบันทึกค่าเฉลี่ยเป็นผลลัพธ์",
              "Keeps a running average of LoRA weights and saves the average as the result."
            ),
          },
          {
            list: [
              T("ข้อดี: ผลเนียนขึ้น ลด noise ระหว่าง step", "Pro: smoother result, less step-to-step noise"),
              T(
                "ข้อเสีย: LoRA อ่อนลง (‖BA‖ ต่ำ) run สั้นจะได้ผลอ่อนเกิน",
                "Con: weaker LoRA (lower ‖BA‖); short runs end up too soft"
              ),
              T(
                "แนะนำ: เปิดเฉพาะ run ยาว ≥2,000 step · ปิดสำหรับ run สั้น/character ที่ต้องการ likeness แรง",
                "Recommendation: on for long runs ≥2,000 steps · off for short runs / strong character likeness"
              ),
              T(
                "EMA decay (0.9–0.9999, ค่าเริ่มต้น 0.999): ยิ่งสูงยิ่งเฉลี่ยนานขึ้น",
                "EMA decay (0.9–0.9999, default 0.999): higher averages over more steps"
              ),
            ],
          },
        ],
      },

      {
        id: "accum",
        icon: "🧱",
        title: T("Grad accumulation", "Grad accumulation"),
        blocks: [
          {
            p: T(
              "สะสม gradient หลาย micro-step ก่อนอัปเดต 1 ครั้ง ทำให้ได้ effective batch ใหญ่ขึ้นโดยใช้ VRAM เท่าเดิม",
              "Accumulates gradients over several micro-steps per update, giving a larger effective batch at the same VRAM."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: gradient เสถียรขึ้น เหมาะ dataset เล็กมากหรือ loss แกว่ง",
                "Pro: steadier gradients, useful for tiny datasets or oscillating loss"
              ),
              T(
                "ข้อเสีย: ช้าลงตามจำนวนเท่า (1 step ใช้เวลานานขึ้น)",
                "Con: proportionally slower (each step takes longer)"
              ),
              T(
                "แนะนำ: 1 (ปกติ) · 2 เมื่อ dataset เล็ก/loss แกว่ง · ไม่เกิน 4 บนการ์ด 10 GB",
                "Recommendation: 1 (normal) · 2 for tiny datasets/oscillating loss · max 4 on a 10 GB card"
              ),
            ],
          },
        ],
      },

      {
        id: "resolution",
        icon: "🖼️",
        title: T("Resolution", "Resolution"),
        blocks: [
          {
            p: T(
              "ขนาดเป้าหมายของ bucket (ไม่ได้บังคับภาพเป็นจัตุรัสเมื่อเปิด bucketing)",
              "Target bucket size (images are not forced square while bucketing is on)."
            ),
          },
          {
            table: {
              head: [T("ตระกูล", "Family"), T("ตัวเลือก", "Options"), T("แนะนำ", "Recommendation")],
              rows: [
                [T("SANA", "SANA"), T("512 / 768 / 1024", "512 / 768 / 1024"), T("1024", "1024")],
                [T("SDXL", "SDXL"), T("512 / 768 / 1024", "512 / 768 / 1024"), T("768 (1024 ถ้า VRAM พอ)", "768 (1024 if VRAM allows)")],
              ],
            },
          },
          {
            list: [
              T(
                "สูง: รายละเอียดดี แต่ใช้ VRAM/เวลามาก",
                "Higher: more detail, more VRAM and time"
              ),
              T(
                "ต่ำ: เร็ว แต่ใบหน้า/ผิวอาจไม่คม",
                "Lower: faster, but faces/skin may be soft"
              ),
            ],
          },
        ],
      },

      {
        id: "seed",
        icon: "🌱",
        title: T("Seed", "Seed"),
        blocks: [
          {
            list: [
              T("-1 = สุ่มใหม่ทุกครั้ง", "-1 = a new random seed every run"),
              T(
                "ใส่ค่าคงที่เมื่อต้องการเทียบผลระหว่างการทดลอง (เช่น เปลี่ยน LR/step)",
                "Use a fixed value to compare experiments (e.g. changing LR/steps)"
              ),
              T(
                "seed มีผลกับลำดับการสุ่มรูป timestep และ noise — ไม่ใช่คุณภาพโดยตรง",
                "The seed controls image order, timesteps and noise — not quality directly"
              ),
            ],
          },
        ],
      },

      {
        id: "enhance",
        icon: "✨",
        title: T("Prompt enhance ตอนเทรน", "Prompt enhance during training"),
        blocks: [
          {
            p: T(
              "ใช้ตัวขยายพรอมป์ตตัวเดียวกับตอน generate ก่อน encode caption",
              "Runs the same prompt enhancer used for generation before encoding captions."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: caption ยาว/ละเอียดขึ้น อาจช่วย composition",
                "Pro: longer, more detailed captions that can help composition"
              ),
              T(
                "ข้อเสีย: caption ไม่ตรงกับที่เขียน 100% เพิ่มเวลาช่วงเตรียม และ debug ยาก",
                "Con: captions no longer match what you wrote, adds preparation time, harder to debug"
              ),
              T(
                "แนะนำ: เปิดถ้า caption สั้น · ปิดเมื่อต้องการควบคุมคำทุกคำหรือกำลังไล่ปัญหา caption",
                "Recommendation: on for short captions · off when every word matters or you are debugging captions"
              ),
            ],
          },
        ],
      },

      {
        id: "flip",
        icon: "↔️",
        title: T("สุ่มพลิกภาพซ้าย-ขวา", "Random left-right flip"),
        blocks: [
          {
            list: [
              T(
                "ข้อดี: เพิ่มความหลากหลาย กัน overfit เมื่อรูปน้อย",
                "Pro: more variety, less overfit on small datasets"
              ),
              T(
                "ข้อเสีย: อันตรายกับลักษณะไม่สมมาตร (ไฝ ผมแสกข้าง เครื่องหมาย) และข้อความในภาพ",
                "Con: harmful for asymmetric features (moles, side partings, marks) and text in images"
              ),
              T(
                "แนะนำ: เปิดเฉพาะ style · ปิดสำหรับ character/outfit ที่มีรายละเอียดข้างชัดเจน",
                "Recommendation: on for style only · off for character/outfit with clear side details"
              ),
            ],
          },
        ],
      },

      {
        id: "checkpoints",
        icon: "💾",
        title: T("Checkpoints (save every N steps)", "Checkpoints (save every N steps)"),
        blocks: [
          {
            p: T(
              "บันทึก LoRA ระหว่างเทรนเป็นไฟล์ ชื่อ-stepN.safetensors · 0 = บันทึกเฉพาะไฟล์สุดท้าย",
              "Saves the LoRA during training as name-stepN.safetensors · 0 = only the final file."
            ),
          },
          {
            list: [
              T(
                "แนะนำ: 100–250 สำหรับ run สั้น · 250–500 สำหรับ run ยาว",
                "Recommendation: 100–250 for short runs · 250–500 for long runs"
              ),
              T(
                "ระบบไม่เขียนทับไฟล์ที่มีอยู่แล้ว",
                "Existing files are never overwritten"
              ),
              T(
                "ไฟล์ checkpoint ทุกไฟล์มี .trainstate.pt คู่กัน (optimizer, scheduler, RNG, EMA, step, loss)",
                "Each checkpoint has a matching .trainstate.pt (optimizer, scheduler, RNG, EMA, step, loss)"
              ),
            ],
          },
        ],
      },

      {
        id: "resume",
        icon: "⏯️",
        title: T("Resume จาก optimizer state", "Resume from optimizer state"),
        blocks: [
          {
            p: T(
              "เปิดสวิตช์ Resume + เลือก checkpoint ที่มี .trainstate.pt → เทรนต่อจาก step เดิมพร้อม LR schedule/optimizer เดิม",
              "Turn on Resume + pick a checkpoint with a .trainstate.pt → training continues from that step with the same LR schedule and optimizer."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: ไม่ต้องเริ่มนับ step ใหม่ สถานะต่อเนื่อง ให้ผลเหมือนเทรนยาวครั้งเดียว",
                "Pro: step count continues and state stays consistent, matching one long run"
              ),
              T(
                "ข้อเสีย: ต้องมีไฟล์ state และ rank ตรงกัน; ถ้าไม่มีจะเริ่มจาก 0",
                "Con: needs the state file and a matching rank; otherwise it starts from 0"
              ),
              T(
                "แนะนำ: เปิดเมื่อเทรนต่อจาก checkpoint เดิม · ปิดเมื่อเริ่ม LoRA ใหม่",
                "Recommendation: on when continuing a checkpoint · off for a brand-new LoRA"
              ),
            ],
          },
        ],
      },

      {
        id: "continue",
        icon: "🔁",
        title: T("Continue from LoRA (init_lora)", "Continue from LoRA (init_lora)"),
        blocks: [
          {
            p: T(
              "โหลดน้ำหนัก LoRA เดิมเป็นจุดเริ่มต้นก่อนเทรน เหมาะกับการ refine LoRA ที่มีอยู่",
              "Loads an existing LoRA as the starting point; useful for refining an adapter you already have."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: ต่อยอดสไตล์/ตัวตนเดิม ใช้ step น้อยลง",
                "Pro: extends an existing style/identity with fewer steps"
              ),
              T(
                "ข้อเสีย: ถ้า LoRA เดิม overfit อยู่แล้วจะยิ่ง overfit · rank ถูกบังคับตามไฟล์เดิม",
                "Con: an already-overfit LoRA gets worse · rank is forced to the file's rank"
              ),
              T(
                "แนะนำ: ตั้งชื่อผลลัพธ์ใหม่เสมอ · ถ้าต้องการเทรนต่อเนื่องจริง ๆ ให้เปิด Resume ด้วย",
                "Recommendation: always use a new output name · enable Resume for a true continuation"
              ),
            ],
          },
        ],
      },

      {
        id: "thermal",
        icon: "🌡️",
        title: T("Thermal guard (ป้องกัน GPU ร้อน)", "Thermal guard"),
        blocks: [
          {
            table: {
              head: [T("โหมด", "Mode"), T("ทำอะไร", "What it does"), T("ข้อดี", "Pros"), T("ข้อเสีย", "Cons")],
              rows: [
                [
                  T("Throttle (แนะนำ)", "Throttle (recommended)"),
                  T("ลด clock ลงทีละระดับ 5% สูงสุด 25%", "Drops clocks 5% per level, max 25%"),
                  T("เทรนไม่หยุด อุณหภูมินิ่ง", "Training never stops, temperature plateaus"),
                  T("ช้าลงเล็กน้อย", "Slightly slower"),
                ],
                [
                  T("Pause", "Pause"),
                  T("หยุดพักจนเย็นถึง threshold", "Pauses until it cools to the threshold"),
                  T("เย็นกว่า", "Cooler"),
                  T("step ใช้เวลานานขึ้นมาก", "Steps take much longer"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "เริ่มชะลอเมื่อ 87°C · กลับมาเต็ม speed 80°C · หยุดฉุกเฉิน 90°C (ปรับได้ในกรอบปลอดภัย)",
                "Throttle at 87°C · full speed at 80°C · emergency pause at 90°C (adjustable within safe bounds)"
              ),
              T(
                "ถ้าไม่มีสิทธิ์ sudo ระบบจะหน่วงระหว่าง step แทนการปรับ clock",
                "Without sudo the engine sleeps between steps instead of changing clocks"
              ),
              T(
                "แนะนำ: เปิด throttle ไว้เสมอ · อย่าตั้ง max เกิน 90°C",
                "Recommendation: always keep throttle on · never set max above 90°C"
              ),
            ],
          },
        ],
      },

      {
        id: "sample",
        icon: "🧪",
        title: T("สร้างภาพทดสอบหลังเทรน", "Sample generation after training"),
        blocks: [
          {
            p: T(
              "ส่งงาน generate ปกติโดยใช้ LoRA ที่เลือก: batch 4, seed คงที่ 12345, พรอมป์ต = trigger ของ dataset ถ้าเว้นว่าง",
              "Queues a normal generation job with the selected LoRA: batch 4, fixed seed 12345, prompt defaults to the dataset trigger."
            ),
          },
          {
            list: [
              T(
                "ข้อดี: เห็นผลจริงทันทีโดยไม่โหลดโมเดลซ้ำ และเทียบ checkpoint ด้วย seed เดียวกันได้",
                "Pro: immediate real output without reloading models, and checkpoints compare fairly at the same seed"
              ),
              T(
                "ข้อเสีย: ใช้เวลาคิวปกติ (ไม่ใช่ instant)",
                "Con: takes normal queue time (not instant)"
              ),
              T(
                "วิธีใช้: เลือก LoRA → ใส่พรอมป์ตหรือเว้นว่าง → กดสร้างภาพทดสอบ",
                "How: pick a LoRA → type a prompt or leave it empty → Generate test images"
              ),
              T(
                "แนะนำ: สร้าง 2–3 checkpoint แล้วเทียบที่ seed เดียวกัน",
                "Recommendation: sample 2–3 checkpoints at the same seed"
              ),
            ],
          },
        ],
      },

      {
        id: "presets",
        icon: "🧾",
        title: T("Presets และ Config export/import", "Presets and config export/import"),
        blocks: [
          {
            table: {
              head: [T("Preset", "Preset"), T("Rank", "Rank"), T("Steps", "Steps"), T("LR", "LR"), T("หมายเหตุ", "Notes")],
              rows: [
                [
                  T("Character", "Character"),
                  T("32", "32"),
                  T("1,200", "1,200"),
                  T("1e-4", "1e-4"),
                  T("bucketing เปิด shuffle/min-SNR ปิด", "bucketing on, shuffle/min-SNR off"),
                ],
                [
                  T("Style", "Style"),
                  T("32", "32"),
                  T("2,000", "2,000"),
                  T("1.5e-4", "1.5e-4"),
                  T("shuffle เปิด noise 0.02 min-SNR 5", "shuffle on, noise 0.02, min-SNR 5"),
                ],
                [
                  T("Outfit", "Outfit"),
                  T("16", "16"),
                  T("900", "900"),
                  T("1e-4", "1e-4"),
                  T("เหมาะชุด/ไอเทม", "For outfits/items"),
                ],
                [
                  T("Fast test", "Fast test"),
                  T("16", "16"),
                  T("300", "300"),
                  T("1.5e-4", "1.5e-4"),
                  T("ทดสอบ pipeline เร็ว ๆ", "Quick pipeline test"),
                ],
              ],
            },
          },
          {
            list: [
              T(
                "preset เป็นจุดเริ่มต้น ไม่ใช่ค่าตายตัว",
                "A preset is a starting point, not a fixed rule"
              ),
              T(
                "เลือก SDXL แล้ว preset จะปรับเป็น 768px และเพิ่ม step 1.5×",
                "With SDXL the preset switches to 768px and multiplies steps by 1.5"
              ),
              T(
                "ส่งออก config → JSON เก็บค่าทั้งหมด (ยกเว้น dataset) · นำเข้า config → ใช้ค่าเดิมกับ dataset ใหม่",
                "Export config → JSON with every setting (except the dataset) · Import config → reuse it on another dataset"
              ),
            ],
          },
        ],
      },

      {
        id: "families",
        icon: "🧬",
        title: T("SANA vs SDXL", "SANA vs SDXL"),
        blocks: [
          {
            table: {
              head: [T("หัวข้อ", "Topic"), T("SANA 1.5", "SANA 1.5"), T("SDXL", "SDXL")],
              rows: [
                [T("VRAM ตอนเทรน", "Training VRAM"), T("ต่ำกว่า", "Lower"), T("สูงกว่า", "Higher")],
                [T("ความเร็ว", "Speed"), T("เร็วกว่า", "Faster"), T("ช้ากว่า", "Slower")],
                [T("Rank แนะนำ", "Recommended rank"), T("32", "32"), T("16–32", "16–32")],
                [T("Resolution", "Resolution"), T("1024", "1024"), T("768 (หรือ 1024 ถ้าไหว)", "768 (or 1024 if it fits)")],
                [T("LR", "LR"), T("1e-4 – 2e-4", "1e-4 – 2e-4"), T("1e-4", "1e-4")],
                [T("Steps", "Steps"), T("1,000–2,500", "1,000–2,500"), T("1,000–2,000", "1,000–2,000")],
                [T("Bucketing", "Bucketing"), T("dynamic RoPE รองรับเอง", "native dynamic RoPE"), T("time_ids ต่อรูป (ระบบทำให้)", "per-image time_ids (automatic)")],
                [T("attn_ff", "attn_ff"), T("conv modules", "conv modules"), T("linear FF modules", "linear FF modules")],
                [T("ขอบเขตเทรน", "Trains"), T("transformer เท่านั้น", "Transformer only"), T("UNet เท่านั้น", "UNet only")],
              ],
            },
          },
          {
            note: {
              tone: "warn",
              text: T(
                "ห้ามโหลดสอง pipeline พร้อมกันบนการ์ด 10 GB — ระบบปล่อยโมเดล generate ก่อนเทรนให้อัตโนมัติ",
                "Never load two pipelines at once on a 10 GB card — the engine unloads the generation model before training"
              ),
            },
          },
        ],
      },

      {
        id: "recipes",
        icon: "🍳",
        title: T("สูตรตั้งค่า (คัดลอกใช้ได้)", "Recipes (copy-ready)"),
        blocks: [
          {
            p: T("Character LoRA (SANA)", "Character LoRA (SANA)"),
          },
          {
            code:
              "kind: character\nrank: 32\nsteps: 1200–1500\nlr: 1e-4\nresolution: 1024\nscheduler: cosine, warmup 0.03\ntimestep: logit-normal\nbucketing: on\ncaption dropout: 0.05\ncaption shuffle: off\nkeep_tokens: 1\nnoise offset: 0\nmin-SNR: 0\nrsLoRA: off\ntarget: attn\nEMA: off\nflip: off\nsave_every: 200",
          },
          { p: T("Style LoRA (SANA)", "Style LoRA (SANA)") },
          {
            code:
              "kind: style\nrank: 32\nsteps: 2000\nlr: 1.5e-4\nresolution: 1024\nbucketing: on\ncaption dropout: 0.03\ncaption shuffle: on, keep_tokens 1\nnoise offset: 0.02\nmin-SNR: 5\ntarget: attn\nflip: on\nsave_every: 250",
          },
          { p: T("SDXL Character", "SDXL Character") },
          {
            code:
              "kind: character\nrank: 16–32\nsteps: 1000–1500\nlr: 1e-4\nresolution: 768\nscheduler: cosine\nbucketing: on\ncaption dropout: 0.05\nmin-SNR: 0\ntarget: attn\nEMA: off\nsave_every: 200",
          },
        ],
      },

      {
        id: "troubleshoot",
        icon: "🩺",
        title: T("ปัญหาที่พบบ่อย", "Troubleshooting"),
        blocks: [
          {
            table: {
              head: [T("อาการ", "Symptom"), T("สาเหตุ", "Cause"), T("แก้", "Fix")],
              rows: [
                [
                  T("ภาพไม่เหมือนตัวละคร", "Output does not look like the character"),
                  T("รูปน้อย/มุมซ้ำ/caption บรรยายลักษณะถาวร", "Few images, repeated angles, permanent traits in captions"),
                  T("เพิ่มรูปหลายมุม ย้ายลักษณะถาวรออกจาก caption เพิ่ม steps", "Add angles, move permanent traits out of captions, add steps"),
                ],
                [
                  T("LoRA ตอบ trigger ไม่แม่น", "LoRA ignores the trigger"),
                  T("dropout สูง steps น้อย trigger ซ้ำคำธรรมดา", "High dropout, few steps, common-word trigger"),
                  T("ลด dropout เพิ่ม steps เปลี่ยน trigger", "Lower dropout, add steps, change the trigger"),
                ],
                [
                  T("ภาพแข็ง/สีเพี้ยน/จำต้นฉบับ", "Stiff, wrong colors, copies training images"),
                  T("overfit", "Overfit"),
                  T("ลด steps/LR ใช้ checkpoint ก่อนหน้า เพิ่ม dropout เล็กน้อย", "Lower steps/LR, use an earlier checkpoint, raise dropout slightly"),
                ],
                [
                  T("ใบหน้าไม่คม", "Soft faces"),
                  T("resolution/rank ต่ำ", "Low resolution or rank"),
                  T("เพิ่ม resolution/rank เปิด attn_ff", "Raise resolution/rank, enable attn_ff"),
                ],
                [
                  T("ภาพแนวตั้งถูกครอป", "Portrait images get cropped"),
                  T("bucketing ปิด", "Bucketing off"),
                  T("เปิด bucketing", "Enable bucketing"),
                ],
                [
                  T("ภาพมืด/contrast เพี้ยน", "Dark or broken contrast"),
                  T("noise offset สูงเกิน", "Noise offset too high"),
                  T("ลดเหลือ 0–0.02", "Lower to 0–0.02"),
                ],
                [
                  T("loss แกว่ง", "Loss oscillates"),
                  T("LR สูง/dataset เล็ก", "High LR / small dataset"),
                  T("ลด LR เพิ่ม warmup grad accum 2", "Lower LR, raise warmup, grad accum 2"),
                ],
                [
                  T("เทรนช้ามาก", "Training is very slow"),
                  T("grad accum/attn_ff/thermal throttle", "Grad accum / attn_ff / thermal throttle"),
                  T("ลด grad accum ใช้ attn ตรวจอุณหภูมิ", "Lower grad accum, use attn, check temperature"),
                ],
                [
                  T("resume ไม่ต่อ step", "Resume restarts at 0"),
                  T("ไม่มี .trainstate.pt หรือ rank ไม่ตรง", "No .trainstate.pt or rank mismatch"),
                  T("ใช้ checkpoint ที่มี state ตั้ง rank ให้ตรง", "Use a checkpoint with state and the matching rank"),
                ],
              ],
            },
          },
        ],
      },

      {
        id: "checklist",
        icon: "✅",
        title: T("เช็กลิสต์ก่อนกดเริ่มเทรน", "Pre-flight checklist"),
        blocks: [
          {
            list: [
              T("โมเดลฐานเป็นตระกูลที่ต้องการ (SANA/SDXL)", "Base model is the right family (SANA/SDXL)"),
              T("dataset มีรูปพอ (character ≥16 รูปเดี่ยว)", "Enough images (character ≥16 solo)"),
              T("ทุกรูปมี caption หรือ tags", "Every image has a caption or tags"),
              T("trigger ไม่ซ้ำคำทั่วไป และปรากฏใน caption (หรือปล่อยให้ระบบเติม)", "Trigger is rare and present in captions (or auto-added)"),
              T("ตรวจ QC แล้วไม่มีไฟล์ซ้ำ/รูปเล็กผิดปกติ", "QC shows no duplicates or abnormal sizes"),
              T("ตั้งชื่อ LoRA ผลลัพธ์ใหม่ (ไม่ซ้ำของเดิม)", "New output LoRA name (not an existing one)"),
              T("เปิด bucketing", "Bucketing on"),
              T("ตั้ง checkpoint ทุก 100–250 step", "Checkpoints every 100–250 steps"),
              T("เปิด thermal guard (throttle)", "Thermal guard on (throttle)"),
              T("รู้ว่าจะเทียบผลด้วย seed คงที่", "You know which fixed seed you will compare with"),
            ],
          },
        ],
      },

      {
        id: "after",
        icon: "🏁",
        title: T("หลังเทรน", "After training"),
        blocks: [
          {
            steps: [
              T("ดูรายการ LoRA → เลือก checkpoint ที่ต้องการ", "Open the LoRA list → pick a checkpoint"),
              T(
                "กดสร้างภาพทดสอบด้วย seed เดียวกัน 2–3 checkpoint",
                "Generate samples for 2–3 checkpoints at the same seed"
              ),
              T(
                "เทียบ: ความเหมือน การตอบพรอมป์ต และการไม่ overfit (ลองพรอมป์ตฉากใหม่)",
                "Compare likeness, prompt adherence and overfit (try a new scene prompt)"
              ),
              T(
                "ถ้ายังไม่พอ: เพิ่ม dataset (ไม่ใช่เพิ่ม steps อย่างเดียว)",
                "If it is not enough: improve the dataset, not just the step count"
              ),
              T(
                "เก็บ config ที่ใช้ได้ → export ไว้เป็น recipe",
                "Export the config that worked as a recipe"
              ),
            ],
          },
        ],
      },
    ],
  };
})();
