# GPU หลุดจากบัส (Xid 79) — P102-100

> **English TL;DR** — `NVRM: Xid 79, GPU has fallen off the bus` means the GPU
> stopped answering PCIe. It is a **link/hardware** event, not a driver install
> problem and not OOM. `apt-get purge '*nvidia*'` does **not** touch this
> driver (it is a `.run` + DKMS build, not a dpkg package) — every apparent
> "fix" was the reboot that followed. Recover with a **reboot**. Reduce risk by
> lowering the power limit, disabling `nvidia-drm` modeset, and keeping the
> GPU off the desktop's X server.
>
> ตรวจสถานะเครื่องด้วย `bash scripts/gpu-status.sh` (อ่านอย่างเดียว ไม่ต้อง sudo)

เอกสารนี้อธิบายอาการ, หลักฐาน, สาเหตุที่เป็นไปได้ และสิ่งที่ทำได้จริง
เขียนจากเหตุการณ์จริงบนเครื่องนี้ (ASRock A320M-HDV R4.0 + Ryzen 5 1400 +
P102-100 10 GB)

---

## 1. อาการ

- ระหว่างใช้งาน (หรือแม้แต่ตอนว่าง) จอภาพค้าง/ดับ แล้วเดสก์ท็อปตายทั้งหน้าจอ
- `dmesg` / `journalctl -k` ขึ้น:

  ```
  NVRM: Xid (PCI:0000:07:00): 79, GPU has fallen off the bus.
  NVRM: Xid (PCI:0000:07:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)
  ```

- ตามด้วย `nvidia-drm ... Removing device`, Xorg `SIGABRT` / `FatalError`,
  และโปรเซสที่ใช้ CUDA (Ollama, engine) ขึ้น
  `CUDA error: invalid device ordinal` หรือ `no CUDA-capable device`
- `nvidia-smi` หลังจากนั้นตอบไม่ได้ หรือไม่เห็นการ์ด
- ต้องรีบูตเท่านั้นถึงจะกลับมา

Xid 79 แปลตรงตัวว่า **การ์ดไม่ตอบสนองต่อการอ่าน/เขียนผ่าน PCIe แล้ว**
ฝั่งไดรเวอร์ทำอะไรไม่ได้นอกจากถอดอุปกรณ์ออกจากระบบ

---

## 2. หลักฐานบนเครื่องนี้

### 2.1 เวลาที่เกิด Xid 79

| เวลา | หมายเหตุ |
|---|---|
| 2026-11-08 17:16:42 | เกิดครั้งแรก |
| 2026-11-11 14:29:54 | **ตอนว่าง** — มีแค่ Ollama poll `/api/tags` ไม่มีการสร้างภาพ |
| 2026-11-11 14:58:40 | |
| 2026-11-11 15:08:52 | |
| 2026-11-12 13:42:14 | |
| _2026-11-13 03:16_ | _ไม่ใช่ Xid 79 — Update Manager ติดตั้ง kernel 7.0.0-43 + `linux-firmware` กลับมา_ |

จุดสำคัญ: **14:29:54 เกิดตอนการ์ดว่าง** ไม่มีงาน generate/train
จึงตัดสาเหตุเรื่องโหลดหนัก, VRAM เต็ม, หรือ OOM ออกได้

สถานะล่าสุด (2026-11-13 14:30): หลัง reboot ครั้งสุดท้ายเมื่อ 11-12 13:49
**ไม่เกิด Xid 79 อีกเลย** และตอน reboot 11-13 12:56
ระบบบูตเข้า kernel **7.0.0-43** โดย DKMS build ไดรเวอร์ 580.196.07
สำหรับ kernel ใหม่ได้สำเร็จ (`dkms status` มีทั้ง 42 และ 43)
· ตรึง kernel ครบแล้ว **7/7 แพ็กเกจ** (`apt-mark showhold`)

### 2.2 ลำดับเหตุการณ์ 2026-11-12

```
13:41:57  ollama: offload 33/33 layers ไป CUDA0 (predicted 4.5 GiB)
13:42:14  NVRM: Xid 79 — GPU has fallen off the bus
13:42:38  nvidia-drm: Removing device (non-zero usage count)
13:43:32  Xorg: SIGABRT / FatalError  ← เดสก์ท็อปตายตรงนี้
13:44:36  ollama: CUDA error: invalid device ordinal
13:45:55  ผู้ใช้รัน apt-get purge '*nvidia*'
13:47:55  reboot
13:49:16  กลับมาทำงาน (การ์ดติดตั้งใหม่ตอน boot)
```

จะเห็นว่า **การ purge เกิดหลังการ์ดหลุดไปแล้ว** และสิ่งที่ทำให้กลับมาคือ reboot

### 2.3 `apt-get purge '*nvidia*'` ลบอะไรจริง

`/var/log/apt/history.log` แสดงว่า purge ทั้งสองครั้ง (11-11 16:15:31 และ
11-12 13:45:55) ลบแพ็กเกจเดียวกันเสมอ:

```
linux-firmware-nvidia-graphics
linux-firmware
linux-image-generic-7.0
linux-generic-7.0
```

**ไม่มีตัวใดเป็นไดรเวอร์** ทั้งสี่ตัวเป็น firmware ของ nouveau และ meta-package
ของเคอร์เนล

ไดรเวอร์ที่ใช้จริงคือ:

```
/lib/modules/7.0.0-42-generic/updates/dkms/nvidia.ko   (580.196.07)
/usr/src/nvidia-580.196.07/                            (source ที่ patch แล้ว)
```

ติดตั้งด้วย `nvidia-installer` แบบ `.run` + DKMS → **ไม่ได้อยู่ใน dpkg**
ดังนั้น `apt purge` มองไม่เห็นและลบไม่ได้

และ `/var/log/nvidia-installer.log` (2026-11-11 16:19) จบด้วย
`Abort` — "There appears to already be a driver installed"
แปลว่า **ไม่มีการติดตั้งไดรเวอร์ใหม่เกิดขึ้นเลย**

> สรุป: ทุกครั้งที่ "purge แล้วหาย" จริง ๆ คือ **reboot ที่ตามมา**
> ซึ่งไป enumerate อุปกรณ์ PCIe ใหม่ การ purge ไม่ได้แก้และไม่ได้ทำอะไรกับไดรเวอร์

> หมายเหตุ: ทั้ง 4 แพ็กเกจกลับมาติดตั้งแล้วเมื่อ 2026-11-13 03:16
> (Update Manager) พร้อม kernel 7.0.0-43 — ยืนยันว่า purge เป็นเพียง
> ผลข้างเคียง ไม่ใช่กลไกการกู้คืน

### 2.4 ทำไมจอถึงตาย ไม่ใช่แค่ GPU ใช้งานไม่ได้

`/usr/share/X11/xorg.conf.d/nvidia-drm-outputclass.conf` (จาก `.run`) และ
`11-nvidia-offload.conf` (gpu-manager) ทำให้ Xorg โหลด `nvidia_drv.so`
ผูกกับ `/dev/dri/card2` ของ P102-100:

```
(II) Applying OutputClass "nvidia" to /dev/dri/card2
(II) Loading /usr/lib/xorg/modules/drivers/nvidia_drv.so
```

`xrandr --listproviders` เห็น 2 provider (`AMD Radeon R7 200` + `NVIDIA-G0`)
เมื่อการ์ดหลุด Xorg จึง abort ทั้งเซิร์ฟเวอร์ → จอดำ

ผลข้างเคียงอีกอย่าง: Xorg abort ทุก boot ราว 8 วินาทีแรก ด้วย
`no screens found` เพราะแย่งกับ udev ที่โหลด nvidia module ช้า แล้วค่อย respawn
(`modeset=0` ช่วยตัดปัญหานี้ด้วย)

### 2.5 สภาพลิงก์ PCIe

```
GPU 0000:07:00.0  ใต้ root port 0000:00:03.1
PCIe Generation: Max 1 / Current 1
Link Width:      Max 4x / Current 4x
```

- ช่อง x16 (`0000:00:01.1`) ถูกจอแสดงผล AMD Radeon R7 200 ใช้อยู่
- P102-100 อยู่ที่ root port `00:03.1` ซึ่งเป็น **GPP x4 ของ CPU** = ช่อง M.2
  บนบอร์ด A320M-HDV R4.0 → แปลว่า **ต้องมี adapter M.2 → PCIe x16**
- การ์ดตั้งไว้ 250 W (default), ขั้นต่ำ 125 W, สูงสุด 300 W
- ไม่มี `AER` / `pcieport` error ใน log — อาการเป็น link drop เงียบ ๆ
  ซึ่งเข้ากับปัญหา **ไฟฟ้า/หน้าสัมผัส/สัญญาณ** มากกว่าฝั่งซอฟต์แวร์

### 2.6 สิ่งที่ลองแล้วไม่ช่วย

- `pcie_aspm=off processor.max_cstate=1` ใน kernel cmdline (แก้ 2026-11-10)
  → ยังหลุดใน 11-11 และ 11-12
- purge + ติดตั้งใหม่ → ไม่ได้แตะไดรเวอร์ (ดู 2.3)

---

## 3. สาเหตุที่เป็นไปได้ (เรียงตามความน่าจะเป็น)

1. **PCIe link ไม่เสถียรทางกายภาพ/ไฟฟ้า** — M.2 → x16 adapter, riser,
   หน้าสัมผัสสกปรก, สาย 12 V ไม่แน่น, PSU จ่ายไฟไม่พอตอน transient
2. **ความร้อน/ไฟเลี้ยงการ์ด** — การ์ด Mining card ไม่มีพัดลม/ท่อลมเหมือนการ์ดปกติ
   VRM ร้อนง่ายเมื่อวิ่งใกล้ 250 W
3. **ไดรเวอร์/kernel** — เป็นไปได้น้อย เพราะเกิดตอนว่างและไม่มี AER error
   (แต่ kernel 7.0.0-42 + DKMS ยังเป็นชุดที่ใหม่มาก)

ไม่ใช่: OOM, VRAM เต็ม, โหลดสูง, หรือ "ไดรเวอร์เสียจากการ purge"

---

## 4. การกู้คืนเมื่อเกิดแล้ว

1. **รีบูต** — วิธีเดียวที่ได้ผลจริง
   ```bash
   sudo reboot
   ```
2. หลัง boot ตรวจว่าการ์ดกลับมา:
   ```bash
   nvidia-smi
   bash scripts/gpu-status.sh
   ```
3. **อย่า** purge หรือลงไดรเวอร์ใหม่ — เสียเวลาและเสี่ยงทำ `/usr/src` พัง
4. ปิด Ollama/engine ที่ค้างก่อนรีบูตก็ได้ แต่ไม่จำเป็น

ถ้าอยากได้ log ไว้วิเคราะห์ก่อนรีบูต:
```bash
journalctl -k -b -1 --no-pager | grep -E 'NVRM|Xid|nvidia-drm' | tail -30
```

---

## 5. มาตรการลดความเสี่ยง

ทำทีละข้อ เริ่มจากข้อที่ไม่ต้อง reboot ก่อน แล้วสังเกต 3–7 วัน

### 5.1 ลด power limit (แนะนำที่สุด, ไม่ต้อง reboot)

ใช้ **Settings → GPU power mode → Low temp (150 W)** ในแอป หรือตั้งถาวร:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 200        # 200 W = 80% ของ 250 W
```

ถ้าอยากให้ติดทุก boot ให้ทำ systemd unit (อย่าใช้พร้อมกับ power mode ของแอป
เพราะจะแย่งกันตั้งค่า):

```ini
# /etc/systemd/system/nvidia-powerlimit.service
[Unit]
Description=Cap NVIDIA GPU power limit
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 200
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-powerlimit.service
```

### 5.2 ปิด nvidia-drm modeset (ตัดไม่ให้ Xorg ผูกการ์ด)

CUDA ยังใช้ได้ปกติ แต่จะไม่มี `/dev/dri/card2` ให้ Xorg ไปจับ

```bash
echo 'options nvidia-drm modeset=0' | sudo tee /etc/modprobe.d/nvidia-drm-modeset.conf
sudo update-initramfs -u
sudo reboot
```

หลัง reboot ตรวจ:
```bash
ls /dev/dri/                 # ควรเหลือ card เฉพาะ AMD
DISPLAY=:0 xrandr --listproviders   # ควรเหลือ provider เดียว
```

ผลข้างเคียง: ถ้าวันหนึ่งอยากใช้ P102-100 เป็นจอแสดงผล จะต้องเอาออก

### 5.3 บังคับ Ollama ใช้ CPU

ลดการแตะ GPU ตอนไม่ได้สร้างภาพ (ซึ่งเป็นตอนที่เกิด Xid 79 ตอนว่าง)

```bash
sudo systemctl edit ollama
```

ใส่:
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES="
Environment="OLLAMA_VULKAN=false"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
ollama ps          # PROCESSOR ควรเป็น 100% CPU
```

ช้ากว่าเดิม แต่เสถียรกว่า และแอปยังสั่ง unload ก่อน generate/train อยู่แล้ว

### 5.4 ตรึงเคอร์เนล (ตอนนี้คือ 7.0.0-43)

กันไม่ให้ `apt upgrade` ดึง kernel รุ่นถัดไปมาแล้ว DKMS ต้อง build ใหม่
ทุกครั้ง (ซึ่งเสี่ยงทำให้ไดรเวอร์ที่ patch อยู่พัง)

> สถานะ 2026-11-13: Update Manager ติดตั้ง kernel **7.0.0-43** ไปแล้ว
> และ DKMS build ไดรเวอร์ 580.196.07 สำหรับ 43 สำเร็จ
> (`dkms status` = installed ทั้ง 42 และ 43) — ความเสี่ยงที่เคยเตือนไว้
> **ไม่เกิดขึ้นจริง** แต่ยังควรตรึงไว้เพื่อกันรอบถัดไป

```bash
# หนึ่งคำสั่งต่อบรรทัด — กันปัญหา paste แล้วมีช่องว่างนำหน้าชื่อแพ็กเกจ
sudo apt-mark hold linux-generic-7.0
sudo apt-mark hold linux-image-generic-7.0
sudo apt-mark hold linux-image-7.0.0-43-generic
sudo apt-mark hold linux-modules-7.0.0-43-generic
sudo apt-mark hold linux-headers-7.0.0-43-generic
sudo apt-mark hold linux-headers-generic-7.0
sudo apt-mark hold linux-headers-generic-hwe-24.04

apt-mark showhold
```

> อย่าใช้รูปแบบหลายบรรทัดที่ลงท้ายด้วย `\` แล้วมี indent — ถ้า paste
> แล้ว backslash หาย ช่องว่างนำหน้าจะกลายเป็นส่วนหนึ่งของชื่อแพ็กเกจ
> (`E: ไม่พบแพกเกจ  linux-generic-7.0`) ซึ่งไม่ใช่ปัญหาของ apt-mark


ข้อแลกเปลี่ยน: จะไม่ได้รับ kernel security update ใหม่จนกว่าจะเอา hold ออก
ถ้าจะอัปเกรดเอง ให้ปลด hold, ติดตั้ง, แล้ว reboot พร้อมเช็ก `dkms status`
ว่าขึ้น kernel ใหม่แล้วจริงก่อนใช้งาน

### 5.5 `linux-firmware` — คืนแล้ว

`linux-firmware` และ meta-package ของเคอร์เนลกลับมาติดตั้งแล้วเมื่อ
2026-11-13 03:16 พร้อม kernel 43 ไม่ต้องทำอะไรเพิ่ม

ตรวจซ้ำได้ด้วย:

```bash
dpkg -l linux-firmware linux-generic-7.0 linux-image-generic-7.0 | tail -n +6
```

> ทั้งสามตัว**ไม่เกี่ยวกับไดรเวอร์ NVIDIA** ที่ใช้ `.run` + DKMS
> การติดตั้งหรือถอนมันจึงไม่มีผลกับการ์ดเลย


### 5.6 ตรวจ hardware (สำคัญที่สุด)

- ยืนยันว่า P102-100 ต่อผ่าน **adapter M.2 → PCIe x16** จริงหรือไม่
  (ถ้าใช่ นี่คือจุดน่าสงสัยอันดับหนึ่ง)
- ถอด-เสียบการ์ดใหม่ ทำความสะอาดหน้าสัมผัส (ยางลบ/แอลกอฮอล์ IPA)
- ตรวจสายไฟ 12 V (PCIe 8-pin) ว่าแน่นและใช้สายแยก ไม่ใช่ Daisy-chain
- ตรวจ PSU ว่าจ่ายพอสำหรับ 250 W + การ์ดจอ + CPU
- ดูพัดลม/ท่อลมของการ์ด และอุณหภูมิ VRM
- ถ้ามีช่อง x16 ว่าง ให้สลับการ์ดไปช่อง x16 โดยตรง (แต่ต้องย้ายจอ AMD
  ไปใช้ iGPU หรือช่องอื่น)

---

## 6. สิ่งที่ **ไม่** ควรทำ

| อย่าทำ | เพราะ |
|---|---|
| `sudo apt-get purge '*nvidia*'` | ไม่ได้แตะไดรเวอร์ และลบ kernel meta/firmware ทิ้ง |
| `sudo apt install nvidia-driver-*` | จะเขียนทับ `/usr/src/nvidia-580.196.07` ที่ patch ไว้ → การ์ดเป็น unknown GPU |
| `sudo apt install linux-generic-7.0` | ไม่ผิดแล้ว (ติดตั้งอยู่ที่ 43) แต่จะดึง kernel รุ่นถัดไป → ต้อง DKMS build ใหม่ทุกครั้ง |
| ถอดการ์ดออกแล้วใส่ใหม่ขณะเปิดเครื่อง | อันตรายต่อ hardware |
| ตั้ง `-pl 300` เพื่อ "ให้แรงขึ้น" | เพิ่มโอกาส link drop และร้อนเกิน |

ไฟล์ pin `/etc/apt/preferences.d/99-block-nvidia-driver` ยังทำงานอยู่
(`Pin-Priority: -1` กับ `nvidia-driver-*`, `nvidia-dkms-*`, `libnvidia-*`)
และตั้งใจ **ไม่** บล็อก `linux-firmware-nvidia-graphics` กับ `libnvidia-egl-*`
เพราะไม่ใช่ไดรเวอร์

---

## 7. เกณฑ์ตัดสินใจ

- ถ้าลดไฟแล้วหลุดน้อยลงชัดเจน → ไปทางไฟฟ้า/ความร้อน ให้ตรวจ hardware ต่อ
- ถ้ายังหลุดแม้ตอนว่างที่ 150 W → สงสัย adapter/หน้าสัมผัสเป็นหลัก
- ถ้าย้ายไปช่อง x16 แล้วหาย → จบที่ adapter M.2
- ถ้ายังหลุดแม้เปลี่ยนช่อง/เปลี่ยน adapter → อาจเป็นการ์ดเอง (VRM/GPU เสีย)

เก็บ log ทุกครั้งที่หลุดไว้เทียบ:
```bash
bash scripts/gpu-status.sh > ~/gpu-status-$(date +%F-%H%M).txt
```
