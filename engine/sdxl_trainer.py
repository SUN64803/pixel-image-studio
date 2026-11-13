#!/usr/bin/env python3
"""SDXL LoRA trainer — diffusion (epsilon / v-prediction) fine-tuning of the UNet.

Why this design (P102 10 GB / 15 GB RAM, no swap):
- SDXL's UNet is ~2.6 B parameters; full fine-tuning is impossible on 10 GB.
  Only PEFT LoRA adapters are injected into the attention projections
  (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``); everything else is frozen.
- The VAE and both text encoders are used once to cache latents / embeddings
  and then freed, so the training loop only holds the UNet + adapters.
- Gradient checkpointing + attention slicing keep activations inside 10 GB.
- The objective mirrors SDXL inference: DDPM noise schedule, the UNet predicts
  epsilon (or v), and the loss is MSE against the added noise.
- Output uses diffusers SDXL LoRA keys (``unet.<path>.lora_A/B.weight``) plus
  metadata (``family=sdxl``) so the normal LoRA loader picks it up unchanged.
"""
from __future__ import annotations

import gc
import math
import random
import threading
import time
from pathlib import Path

import sana_engine as se
import sana_trainer as st

torch = se.torch

_SDXL_RESOLUTIONS = (512, 768, 1024)
_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


class SdxlTrainer:
    """Runs SDXL LoRA training jobs inside the normal job queue."""

    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.Lock()

    # -- public entry (called by the worker like manager.generate) -----------
    def train(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        with self.lock:
            return self._train(params, job, cancel)

    def _sanitize(self, params: dict) -> dict:
        ds = st.get_dataset(str(params.get("dataset_id") or ""))
        if ds is None:
            raise RuntimeError("ไม่พบ dataset ที่จะเทรน")
        if not ds.get("images"):
            raise RuntimeError("dataset ยังไม่มีรูป — เพิ่มรูปก่อนเทรน")
        model_id = str(params.get("model_id") or "").strip()
        try:
            entry = self.manager.get_model(model_id or None)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if entry.get("family") != "sdxl":
            raise RuntimeError("เทรนด้วย SDXL trainer ต้องเลือกโมเดลฐานตระกูล SDXL")
        model_path = Path(entry["path"])
        if not model_path.exists():
            raise RuntimeError(f"ไม่พบไฟล์โมเดล SDXL: {model_path}")
        rank = max(4, min(128, int(params.get("rank") or 16)))
        steps = max(10, min(20000, int(params.get("steps") or 1000)))
        lr = float(params.get("lr") or 1e-4)
        lr = max(1e-5, min(5e-4, lr))
        resolution = int(params.get("resolution") or 768)
        resolution = resolution if resolution in _SDXL_RESOLUTIONS else 768
        lora_name = st.sanitize_lora_name(params.get("lora_name"))
        scheduler = str(params.get("scheduler") or "cosine").lower()
        if scheduler not in ("cosine", "constant"):
            scheduler = "cosine"
        try:
            warmup_ratio = float(params.get("warmup_ratio"))
        except (TypeError, ValueError):
            warmup_ratio = 0.03
        warmup_ratio = max(0.0, min(0.2, warmup_ratio))
        ema = bool(params.get("ema", False))
        try:
            ema_decay = float(params.get("ema_decay"))
        except (TypeError, ValueError):
            ema_decay = 0.999
        ema_decay = max(0.9, min(0.9999, ema_decay))
        sampling = str(params.get("timestep_sampling") or "logit_normal").lower()
        if sampling not in ("logit_normal", "uniform", "shift"):
            sampling = "logit_normal"
        if sampling == "shift":
            sampling = "uniform"
        try:
            save_every = int(params.get("save_every") or 0)
        except (TypeError, ValueError):
            save_every = 0
        save_every = max(0, min(steps, save_every))
        augment_flip = bool(params.get("augment_flip", False))
        try:
            grad_accum = int(params.get("grad_accum") or 1)
        except (TypeError, ValueError):
            grad_accum = 1
        grad_accum = max(1, min(8, grad_accum))
        multi_mode = str(params.get("multi_mode") or "exclude").lower()
        if multi_mode not in ("exclude", "weighted", "include"):
            multi_mode = "exclude"
        try:
            multi_weight = float(params.get("multi_weight"))
        except (TypeError, ValueError):
            multi_weight = 0.25
        multi_weight = max(0.05, min(1.0, multi_weight))
        bucketing = st.sanitize_bool(params.get("bucketing"), True)
        try:
            caption_dropout = float(params.get("caption_dropout"))
        except (TypeError, ValueError):
            caption_dropout = 0.05
        caption_dropout = max(0.0, min(0.3, caption_dropout))
        caption_shuffle = st.sanitize_bool(params.get("caption_shuffle"), False)
        try:
            keep_tokens = int(params.get("keep_tokens"))
        except (TypeError, ValueError):
            keep_tokens = 1
        keep_tokens = max(0, min(4, keep_tokens))
        try:
            noise_offset = float(params.get("noise_offset"))
        except (TypeError, ValueError):
            noise_offset = 0.0
        noise_offset = max(0.0, min(0.2, noise_offset))
        try:
            min_snr_gamma = float(params.get("min_snr_gamma"))
        except (TypeError, ValueError):
            min_snr_gamma = 0.0
        min_snr_gamma = max(0.0, min(20.0, min_snr_gamma))
        use_rslora = st.sanitize_bool(params.get("use_rslora"), False)
        target_set = str(params.get("target_modules") or "attn").lower()
        if target_set not in ("attn", "attn_ff"):
            target_set = "attn"
        resume_state = st.sanitize_bool(params.get("resume_state"), False)
        thermal_guard = bool(params.get("thermal_guard", True))
        thermal_mode = str(params.get("thermal_mode") or "throttle").lower()
        if thermal_mode not in ("throttle", "pause"):
            thermal_mode = "throttle"
        try:
            thermal_max_c = int(params.get("thermal_max_c") or 90)
        except (TypeError, ValueError):
            thermal_max_c = 90
        thermal_max_c = max(80, min(93, thermal_max_c))
        try:
            thermal_throttle_c = int(
                params.get("thermal_throttle_c") or min(87, thermal_max_c - 3)
            )
        except (TypeError, ValueError):
            thermal_throttle_c = min(87, thermal_max_c - 3)
        thermal_throttle_c = max(60, min(thermal_max_c - 1, thermal_throttle_c))
        try:
            thermal_resume_c = int(
                params.get("thermal_resume_c")
                or max(45, thermal_throttle_c - 7)
            )
        except (TypeError, ValueError):
            thermal_resume_c = max(45, thermal_throttle_c - 7)
        thermal_resume_c = max(40, min(thermal_throttle_c - 1, thermal_resume_c))
        try:
            thermal_step_pct = int(params.get("thermal_step_pct") or 5)
        except (TypeError, ValueError):
            thermal_step_pct = 5
        thermal_step_pct = max(1, min(25, thermal_step_pct))
        thermal_max_level = max(1, min(8, 25 // thermal_step_pct))
        try:
            thermal_check_every = int(params.get("thermal_check_every") or 10)
        except (TypeError, ValueError):
            thermal_check_every = 10
        thermal_check_every = max(1, min(200, thermal_check_every))
        try:
            seed = int(params.get("seed"))
        except (TypeError, ValueError):
            seed = -1
        seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        init_lora = st._resolve_init_lora(params.get("init_lora"))
        if init_lora is not None:
            file_rank = st._lora_file_rank(init_lora)
            if file_rank is not None:
                rank = max(4, min(128, int(file_rank)))
        out_path = se.MODELS_ROOT / "loras" / f"{lora_name}.safetensors"
        return {
            "ds": ds,
            "model_id": entry["id"],
            "model_name": entry["name"],
            "model_path": model_path,
            "model_kind": entry.get("kind") or "single",
            "rank": rank,
            "steps": steps,
            "lr": lr,
            "resolution": resolution,
            "lora_name": lora_name,
            "out_path": out_path,
            "seed": seed,
            "scheduler": scheduler,
            "warmup_ratio": warmup_ratio,
            "ema": ema,
            "ema_decay": ema_decay,
            "sampling": sampling,
            "save_every": save_every,
            "augment_flip": augment_flip,
            "grad_accum": grad_accum,
            "multi_mode": multi_mode,
            "multi_weight": multi_weight,
            "bucketing": bucketing,
            "caption_dropout": caption_dropout,
            "caption_shuffle": caption_shuffle,
            "keep_tokens": keep_tokens,
            "noise_offset": noise_offset,
            "min_snr_gamma": min_snr_gamma,
            "use_rslora": use_rslora,
            "target_set": target_set,
            "resume_state": resume_state,
            "thermal_guard": thermal_guard,
            "thermal_mode": thermal_mode,
            "thermal_max_c": thermal_max_c,
            "thermal_throttle_c": thermal_throttle_c,
            "thermal_resume_c": thermal_resume_c,
            "thermal_step_pct": thermal_step_pct,
            "thermal_max_level": thermal_max_level,
            "thermal_check_every": thermal_check_every,
            "init_lora": init_lora,
        }

    def _train(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        """Run training, then return GPU memory to the driver."""
        try:
            return self._train_inner(params, job, cancel)
        finally:
            st._clock_reset()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_inner(self, params: dict, job: dict, cancel: threading.Event) -> dict:
        cfg = self._sanitize(params)
        ds = cfg["ds"]
        train_images, sample_weights, multi_mode = st.select_train_images(
            ds.get("images") or [],
            cfg["multi_mode"],
            cfg["multi_weight"],
            ds.get("kind") or "character",
        )
        solo_count = sum(1 for im in train_images if not im.get("multi"))
        multi_count = len(train_images) - solo_count

        def status(msg):
            job.update({"phase": "preparing", "message": msg, "status": "running"})
            print(f"[sdxl-train] {msg}", flush=True)

        mode_label = {
            "exclude": "เฉพาะตัวละครเดี่ยว",
            "weighted": f"ลดน้ำหนักรูปหลายตัวละครเหลือ {int(cfg['multi_weight'] * 100)}%",
            "include": "ใช้ทุกรูปน้ำหนักเท่ากัน",
        }[multi_mode]
        print(
            f"[sdxl-train] dataset policy: {mode_label} · ใช้ {len(train_images)}/"
            f"{len(ds['images'])} รูป (เดี่ยว {solo_count} · หลายตัวละคร {multi_count})",
            flush=True,
        )

        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("เทรนต้องใช้ GPU — ตรวจ PyTorch/CUDA ใน engine/setup.sh")

        status(f"ปล่อยหน่วยความจำโมเดลเดิมก่อนเทรน ({cfg['lora_name']})…")
        self.manager.unload()

        import numpy as np
        from PIL import Image, ImageOps
        from peft import LoraConfig, inject_adapter_in_model

        from diffusers import StableDiffusionXLPipeline

        status(f"กำลังโหลดโมเดล SDXL สำหรับเทรน ({cfg['model_name']})…")
        dtype = torch.float16
        if cfg["model_kind"] == "single":
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(cfg["model_path"]),
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(cfg["model_path"]),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        pipe.set_progress_bar_config(disable=True)
        try:
            # UNet2DConditionModel has no enable_attention_slicing() in recent
            # diffusers; the pipeline-level helper applies it to the UNet.
            pipe.enable_attention_slicing()
        except Exception as exc:  # noqa: BLE001
            print("[sdxl-train] attention slicing warning:", exc, flush=True)
        device = torch.device("cuda")
        unet = pipe.unet

        try:
            # ── 1. VAE pass: cache latents for every image, then free the VAE ──
            status(f"เตรียม latent ({len(train_images)} รูป · {cfg['resolution']}px)…")
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception as exc:  # noqa: BLE001
                print("[sdxl-train] vae tiling warning:", exc, flush=True)
            # SDXL's VAE is numerically unstable in fp16 → encode in fp32.
            pipe.vae.to(device=device, dtype=torch.float32)
            scale = float(pipe.vae.config.scaling_factor)
            latents = []
            sizes = []
            with torch.no_grad():
                for index, image in enumerate(train_images):
                    if cancel.is_set():
                        raise se.Cancelled()
                    path = st.TRAIN_DIR / ds["id"] / image["file"]
                    img = Image.open(path).convert("RGB")
                    bucket = (
                        st.pick_bucket(img.width, img.height, cfg["resolution"])
                        if cfg["bucketing"]
                        else (cfg["resolution"], cfg["resolution"])
                    )
                    sizes.append(bucket)
                    img = ImageOps.fit(img, bucket, method=Image.LANCZOS)
                    arr = np.asarray(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
                    pixel = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
                    latent = pipe.vae.encode(pixel).latent_dist.sample()
                    latents.append((latent * scale).to(dtype=torch.float16))
                    job["message"] = (
                        f"เตรียม latent {index + 1}/{len(train_images)} "
                        f"({bucket[0]}×{bucket[1]})"
                    )
                    job["progress"] = round(0.05 * (index + 1) / len(train_images), 4)
            pipe.vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── 2. text encoders: embed every caption once, then free ──
            variant_count = 2 if cfg["caption_shuffle"] else 1
            caption_groups = [
                st.caption_variants(
                    image,
                    ds,
                    cfg["caption_shuffle"],
                    cfg["keep_tokens"],
                    variant_count,
                )
                for image in train_images
            ]
            status(
                f"เตรียม embedding ของ caption ({len(caption_groups)} รูป · "
                f"{sum(len(g) for g in caption_groups)} แบบ) — ตัวอย่าง: "
                f"“{caption_groups[0][0][:70]}”"
            )
            pipe.text_encoder.to(device)
            pipe.text_encoder_2.to(device)
            embeds, pooled = [], []
            with torch.no_grad():
                for group in caption_groups:
                    group_embeds, group_pooled = [], []
                    for text in group:
                        p_emb, _, p_pooled, _ = pipe.encode_prompt(
                            prompt=text,
                            prompt_2=text,
                            device=device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        group_embeds.append(p_emb.to(dtype=torch.float16))
                        group_pooled.append(p_pooled.to(dtype=torch.float16))
                    embeds.append(group_embeds)
                    pooled.append(group_pooled)
                empty_emb, _, empty_pooled, _ = pipe.encode_prompt(
                    prompt="",
                    prompt_2="",
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                empty_emb = empty_emb.to(dtype=torch.float16)
                empty_pooled = empty_pooled.to(dtype=torch.float16)
            pipe.text_encoder = None
            pipe.text_encoder_2 = None
            pipe.tokenizer = None
            pipe.tokenizer_2 = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── 3. inject LoRA into the UNet ──
            targets = list(_TARGET_MODULES)
            if cfg["target_set"] == "attn_ff":
                targets += ["ff.net.0.proj", "ff.net.2"]
            rs_note = " · rsLoRA" if cfg["use_rslora"] else ""
            status(
                f"ฉีด LoRA rank {cfg['rank']} "
                f"(alpha={cfg['rank']}, targets={cfg['target_set']}{rs_note})…"
            )
            unet.requires_grad_(False)
            unet.to(device)
            lora_cfg = LoraConfig(
                r=cfg["rank"],
                lora_alpha=cfg["rank"],
                init_lora_weights="gaussian",
                target_modules=targets,
                bias="none",
                use_rslora=cfg["use_rslora"],
            )
            peft_unet = inject_adapter_in_model(lora_cfg, unet)
            trainable = [p for p in peft_unet.parameters() if p.requires_grad]
            for p in trainable:
                p.data = p.data.float()
            rs_scale = math.sqrt(cfg["rank"]) if cfg["use_rslora"] else 1.0
            if cfg["init_lora"] is not None:
                status(
                    f"โหลดน้ำหนักตั้งต้นจาก {cfg['init_lora'].name} "
                    f"(rank {cfg['rank']})…"
                )
                loaded, skipped = st._load_lora_weights(
                    peft_unet,
                    cfg["init_lora"],
                    prefix="unet.",
                    lora_b_scale=1.0 / rs_scale if cfg["use_rslora"] else 1.0,
                )
                print(
                    f"[sdxl-train] init LoRA {cfg['init_lora'].name}: "
                    f"{loaded} tensors loaded, {len(skipped)} skipped",
                    flush=True,
                )
            try:
                peft_unet.enable_gradient_checkpointing()
            except Exception as exc:  # noqa: BLE001
                print("[sdxl-train] gradient checkpointing warning:", exc, flush=True)
            peft_unet.train()

            opt = torch.optim.AdamW(
                trainable, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.99)
            )
            total_opt_steps = max(1, cfg["steps"])
            warmup = max(1, int(total_opt_steps * cfg["warmup_ratio"]))
            min_ratio = 0.1

            if cfg["scheduler"] == "cosine":

                def _lr_lambda(s):
                    if s < warmup:
                        return (s + 1) / warmup
                    prog = (s - warmup) / max(1, total_opt_steps - warmup)
                    return min_ratio + (1 - min_ratio) * 0.5 * (
                        1 + math.cos(math.pi * prog)
                    )

            else:

                def _lr_lambda(s):
                    return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
            rng = random.Random(cfg["seed"])
            g_rng = torch.Generator(device="cuda").manual_seed(cfg["seed"])
            total = len(latents)
            ema_loss = None

            # DDPM schedule used by SDXL (scaled_linear 0.00085 → 0.012).
            from diffusers import DDPMScheduler

            pred_type = "epsilon"
            try:
                pred_type = str(pipe.scheduler.config.prediction_type)
            except Exception:  # noqa: BLE001
                pass
            if pred_type not in ("epsilon", "v_prediction"):
                pred_type = "epsilon"
            noise_sched = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                prediction_type=pred_type,
                clip_sample=False,
            )
            alphas_cumprod = noise_sched.alphas_cumprod.to(device=device)
            timestep_count = int(noise_sched.config.num_train_timesteps)
            # Per-image conditioning: SDXL expects the original size/crop
            # coordinates, and bucketed images each have their own width/height.
            time_ids = [
                torch.tensor(
                    [[width, height, 0, 0, width, height]],
                    device=device,
                    dtype=torch.float16,
                )
                for width, height in sizes
            ]

            ema_params: dict[str, torch.Tensor] = {}
            if cfg["ema"]:
                for name, p in peft_unet.named_parameters():
                    if p.requires_grad and "lora_" in name:
                        ema_params[name] = p.detach().clone().float()
            ema_decay = cfg["ema_decay"] if ema_params else 0.0

            start_step = 0
            state_file = st.trainstate_path(cfg["out_path"])
            if cfg["resume_state"] and state_file.is_file():
                status(f"กู้สถานะการเทรนจาก {state_file.name}…")
                try:
                    start_step, saved_loss = st.load_train_state(
                        state_file,
                        optimizer=opt,
                        scheduler=sched,
                        ema_params=ema_params,
                        rng=rng,
                        g_rng=g_rng,
                        rank=cfg["rank"],
                    )
                    if saved_loss is not None:
                        ema_loss = float(saved_loss)
                    print(
                        f"[sdxl-train] resumed at step {start_step}/{cfg['steps']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[sdxl-train] resume skipped: {exc}", flush=True)
                    start_step = 0

            def _write_lora(path: Path, step_no: int, final: bool) -> None:
                backup = {}
                if ema_params:
                    for name, p in peft_unet.named_parameters():
                        if name in ema_params:
                            backup[name] = p.detach().clone()
                            p.data.copy_(ema_params[name].to(p.dtype))
                try:
                    st._save_lora_state(
                        peft_unet,
                        path,
                        {
                            "lora_name": cfg["lora_name"],
                            "trigger": ds.get("trigger") or "",
                            "family": "sdxl",
                            "base_model": cfg["model_name"],
                            "dataset": ds.get("name") or "",
                            "kind": ds.get("kind") or "",
                            "rank": str(cfg["rank"]),
                            "steps": str(cfg["steps"]),
                            "lr": str(cfg["lr"]),
                            "resolution": str(cfg["resolution"]),
                            "scheduler": cfg["scheduler"],
                            "timestep_sampling": cfg["sampling"],
                            "ema": str(bool(ema_params)),
                            "grad_accum": str(cfg["grad_accum"]),
                            "multi_mode": multi_mode,
                            "multi_weight": str(cfg["multi_weight"]),
                            "bucketing": str(cfg["bucketing"]),
                            "caption_dropout": str(cfg["caption_dropout"]),
                            "caption_shuffle": str(cfg["caption_shuffle"]),
                            "keep_tokens": str(cfg["keep_tokens"]),
                            "noise_offset": str(cfg["noise_offset"]),
                            "min_snr_gamma": str(cfg["min_snr_gamma"]),
                            "use_rslora": str(cfg["use_rslora"]),
                            "target_modules": cfg["target_set"],
                            "train_images": str(len(train_images)),
                            "solo_images": str(solo_count),
                            "multi_images": str(multi_count),
                            "init_from": (
                                cfg["init_lora"].name if cfg["init_lora"] else ""
                            ),
                            "checkpoint_step": str(step_no),
                            "loss": f"{ema_loss:.4f}" if ema_loss is not None else "",
                            "trained_at": st._now(),
                        },
                        prefix="unet.",
                        lora_b_scale=rs_scale,
                    )
                finally:
                    for name, p in peft_unet.named_parameters():
                        if name in backup:
                            p.data.copy_(backup[name])

            def _save_state(lora_file: Path, step_no: int) -> None:
                try:
                    st.save_train_state(
                        st.trainstate_path(lora_file),
                        optimizer=opt,
                        scheduler=sched,
                        step=step_no,
                        ema_params=ema_params,
                        rng=rng,
                        g_rng=g_rng,
                        ema_loss=ema_loss,
                        rank=cfg["rank"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[sdxl-train] trainstate save skipped: {exc}", flush=True)

            job.update(
                {
                    "phase": "generating",
                    "message": "เริ่มเทรน LoRA (SDXL)",
                    "steps": cfg["steps"],
                    "step": start_step,
                    "batch": 1,
                    "progress": 0.05,
                    "status": "running",
                    "loss": None,
                    "lora_name": cfg["lora_name"],
                }
            )

            accum = cfg["grad_accum"]
            opt.zero_grad(set_to_none=True)
            thermal_state = {
                "level": 0,
                "clocks": None,
                "sleep": 0.0,
                "step_time": None,
                "applied": False,
                "temp": None,
            }
            for step in range(start_step, cfg["steps"]):
                if cancel.is_set():
                    raise se.Cancelled()
                step_started = time.time()
                if (
                    cfg["thermal_guard"]
                    and step % cfg["thermal_check_every"] == 0
                ):
                    if cfg["thermal_mode"] == "pause":
                        temp = st.gpu_temperature()
                        if temp is not None and temp >= cfg["thermal_max_c"]:
                            print(
                                f"[sdxl-train] thermal pause at {temp:.0f}°C "
                                f"(limit {cfg['thermal_max_c']}°C)",
                                flush=True,
                            )
                            st.thermal_wait(cfg, job, cancel)
                            job["phase"] = "generating"
                            job["message"] = (
                                f"เทรนต่อ · step {step + 1}/{cfg['steps']}"
                            )
                    else:
                        st.thermal_governor(cfg, job, cancel, thermal_state)
                step_loss = 0.0
                for _micro in range(accum):
                    if sample_weights:
                        index = rng.choices(
                            range(total), weights=sample_weights, k=1
                        )[0]
                    else:
                        index = rng.randrange(total)
                    x0 = latents[index]
                    group = embeds[index]
                    variant = rng.randrange(len(group))
                    emb = group[variant]
                    pool = pooled[index][variant]
                    if cfg["augment_flip"] and rng.random() < 0.5:
                        x0 = torch.flip(x0, dims=[-1])
                    if cfg["caption_dropout"] and rng.random() < cfg["caption_dropout"]:
                        emb, pool = empty_emb, empty_pooled
                    if cfg["sampling"] == "logit_normal":
                        u = 1.0 / (1.0 + math.exp(-rng.gauss(0.0, 1.0)))
                    else:
                        u = rng.random()
                    t_int = min(timestep_count - 1, max(0, int(u * timestep_count)))
                    noise = torch.randn(
                        x0.shape, generator=g_rng, device=device, dtype=x0.dtype
                    )
                    noise = st.add_noise_offset(noise, cfg["noise_offset"], g_rng)
                    alpha = alphas_cumprod[t_int]
                    sqrt_a = alpha.sqrt()
                    sqrt_1ma = (1.0 - alpha).sqrt()
                    x_t = (sqrt_a * x0 + sqrt_1ma * noise).to(dtype=torch.float16)
                    if pred_type == "v_prediction":
                        target = (sqrt_a * noise - sqrt_1ma * x0).to(
                            dtype=torch.float16
                        )
                    else:
                        target = noise
                    t_in = torch.tensor([t_int], device=device, dtype=torch.long)

                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = peft_unet(
                            x_t,
                            t_in,
                            encoder_hidden_states=emb,
                            added_cond_kwargs={
                                "text_embeds": pool,
                                "time_ids": time_ids[index],
                            },
                            return_dict=False,
                        )[0]
                    pred = pred.float()
                    if cfg["min_snr_gamma"]:
                        snr = alpha / torch.clamp(1.0 - alpha, min=1e-8)
                        weight = st.min_snr_weight(
                            snr.float(), cfg["min_snr_gamma"]
                        )
                        per_sample = torch.nn.functional.mse_loss(
                            pred, target.float(), reduction="none"
                        ).mean(dim=tuple(range(1, pred.dim())))
                        loss = (per_sample * weight).mean() / accum
                    else:
                        loss = (
                            torch.nn.functional.mse_loss(pred, target.float())
                            / accum
                        )
                    loss.backward()
                    step_loss += float(loss.item()) * accum

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                if thermal_state["sleep"] > 0:
                    time.sleep(thermal_state["sleep"])
                elapsed_step = time.time() - step_started - thermal_state["sleep"]
                previous = thermal_state.get("step_time")
                thermal_state["step_time"] = (
                    elapsed_step
                    if previous is None
                    else previous * 0.8 + elapsed_step * 0.2
                )

                if ema_params:
                    with torch.no_grad():
                        for name, p in peft_unet.named_parameters():
                            if name in ema_params:
                                ema_params[name].mul_(ema_decay).add_(
                                    p.detach().float(), alpha=1.0 - ema_decay
                                )

                value = step_loss / accum
                ema_loss = value if ema_loss is None else ema_loss * 0.95 + value * 0.05
                job["step"] = step + 1
                job["loss"] = round(ema_loss, 6)
                job["progress"] = round(0.05 + 0.9 * (step + 1) / cfg["steps"], 4)
                throttle_note = ""
                if thermal_state["level"] > 0:
                    throttle_note = (
                        f" · ชะลอ {cfg['thermal_step_pct'] * thermal_state['level']}%"
                    )
                job["message"] = (
                    f"เทรน LoRA SDXL {cfg['lora_name']} · step {step + 1}/"
                    f"{cfg['steps']} · loss {ema_loss:.4f}{throttle_note}"
                )
                if (step + 1) % 10 == 0 or step == 0:
                    print(
                        f"[sdxl-train] step {step + 1}/{cfg['steps']} "
                        f"loss {ema_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"({thermal_state['step_time']:.1f}s/step)",
                        flush=True,
                    )
                if (
                    cfg["save_every"]
                    and (step + 1) % cfg["save_every"] == 0
                    and (step + 1) < cfg["steps"]
                ):
                    ckpt = cfg["out_path"].with_name(
                        f"{cfg['out_path'].stem}-step{step + 1}.safetensors"
                    )
                    if not ckpt.exists():
                        print(
                            f"[sdxl-train] checkpoint step {step + 1} → {ckpt.name}",
                            flush=True,
                        )
                        job["message"] = (
                            f"บันทึก checkpoint step {step + 1}/{cfg['steps']}…"
                        )
                        _write_lora(ckpt, step + 1, final=False)
                        _save_state(ckpt, step + 1)

            if cancel.is_set():
                raise se.Cancelled()

            # ── 4. package the final LoRA file (EMA weights if enabled) ──
            status("บันทึกไฟล์ LoRA…")
            out_path = cfg["out_path"]
            if out_path.exists():
                raise RuntimeError(f"มีไฟล์ชื่อ {out_path.name} อยู่แล้ว — ตั้งชื่อใหม่")
            se.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            (se.MODELS_ROOT / "loras").mkdir(parents=True, exist_ok=True)
            _write_lora(out_path, cfg["steps"], final=True)
            _save_state(out_path, cfg["steps"])

            for lora in se.discover_loras():
                if Path(lora["path"]).resolve() == out_path.resolve():
                    job["progress"] = 1.0
                    status(f"เทรนเสร็จ — บันทึก {out_path.name} แล้ว")
                    return {"lora": lora}
            raise RuntimeError("บันทึกแล้วแต่หา LoRA ไม่เจอในรายการ")
        finally:
            try:
                del pipe
            except Exception:  # noqa: BLE001
                pass
