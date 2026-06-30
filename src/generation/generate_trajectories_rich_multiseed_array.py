from __future__ import annotations

import argparse
import csv
import json
import gc
import logging
import os
import psutil
import sys
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline


# -----------------------
# General helpers
# -----------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("sd35_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def append_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def linear_slope(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    y = np.asarray(vals, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def safe_max(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.max(vals)) if vals else 0.0


def safe_std(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.std(vals)) if vals else 0.0


def tensor_rms(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(torch.sqrt(torch.mean(x ** 2)).item())


def tensor_abs_mean(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(torch.mean(torch.abs(x)).item())


def tensor_std(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(torch.std(x).item())


def tensor_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    denom = torch.norm(a) * torch.norm(b) + eps
    return float(torch.dot(a, b).item() / denom.item())


def normalized_residual(delta: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Fit best scalar alpha so alpha * pred approximates delta.
    Return ||delta - alpha pred|| / ||delta||.

    This is a scheduler-agnostic denoising consistency residual.
    Lower = denoiser direction explains the actual latent update better.
    """
    d = delta.detach().float().flatten()
    p = pred.detach().float().flatten()

    denom = torch.dot(p, p) + eps
    alpha = torch.dot(d, p) / denom
    residual = d - alpha * p

    return float((torch.norm(residual) / (torch.norm(d) + eps)).item())


def maybe_resume_skip(manifest_csv: Path) -> set[tuple[str, int]]:
    """Return completed (prompt_id, seed) pairs so reruns can resume safely."""
    if not manifest_csv.exists():
        return set()

    old = pd.read_csv(manifest_csv)
    if not {"prompt_id", "seed"}.issubset(old.columns):
        return set()

    ok = old.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"] == "ok"]

    return set((str(r["prompt_id"]), int(r["seed"])) for _, r in ok.iterrows())


def parse_seed_list(
    seed_list: Optional[str],
    seeds_per_prompt: int,
    base_seed: int,
    original_prompt_index: int,
    seed_stride: int,
    replicate_start: int,
) -> list[tuple[int, int]]:
    """
    Build a systematic list of (replicate_id, seed) pairs for one prompt.

    Default formula:
        replicate_id = replicate_start + local_replicate_id
        seed = base_seed + original_prompt_index * seed_stride + replicate_id

    This keeps seed identities stable across chunked SLURM array jobs. For example,
    the same prompt at original CSV row 250 always gets seed
    base_seed + 250 * seed_stride + replicate_id, even if this script is run only
    on a slice of the CSV.

    If --seed-list is passed, those exact seeds are used for every prompt; the
    replicate_id values still begin at --replicate-start.
    """
    replicate_ids = [replicate_start + i for i in range(seeds_per_prompt)]

    if seed_list:
        parsed = [int(x.strip()) for x in seed_list.split(",") if x.strip()]
        if len(parsed) != seeds_per_prompt:
            raise ValueError(
                f"--seed-list contains {len(parsed)} seeds, but --seeds-per-prompt={seeds_per_prompt}."
            )
        return list(zip(replicate_ids, parsed))

    seeds = [int(base_seed + original_prompt_index * seed_stride + rid) for rid in replicate_ids]
    return list(zip(replicate_ids, seeds))


# -----------------------
# Signal recorder
# -----------------------

class InternalSignalRecorder:
    """
    Captures rich early trajectory signals.

    Captured from callback:
    - latent RMS/std/absmean
    - latent step-to-step volatility
    - latent update cosine
    - denoising consistency residual

    Captured from transformer hook:
    - denoiser/model prediction norms
    - denoiser step-to-step consistency
    - CFG divergence if unconditional/conditional chunks are visible
    """

    def __init__(self, early_steps: int, guidance_scale: float):
        self.early_steps = early_steps
        self.guidance_scale = guidance_scale

        self.timesteps: list[int] = []
        self.step_times_sec: list[float] = []

        # Latent state
        self.latent_rms: list[float] = []
        self.latent_std: list[float] = []
        self.latent_abs_mean: list[float] = []
        self.latent_volatility_rms: list[float] = []
        self.latent_update_cosine: list[float] = []

        # Denoiser/model output
        self.denoiser_pred_rms: list[float] = []
        self.denoiser_pred_std: list[float] = []
        self.denoiser_pred_abs_mean: list[float] = []
        self.denoiser_pred_delta_rms: list[float] = []
        self.denoiser_pred_cosine_prev: list[float] = []

        # CFG / guidance-specific
        self.cfg_divergence_rms: list[float] = []
        self.cfg_divergence_abs_mean: list[float] = []
        self.cfg_divergence_relative: list[float] = []
        self.cfg_alignment_cosine: list[float] = []
        self.guided_minus_cond_rms: list[float] = []
        self.cfg_chunk_count: list[int] = []

        # Denoising consistency
        self.denoising_consistency_residual: list[float] = []
        self.denoising_update_pred_cosine: list[float] = []

        # Internal state
        self._prev_latents_cpu: Optional[torch.Tensor] = None
        self._prev_delta_cpu: Optional[torch.Tensor] = None
        self._prev_pred_cpu: Optional[torch.Tensor] = None
        self._last_pred_cpu: Optional[torch.Tensor] = None
        self._last_callback_time: Optional[float] = None
        self._hook_step_count = 0

    def transformer_hook(self, output: Any) -> None:
        """
        Called after pipe.transformer.forward.

        The transformer output is usually either:
        - tuple/list where first item is tensor
        - object with .sample
        - tensor directly
        """
        if self._hook_step_count >= self.early_steps:
            self._hook_step_count += 1
            return

        try:
            if isinstance(output, (tuple, list)):
                pred = output[0]
            elif hasattr(output, "sample"):
                pred = output.sample
            else:
                pred = output

            if not isinstance(pred, torch.Tensor):
                self._hook_step_count += 1
                return

            pred_cpu = pred.detach().to("cpu", dtype=torch.float32)

            # If CFG is active, the batch may contain uncond/cond chunks.
            # For batch size 1, commonly this is 2 chunks; some SD3 variants may expose 3.
            chunk_count = int(pred_cpu.shape[0]) if pred_cpu.ndim >= 1 else 1
            self.cfg_chunk_count.append(chunk_count)

            pred_for_consistency = pred_cpu
            if chunk_count >= 2:
                uncond = pred_cpu[0:1]
                cond = pred_cpu[1:2]
                cfg_delta = cond - uncond
                guided = uncond + self.guidance_scale * cfg_delta

                self.cfg_divergence_rms.append(tensor_rms(cfg_delta))
                self.cfg_divergence_abs_mean.append(tensor_abs_mean(cfg_delta))
                self.cfg_alignment_cosine.append(tensor_cosine(uncond, cond))
                self.guided_minus_cond_rms.append(tensor_rms(guided - cond))

                guided_rms = tensor_rms(guided)
                self.cfg_divergence_relative.append(
                    tensor_rms(cfg_delta) / (guided_rms + 1e-8)
                )

                pred_for_consistency = guided
            else:
                self.cfg_divergence_rms.append(np.nan)
                self.cfg_divergence_abs_mean.append(np.nan)
                self.cfg_alignment_cosine.append(np.nan)
                self.guided_minus_cond_rms.append(np.nan)
                self.cfg_divergence_relative.append(np.nan)

            self.denoiser_pred_rms.append(tensor_rms(pred_for_consistency))
            self.denoiser_pred_std.append(tensor_std(pred_for_consistency))
            self.denoiser_pred_abs_mean.append(tensor_abs_mean(pred_for_consistency))

            if self._prev_pred_cpu is not None:
                self.denoiser_pred_delta_rms.append(
                    tensor_rms(pred_for_consistency - self._prev_pred_cpu)
                )
                self.denoiser_pred_cosine_prev.append(
                    tensor_cosine(pred_for_consistency, self._prev_pred_cpu)
                )

            self._prev_pred_cpu = pred_for_consistency.clone()
            self._last_pred_cpu = pred_for_consistency.clone()

        except Exception:
            # Do not crash generation because signal capture failed.
            pass
        finally:
            self._hook_step_count += 1

    def callback(self, step_index: int, timestep: int, latents: torch.Tensor) -> None:
        if step_index >= self.early_steps:
            return

        now = time.time()
        if self._last_callback_time is not None:
            self.step_times_sec.append(now - self._last_callback_time)
        self._last_callback_time = now

        lat_cpu = latents.detach().to("cpu", dtype=torch.float32)

        self.timesteps.append(int(timestep))
        self.latent_rms.append(tensor_rms(lat_cpu))
        self.latent_std.append(tensor_std(lat_cpu))
        self.latent_abs_mean.append(tensor_abs_mean(lat_cpu))

        if self._prev_latents_cpu is not None:
            delta = lat_cpu - self._prev_latents_cpu
            self.latent_volatility_rms.append(tensor_rms(delta))

            if self._prev_delta_cpu is not None:
                self.latent_update_cosine.append(tensor_cosine(delta, self._prev_delta_cpu))

            # Compare actual latent update to latest denoiser prediction direction.
            if self._last_pred_cpu is not None:
                try:
                    self.denoising_consistency_residual.append(
                        normalized_residual(delta, self._last_pred_cpu)
                    )
                    self.denoising_update_pred_cosine.append(
                        tensor_cosine(delta, self._last_pred_cpu)
                    )
                except Exception:
                    self.denoising_consistency_residual.append(np.nan)
                    self.denoising_update_pred_cosine.append(np.nan)

            self._prev_delta_cpu = delta.clone()

        self._prev_latents_cpu = lat_cpu.clone()

    def to_summary_row(self) -> dict:
        return {
            # Raw sequences
            "timesteps_json": json.dumps(self.timesteps),
            "latent_rms_json": json.dumps(self.latent_rms),
            "latent_std_json": json.dumps(self.latent_std),
            "latent_abs_mean_json": json.dumps(self.latent_abs_mean),
            "latent_volatility_rms_json": json.dumps(self.latent_volatility_rms),
            "latent_update_cosine_json": json.dumps(self.latent_update_cosine),
            "step_times_sec_json": json.dumps(self.step_times_sec),

            "denoiser_pred_rms_json": json.dumps(self.denoiser_pred_rms),
            "denoiser_pred_std_json": json.dumps(self.denoiser_pred_std),
            "denoiser_pred_abs_mean_json": json.dumps(self.denoiser_pred_abs_mean),
            "denoiser_pred_delta_rms_json": json.dumps(self.denoiser_pred_delta_rms),
            "denoiser_pred_cosine_prev_json": json.dumps(self.denoiser_pred_cosine_prev),

            "cfg_divergence_rms_json": json.dumps(self.cfg_divergence_rms),
            "cfg_divergence_abs_mean_json": json.dumps(self.cfg_divergence_abs_mean),
            "cfg_divergence_relative_json": json.dumps(self.cfg_divergence_relative),
            "cfg_alignment_cosine_json": json.dumps(self.cfg_alignment_cosine),
            "guided_minus_cond_rms_json": json.dumps(self.guided_minus_cond_rms),
            "cfg_chunk_count_json": json.dumps(self.cfg_chunk_count),

            "denoising_consistency_residual_json": json.dumps(self.denoising_consistency_residual),
            "denoising_update_pred_cosine_json": json.dumps(self.denoising_update_pred_cosine),

            # Latent summaries
            "latent_rms_mean": round(safe_mean(self.latent_rms), 8),
            "latent_rms_max": round(safe_max(self.latent_rms), 8),
            "latent_rms_std": round(safe_std(self.latent_rms), 8),
            "latent_rms_slope": round(linear_slope(self.latent_rms), 8),

            "latent_std_mean": round(safe_mean(self.latent_std), 8),
            "latent_abs_mean_mean": round(safe_mean(self.latent_abs_mean), 8),

            "latent_volatility_mean": round(safe_mean(self.latent_volatility_rms), 8),
            "latent_volatility_max": round(safe_max(self.latent_volatility_rms), 8),
            "latent_volatility_std": round(safe_std(self.latent_volatility_rms), 8),
            "latent_volatility_slope": round(linear_slope(self.latent_volatility_rms), 8),
            "latent_update_cosine_mean": round(safe_mean(self.latent_update_cosine), 8),

            # Denoiser summaries
            "denoiser_pred_rms_mean": round(safe_mean(self.denoiser_pred_rms), 8),
            "denoiser_pred_rms_max": round(safe_max(self.denoiser_pred_rms), 8),
            "denoiser_pred_rms_slope": round(linear_slope(self.denoiser_pred_rms), 8),
            "denoiser_pred_std_mean": round(safe_mean(self.denoiser_pred_std), 8),
            "denoiser_pred_abs_mean_mean": round(safe_mean(self.denoiser_pred_abs_mean), 8),

            "denoiser_pred_delta_rms_mean": round(safe_mean(self.denoiser_pred_delta_rms), 8),
            "denoiser_pred_delta_rms_max": round(safe_max(self.denoiser_pred_delta_rms), 8),
            "denoiser_pred_delta_rms_slope": round(linear_slope(self.denoiser_pred_delta_rms), 8),
            "denoiser_pred_cosine_prev_mean": round(safe_mean(self.denoiser_pred_cosine_prev), 8),

            # CFG summaries
            "cfg_divergence_mean": round(safe_mean(self.cfg_divergence_rms), 8),
            "cfg_divergence_max": round(safe_max(self.cfg_divergence_rms), 8),
            "cfg_divergence_std": round(safe_std(self.cfg_divergence_rms), 8),
            "cfg_divergence_slope": round(linear_slope(self.cfg_divergence_rms), 8),
            "cfg_divergence_abs_mean_mean": round(safe_mean(self.cfg_divergence_abs_mean), 8),
            "cfg_divergence_relative_mean": round(safe_mean(self.cfg_divergence_relative), 8),
            "cfg_alignment_cosine_mean": round(safe_mean(self.cfg_alignment_cosine), 8),
            "guided_minus_cond_rms_mean": round(safe_mean(self.guided_minus_cond_rms), 8),

            # Consistency summaries
            "denoising_consistency_residual_mean": round(safe_mean(self.denoising_consistency_residual), 8),
            "denoising_consistency_residual_max": round(safe_max(self.denoising_consistency_residual), 8),
            "denoising_consistency_residual_slope": round(linear_slope(self.denoising_consistency_residual), 8),
            "denoising_update_pred_cosine_mean": round(safe_mean(self.denoising_update_pred_cosine), 8),
        }


# -----------------------
# Main
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("data/final_generation_prompts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sd35_trajectories_rich_250"))
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--early-steps", type=int, default=5)
    # Multi-seed design:
    #   --max-prompts 250 --seeds-per-prompt 5 creates 250 prompt groups × 5 seeds = 1250 images.
    #   prompt_id stays fixed across the 5 runs, while replicate_id and seed change.
    #   Default seed formula: seed = base_seed + prompt_index * seed_stride + replicate_id.
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--seed-list", type=str, default=None, help="Optional comma-separated exact seeds used for every prompt, e.g. '12345,13345,14345,15345,16345'.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-prompts", type=int, default=None, help="Maximum number of prompts to use after slicing. Use None/all for full CSV.")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive row index in the input CSV. Intended for SLURM array chunking.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive row index in the input CSV. Intended for SLURM array chunking.")
    parser.add_argument("--replicate-start", type=int, default=0, help="First replicate_id to generate. Use 0 for the first seed, 1 for extra seeds after seed 0 already exists.")
    parser.add_argument("--max-generations", type=int, default=None, help="Optional hard cap on total prompt×seed generations. If omitted, all selected prompts × seeds_per_prompt are attempted.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    logs_dir = output_dir / "logs"
    ensure_dir(images_dir)
    ensure_dir(logs_dir)

    logger = setup_logger(logs_dir / "run.log")
    logger.info("Starting rich SD3.5 trajectory batch run")

    user = os.environ.get("USER", "unknown")
    hf_home = os.environ.get("HF_HOME", f"/scratch/{user}/huggingface")
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_DATASETS_CACHE", hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    logger.info("HF_HOME=%s", os.environ["HF_HOME"])
    logger.info("HF_HUB_DISABLE_XET=%s", os.environ["HF_HUB_DISABLE_XET"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    df = pd.read_csv(args.input_csv)
    required = {"prompt_id", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    if "split" not in df.columns:
        df["split"] = "unknown"

    # Preserve the original CSV row index for stable seed calculation across chunks.
    df = df.reset_index(drop=False).rename(columns={"index": "original_prompt_index"})

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")

    if args.end_index is None:
        df = df.iloc[args.start_index:].copy()
    else:
        df = df.iloc[args.start_index:args.end_index].copy()

    if args.max_prompts is not None:
        df = df.head(args.max_prompts).copy()

    expected_generations = int(len(df) * args.seeds_per_prompt)
    max_generations = args.max_generations if args.max_generations is not None else expected_generations

    manifest_csv = output_dir / "generation_manifest.csv"
    traj_csv = output_dir / "trajectory_signals.csv"
    fail_csv = output_dir / "failures.csv"
    config_json = output_dir / "run_config.json"

    config_json.write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    done = maybe_resume_skip(manifest_csv) if not args.overwrite else set()
    logger.info("Loaded %d selected prompts from %s", len(df), args.input_csv)
    logger.info("CSV slice: start_index=%s end_index=%s", args.start_index, str(args.end_index))
    logger.info("Seeds per prompt in this run: %d", args.seeds_per_prompt)
    logger.info("Replicate start: %d", args.replicate_start)
    logger.info("Expected prompt-seed generations: %d", expected_generations)
    logger.info("Found %d completed runs to skip", len(done))
    logger.info("Max generations for this run: %s", str(max_generations))

    logger.info("Loading pipeline...")
    t0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    logger.info("Pipeline loaded in %.2f seconds", time.time() - t0)

    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    logger.info("Memory after pipeline load | host_rss=%.2f GB | gpu_alloc=%.2f GB", rss_gb, vram_gb)

    generated_count = 0

    for local_row_idx, row in df.reset_index(drop=True).iterrows():
        if max_generations is not None and generated_count >= max_generations:
            logger.info("Reached max_generations=%d. Stopping.", max_generations)
            break

        prompt_id = str(row["prompt_id"])
        prompt = str(row["prompt"]).strip()
        split = str(row["split"])
        # prompt_index is the original row number in the full input CSV, not the local chunk row.
        # This is important: array jobs must not all reuse prompt_index 0..999 for seed calculation.
        prompt_index = int(row["original_prompt_index"])
        prompt_seed_pairs = parse_seed_list(
            args.seed_list,
            args.seeds_per_prompt,
            args.base_seed,
            prompt_index,
            args.seed_stride,
            args.replicate_start,
        )

        for replicate_id, seed in prompt_seed_pairs:
            if max_generations is not None and generated_count >= max_generations:
                break

            seed = int(seed)

            if (prompt_id, seed) in done:
                logger.info("Skipping existing run prompt_id=%s seed=%d", prompt_id, seed)
                continue

            image_path = images_dir / f"{prompt_id}__seed_{seed}.png"

            if image_path.exists() and not args.overwrite:
                logger.info("Image already exists, skipping prompt_id=%s seed=%d", prompt_id, seed)
                continue

            logger.info("Running prompt_id=%s split=%s seed=%d", prompt_id, split, seed)

            result = None
            image = None
            generator = None
            recorder = None
            original_forward = None

            try:
                generator = torch.Generator(device=device).manual_seed(seed)
                recorder = InternalSignalRecorder(
                    early_steps=args.early_steps,
                    guidance_scale=args.guidance_scale,
                )

                # Hook transformer.forward to capture denoiser and CFG-related signals.
                original_forward = pipe.transformer.forward

                def wrapped_forward(*f_args, **f_kwargs):
                    out = original_forward(*f_args, **f_kwargs)
                    recorder.transformer_hook(out)
                    return out

                pipe.transformer.forward = wrapped_forward

                def on_step_end(pipe_obj, step_index: int, timestep: int, callback_kwargs: dict):
                    latents = callback_kwargs["latents"]
                    recorder.callback(step_index, int(timestep), latents)
                    return callback_kwargs

                started = time.time()

                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )

                total_runtime_sec = time.time() - started

                image = result.images[0]
                image.save(image_path)

                manifest_row = {
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "local_prompt_index": int(local_row_idx),
                    "replicate_id": int(replicate_id),
                    "prompt": prompt,
                    "split": split,
                    "seed": seed,
                    "seeds_per_prompt": args.seeds_per_prompt,
                    "seed_stride": args.seed_stride,
                    "model_id": args.model_id,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "height": args.height,
                    "width": args.width,
                    "image_path": str(image_path),
                    "total_runtime_sec": round(total_runtime_sec, 6),
                    "status": "ok",
                }
                append_row(manifest_csv, manifest_row)

                signal_row = recorder.to_summary_row()

                traj_row = {
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "local_prompt_index": int(local_row_idx),
                    "replicate_id": int(replicate_id),
                    "split": split,
                    "seed": seed,
                    "early_steps": args.early_steps,
                    **signal_row,
                    "total_runtime_sec": round(total_runtime_sec, 6),
                }
                append_row(traj_csv, traj_row)

                generated_count += 1

                rss_gb = process.memory_info().rss / (1024 ** 3)
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

                logger.info(
                    "Memory after prompt_id=%s seed=%d | host_rss=%.2f GB | gpu_alloc=%.2f GB",
                    prompt_id, seed, rss_gb, vram_gb
                )

                logger.info(
                    "Done prompt_id=%s seed=%d runtime=%.2fs generated=%d cfg_div_mean=%.6f consistency=%.6f",
                    prompt_id,
                    seed,
                    total_runtime_sec,
                    generated_count,
                    signal_row.get("cfg_divergence_mean", 0.0),
                    signal_row.get("denoising_consistency_residual_mean", 0.0),
                )

            except Exception as e:
                logger.exception("FAILED prompt_id=%s seed=%d", prompt_id, seed)
                fail_row = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "split": split,
                    "seed": seed,
                    "model_id": args.model_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                append_row(fail_csv, fail_row)

            finally:
                # Always restore original transformer.forward.
                if original_forward is not None:
                    pipe.transformer.forward = original_forward

                if image is not None:
                    del image
                if result is not None:
                    del result
                if generator is not None:
                    del generator
                if recorder is not None:
                    del recorder

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("Batch run complete. Generated %d new images.", generated_count)


if __name__ == "__main__":
    main()