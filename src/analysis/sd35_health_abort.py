"""
sd35_health_abort.py

A small importable utility for early-aborting Stable Diffusion 3.5 generations
using a fixed, training-free trajectory health score.

Key idea
--------
1. Run SD3.5 normally for the first K denoising steps.
2. Record internal trajectory signals from latents and the transformer output.
3. Compute a fixed health score from those early signals.
4. Abort the generation if the score is below a chosen threshold.

This module does NOT train a CNN/probe. The score formula is fixed.
For practical use, you should provide a calibration JSON containing feature
means/stds and raw score min/max estimated from a separate calibration set.
That calibration only fixes the scale; it does not fit feature weights.

Typical usage
-------------
Calibrate from a previous trajectory CSV:

    python sd35_health_abort.py calibrate \
      --input-csv results/rich_analysis_clip_250x5_prefix5_v6/rich_merged_dataset.csv \
      --output-json results/health_calibration_prefix5.json \
      --prefix-steps 5

Use as a library:

    from pathlib import Path
    from sd35_health_abort import TrainingFreeHealthScorer, SD35HealthAborter

    scorer = TrainingFreeHealthScorer.from_json("results/health_calibration_prefix5.json")
    aborter = SD35HealthAborter(
        scorer=scorer,
        threshold=50.0,
        early_steps=5,
        num_inference_steps=25,
    )

    result = aborter.generate(
        prompt="a glass cathedral in a storm, cinematic lighting",
        seed=12345,
        output_path=Path("out.png"),
    )

    if result.aborted:
        print("Aborted early", result.health_score_0_100)
    else:
        print("Finished", result.health_score_0_100, result.image_path)

Notes
-----
- The default fixed formula mirrors the thesis analysis score:
    positive: early volatility, denoiser change, update/prediction cosine, mild CFG divergence
    negative: excessive latent RMS and denoising inconsistency residual
- The 0-100 health scale is only meaningful with a calibration JSON.
- If you do not have calibration, inspect `raw_health` and choose a raw threshold,
  but that is less portable.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from diffusers import StableDiffusion3Pipeline
except Exception:  # pragma: no cover
    StableDiffusion3Pipeline = None


# ---------------------------------------------------------------------------
# Low-level tensor helpers
# ---------------------------------------------------------------------------


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
    Fit scalar alpha so alpha * pred approximates delta.
    Return ||delta - alpha pred|| / ||delta||.

    Lower means the denoiser direction explains the actual latent update better.
    """
    d = delta.detach().float().flatten()
    p = pred.detach().float().flatten()
    denom = torch.dot(p, p) + eps
    alpha = torch.dot(d, p) / denom
    residual = d - alpha * p
    return float((torch.norm(residual) / (torch.norm(d) + eps)).item())


def safe_mean(vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def slope(vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def parse_json_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, str):
        try:
            return [float(v) for v in json.loads(x)]
        except Exception:
            return []
    try:
        if pd is not None and pd.isna(x):
            return []
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Signal recorder
# ---------------------------------------------------------------------------


class InternalSignalRecorder:
    """Records early latent, denoiser, CFG, and consistency signals."""

    def __init__(self, early_steps: int, guidance_scale: float):
        self.early_steps = int(early_steps)
        self.guidance_scale = float(guidance_scale)

        self.timesteps: list[int] = []
        self.step_times_sec: list[float] = []

        self.latent_rms: list[float] = []
        self.latent_std: list[float] = []
        self.latent_abs_mean: list[float] = []
        self.latent_volatility_rms: list[float] = []
        self.latent_update_cosine: list[float] = []

        self.denoiser_pred_rms: list[float] = []
        self.denoiser_pred_std: list[float] = []
        self.denoiser_pred_abs_mean: list[float] = []
        self.denoiser_pred_delta_rms: list[float] = []
        self.denoiser_pred_cosine_prev: list[float] = []

        self.cfg_divergence_rms: list[float] = []
        self.cfg_divergence_abs_mean: list[float] = []
        self.cfg_divergence_relative: list[float] = []
        self.cfg_alignment_cosine: list[float] = []
        self.guided_minus_cond_rms: list[float] = []
        self.cfg_chunk_count: list[int] = []

        self.denoising_consistency_residual: list[float] = []
        self.denoising_update_pred_cosine: list[float] = []

        self._prev_latents_cpu: Optional[torch.Tensor] = None
        self._prev_delta_cpu: Optional[torch.Tensor] = None
        self._prev_pred_cpu: Optional[torch.Tensor] = None
        self._last_pred_cpu: Optional[torch.Tensor] = None
        self._last_callback_time: Optional[float] = None
        self._hook_step_count = 0

    def transformer_hook(self, output: Any) -> None:
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
                return

            pred_cpu = pred.detach().to("cpu", dtype=torch.float32)
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
                self.cfg_divergence_relative.append(tensor_rms(cfg_delta) / (guided_rms + 1e-8))
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
                self.denoiser_pred_delta_rms.append(tensor_rms(pred_for_consistency - self._prev_pred_cpu))
                self.denoiser_pred_cosine_prev.append(tensor_cosine(pred_for_consistency, self._prev_pred_cpu))

            self._prev_pred_cpu = pred_for_consistency.clone()
            self._last_pred_cpu = pred_for_consistency.clone()
        except Exception:
            # Never let signal recording crash generation.
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

    def raw_sequences(self) -> dict[str, list[float] | list[int]]:
        return {
            "timesteps": self.timesteps,
            "latent_rms": self.latent_rms,
            "latent_std": self.latent_std,
            "latent_abs_mean": self.latent_abs_mean,
            "latent_volatility_rms": self.latent_volatility_rms,
            "latent_update_cosine": self.latent_update_cosine,
            "denoiser_pred_rms": self.denoiser_pred_rms,
            "denoiser_pred_std": self.denoiser_pred_std,
            "denoiser_pred_abs_mean": self.denoiser_pred_abs_mean,
            "denoiser_pred_delta_rms": self.denoiser_pred_delta_rms,
            "denoiser_pred_cosine_prev": self.denoiser_pred_cosine_prev,
            "cfg_divergence_rms": self.cfg_divergence_rms,
            "cfg_divergence_abs_mean": self.cfg_divergence_abs_mean,
            "cfg_divergence_relative": self.cfg_divergence_relative,
            "cfg_alignment_cosine": self.cfg_alignment_cosine,
            "guided_minus_cond_rms": self.guided_minus_cond_rms,
            "denoising_consistency_residual": self.denoising_consistency_residual,
            "denoising_update_pred_cosine": self.denoising_update_pred_cosine,
        }


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HealthResult:
    raw_health: float
    health_score_0_100: Optional[float]
    features: dict[str, float]
    terms: dict[str, float]


class TrainingFreeHealthScorer:
    """
    Fixed training-free health formula.

    Calibration stores feature mean/std and raw score min/max.
    It does not fit weights.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "latent_volatility_rms_prefix{K}_mean": 1.00,
        "latent_volatility_rms_prefix{K}_slope": 0.50,
        "denoiser_pred_delta_rms_prefix{K}_mean": 0.75,
        "denoising_update_pred_cosine_prefix{K}_mean": 0.50,
        "latent_rms_prefix{K}_mean": -0.25,
        "latent_rms_prefix{K}_slope": -0.15,
        "denoising_consistency_residual_prefix{K}_mean": -0.75,
        "cfg_divergence_rms_prefix{K}_mean": 0.25,
    }

    def __init__(
        self,
        prefix_steps: int = 5,
        feature_means: Optional[dict[str, float]] = None,
        feature_stds: Optional[dict[str, float]] = None,
        raw_min: Optional[float] = None,
        raw_max: Optional[float] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.prefix_steps = int(prefix_steps)
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingFreeHealthScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            prefix_steps=int(data["prefix_steps"]),
            feature_means={k: float(v) for k, v in data.get("feature_means", {}).items()},
            feature_stds={k: float(v) for k, v in data.get("feature_stds", {}).items()},
            raw_min=float(data["raw_min"]) if data.get("raw_min") is not None else None,
            raw_max=float(data["raw_max"]) if data.get("raw_max") is not None else None,
            weights={k: float(v) for k, v in data.get("weights", cls.DEFAULT_WEIGHTS).items()},
        )

    def to_json(self, path: str | Path) -> None:
        data = {
            "prefix_steps": self.prefix_steps,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _feature_name(self, template: str) -> str:
        return template.format(K=self.prefix_steps)

    def extract_features_from_recorder(self, recorder: InternalSignalRecorder) -> dict[str, float]:
        seqs = recorder.raw_sequences()
        return self.extract_features_from_sequences(seqs)

    def extract_features_from_sequences(self, seqs: dict[str, list[float]]) -> dict[str, float]:
        features: dict[str, float] = {}
        for signal_name, values in seqs.items():
            if signal_name == "timesteps":
                continue
            vals = [float(v) for v in values[: self.prefix_steps] if v is not None and not np.isnan(v)]
            prefix = f"{signal_name}_prefix{self.prefix_steps}"
            features[f"{prefix}_mean"] = safe_mean(vals)
            features[f"{prefix}_slope"] = slope(vals)
            features[f"{prefix}_last"] = float(vals[-1]) if vals else np.nan
        return features

    def _z(self, name: str, value: float) -> float:
        if np.isnan(value):
            return 0.0
        if name in self.feature_means and name in self.feature_stds:
            return (value - self.feature_means[name]) / (self.feature_stds[name] + 1e-8)
        # Fallback: uncalibrated use. This makes raw_health dataset-dependent.
        return float(value)

    def score_features(self, features: dict[str, float]) -> HealthResult:
        raw = 0.0
        terms: dict[str, float] = {}
        for template, weight in self.weights.items():
            name = self._feature_name(template)
            value = features.get(name, np.nan)
            term = float(weight) * self._z(name, float(value))
            terms[name] = term
            raw += term

        score = None
        if self.raw_min is not None and self.raw_max is not None and not np.isclose(self.raw_min, self.raw_max):
            score = 100.0 * (raw - self.raw_min) / (self.raw_max - self.raw_min)
            score = float(np.clip(score, 0.0, 100.0))

        return HealthResult(raw_health=float(raw), health_score_0_100=score, features=features, terms=terms)

    def score_recorder(self, recorder: InternalSignalRecorder) -> HealthResult:
        return self.score_features(self.extract_features_from_recorder(recorder))

    @classmethod
    def calibrate_from_csv(
        cls,
        input_csv: str | Path,
        output_json: str | Path,
        prefix_steps: int = 5,
    ) -> "TrainingFreeHealthScorer":
        if pd is None:
            raise RuntimeError("pandas is required for calibration")
        df = pd.read_csv(input_csv)
        scorer = cls(prefix_steps=prefix_steps)

        seq_col_map = {
            "latent_rms": "latent_rms_json",
            "latent_volatility_rms": "latent_volatility_rms_json",
            "denoiser_pred_delta_rms": "denoiser_pred_delta_rms_json",
            "denoising_update_pred_cosine": "denoising_update_pred_cosine_json",
            "denoising_consistency_residual": "denoising_consistency_residual_json",
            "cfg_divergence_rms": "cfg_divergence_rms_json",
        }

        feature_rows: list[dict[str, float]] = []
        for _, row in df.iterrows():
            seqs = {}
            for signal_name, col in seq_col_map.items():
                if col in df.columns:
                    seqs[signal_name] = parse_json_list(row[col])
            feature_rows.append(scorer.extract_features_from_sequences(seqs))

        feat_df = pd.DataFrame(feature_rows)
        needed = [scorer._feature_name(t) for t in scorer.weights.keys()]
        needed = [c for c in needed if c in feat_df.columns]

        feature_means = {c: float(feat_df[c].mean()) for c in needed}
        feature_stds = {c: float(feat_df[c].std(ddof=0)) for c in needed}

        calibrated = cls(
            prefix_steps=prefix_steps,
            feature_means=feature_means,
            feature_stds=feature_stds,
            weights=scorer.weights.copy(),
        )
        raw_scores = []
        for features in feature_rows:
            raw_scores.append(calibrated.score_features(features).raw_health)
        calibrated.raw_min = float(np.nanmin(raw_scores))
        calibrated.raw_max = float(np.nanmax(raw_scores))
        calibrated.to_json(output_json)
        return calibrated


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------


class AbortGeneration(Exception):
    pass


@dataclasses.dataclass
class GenerationResult:
    prompt: str
    seed: Optional[int]
    aborted: bool
    abort_step: Optional[int]
    health_score_0_100: Optional[float]
    raw_health: Optional[float]
    image: Optional[Image.Image]
    image_path: Optional[str]
    runtime_sec: float
    metadata: dict[str, Any]


class SD35HealthAborter:
    """Wrapper around StableDiffusion3Pipeline with early health-based aborting."""

    def __init__(
        self,
        scorer: TrainingFreeHealthScorer,
        threshold: float,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        early_steps: int = 5,
        height: int = 1024,
        width: int = 1024,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        pipe: Optional[Any] = None,
    ):
        if StableDiffusion3Pipeline is None and pipe is None:
            raise RuntimeError("diffusers is required unless an existing pipe is passed")

        self.scorer = scorer
        self.threshold = float(threshold)
        self.model_id = model_id
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.early_steps = int(early_steps)
        self.height = int(height)
        self.width = int(width)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.pipe = pipe
        if self.pipe is None:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            self.pipe = self.pipe.to(self.device)
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        output_path: Optional[str | Path] = None,
        negative_prompt: Optional[str] = None,
        extra_pipe_kwargs: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        recorder = InternalSignalRecorder(
            early_steps=self.early_steps,
            guidance_scale=self.guidance_scale,
        )
        state: dict[str, Any] = {
            "health": None,
            "abort_step": None,
            "decision_made": False,
        }
        original_forward = self.pipe.transformer.forward

        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            recorder.transformer_hook(out)
            return out

        def on_step_end(pipe_obj, step_index: int, timestep: int, callback_kwargs: dict):
            latents = callback_kwargs["latents"]
            recorder.callback(step_index, int(timestep), latents)

            if (step_index + 1) >= self.early_steps and not state["decision_made"]:
                health = self.scorer.score_recorder(recorder)
                state["health"] = health
                state["abort_step"] = int(step_index + 1)
                state["decision_made"] = True

                # Prefer calibrated 0-100 score. If unavailable, compare raw health.
                value = health.health_score_0_100 if health.health_score_0_100 is not None else health.raw_health
                if value < self.threshold:
                    raise AbortGeneration(
                        f"Aborted at step {step_index + 1}: health={value:.4f} < threshold={self.threshold:.4f}"
                    )
            return callback_kwargs

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        started = time.time()
        image = None
        aborted = False
        metadata: dict[str, Any] = {}

        try:
            self.pipe.transformer.forward = wrapped_forward
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.height,
                width=self.width,
                generator=generator,
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            if negative_prompt is not None:
                kwargs["negative_prompt"] = negative_prompt
            if extra_pipe_kwargs:
                kwargs.update(extra_pipe_kwargs)

            result = self.pipe(**kwargs)
            image = result.images[0]
            if state["health"] is None:
                state["health"] = self.scorer.score_recorder(recorder)

            if output_path is not None:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                image.save(out)
                image_path = str(out)
            else:
                image_path = None

        except AbortGeneration as e:
            aborted = True
            image_path = None
            metadata["abort_reason"] = str(e)
        finally:
            self.pipe.transformer.forward = original_forward
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        runtime = time.time() - started
        health: Optional[HealthResult] = state.get("health")
        if health is not None:
            metadata["health_terms"] = health.terms
            metadata["health_features"] = health.features

        return GenerationResult(
            prompt=prompt,
            seed=seed,
            aborted=aborted,
            abort_step=state.get("abort_step"),
            health_score_0_100=health.health_score_0_100 if health is not None else None,
            raw_health=health.raw_health if health is not None else None,
            image=image,
            image_path=image_path,
            runtime_sec=runtime,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# CLI: calibration utility
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="SD3.5 training-free health abort utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cal = sub.add_parser("calibrate", help="Create health-score calibration JSON from trajectory CSV")
    cal.add_argument("--input-csv", type=Path, required=True)
    cal.add_argument("--output-json", type=Path, required=True)
    cal.add_argument("--prefix-steps", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "calibrate":
        scorer = TrainingFreeHealthScorer.calibrate_from_csv(
            input_csv=args.input_csv,
            output_json=args.output_json,
            prefix_steps=args.prefix_steps,
        )
        print(f"Wrote calibration to: {args.output_json}")
        print(f"prefix_steps={scorer.prefix_steps}")
        print(f"raw_min={scorer.raw_min:.6f} raw_max={scorer.raw_max:.6f}")


if __name__ == "__main__":
    _main()
