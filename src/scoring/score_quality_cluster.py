from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------------------------------------------------------------
# Workaround for hpsv2 bundled open_clip bug:
# hpsv2/src/open_clip/factory.py does `from turtle import forward`,
# which imports tkinter and fails on headless HPC Python builds.
# The imported `forward` is unused for HPS scoring, so we stub it.
# ---------------------------------------------------------------------
import sys
import types

if "turtle" not in sys.modules:
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.forward = lambda *args, **kwargs: None
    sys.modules["turtle"] = turtle_stub

# CLIP imports are reasonably lightweight; HPSv2 and ImageReward are imported lazily.
from transformers import CLIPModel, CLIPProcessor


# -----------------------------------------------------------------------------
# Cluster quality scorer for SD3.5 trajectory runs
# -----------------------------------------------------------------------------
# Expected run directory:
#   results/sd35_trajectories_rich_250x5/
#   ├── generation_manifest.csv   # prompt_id, seed, prompt, image_path, ...
#   ├── trajectory_signals.csv     # prompt_id, seed, trajectory JSON columns, ...
#   └── images/
#       ├── p001__seed_12345.png
#       └── ...
#
# Computes one or more text-image quality metrics on the cluster:
#   --quality-targets clip
#   --quality-targets hpsv2
#   --quality-targets imagereward
#   --quality-targets clip,hpsv2,imagereward
#
# Output columns:
#   clip        -> new_clip_score, clip_model_name
#   hpsv2       -> hpsv2_score, hpsv2_version
#   imagereward -> image_reward_score, image_reward_model_name
#
# Writes:
#   generation_manifest_with_quality.csv
#   trajectory_signals_with_quality.csv
#   quality_scoring_summary.json
#
# Notes:
# - CLIP loading uses safetensors to avoid torch<2.6 torch.load restrictions.
# - HPSv2 and ImageReward are optional. Install only what you need.
# - The path resolver is robust to old absolute cluster paths in image_path.
# -----------------------------------------------------------------------------


TARGET_TO_SCORE_COL = {
    "clip": "new_clip_score",
    "hpsv2": "hpsv2_score",
    "imagereward": "image_reward_score",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("quality_cluster_score")
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


def parse_targets(value: str) -> list[str]:
    targets = []
    for part in value.split(","):
        t = part.strip().lower()
        if not t:
            continue
        if t not in TARGET_TO_SCORE_COL:
            raise ValueError(f"Unknown quality target {t!r}. Valid: {sorted(TARGET_TO_SCORE_COL)}")
        if t not in targets:
            targets.append(t)
    if not targets:
        raise ValueError("No quality targets supplied.")
    return targets


def resolve_image_path(raw_path: str, run_dir: Path, image_root: Optional[Path]) -> Path:
    raw = Path(str(raw_path))
    candidates: list[Path] = []

    candidates.append(raw)

    if image_root is not None:
        candidates.append(image_root / raw)
        candidates.append(image_root / raw.name)

    candidates.append(run_dir / raw)
    candidates.append(run_dir / "images" / raw.name)
    candidates.append(run_dir / raw.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not resolve image path. Tried: " + "; ".join(str(c) for c in candidates)
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class CLIPScorer:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name
        # Force safetensors so Transformers does not call torch.load on .bin weights
        # in older cluster environments with torch<2.6.
        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, prompts: list[str], image_paths: list[Path]) -> np.ndarray:
        images = [load_image(p) for p in image_paths]
        text_inputs = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        image_inputs = self.processor.image_processor(images, return_tensors="pt")

        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
        }

        outputs = self.model(**inputs)
        text_emb = outputs.text_embeds
        image_emb = outputs.image_embeds
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        sims = (text_emb * image_emb).sum(dim=-1)
        return sims.detach().cpu().numpy()


class HPSv2Scorer:
    def __init__(self, hps_version: str, device: str):
        self.hps_version = hps_version
        self.device = device
        import hpsv2  # type: ignore
        self.hpsv2 = hpsv2

    def score_one(self, prompt: str, image_path: Path) -> float:
        # hpsv2.score returns list/array-like for list of image paths.
        val = self.hpsv2.score([str(image_path)], prompt, hps_version=self.hps_version)
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0])
        return float(val)

    def score_batch(self, prompts: list[str], image_paths: list[Path]) -> np.ndarray:
        scores = [self.score_one(prompt, path) for prompt, path in zip(prompts, image_paths)]
        return np.asarray(scores, dtype=float)


class ImageRewardScorer:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        import ImageReward as RM  # type: ignore
        self.RM = RM
        self.model = RM.load(model_name)
        # ImageReward load normally places model on cuda if available, but be explicit when possible.
        try:
            self.model = self.model.to(device)
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    def score_one(self, prompt: str, image_path: Path) -> float:
        with torch.no_grad():
            val = self.model.score(prompt, str(image_path))
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0])
        return float(val)

    def score_batch(self, prompts: list[str], image_paths: list[Path]) -> np.ndarray:
        scores = [self.score_one(prompt, path) for prompt, path in zip(prompts, image_paths)]
        return np.asarray(scores, dtype=float)


def build_scorer(target: str, args: argparse.Namespace, logger: logging.Logger):
    if target == "clip":
        logger.info("Loading CLIP scorer: %s", args.clip_model_name)
        return CLIPScorer(args.clip_model_name, args.device)
    if target == "hpsv2":
        logger.info("Loading HPSv2 scorer: %s", args.hps_version)
        return HPSv2Scorer(args.hps_version, args.device)
    if target == "imagereward":
        logger.info("Loading ImageReward scorer: %s", args.image_reward_model_name)
        return ImageRewardScorer(args.image_reward_model_name, args.device)
    raise ValueError(target)


def ensure_score_columns(manifest: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    out = manifest.copy()
    for target in targets:
        score_col = TARGET_TO_SCORE_COL[target]
        if score_col not in out.columns:
            out[score_col] = np.nan

        if target == "clip" and "clip_model_name" not in out.columns:
            out["clip_model_name"] = ""
        if target == "hpsv2" and "hpsv2_version" not in out.columns:
            out["hpsv2_version"] = ""
        if target == "imagereward" and "image_reward_model_name" not in out.columns:
            out["image_reward_model_name"] = ""

    if "image_path_resolved" not in out.columns:
        out["image_path_resolved"] = ""
    return out


def merge_resume_scores(
    manifest: pd.DataFrame,
    output_manifest_csv: Path,
    targets: list[str],
) -> pd.DataFrame:
    if not output_manifest_csv.exists():
        return manifest

    old = pd.read_csv(output_manifest_csv)
    if not {"prompt_id", "seed"}.issubset(old.columns):
        return manifest

    keep_cols = ["prompt_id", "seed"]
    for target in targets:
        score_col = TARGET_TO_SCORE_COL[target]
        if score_col in old.columns:
            keep_cols.append(score_col)
        for meta_col in ["clip_model_name", "hpsv2_version", "image_reward_model_name", "image_path_resolved"]:
            if meta_col in old.columns and meta_col not in keep_cols:
                keep_cols.append(meta_col)

    if len(keep_cols) <= 2:
        return manifest

    # Avoid _x/_y confusion: drop old score cols from current manifest before merging.
    drop_cols = [c for c in keep_cols if c not in {"prompt_id", "seed"} and c in manifest.columns]
    base = manifest.drop(columns=drop_cols, errors="ignore")
    return base.merge(
        old[keep_cols].drop_duplicates(subset=["prompt_id", "seed"]),
        on=["prompt_id", "seed"],
        how="left",
    )
    

def ensure_metadata_columns_are_object(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()

    for col in ["clip_model_name", "hpsv2_version", "image_reward_model_name", "image_path_resolved"]:
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")

    return out


def save_outputs(
    manifest: pd.DataFrame,
    trajectory_csv: Path,
    output_manifest_csv: Path,
    output_merged_csv: Path,
    targets: list[str],
    logger: logging.Logger,
) -> None:
    manifest.to_csv(output_manifest_csv, index=False)

    if trajectory_csv.exists():
        traj = pd.read_csv(trajectory_csv)
        merge_cols = ["prompt_id", "seed", "image_path_resolved"]
        for target in targets:
            score_col = TARGET_TO_SCORE_COL[target]
            if score_col in manifest.columns:
                merge_cols.append(score_col)
        for c in ["clip_model_name", "hpsv2_version", "image_reward_model_name", "prompt", "image_path", "prompt_index", "replicate_id"]:
            if c in manifest.columns and c not in merge_cols:
                merge_cols.append(c)

        scored = manifest[merge_cols].drop_duplicates(subset=["prompt_id", "seed"])
        merged = traj.merge(scored, on=["prompt_id", "seed"], how="left", suffixes=("", "_manifest"))
        merged.to_csv(output_merged_csv, index=False)
        logger.info("Saved trajectory+quality CSV: %s", output_merged_csv)
    else:
        logger.warning("Trajectory CSV not found, skipping merged output: %s", trajectory_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-merged-csv", type=Path, default=None)

    parser.add_argument(
        "--quality-targets",
        default="clip",
        help="Comma-separated targets: clip,hpsv2,imagereward. Example: --quality-targets clip,hpsv2",
    )
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--hps-version", default="v2.1")
    parser.add_argument("--image-reward-model-name", default="ImageReward-v1.0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Reuse scores already present in output-manifest-csv.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--save-every-batches", type=int, default=1, help="Save progress every N batches.")
    args = parser.parse_args()

    targets = parse_targets(args.quality_targets)
    run_dir = args.run_dir.resolve()
    logs_dir = run_dir / "logs"
    ensure_dir(logs_dir)
    logger = setup_logger(logs_dir / "quality_scoring.log")

    manifest_csv = args.manifest_csv or (run_dir / "generation_manifest.csv")
    trajectory_csv = args.trajectory_csv or (run_dir / "trajectory_signals.csv")
    output_manifest_csv = args.output_manifest_csv or (run_dir / "generation_manifest_with_quality.csv")
    output_merged_csv = args.output_merged_csv or (run_dir / "trajectory_signals_with_quality.csv")

    logger.info("Quality targets: %s", ",".join(targets))
    logger.info("Loading manifest: %s", manifest_csv)
    manifest = pd.read_csv(manifest_csv)
    required = {"prompt_id", "seed", "prompt", "image_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    if args.resume:
        manifest = merge_resume_scores(manifest, output_manifest_csv, targets)

    manifest = ensure_score_columns(manifest, targets)
    manifest = ensure_metadata_columns_are_object(manifest)

    logger.info("Rows in manifest: %d", len(manifest))
    logger.info("Device: %s", args.device)
    logger.info("Output manifest: %s", output_manifest_csv)
    logger.info("Output merged CSV: %s", output_merged_csv)

    for target in targets:
        score_col = TARGET_TO_SCORE_COL[target]
        to_score = manifest[manifest[score_col].isna()].copy()
        logger.info("Rows needing %s score (%s): %d", target, score_col, len(to_score))
        if len(to_score) == 0:
            continue

        scorer = build_scorer(target, args, logger)
        t_target = time.time()

        for batch_num, start in enumerate(range(0, len(to_score), args.batch_size), start=1):
            batch = to_score.iloc[start:start + args.batch_size]
            prompts = batch["prompt"].astype(str).tolist()
            paths = [
                resolve_image_path(p, run_dir=run_dir, image_root=args.image_root)
                for p in batch["image_path"].tolist()
            ]
            t0 = time.time()
            scores = scorer.score_batch(prompts, paths)
            elapsed = time.time() - t0

            for idx, score, path in zip(batch.index, scores, paths):
                manifest.loc[idx, score_col] = float(score)
                manifest.loc[idx, "image_path_resolved"] = str(path)
                if target == "clip":
                    manifest.loc[idx, "clip_model_name"] = args.clip_model_name
                elif target == "hpsv2":
                    manifest.loc[idx, "hpsv2_version"] = args.hps_version
                elif target == "imagereward":
                    manifest.loc[idx, "image_reward_model_name"] = args.image_reward_model_name

            done = min(start + len(batch), len(to_score))
            logger.info(
                "%s: scored %d/%d rows | batch_size=%d | batch_time=%.2fs | %.2fs/image",
                target,
                done,
                len(to_score),
                len(batch),
                elapsed,
                elapsed / max(len(batch), 1),
            )

            if batch_num % max(args.save_every_batches, 1) == 0:
                save_outputs(manifest, trajectory_csv, output_manifest_csv, output_merged_csv, targets, logger)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Finished %s in %.2f minutes", target, (time.time() - t_target) / 60.0)
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_outputs(manifest, trajectory_csv, output_manifest_csv, output_merged_csv, targets, logger)
    logger.info("Saved scored manifest: %s", output_manifest_csv)

    summary = {
        "run_dir": str(run_dir),
        "manifest_csv": str(manifest_csv),
        "trajectory_csv": str(trajectory_csv),
        "output_manifest_csv": str(output_manifest_csv),
        "output_merged_csv": str(output_merged_csv),
        "quality_targets": targets,
        "clip_model_name": args.clip_model_name,
        "hps_version": args.hps_version,
        "image_reward_model_name": args.image_reward_model_name,
        "n_rows": int(len(manifest)),
    }
    for target in targets:
        score_col = TARGET_TO_SCORE_COL[target]
        if score_col in manifest.columns:
            s = pd.to_numeric(manifest[score_col], errors="coerce")
            summary[f"n_scored_{target}"] = int(s.notna().sum())
            summary[f"mean_{score_col}"] = float(s.mean()) if s.notna().any() else None
            summary[f"std_{score_col}"] = float(s.std()) if s.notna().any() else None
            summary[f"min_{score_col}"] = float(s.min()) if s.notna().any() else None
            summary[f"max_{score_col}"] = float(s.max()) if s.notna().any() else None

    (run_dir / "quality_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
