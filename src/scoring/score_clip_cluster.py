from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# -----------------------------------------------------------------------------
# Cluster CLIP scorer for SD3.5 trajectory runs
# -----------------------------------------------------------------------------
# Expected run directory:
#   results/sd35_trajectories_rich_250x5/
#   ├── generation_manifest.csv   # prompt_id, seed, prompt, image_path, ...
#   ├── trajectory_signals.csv     # prompt_id, seed, trajectory JSON columns, ...
#   └── images/
#       ├── p001__seed_12345.png
#       └── ...
#
# This script computes CLIP image-text scores on the cluster so images do not
# have to be downloaded locally. It writes:
#   generation_manifest_with_clip.csv
#   trajectory_signals_with_clip.csv
#
# The path resolver is robust to old absolute image_path values from a different
# machine/cluster path: it tries the exact path first, then image_root/filename,
# run_dir/images/filename, etc.
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("clip_cluster_score")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-merged-csv", type=Path, default=None)
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Reuse scores already present in output-manifest-csv.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    logs_dir = run_dir / "logs"
    ensure_dir(logs_dir)
    logger = setup_logger(logs_dir / "clip_scoring.log")

    manifest_csv = args.manifest_csv or (run_dir / "generation_manifest.csv")
    trajectory_csv = args.trajectory_csv or (run_dir / "trajectory_signals.csv")
    output_manifest_csv = args.output_manifest_csv or (run_dir / "generation_manifest_with_clip.csv")
    output_merged_csv = args.output_merged_csv or (run_dir / "trajectory_signals_with_clip.csv")

    logger.info("Loading manifest: %s", manifest_csv)
    manifest = pd.read_csv(manifest_csv)
    required = {"prompt_id", "seed", "prompt", "image_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    # Resume by merging old scores when available.
    if args.resume and output_manifest_csv.exists():
        old = pd.read_csv(output_manifest_csv)
        if {"prompt_id", "seed", "new_clip_score"}.issubset(old.columns):
            keep_cols = ["prompt_id", "seed", "new_clip_score", "clip_model_name", "image_path_resolved"]
            keep_cols = [c for c in keep_cols if c in old.columns]
            manifest = manifest.merge(
                old[keep_cols].drop_duplicates(subset=["prompt_id", "seed"]),
                on=["prompt_id", "seed"],
                how="left",
            )

    if "new_clip_score" not in manifest.columns:
        manifest["new_clip_score"] = np.nan
    if "clip_model_name" not in manifest.columns:
        manifest["clip_model_name"] = ""
    if "image_path_resolved" not in manifest.columns:
        manifest["image_path_resolved"] = ""

    to_score = manifest[manifest["new_clip_score"].isna()].copy()
    logger.info("Rows in manifest: %d", len(manifest))
    logger.info("Rows needing CLIP score: %d", len(to_score))
    logger.info("Device: %s", args.device)

    scorer = CLIPScorer(args.clip_model_name, args.device)

    for start in range(0, len(to_score), args.batch_size):
        batch = to_score.iloc[start:start + args.batch_size]
        prompts = batch["prompt"].astype(str).tolist()
        paths = [
            resolve_image_path(p, run_dir=run_dir, image_root=args.image_root)
            for p in batch["image_path"].tolist()
        ]
        scores = scorer.score_batch(prompts, paths)

        for idx, score, path in zip(batch.index, scores, paths):
            manifest.loc[idx, "new_clip_score"] = float(score)
            manifest.loc[idx, "clip_model_name"] = args.clip_model_name
            manifest.loc[idx, "image_path_resolved"] = str(path)

        manifest.to_csv(output_manifest_csv, index=False)
        logger.info("Scored %d/%d rows", min(start + len(batch), len(to_score)), len(to_score))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest.to_csv(output_manifest_csv, index=False)
    logger.info("Saved scored manifest: %s", output_manifest_csv)

    if trajectory_csv.exists():
        traj = pd.read_csv(trajectory_csv)
        merge_cols = [
            "prompt_id", "seed", "new_clip_score", "clip_model_name", "image_path_resolved"
        ]
        extra_cols = ["prompt", "image_path", "prompt_index", "replicate_id"]
        for c in extra_cols:
            if c in manifest.columns and c not in merge_cols:
                merge_cols.append(c)

        scored = manifest[merge_cols].drop_duplicates(subset=["prompt_id", "seed"])
        merged = traj.merge(scored, on=["prompt_id", "seed"], how="left", suffixes=("", "_manifest"))
        merged.to_csv(output_merged_csv, index=False)
        logger.info("Saved trajectory+CLIP CSV: %s", output_merged_csv)
    else:
        logger.warning("Trajectory CSV not found, skipping merged output: %s", trajectory_csv)

    summary = {
        "run_dir": str(run_dir),
        "manifest_csv": str(manifest_csv),
        "trajectory_csv": str(trajectory_csv),
        "output_manifest_csv": str(output_manifest_csv),
        "output_merged_csv": str(output_merged_csv),
        "clip_model_name": args.clip_model_name,
        "n_rows": int(len(manifest)),
        "n_scored": int(manifest["new_clip_score"].notna().sum()),
        "mean_clip": float(manifest["new_clip_score"].mean()),
    }
    (run_dir / "clip_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
