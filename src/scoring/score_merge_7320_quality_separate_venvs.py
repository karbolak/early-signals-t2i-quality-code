#!/usr/bin/env python3
"""
Score SD3.5 7320-prompt trajectory chunks and merge all 5 seed passes.

This version is designed for separate environments:
  - CLIP + HPSv2 in ~/venvs/bscthesis
  - ImageReward in a separate ImageReward environment

Directory layout expected:
  results/sd35_7320_1seed/chunk_0/generation_manifest.csv
  results/sd35_7320_1seed/chunk_0/trajectory_signals.csv
  ...
  results/sd35_7320_5thseed/chunk_7/generation_manifest.csv
  results/sd35_7320_5thseed/chunk_7/trajectory_signals.csv

Output columns match the earlier score_quality_cluster.py conventions:
  CLIP        -> new_clip_score, clip_model_name
  HPSv2       -> hpsv2_score, hpsv2_version
  ImageReward -> image_reward_score, image_reward_model_name
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

# hpsv2 bundled open_clip can import turtle, which can pull tkinter on headless HPC.
# The imported forward symbol is unused for scoring, so stub it before hpsv2 import.
if "turtle" not in sys.modules:
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.forward = lambda *args, **kwargs: None
    sys.modules["turtle"] = turtle_stub

LOG = logging.getLogger("score_merge_7320_quality")

PASS_DIRS_DEFAULT = [
    "sd35_7320_1seed",
    "sd35_7320_2ndseed",
    "sd35_7320_3rdseed",
    "sd35_7320_4thseed",
    "sd35_7320_5thseed",
]

IMAGE_PATH_CANDIDATES = [
    "image_path",
    "image_file",
    "output_path",
    "output_image_path",
    "path",
    "filepath",
    "filename",
    "file_name",
]
PROMPT_CANDIDATES = ["prompt", "text", "caption", "prompt_text", "positive_prompt"]

QUALITY_COLUMNS = {
    "clip": "new_clip_score",
    "hpsv2": "hpsv2_score",
    "imagereward": "image_reward_score",
}
VERSION_COLUMNS = {
    "clip": "clip_model_name",
    "hpsv2": "hpsv2_version",
    "imagereward": "image_reward_model_name",
}

# Backward-compatible aliases in case an older draft already wrote these names.
ALIASES_TO_CANONICAL = {
    "clip_score": "new_clip_score",
    "imagereward_score": "image_reward_score",
    "clip_model": "clip_model_name",
    "imagereward_model": "image_reward_model_name",
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_targets(raw: str) -> list[str]:
    targets = [t.strip().lower() for t in raw.split(",") if t.strip()]
    valid = set(QUALITY_COLUMNS)
    bad = [t for t in targets if t not in valid]
    if bad:
        raise ValueError(f"Unknown target(s): {bad}. Valid targets: {sorted(valid)}")
    if not targets:
        raise ValueError("No targets requested.")
    return targets


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {kind} column. Tried {candidates}. Available: {list(df.columns)}")


def adopt_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for old, new in ALIASES_TO_CANONICAL.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def resolve_image_path(raw: object, chunk_dir: Path) -> str:
    if pd.isna(raw):
        return ""
    raw_path = Path(str(raw))
    candidates: list[Path] = []
    candidates.append(raw_path)
    candidates.append(chunk_dir / raw_path)
    candidates.append(chunk_dir / "images" / raw_path.name)
    candidates.append(chunk_dir / raw_path.name)
    candidates.append(Path.cwd() / raw_path)

    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return str((chunk_dir / raw_path).resolve())


def load_best_manifest(chunk_dir: Path) -> tuple[pd.DataFrame, Path]:
    scored = chunk_dir / "generation_manifest_with_quality.csv"
    plain = chunk_dir / "generation_manifest.csv"
    if scored.exists():
        LOG.info("Loading existing scored manifest: %s", scored)
        return adopt_alias_columns(pd.read_csv(scored)), scored
    if plain.exists():
        LOG.info("Loading plain manifest: %s", plain)
        return adopt_alias_columns(pd.read_csv(plain)), plain
    raise FileNotFoundError(f"No generation manifest found in {chunk_dir}")


def ensure_columns(df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    df = adopt_alias_columns(df).copy()
    for target in targets:
        score_col = QUALITY_COLUMNS[target]
        version_col = VERSION_COLUMNS[target]
        if score_col not in df.columns:
            df[score_col] = np.nan
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

        # Important for hpsv2_version='v2.1': make metadata columns object dtype.
        if version_col not in df.columns:
            df[version_col] = pd.Series(pd.NA, index=df.index, dtype="object")
        else:
            df[version_col] = df[version_col].astype("object")

    if "image_path_resolved" not in df.columns:
        df["image_path_resolved"] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df["image_path_resolved"] = df["image_path_resolved"].astype("object")
    return df


def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ClipScorer:
    def __init__(self, device: str, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model_name = model_name
        LOG.info("Loading CLIP scorer: %s", model_name)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def score_batch(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int) -> list[float]:
        out: list[float] = []
        torch = self.torch
        with torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                paths = image_paths[start : start + batch_size]
                texts = [str(x) for x in prompts[start : start + batch_size]]
                images = [load_image(p) for p in paths]
                text_inputs = self.processor.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=77
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
                out.extend(sims.detach().float().cpu().numpy().tolist())
        return out


class HPSv2Scorer:
    def __init__(self, device: str, hps_version: str):
        import hpsv2

        self.hpsv2 = hpsv2
        self.device = device
        self.hps_version = hps_version
        LOG.info("Loading HPSv2 scorer: %s", hps_version)

    def score_batch(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int) -> list[float]:
        out: list[float] = []
        for path, prompt in zip(image_paths, prompts):
            val = self.hpsv2.score([str(path)], str(prompt), hps_version=self.hps_version)
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0]
            out.append(float(val))
        return out


class ImageRewardScorer:
    def __init__(self, device: str, model_name: str):
        import torch
        import ImageReward as RM

        self.torch = torch
        self.device = device
        self.model_name = model_name
        LOG.info("Loading ImageReward scorer: %s", model_name)
        self.model = RM.load(model_name)
        try:
            self.model = self.model.to(device)
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    def score_batch(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for path, prompt in zip(image_paths, prompts):
                val = self.model.score(str(prompt), str(path))
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = val[0]
                out.append(float(val))
        return out


def build_scorer(target: str, args: argparse.Namespace):
    if target == "clip":
        return ClipScorer(args.device, args.clip_model_name)
    if target == "hpsv2":
        return HPSv2Scorer(args.device, args.hps_version)
    if target == "imagereward":
        return ImageRewardScorer(args.device, args.image_reward_model_name)
    raise AssertionError(target)


def save_chunk_trajectory_with_quality(chunk_dir: Path, manifest: pd.DataFrame) -> Path | None:
    traj_path = chunk_dir / "trajectory_signals.csv"
    if not traj_path.exists():
        LOG.warning("No trajectory_signals.csv in %s; skipping trajectory merge", chunk_dir)
        return None

    traj = pd.read_csv(traj_path)
    manifest_for_join = adopt_alias_columns(manifest.copy())
    quality_cols = [
        c for c in ["image_path_resolved", *QUALITY_COLUMNS.values(), *VERSION_COLUMNS.values(), "prompt", "image_path"]
        if c in manifest_for_join.columns
    ]

    join_keys_priority = [
        ["image_path"],
        ["prompt_id", "seed", "replicate_id"],
        ["prompt_id", "seed"],
        ["global_index"],
        ["row_index"],
    ]

    chosen_keys: list[str] | None = None
    for keys in join_keys_priority:
        if all(k in traj.columns for k in keys) and all(k in manifest_for_join.columns for k in keys):
            chosen_keys = keys
            break

    if chosen_keys is None:
        LOG.warning("No stable join keys found in %s; falling back to row-order assignment", chunk_dir)
        merged = traj.copy()
        for col in quality_cols:
            merged[col] = manifest_for_join[col].to_numpy()[: len(merged)]
    else:
        LOG.info("Merging trajectory with quality on keys: %s", chosen_keys)
        right_cols = chosen_keys + [c for c in quality_cols if c not in chosen_keys]
        right = manifest_for_join[right_cols].drop_duplicates(chosen_keys)
        merged = traj.merge(right, on=chosen_keys, how="left", suffixes=("", "_manifest"))

    out_path = chunk_dir / "trajectory_signals_with_quality.csv"
    merged.to_csv(out_path, index=False)
    LOG.info("Wrote %d rows to %s", len(merged), out_path)
    return out_path


def score_chunk(args: argparse.Namespace) -> None:
    chunk_dir = Path(args.chunk_dir).resolve()
    targets = normalize_targets(args.targets)

    manifest, source_path = load_best_manifest(chunk_dir)
    manifest = ensure_columns(manifest, targets)

    prompt_col = args.prompt_col or first_existing_column(manifest, PROMPT_CANDIDATES, "prompt")
    image_col = args.image_col or first_existing_column(manifest, IMAGE_PATH_CANDIDATES, "image path")
    LOG.info("Using prompt column: %s", prompt_col)
    LOG.info("Using image column: %s", image_col)

    resolved_paths = manifest[image_col].map(lambda x: resolve_image_path(x, chunk_dir))
    manifest["image_path_resolved"] = resolved_paths.astype("object")

    missing = [p for p in resolved_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"First missing image path: {missing[0]} ({len(missing)} missing total)")

    out_path = chunk_dir / "generation_manifest_with_quality.csv"
    summary: dict[str, object] = {
        "chunk_dir": str(chunk_dir),
        "source_manifest": str(source_path),
        "rows": int(len(manifest)),
        "targets_requested_this_run": targets,
    }

    for target in targets:
        score_col = QUALITY_COLUMNS[target]
        version_col = VERSION_COLUMNS[target]
        todo_mask = manifest[score_col].isna()
        todo_idx = manifest.index[todo_mask].tolist()
        LOG.info("Rows needing %s (%s): %d", target, score_col, len(todo_idx))
        if not todo_idx:
            continue

        scorer = build_scorer(target, args)
        version_value = {
            "clip": args.clip_model_name,
            "hpsv2": args.hps_version,
            "imagereward": args.image_reward_model_name,
        }[target]

        for start in range(0, len(todo_idx), args.save_every):
            block_idx = todo_idx[start : start + args.save_every]
            paths = manifest.loc[block_idx, "image_path_resolved"].astype(str).tolist()
            prompts = manifest.loc[block_idx, prompt_col].astype(str).tolist()
            scores = scorer.score_batch(paths, prompts, batch_size=args.batch_size)
            manifest.loc[block_idx, score_col] = scores
            manifest.loc[block_idx, version_col] = version_value
            manifest.to_csv(out_path, index=False)
            LOG.info(
                "Saved %s progress: %d/%d newly scored rows",
                target,
                min(start + len(block_idx), len(todo_idx)),
                len(todo_idx),
            )

        del scorer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    manifest.to_csv(out_path, index=False)
    LOG.info("Wrote scored manifest: %s", out_path)
    save_chunk_trajectory_with_quality(chunk_dir, manifest)

    summary_path = chunk_dir / "quality_scoring_summary.json"
    for target, col in QUALITY_COLUMNS.items():
        if col in manifest.columns:
            summary[f"non_null_{col}"] = int(pd.to_numeric(manifest[col], errors="coerce").notna().sum())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote summary: %s", summary_path)


def parse_chunk_id(path: Path) -> int:
    m = re.search(r"chunk_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def merge_all(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pass_dirs = [p.strip() for p in args.pass_dirs.split(",") if p.strip()]
    traj_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for seed_pass, pass_dir_name in enumerate(pass_dirs):
        pass_dir = results_root / pass_dir_name
        for chunk_dir in sorted(pass_dir.glob("chunk_*"), key=parse_chunk_id):
            chunk_id = parse_chunk_id(chunk_dir)
            traj_path = chunk_dir / "trajectory_signals_with_quality.csv"
            manifest_path = chunk_dir / "generation_manifest_with_quality.csv"

            if not traj_path.exists():
                traj_path = chunk_dir / "trajectory_signals.csv"
                missing.append(str(chunk_dir / "trajectory_signals_with_quality.csv"))
            if not manifest_path.exists():
                manifest_path = chunk_dir / "generation_manifest.csv"
                missing.append(str(chunk_dir / "generation_manifest_with_quality.csv"))

            if traj_path.exists():
                df = adopt_alias_columns(pd.read_csv(traj_path))
                df.insert(0, "seed_pass", seed_pass)
                df.insert(1, "seed_pass_name", pass_dir_name)
                df.insert(2, "chunk_id", chunk_id)
                traj_frames.append(df)
                LOG.info("Loaded trajectory %s rows=%d", traj_path, len(df))
            else:
                missing.append(str(traj_path))

            if manifest_path.exists():
                mf = adopt_alias_columns(pd.read_csv(manifest_path))
                mf.insert(0, "seed_pass", seed_pass)
                mf.insert(1, "seed_pass_name", pass_dir_name)
                mf.insert(2, "chunk_id", chunk_id)
                manifest_frames.append(mf)
                LOG.info("Loaded manifest %s rows=%d", manifest_path, len(mf))
            else:
                missing.append(str(manifest_path))

    if not traj_frames:
        raise RuntimeError("No trajectory files found to merge.")

    traj_all = pd.concat(traj_frames, ignore_index=True, sort=False)
    manifest_all = pd.concat(manifest_frames, ignore_index=True, sort=False) if manifest_frames else pd.DataFrame()

    traj_out = output_dir / "trajectory_signals_with_quality_all5seeds.csv"
    manifest_out = output_dir / "generation_manifest_with_quality_all5seeds.csv"
    summary_out = output_dir / "merge_summary.json"

    traj_all.to_csv(traj_out, index=False)
    if not manifest_all.empty:
        manifest_all.to_csv(manifest_out, index=False)

    summary = {
        "results_root": str(results_root),
        "pass_dirs": pass_dirs,
        "trajectory_rows": int(len(traj_all)),
        "manifest_rows": int(len(manifest_all)) if not manifest_all.empty else 0,
        "trajectory_output": str(traj_out),
        "manifest_output": str(manifest_out) if not manifest_all.empty else None,
        "missing_files_or_unscored_chunk_outputs": missing,
        "quality_non_null_counts_trajectory": {
            col: int(pd.to_numeric(traj_all[col], errors="coerce").notna().sum())
            for col in QUALITY_COLUMNS.values()
            if col in traj_all.columns
        },
        "quality_non_null_counts_manifest": {
            col: int(pd.to_numeric(manifest_all[col], errors="coerce").notna().sum())
            for col in QUALITY_COLUMNS.values()
            if not manifest_all.empty and col in manifest_all.columns
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOG.info("Wrote merged trajectory CSV: %s", traj_out)
    if not manifest_all.empty:
        LOG.info("Wrote merged manifest CSV: %s", manifest_out)
    LOG.info("Wrote merge summary: %s", summary_out)
    if missing:
        LOG.warning("Some chunk quality outputs were missing. See merge_summary.json.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score-chunk")
    score.add_argument("--chunk-dir", required=True)
    score.add_argument("--targets", required=True, help="Comma-separated: clip,hpsv2,imagereward")
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--save-every", type=int, default=100)
    score.add_argument("--prompt-col", default=None)
    score.add_argument("--image-col", default=None)
    score.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    score.add_argument("--hps-version", default="v2.1")
    score.add_argument("--image-reward-model-name", default="ImageReward-v1.0")
    score.set_defaults(func=score_chunk)

    merge = sub.add_parser("merge-all")
    merge.add_argument("--results-root", default="results")
    merge.add_argument("--output-dir", default="results/sd35_7320_all5seeds_quality_merged")
    merge.add_argument("--pass-dirs", default=",".join(PASS_DIRS_DEFAULT))
    merge.set_defaults(func=merge_all)

    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
