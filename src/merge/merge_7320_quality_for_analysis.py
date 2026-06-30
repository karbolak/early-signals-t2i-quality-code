#!/usr/bin/env python3
"""
Cluster-safe merge script for the 7320-prompt x 5-seed SD3.5 chunk outputs.

Run after all chunk scoring is complete. It merges:
  - generation manifests from all seed-pass/chunk directories
  - trajectory signal CSVs from all seed-pass/chunk directories
  - existing quality columns such as clip_score, hpsv2_score, image_reward_score

Main output:
  results/sd35_7320_5seeds_merged_quality/analysis_input_all_quality.csv

This output is intended for:
  src/anal_rich_traj_single_csv_v6.py --input-csv ... --quality-target existing --quality-col ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_RUN_DIRS = [
    "sd35_7320_1seed",
    "sd35_7320_2ndseed",
    "sd35_7320_3rdseed",
    "sd35_7320_4thseed",
    "sd35_7320_5thseed",
]

KNOWN_QUALITY_COLS = [
    "clip_score",
    "new_clip_score",
    "hpsv2_score",
    "hspv2_score",
    "imagereward_score",
    "image_reward_score",
    "quality_score",
    "clip_version",
    "hpsv2_version",
    "hspv2_version",
    "imagereward_version",
    "image_reward_version",
    "clip_model",
    "clip_pretrained",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def choose_first_existing(chunk_dir: Path, names: Iterable[str]) -> tuple[Path | None, pd.DataFrame | None]:
    for name in names:
        path = chunk_dir / name
        df = read_csv_if_exists(path)
        if df is not None:
            return path, df
    return None, None


def seed_pass_from_run_dir(run_dir: str) -> int | None:
    mapping = {
        "sd35_7320_1seed": 1,
        "sd35_7320_2ndseed": 2,
        "sd35_7320_3rdseed": 3,
        "sd35_7320_4thseed": 4,
        "sd35_7320_5thseed": 5,
    }
    return mapping.get(run_dir)


def normalise_quality_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # HPS typo fallback.
    if "hpsv2_score" not in out.columns and "hspv2_score" in out.columns:
        out["hpsv2_score"] = out["hspv2_score"]
    if "hpsv2_version" not in out.columns and "hspv2_version" in out.columns:
        out["hpsv2_version"] = out["hspv2_version"]

    # ImageReward spelling fallback.
    if "image_reward_score" not in out.columns and "imagereward_score" in out.columns:
        out["image_reward_score"] = out["imagereward_score"]
    if "imagereward_score" not in out.columns and "image_reward_score" in out.columns:
        out["imagereward_score"] = out["image_reward_score"]

    return out


def add_source_columns(df: pd.DataFrame, run_dir: str, chunk_index: int, source_file: Path) -> pd.DataFrame:
    out = normalise_quality_aliases(df)
    out = out.copy()
    seed_pass = seed_pass_from_run_dir(run_dir)

    out["seed_pass"] = seed_pass
    out["seed_pass_name"] = run_dir
    out["chunk_index"] = int(chunk_index)
    out["source_file"] = str(source_file)
    out["source_chunk_dir"] = str(source_file.parent)

    # Safe prompt identifier for prompt-controlled analysis.
    # This avoids accidentally mixing prompt_id values that restart inside chunks.
    if "prompt_id" in out.columns:
        out["prompt_id_original"] = out["prompt_id"].astype(str)
        out["prompt_group"] = "chunk_" + out["chunk_index"].astype(str) + "__" + out["prompt_id"].astype(str)
    elif "prompt" in out.columns:
        out["prompt_group"] = out["prompt"].astype(str)

    return out


def existing_quality_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in KNOWN_QUALITY_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in df.columns:
        low = c.lower()
        if (
            ("clip" in low or "hps" in low or "hsp" in low or "reward" in low or "quality" in low)
            and c not in cols
        ):
            cols.append(c)
    return cols


def choose_merge_keys(manifest: pd.DataFrame, traj: pd.DataFrame) -> list[str]:
    candidate_sets = [
        ["seed_pass", "chunk_index", "prompt_id", "seed"],
        ["seed_pass", "chunk_index", "prompt_id_original", "seed"],
        ["seed_pass_name", "chunk_index", "prompt_id", "seed"],
        ["prompt_group", "seed"],
        ["prompt_id", "seed"],
        ["image_path"],
    ]

    for keys in candidate_sets:
        if all(k in manifest.columns and k in traj.columns for k in keys):
            if not manifest.duplicated(keys).any():
                return keys

    raise ValueError(
        "Could not find safe merge keys between manifest and trajectory files.\n"
        f"Manifest columns include: {list(manifest.columns)[:80]}\n"
        f"Trajectory columns include: {list(traj.columns)[:80]}"
    )


def merge_manifest_into_trajectories(manifest_all: pd.DataFrame, traj_all: pd.DataFrame) -> pd.DataFrame:
    keys = choose_merge_keys(manifest_all, traj_all)
    quality_cols = existing_quality_cols(manifest_all)

    keep_cols = list(keys)
    useful_manifest_cols = [
        "prompt",
        "prompt_id",
        "prompt_id_original",
        "prompt_group",
        "seed",
        "seed_pass",
        "seed_pass_name",
        "chunk_index",
        "split",
        "image_path",
        "output_path",
        "filename",
        "image_file",
    ]
    for c in useful_manifest_cols + quality_cols:
        if c in manifest_all.columns and c not in keep_cols:
            keep_cols.append(c)

    manifest_small = manifest_all[keep_cols].drop_duplicates(keys)

    merged = traj_all.merge(
        manifest_small,
        on=keys,
        how="left",
        suffixes=("", "__from_manifest"),
        validate="many_to_one",
    )

    for c in useful_manifest_cols + quality_cols:
        from_c = f"{c}__from_manifest"
        if from_c in merged.columns:
            if c in merged.columns:
                merged[c] = merged[c].combine_first(merged[from_c])
                merged = merged.drop(columns=[from_c])
            else:
                merged = merged.rename(columns={from_c: c})

    merged = normalise_quality_aliases(merged)
    merged["merge_keys_used"] = ",".join(keys)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all 7320 x 5 seed chunks into one analysis-ready CSV.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sd35_7320_5seeds_merged_quality"))
    parser.add_argument("--require-quality", action="store_true", help="Fail if no quality columns are found in the final merge.")
    parser.add_argument("--expected-rows", type=int, default=36600, help="Expected final rows. Use 0 to disable this check.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_parts: list[pd.DataFrame] = []
    traj_parts: list[pd.DataFrame] = []
    summary: dict = {
        "results_root": str(args.results_root),
        "output_dir": str(args.output_dir),
        "run_dirs": args.run_dirs,
        "n_chunks": args.n_chunks,
        "chunks": [],
        "missing_chunks": [],
    }

    for run_dir in args.run_dirs:
        for chunk_index in range(args.n_chunks):
            chunk_dir = args.results_root / run_dir / f"chunk_{chunk_index}"
            chunk_info = {
                "run_dir": run_dir,
                "seed_pass": seed_pass_from_run_dir(run_dir),
                "chunk_index": int(chunk_index),
                "chunk_dir": str(chunk_dir),
            }

            if not chunk_dir.exists():
                chunk_info["status"] = "missing_chunk_dir"
                summary["missing_chunks"].append(chunk_info)
                summary["chunks"].append(chunk_info)
                continue

            manifest_path, manifest = choose_first_existing(
                chunk_dir,
                [
                    "generation_manifest_with_quality.csv",
                    "generation_manifest_with_clip.csv",
                    "generation_manifest.csv",
                ],
            )
            traj_path, traj = choose_first_existing(
                chunk_dir,
                [
                    "trajectory_signals_with_quality.csv",
                    "trajectory_signals_with_clip.csv",
                    "trajectory_signals.csv",
                ],
            )

            if manifest is None:
                chunk_info["manifest_status"] = "missing"
            else:
                manifest = add_source_columns(manifest, run_dir, chunk_index, manifest_path)  # type: ignore[arg-type]
                qcols = existing_quality_cols(manifest)
                chunk_info["manifest_file"] = str(manifest_path)
                chunk_info["manifest_rows"] = int(len(manifest))
                chunk_info["manifest_quality_cols"] = qcols
                manifest_parts.append(manifest)

            if traj is None:
                chunk_info["trajectory_status"] = "missing"
            else:
                traj = add_source_columns(traj, run_dir, chunk_index, traj_path)  # type: ignore[arg-type]
                chunk_info["trajectory_file"] = str(traj_path)
                chunk_info["trajectory_rows"] = int(len(traj))
                chunk_info["trajectory_quality_cols"] = existing_quality_cols(traj)
                traj_parts.append(traj)

            chunk_info["status"] = "ok"
            summary["chunks"].append(chunk_info)

    if not manifest_parts:
        raise RuntimeError("No manifest files found.")
    if not traj_parts:
        raise RuntimeError("No trajectory files found.")

    manifest_all = pd.concat(manifest_parts, ignore_index=True, sort=False).copy()
    traj_all = pd.concat(traj_parts, ignore_index=True, sort=False).copy()

    manifest_all.insert(0, "manifest_uid", [f"m{i:08d}" for i in range(len(manifest_all))])
    traj_all.insert(0, "trajectory_uid", [f"t{i:08d}" for i in range(len(traj_all))])

    merged = merge_manifest_into_trajectories(manifest_all, traj_all)

    quality_cols = existing_quality_cols(merged)
    rows_per_prompt = None
    if "prompt_group" in merged.columns:
        counts = merged.groupby("prompt_group").size()
        rows_per_prompt = {
            "n_prompt_groups": int(counts.shape[0]),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(counts.median()),
        }

    if args.expected_rows and len(merged) != args.expected_rows:
        print(f"WARNING: expected {args.expected_rows} merged rows, got {len(merged)}")

    if args.require_quality and not quality_cols:
        raise RuntimeError("No quality columns found in merged output.")

    manifest_out = args.output_dir / "generation_manifest_all_quality.csv"
    traj_out = args.output_dir / "trajectory_signals_all.csv"
    analysis_out = args.output_dir / "analysis_input_all_quality.csv"
    summary_out = args.output_dir / "merge_summary.json"

    manifest_all.to_csv(manifest_out, index=False)
    traj_all.to_csv(traj_out, index=False)
    merged.to_csv(analysis_out, index=False)

    summary.update(
        {
            "manifest_rows_total": int(len(manifest_all)),
            "trajectory_rows_total": int(len(traj_all)),
            "analysis_input_rows_total": int(len(merged)),
            "quality_cols_in_analysis_input": quality_cols,
            "rows_per_prompt_group": rows_per_prompt,
            "outputs": {
                "generation_manifest_all_quality": str(manifest_out),
                "trajectory_signals_all": str(traj_out),
                "analysis_input_all_quality": str(analysis_out),
                "summary": str(summary_out),
            },
        }
    )

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Merge complete.")
    print(f"Manifest rows:       {len(manifest_all)}")
    print(f"Trajectory rows:     {len(traj_all)}")
    print(f"Analysis input rows: {len(merged)}")
    print(f"Quality columns:     {quality_cols}")
    if rows_per_prompt:
        print(f"Prompt groups:       {rows_per_prompt}")
    print(f"Main analysis input: {analysis_out}")
    print(f"Summary:             {summary_out}")


if __name__ == "__main__":
    main()
