#!/usr/bin/env python3
"""
Normalize Yitong certain/uncertain prompt CSVs into the prompt format used by the
SD3.5 trajectory generation/scoring pipeline.

Input files have prompt text in `prompt_text`. Output contains at least:
  prompt_id, prompt, split
plus metadata columns preserved for later analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_one(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_text" not in df.columns:
        raise ValueError(f"{path} has no `prompt_text` column. Columns: {list(df.columns)}")

    df = df.copy()
    df["dataset_source"] = source_name
    df["prompt"] = df["prompt_text"].astype(str)

    if "prompt_id" in df.columns:
        df["original_prompt_id"] = df["prompt_id"].astype(str)
    else:
        df["original_prompt_id"] = [f"{source_name}_{i:05d}" for i in range(1, len(df) + 1)]

    if "category" not in df.columns:
        df["category"] = source_name
    df["split"] = df["category"].astype(str)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certain-csv", required=True, type=Path)
    ap.add_argument("--uncertain-csv", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--base-seed", type=int, default=12345)
    args = ap.parse_args()

    certain = load_one(args.certain_csv, "certain")
    uncertain = load_one(args.uncertain_csv, "uncertain")

    df = pd.concat([certain, uncertain], ignore_index=True, sort=False)
    df.insert(0, "row_id", range(len(df)))
    df["prompt_id"] = [f"yitong_{i:05d}" for i in range(1, len(df) + 1)]
    df["seed"] = args.base_seed + df["row_id"].astype(int) * 1000

    # Put the generation-critical columns first, keep all original metadata after.
    first = ["prompt_id", "prompt", "split", "seed", "row_id", "dataset_source", "original_prompt_id"]
    rest = [c for c in df.columns if c not in first]
    df = df[first + rest]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(df)} prompts to {args.output_csv}")
    print(df[["prompt_id", "split", "original_prompt_id", "prompt"]].head().to_string(index=False))
    print("Counts by split:")
    print(df["split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
