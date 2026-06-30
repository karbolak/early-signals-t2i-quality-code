"""Prepare the prompt CSV used for thesis generation.

This script is intentionally configurable because Hugging Face dataset column
names can change. It loads a HF dataset, keeps the prompt column, optionally
filters rows by model/source column, removes duplicate prompts, and writes a
CSV compatible with the generation scripts.

Example:
    python src/data/prepare_prompt_dataset.py \
        --dataset-id data-is-better-together/open-image-preferences-v1-binarized \
        --split train \
        --prompt-col prompt \
        --model-col model \
        --model-filter "Stable Diffusion 3.5 Large" \
        --output-csv data/processed/final_generation_prompts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install datasets first: pip install datasets") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--model-filter", default=None)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_id, split=args.split)
    df = ds.to_pandas()

    if args.prompt_col not in df.columns:
        raise ValueError(f"Prompt column {args.prompt_col!r} not found. Available columns: {list(df.columns)}")

    if args.model_col and args.model_filter:
        if args.model_col not in df.columns:
            raise ValueError(f"Model column {args.model_col!r} not found. Available columns: {list(df.columns)}")
        mask = df[args.model_col].astype(str).str.contains(args.model_filter, case=False, na=False)
        df = df.loc[mask].copy()

    out = (
        df[[args.prompt_col]]
        .rename(columns={args.prompt_col: "prompt"})
        .dropna()
        .drop_duplicates("prompt")
        .reset_index(drop=True)
    )
    out.insert(0, "prompt_id", range(len(out)))
    if args.max_prompts is not None:
        out = out.head(args.max_prompts)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} prompts to {args.output_csv}")


if __name__ == "__main__":
    main()
