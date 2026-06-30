from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def merge_csvs(root: Path, filename: str, out_path: Path) -> None:
    paths = sorted(root.glob(f"chunk_*/{filename}"))
    if not paths:
        print(f"No {filename} files found under {root}/chunk_*/")
        return

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_chunk_dir"] = p.parent.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["prompt_index", "replicate_id", "seed"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Root directory containing chunk_*/ subdirectories.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for merged CSV outputs.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merge_csvs(args.root, "generation_manifest.csv", args.out_dir / "generation_manifest.csv")
    merge_csvs(args.root, "trajectory_signals.csv", args.out_dir / "trajectory_signals.csv")
    merge_csvs(args.root, "failures.csv", args.out_dir / "failures.csv")


if __name__ == "__main__":
    main()
