from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


def merge_csvs(paths: list[Path], out_path: Path) -> int:
    frames = []
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            frames.append(pd.read_csv(p))
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    # Remove exact duplicate prompt/seed rows if a chunk was resumed/rerun.
    if {"prompt_id", "seed"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["prompt_id", "seed"], keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/sd35_trajectories_rich_250x5"))
    ap.add_argument("--chunks-dir", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    chunks_dir = args.chunks_dir or (root / "chunks")
    chunk_dirs = sorted([p for p in chunks_dir.glob("chunk_*") if p.is_dir()])
    if not chunk_dirs:
        raise SystemExit(f"No chunk directories found in {chunks_dir}")

    manifest_n = merge_csvs([d / "generation_manifest.csv" for d in chunk_dirs], root / "generation_manifest.csv")
    traj_n = merge_csvs([d / "trajectory_signals.csv" for d in chunk_dirs], root / "trajectory_signals.csv")
    fail_n = merge_csvs([d / "failures.csv" for d in chunk_dirs], root / "failures.csv")

    summary = {
        "root": str(root),
        "chunks": [str(d) for d in chunk_dirs],
        "manifest_rows": manifest_n,
        "trajectory_rows": traj_n,
        "failure_rows": fail_n,
    }
    (root / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
