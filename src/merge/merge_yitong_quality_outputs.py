#!/usr/bin/env python3
"""
Merge chunk-level generation manifests and trajectory signals after quality scoring.
This does NOT run any analysis. It only creates final combined CSVs.

Expected layout:
  RUN_DIR/chunk_0/generation_manifest_with_quality.csv
  RUN_DIR/chunk_0/trajectory_signals_with_quality.csv
  ...

If *_with_quality.csv is missing, it falls back to the unscored CSV and warns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_first_existing(chunk: Path, names: list[str]) -> tuple[pd.DataFrame, Path]:
    for name in names:
        p = chunk / name
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError(f"None of {names} found in {chunk}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.output_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted([p for p in run_dir.glob("chunk_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[-1]))
    if not chunks:
        raise FileNotFoundError(f"No chunk_* directories found in {run_dir}")

    manifest_parts = []
    traj_parts = []

    for chunk in chunks:
        chunk_id = int(chunk.name.split("_")[-1])

        m, mp = read_first_existing(chunk, ["generation_manifest_with_quality.csv", "generation_manifest.csv"])
        t, tp = read_first_existing(chunk, ["trajectory_signals_with_quality.csv", "trajectory_signals.csv"])

        m = m.copy()
        t = t.copy()
        m["chunk_id"] = chunk_id
        t["chunk_id"] = chunk_id

        if "with_quality" not in mp.name:
            print(f"WARNING: using unscored manifest for {chunk}: {mp}")
        if "with_quality" not in tp.name:
            print(f"WARNING: using unscored trajectory file for {chunk}: {tp}")

        manifest_parts.append(m)
        traj_parts.append(t)
        print(f"Loaded {chunk.name}: manifest={len(m)} from {mp.name}, trajectories={len(t)} from {tp.name}")

    manifest = pd.concat(manifest_parts, ignore_index=True, sort=False)
    traj = pd.concat(traj_parts, ignore_index=True, sort=False)

    # Make hpsv2_version/other version columns string-safe if present.
    for df in (manifest, traj):
        for col in df.columns:
            if col.endswith("_version") or col in {"hpsv2_version", "clip_model", "imagereward_model"}:
                df[col] = df[col].astype("string")

    manifest_out = out_dir / "generation_manifest_with_quality_all.csv"
    traj_out = out_dir / "trajectory_signals_with_quality_all.csv"

    manifest.to_csv(manifest_out, index=False)
    traj.to_csv(traj_out, index=False)

    print("\nDone.")
    print(f"Merged manifest:   {manifest_out} ({len(manifest)} rows, {manifest.shape[1]} cols)")
    print(f"Merged trajectories: {traj_out} ({len(traj)} rows, {traj.shape[1]} cols)")
    print("Quality columns in manifest:", [c for c in manifest.columns if c.endswith("_score") or c in ["clip_score", "hpsv2_score", "imagereward_score"]])
    print("Quality columns in trajectories:", [c for c in traj.columns if c.endswith("_score") or c in ["clip_score", "hpsv2_score", "imagereward_score"]])


if __name__ == "__main__":
    main()
