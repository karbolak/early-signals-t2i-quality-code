from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def resolve_image_path(path_value: str, image_root: Optional[Path], run_dir: Optional[Path]) -> Path:
    p = Path(str(path_value))
    candidates = []

    # 1. Exact path from manifest.
    candidates.append(p)

    # 2. Relative to image root.
    if image_root is not None:
        candidates.append(image_root / p)
        candidates.append(image_root / p.name)

    # 3. Relative to run dir.
    if run_dir is not None:
        candidates.append(run_dir / p)
        candidates.append(run_dir / "images" / p.name)
        candidates.append(run_dir / p.name)

    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            pass

    raise FileNotFoundError(
        f"Could not resolve image path {path_value!r}. Tried: "
        + "; ".join(str(c) for c in candidates[:8])
    )


def load_inputs(run_dir: Path, trajectory_csv: Optional[Path], manifest_csv: Optional[Path]) -> pd.DataFrame:
    manifest_csv = manifest_csv or (run_dir / "generation_manifest.csv")
    trajectory_csv = trajectory_csv or (run_dir / "trajectory_signals.csv")

    manifest = pd.read_csv(manifest_csv)
    traj = pd.read_csv(trajectory_csv)

    required_manifest = {"prompt_id", "seed", "prompt", "image_path"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    required_traj = {"prompt_id", "seed"}
    missing = required_traj - set(traj.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {sorted(missing)}")

    df = manifest.merge(traj, on=["prompt_id", "seed"], how="inner")
    if len(df) == 0:
        raise ValueError("Manifest and trajectory CSV merged to zero rows. Check prompt_id/seed types.")
    return df


def score_one(hpsv2, image_path: Path, prompt: str, hps_version: str) -> float:
    val = hpsv2.score([str(image_path)], prompt, hps_version=hps_version)
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[0])
    try:
        return float(val.detach().cpu().numpy().reshape(-1)[0])
    except Exception:
        return float(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute HPSv2 scores with progress/resume, then save a merged CSV that can be "
            "passed to anal_rich_traj_single_csv_v5.py with --quality-target existing."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--hps-version", default="v2.1")
    parser.add_argument("--limit", type=int, default=None, help="Score only first N rows for testing.")
    parser.add_argument("--save-every", type=int, default=5, help="Write partial CSV every N newly scored rows.")
    parser.add_argument("--resume", action="store_true", help="Resume from output CSV if it already exists.")
    args = parser.parse_args()

    import hpsv2

    if args.resume and args.output_csv.exists():
        df = pd.read_csv(args.output_csv)
        print(f"Resuming from existing output: {args.output_csv} ({len(df)} rows)", flush=True)
    else:
        df = load_inputs(args.run_dir, args.trajectory_csv, args.manifest_csv)
        if args.limit is not None:
            df = df.iloc[: args.limit].copy()
        df["hpsv2_score"] = np.nan

    if "hpsv2_score" not in df.columns:
        df["hpsv2_score"] = np.nan

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    remaining = df["hpsv2_score"].isna()
    print(f"Rows: {len(df)} | already scored: {(~remaining).sum()} | remaining: {remaining.sum()}", flush=True)
    print("If this pauses on the first item, HPSv2 is probably loading the model onto GPU/CPU.", flush=True)

    n_new = 0
    start_all = time.time()

    for idx in tqdm(df.index[remaining], desc="HPSv2 scoring", unit="img"):
        prompt = str(df.at[idx, "prompt"])
        img_path = resolve_image_path(str(df.at[idx, "image_path"]), args.image_root, args.run_dir)
        t0 = time.time()
        score = score_one(hpsv2, img_path, prompt, args.hps_version)
        dt = time.time() - t0
        df.at[idx, "image_path_resolved"] = str(img_path)
        df.at[idx, "hpsv2_score"] = score
        n_new += 1
        tqdm.write(f"{idx}: score={score:.6f} time={dt:.2f}s image={img_path.name}")

        if n_new % args.save_every == 0:
            df.to_csv(args.output_csv, index=False)
            tqdm.write(f"Saved partial output to {args.output_csv}")

    df.to_csv(args.output_csv, index=False)
    print(f"Done. Scored {n_new} new rows in {(time.time() - start_all) / 60:.2f} min")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
