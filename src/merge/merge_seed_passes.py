from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_FILES = ["generation_manifest.csv", "trajectory_signals.csv", "failures.csv"]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple already-merged seed-pass directories into one dataset."
    )
    parser.add_argument(
        "--pass-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Merged pass directories, e.g. results/sd35_7320_1seed_merged results/sd35_7320_2ndseed_merged results/sd35_7320_3rdseed_merged",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for combined dataset.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="CSV filenames to merge from each pass directory.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.files:
        frames: list[pd.DataFrame] = []

        for i, pass_dir in enumerate(args.pass_dirs, start=1):
            df = read_csv_if_exists(pass_dir / name)
            if df is None or len(df) == 0:
                print(f"Skipping missing/empty {pass_dir / name}")
                continue

            df = df.copy()
            df["seed_pass"] = f"pass{i}"
            frames.append(df)

        if not frames:
            print(f"No data found for {name}; skipping")
            continue

        out = pd.concat(frames, ignore_index=True)

        sort_cols = [
            c for c in ["prompt_index", "prompt_id", "replicate_id", "seed", "timestep"]
            if c in out.columns
        ]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)

        output_path = args.out_dir / name
        out.to_csv(output_path, index=False)
        print(f"Wrote {output_path} with {len(out)} rows")


if __name__ == "__main__":
    main()
