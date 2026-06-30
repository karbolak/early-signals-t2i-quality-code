"""
Rich SD3.5 trajectory analysis.

FLAG GUIDE
----------
Input modes:
  --input-csv PATH
      Use one merged CSV containing trajectory JSON columns. This is the easiest mode
      when your internal signals are already stored in one file.
  --run-dir PATH
      Use the original two-file layout. The directory should contain:
        generation_manifest.csv  with prompt_id, seed, prompt, image_path
        trajectory_signals.csv   with prompt_id, seed, *_json signal columns

Quality / reward modes:
  --quality-target clip
      Use CLIP image-text similarity as the quality target. This matches the previous script.
  --quality-target imagereward
      Use ImageReward as the quality target. Requires the ImageReward package and model weights.
  --quality-target hpsv2
      Use HPSv2 as the quality target. Requires the hpsv2 package and model weights.
  --quality-target existing --quality-col COL
      Use an already computed quality/reward column from the CSV.
  --skip-clip
      Backwards-compatible alias for not computing CLIP/reward. Use this for signal-only analysis.
  --quality-col COL
      Use an existing quality column and map it internally to new_clip_score for labels/plots.
  --force-compute-clip / --force-compute-quality
      Recompute the chosen quality score even if a quality column already exists.

Label modes:
  --label-from-split
      Use split values such as top/bottom to create good_label when quality scores
      are unavailable.
  --relabel
      Rebuild labels even if good_label/failure_label already exist.

Feature modes:
  --feature-set rich
      Use all derived sequence features plus legacy summary columns. This is the
      most expressive but can overfit on small datasets.
  --feature-set compact
      Use a smaller, less redundant set of representative features. Prefer this for
      thesis reporting and robustness checks.
  --feature-set compact-prefix
      Use compact features computed only from the first --compact-prefix-steps
      denoising steps. This tests whether early abort is feasible.

Early-abort health score:
  --health-prefix-steps K
      Build the 0-100 training-free health score only from the first K steps. This
      is the correct option when simulating an abort decision made at step K.
  --learned-health-score
      Also create a calibrated learned health score from cross-validated logistic
      regression probabilities: 100 * P(good). This is usually the best score for
      threshold tuning, while the training-free score is more interpretable.
  --supervised-prefix-auc
      In addition to the hand-written prefix ROC curve, train a compact-prefix
      logistic model separately at every prefix length. This tests whether more
      internal signal data actually helps a learned model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:  # Allows feature-only analysis without PyTorch installed.
    torch = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # Loaded lazily only when CLIP scoring is requested.
    CLIPModel = None
    CLIPProcessor = None

# ImageReward and HPSv2 are optional. They are used only when requested by
# --quality-target imagereward or --quality-target hpsv2. Keeping imports lazy
# means the rest of the analysis can still run without these packages installed.
ImageReward = None
hpsv2 = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_json_list(x) -> list[float]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, str):
        try:
            return [float(v) for v in json.loads(x)]
        except Exception:
            return []
    return []


def slope(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    y = np.asarray(vals, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def mean_abs_diff(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(np.abs(np.diff(arr))))


def zscore_np(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def scale_0_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if np.isclose(mn, mx):
        return np.full_like(x, 50.0)
    return 100.0 * (x - mn) / (mx - mn)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class CLIPScorer:
    def __init__(self, model_name: str, device: str):
        if torch is None or CLIPModel is None or CLIPProcessor is None:
            raise ImportError(
                "CLIP scoring requires torch and transformers. Install them, provide an existing "
                "new_clip_score/--quality-col, or pass --skip-clip for feature-only analysis."
            )
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def score_batch(self, prompts: list[str], images: list[Image.Image]) -> np.ndarray:
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

        with torch.no_grad():
            outputs = self.model(**inputs)

        text_emb = outputs.text_embeds
        image_emb = outputs.image_embeds

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        sims = (text_emb * image_emb).sum(dim=-1)
        return sims.detach().cpu().numpy()


def resolve_image_path(raw_path, image_root: Optional[Path] = None, run_dir: Optional[Path] = None) -> Path:
    """
    Resolve image paths in a portable way.

    This is useful when generation_manifest.csv was created on a cluster and
    contains absolute paths such as:
        /scratch/.../sd35_trajectories_rich_250/images/p001__seed_12345.png

    The function first tries the path exactly as stored. If that does not exist,
    it tries common local fallbacks using only the filename:
        --image-root/<filename>
        --run-dir/images/<filename>

    It also supports normal relative paths from either --image-root or --run-dir.
    """
    path = Path(str(raw_path))

    candidates: list[Path] = []
    candidates.append(path)

    if image_root is not None:
        candidates.append(image_root / path)
        candidates.append(image_root / path.name)

    if run_dir is not None:
        candidates.append(run_dir / path)
        candidates.append(run_dir / "images" / path.name)
        candidates.append(run_dir / path.name)

    # Remove duplicates while preserving order.
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c

    tried = "\n  - ".join(str(c) for c in unique_candidates[:8])
    raise FileNotFoundError(
        "Could not locate image for manifest path:\n"
        f"  {raw_path}\n"
        "Tried:\n"
        f"  - {tried}\n"
        "Pass --image-root pointing to the local images directory, or place images under --run-dir/images."
    )


def rewrite_image_paths_for_local_files(
    df: pd.DataFrame,
    image_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Replace image_path values with locally resolvable paths before CLIP scoring."""
    if "image_path" not in df.columns:
        return df

    out = df.copy()
    resolved = []
    missing_examples = []

    for raw in out["image_path"].tolist():
        try:
            resolved.append(str(resolve_image_path(raw, image_root=image_root, run_dir=run_dir)))
        except FileNotFoundError as exc:
            missing_examples.append(str(exc))
            resolved.append(None)

    if missing_examples:
        shown = "\n\n".join(missing_examples[:3])
        raise FileNotFoundError(
            f"Could not resolve {len(missing_examples)} image paths. Examples:\n\n{shown}"
        )

    out["image_path"] = resolved
    return out


def compute_clip_scores(
    df: pd.DataFrame,
    batch_size: int,
    model_name: str,
    device: str,
    image_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> pd.DataFrame:
    scorer = CLIPScorer(model_name, device)
    df = rewrite_image_paths_for_local_files(df, image_root=image_root, run_dir=run_dir)
    scores = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        prompts = batch["prompt"].tolist()
        images = [load_image(Path(p)) for p in batch["image_path"].tolist()]
        scores.extend(scorer.score_batch(prompts, images).tolist())

    out = df.copy()
    out["new_clip_score"] = scores
    return out



class ImageRewardScorer:
    """Small wrapper around ImageReward-v1.0.

    Install separately, for example:
        pip install image-reward

    The ImageReward API has changed slightly across versions, so score_one tries
    both list-style and single-image calls and normalises the return value to float.
    """

    def __init__(self, model_name: str, device: str):
        global ImageReward
        if ImageReward is None:
            try:
                import ImageReward as RM  # type: ignore
                ImageReward = RM
            except ImportError as exc:
                raise ImportError(
                    "ImageReward scoring requires the ImageReward package. Install it, "
                    "use --quality-target clip, or provide --quality-col with precomputed rewards."
                ) from exc
        self.device = device
        try:
            self.model = ImageReward.load(model_name, device=device)
        except TypeError:
            self.model = ImageReward.load(model_name)

    def score_one(self, prompt: str, image_path: Path) -> float:
        # Prefer path-based scoring because ImageReward examples typically use paths.
        try:
            val = self.model.score(prompt, [str(image_path)])
        except Exception:
            val = self.model.score(prompt, str(image_path))

        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0])
        return float(val)

    def score_batch(self, prompts: list[str], image_paths: list[Path]) -> np.ndarray:
        scores = [self.score_one(prompt, path) for prompt, path in zip(prompts, image_paths)]
        return np.asarray(scores, dtype=float)


class HPSv2Scorer:
    """Small wrapper around HPSv2.

    Install separately, for example:
        pip install hpsv2

    HPSv2 is optional and can also be run outside this script, then passed back
    through --quality-target existing --quality-col your_hps_column.
    """

    def __init__(self, hps_version: str):
        global hpsv2
        if hpsv2 is None:
            try:
                import hpsv2 as hpsv2_module  # type: ignore
                hpsv2 = hpsv2_module
            except ImportError as exc:
                raise ImportError(
                    "HPSv2 scoring requires the hpsv2 package. Install it, "
                    "use --quality-target clip/imagereward, or provide --quality-col with precomputed rewards."
                ) from exc
        self.hps_version = hps_version

    def score_one(self, prompt: str, image_path: Path) -> float:
        val = hpsv2.score([str(image_path)], prompt, hps_version=self.hps_version)
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0])
        # Some versions return a tensor.
        try:
            return float(val.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return float(val)

    def score_batch(self, prompts: list[str], image_paths: list[Path]) -> np.ndarray:
        scores = [self.score_one(prompt, path) for prompt, path in zip(prompts, image_paths)]
        return np.asarray(scores, dtype=float)


def compute_image_reward_scores(
    df: pd.DataFrame,
    batch_size: int,
    model_name: str,
    device: str,
    image_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> pd.DataFrame:
    scorer = ImageRewardScorer(model_name, device)
    df = rewrite_image_paths_for_local_files(df, image_root=image_root, run_dir=run_dir)
    scores = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        prompts = batch["prompt"].tolist()
        image_paths = [Path(p) for p in batch["image_path"].tolist()]
        scores.extend(scorer.score_batch(prompts, image_paths).tolist())
    out = df.copy()
    out["image_reward_score"] = scores
    # Internally the analysis still uses new_clip_score as the generic quality column
    # for labels, AUC, plots, and abort simulation. The metric-specific column is kept too.
    out["new_clip_score"] = out["image_reward_score"]
    return out


def compute_hpsv2_scores(
    df: pd.DataFrame,
    batch_size: int,
    hps_version: str,
    image_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> pd.DataFrame:
    scorer = HPSv2Scorer(hps_version)
    df = rewrite_image_paths_for_local_files(df, image_root=image_root, run_dir=run_dir)
    scores = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        prompts = batch["prompt"].tolist()
        image_paths = [Path(p) for p in batch["image_path"].tolist()]
        scores.extend(scorer.score_batch(prompts, image_paths).tolist())
    out = df.copy()
    out["hpsv2_score"] = scores
    out["new_clip_score"] = out["hpsv2_score"]
    return out


def selected_quality_label(args: argparse.Namespace) -> str:
    if args.quality_target == "clip":
        return "CLIP"
    if args.quality_target == "imagereward":
        return "ImageReward"
    if args.quality_target == "hpsv2":
        return "HPSv2"
    if args.quality_col:
        return args.quality_col
    return "quality"

def merge_inputs(manifest_csv: Path, trajectory_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    traj = pd.read_csv(trajectory_csv)

    required_manifest = {"prompt_id", "seed", "prompt", "image_path"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    required_traj = {"prompt_id", "seed"}
    missing = required_traj - set(traj.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {missing}")

    return manifest.merge(traj, on=["prompt_id", "seed"], how="inner")


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Load either a single merged CSV or the original run directory layout."""
    if args.input_csv is not None:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        return pd.read_csv(args.input_csv)

    manifest_csv = args.run_dir / "generation_manifest.csv"
    trajectory_csv = args.run_dir / "trajectory_signals.csv"
    return merge_inputs(manifest_csv, trajectory_csv)


def coerce_existing_quality_column(df: pd.DataFrame, quality_col: Optional[str]) -> pd.DataFrame:
    """Standardise a user-provided quality column to new_clip_score."""
    out = df.copy()

    if quality_col:
        if quality_col not in out.columns:
            raise ValueError(f"--quality-col '{quality_col}' was not found in the input CSV.")
        if quality_col != "new_clip_score":
            out["new_clip_score"] = pd.to_numeric(out[quality_col], errors="coerce")
        return out

    # Common names from previous analysis outputs. Keep new_clip_score if already present.
    for candidate in ["new_clip_score", "clip_score", "clip", "quality_score"]:
        if candidate in out.columns:
            if candidate != "new_clip_score":
                out["new_clip_score"] = pd.to_numeric(out[candidate], errors="coerce")
            return out

    return out


def prepare_quality_and_labels(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Add/standardise the selected quality target, good_label, and failure_label.

    Internally, the selected target is mapped to new_clip_score so the rest of the
    older analysis code can be reused. If --quality-target imagereward is used,
    the raw ImageReward values are also saved as image_reward_score. If --quality-target
    hpsv2 is used, the raw HPSv2 values are saved as hpsv2_score.
    """
    out = coerce_existing_quality_column(df, args.quality_col)

    has_quality_inputs = {"prompt", "image_path"}.issubset(out.columns)
    force_quality = args.force_compute_quality or args.force_compute_clip
    should_compute = (
        force_quality
        or ("new_clip_score" not in out.columns and has_quality_inputs and not args.skip_clip and args.quality_target != "existing")
    )

    if should_compute:
        missing = {"prompt", "image_path"} - set(out.columns)
        if missing:
            raise ValueError(
                f"Cannot compute {args.quality_target} quality because the input is missing columns: {missing}. "
                "Pass --quality-col, include a precomputed quality column, or use --skip-clip for feature-only analysis."
            )
        if args.quality_target == "clip":
            print("Computing CLIP scores...")
            out = compute_clip_scores(
                out,
                args.batch_size,
                args.clip_model_name,
                args.device,
                image_root=args.image_root,
                run_dir=args.run_dir,
            )
        elif args.quality_target == "imagereward":
            print("Computing ImageReward scores...")
            out = compute_image_reward_scores(
                out,
                args.batch_size,
                args.image_reward_model_name,
                args.device,
                image_root=args.image_root,
                run_dir=args.run_dir,
            )
        elif args.quality_target == "hpsv2":
            print("Computing HPSv2 scores...")
            out = compute_hpsv2_scores(
                out,
                args.batch_size,
                args.hps_version,
                image_root=args.image_root,
                run_dir=args.run_dir,
            )
        elif args.quality_target == "existing":
            if "new_clip_score" not in out.columns:
                raise ValueError("--quality-target existing requires --quality-col or a new_clip_score column.")
        else:
            raise ValueError(f"Unknown --quality-target: {args.quality_target}")
    elif "new_clip_score" in out.columns:
        out["new_clip_score"] = pd.to_numeric(out["new_clip_score"], errors="coerce")
    else:
        print(
            "No selected quality column found and quality cannot be computed from this CSV. "
            "Quality-dependent plots, abort quality summaries, and percentile labels may be skipped."
        )

    if args.relabel:
        out = out.drop(columns=[c for c in ["good_label", "failure_label", "failure_cutoff"] if c in out.columns])

    if "good_label" in out.columns:
        out["good_label"] = pd.to_numeric(out["good_label"], errors="coerce").astype("Int64")
        if "failure_label" not in out.columns:
            out["failure_label"] = (1 - out["good_label"]).astype("Int64")
        return out

    if "failure_label" in out.columns:
        out["failure_label"] = pd.to_numeric(out["failure_label"], errors="coerce").astype("Int64")
        out["good_label"] = (1 - out["failure_label"]).astype("Int64")
        return out

    if "new_clip_score" in out.columns and out["new_clip_score"].notna().any():
        print(f"Deriving labels from selected quality target: {selected_quality_label(args)}")
        return assign_labels(out, args.failure_percentile)

    if args.label_from_split and "split" in out.columns:
        good_values = {v.strip().lower() for v in args.good_split_values.split(",") if v.strip()}
        bad_values = {v.strip().lower() for v in args.bad_split_values.split(",") if v.strip()}
        split_norm = out["split"].astype(str).str.lower().str.strip()
        good = split_norm.isin(good_values)
        bad = split_norm.isin(bad_values)
        if (good | bad).any():
            out.loc[good, "good_label"] = 1
            out.loc[bad, "good_label"] = 0
            out["good_label"] = out["good_label"].astype("Int64")
            out["failure_label"] = (1 - out["good_label"]).astype("Int64")
            print("Derived labels from split column.")
        else:
            print("split column exists, but no values matched --good-split-values or --bad-split-values.")

    return out

def has_labels(df: pd.DataFrame) -> bool:
    return "good_label" in df.columns and df["good_label"].notna().any()


def has_two_label_classes(df: pd.DataFrame) -> bool:
    return has_labels(df) and df["good_label"].dropna().nunique() >= 2


def has_quality(df: pd.DataFrame) -> bool:
    return "new_clip_score" in df.columns and df["new_clip_score"].notna().any()


def assign_labels(df: pd.DataFrame, failure_percentile: float) -> pd.DataFrame:
    out = df.copy()
    cutoff = out["new_clip_score"].quantile(failure_percentile)
    out["failure_label"] = (out["new_clip_score"] <= cutoff).astype(int)
    out["good_label"] = 1 - out["failure_label"]
    out["failure_cutoff"] = cutoff
    return out


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive sequence summary features from *_json columns.

    This version collects all new columns in a dictionary and concatenates once,
    avoiding pandas DataFrame fragmentation warnings from repeated column inserts.
    """
    out = df.copy()

    json_cols = [
        "latent_rms_json",
        "latent_std_json",
        "latent_abs_mean_json",
        "latent_volatility_rms_json",
        "latent_update_cosine_json",
        "denoiser_pred_rms_json",
        "denoiser_pred_std_json",
        "denoiser_pred_abs_mean_json",
        "denoiser_pred_delta_rms_json",
        "denoiser_pred_cosine_prev_json",
        "cfg_divergence_rms_json",
        "cfg_divergence_relative_json",
        "cfg_alignment_cosine_json",
        "denoising_consistency_residual_json",
        "denoising_update_pred_cosine_json",
    ]

    new_cols: dict[str, pd.Series] = {}

    for col in json_cols:
        if col not in out.columns:
            continue

        base = col.replace("_json", "")
        lists = out[col].apply(parse_json_list)

        seq_mean = lists.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
        seq_std = lists.apply(lambda v: float(np.std(v)) if len(v) else np.nan)
        seq_min = lists.apply(lambda v: float(np.min(v)) if len(v) else np.nan)
        seq_max = lists.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
        seq_first = lists.apply(lambda v: float(v[0]) if len(v) else np.nan)
        seq_last = lists.apply(lambda v: float(v[-1]) if len(v) else np.nan)

        candidates = {
            f"{base}_seq_mean": seq_mean,
            f"{base}_seq_std": seq_std,
            f"{base}_seq_min": seq_min,
            f"{base}_seq_max": seq_max,
            f"{base}_seq_range": seq_max - seq_min,
            f"{base}_seq_first": seq_first,
            f"{base}_seq_last": seq_last,
            f"{base}_seq_delta": seq_last - seq_first,
            f"{base}_seq_slope": lists.apply(slope),
            f"{base}_seq_mean_abs_diff": lists.apply(mean_abs_diff),
        }
        for name, values in candidates.items():
            if name not in out.columns:
                new_cols[name] = values

    if not new_cols:
        return out

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1).copy()


def add_compact_prefix_features(df: pd.DataFrame, prefix_steps: int) -> pd.DataFrame:
    """Add compact features computed only from the first K denoising steps.

    These features are intended for early-abort experiments, where the decision
    must be made before the full trajectory is available.
    """
    out = df.copy()
    prefix_steps = max(1, int(prefix_steps))

    json_cols = [c for c in out.columns if c.endswith("_json")]
    new_cols: dict[str, pd.Series] = {}

    for col in json_cols:
        base = col.replace("_json", "")
        # Skip non-signal JSON columns such as timesteps_json and cfg_chunk_count_json.
        if base in {"timesteps", "cfg_chunk_count", "step_times_sec"}:
            continue

        lists = out[col].apply(parse_json_list).apply(lambda v: v[:prefix_steps])
        prefix = f"{base}_prefix{prefix_steps}"

        candidates = {
            f"{prefix}_mean": lists.apply(lambda v: float(np.mean(v)) if len(v) else np.nan),
            f"{prefix}_slope": lists.apply(slope),
            f"{prefix}_last": lists.apply(lambda v: float(v[-1]) if len(v) else np.nan),
        }
        for name, values in candidates.items():
            if name not in out.columns:
                new_cols[name] = values

    if not new_cols:
        return out

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1).copy()

def signal_prefixes() -> list[str]:
    return [
        "latent_rms",
        "latent_std",
        "latent_abs_mean",
        "latent_volatility_rms",
        "latent_update_cosine",
        "denoiser_pred_rms",
        "denoiser_pred_std",
        "denoiser_pred_abs_mean",
        "denoiser_pred_delta_rms",
        "denoiser_pred_cosine_prev",
        "cfg_divergence_rms",
        "cfg_divergence_relative",
        "cfg_alignment_cosine",
        "denoising_consistency_residual",
        "denoising_update_pred_cosine",
    ]


def trajectory_feature_columns(
    df: pd.DataFrame,
    feature_set: str = "rich",
    compact_prefix_steps: int = 5,
    include_runtime_feature: bool = False,
) -> list[str]:
    prefixes = signal_prefixes()

    rich_suffixes = [
        "_seq_mean",
        "_seq_std",
        "_seq_min",
        "_seq_max",
        "_seq_range",
        "_seq_first",
        "_seq_last",
        "_seq_delta",
        "_seq_slope",
        "_seq_mean_abs_diff",
    ]

    compact_suffixes = [
        "_seq_mean",
        "_seq_slope",
        "_seq_mean_abs_diff",
    ]

    compact_prefix_suffixes = [
        f"_prefix{compact_prefix_steps}_mean",
        f"_prefix{compact_prefix_steps}_slope",
        f"_prefix{compact_prefix_steps}_last",
    ]

    if feature_set == "compact":
        suffixes = compact_suffixes
    elif feature_set == "compact-prefix":
        suffixes = compact_prefix_suffixes
    else:
        suffixes = rich_suffixes

    cols = []
    for p in prefixes:
        for s in suffixes:
            c = f"{p}{s}"
            if c in df.columns:
                cols.append(c)

    if feature_set == "rich":
        legacy = [
            "latent_rms_mean",
            "latent_rms_max",
            "latent_rms_std",
            "latent_rms_slope",
            "latent_volatility_mean",
            "latent_volatility_max",
            "latent_volatility_std",
            "latent_volatility_slope",
            "cfg_divergence_mean",
            "cfg_divergence_max",
            "cfg_divergence_std",
            "cfg_divergence_slope",
            "denoiser_pred_rms_mean",
            "denoiser_pred_delta_rms_mean",
            "denoising_consistency_residual_mean",
            "denoising_update_pred_cosine_mean",
        ]
        for c in legacy:
            if c in df.columns and c not in cols:
                cols.append(c)

    # Total runtime is not available at an early-abort decision point.
    # Keep it out by default; only include it for explicit diagnostic runs.
    if include_runtime_feature and "total_runtime_sec" in df.columns and "total_runtime_sec" not in cols:
        cols.append("total_runtime_sec")

    return cols

def build_training_free_health(df: pd.DataFrame, prefix_steps: Optional[int] = None) -> pd.DataFrame:
    """Build a simple 0-100 hand-designed trajectory health score.

    If prefix_steps is provided, the score uses only columns computed from the
    first K steps. That is the appropriate setting for early-abort simulations.
    """
    out = df.copy()

    def z(col: str) -> np.ndarray:
        return zscore_np(out[col].to_numpy(dtype=float))

    if prefix_steps is None:
        name = lambda base, stat: f"{base}_seq_{stat}"
    else:
        name = lambda base, stat: f"{base}_prefix{prefix_steps}_{stat}"

    terms = []

    # Positive: productive trajectory movement / denoiser activity.
    # These signs are assumptions, not learned weights; validate them against CLIP.
    for col, weight in [
        (name("latent_volatility_rms", "mean"), 1.00),
        (name("latent_volatility_rms", "slope"), 0.50),
        (name("denoiser_pred_delta_rms", "mean"), 0.75),
        (name("denoising_update_pred_cosine", "mean"), 0.50),
    ]:
        if col in out.columns:
            terms.append(weight * z(col))

    # Negative: excessive latent magnitude / inconsistency residual.
    for col, weight in [
        (name("latent_rms", "mean"), -0.25),
        (name("latent_rms", "slope"), -0.15),
        (name("denoising_consistency_residual", "mean"), -0.75),
    ]:
        if col in out.columns:
            terms.append(weight * z(col))

    # CFG divergence can be ambiguous, so keep it mild.
    cfg_col = name("cfg_divergence_rms", "mean")
    if cfg_col in out.columns:
        terms.append(0.25 * z(cfg_col))

    if not terms:
        raise ValueError("No usable features for training-free health score.")

    raw = np.sum(np.column_stack(terms), axis=1)

    out["health_score_raw"] = raw
    out["health_score_0_100"] = scale_0_100(raw)
    out["health_score_mode"] = "prefix" if prefix_steps is not None else "full"
    out["health_score_prefix_steps"] = prefix_steps if prefix_steps is not None else np.nan
    return out

def run_logreg_cv(df: pd.DataFrame, feature_cols: list[str], n_splits: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    feature_cols = list(dict.fromkeys(feature_cols))
    df = df.loc[:, ~df.columns.duplicated()].copy()
    clean = df.dropna(subset=feature_cols + ["good_label"]).copy()

    if not feature_cols:
        raise ValueError("No trajectory feature columns found for logistic regression.")

    X = clean[feature_cols].to_numpy(dtype=float)
    y = clean["good_label"].to_numpy(dtype=int)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Logistic regression requires at least two label classes in good_label.")

    effective_splits = min(n_splits, int(counts.min()))
    if effective_splits < 2:
        raise ValueError("Logistic regression requires at least two examples in each class.")
    if effective_splits != n_splits:
        print(f"Reducing n_splits from {n_splits} to {effective_splits} because of class counts.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    metrics = {
        "n": int(len(clean)),
        "n_features": int(len(feature_cols)),
        "n_splits": int(effective_splits),
        **safe_auc(y, probs),
    }

    model.fit(X, y)
    coef = model.named_steps["clf"].coef_[0]

    coef_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "coefficient_standardized": coef,
        })
        .sort_values("coefficient_standardized", ascending=False)
        .reset_index(drop=True)
    )

    pred_df = clean[["prompt_id", "seed", "good_label"]].copy() if {"prompt_id", "seed", "good_label"}.issubset(clean.columns) else clean[["good_label"]].copy()
    pred_df["cv_pred_good_prob"] = probs
    pred_df["learned_health_score_0_100"] = 100.0 * probs
    return metrics, coef_df, pred_df


def prefix_health_auc(df: pd.DataFrame, max_prefix: Optional[int] = None) -> pd.DataFrame:
    required = {"latent_rms_json", "latent_volatility_rms_json", "good_label"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping prefix analysis; missing columns: {missing}")
        return pd.DataFrame(columns=["prefix_step", "n", "roc_auc", "pr_auc"])

    rms_lists = df["latent_rms_json"].apply(parse_json_list)
    vol_lists = df["latent_volatility_rms_json"].apply(parse_json_list)

    available = int(rms_lists.apply(len).max())
    if max_prefix is not None:
        available = min(available, max_prefix)

    rows = []

    for k in range(1, available + 1):
        tmp = df.copy()

        rms_prefix = rms_lists.apply(lambda v: v[:k])
        vol_prefix = vol_lists.apply(lambda v: v[:max(0, k - 1)])

        tmp["prefix_rms_mean"] = rms_prefix.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
        tmp["prefix_rms_slope"] = rms_prefix.apply(slope)
        tmp["prefix_vol_mean"] = vol_prefix.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
        tmp["prefix_vol_max"] = vol_prefix.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
        tmp["prefix_vol_slope"] = vol_prefix.apply(slope)

        if k == 1:
            raw = -zscore_np(tmp["prefix_rms_mean"].to_numpy(dtype=float))
        else:
            raw = (
                1.0 * zscore_np(tmp["prefix_vol_mean"].to_numpy(dtype=float))
                + 0.5 * zscore_np(tmp["prefix_vol_max"].to_numpy(dtype=float))
                + 0.5 * zscore_np(tmp["prefix_vol_slope"].to_numpy(dtype=float))
                - 0.25 * zscore_np(tmp["prefix_rms_mean"].to_numpy(dtype=float))
                - 0.15 * zscore_np(tmp["prefix_rms_slope"].to_numpy(dtype=float))
            )

        tmp["prefix_health"] = scale_0_100(raw)
        clean = tmp.dropna(subset=["prefix_health", "good_label"])

        metrics = safe_auc(
            clean["good_label"].to_numpy(dtype=int),
            clean["prefix_health"].to_numpy(dtype=float),
        )

        rows.append({
            "prefix_step": k,
            "n": int(len(clean)),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })

    return pd.DataFrame(rows)


def parse_float_list_csv(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def simulate_abort(
    df: pd.DataFrame,
    total_steps: int,
    early_steps: int,
    score_col: str = "health_score_0_100",
    lambdas: Optional[list[float]] = None,
) -> pd.DataFrame:
    rows = []
    quality_available = has_quality(df)
    if score_col not in df.columns:
        raise ValueError(f"Missing score column for abort simulation: {score_col}")
    if lambdas is None:
        lambdas = []

    for tau in np.linspace(0, 100, 41):
        tmp = df.copy()
        tmp["aborted"] = (tmp[score_col] < tau).astype(int)
        tmp["used_steps"] = np.where(tmp["aborted"] == 1, early_steps, total_steps)

        kept = tmp[tmp["aborted"] == 0]
        abort_rate = float(tmp["aborted"].mean())
        compute_saved = float(1.0 - tmp["used_steps"].mean() / total_steps)

        row = {
            "score_col": score_col,
            "threshold": float(tau),
            "abort_rate": abort_rate,
            "mean_used_steps": float(tmp["used_steps"].mean()),
            "compute_saved_fraction": compute_saved,
            "kept_count": int(len(kept)),
            "total_count": int(len(tmp)),
        }
        if quality_available:
            mean_all = float(tmp["new_clip_score"].mean())
            mean_kept = float(kept["new_clip_score"].mean()) if len(kept) else np.nan
            row["mean_quality_all"] = mean_all
            row["mean_quality_kept"] = mean_kept
            row["quality_lift_vs_all"] = mean_kept - mean_all if len(kept) else np.nan
            row["quality_lift_pct_vs_all"] = 100.0 * (mean_kept / mean_all - 1.0) if len(kept) and mean_all != 0 else np.nan
            # Random aborting at the same abort rate has expected retained quality equal to the dataset mean.
            row["random_abort_expected_mean_quality_kept"] = mean_all
            row["quality_lift_vs_random_abort"] = mean_kept - mean_all if len(kept) else np.nan
            for lam in lambdas:
                row[f"utility_lambda_{lam}"] = mean_kept + lam * compute_saved if len(kept) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def supervised_prefix_auc(df: pd.DataFrame, max_prefix: int, n_splits: int) -> pd.DataFrame:
    if not has_two_label_classes(df):
        return pd.DataFrame(columns=["prefix_step", "n", "n_features", "n_splits", "roc_auc", "pr_auc"])

    available = 0
    for col in [c for c in df.columns if c.endswith("_json")]:
        lengths = df[col].apply(parse_json_list).apply(len)
        if len(lengths):
            available = max(available, int(lengths.max()))
    available = min(available, max_prefix)

    rows = []
    for k in range(1, available + 1):
        tmp = add_compact_prefix_features(df, k)
        cols = trajectory_feature_columns(tmp, feature_set="compact-prefix", compact_prefix_steps=k)
        try:
            metrics, _, _ = run_logreg_cv(tmp, cols, n_splits)
            rows.append({"prefix_step": k, **metrics})
        except ValueError as exc:
            rows.append({"prefix_step": k, "n": len(tmp), "n_features": len(cols), "n_splits": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)


def save_hist(s: pd.Series, title: str, xlabel: str, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(s.dropna(), bins=25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, out: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, cols: list[str], out: Path) -> None:
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(im, label="Pearson correlation")
    plt.title("Rich trajectory correlation heatmap")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def save_prefix_plot(prefix_df: pd.DataFrame, out: Path) -> None:
    if prefix_df.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(prefix_df["prefix_step"], prefix_df["roc_auc"], marker="o", label="ROC-AUC")
    plt.plot(prefix_df["prefix_step"], prefix_df["pr_auc"], marker="o", label="PR-AUC")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="random ROC")
    plt.ylim(0, 1)
    plt.xlabel("Available denoising step prefix")
    plt.ylabel("Score")
    plt.title("Predictive performance by timestep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def save_coef_plot(coef_df: pd.DataFrame, out: Path) -> None:
    plot_df = coef_df.sort_values("coefficient_standardized")

    plt.figure(figsize=(9, max(5, 0.28 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["coefficient_standardized"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title("Trajectory-only logistic regression coefficients")
    plt.xlabel("Standardized coefficient toward good generation")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse SD3.5 internal trajectory signals for quality prediction and early abort.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Single merged CSV containing trajectory JSON columns. If omitted, --run-dir is used.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/sd35_trajectories_rich_250"),
        help="Directory containing generation_manifest.csv and trajectory_signals.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/rich_analysis"))
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--quality-col", default=None, help="Existing quality/reward column to use as the selected quality target.")
    parser.add_argument(
        "--quality-target",
        choices=["clip", "imagereward", "hpsv2", "existing"],
        default="clip",
        help=(
            "Which final-image quality target to compute/use for labels and abort summaries. "
            "clip = CLIP similarity; imagereward = ImageReward-v1.0; hpsv2 = HPSv2; "
            "existing = use --quality-col or an existing new_clip_score column."
        ),
    )
    parser.add_argument("--image-reward-model-name", default="ImageReward-v1.0", help="ImageReward model name for --quality-target imagereward.")
    parser.add_argument("--hps-version", default="v2.1", help="HPSv2 version for --quality-target hpsv2.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Local directory containing generated images. Useful when generation_manifest.csv "
            "contains absolute cluster paths. If omitted, the script also tries --run-dir/images."
        ),
    )
    parser.add_argument("--skip-clip", action="store_true", help="Do not compute CLIP scores even if prompt/image_path are present.")
    parser.add_argument("--force-compute-clip", action="store_true", help="Backwards-compatible alias for --force-compute-quality when --quality-target clip is used.")
    parser.add_argument("--force-compute-quality", action="store_true", help="Recompute the selected quality/reward target even if a quality column exists.")
    parser.add_argument("--failure-percentile", type=float, default=0.5)
    parser.add_argument("--relabel", action="store_true", help="Rebuild labels even if good_label/failure_label already exist.")
    parser.add_argument("--label-from-split", action="store_true", help="Use split values as labels when no quality score is available.")
    parser.add_argument("--good-split-values", default="top,good,success")
    parser.add_argument("--bad-split-values", default="bottom,bad,failure")
    parser.add_argument(
        "--feature-set",
        choices=["rich", "compact", "compact-prefix"],
        default="rich",
        help=(
            "Feature set for supervised logistic regression: rich = all sequence and legacy features; "
            "compact = reduced non-legacy sequence features; compact-prefix = reduced features from only the first K steps."
        ),
    )
    parser.add_argument(
        "--include-runtime-feature",
        action="store_true",
        help=(
            "Include total_runtime_sec in supervised feature sets. Off by default because total runtime "
            "is unavailable at an early-abort decision point and would leak future information."
        ),
    )
    parser.add_argument(
        "--compact-prefix-steps",
        type=int,
        default=5,
        help="Number of early denoising steps used by --feature-set compact-prefix.",
    )
    parser.add_argument(
        "--health-prefix-steps",
        type=int,
        default=None,
        help="If set, compute health_score_0_100 using only the first K steps for realistic early-abort simulation.",
    )
    parser.add_argument(
        "--learned-health-score",
        action="store_true",
        help=(
            "After supervised CV, save learned_health_score_0_100 = 100 * cross-validated P(good). "
            "Use this for threshold tuning when labels are available."
        ),
    )
    parser.add_argument(
        "--supervised-prefix-auc",
        action="store_true",
        help=(
            "Train/evaluate a compact-prefix logistic model at every prefix length. "
            "This is a learned comparison to the hand-written prefix health curve."
        ),
    )
    parser.add_argument(
        "--utility-lambdas",
        default="0,0.02,0.05,0.10",
        help="Comma-separated lambda values for utility = mean_quality_kept + lambda * compute_saved_fraction.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=25)
    parser.add_argument("--early-steps-for-abort", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    plots_dir = args.output_dir / "plots"
    ensure_dir(plots_dir)

    df = load_input_dataframe(args)
    if df.columns.duplicated().any():
        print("Input contains duplicate column names; keeping the first occurrence of each.")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    df = prepare_quality_and_labels(df, args)

    print("Deriving rich sequence features...")
    df = add_sequence_features(df)

    if args.feature_set == "compact-prefix" or args.health_prefix_steps is not None:
        prefix_k = args.health_prefix_steps if args.health_prefix_steps is not None else args.compact_prefix_steps
        print(f"Deriving compact prefix features from the first {prefix_k} steps...")
        df = add_compact_prefix_features(df, prefix_k)

    print("Building training-free rich health score...")
    df = build_training_free_health(df, prefix_steps=args.health_prefix_steps)

    feature_cols = trajectory_feature_columns(
        df,
        feature_set=args.feature_set,
        compact_prefix_steps=args.compact_prefix_steps,
        include_runtime_feature=args.include_runtime_feature,
    )
    print(f"Trajectory feature set: {args.feature_set}")
    print(f"Trajectory feature count: {len(feature_cols)}")
    if not feature_cols:
        raise ValueError("No trajectory feature columns found after deriving sequence features.")

    logreg_metrics = None
    coef_df = pd.DataFrame(columns=["feature", "coefficient_standardized"])
    pred_df = pd.DataFrame()
    model_rows = []
    lambdas = parse_float_list_csv(args.utility_lambdas)

    if has_two_label_classes(df):
        health_clean = df.dropna(subset=["good_label", "health_score_0_100"])
        health_metrics = safe_auc(
            health_clean["good_label"].to_numpy(dtype=int),
            health_clean["health_score_0_100"].to_numpy(dtype=float),
        )
        model_rows.append({
            "model": "training_free_rich_health",
            "n_features": 1,
            **health_metrics,
        })

        print("Running trajectory-only logistic regression baseline...")
        try:
            logreg_metrics, coef_df, pred_df = run_logreg_cv(df, feature_cols, args.n_splits)
            if args.learned_health_score:
                df.loc[pred_df.index, "learned_health_score_0_100"] = pred_df["learned_health_score_0_100"]
            model_rows.append({
                "model": "trajectory_logreg_supervised_baseline",
                "n_features": len(feature_cols),
                **logreg_metrics,
            })
        except ValueError as exc:
            print(f"Skipping logistic regression: {exc}")
    elif has_labels(df):
        print("Only one label class is available; skipping AUC metrics and supervised logistic regression.")
    else:
        print("No labels available; skipping AUC metrics and supervised logistic regression.")

    model_comparison = pd.DataFrame(model_rows)

    if has_two_label_classes(df):
        print("Running hand-written ROC-by-timestep analysis...")
        prefix_df = prefix_health_auc(df, max_prefix=args.total_steps)
        if args.supervised_prefix_auc:
            print("Running supervised compact-prefix ROC-by-timestep analysis...")
            supervised_prefix_df = supervised_prefix_auc(df, max_prefix=args.total_steps, n_splits=args.n_splits)
        else:
            supervised_prefix_df = pd.DataFrame(columns=["prefix_step", "n", "n_features", "n_splits", "roc_auc", "pr_auc"])
    else:
        prefix_df = pd.DataFrame(columns=["prefix_step", "n", "roc_auc", "pr_auc"])
        supervised_prefix_df = pd.DataFrame(columns=["prefix_step", "n", "n_features", "n_splits", "roc_auc", "pr_auc"])

    print("Running abort simulation for training-free health score...")
    abort_df = simulate_abort(
        df,
        total_steps=args.total_steps,
        early_steps=args.early_steps_for_abort,
        score_col="health_score_0_100",
        lambdas=lambdas,
    )
    if args.learned_health_score and "learned_health_score_0_100" in df.columns:
        print("Running abort simulation for learned health score...")
        learned_abort_df = simulate_abort(
            df,
            total_steps=args.total_steps,
            early_steps=args.early_steps_for_abort,
            score_col="learned_health_score_0_100",
            lambdas=lambdas,
        )
    else:
        learned_abort_df = pd.DataFrame()

    df.to_csv(args.output_dir / "rich_merged_dataset.csv", index=False)
    coef_df.to_csv(args.output_dir / "trajectory_logreg_coefficients.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(args.output_dir / "trajectory_logreg_cv_predictions.csv", index=False)
    model_comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    prefix_df.to_csv(args.output_dir / "prefix_step_roc_auc.csv", index=False)
    supervised_prefix_df.to_csv(args.output_dir / "supervised_prefix_step_roc_auc.csv", index=False)
    abort_df.to_csv(args.output_dir / "abort_simulation.csv", index=False)
    if not learned_abort_df.empty:
        learned_abort_df.to_csv(args.output_dir / "learned_abort_simulation.csv", index=False)

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(args.input_csv) if args.input_csv else None,
                "run_dir": str(args.run_dir) if args.input_csv is None else None,
                "n_rows": int(len(df)),
                "quality_target": args.quality_target,
                "quality_label": selected_quality_label(args),
                "trajectory_feature_set": args.feature_set,
                "compact_prefix_steps": int(args.compact_prefix_steps),
                "health_prefix_steps": args.health_prefix_steps,
                "trajectory_feature_count": int(len(feature_cols)),
                "feature_cols": feature_cols,
                "has_quality": bool(has_quality(df)),
                "has_labels": bool(has_labels(df)),
                "has_two_label_classes": bool(has_two_label_classes(df)),
                "logreg_metrics": logreg_metrics,
                "learned_health_score_available": bool("learned_health_score_0_100" in df.columns),
                "supervised_prefix_auc_available": bool(not supervised_prefix_df.empty),
                "model_comparison": model_comparison.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    if has_quality(df):
        q_label = selected_quality_label(args)
        save_hist(df["new_clip_score"], f"{q_label} quality distribution", "selected_quality_score", plots_dir / "target_quality_hist.png")
        save_scatter(df, "health_score_0_100", "new_clip_score", f"Training-free health vs {q_label}", plots_dir / "health_vs_quality.png")
        # Backwards-compatible filenames for existing thesis notebooks/scripts.
        save_hist(df["new_clip_score"], f"{q_label} quality distribution", "selected_quality_score", plots_dir / "new_clip_hist.png")
        save_scatter(df, "health_score_0_100", "new_clip_score", f"Training-free health vs {q_label}", plots_dir / "health_vs_clip.png")
        if "learned_health_score_0_100" in df.columns:
            save_scatter(df, "learned_health_score_0_100", "new_clip_score", f"Learned health vs {q_label}", plots_dir / "learned_health_vs_quality.png")
            save_scatter(df, "learned_health_score_0_100", "new_clip_score", f"Learned health vs {q_label}", plots_dir / "learned_health_vs_clip.png")

    save_prefix_plot(prefix_df, plots_dir / "prefix_step_roc_auc.png")
    if not supervised_prefix_df.empty:
        save_prefix_plot(supervised_prefix_df, plots_dir / "supervised_prefix_step_roc_auc.png")
    if not coef_df.empty:
        save_coef_plot(coef_df, plots_dir / "trajectory_logreg_coefficients.png")

    corr_cols = [
        c for c in [
            "new_clip_score",
            "health_score_0_100",
            "latent_volatility_rms_seq_mean",
            "latent_volatility_rms_seq_slope",
            "latent_rms_seq_mean",
            "latent_rms_seq_slope",
            "denoiser_pred_rms_seq_mean",
            "denoiser_pred_delta_rms_seq_mean",
            "cfg_divergence_rms_seq_mean",
            "cfg_divergence_relative_seq_mean",
            "cfg_alignment_cosine_seq_mean",
            "denoising_consistency_residual_seq_mean",
            "denoising_update_pred_cosine_seq_mean",
        ]
        if c in df.columns
    ]

    if len(corr_cols) >= 2:
        save_correlation_heatmap(df, corr_cols, plots_dir / "correlation_heatmap.png")

    print("Done.")
    if not model_comparison.empty:
        print(model_comparison.to_string(index=False))
    else:
        print("Feature-only run completed; no model comparison because labels were unavailable.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()