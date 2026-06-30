from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def safe_corr(x: pd.Series, y: pd.Series) -> dict:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return {"n": int(len(clean)), "pearson_r": np.nan, "spearman_rho": np.nan}
    return {
        "n": int(len(clean)),
        "pearson_r": float(pearsonr(clean["x"], clean["y"]).statistic),
        "spearman_rho": float(spearmanr(clean["x"], clean["y"]).statistic),
    }


def infer_prefix_features(df: pd.DataFrame, prefix_steps: int) -> list[str]:
    marker = f"_prefix{prefix_steps}_"
    return [c for c in df.columns if marker in c and c.endswith(("_mean", "_slope", "_last"))]


def within_prompt_pairwise_accuracy(df: pd.DataFrame, score_col: str, quality_col: str, prompt_col: str) -> dict:
    concordant = 0
    discordant = 0
    ties_score = 0
    ties_quality = 0

    for _, g in df[[prompt_col, score_col, quality_col]].dropna().groupby(prompt_col):
        rows = g[[score_col, quality_col]].to_numpy(dtype=float)
        for a, b in combinations(rows, 2):
            ds = a[0] - b[0]
            dq = a[1] - b[1]
            if np.isclose(ds, 0):
                ties_score += 1
                continue
            if np.isclose(dq, 0):
                ties_quality += 1
                continue
            if ds * dq > 0:
                concordant += 1
            else:
                discordant += 1

    usable = concordant + discordant
    return {
        "score_col": score_col,
        "usable_pairs": int(usable),
        "concordant_pairs": int(concordant),
        "discordant_pairs": int(discordant),
        "score_ties": int(ties_score),
        "quality_ties": int(ties_quality),
        "pairwise_accuracy": float(concordant / usable) if usable else np.nan,
    }


def best_of_prompt_selection(df: pd.DataFrame, score_col: str, quality_col: str, prompt_col: str) -> dict:
    clean = df[[prompt_col, score_col, quality_col]].dropna().copy()
    grouped = clean.groupby(prompt_col, sort=False)
    selected_idx = grouped[score_col].idxmax()
    selected = clean.loc[selected_idx]
    random_expected = grouped[quality_col].mean()
    oracle_best = grouped[quality_col].max()

    return {
        "score_col": score_col,
        "n_prompts": int(grouped.ngroups),
        "mean_selected_quality": float(selected[quality_col].mean()),
        "mean_random_expected_quality": float(random_expected.mean()),
        "mean_oracle_best_quality": float(oracle_best.mean()),
        "lift_vs_random": float(selected[quality_col].mean() - random_expected.mean()),
        "fraction_of_oracle_gain": float(
            (selected[quality_col].mean() - random_expected.mean())
            / (oracle_best.mean() - random_expected.mean() + 1e-12)
        ),
    }


def grouped_logreg_cv(df: pd.DataFrame, feature_cols: list[str], label_col: str, group_col: str, n_splits: int) -> tuple[dict, pd.Series, pd.DataFrame]:
    clean = df.dropna(subset=feature_cols + [label_col, group_col]).copy()
    X = clean[feature_cols].to_numpy(dtype=float)
    y = clean[label_col].to_numpy(dtype=int)
    groups = clean[group_col].to_numpy()

    n_groups = clean[group_col].nunique()
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        raise ValueError("Need at least two prompt groups for GroupKFold.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
    ])

    cv = GroupKFold(n_splits=actual_splits)
    probs = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    metrics = {
        "n": int(len(clean)),
        "n_features": int(len(feature_cols)),
        "n_groups": int(n_groups),
        "n_splits": int(actual_splits),
        **safe_auc(y, probs),
    }

    model.fit(X, y)
    coefs = model.named_steps["clf"].coef_[0]
    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coefficient_standardized": coefs})
        .sort_values("coefficient_standardized", ascending=False)
        .reset_index(drop=True)
    )

    pred = pd.Series(index=clean.index, data=100.0 * probs, name="group_cv_learned_health_score_0_100")
    return metrics, pred, coef_df


def prompt_variance_decomposition(df: pd.DataFrame, quality_col: str, prompt_col: str) -> dict:
    clean = df[[prompt_col, quality_col]].dropna()
    prompt_means = clean.groupby(prompt_col)[quality_col].mean()
    overall_mean = clean[quality_col].mean()
    n_per = clean.groupby(prompt_col)[quality_col].size()

    ss_between = float(((prompt_means - overall_mean) ** 2 * n_per).sum())
    ss_total = float(((clean[quality_col] - overall_mean) ** 2).sum())
    ss_within = float(clean.groupby(prompt_col)[quality_col].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())

    return {
        "quality_col": quality_col,
        "n_rows": int(len(clean)),
        "n_prompts": int(clean[prompt_col].nunique()),
        "between_prompt_fraction_of_total_ss": float(ss_between / ss_total) if ss_total else np.nan,
        "within_prompt_fraction_of_total_ss": float(ss_within / ss_total) if ss_total else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-controlled analysis for multiseed SD3.5 trajectories.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Use rich_merged_dataset.csv from the analysis run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-col", default="new_clip_score")
    parser.add_argument("--prompt-col", default="prompt_id")
    parser.add_argument("--label-col", default="good_label")
    parser.add_argument("--health-col", default="health_score_0_100")
    parser.add_argument("--prefix-steps", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_csv)

    required = {args.quality_col, args.prompt_col, args.label_col, args.health_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    counts = df.groupby(args.prompt_col).size()
    centered_quality_col = f"{args.quality_col}_prompt_centered"
    centered_health_col = f"{args.health_col}_prompt_centered"
    df[centered_quality_col] = df[args.quality_col] - df.groupby(args.prompt_col)[args.quality_col].transform("mean")
    df[centered_health_col] = df[args.health_col] - df.groupby(args.prompt_col)[args.health_col].transform("mean")

    prefix_features = infer_prefix_features(df, args.prefix_steps)
    if not prefix_features:
        raise ValueError(f"No compact prefix features found for prefix {args.prefix_steps}.")

    grouped_metrics, grouped_pred, grouped_coef = grouped_logreg_cv(
        df,
        feature_cols=prefix_features,
        label_col=args.label_col,
        group_col=args.prompt_col,
        n_splits=args.n_splits,
    )
    df.loc[grouped_pred.index, grouped_pred.name] = grouped_pred

    correlations = pd.DataFrame([
        {"analysis": "raw_training_free_health_vs_quality", **safe_corr(df[args.health_col], df[args.quality_col])},
        {"analysis": "prompt_centered_training_free_health_vs_quality", **safe_corr(df[centered_health_col], df[centered_quality_col])},
        {"analysis": "group_cv_learned_health_vs_quality", **safe_corr(df["group_cv_learned_health_score_0_100"], df[args.quality_col])},
    ])

    pairwise = pd.DataFrame([
        within_prompt_pairwise_accuracy(df, args.health_col, args.quality_col, args.prompt_col),
        within_prompt_pairwise_accuracy(df, "group_cv_learned_health_score_0_100", args.quality_col, args.prompt_col),
    ])

    selection = pd.DataFrame([
        best_of_prompt_selection(df, args.health_col, args.quality_col, args.prompt_col),
        best_of_prompt_selection(df, "group_cv_learned_health_score_0_100", args.quality_col, args.prompt_col),
    ])

    variance = prompt_variance_decomposition(df, args.quality_col, args.prompt_col)

    summary = {
        "input_csv": str(args.input_csv),
        "n_rows": int(len(df)),
        "n_prompts": int(df[args.prompt_col].nunique()),
        "rows_per_prompt": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(counts.median()),
        },
        "prefix_steps": int(args.prefix_steps),
        "prefix_feature_count": int(len(prefix_features)),
        "prefix_features": prefix_features,
        "grouped_logreg_metrics": grouped_metrics,
        "prompt_variance_decomposition": variance,
        "correlations": correlations.to_dict(orient="records"),
        "within_prompt_pairwise_accuracy": pairwise.to_dict(orient="records"),
        "best_of_prompt_selection": selection.to_dict(orient="records"),
    }

    df.to_csv(args.output_dir / "prompt_controlled_dataset.csv", index=False)
    correlations.to_csv(args.output_dir / "prompt_controlled_correlations.csv", index=False)
    pairwise.to_csv(args.output_dir / "within_prompt_pairwise_accuracy.csv", index=False)
    selection.to_csv(args.output_dir / "best_of_prompt_selection.csv", index=False)
    grouped_coef.to_csv(args.output_dir / "grouped_logreg_coefficients.csv", index=False)
    pd.DataFrame([variance]).to_csv(args.output_dir / "prompt_variance_decomposition.csv", index=False)
    with open(args.output_dir / "prompt_controlled_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Prompt-controlled analysis complete.")
    print(f"Rows: {len(df)} | prompts: {df[args.prompt_col].nunique()} | rows/prompt: {counts.min()}-{counts.max()}")
    print("\nGrouped logistic regression:")
    print(pd.DataFrame([grouped_metrics]).to_string(index=False))
    print("\nWithin-prompt pairwise ranking:")
    print(pairwise.to_string(index=False))
    print("\nBest-of-prompt selection:")
    print(selection.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
