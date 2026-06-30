# Reproducing results

The recommended reproducibility path starts from the final scored analysis dataset:

```text
artifacts/final_dataset/analysis_input_all_quality.csv
```

This file is the merged table used for the thesis analyses. It contains the prompts, seeds, final automatic quality scores, and trajectory-derived features needed for global prediction, prompt-controlled ranking, best-of-five selection, and abort simulation.

## 1. Create environments

Most scripts use the core environment:

```bash
conda env create -f envs/core.yml
conda activate early-signals-core
```

ImageReward scoring used a separate environment:

```bash
conda env create -f envs/imagereward.yml
conda activate early-signals-imagereward
```

The separate ImageReward environment is kept because it reflects how the scoring pipeline was actually run.

## 2. Inspect the dataset

```bash
python - <<'PY'
import pandas as pd
path = 'artifacts/final_dataset/analysis_input_all_quality.csv'
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist()[:50])
PY
```

## 3. Rerun the analysis scripts

The main scripts are in:

```text
src/analysis/
```

They reproduce the analyses reported in the thesis from the final scored dataset:

- global quality prediction;
- prefix-length diagnostics;
- prompt-controlled correlations and pairwise ranking;
- best-of-five selection;
- early-abort simulation;
- feature diagnostics.

Some scripts may contain cluster-specific paths from the original run. Replace those paths with:

```text
artifacts/final_dataset/analysis_input_all_quality.csv
```

## 4. Full pipeline rerun

A full rerun from prompts to generated images is possible in principle but is much more expensive. The relevant scripts are included for auditability:

```text
src/data/         prompt preparation
src/generation/   image generation and trajectory logging
src/scoring/      automatic scoring
src/merge/        merging into final analysis table
slurm/            cluster job scripts
```

The repository therefore supports two levels of reproducibility:

1. **analysis reproducibility** from the included final scored dataset;
2. **pipeline auditability** through the generation, logging, scoring, merge, and SLURM scripts.
