# Early Signals Predict Quality in Image Generation — implementation code

This repository accompanies the thesis **Early Signals Predict Quality in Image Generation**.

It has two purposes:

1. **Documentation first:** document the final analysis dataset and the experiment artifacts used in the thesis.
2. **Reproducibility second:** provide the scripts, environments, and SLURM jobs used to prepare prompts, generate images, log trajectories, score outputs, merge results, and run the analyses.

This is not intended to be a polished Python package. It is an implementation and documentation repository for the thesis experiments.

---

## Related repository

This repository contains the implementation code and experiment artifacts used in the thesis.

The reusable implementation of the early-abort filtering framework is available as the separate package **[DiffGate](https://github.com/<karbolak>/DiffGate)**.

## Repository map

```text
.
├── artifacts/
│   ├── final_dataset/
│   │   ├── analysis_input_all_quality.csv
│   │   └── README.md
│   │
│   ├── trajectory_runs/
│   │   ├── sd35_7320_1seed/
│   │   │   ├── chunk_0/
│   │   │   │   ├── generation_manifest.csv
│   │   │   │   ├── generation_manifest_with_quality.csv
│   │   │   │   ├── logs/
│   │   │   │   │   └── run.log
│   │   │   │   ├── quality_scoring_summary.json
│   │   │   │   ├── run_config.json
│   │   │   │   ├── trajectory_signals.csv
│   │   │   │   └── trajectory_signals_with_quality.csv
│   │   │   ├── chunk_1/
│   │   │   ├── ...
│   │   │   └── chunk_7/
│   │   ├── sd35_7320_2ndseed/
│   │   ├── sd35_7320_3rdseed/
│   │   ├── sd35_7320_4thseed/
│   │   └── sd35_7320_5thseed/
│   │
│   ├── analysis_logs/
│   │   ├── merge_analyze_7320_quality_29141287.out
│   │   └── merge_analyze_7320_quality_29141317.out
│   │
│   ├── trajectory_runs_no_images.tar.gz
│   ├── trajectory_runs_no_images.tar.gz.sha256
│   └── README.md
│
├── src/
│   ├── data/          # prompt dataset preparation
│   ├── generation/    # SD3.5 generation and trajectory logging
│   ├── scoring/       # CLIP, HPSv2, ImageReward scoring and score merging
│   ├── merge/         # merging chunk/seed/scoring outputs into analysis tables
│   └── analysis/      # global, prompt-controlled, best-of-five, and abort analyses
│
├── slurm/             # cluster job scripts used for generation, scoring, merging, and analysis
├── envs/              # conda environment files
├── manifests/         # artifact descriptions and expected files
├── docs/              # minimal reproduction notes
├── requirements-core.txt
├── requirements-imagereward.txt
├── DATA_NOTICE.md
├── LICENSE
└── CITATION.cff
```

---

## Main artifact: final scored analysis dataset

The main data artifact is:

```text
artifacts/final_dataset/analysis_input_all_quality.csv
```

This is the final merged dataset used for the thesis analyses. It is the easiest entry point for inspecting the experiment.

Each row corresponds to one generated image and combines:

* prompt and run identifiers;
* prompt text;
* seed/run metadata;
* final automatic quality scores: CLIP, HPSv2, and ImageReward;
* trajectory-derived prefix features;
* derived columns used for global prediction, prompt-controlled ranking, best-of-five selection, and abort simulation.

Start here if you want to understand or rerun the analyses without reconstructing the full generation pipeline.

For details, see:

```text
artifacts/final_dataset/README.md
```

---

## Trajectory-run artifacts

The per-run generation and trajectory artifacts are stored in:

```text
artifacts/trajectory_runs/
```

These folders document the five-seed generation setup used in the thesis. They are split by seed pass and then by chunk:

```text
artifacts/trajectory_runs/
├── sd35_7320_1seed/
│   ├── chunk_0/
│   ├── ...
│   └── chunk_7/
├── sd35_7320_2ndseed/
├── sd35_7320_3rdseed/
├── sd35_7320_4thseed/
└── sd35_7320_5thseed/
```

Each chunk contains non-image artifacts such as:

```text
generation_manifest.csv
generation_manifest_with_quality.csv
logs/run.log
quality_scoring_summary.json
run_config.json
trajectory_signals.csv
trajectory_signals_with_quality.csv
```

These files document what was generated, how the run was configured, what trajectory signals were logged, and how quality scores were attached to the generated samples.

The generated image directories are intentionally not included because they are large and redundant for the analysis reported in the thesis.

---

## Compressed trajectory archive

The uncompressed directory:

```text
artifacts/trajectory_runs/
```

is included for easy inspection in GitHub and local tools.

A compressed duplicate is also included:

```text
artifacts/trajectory_runs_no_images.tar.gz
artifacts/trajectory_runs_no_images.tar.gz.sha256
```

The compressed version is provided only for convenience: it preserves the same directory structure, is easier to download as one file, and can be verified with the checksum.

To verify the compressed archive:

```bash
cd artifacts
sha256sum -c trajectory_runs_no_images.tar.gz.sha256
```

---

## Analysis logs

The final merge/analysis job logs are stored in:

```text
artifacts/analysis_logs/
```

This directory currently contains:

```text
merge_analyze_7320_quality_29141287.out
merge_analyze_7320_quality_29141317.out
```

These files are included as an audit trail for the cluster jobs that merged the scored data and produced analysis outputs. They are not needed for running the analysis scripts, but they document the execution of the final processing stage.

---

## How to inspect the documented experiment

Start with the final dataset:

```bash
python - <<'PY'
import pandas as pd

path = "artifacts/final_dataset/analysis_input_all_quality.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("First columns:")
print(df.columns.tolist()[:40])
print(df.head())
PY
```

Then inspect the per-run artifacts if you want to audit how the final dataset was produced:

```text
artifacts/trajectory_runs/sd35_7320_1seed/chunk_0/run_config.json
artifacts/trajectory_runs/sd35_7320_1seed/chunk_0/generation_manifest.csv
artifacts/trajectory_runs/sd35_7320_1seed/chunk_0/trajectory_signals_with_quality.csv
artifacts/trajectory_runs/sd35_7320_1seed/chunk_0/logs/run.log
```

The final dataset is the recommended entry point for analysis. The trajectory-run artifacts are included mainly for documentation and traceability.

---

## Reproducing analyses from the final dataset

The scripts in `src/analysis/` operate on the final scored dataset. They cover:

* global median-split prediction;
* prefix-length diagnostics;
* prompt-controlled correlations and pairwise ranking;
* best-of-five seed selection;
* early-abort threshold simulation;
* feature diagnostics.

The intended input is:

```text
artifacts/final_dataset/analysis_input_all_quality.csv
```

Depending on the script, paths may need to be edited at the top of the file or supplied as command-line arguments.

For more detail, see:

```text
docs/REPRODUCING_RESULTS.md
```

---

## Re-running the full pipeline

The full pipeline is substantially more expensive than rerunning the analyses from the final CSV. The code is included for transparency and reproducibility:

1. `src/data/` prepares the prompt set.
2. `src/generation/` generates images with Stable Diffusion 3.5 Large and logs trajectory signals.
3. `src/scoring/` scores generated images with CLIP, HPSv2, and ImageReward.
4. `src/merge/` merges chunk, seed, and scoring outputs into the final analysis table.
5. `src/analysis/` reproduces the thesis analyses from the final scored table.

The corresponding SLURM scripts are in:

```text
slurm/
```

---

## Environments

Two environments are documented because ImageReward was run separately from the rest of the pipeline.

Core environment:

```bash
conda env create -f envs/core.yml
conda activate early-signals-core
```

ImageReward environment:

```bash
conda env create -f envs/imagereward.yml
conda activate early-signals-imagereward
```

The root `environment.yml`, if present, is a convenience alias for the core environment only. It is not meant to represent the ImageReward setup.

---

## What is intentionally not included

This repository does not include:

* generated image directories;
* model weights;
* Hugging Face caches;
* virtual environments;
* large temporary scratch outputs.

The trajectory-run folders are included only in non-image form. They document the experiment structure and logged trajectory/scoring artifacts without storing the generated images.

The MIT licence applies to the code in this repository. The final dataset and derived artifacts are provided for thesis documentation and reproducibility and remain subject to the licences and terms of the underlying datasets, models, and scoring tools. See `DATA_NOTICE.md`.

---

## Citation

If you use or refer to this repository, please cite it using the metadata in:

```text
CITATION.cff
```
