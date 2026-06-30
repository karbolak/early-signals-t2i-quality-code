#!/bin/bash
#SBATCH --job-name=merge_analyze
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_PATH:-$HOME/venvs/bscthesis}/bin/activate"

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/bscthesis}"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

MERGED_DIR="results/sd35_7320_5seeds_merged_quality"
MERGED_CSV="$MERGED_DIR/analysis_input_all_quality.csv"

python src/merge/merge_7320_quality_for_analysis.py \
  --results-root results \
  --run-dirs sd35_7320_1seed sd35_7320_2ndseed sd35_7320_3rdseed sd35_7320_4thseed sd35_7320_5thseed \
  --n-chunks 8 \
  --output-dir "$MERGED_DIR" \
  --require-quality \
  --expected-rows 36600

# Edit the quality column names below to match the merged CSV.
for TARGET in clip hpsv2 imagereward; do
  case "$TARGET" in
    clip) QUALITY_COL="clip_score" ;;
    hpsv2) QUALITY_COL="hpsv2_score" ;;
    imagereward) QUALITY_COL="imagereward_score" ;;
  esac
  OUT="results/rich_analysis_${TARGET}_prefix5"
  python src/analysis/analyze_trajectory_features.py \
    --input-csv "$MERGED_CSV" \
    --output-dir "$OUT" \
    --quality-target existing \
    --quality-col "$QUALITY_COL" \
    --feature-set compact \
    --prefix-steps 5

  python src/analysis/prompt_controlled_analysis.py \
    --input-csv "$OUT/rich_merged_dataset.csv" \
    --output-dir "results/prompt_controlled_${TARGET}_prefix5" \
    --quality-col "$QUALITY_COL" \
    --prefix-steps 5
done
