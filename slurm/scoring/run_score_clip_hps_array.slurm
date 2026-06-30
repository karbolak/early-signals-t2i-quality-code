#!/bin/bash
#SBATCH --job-name=score_clip_hps
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_PATH:-$HOME/venvs/bscthesis}/bin/activate"

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/bscthesis}"
export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
mkdir -p slurm_logs

PASS_DIRS=(sd35_7320_1seed sd35_7320_2ndseed sd35_7320_3rdseed sd35_7320_4thseed sd35_7320_5thseed)
PASS_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % 8))
CHUNK_DIR="results/${PASS_DIRS[$PASS_INDEX]}/chunk_${CHUNK_INDEX}"

python src/scoring/score_merge_7320_quality_separate_venvs.py score-chunk \
  --chunk-dir "$CHUNK_DIR" \
  --targets clip,hpsv2 \
  --device cuda \
  --batch-size 16 \
  --save-every 100 \
  --clip-model-name openai/clip-vit-base-patch32 \
  --hps-version v2.1
