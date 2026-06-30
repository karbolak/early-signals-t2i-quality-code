#!/bin/bash
#SBATCH --job-name=sd35_generation
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --array=0-7
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

mkdir -p "$HF_HOME" "$PROJECT_ROOT/results" "$PROJECT_ROOT/data" "$PROJECT_ROOT/slurm_logs"
cd "$PROJECT_ROOT"

CHUNK_SIZE="${CHUNK_SIZE:-1000}"
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_INDEX=$((START_INDEX + CHUNK_SIZE))
REPLICATE_START="${REPLICATE_START:-0}"
OUTDIR="results/sd35_7320_seed${REPLICATE_START}/chunk_${SLURM_ARRAY_TASK_ID}"

python src/generation/generate_trajectories_rich_multiseed_array.py \
  --input-csv data/processed/final_generation_prompts.csv \
  --output-dir "$OUTDIR" \
  --num-inference-steps 25 \
  --guidance-scale 7.0 \
  --early-steps 25 \
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX" \
  --seeds-per-prompt 1 \
  --replicate-start "$REPLICATE_START" \
  --base-seed 12345 \
  --seed-stride 1000
