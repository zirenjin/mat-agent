#!/usr/bin/env bash
# Fine-tune a MACE foundation model on 950 rMD17 aspirin frames.
# Identical config for both branches (controlled experiment); only NAME + FOUNDATION differ.
# Strategy: full-weight NAIVE single-head FT (multiheads_finetuning=False) -> max adaptation to
# rMD17 aspirin, no replay-data download, fully controlled. float64 (both foundations are f64).
# Loss: force-dominated stage-1 (e=1,f=1000) then SWA stage-2 energy-tightening (e=1000,f=100).
set -euo pipefail
ROOT=/GenSIvePFS/users/public/mat-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/envs/mace_env"
cd "$ROOT"

NAME="$1"; FOUNDATION="$2"
MAXEP="${3:-250}"; STARTSWA="${4:-200}"; LR="${5:-0.01}"; WD="${6:-5e-7}"
RUNDIR="runs/ft/$NAME"
mkdir -p "$RUNDIR"
echo "[run_mace_ft] NAME=$NAME FOUNDATION=$FOUNDATION MAXEP=$MAXEP STARTSWA=$STARTSWA LR=$LR WD=$WD"

mace_run_train \
  --name="$NAME" \
  --foundation_model="$FOUNDATION" \
  --multiheads_finetuning=False \
  --E0s="average" \
  --train_file="data/rmd17_aspirin_ft/train.extxyz" \
  --valid_file="data/rmd17_aspirin_ft/ftest.extxyz" \
  --test_file="data/rmd17_aspirin_ft/ftest.extxyz" \
  --energy_key="ref_energy" --forces_key="ref_forces" \
  --loss="weighted" --energy_weight=1 --forces_weight=1000 \
  --swa --start_swa="$STARTSWA" --swa_energy_weight=1000 --swa_forces_weight=100 \
  --ema --ema_decay=0.99 \
  --lr="$LR" --weight_decay="$WD" \
  --batch_size=10 --valid_batch_size=25 \
  --max_num_epochs="$MAXEP" --patience=40 \
  --eval_interval=1 \
  --default_dtype=float64 --device=cuda --seed=1234 \
  --model_dir="$RUNDIR" --log_dir="$RUNDIR/logs" \
  --checkpoints_dir="$RUNDIR/checkpoints" --results_dir="$RUNDIR/results" \
  --save_cpu --restart_latest
echo "[run_mace_ft] DONE name=$NAME exit=$?"
