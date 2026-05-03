#!/usr/bin/env bash
# Long-running pipeline: generate SFT + DPO datasets, then train QLoRA.
# Designed to be left running unattended (e.g. nohup or tmux).
#
# Usage:
#     cd final
#     ./background_task.sh                          # foreground
#     nohup ./background_task.sh &                  # background, log->./background_task.log
#     SFT_TARGET=10000 ./background_task.sh         # override knobs via env
#
# Override via env vars:
#     SFT_TARGET (default 20000), DPO_TARGET (default 5000)
#     QLORA_EPOCHS (default 3), QLORA_RUN (default qwen-qlora)
#     SCAN_PATH (default ../scan/scan_31/scan_linux)
#     SKIP_EXISTING (default 1) — skip stages whose output files already exist
#
# To re-run a stage from scratch, delete its output files (or set
# SKIP_EXISTING=0 to ignore the cache).

set -euo pipefail

# Always work from final/ — datasets land here, training resolves paths
# relative to it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/ubuntu/gpu-shi/tf-learn/.env/bin/python}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/background_task.log}"
SFT_TARGET="${SFT_TARGET:-20000}"
DPO_TARGET="${DPO_TARGET:-5000}"
QLORA_EPOCHS="${QLORA_EPOCHS:-3}"
QLORA_RUN="${QLORA_RUN:-qwen-qlora}"
SCAN_PATH="${SCAN_PATH:-../scan/scan_31/scan_linux}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Mirror stdout+stderr to the log file. `tee -a` works whether stdin is a
# tty (foreground) or detached (nohup).
exec > >(tee -a "$LOG_FILE") 2>&1

start_ts=$(date +%s)
section() {
  printf '\n==== [%s] %s ====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}
elapsed_since() {
  local since=$1
  local now=$(date +%s)
  printf '%dm%02ds' $(( (now - since) / 60 )) $(( (now - since) % 60 ))
}

section "config"
echo "  PYTHON         = $PYTHON"
echo "  SFT_TARGET     = $SFT_TARGET"
echo "  DPO_TARGET     = $DPO_TARGET"
echo "  QLORA_EPOCHS   = $QLORA_EPOCHS"
echo "  QLORA_RUN      = $QLORA_RUN"
echo "  SCAN_PATH      = $SCAN_PATH"
echo "  SKIP_EXISTING  = $SKIP_EXISTING"
echo "  LOG_FILE       = $LOG_FILE"
echo "  cwd            = $(pwd)"

# ---- Pre-flight checks -------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not executable at $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$SCAN_PATH" ]]; then
  echo "ERROR: Scan binary missing/unexecutable at $SCAN_PATH" >&2
  echo "       (extract scan.zip or set SCAN_PATH to the right path)" >&2
  exit 1
fi

# ---- Stage 1: SFT (valid_move) dataset ---------------------------------
section "stage 1: SFT dataset (target=$SFT_TARGET)"
sft_start=$(date +%s)
if [[ "$SKIP_EXISTING" == "1" && -f "valid_move_train.jsonl" && -f "valid_move_test.jsonl" ]]; then
  echo "  valid_move_{train,test}.jsonl already exist — skipping"
else
  "$PYTHON" generate_dataset_cli.py valid \
    --target "$SFT_TARGET" --output-base valid_move
fi
echo "stage 1 done in $(elapsed_since "$sft_start")"

# ---- Stage 2: DPO (optimal_move) dataset -------------------------------
section "stage 2: DPO dataset (target=$DPO_TARGET)"
dpo_start=$(date +%s)
if [[ "$SKIP_EXISTING" == "1" && -f "optimal_move_train.jsonl" && -f "optimal_move_test.jsonl" ]]; then
  echo "  optimal_move_{train,test}.jsonl already exist — skipping"
else
  "$PYTHON" generate_dataset_cli.py optimal \
    --target "$DPO_TARGET" --output-base optimal_move \
    --scan-path "$SCAN_PATH"
fi
echo "stage 2 done in $(elapsed_since "$dpo_start")"

# ---- Stage 3: QLoRA training ------------------------------------------
section "stage 3: QLoRA training (epochs=$QLORA_EPOCHS, run=$QLORA_RUN)"
qlora_start=$(date +%s)
qlora_done="checkpoints/${QLORA_RUN}/DONE"
if [[ "$SKIP_EXISTING" == "1" && -f "$qlora_done" ]]; then
  echo "  $qlora_done exists — skipping"
else
  "$PYTHON" training-jobs/qwen_qlora.py \
    --run-name "$QLORA_RUN" \
    --epochs "$QLORA_EPOCHS" \
    --dataset valid_move_train.jsonl
fi
echo "stage 3 done in $(elapsed_since "$qlora_start")"

section "all done in $(elapsed_since "$start_ts")"
