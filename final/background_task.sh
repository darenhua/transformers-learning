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

DPO_RUN="${DPO_RUN:-qwen-dpo}"
DPO_EPOCHS="${DPO_EPOCHS:-3}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-1e-5}"
MERGED_BASE="${MERGED_BASE:-checkpoints/${QLORA_RUN}-merged}"

section "config"
echo "  PYTHON         = $PYTHON"
echo "  SFT_TARGET     = $SFT_TARGET"
echo "  DPO_TARGET     = $DPO_TARGET"
echo "  QLORA_EPOCHS   = $QLORA_EPOCHS"
echo "  QLORA_RUN      = $QLORA_RUN"
echo "  MERGED_BASE    = $MERGED_BASE"
echo "  DPO_RUN        = $DPO_RUN"
echo "  DPO_EPOCHS     = $DPO_EPOCHS"
echo "  DPO_BETA       = $DPO_BETA"
echo "  DPO_LR         = $DPO_LR"
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

# ---- Stage 4: merge QLoRA adapter into a full model -------------------
# DPO needs a self-contained checkpoint (config.json + safetensors) for
# AutoModelForCausalLM.from_pretrained, not an adapter. Merge once.
MERGED_BASE="${MERGED_BASE:-checkpoints/${QLORA_RUN}-merged}"
section "stage 4: merge ${QLORA_RUN} adapter -> ${MERGED_BASE}"
merge_start=$(date +%s)
if [[ "$SKIP_EXISTING" == "1" && -f "${MERGED_BASE}/DONE" ]]; then
  echo "  ${MERGED_BASE}/DONE exists — skipping"
else
  "$PYTHON" training-jobs/merge_adapter.py \
    --adapter-path "checkpoints/${QLORA_RUN}" \
    --output-path "$MERGED_BASE"
fi
echo "stage 4 done in $(elapsed_since "$merge_start")"

# ---- Stage 5: DPO training on the merged base -------------------------
DPO_RUN="${DPO_RUN:-qwen-dpo}"
DPO_EPOCHS="${DPO_EPOCHS:-3}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-1e-5}"
section "stage 5: DPO training (epochs=$DPO_EPOCHS, run=$DPO_RUN, beta=$DPO_BETA, lr=$DPO_LR)"
dpo_train_start=$(date +%s)
dpo_done="checkpoints/${DPO_RUN}/DONE"
if [[ "$SKIP_EXISTING" == "1" && -f "$dpo_done" ]]; then
  echo "  $dpo_done exists — skipping"
else
  "$PYTHON" training-jobs/qwen_dpo.py \
    --run-name "$DPO_RUN" \
    --epochs "$DPO_EPOCHS" \
    --dataset optimal_move_train.jsonl \
    --base-model "$MERGED_BASE" \
    --beta "$DPO_BETA" \
    --learning-rate "$DPO_LR" \
    --peft
fi
echo "stage 5 done in $(elapsed_since "$dpo_train_start")"

section "all done in $(elapsed_since "$start_ts")"
