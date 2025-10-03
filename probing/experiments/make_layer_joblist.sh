#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# turns yaml to a joblist (txt) if you run in terminal 
BASE_CFG="probing/experiments/single_layer_sweep.yaml"   # your YAML snippet at top; must contain a 'params:' block with 'layer_idx:' and 'run_name:'
TRAIN_SCRIPT="/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/train_probe.py"
OUTROOT="/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/pr,obing"   # repo root for consistency
GEN_DIR="$OUTROOT/experiments/layers"      # generated YAMLs go here
JOBLIST="$OUTROOT/experiments/layers/joblist.txt"

mkdir -p "$GEN_DIR"
: > "$JOBLIST"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "ERROR: base config not found: $BASE_CFG" >&2
  exit 1
fi

# Make sure base has the lines we will rewrite
grep -qE '^[[:space:]]*layer_idx:' "$BASE_CFG" || { echo "ERROR: base YAML missing 'layer_idx:' under params"; exit 1; }
grep -qE '^[[:space:]]*run_name:'   "$BASE_CFG" || { echo "ERROR: base YAML missing 'run_name:' under params"; exit 1; }

for L in $(seq 0 60); do
  CFG="$GEN_DIR/layer_${L}.yaml"
  cp "$BASE_CFG" "$CFG"

  # Update layer_idx in-place (first matching line)
  # also make run_name unique per-layer to avoid clobbering outputs
  # these sed expressions are conservative (match line beginning + optional spaces)
  sed -i -E "0,/^[[:space:]]*layer_idx:[[:space:]]*[0-9]+/s//  layer_idx: ${L}/" "$CFG"
  sed -i -E "0,/^[[:space:]]*run_name:[[:space:]]*\"?[^\"#]+\"?/s//  run_name: \"last_layer_lr_sweep_layer${L}\"/" "$CFG"

  # Each job runs a single layer
  echo "python -u $TRAIN_SCRIPT --config $CFG --mode single" >> "$JOBLIST"
done

echo "Wrote $(wc -l < "$JOBLIST") jobs to $JOBLIST"
echo "Generated YAMLs in: $GEN_DIR"
