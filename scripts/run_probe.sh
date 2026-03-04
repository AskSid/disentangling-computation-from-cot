#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_probe.sh experiments/example_analysis.yaml 15

CFG_PATH=$(realpath ${1:?Provide path to experiment YAML})
LAYER=${2:?Provide layer index}

ROOT_DIR="$(cd "$(dirname "${CFG_PATH}")" && git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

RUN_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('run',{}).get('run_name','run'))" "${CFG_PATH}")
RESULTS_DIR=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('run',{}).get('results_dir','results'))" "${CFG_PATH}")
RUN_ROOT="${ROOT_DIR}/${RESULTS_DIR}/${RUN_NAME}"
mkdir -p "${RUN_ROOT}/logs"

LOG_PATH="${RUN_ROOT}/logs/probe_layer_${LAYER}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Training probe for layer ${LAYER}..."
uv run python -m src.analysis.run_probing --config "${CFG_PATH}" --layer "${LAYER}"
