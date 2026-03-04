#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_pipeline.sh experiments/example_analysis.yaml

CFG_PATH=$(realpath ${1:?Provide path to experiment YAML})

ROOT_DIR="$(cd "$(dirname "${CFG_PATH}")" && git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

RUN_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('run',{}).get('run_name','run'))" "${CFG_PATH}")
RESULTS_DIR=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('run',{}).get('results_dir','results'))" "${CFG_PATH}")
RUN_ROOT="${ROOT_DIR}/${RESULTS_DIR}/${RUN_NAME}"
mkdir -p "${RUN_ROOT}/logs"

LOG_PATH="${RUN_ROOT}/logs/pipeline.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "===== PIPELINE START ====="
echo "Host: $(hostname)"
echo "Config: ${CFG_PATH}"
echo "Run root: ${RUN_ROOT}"

# Setup
SETUP_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('setup',{}).get('enabled',True))" "${CFG_PATH}")
if [[ "${SETUP_ENABLED}" == "True" ]]; then
    echo "Running setup_data..."
    uv run python -m src.analysis.setup_data --config "${CFG_PATH}"
else
    echo "Skipping setup_data (disabled)"
fi

# Probe training
PROBE_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('probe',{}).get('enabled',True))" "${CFG_PATH}")
if [[ "${PROBE_ENABLED}" == "True" ]]; then
    echo "Running probe training/evaluation..."
    LAYERS_JSON=$(uv run python - <<'PY' "${CFG_PATH}"
import yaml, sys, json
cfg = yaml.safe_load(open(sys.argv[1]))
probe = cfg.get("probe", {})
if "num_layers" not in probe:
    raise SystemExit("probe.num_layers is required")
num_layers = int(probe["num_layers"])
selected_layer = int(probe.get("selected_layer", -1))
if selected_layer == -1:
    layers = list(range(num_layers))
else:
    layers = [selected_layer]
print(json.dumps(layers))
PY
    )

    for LAYER in $(uv run python -c "import json; [print(l) for l in json.loads('${LAYERS_JSON}')]"); do
        echo "Training probe for layer ${LAYER}..."
        uv run python -m src.analysis.run_probing --config "${CFG_PATH}" --layer "${LAYER}"
    done
else
    echo "Skipping probe training (disabled)"
fi

# Forced answering
FORCED_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('forced_answer',{}).get('enabled',True))" "${CFG_PATH}")
if [[ "${FORCED_ENABLED}" == "True" ]]; then
    echo "Running forced-answer injection..."
    uv run python -m src.analysis.run_forced_answering --config "${CFG_PATH}"
else
    echo "Skipping forced-answer injection (disabled)"
fi

# CoT monitor
COT_MONITOR_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('cot_monitor',{}).get('enabled',True))" "${CFG_PATH}")
if [[ "${COT_MONITOR_ENABLED}" == "True" ]]; then
    echo "Running CoT monitor inference and injection..."
    uv run python -m src.analysis.run_cot_monitor --config "${CFG_PATH}"
else
    echo "Skipping CoT monitor (disabled)"
fi

# Plots
PLOTS_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('plots',{}).get('enabled',False))" "${CFG_PATH}")
if [[ "${PLOTS_ENABLED}" == "True" ]]; then
    echo "Generating plots..."
    MODEL_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('plots',{}).get('model_name','model'))" "${CFG_PATH}")
    DATASET_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('plots',{}).get('dataset_name','dataset'))" "${CFG_PATH}")
    uv run python -m src.analysis.plots \
        --results_dir "${RUN_ROOT}" \
        --model_name "${MODEL_NAME}" \
        --dataset_name "${DATASET_NAME}"
else
    echo "Skipping plots (disabled)"
fi

# Inflection point analysis
INFLECTIONS_ENABLED=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('inflections',{}).get('enabled',False))" "${CFG_PATH}")
if [[ "${INFLECTIONS_ENABLED}" == "True" ]]; then
    echo "Finding inflection points..."
    INFLECTION_MODEL=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('cot_monitor',{}).get('model','google/gemini-2.5-flash'))" "${CFG_PATH}")
    INFLECTION_API_KEY_ENV=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('cot_monitor',{}).get('api_key_env','OPENROUTER_API_KEY'))" "${CFG_PATH}")
    INFLECTION_WORKERS=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('inflections',{}).get('workers',50))" "${CFG_PATH}")
    RESPONSES_DIR=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('data',{}).get('responses_dir',''))" "${CFG_PATH}")
    uv run python -m src.analysis.find_inflection_points \
        --responses-dir "${RESPONSES_DIR}" \
        --output "${RUN_ROOT}/inflection_results.json" \
        --split-path "${RUN_ROOT}/train_val_test_split.json" \
        --model "${INFLECTION_MODEL}" \
        --api-key-env "${INFLECTION_API_KEY_ENV}" \
        --workers "${INFLECTION_WORKERS}"
    echo "Running inflection-probe correlation analysis..."
    INFLECTION_LAYER=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('inflections',{}).get('layer', cfg.get('probe',{}).get('num_layers',28) - 1))" "${CFG_PATH}")
    MODEL_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('plots',{}).get('model_name',''))" "${CFG_PATH}")
    DATASET_NAME=$(uv run python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('plots',{}).get('dataset_name',''))" "${CFG_PATH}")
    uv run python -m src.analysis.inflection_point_analysis \
        --inflection-path "${RUN_ROOT}/inflection_results.json" \
        --step-level-dir "${RUN_ROOT}/step_level" \
        --metadata-path "${RUN_ROOT}/predictions_metadata.csv" \
        --layer "${INFLECTION_LAYER}" \
        --model-name "${MODEL_NAME}" \
        --dataset-name "${DATASET_NAME}"
else
    echo "Skipping inflection point analysis (disabled)"
fi

echo "===== PIPELINE DONE ====="
