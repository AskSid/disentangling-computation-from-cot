#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_datagen.sh experiments/example_datagen.yaml [stage1|stage2|both] [--layer L] [--shard N --total-shards M]

CFG_PATH=$(realpath ${1:?Provide path to experiment YAML})
STAGE=${2:-both}
shift 2 || shift $#

SHARD=""
TOTAL_SHARDS=""
LAYER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shard) SHARD="$2"; shift 2 ;;
        --total-shards) TOTAL_SHARDS="$2"; shift 2 ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${CFG_PATH}")" && git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

OUTPUT_DIR=$(uv run python -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
dg = cfg.get('data_generation', {})
print(dg.get('output_dir', 'datagen_output'))
" "${CFG_PATH}")
mkdir -p "${OUTPUT_DIR}/logs"

LOG_SUFFIX=""
[[ -n "${SHARD}" ]] && LOG_SUFFIX="_shard${SHARD}"
[[ -n "${LAYER}" ]] && LOG_SUFFIX="_layer${LAYER}"
LOG_PATH="${OUTPUT_DIR}/logs/datagen_${STAGE}${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "===== DATA GENERATION START ====="
echo "Host: $(hostname)"
echo "Config: ${CFG_PATH}"
echo "Stage: ${STAGE}"

if [[ "${STAGE}" == "stage1" || "${STAGE}" == "both" ]]; then
    echo "=== Stage 1: Response Generation ==="
    uv run python -c "
import yaml, sys
from pathlib import Path
from src.data_generation.data_gen_config import DataGenerationConfig
from src.data_generation.stage1_responses import generate_responses

cfg = yaml.safe_load(open(sys.argv[1]))
dg = cfg['data_generation']
config = DataGenerationConfig(
    model_id=dg['model_id'],
    dataset_name=dg['dataset_name'],
    output_dir=Path(dg['output_dir']),
    max_new_tokens=int(dg.get('max_new_tokens', 8192)),
    temperature=float(dg.get('temperature', 0.6)),
    top_p=float(dg.get('top_p', 0.95)),
    tensor_parallel_size=int(dg.get('tensor_parallel_size', 1)),
    num_layers=int(dg.get('num_layers', 28)),
    dtype=dg.get('dtype', 'bfloat16'),
    limit=int(dg['limit']) if dg.get('limit') else None,
    existing_responses_dir=Path(dg['existing_responses_dir']) if dg.get('existing_responses_dir') else None,
    system_prompt=dg.get('system_prompt', 'The assistant is DeepSeek-R1, created by DeepSeek.'),
)
generate_responses(config)
" "${CFG_PATH}"
fi

if [[ "${STAGE}" == "stage2" || "${STAGE}" == "both" ]]; then
    echo "=== Stage 2: Activation Harvesting ==="
    STAGE2_ARGS=""
    [[ -n "${SHARD}" ]] && STAGE2_ARGS="${STAGE2_ARGS} --shard ${SHARD} --total-shards ${TOTAL_SHARDS}"
    [[ -n "${LAYER}" ]] && STAGE2_ARGS="${STAGE2_ARGS} --layer ${LAYER}"

    uv run python -c "
import yaml, sys, argparse
from pathlib import Path
from src.data_generation.data_gen_config import DataGenerationConfig
from src.data_generation.stage2_activations import harvest_activations

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--shard', type=int, default=None)
parser.add_argument('--total-shards', type=int, default=None)
parser.add_argument('--layer', type=int, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
dg = cfg['data_generation']
config = DataGenerationConfig(
    model_id=dg['model_id'],
    dataset_name=dg.get('dataset_name', ''),
    output_dir=Path(dg['output_dir']),
    num_layers=int(dg.get('num_layers', 28)),
    dtype=dg.get('dtype', 'bfloat16'),
)

harvest_activations(config, shard_idx=args.shard, total_shards=args.total_shards, layer_idx=args.layer)
" "${CFG_PATH}" ${STAGE2_ARGS}
fi

echo "===== DATA GENERATION DONE ====="
