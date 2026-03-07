# Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought

Codebase for the paper "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought" (https://arxiv.org/abs/2603.05488) 

Interactive app of our experiments: https://reasoning-theater.streamlit.app/

Experiments on the app were run on DeepSeek-R1-0528-671B and GPT-OSS-120B across a subset of MMLU-Redux-2.0 and GPQA-Diamond.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # add your OPENROUTER_API_KEY if running the CoT monitor / inflection stages
```

The project has two dependency surfaces:
- `pyproject.toml` — full project (training, inference, analysis). Managed by `uv`.
- `streamlit_app/requirements.txt` — lightweight dependencies for the Streamlit app.

## Usage

Everything is configured via YAML files in `experiments/`. See `experiments/example_datagen.yaml` and `experiments/example_analysis.yaml` for the full schema.

The pipeline has two phases:
- **Data generation**: 1) collect model responses via vLLM and 2) harvest hidden-state activations via nnsight.
- **Analysis**: 1) train probes on activations, 2) collect forced-answering predictions, 3) collect CoT monitor predictions, and 4) identify and analyze inflection points.

### Example 1: Full pipeline with a single layer's activations

Generate data, collect activations for one layer, train a single probe, and run the full analysis:

```bash
bash scripts/run_datagen.sh experiments/example_datagen.yaml both --layer 17
bash scripts/run_pipeline.sh experiments/example_analysis.yaml
```

The example analysis config is set up for this by default (`selected_layer: 17`).

### Example 2: Full pipeline with all layer activations

For multi-GPU setups, use `tensor_parallel_size` for vLLM inference and sharding for parallel activation harvesting. See `experiments/example_full_datagen.yaml` and `experiments/example_full_analysis.yaml` for the full configs.

```bash
# Stage 1: generate responses (uses tensor parallelism across 4 GPUs)
bash scripts/run_datagen.sh experiments/example_full_datagen.yaml stage1

# Stage 2: harvest activations in parallel across 4 jobs
for i in 0 1 2 3; do
  bash scripts/run_datagen.sh experiments/example_full_datagen.yaml stage2 --shard $i --total-shards 4 &
done
wait

# Analysis: train probes for all layers, run all stages
bash scripts/run_pipeline.sh experiments/example_full_analysis.yaml
```

### Reference

Individual stages can be run separately:

```bash
# Data generation
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage1              # responses only
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage2 --layer 17   # single layer activations
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage2              # all layer activations

# Single probe layer (useful for debugging)
bash scripts/run_probe.sh experiments/example_analysis.yaml 17
```

Toggle individual analysis stages in the config YAML:

```yaml
setup:
  enabled: true           # prepare metadata and train/val/test split
probe:
  enabled: true           # train attention probes on hidden-state activations
forced_answer:
  enabled: false          # force the model to answer at each reasoning step
cot_monitor:
  enabled: false          # have an external LLM predict the answer from partial reasoning text
plots:
  enabled: true           # generate comparison plots (probe vs forced answer vs CoT monitor)
inflections:
  enabled: false          # detect backtracking / realization moments and correlate with probe shifts
```

Results are written to `results/<run_name>/`, with logs in `results/<run_name>/logs/`.
