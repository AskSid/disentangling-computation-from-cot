"""Plotting functions for early decoder analysis."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib import cm
import seaborn as sns

if TYPE_CHECKING:
    from .data_loading import RunData


AGREEMENT_COLORS = {
    'both_correct': '#4CAF50',      # soft green - agreement, both right
    'both_incorrect': '#EF5350',    # soft coral red - agreement, both wrong
    'probe_only': '#42A5F5',        # sky blue - disagreement, probe wins
    'forced_only': '#FFA726',       # warm amber - disagreement, forced wins
}

METHOD_COLORS = {
    'probe': ('#1976D2', '#64B5F6'),       # blue / light blue
    'forced': ('#388E3C', '#81C784'),      # green / light green
    'cot_monitor': ('#D32F2F', '#E57373'),    # red / light red
}

ECE_COLOR = '#1565C0'               # dark blue
BRIER_COLOR = '#388E3C'             # dark green

CALIBRATION_LINE_COLOR = '#2196F3'  # blue
PERFECT_CALIBRATION_COLOR = 'black'

# R1 uses <｜Assistant｜> (capital A, special Unicode), GPT-OSS uses <|message|>
ASSISTANT_MARKERS = {
    "<｜Assistant｜>",  # R1
    "<|message|>",      # GPT-OSS
}

FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS_LABEL = 20
FONT_SIZE_TICK_LABEL = 16
FONT_SIZE_LEGEND = 14
FONT_SIZE_ANNOTATION = 16
FONT_SIZE_SUBPLOT_TITLE = 20

FONT_SIZE_HEATMAP_TITLE = 36
FONT_SIZE_HEATMAP_AXIS_LABEL = 32
FONT_SIZE_HEATMAP_TICK_LABEL = 28
FONT_SIZE_HEATMAP_ANNOTATION = 28

FONT_SIZE_EARLY_EXIT_TITLE = 32
FONT_SIZE_EARLY_EXIT_AXIS_LABEL = 24
FONT_SIZE_EARLY_EXIT_TICK_LABEL = 20
FONT_SIZE_EARLY_EXIT_LEGEND = 18
FONT_SIZE_EARLY_EXIT_SUBTITLE = 24


def get_answers(metadata_df: pd.DataFrame, col: str = 'model_answer') -> tuple[dict, str]:
    """Get answer dict and key column from metadata."""
    key_col = 'question_hash' if 'question_hash' in metadata_df.columns else 'question_idx'
    answers = (
        metadata_df.set_index(key_col)[col]
        .astype(str).str.strip().str.upper()
        .to_dict()
    )
    return answers, key_col


def save_figure(fig: plt.Figure, filename: str, runs: "RunData | list[RunData]"):
    """Save figure to plots directory for each run."""
    if not isinstance(runs, list):
        runs = [runs]

    for run in runs:
        save_path = run.plots_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved plot to {save_path}')


def compute_probe_accuracy_heatmap(
    token_tables: dict,
    metadata_df: pd.DataFrame,
    num_bins: int = 100,
    layer_stride: int = 1,
) -> tuple[pd.DataFrame, Optional[float]]:
    """Compute probe accuracy by layer and relative position."""
    model_answers, _ = get_answers(metadata_df, 'model_answer')

    all_layers = set()
    for df_token in token_tables.values():
        probe_df = df_token[df_token['layer_idx'] >= 0]
        if not probe_df.empty:
            all_layers.update(probe_df['layer_idx'].unique())

    layers_sorted = sorted(all_layers)
    if layer_stride > 1:
        layers_to_keep = set(
            layers_sorted[i] for i in range(len(layers_sorted))
            if i % layer_stride == 0 or i == len(layers_sorted) - 1
        )
    else:
        layers_to_keep = set(layers_sorted)

    counts = {}
    assistant_rel_positions = []

    for question_hash, df_token in token_tables.items():
        target = model_answers.get(str(question_hash))
        if not target:
            continue

        probe_df = df_token[
            (df_token['layer_idx'] >= 0) &
            (df_token['layer_idx'].isin(layers_to_keep))
        ].copy()

        if probe_df.empty:
            continue

        probe_df['token_idx'] = pd.to_numeric(probe_df['token_idx'], errors='coerce')
        probe_df = probe_df.dropna(subset=['token_idx'])

        if probe_df.empty:
            continue

        seq_len = int(probe_df['token_idx'].max()) + 1
        if seq_len == 0:
            continue

        if 'token_text' in df_token.columns:
            for marker in ASSISTANT_MARKERS:
                assistant_rows = df_token[df_token['token_text'] == marker]
                if not assistant_rows.empty:
                    assistant_idx = int(assistant_rows['token_idx'].max())
                    full_seq_len = int(df_token['token_idx'].max()) + 1
                    if full_seq_len > 0:
                        assistant_rel_pos = (assistant_idx + 1) / full_seq_len
                        assistant_rel_positions.append(assistant_rel_pos)
                    break

        rel_position = (probe_df['token_idx'].values + 1) / seq_len
        rel_bin = np.minimum(
            (np.ceil(rel_position * num_bins).astype(int) - 1),
            num_bins - 1
        )

        probe_pred = probe_df['probe_pred'].astype(str).str.strip().str.upper().values
        correct = (probe_pred == target).astype(int)

        layers = probe_df['layer_idx'].values
        for i in range(len(layers)):
            key = (int(layers[i]), int(rel_bin[i]))
            if key not in counts:
                counts[key] = [0, 0]
            counts[key][0] += correct[i]
            counts[key][1] += 1

    layers = sorted({layer for layer, _ in counts.keys()})
    bins = sorted({rel_bin for _, rel_bin in counts.keys()})

    accuracy = pd.DataFrame(np.nan, index=layers, columns=bins)
    for (layer, rel_bin), (wins, total) in counts.items():
        if total > 0:
            accuracy.at[layer, rel_bin] = wins / total

    accuracy.index.name = 'layer_idx'

    avg_assistant_rel_pos = np.mean(assistant_rel_positions) if assistant_rel_positions else None

    return accuracy, avg_assistant_rel_pos


def plot_heatmap(
    accuracy: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    label_stride: int = 5,
    figsize: tuple = (18, 8),
    avg_assistant_rel_pos: Optional[float] = None,
) -> plt.Figure:
    """Internal function to plot a precomputed accuracy heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    plot_norm = TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=1.0)

    im = ax.imshow(
        np.ma.masked_invalid(accuracy.values),
        aspect='auto',
        origin='lower',
        cmap='RdBu',
        norm=plot_norm,
    )

    ax.set_xlabel('Relative Position (%)', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)
    ax.set_ylabel('Layer', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)
    ax.set_title(
        f'Probe Accuracy by Layer and Position, {model_name} on {dataset_name}',
        fontsize=FONT_SIZE_HEATMAP_TITLE,
        fontweight='bold',
        pad=60,
    )

    num_cols = accuracy.shape[1]
    xtick_positions = np.arange(0, num_cols + 1, max(1, num_cols // 10))
    xtick_labels = [str(int(i / num_cols * 100)) for i in xtick_positions]
    ax.set_xticks(xtick_positions - 0.5)  # Align to left edge of bins
    ax.set_xticklabels(xtick_labels, fontsize=FONT_SIZE_HEATMAP_TICK_LABEL)

    layer_indices = accuracy.index.tolist()
    num_layers = len(layer_indices)
    ytick_positions = [i for i in range(num_layers) if i % label_stride == 0]
    ytick_labels = [layer_indices[i] for i in ytick_positions]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=FONT_SIZE_HEATMAP_TICK_LABEL)

    if avg_assistant_rel_pos is not None:
        num_bins = num_cols
        avg_assistant_bin = int(np.minimum(
            np.ceil(avg_assistant_rel_pos * num_bins) - 1,
            num_bins - 1
        ))
        bin_values = list(accuracy.columns)
        if avg_assistant_bin in bin_values:
            avg_assistant_bin_idx = bin_values.index(avg_assistant_bin)
        else:
            avg_assistant_bin_idx = int(np.argmin(np.abs(np.array(bin_values) - avg_assistant_bin)))

        if 0 <= avg_assistant_bin_idx < num_cols:
            ax.axvline(
                x=avg_assistant_bin_idx + 0.5,
                color='black',
                linestyle='--',
                linewidth=2.5,
                alpha=0.9,
                zorder=10,
            )
            ax.text(
                avg_assistant_bin_idx + 1,
                0.5,  # Fixed position near bottom
                'Avg. start of reasoning',
                rotation=90,
                va='bottom',
                ha='left',
                fontsize=FONT_SIZE_HEATMAP_ANNOTATION,
                color='black',
                alpha=0.9,
            )

    # Create colorbar with evenly spaced ticks despite non-linear norm
    # Map data values through TwoSlopeNorm to get colormap positions, then create
    # a new linear colormap that preserves the colors but has even tick spacing
    n_colors = 256
    data_values = np.linspace(0.0, 1.0, n_colors)
    norm_positions = plot_norm(data_values)
    cbar_colors = cm.RdBu(norm_positions)
    cbar_cmap = ListedColormap(cbar_colors)

    # Create ScalarMappable with linear norm for even spacing
    cbar_norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=cbar_norm, cmap=cbar_cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label('Accuracy', rotation=0, labelpad=-50, y=1.12, fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_HEATMAP_TICK_LABEL - 4)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout()

    return fig


def plot_probe_accuracy_heatmap(
    run: "RunData",
    num_bins: int = 100,
    layer_stride: int = 5,
    label_stride: int = 10,
    figsize: tuple = (18, 8),
    save: bool = False,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot probe accuracy heatmap by layer and relative token position."""
    accuracy_strided, avg_assistant_rel_pos = compute_probe_accuracy_heatmap(
        run.token_tables, run.metadata_df,
        num_bins=num_bins,
        layer_stride=layer_stride,
    )

    accuracy_full, _ = compute_probe_accuracy_heatmap(
        run.token_tables, run.metadata_df,
        num_bins=num_bins,
        layer_stride=1,
    )

    fig_strided = plot_heatmap(
        accuracy_strided,
        run.model_name, run.dataset_name,
        label_stride=1,
        figsize=figsize,
        avg_assistant_rel_pos=avg_assistant_rel_pos,
    )

    fig_full = plot_heatmap(
        accuracy_full,
        run.model_name, run.dataset_name,
        label_stride=label_stride,
        figsize=figsize,
        avg_assistant_rel_pos=avg_assistant_rel_pos,
    )

    if save:
        save_figure(fig_strided, "probe_accuracy_heatmap_strided.pdf", run)
        save_figure(fig_full, "probe_accuracy_heatmap_full.pdf", run)

    return fig_strided, fig_full


def compute_agreement_by_position(
    step_level_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    probe_layer: int,
    num_bins: int = 100,
) -> pd.DataFrame:
    """Compute agreement between probe and forced answering by relative position."""
    model_answers, key_col = get_answers(metadata_df, 'model_answer')

    forced_df = step_level_df[step_level_df['layer_idx'] == -1][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()
    forced_df = forced_df.rename(columns={'probe_pred': 'forced_pred'})

    probe_df = step_level_df[step_level_df['layer_idx'] == probe_layer][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()

    merged = forced_df.merge(probe_df, on=[key_col, 'step_idx'], how='inner')

    if merged.empty:
        raise ValueError(f"No matching steps between forced answer and probe layer {probe_layer}")

    merged['target'] = merged[key_col].map(model_answers)
    merged = merged[merged['target'].notna()]

    merged['forced_pred_norm'] = merged['forced_pred'].astype(str).str.strip().str.upper()
    merged['probe_pred_norm'] = merged['probe_pred'].astype(str).str.strip().str.upper()

    merged['forced_correct'] = merged['forced_pred_norm'] == merged['target']
    merged['probe_correct'] = merged['probe_pred_norm'] == merged['target']

    step_range = merged.groupby(key_col)['step_idx'].agg(['min', 'max'])
    merged = merged.merge(step_range, on=key_col)
    merged['rel_position'] = np.where(
        merged['max'] > merged['min'],
        (merged['step_idx'] - merged['min']) / (merged['max'] - merged['min']),
        0.5  # single step questions get middle position
    )

    merged['bin'] = (merged['rel_position'] * num_bins).astype(int).clip(0, num_bins - 1)

    merged['category'] = np.select(
        [
            merged['forced_correct'] & merged['probe_correct'],
            ~merged['forced_correct'] & ~merged['probe_correct'],
            ~merged['forced_correct'] & merged['probe_correct'],
            merged['forced_correct'] & ~merged['probe_correct'],
        ],
        ['both_correct', 'both_incorrect', 'probe_only', 'forced_only'],
        default='other'
    )

    counts = merged.groupby(['bin', 'category']).size().unstack(fill_value=0)

    for cat in ['both_correct', 'both_incorrect', 'probe_only', 'forced_only']:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts[['both_correct', 'both_incorrect', 'probe_only', 'forced_only']]
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    return percentages.reset_index()


def plot_probe_forced_agreement(
    run: "RunData",
    probe_layer: Optional[int] = None,
    num_bins: int = 100,
    figsize: tuple = (14, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot agreement between probe and forced answering by relative position."""
    if probe_layer is None:
        probe_layer = run.best_layer

    data = compute_agreement_by_position(
        run.step_level_df, run.metadata_df, probe_layer, num_bins
    )

    fig, ax = plt.subplots(figsize=figsize)

    bins = data['bin'].values
    bottom = np.zeros(len(bins))

    categories = [
        ('both_correct', 'P=1, F=1'),
        ('both_incorrect', 'P=0, F=0'),
        ('probe_only', 'P=1, F=0'),
        ('forced_only', 'P=0, F=1'),
    ]

    for cat_key, cat_label in categories:
        values = data[cat_key].values
        ax.bar(
            bins,
            values,
            bottom=bottom,
            width=1.0,
            label=cat_label,
            color=AGREEMENT_COLORS[cat_key],
            edgecolor='none',
        )
        bottom += values

    ax.set_xlabel('Relative Position (%)', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)
    ax.set_ylabel('Agreement Rate', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)
    ax.set_title(
        f'Probe (L{probe_layer}) vs Forced Answer Agreement, {run.model_name} on {run.dataset_name}',
        fontsize=FONT_SIZE_HEATMAP_TITLE,
        fontweight='bold',
        pad=25,
    )

    xtick_positions = np.arange(0, num_bins + 1, num_bins // 10)
    xtick_labels = [str(int(x / num_bins * 100)) for x in xtick_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=FONT_SIZE_HEATMAP_TICK_LABEL)

    ax.set_xlim(-0.5, num_bins - 0.5)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=FONT_SIZE_HEATMAP_TICK_LABEL)

    ax.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        save_figure(fig, "probe_forced_agreement.pdf", run)

    return fig


def compute_method_accuracy_by_position(
    step_level_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    probe_layer: int,
    num_bins: int = 100,
) -> dict[str, pd.DataFrame]:
    """Compute accuracy by relative position for probe, forced, and CoT monitor methods."""
    model_answers, key_col = get_answers(metadata_df, 'model_answer')

    methods = {
        'probe': probe_layer,
        'forced': -1,
        'cot_monitor': -2,
    }

    results = {}

    for method_name, layer_idx in methods.items():
        method_df = step_level_df[step_level_df['layer_idx'] == layer_idx].copy()

        if method_df.empty:
            continue

        method_df['target'] = method_df[key_col].map(model_answers)
        method_df = method_df[method_df['target'].notna()]

        if method_df.empty:
            continue

        method_df['probe_pred_norm'] = method_df['probe_pred'].astype(str).str.strip().str.upper()

        method_df['is_correct'] = (method_df['probe_pred_norm'] == method_df['target']).astype(float)
        if method_name == 'cot_monitor':
            na_mask = (
                method_df['probe_pred'].isna() |
                method_df['probe_pred_norm'].isin(['N/A', 'NA', 'NAN'])
            )
            method_df.loc[na_mask, 'is_correct'] = 0.25

        step_range = method_df.groupby(key_col)['step_idx'].agg(['min', 'max'])
        method_df = method_df.merge(step_range, on=key_col)
        method_df['rel_position'] = np.where(
            method_df['max'] > method_df['min'],
            (method_df['step_idx'] - method_df['min']) / (method_df['max'] - method_df['min']),
            0.5
        )

        method_df['bin'] = (method_df['rel_position'] * num_bins).astype(int).clip(0, num_bins - 1)

        bin_stats = method_df.groupby('bin').agg(
            accuracy=('is_correct', 'mean'),
            count=('is_correct', 'count'),
        ).reset_index()

        results[method_name] = bin_stats

    return results


def plot_early_decoding_accuracy(
    runs: "RunData | list[RunData]",
    probe_layer: Optional[int] = None,
    num_bins: int = 100,
    figsize: tuple = (12, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot accuracy by position for probe, forced answer, and CoT monitor methods."""
    if not isinstance(runs, list):
        runs = [runs]

    if len(runs) > 2:
        raise ValueError("Maximum 2 runs supported for comparison")

    if probe_layer is None:
        probe_layer = runs[0].best_layer

    fig, ax = plt.subplots(figsize=figsize)

    method_labels = {
        'probe': f'Probe (Layer {probe_layer})',
        'forced': 'Forced Answer',
        'cot_monitor': 'CoT Monitor',
    }

    same_model = len(runs) > 1 and len(set(r.model_name for r in runs)) == 1

    for run_idx, run in enumerate(runs):
        method_data = compute_method_accuracy_by_position(
            run.step_level_df, run.metadata_df, probe_layer, num_bins
        )

        for method_name, data in method_data.items():
            if len(runs) > 1:
                label = f'{method_labels[method_name]} ({run.dataset_name})'
            else:
                label = method_labels[method_name]

            x = data['bin'].values / num_bins * 100

            color = METHOD_COLORS[method_name][run_idx]

            ax.plot(
                x,
                data['accuracy'].values,
                color=color,
                linestyle='-',
                linewidth=2,
                label=label,
                marker='o' if num_bins <= 20 else None,
                markersize=4,
            )

    ax.set_xlabel('Relative Position (%)', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)

    title_line1 = 'Early Decoding Accuracy (Best Layer Probe vs. Forced Answer vs. CoT Monitor)'
    if len(runs) == 1:
        title_line2 = f'{runs[0].model_name} on {runs[0].dataset_name}'
    elif same_model:
        dataset_names = ' & '.join(r.dataset_name for r in runs)
        title_line2 = f'{runs[0].model_name} on {dataset_names}'
    else:
        model_names = ' & '.join(r.model_name for r in runs)
        dataset_name = runs[0].dataset_name
        title_line2 = f'{model_names} on {dataset_name}'

    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=FONT_SIZE_TITLE, pad=15)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)

    if len(runs) > 1:
        ax.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, ncol=2)
    else:
        ax.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        save_figure(fig, "early_decoding_accuracy.pdf", runs)

    return fig


def compute_probe_forced_agreement_stats(
    step_level_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    probe_layer: int,
) -> dict:
    """Compute aggregated agreement stats between probe and forced answer across all steps."""
    model_answers, key_col = get_answers(metadata_df, 'model_answer')

    forced_df = step_level_df[step_level_df['layer_idx'] == -1][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()
    forced_df = forced_df.rename(columns={'probe_pred': 'forced_pred'})

    probe_df = step_level_df[step_level_df['layer_idx'] == probe_layer][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()

    merged = forced_df.merge(probe_df, on=[key_col, 'step_idx'], how='inner')

    if merged.empty:
        return {'error': f'No matching steps between forced answer and probe layer {probe_layer}'}

    merged['target'] = merged[key_col].map(model_answers)
    merged = merged[merged['target'].notna()]

    if merged.empty:
        return {'error': 'No valid targets found'}

    merged['forced_pred_norm'] = merged['forced_pred'].astype(str).str.strip().str.upper()
    merged['probe_pred_norm'] = merged['probe_pred'].astype(str).str.strip().str.upper()

    merged['forced_correct'] = merged['forced_pred_norm'] == merged['target']
    merged['probe_correct'] = merged['probe_pred_norm'] == merged['target']

    both_correct = (merged['forced_correct'] & merged['probe_correct']).sum()
    both_incorrect = (~merged['forced_correct'] & ~merged['probe_correct']).sum()
    probe_only = (~merged['forced_correct'] & merged['probe_correct']).sum()
    forced_only = (merged['forced_correct'] & ~merged['probe_correct']).sum()
    total = len(merged)

    return {
        'both_correct': int(both_correct),
        'both_incorrect': int(both_incorrect),
        'probe_only': int(probe_only),
        'forced_only': int(forced_only),
        'total': int(total),
        'both_correct_pct': both_correct / total * 100,
        'both_incorrect_pct': both_incorrect / total * 100,
        'probe_only_pct': probe_only / total * 100,
        'forced_only_pct': forced_only / total * 100,
    }


def compute_probe_logit_stability(
    run: "RunData",
    probe_layer: Optional[int] = None,
) -> dict:
    """Compute probe logit stability across consecutive reasoning steps."""
    if probe_layer is None:
        probe_layer = run.best_layer

    key_col = 'question_hash' if 'question_hash' in run.step_level_df.columns else 'question_idx'

    probe_df = run.step_level_df[run.step_level_df['layer_idx'] == probe_layer].copy()

    if probe_df.empty:
        return {'error': f'No data found for layer {probe_layer}'}

    if 'probe_output_parsed' in probe_df.columns:
        probe_df['logits'] = probe_df['probe_output_parsed']
    else:
        def parse_logits(s):
            if pd.isna(s) or s == '':
                return None
            try:
                s = str(s).strip('[]')
                return np.array([float(v.strip()) for v in s.split(',')])
            except (ValueError, AttributeError):
                return None

        probe_df['logits'] = probe_df['probe_output'].apply(parse_logits)

    probe_df = probe_df[probe_df['logits'].apply(lambda x: x is not None and len(x) == 4)]

    if probe_df.empty:
        return {'error': 'No valid logits found in probe_output'}

    per_question_msd = []
    total_step_pairs = 0

    for question_key, group in probe_df.groupby(key_col):
        group = group.sort_values('step_idx')

        if len(group) < 2:
            # Need at least 2 steps to compute consecutive differences
            continue

        logits_list = group['logits'].tolist()

        ssd_values = []
        for i in range(len(logits_list) - 1):
            p_curr = logits_list[i]
            p_next = logits_list[i + 1]
            ssd = np.sum((p_next - p_curr) ** 2)
            ssd_values.append(ssd)

        msd_q = np.mean(ssd_values)
        per_question_msd.append(msd_q)
        total_step_pairs += len(ssd_values)

    if not per_question_msd:
        return {'error': 'No questions with multiple steps found'}

    probe_logit_stability = np.mean(per_question_msd)

    return {
        'probe_logit_stability': float(probe_logit_stability),
        'num_questions': len(per_question_msd),
        'num_step_pairs': total_step_pairs,
        'per_question_msd': per_question_msd,
        'std_across_questions': float(np.std(per_question_msd)),
        'median_across_questions': float(np.median(per_question_msd)),
    }


def compute_max_gap_stats(
    step_level_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    probe_layer: int,
) -> dict:
    """Compute the average maximum gap between probe/forced vs CoT monitor per question."""
    correct_answers, key_col = get_answers(metadata_df, 'correct_answer')

    cot_monitor_df = step_level_df[step_level_df['layer_idx'] == -2][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()
    cot_monitor_df = cot_monitor_df.rename(columns={'probe_pred': 'cot_monitor_pred'})

    probe_df = step_level_df[step_level_df['layer_idx'] == probe_layer][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()

    forced_df = step_level_df[step_level_df['layer_idx'] == -1][
        [key_col, 'step_idx', 'probe_pred']
    ].copy()
    forced_df = forced_df.rename(columns={'probe_pred': 'forced_pred'})

    probe_merged = probe_df.merge(cot_monitor_df, on=[key_col, 'step_idx'], how='inner')

    forced_merged = forced_df.merge(cot_monitor_df, on=[key_col, 'step_idx'], how='inner')

    def compute_score(pred, correct_answer, is_cot_monitor=False):
        """Compute score: 1 if correct, 0 if incorrect, 0.25 if N/A (cot_monitor only)."""
        if pd.isna(pred):
            return 0.25 if is_cot_monitor else 0.0
        pred_norm = str(pred).strip().upper()
        if pred_norm in ('N/A', 'NA'):
            return 0.25 if is_cot_monitor else 0.0
        return 1.0 if pred_norm == correct_answer else 0.0

    def _max_gaps_for_method(merged_df, method_pred_col):
        """Compute per-question max gap between method and cot_monitor."""
        max_gaps = []
        for question_key, group in merged_df.groupby(key_col):
            correct = correct_answers.get(question_key)
            if not correct:
                continue
            max_gap = float('-inf')
            for _, row in group.iterrows():
                method_score = compute_score(row[method_pred_col], correct, is_cot_monitor=False)
                cot_monitor_score = compute_score(row['cot_monitor_pred'], correct, is_cot_monitor=True)
                max_gap = max(max_gap, method_score - cot_monitor_score)
            if max_gap > float('-inf'):
                max_gaps.append(max_gap)
        return max_gaps

    results = {}
    for label, merged_df, pred_col in [
        ('probe_vs_cot_monitor', probe_merged, 'probe_pred'),
        ('forced_vs_cot_monitor', forced_merged, 'forced_pred'),
    ]:
        gaps = _max_gaps_for_method(merged_df, pred_col)
        if gaps:
            results[label] = {
                'avg_max_gap': np.mean(gaps),
                'num_questions': len(gaps),
            }

    return results


def compute_average_slope(accuracy_series: pd.Series, num_bins: int = 20) -> float:
    """Compute average slope of accuracy curve using point-wise derivative."""
    if len(accuracy_series) > num_bins:
        original_bins = accuracy_series.index.values
        max_bin = original_bins.max()
        bin_size = (max_bin + 1) / num_bins

        rebinned = {}
        for i in range(num_bins):
            bin_start = int(i * bin_size)
            bin_end = int((i + 1) * bin_size)
            mask = (original_bins >= bin_start) & (original_bins < bin_end)
            if mask.any():
                rebinned[i] = accuracy_series.iloc[mask].mean()

        accuracy_series = pd.Series(rebinned)

    accuracy_series = accuracy_series.sort_index()

    if len(accuracy_series) < 2:
        return np.nan

    values = accuracy_series.values
    derivatives = np.diff(values)

    return float(np.mean(derivatives))


def compute_quadratic_slope(accuracy_series: pd.Series) -> tuple[float, tuple[float, float, float]]:
    """Compute average slope by fitting a quadratic function to the accuracy curve."""
    accuracy_series = accuracy_series.sort_index()

    if len(accuracy_series) < 3:
        return np.nan, (np.nan, np.nan, np.nan)

    x = accuracy_series.index.values
    x_normalized = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
    y = accuracy_series.values

    coeffs = np.polyfit(x_normalized, y, 2)
    a, b, c = coeffs

    # Average slope over [0, 1] for quadratic is a + b
    avg_slope = a + b

    return float(avg_slope), (float(a), float(b), float(c))


def compute_slope_comparison_stats(
    step_level_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    probe_layer: int,
    num_bins: int = 20,
) -> dict:
    """Compute slope comparison stats between probe/forced vs CoT monitor."""
    method_data = compute_method_accuracy_by_position(
        step_level_df, metadata_df, probe_layer, num_bins=num_bins
    )

    results = {}

    if 'cot_monitor' not in method_data:
        results['error'] = 'No CoT monitor data available'
        return results

    cot_monitor_acc = method_data['cot_monitor'].set_index('bin')['accuracy']

    cot_monitor_slope_pw = compute_average_slope(cot_monitor_acc, num_bins)
    cot_monitor_slope_quad, cot_monitor_coeffs = compute_quadratic_slope(cot_monitor_acc)

    results['cot_monitor_slope_pointwise'] = cot_monitor_slope_pw
    results['cot_monitor_slope_quadratic'] = cot_monitor_slope_quad
    results['cot_monitor_quadratic_coeffs'] = cot_monitor_coeffs

    if 'probe' in method_data:
        probe_acc = method_data['probe'].set_index('bin')['accuracy']
        probe_slope_pw = compute_average_slope(probe_acc, num_bins)
        probe_slope_quad, probe_coeffs = compute_quadratic_slope(probe_acc)

        results['probe_vs_cot_monitor'] = {
            'probe_slope_pointwise': probe_slope_pw,
            'probe_slope_quadratic': probe_slope_quad,
            'probe_quadratic_coeffs': probe_coeffs,
            'cot_monitor_slope_pointwise': cot_monitor_slope_pw,
            'cot_monitor_slope_quadratic': cot_monitor_slope_quad,
            'slope_difference_pointwise': probe_slope_pw - cot_monitor_slope_pw,
            'slope_difference_quadratic': probe_slope_quad - cot_monitor_slope_quad,
        }

    if 'forced' in method_data:
        forced_acc = method_data['forced'].set_index('bin')['accuracy']
        forced_slope_pw = compute_average_slope(forced_acc, num_bins)
        forced_slope_quad, forced_coeffs = compute_quadratic_slope(forced_acc)

        results['forced_vs_cot_monitor'] = {
            'forced_slope_pointwise': forced_slope_pw,
            'forced_slope_quadratic': forced_slope_quad,
            'forced_quadratic_coeffs': forced_coeffs,
            'cot_monitor_slope_pointwise': cot_monitor_slope_pw,
            'cot_monitor_slope_quadratic': cot_monitor_slope_quad,
            'slope_difference_pointwise': forced_slope_pw - cot_monitor_slope_pw,
            'slope_difference_quadratic': forced_slope_quad - cot_monitor_slope_quad,
        }

    return results


def compute_area_between_curves(
    run: "RunData",
    probe_layer: Optional[int] = None,
    num_bins: int = 100,
    save: bool = False,
) -> dict:
    """Compute area between accuracy curves for probe vs CoT monitor and forced vs CoT monitor."""
    if probe_layer is None:
        probe_layer = run.best_layer

    method_data = compute_method_accuracy_by_position(
        run.step_level_df, run.metadata_df, probe_layer, num_bins
    )

    results = {}

    if 'cot_monitor' not in method_data:
        results['error'] = 'No CoT monitor data available'
        return results

    cot_monitor_acc = method_data['cot_monitor'].set_index('bin')['accuracy']

    if 'probe' in method_data:
        probe_acc = method_data['probe'].set_index('bin')['accuracy']

        common_bins = probe_acc.index.intersection(cot_monitor_acc.index)
        probe_aligned = probe_acc.loc[common_bins]
        cot_monitor_aligned_probe = cot_monitor_acc.loc[common_bins]

        diff_probe = probe_aligned - cot_monitor_aligned_probe
        area_probe = diff_probe.sum() / num_bins * 100  # Scale to percentage-points × percentage
        mean_diff_probe = diff_probe.mean()

        results['probe_vs_cot_monitor'] = {
            'area': area_probe,
            'mean_accuracy_difference': mean_diff_probe,
            'num_bins_compared': len(common_bins),
            'probe_mean_accuracy': probe_aligned.mean(),
            'cot_monitor_mean_accuracy': cot_monitor_aligned_probe.mean(),
        }

    if 'forced' in method_data:
        forced_acc = method_data['forced'].set_index('bin')['accuracy']

        common_bins = forced_acc.index.intersection(cot_monitor_acc.index)
        forced_aligned = forced_acc.loc[common_bins]
        cot_monitor_aligned_forced = cot_monitor_acc.loc[common_bins]

        diff_forced = forced_aligned - cot_monitor_aligned_forced
        area_forced = diff_forced.sum() / num_bins * 100
        mean_diff_forced = diff_forced.mean()

        results['forced_vs_cot_monitor'] = {
            'area': area_forced,
            'mean_accuracy_difference': mean_diff_forced,
            'num_bins_compared': len(common_bins),
            'forced_mean_accuracy': forced_aligned.mean(),
            'cot_monitor_mean_accuracy': cot_monitor_aligned_forced.mean(),
        }

    agreement_stats = compute_probe_forced_agreement_stats(
        run.step_level_df, run.metadata_df, probe_layer
    )
    results['probe_vs_forced_agreement'] = agreement_stats

    max_gap_stats = compute_max_gap_stats(
        run.step_level_df, run.metadata_df, probe_layer
    )
    results['max_gap_stats'] = max_gap_stats

    logit_stability_stats = compute_probe_logit_stability(run, probe_layer)
    results['probe_logit_stability'] = logit_stability_stats

    slope_stats = compute_slope_comparison_stats(
        run.step_level_df, run.metadata_df, probe_layer, num_bins=20
    )
    results['slope_comparison'] = slope_stats

    if save:
        save_path = run.plots_dir / "stats.txt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            f.write(f"Run Statistics\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model: {run.model_name}\n")
            f.write(f"Dataset: {run.dataset_name}\n")
            f.write(f"Probe Layer: {probe_layer}\n")
            f.write(f"Number of Bins: {num_bins}\n\n")

            f.write(f"Probe vs Forced Answer Agreement (All Steps)\n")
            f.write(f"{'-'*50}\n")
            if 'error' not in agreement_stats:
                f.write(f"  Both correct:      {agreement_stats['both_correct']:>6} ({agreement_stats['both_correct_pct']:.1f}%)\n")
                f.write(f"  Both incorrect:    {agreement_stats['both_incorrect']:>6} ({agreement_stats['both_incorrect_pct']:.1f}%)\n")
                f.write(f"  Probe only correct:{agreement_stats['probe_only']:>6} ({agreement_stats['probe_only_pct']:.1f}%)\n")
                f.write(f"  Forced only correct:{agreement_stats['forced_only']:>5} ({agreement_stats['forced_only_pct']:.1f}%)\n")
                f.write(f"  Total steps:       {agreement_stats['total']:>6}\n")
            else:
                f.write(f"  Error: {agreement_stats['error']}\n")
            f.write(f"\n")

            f.write(f"Area Between Curves Analysis\n")
            f.write(f"{'-'*50}\n")

            if 'probe_vs_cot_monitor' in results:
                r = results['probe_vs_cot_monitor']
                f.write(f"\nProbe (Layer {probe_layer}) vs CoT Monitor\n")
                f.write(f"  Area between curves: {r['area']:.2f}\n")
                f.write(f"  Mean accuracy difference: {r['mean_accuracy_difference']:.4f}\n")
                f.write(f"  Probe mean accuracy: {r['probe_mean_accuracy']:.4f}\n")
                f.write(f"  CoT Monitor mean accuracy: {r['cot_monitor_mean_accuracy']:.4f}\n")
                f.write(f"  Bins compared: {r['num_bins_compared']}\n")
                f.write(f"  Interpretation: Probe is {'better' if r['mean_accuracy_difference'] > 0 else 'worse'} than CoT Monitor by {abs(r['mean_accuracy_difference'])*100:.2f} percentage points on average\n")

            if 'forced_vs_cot_monitor' in results:
                r = results['forced_vs_cot_monitor']
                f.write(f"\nForced Answer vs CoT Monitor\n")
                f.write(f"  Area between curves: {r['area']:.2f}\n")
                f.write(f"  Mean accuracy difference: {r['mean_accuracy_difference']:.4f}\n")
                f.write(f"  Forced mean accuracy: {r['forced_mean_accuracy']:.4f}\n")
                f.write(f"  CoT Monitor mean accuracy: {r['cot_monitor_mean_accuracy']:.4f}\n")
                f.write(f"  Bins compared: {r['num_bins_compared']}\n")
                f.write(f"  Interpretation: Forced is {'better' if r['mean_accuracy_difference'] > 0 else 'worse'} than CoT Monitor by {abs(r['mean_accuracy_difference'])*100:.2f} percentage points on average\n")

            f.write(f"\nMax Gap Analysis (N/A = 0.25 accuracy)\n")
            f.write(f"{'-'*50}\n")
            f.write(f"For each question, finds the step with the largest gap between\n")
            f.write(f"method and CoT monitor, then averages across all questions.\n")

            if 'max_gap_stats' in results and results['max_gap_stats']:
                gap_stats = results['max_gap_stats']
                if 'probe_vs_cot_monitor' in gap_stats:
                    r = gap_stats['probe_vs_cot_monitor']
                    f.write(f"\nProbe (Layer {probe_layer}) vs CoT Monitor\n")
                    f.write(f"  Average max gap: {r['avg_max_gap']:.4f}\n")
                    f.write(f"  Questions analyzed: {r['num_questions']}\n")

                if 'forced_vs_cot_monitor' in gap_stats:
                    r = gap_stats['forced_vs_cot_monitor']
                    f.write(f"\nForced Answer vs CoT Monitor\n")
                    f.write(f"  Average max gap: {r['avg_max_gap']:.4f}\n")
                    f.write(f"  Questions analyzed: {r['num_questions']}\n")

            f.write(f"\nProbe Logit Stability (Consecutive Steps)\n")
            f.write(f"{'-'*50}\n")
            f.write(f"Measures how smoothly probe predictions change during reasoning.\n")
            f.write(f"Lower values = smoother transitions; higher = more erratic.\n")

            if 'probe_logit_stability' in results and 'error' not in results['probe_logit_stability']:
                r = results['probe_logit_stability']
                f.write(f"\n  Mean squared difference (MSD): {r['probe_logit_stability']:.6f}\n")
                f.write(f"  Std across questions: {r['std_across_questions']:.6f}\n")
                f.write(f"  Median across questions: {r['median_across_questions']:.6f}\n")
                f.write(f"  Questions analyzed: {r['num_questions']}\n")
                f.write(f"  Total step pairs: {r['num_step_pairs']}\n")
            elif 'probe_logit_stability' in results:
                f.write(f"  Error: {results['probe_logit_stability']['error']}\n")

            f.write(f"\nSlope Comparison (5% bins, N/A = 0.25 for cot_monitor)\n")
            f.write(f"{'-'*50}\n")
            f.write(f"Two methods for computing average slope:\n")
            f.write(f"  1. Point-wise: mean of consecutive bin differences\n")
            f.write(f"  2. Quadratic fit: fits y = ax² + bx + c, avg slope = a + b\n")
            f.write(f"Positive slope = accuracy increases with position.\n")
            f.write(f"Slope difference = method slope - cot_monitor slope.\n")

            if 'slope_comparison' in results and 'error' not in results['slope_comparison']:
                slope_stats = results['slope_comparison']

                if 'probe_vs_cot_monitor' in slope_stats:
                    r = slope_stats['probe_vs_cot_monitor']
                    f.write(f"\nProbe (Layer {probe_layer}) vs CoT Monitor\n")
                    f.write(f"  Point-wise method:\n")
                    f.write(f"    Probe slope: {r['probe_slope_pointwise']:.6f}\n")
                    f.write(f"    CoT Monitor slope: {r['cot_monitor_slope_pointwise']:.6f}\n")
                    f.write(f"    Slope difference: {r['slope_difference_pointwise']:.6f}\n")
                    interpretation = "rises faster" if r['slope_difference_pointwise'] > 0 else "rises slower"
                    f.write(f"    Interpretation: Probe {interpretation} than CoT Monitor\n")
                    f.write(f"  Quadratic fit method:\n")
                    f.write(f"    Probe slope: {r['probe_slope_quadratic']:.6f}\n")
                    f.write(f"    CoT Monitor slope: {r['cot_monitor_slope_quadratic']:.6f}\n")
                    f.write(f"    Slope difference: {r['slope_difference_quadratic']:.6f}\n")
                    interpretation = "rises faster" if r['slope_difference_quadratic'] > 0 else "rises slower"
                    f.write(f"    Interpretation: Probe {interpretation} than CoT Monitor\n")
                    if r['probe_quadratic_coeffs'][0] is not np.nan:
                        a, b, c = r['probe_quadratic_coeffs']
                        f.write(f"    Probe fit: y = {a:.4f}x² + {b:.4f}x + {c:.4f}\n")

                if 'forced_vs_cot_monitor' in slope_stats:
                    r = slope_stats['forced_vs_cot_monitor']
                    f.write(f"\nForced Answer vs CoT Monitor\n")
                    f.write(f"  Point-wise method:\n")
                    f.write(f"    Forced slope: {r['forced_slope_pointwise']:.6f}\n")
                    f.write(f"    CoT Monitor slope: {r['cot_monitor_slope_pointwise']:.6f}\n")
                    f.write(f"    Slope difference: {r['slope_difference_pointwise']:.6f}\n")
                    interpretation = "rises faster" if r['slope_difference_pointwise'] > 0 else "rises slower"
                    f.write(f"    Interpretation: Forced {interpretation} than CoT Monitor\n")
                    f.write(f"  Quadratic fit method:\n")
                    f.write(f"    Forced slope: {r['forced_slope_quadratic']:.6f}\n")
                    f.write(f"    CoT Monitor slope: {r['cot_monitor_slope_quadratic']:.6f}\n")
                    f.write(f"    Slope difference: {r['slope_difference_quadratic']:.6f}\n")
                    interpretation = "rises faster" if r['slope_difference_quadratic'] > 0 else "rises slower"
                    f.write(f"    Interpretation: Forced {interpretation} than CoT Monitor\n")
                    if r['forced_quadratic_coeffs'][0] is not np.nan:
                        a, b, c = r['forced_quadratic_coeffs']
                        f.write(f"    Forced fit: y = {a:.4f}x² + {b:.4f}x + {c:.4f}\n")
            elif 'slope_comparison' in results:
                f.write(f"  Error: {results['slope_comparison'].get('error', 'Unknown error')}\n")

        print(f'Saved stats to {save_path}')

    return results


def require_parsed_probe_output(token_level_df: pd.DataFrame):
    """Check that probe_output_parsed column exists."""
    if 'probe_output_parsed' not in token_level_df.columns:
        raise ValueError(
            "probe_output_parsed column not found. "
            "Load data with parse_probe_output=True for calibration/ECE plots."
        )


def compute_calibration_bins(
    probs_list: list,
    targets: np.ndarray,
    num_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute calibration bins from probability lists and targets."""
    options = np.array(['A', 'B', 'C', 'D'])

    confidences = []
    corrects = []
    for i, probs in enumerate(probs_list):
        if probs is None or len(probs) != 4:
            continue
        for j, option in enumerate(options):
            confidences.append(probs[j])
            corrects.append(1 if option == targets[i] else 0)

    confidences = np.array(confidences)
    corrects = np.array(corrects)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    accuracy_by_bin = np.full(num_bins, np.nan)
    count_by_bin = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            accuracy_by_bin[i] = corrects[mask].mean()
            count_by_bin[i] = mask.sum()

    total = len(confidences)
    ece = 0.0
    for i in range(num_bins):
        if count_by_bin[i] > 0:
            avg_conf = confidences[bin_indices == i].mean()
            ece += (count_by_bin[i] / total) * abs(accuracy_by_bin[i] - avg_conf)

    return bin_centers, accuracy_by_bin, count_by_bin, ece


def plot_calibration(
    run: "RunData",
    probe_layer: Optional[int] = None,
    num_bins: int = 20,
    figsize: tuple = (10, 8),
    save: bool = False,
) -> plt.Figure:
    """Plot calibration curve (confidence vs accuracy) for probe predictions."""
    require_parsed_probe_output(run.token_level_df)

    if probe_layer is None:
        probe_layer = run.best_layer

    model_answers, key_col = get_answers(run.metadata_df, 'model_answer')

    probe_df = run.token_level_df[run.token_level_df['layer_idx'] == probe_layer].copy()
    probe_df = probe_df[probe_df['probe_output_parsed'].notna()]
    probe_df['target'] = probe_df[key_col].map(model_answers)
    probe_df = probe_df[probe_df['target'].notna()]

    if probe_df.empty:
        raise ValueError(f"No valid data for layer {probe_layer}")

    bin_centers, accuracy_by_bin, _, _ = compute_calibration_bins(
        probe_df['probe_output_parsed'].tolist(), probe_df['target'].values, num_bins
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bin_centers, accuracy_by_bin, marker='o', linestyle='-', linewidth=2,
            markersize=6, color=CALIBRATION_LINE_COLOR, label='Probe calibration')
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2,
            color=PERFECT_CALIBRATION_COLOR, label='Perfect calibration')

    ax.set_xlabel('Confidence', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)
    ax.set_title(
        f'Probe Calibration (Layer {probe_layer})\n{run.model_name} on {run.dataset_name}',
        fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
    ax.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()

    if save:
        save_figure(fig, "calibration.pdf", run)

    return fig


def plot_calibration_comparison(
    runs: "list[RunData]",
    num_bins: int = 20,
    figsize: tuple = (12, 10),
    save: bool = False,
) -> plt.Figure:
    """Plot calibration curves for multiple runs overlayed on the same plot."""
    RUN_COLORS = [
        '#1976D2',  # Blue
        '#D32F2F',  # Red
        '#388E3C',  # Green
        '#7B1FA2',  # Purple
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for run_idx, run in enumerate(runs):
        require_parsed_probe_output(run.token_level_df)

        probe_layer = run.best_layer
        model_answers, key_col = get_answers(run.metadata_df, 'model_answer')

        probe_df = run.token_level_df[run.token_level_df['layer_idx'] == probe_layer].copy()
        probe_df = probe_df[probe_df['probe_output_parsed'].notna()]
        probe_df['target'] = probe_df[key_col].map(model_answers)
        probe_df = probe_df[probe_df['target'].notna()]

        if probe_df.empty:
            print(f"Warning: No valid data for {run.model_name} on {run.dataset_name}")
            continue

        bin_centers, accuracy_by_bin, _, _ = compute_calibration_bins(
            probe_df['probe_output_parsed'].tolist(), probe_df['target'].values, num_bins
        )

        color = RUN_COLORS[run_idx % len(RUN_COLORS)]
        label = f'{run.dataset_name} (L{probe_layer})'
        ax.plot(bin_centers, accuracy_by_bin, marker='o', linestyle='-', linewidth=2,
                markersize=6, color=color, label=label)

    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='black', label='Perfect calibration')

    ax.set_xlabel('Confidence', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL, labelpad=10)

    if len(runs) > 0:
        model_names = list(set(r.model_name for r in runs))
        if len(model_names) == 1:
            title = f'{model_names[0]} Probe Calibration'
        else:
            title = 'Probe Calibration Comparison'
    else:
        title = 'Probe Calibration Comparison'

    ax.set_title(title, fontsize=FONT_SIZE_HEATMAP_TITLE, fontweight='bold', pad=25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=FONT_SIZE_HEATMAP_TICK_LABEL)
    ax.legend(loc='lower right', fontsize=FONT_SIZE_HEATMAP_TICK_LABEL)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()

    if save:
        save_figure(fig, "calibration_comparison.pdf", runs)

    return fig


def compute_ece_brier_by_position(
    run: "RunData",
    probe_layer: int,
    num_position_bins: int = 100,
    num_confidence_bins: int = 10,
) -> pd.DataFrame:
    """Compute ECE and Brier score by relative position."""
    require_parsed_probe_output(run.token_level_df)

    model_answers, key_col = get_answers(run.metadata_df, 'model_answer')

    probe_df = run.token_level_df[run.token_level_df['layer_idx'] == probe_layer].copy()
    probe_df = probe_df[probe_df['probe_output_parsed'].notna()]
    probe_df['target'] = probe_df[key_col].map(model_answers)
    probe_df = probe_df[probe_df['target'].notna()]

    if probe_df.empty:
        raise ValueError(f"No valid data for layer {probe_layer}")

    probe_df['token_idx'] = pd.to_numeric(probe_df['token_idx'], errors='coerce')
    seq_lengths = probe_df.groupby(key_col)['token_idx'].transform('max') + 1
    probe_df['rel_position'] = (probe_df['token_idx'] + 1) / seq_lengths
    probe_df['pos_bin'] = (probe_df['rel_position'] * num_position_bins).astype(int).clip(0, num_position_bins - 1)

    options = np.array(['A', 'B', 'C', 'D'])
    probs_array = np.array(probe_df['probe_output_parsed'].tolist())
    targets = probe_df['target'].values

    predictions = options[probs_array.argmax(axis=1)]
    max_confidences = probs_array.max(axis=1)
    correct = (predictions == targets).astype(int)

    target_one_hot = np.zeros_like(probs_array)
    for i, t in enumerate(targets):
        idx = np.where(options == t)[0]
        if len(idx) > 0:
            target_one_hot[i, idx[0]] = 1.0

    brier_scores = np.sum((probs_array - target_one_hot) ** 2, axis=1)

    probe_df['confidence'] = max_confidences
    probe_df['correct'] = correct
    probe_df['brier'] = brier_scores

    conf_bin_edges = np.linspace(0, 1, num_confidence_bins + 1)
    probe_df['conf_bin'] = np.digitize(probe_df['confidence'], conf_bin_edges) - 1
    probe_df['conf_bin'] = probe_df['conf_bin'].clip(0, num_confidence_bins - 1)

    results = []
    for pos_bin in range(num_position_bins):
        pos_df = probe_df[probe_df['pos_bin'] == pos_bin]
        if len(pos_df) == 0:
            continue

        brier = pos_df['brier'].mean()

        n_pos = len(pos_df)
        ece = 0.0
        for conf_bin in range(num_confidence_bins):
            conf_df = pos_df[pos_df['conf_bin'] == conf_bin]
            if len(conf_df) > 0:
                acc = conf_df['correct'].mean()
                avg_conf = conf_df['confidence'].mean()
                ece += (len(conf_df) / n_pos) * abs(acc - avg_conf)

        results.append({
            'pos_bin': pos_bin,
            'rel_position': (pos_bin + 0.5) / num_position_bins * 100,
            'ece': ece,
            'brier': brier,
            'count': len(pos_df),
        })

    return pd.DataFrame(results)


def plot_ece_brier_by_position(
    run: "RunData",
    probe_layer: Optional[int] = None,
    num_position_bins: int = 100,
    num_confidence_bins: int = 10,
    figsize: tuple = (12, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot ECE and Brier score by relative position (combined dual-axis plot)."""
    if probe_layer is None:
        probe_layer = run.best_layer

    results_df = compute_ece_brier_by_position(
        run, probe_layer, num_position_bins, num_confidence_bins
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(results_df['rel_position'], results_df['ece'],
             color=ECE_COLOR, linewidth=2, label='ECE')
    ax1.set_xlabel('Relative Position (%)', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)
    ax1.set_ylabel('ECE', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10, color=ECE_COLOR)
    ax1.tick_params(axis='y', labelcolor=ECE_COLOR, labelsize=FONT_SIZE_TICK_LABEL)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK_LABEL)

    ax2 = ax1.twinx()
    ax2.plot(results_df['rel_position'], results_df['brier'],
             color=BRIER_COLOR, linewidth=2, label='Brier Score')
    ax2.set_ylabel('Brier Score', fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10, color=BRIER_COLOR)
    ax2.tick_params(axis='y', labelcolor=BRIER_COLOR, labelsize=FONT_SIZE_TICK_LABEL)

    ax1.set_xlim(0, 100)
    ax1.set_title(
        f'ECE and Brier Score by Position (Layer {probe_layer})\n{run.model_name} on {run.dataset_name}',
        fontsize=FONT_SIZE_TITLE, pad=15
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=FONT_SIZE_LEGEND)

    ax1.grid(True, alpha=0.3)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, right=False)
    plt.tight_layout()

    if save:
        save_figure(fig, "ece_brier_by_position.pdf", run)

    return fig


EARLY_EXIT_ACCURACY_COLOR = '#0D47A1'   # Dark blue (Material Blue 900)
EARLY_EXIT_TOKENS_COLOR = '#1B5E20'     # Dark green (Material Green 900)


def compute_early_exit_metrics(
    run: "RunData",
    probe_layer: int,
) -> pd.DataFrame:
    """Compute early exit accuracy and tokens saved for each confidence threshold."""
    require_parsed_probe_output(run.token_level_df)

    correct_answers, key_col = get_answers(run.metadata_df, 'correct_answer')

    options = np.array(['A', 'B', 'C', 'D'])
    thresholds = np.arange(0.25, 1.01, 0.05)

    threshold_results = {t: {
        'correct': 0,
        'total': 0,
        'tokens_saved_pct': [],
    } for t in thresholds}

    for question_key, group in run.token_level_df[
        run.token_level_df['layer_idx'] == probe_layer
    ].groupby(key_col):

        target = correct_answers.get(question_key)
        group = group.sort_values('token_idx')
        group = group[group['probe_output_parsed'].notna()]

        if group.empty:
            continue

        max_token_idx = group['token_idx'].max()
        if max_token_idx == 0:
            continue

        token_data = []
        for _, row in group.iterrows():
            probs = row['probe_output_parsed']
            if probs is not None and len(probs) == 4:
                conf = max(probs)
                pred = options[np.argmax(probs)]
                token_data.append({
                    'token_idx': row['token_idx'],
                    'confidence': conf,
                    'prediction': pred,
                })

        if not token_data:
            continue

        token_data = sorted(token_data, key=lambda x: x['token_idx'])

        for threshold in thresholds:
            exit_token = None
            for td in token_data:
                if td['confidence'] >= threshold:
                    exit_token = td
                    break

            if exit_token is None:
                exit_token = token_data[-1]
                tokens_saved_pct = 0.0
            else:
                tokens_saved = max_token_idx - exit_token['token_idx']
                tokens_saved_pct = (tokens_saved / max_token_idx) * 100

            if target and target in options:
                threshold_results[threshold]['total'] += 1
                if exit_token['prediction'] == target:
                    threshold_results[threshold]['correct'] += 1

            threshold_results[threshold]['tokens_saved_pct'].append(tokens_saved_pct)

    results = []
    for threshold in thresholds:
        r = threshold_results[threshold]
        if r['total'] > 0 and r['tokens_saved_pct']:
            results.append({
                'threshold': threshold,
                'accuracy': r['correct'] / r['total'],
                'avg_tokens_saved_pct': np.mean(r['tokens_saved_pct']),
            })

    results_df = pd.DataFrame(results)

    model_answers = run.metadata_df['model_answer'].astype(str).str.strip().str.upper()
    correct_answers_series = run.metadata_df['correct_answer'].astype(str).str.strip().str.upper()
    baseline_accuracy = (model_answers == correct_answers_series).mean()
    results_df['baseline_accuracy'] = baseline_accuracy

    return results_df


def plot_early_exit_combined(
    run: "RunData",
    probe_layer: Optional[int] = None,
    figsize: tuple = (12, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot early exit accuracy and tokens saved on dual y-axes."""
    if probe_layer is None:
        probe_layer = run.best_layer

    results_df = compute_early_exit_metrics(run, probe_layer)

    if results_df.empty:
        raise ValueError(f"No valid data for layer {probe_layer}")

    baseline_accuracy = results_df['baseline_accuracy'].iloc[0]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(results_df['threshold'], results_df['accuracy'],
             color=EARLY_EXIT_ACCURACY_COLOR, linewidth=2, marker='o', markersize=5)
    ax1.axhline(y=baseline_accuracy, color='black', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Baseline Model Accuracy ({baseline_accuracy:.1%})')
    ax1.set_xlabel('Confidence Threshold', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL, labelpad=10)
    ax1.set_ylabel('Accuracy', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL, labelpad=10, color=EARLY_EXIT_ACCURACY_COLOR)
    ax1.tick_params(axis='y', labelcolor=EARLY_EXIT_ACCURACY_COLOR, labelsize=FONT_SIZE_EARLY_EXIT_TICK_LABEL)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_EARLY_EXIT_TICK_LABEL)
    ax1.set_xlim(0.25, 1.0)
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['avg_tokens_saved_pct'],
             color=EARLY_EXIT_TOKENS_COLOR, linewidth=2, marker='s', markersize=5)
    ax2.set_ylabel('Tokens Saved (%)', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL, labelpad=10, color=EARLY_EXIT_TOKENS_COLOR)
    ax2.tick_params(axis='y', labelcolor=EARLY_EXIT_TOKENS_COLOR, labelsize=FONT_SIZE_EARLY_EXIT_TICK_LABEL)
    ax2.set_ylim(0, 100)

    ax1.set_title(
        f'Early Exit: Accuracy vs Tokens Saved (Layer {probe_layer})\n{run.model_name} on {run.dataset_name}',
        fontsize=FONT_SIZE_EARLY_EXIT_TITLE, fontweight='bold', pad=15
    )

    ax1.legend(loc='lower center', fontsize=FONT_SIZE_EARLY_EXIT_LEGEND)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        save_figure(fig, "early_exit_combined.pdf", run)

    return fig


def plot_early_decoding_accuracy_sidebyside(
    runs: "list[RunData]",
    probe_layer: Optional[int] = None,
    num_bins: int = 100,
    figsize: tuple = (18, 7),
    save: bool = False,
) -> plt.Figure:
    """Plot accuracy by position for two runs side-by-side with shared legend."""
    if len(runs) != 2:
        raise ValueError("Side-by-side plot requires exactly 2 runs")

    if probe_layer is None:
        probe_layer = runs[0].best_layer

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    method_labels = {
        'probe': f'Probe (L{probe_layer})',
        'forced': 'Forced Answer',
        'cot_monitor': 'CoT Monitor',
    }

    legend_handles = []
    legend_labels = []

    for ax_idx, (ax, run) in enumerate(zip(axes, runs)):
        method_data = compute_method_accuracy_by_position(
            run.step_level_df, run.metadata_df, probe_layer, num_bins
        )

        for method_name, data in method_data.items():
            x = data['bin'].values / num_bins * 100

            color = METHOD_COLORS[method_name][0]

            line, = ax.plot(
                x,
                data['accuracy'].values,
                color=color,
                linestyle='-',
                linewidth=2,
                marker=None,
            )

            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(method_labels[method_name])

        chance_line = ax.axhline(y=0.25, color='black', linestyle=':', linewidth=1.5)

        if ax_idx == 0:
            legend_handles.append(chance_line)
            legend_labels.append('Chance Performance')

        ax.set_title(f'{run.dataset_name}', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL)

        ax.set_xlabel('Relative Position (%)', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=FONT_SIZE_HEATMAP_TICK_LABEL)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Accuracy', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL)

    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        ncol=4,
        fontsize=FONT_SIZE_HEATMAP_TICK_LABEL - 6,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        f'{runs[0].model_name} Early Decoding Accuracy by Method',
        fontsize=FONT_SIZE_HEATMAP_TITLE,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.88)

    if save:
        save_figure(fig, "early_decoding_accuracy_sidebyside.pdf", runs)

    return fig


def plot_early_exit_sidebyside(
    runs: "list[RunData]",
    probe_layer: Optional[int] = None,
    figsize: tuple = (14, 5),
    save: bool = False,
) -> plt.Figure:
    """Plot early exit accuracy and tokens saved side-by-side for two runs."""
    if len(runs) != 2:
        raise ValueError("Side-by-side plot requires exactly 2 runs")

    dataset_colors = ['#1976D2', '#D32F2F']  # Blue for first, Red for second

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    all_metrics = []
    for run in runs:
        layer = probe_layer if probe_layer is not None else run.best_layer
        all_metrics.append(compute_early_exit_metrics(run, layer))

    same_dataset = runs[0].dataset_name == runs[1].dataset_name

    ax_acc = axes[0]
    for run_idx, (run, metrics_df) in enumerate(zip(runs, all_metrics)):
        color = dataset_colors[run_idx]
        baseline = metrics_df['baseline_accuracy'].iloc[0]
        run_label = run.model_name if same_dataset else run.dataset_name
        ax_acc.plot(
            metrics_df['threshold'], metrics_df['accuracy'],
            color=color, linewidth=2, marker='o', markersize=5,
            label=f'{run_label}'
        )
        ax_acc.axhline(
            y=baseline, color=color, linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'{run_label} Baseline ({baseline:.1%})'
        )

    ax_acc.set_xlabel('Confidence Threshold', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL)
    ax_acc.set_ylabel('Accuracy at Exit', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL)
    ax_acc.set_title('Early Exit Accuracy', fontsize=FONT_SIZE_EARLY_EXIT_SUBTITLE, fontweight='bold')
    ax_acc.set_xlim(0.25, 1.0)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.tick_params(labelsize=FONT_SIZE_EARLY_EXIT_TICK_LABEL)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc='lower right', fontsize=FONT_SIZE_EARLY_EXIT_LEGEND)
    sns.despine(ax=ax_acc)

    ax_tok = axes[1]
    for run_idx, (run, metrics_df) in enumerate(zip(runs, all_metrics)):
        color = dataset_colors[run_idx]
        run_label = run.model_name if same_dataset else run.dataset_name
        ax_tok.plot(
            metrics_df['threshold'], metrics_df['avg_tokens_saved_pct'],
            color=color, linewidth=2, marker='o', markersize=5,
            label=f'{run_label}'
        )

    ax_tok.set_xlabel('Confidence Threshold', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL)
    ax_tok.set_ylabel('Average Tokens Saved (%)', fontsize=FONT_SIZE_EARLY_EXIT_AXIS_LABEL)
    ax_tok.set_title('Tokens Saved by Early Exit', fontsize=FONT_SIZE_EARLY_EXIT_SUBTITLE, fontweight='bold')
    ax_tok.set_xlim(0.25, 1.0)
    ax_tok.set_ylim(0, 100)
    ax_tok.tick_params(labelsize=FONT_SIZE_EARLY_EXIT_TICK_LABEL)
    ax_tok.grid(True, alpha=0.3)
    ax_tok.legend(loc='upper left', fontsize=FONT_SIZE_EARLY_EXIT_LEGEND)
    sns.despine(ax=ax_tok)

    suptitle = f'Early Exit Performance on {runs[0].dataset_name}' if same_dataset else f'Early Exit Performance for {runs[0].model_name}'
    fig.suptitle(
        suptitle,
        fontsize=FONT_SIZE_EARLY_EXIT_TITLE,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save:
        save_figure(fig, "early_exit_sidebyside.pdf", runs)

    return fig


def plot_early_exit_vertical(
    runs: "list[RunData]",
    probe_layer: Optional[int] = None,
    figsize: tuple = (12, 16),
    save: bool = False,
) -> plt.Figure:
    """Plot early exit accuracy and tokens saved vertically stacked for two runs."""
    if len(runs) != 2:
        raise ValueError("Vertical plot requires exactly 2 runs")

    require_parsed_probe_output(runs[0].token_level_df)
    require_parsed_probe_output(runs[1].token_level_df)

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    same_dataset = runs[0].dataset_name == runs[1].dataset_name
    legend_handles = []
    legend_labels = []

    for run_idx, (ax, run) in enumerate(zip(axes, runs)):
        layer = probe_layer if probe_layer is not None else run.best_layer

        results_df = compute_early_exit_metrics(run, layer)

        if results_df.empty:
            raise ValueError(f"No valid data for {run.dataset_name} layer {layer}")

        baseline_accuracy = results_df['baseline_accuracy'].iloc[0]

        line_acc, = ax.plot(
            results_df['threshold'], results_df['accuracy'] * 100,
            color=EARLY_EXIT_ACCURACY_COLOR, linewidth=3, marker='o', markersize=6
        )
        baseline_line = ax.axhline(
            y=baseline_accuracy * 100, color=EARLY_EXIT_ACCURACY_COLOR, linestyle='--',
            linewidth=2.5, alpha=0.7
        )

        ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL + 6, labelpad=10, color=EARLY_EXIT_ACCURACY_COLOR)
        ax.tick_params(axis='y', labelcolor=EARLY_EXIT_ACCURACY_COLOR, labelsize=FONT_SIZE_HEATMAP_TICK_LABEL + 6)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_HEATMAP_TICK_LABEL + 6)
        ax.set_xlim(0.25, 1.0)
        ax.set_ylim(0, 100)

        ax2 = ax.twinx()
        line_tok, = ax2.plot(
            results_df['threshold'], results_df['avg_tokens_saved_pct'],
            color=EARLY_EXIT_TOKENS_COLOR, linewidth=3, marker='s', markersize=6
        )
        ax2.set_ylabel('Tokens Saved (%)', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL + 6, labelpad=10, color=EARLY_EXIT_TOKENS_COLOR)
        ax2.tick_params(axis='y', labelcolor=EARLY_EXIT_TOKENS_COLOR, labelsize=FONT_SIZE_HEATMAP_TICK_LABEL + 6)
        ax2.set_ylim(0, 100)

        panel_label = f'{run.model_name} on {run.dataset_name}' if same_dataset else run.dataset_name
        ax.set_title(
            f'{panel_label}, Probe L{layer}',
            fontsize=FONT_SIZE_HEATMAP_TITLE, pad=15
        )

        ax.grid(True, alpha=0.3)

        if run_idx == len(runs) - 1:
            ax.set_xlabel('Confidence Threshold', fontsize=FONT_SIZE_HEATMAP_AXIS_LABEL + 6, labelpad=10)
        else:
            ax.set_xticklabels([])

        if run_idx == 0:
            legend_handles.extend([line_acc, line_tok, baseline_line])
            legend_labels.extend(['Accuracy (%)', 'Tokens Saved (%)', 'Baseline Model Accuracy'])

    fig.suptitle(
        f'{runs[0].model_name} Early Exit: Accuracy vs Tokens Saved',
        fontsize=FONT_SIZE_HEATMAP_TITLE,
        fontweight='bold',
        y=0.98,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        ncol=3,
        fontsize=FONT_SIZE_HEATMAP_TICK_LABEL + 6,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.14, right=0.88, hspace=0.30)

    # Pad bottom panel x-axis ticks down after layout to avoid overlap with right y-axis 0
    axes[-1].tick_params(axis='x', pad=18)

    if save:
        save_figure(fig, "early_exit_vertical.pdf", runs)

    return fig


def main():
    """Generate all plots for specified runs."""
    import argparse
    import matplotlib
    matplotlib.use('Agg')

    from .data_loading import load_results

    parser = argparse.ArgumentParser(description='Generate early decoder plots')
    parser.add_argument('--results_dir', type=str, required=True, nargs='+',
                        help='Path(s) to results directory')
    parser.add_argument('--model_name', type=str, required=True, nargs='+',
                        help='Model name(s) for plot titles')
    parser.add_argument('--dataset_name', type=str, required=True, nargs='+',
                        help='Dataset name(s) for plot titles')
    parser.add_argument('--plots_dir', type=str, nargs='+',
                        help='Custom path(s) to save plots (default: plots/{run_name}/ adjacent to results)')
    parser.add_argument('--parse_probe_output', action='store_true',
                        help='Parse probe output for calibration/ECE plots (slower)')

    args = parser.parse_args()

    if len(args.results_dir) != len(args.model_name):
        parser.error("Must provide same number of results_dir and model_name arguments")
    if len(args.results_dir) != len(args.dataset_name):
        parser.error("Must provide same number of results_dir and dataset_name arguments")
    if args.plots_dir and len(args.plots_dir) != len(args.results_dir):
        parser.error("Must provide same number of plots_dir and results_dir arguments")

    plots_dirs = args.plots_dir if args.plots_dir else [None] * len(args.results_dir)

    runs = []
    for results_dir, model_name, dataset_name, plots_dir in zip(
        args.results_dir, args.model_name, args.dataset_name, plots_dirs
    ):
        print(f"\nLoading {model_name} from {results_dir}...")
        run = load_results(
            results_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            parse_probe_output=args.parse_probe_output,
            plots_dir=plots_dir,
        )
        print(f"  {run}")
        print(f"  Best layer: {run.best_layer} (accuracy: {run.best_accuracy:.3f})")
        runs.append(run)

    for run in runs:
        print(f"\nGenerating plots for {run.run_name}...")

        print("  - Probe accuracy heatmap")
        plot_probe_accuracy_heatmap(run, save=True)
        plt.close('all')

        print("  - Probe vs forced agreement")
        try:
            plot_probe_forced_agreement(run, save=True)
        except ValueError as e:
            print(f"    Skipped: {e}")
        plt.close('all')

        print("  - Early decoding accuracy")
        try:
            plot_early_decoding_accuracy(run, save=True)
        except ValueError as e:
            print(f"    Skipped: {e}")
        plt.close('all')

        print("  - Area between curves")
        try:
            compute_area_between_curves(run, save=True)
        except ValueError as e:
            print(f"    Skipped: {e}")

        if args.parse_probe_output:
            print("  - Calibration")
            plot_calibration(run, save=True)
            plt.close('all')

            print("  - ECE/Brier by position")
            plot_ece_brier_by_position(run, save=True)
            plt.close('all')

            print("  - Early exit combined (accuracy + tokens saved)")
            plot_early_exit_combined(run, save=True)
            plt.close('all')

    if len(runs) > 1:
        print(f"\nGenerating comparison plots...")
        print("  - Early decoding accuracy (comparison)")
        fig = plot_early_decoding_accuracy(runs, save=False)
        save_figure(fig, "early_decoding_accuracy_comparison.pdf", runs)
        plt.close('all')

        if len(runs) == 2:
            print("  - Early decoding accuracy (side-by-side)")
            fig_sidebyside = plot_early_decoding_accuracy_sidebyside(runs, save=False)
            save_figure(fig_sidebyside, "early_decoding_accuracy_sidebyside.pdf", runs)
            plt.close('all')

            if args.parse_probe_output:
                print("  - Early exit (side-by-side)")
                fig_exit_sidebyside = plot_early_exit_sidebyside(runs, save=False)
                save_figure(fig_exit_sidebyside, "early_exit_sidebyside.pdf", runs)
                plt.close('all')

                print("  - Early exit (vertical)")
                fig_exit_vertical = plot_early_exit_vertical(runs, save=False)
                save_figure(fig_exit_vertical, "early_exit_vertical.pdf", runs)
                plt.close('all')

                print("  - Calibration comparison")
                fig_cal_comparison = plot_calibration_comparison(runs, save=False)
                save_figure(fig_cal_comparison, "calibration_comparison.pdf", runs)
                plt.close('all')

    print(f"\nDone! Plots saved to:")
    for run in runs:
        print(f"  {run.plots_dir}")


if __name__ == "__main__":
    main()
