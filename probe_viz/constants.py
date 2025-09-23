"""Constants used across the probe visualization app."""

from __future__ import annotations

from typing import Dict, List

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Default textual labels for probe classes depending on output dimensionality.
DEFAULT_BINARY_LABELS: Dict[int, str] = {0: "Incorrect", 1: "Correct"}

# Default qualitative colors. We cycle when more classes are requested than
# available hues in any single palette.
COLOR_PALETTES: List[List[str]] = [
    ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"],
    ["#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
]


def label_for_choice(index: int) -> str:
    """Map a zero-based choice index to an uppercase alphabetical label."""
    if 0 <= index < len(CHOICE_LABELS):
        return CHOICE_LABELS[index]
    return f"Choice {index}"


def build_default_class_labels(num_classes: int) -> Dict[int, str]:
    """Return sensible default labels for the provided number of classes."""
    if num_classes == 2:
        return dict(DEFAULT_BINARY_LABELS)
    labels = {}
    for idx in range(num_classes):
        labels[idx] = label_for_choice(idx)
    return labels


def cycle_palette(num_classes: int) -> List[str]:
    """Produce a list of colors of length ``num_classes``."""
    colors: List[str] = []
    palette_idx = 0
    while len(colors) < num_classes:
        palette = COLOR_PALETTES[palette_idx % len(COLOR_PALETTES)]
        for color in palette:
            colors.append(color)
            if len(colors) == num_classes:
                break
        palette_idx += 1
    return colors
