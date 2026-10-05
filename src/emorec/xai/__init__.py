"""Explainability methods (integrated gradients, gate analysis, figures; CC-8).

`ig` — per-token / per-frame Integrated Gradients (captum imported lazily; `[xai]` extra).
`gate` — modality-gate collection and dominance summary.
`viz` — token heatmap, audio timeline, gate distribution, per-class F1 figures.
Attention attribution, occlusion, faithfulness and dominance (T-19+) are not implemented.
"""

from emorec.xai.gate import collect_gates, gate_summary
from emorec.xai.ig import (
    attribute_audio,
    attribute_text,
    attribution_path,
    cur_span,
    frames_to_times,
    save_attribution,
)
from emorec.xai.viz import (
    plot_audio_timeline,
    plot_gate_distribution,
    plot_per_class_comparison,
    plot_token_heatmap,
)

__all__ = [
    "attribute_audio",
    "attribute_text",
    "attribution_path",
    "collect_gates",
    "cur_span",
    "frames_to_times",
    "gate_summary",
    "plot_audio_timeline",
    "plot_gate_distribution",
    "plot_per_class_comparison",
    "plot_token_heatmap",
    "save_attribution",
]
