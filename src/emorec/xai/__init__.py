"""Explainability methods (integrated gradients, gate analysis, ablation, faithfulness, figures; CC-8, CC-9).

`ig` — per-token / per-frame Integrated Gradients (captum imported lazily; `[xai]` extra).
`gate` — modality-gate collection and dominance summary.
`ablation` — inference-time modality ablation (zero / shuffle) on a trained model.
`faithfulness` — comprehensiveness / sufficiency curves, random control, randomisation sanity check.
`viz` — token heatmap, audio timeline, gate distribution, per-class F1, faithfulness and ablation figures.
Attention attribution, occlusion and dominance (T-19+) are not implemented.
"""

from emorec.xai.ablation import ablate_modality, run_ablation_suite
from emorec.xai.faithfulness import (
    comprehensiveness,
    faithfulness_auc,
    random_baseline_curves,
    sanity_check_randomization,
    sufficiency,
)
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
    plot_ablation_bars,
    plot_audio_timeline,
    plot_faithfulness_curves,
    plot_gate_distribution,
    plot_per_class_comparison,
    plot_token_heatmap,
)

__all__ = [
    "ablate_modality",
    "attribute_audio",
    "attribute_text",
    "attribution_path",
    "collect_gates",
    "comprehensiveness",
    "cur_span",
    "faithfulness_auc",
    "frames_to_times",
    "gate_summary",
    "plot_ablation_bars",
    "plot_audio_timeline",
    "plot_faithfulness_curves",
    "plot_gate_distribution",
    "plot_per_class_comparison",
    "plot_token_heatmap",
    "random_baseline_curves",
    "run_ablation_suite",
    "sanity_check_randomization",
    "save_attribution",
    "sufficiency",
]
