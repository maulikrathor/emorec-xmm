"""Inference-time modality ablation of a trained xattn checkpoint (CC-9).

Usage:
    python scripts/run_ablation.py --ckpt artifacts/runs/<run>/best.pt --config configs/xattn.yaml \\
        [--manifest artifacts/manifests/dev.jsonl] [--seed 42] [--device auto] [--allow-test]

Evaluates the SAME trained weights with the modality removed or corrupted at
inference (`none`, `zero_audio`, `zero_text`, `shuffle_audio`; see
`emorec.xai.ablation`) and reports whether audio is load-bearing despite a
text-saturated gate. Writes, under `cfg.data.report_dir` (artifacts/reports):
`ablation.md`, `ablation.csv`, `figures/ablation.png` and one
`ablation_predictions/predictions_<mode>.jsonl` per mode (pair them with
`emorec.training.metrics.align_predictions` + `mcnemar` for further tests).

Like scripts/evaluate.py, refuses a manifest whose filename contains "test"
without --allow-test (A6).
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import MeldDataset, make_collate_fn, make_dataloader
from emorec.data.labels import EMOTION_NAMES
from emorec.models.registry import build_model
from emorec.training.metrics import bootstrap_ci, mcnemar
from emorec.training.trainer import save_predictions
from emorec.utils.config import load_experiment_config
from emorec.utils.seed import seed_everything
from emorec.xai.ablation import MODES, ablate_modality, ablation_frame
from emorec.xai.viz import plot_ablation_bars


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/dev.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required if the manifest filename contains 'test' (A6 discipline).",
    )
    return parser.parse_args(argv)


def _interpretation(df, stats: dict, n: int, g_audio: float) -> str:
    """A short, factual paragraph built ONLY from the measured numbers (no invented values)."""
    row = df.set_index("mode")

    def sig(mode: str) -> bool:
        return stats[mode]["p_value"] < 0.05

    def line(mode: str, what: str) -> str:
        s = stats[mode]
        verdict = "statistically significant (p < 0.05)" if sig(mode) else "not statistically significant (p >= 0.05)"
        return (
            f"{what} changes weighted-F1 by {row.loc[mode, 'delta_wF1']:+.4f} and macro-F1 by "
            f"{row.loc[mode, 'delta_macro']:+.4f}; McNemar on correctness, none vs {mode}: n01={s['n01']}, "
            f"n10={s['n10']}, p={s['p_value']:.3g}, {verdict}."
        )

    parts = [
        (f"On {n} utterances the unmodified model scores weighted-F1 {row.loc['none', 'weighted_f1']:.4f} "
        f"(macro-F1 {row.loc['none', 'macro_f1']:.4f}) with mean g_audio = {g_audio:.4f}."),
        line("zero_audio", "Replacing the audio with zeros"),
        line("zero_text", "Replacing the text with pad embeddings"),
        line("shuffle_audio", "Pairing each utterance with another utterance's audio"),
    ]
    if sig("zero_audio") and row.loc["zero_audio", "delta_wF1"] < 0:
        parts.append(
            "Removing audio significantly lowers accuracy, so the audio stream is used by the model even though the "
            f"gate weights the pooled audio summary at {g_audio:.4f}."
        )
        if sig("shuffle_audio") and row.loc["shuffle_audio", "delta_wF1"] < 0:
            parts.append(
                "Mismatched audio also hurts significantly, which is consistent with the model using "
                "utterance-specific audio content and not merely the presence of an audio stream."
            )
        elif not sig("shuffle_audio"):
            parts.append(
                "Mismatched audio does not significantly change accuracy while zeroed audio does, which is "
                "consistent with audio presence or statistics acting as a bias term rather than "
                "utterance-specific signal."
            )
    else:
        parts.append(
            "Removing audio does not significantly change accuracy in this run; no claim that audio is "
            "load-bearing is supported by this measurement."
        )
    parts.append(
        "These are inference-time interventions on a model trained with real audio and text: zeroed inputs are "
        "out-of-distribution, so the size of a drop measures reliance on that stream under the intervention."
    )
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if "test" in Path(args.manifest).name.lower() and not args.allow_test:
        print(
            f"REFUSING to run ablation on {args.manifest!r} (filename contains 'test') without --allow-test (A6).",
            file=sys.stderr,
        )
        return 2

    seed_everything(args.seed)
    cfg = load_experiment_config(args.config)
    if cfg.model.name != "xattn":
        print(
            f"REFUSING: modality ablation needs the fusion model (xattn); config model is {cfg.model.name!r}.",
            file=sys.stderr,
        )
        return 2
    device = _resolve_device(args.device)

    if cfg.train.use_cache:
        ds = CachedAudioMeldDataset(args.manifest, cfg, use_cache=True)
        collate_fn = make_collate_fn_cached(ds.tokenizer)
    else:
        ds = MeldDataset(args.manifest, cfg, load_audio=True)
        collate_fn = make_collate_fn(ds.tokenizer)
    loader = make_dataloader(ds, cfg, shuffle=False, seed=args.seed, collate_fn=collate_fn)

    model = build_model(cfg, len(ds.tokenizer), [ds.cur_token_id, ds.end_cur_token_id]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    print(f"running {len(MODES)} ablation modes over {len(ds)} utterances ...")
    results = {}
    for mode in MODES:
        results[mode] = ablate_modality(model, loader, mode, device=device, seed=args.seed)
        print(f"  {mode:<14} wF1={results[mode]['weighted_f1']:.4f} macro={results[mode]['macro_f1']:.4f}")
    df = ablation_frame(results)

    # Statistics (ED12): bootstrap CI on weighted-F1 per mode; exact McNemar none vs each other mode.
    cis = {m: bootstrap_ci(r["y_true"], r["y_pred"], "weighted_f1", n=1000, seed=args.seed) for m, r in results.items()}
    stats = {
        m: mcnemar(results["none"]["y_true"], results["none"]["y_pred"], results[m]["y_pred"])
        for m in MODES
        if m != "none"
    }

    report_dir = Path(cfg.data.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "ablation.csv", index=False)
    plot_ablation_bars(df, report_dir / "figures" / "ablation.png")
    for mode, r in results.items():
        save_predictions(
            report_dir / "ablation_predictions" / f"predictions_{mode}.jsonl",
            r["utt_ids"],
            r["y_true"],
            r["y_pred"],
            r["y_prob"],
        )

    n = len(results["none"]["y_true"])
    g_audio = results["none"]["mean_g_audio"]
    lines = [
        "# Modality ablation",
        "",
        f"- checkpoint: `{args.ckpt}`  |  config: `{args.config}`  |  manifest: `{args.manifest}`  |  n = {n}",
        f"- seed: {args.seed}  |  generated (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}",
        ("- Same trained weights in every row; only the inference-time input is changed "
        "(`emorec.xai.ablation`). `none` is the control. The gate is never rescaled."),
        "",
        "## Results",
        "",
        "| mode | accuracy | macro-F1 | weighted-F1 [95% CI] | ΔwF1 | Δmacro | mean g_audio |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        ci = cis[r["mode"]]
        lines.append(
            f"| {r['mode']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} "
            f"[{ci['lo']:.4f}, {ci['hi']:.4f}] | {r['delta_wF1']:+.4f} | {r['delta_macro']:+.4f} | "
            f"{r['mean_g_audio']:.4f} |"
        )
    lines += [
        "",
        "## Paired McNemar (correctness), `none` vs each ablation",
        "",
        "n01 = control right & ablated wrong; n10 = control wrong & ablated right.",
        "",
        "| ablation | n01 | n10 | discordant | p (exact, two-sided) |",
        "|---|---|---|---|---|",
    ]
    for m, s in stats.items():
        lines.append(f"| {m} | {s['n01']} | {s['n10']} | {s['n_discordant']} | {s['p_value']:.3g} |")
    lines += ["", "## Per-class F1", "", "| mode | " + " | ".join(EMOTION_NAMES) + " |", "|---|" + "---|" * len(EMOTION_NAMES)]
    for _, r in df.iterrows():
        lines.append(f"| {r['mode']} | " + " | ".join(f"{r[f'f1_{e}']:.4f}" for e in EMOTION_NAMES) + " |")
    if results["shuffle_audio"]["n_unshuffled"]:
        lines += ["", f"_{results['shuffle_audio']['n_unshuffled']} example(s) sat alone in a batch and kept their own audio._"]
    lines += ["", "## Interpretation", "", _interpretation(df, stats, n, g_audio), "", "![ablation](figures/ablation.png)", ""]
    (report_dir / "ablation.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwritten: {report_dir / 'ablation.md'}, {report_dir / 'ablation.csv'}, {report_dir / 'figures' / 'ablation.png'}")
    print(f"predictions: {report_dir / 'ablation_predictions'}/predictions_<mode>.jsonl")
    for m, s in stats.items():
        print(f"McNemar none vs {m}: n01={s['n01']} n10={s['n10']} p={s['p_value']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
