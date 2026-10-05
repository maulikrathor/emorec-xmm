"""Gate analysis + Integrated Gradients examples for one trained checkpoint (CC-8).

Usage:
    python scripts/run_xai.py --ckpt artifacts/runs/<run>/best.pt --config configs/xattn.yaml \\
        [--manifest artifacts/manifests/dev.jsonl] [--n-examples 20] [--utt-ids a,b,c] \\
        [--n-steps 32] [--device auto] [--out-dir artifacts/attributions/ig] [--seed 42] \\
        [--allow-test]

STAGE A runs the model over the WHOLE manifest and reports the modality gate
exactly as the model produced it (`gates.csv`, `gate_analysis.md`, a
distribution figure) — a saturated gate is a finding, not something to fix
(docs/BACKLOG.md J-08). STAGE B runs Integrated Gradients (text, then audio,
one modality per pass) on a selected set of examples and writes one `.npz`,
a token heatmap and an audio timeline per example, plus `xai_examples.md`.

A text-only checkpoint (no gate) runs STAGE B text attribution only, and says
so. A checkpoint with no IG entry point (audio-only, concat) is refused.
Like scripts/evaluate.py, refuses a manifest whose filename contains "test"
without --allow-test (A6).
"""

import argparse
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import MeldDataset, make_collate_fn, make_dataloader
from emorec.data.labels import EMOTION_NAMES, ID_TO_EMOTION, NUM_CLASSES
from emorec.models.registry import build_model
from emorec.utils.config import load_experiment_config
from emorec.utils.seed import seed_everything
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
    AUDIO_BIN_SECONDS,
    bin_attribution,
    plot_audio_timeline,
    plot_gate_distribution,
    plot_token_heatmap,
)

_STRUCTURAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>", "<cur>", "</cur>"}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/dev.jsonl")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--utt-ids", default=None, help="Comma-separated utt_ids; overrides the automatic selection.")
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="artifacts/attributions/ig")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required if the manifest filename contains 'test' (A6 discipline).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------


def select_examples(
    df: pd.DataFrame, n_examples: int, seed: int, forced: list[str] | None = None, has_gate: bool = True
) -> list[tuple[str, list[str]]]:
    """Choose `(utt_id, [reasons])` pairs for STAGE B, deterministically for a given `seed`.

    `forced`: exactly those utt_ids (each must exist in `df`). Otherwise: the 2
    most audio-dominant and 2 most text-dominant (only if the model has a gate),
    2 randomly chosen misclassified, then a class-stratified fill — first one
    example for every true class not yet covered, then round-robin over the 7
    classes — up to `n_examples` (the deliberate picks come first and count
    toward it; if they alone exceed it the list is truncated in that order).
    """
    known = set(df["utt_id"])
    if forced:
        unknown = [u for u in forced if u not in known]
        if unknown:
            raise SystemExit(f"--utt-ids not found in the manifest: {unknown}")
        return [(u, ["requested"]) for u in forced]

    rng = random.Random(seed)
    reasons: dict[str, list[str]] = {}

    def add(utt_id: str, why: str) -> None:
        reasons.setdefault(utt_id, []).append(why)

    if has_gate:
        ranked = df.sort_values(["g_audio", "utt_id"], ascending=[False, True])
        for u in ranked["utt_id"].head(2):
            add(u, "audio-dominant")
        for u in ranked["utt_id"].tail(2).iloc[::-1]:
            add(u, "text-dominant")

    wrong = sorted(u for u in df.loc[~df["correct"], "utt_id"] if u not in reasons)
    for u in rng.sample(wrong, min(2, len(wrong))):
        add(u, "misclassified")

    label_of = dict(zip(df["utt_id"], df["label"], strict=True))
    pools = {
        c: sorted(u for u in df.loc[df["label"] == c, "utt_id"] if u not in reasons) for c in range(NUM_CLASSES)
    }
    for pool in pools.values():
        rng.shuffle(pool)

    def take(c: int) -> bool:
        if pools[c] and len(reasons) < n_examples:
            add(pools[c].pop(), f"stratified:{ID_TO_EMOTION[c]}")
            return True
        return False

    covered = {label_of[u] for u in reasons}
    for c in range(NUM_CLASSES):
        if c not in covered:
            take(c)
    progress = True
    while progress and len(reasons) < n_examples:
        took = [take(c) for c in range(NUM_CLASSES)]  # list, not a generator: every class gets its turn
        progress = any(took)

    return list(reasons.items())[:n_examples]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    return "nan" if math.isnan(x) else f"{x:.{digits}f}"


def _sci(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.2e}"


def _gate_markdown(summary: dict, df: pd.DataFrame, meta: dict) -> str:
    o, q = summary["overall"], summary["quantiles"]
    lines = [
        "# Gate analysis",
        "",
        f"- checkpoint: `{meta['ckpt']}`  |  config: `{meta['config']}`  |  model: `{meta['model']}`",
        f"- manifest: `{meta['manifest']}`  |  n = {summary['n']} utterances  |  seed: {meta['seed']}",
        f"- generated (UTC): {meta['date']}",
        "",
        (
            "The gate `g = (g_text, g_audio)` is reported exactly as the model produced it; nothing is rescaled "
            "or re-centred (docs/BACKLOG.md J-08)."
        ),
        "",
        "## g_audio overall",
        "",
        "| mean | median | std | p05 | p25 | p50 | p75 | p95 |",
        "|---|---|---|---|---|---|---|---|",
        (
            f"| {_fmt(o['mean'])} | {_fmt(o['median'])} | {_fmt(o['std'])} | {_fmt(q['p05'])} | {_fmt(q['p25'])} "
            f"| {_fmt(q['p50'])} | {_fmt(q['p75'])} | {_fmt(q['p95'])} |"
        ),
        "",
        "## Dominance",
        "",
        f"- audio-dominant (g_audio > 0.5): {summary['frac_audio_dominant']:.4f}",
        f"- saturated toward audio (g_audio > 0.95): {summary['frac_saturated_audio']:.4f}",
        f"- saturated toward text (g_audio < 0.05): {summary['frac_saturated_text']:.4f}",
        "",
        "## g_audio per TRUE class",
        "",
        "| class | n | mean g_audio | std |",
        "|---|---|---|---|",
    ]
    for name in EMOTION_NAMES:
        c = summary["per_class"][name]
        lines.append(f"| {name} | {c['n']} | {_fmt(c['mean'])} | {_fmt(c['std'])} |")
    lines += ["", "## g_audio by prediction correctness", "", "| | n | mean g_audio |", "|---|---|---|"]
    for key in ("correct", "incorrect"):
        c = summary["by_correctness"][key]
        lines.append(f"| {key} | {c['n']} | {_fmt(c['mean'])} |")

    indexed = df.set_index("utt_id")
    for title, key in (("Most audio-dominant", "top_audio_dominant"), ("Most text-dominant", "top_text_dominant")):
        lines += ["", f"## {title} (top {len(summary[key])})", "", "| utt_id | g_audio | true | pred |", "|---|---|---|---|"]
        for u in summary[key]:
            r = indexed.loc[u]
            lines.append(f"| {u} | {_fmt(float(r['g_audio']))} | {ID_TO_EMOTION[int(r['label'])]} | {ID_TO_EMOTION[int(r['pred'])]} |")
    return "\n".join(lines) + "\n"


def _top_tokens(tokens: list[str], attr: np.ndarray, span: tuple[int, int], k: int = 5) -> list[tuple[str, float, str]]:
    cands = [i for i, t in enumerate(tokens) if t not in _STRUCTURAL_TOKENS]
    cands.sort(key=lambda i: -abs(float(attr[i])))
    return [
        (tokens[i].replace("Ġ", ""), float(attr[i]), "cur" if span[0] <= i < span[1] else "ctx") for i in cands[:k]
    ]


def _top_windows(attr: np.ndarray, times: np.ndarray, k: int = 3) -> list[tuple[float, float, float]]:
    starts, means = bin_attribution(attr, times)
    order = np.argsort(-np.abs(means))[:k]
    return [(float(starts[i]), float(starts[i] + AUDIO_BIN_SECONDS), float(means[i])) for i in order]


def _example_markdown(i: int, ex: dict) -> str:
    probs = ", ".join(f"{EMOTION_NAMES[c]} {ex['probs'][c]:.3f}" for c in np.argsort(-ex["probs"]))
    lines = [
        f"## {i}. `{ex['utt_id']}`  ({', '.join(ex['reasons'])})",
        "",
        f"- speaker/text: **{ex['speaker']}**: {ex['text']}",
        (
            f"- true: **{ID_TO_EMOTION[ex['label']]}**  |  predicted: **{ID_TO_EMOTION[ex['pred']]}** "
            f"({'correct' if ex['label'] == ex['pred'] else 'incorrect'})"
        ),
        f"- probs: {probs}",
    ]
    if ex["gate"] is None:
        lines.append("- gate: n/a (no gate in this model)")
    else:
        lines.append(f"- gate: g_text={ex['gate'][0]:.4f}, g_audio={ex['gate'][1]:.4f}")
    lines.append(
        f"- IG target: predicted class `{ID_TO_EMOTION[ex['pred']]}`; convergence delta text={_sci(ex['delta_text'])} "
        f"(n_steps={ex['steps_text']})"
        + (
            f", audio={_sci(ex['delta_audio'])} (n_steps={ex['steps_audio']})"
            if ex["top_windows"] is not None
            else ", audio: not attributed"
        )
    )
    lines.append(
        "- top-5 text tokens by |score| (structural tokens excluded; cur = current utterance, ctx = context): "
        + ", ".join(f"`{t}` {s:+.4f} [{where}]" for t, s, where in ex["top_tokens"])
    )
    if ex["top_windows"] is not None:
        lines.append(
            f"- top-3 audio windows by |mean score| ({AUDIO_BIN_SECONDS} s bins): "
            + ", ".join(f"[{a:.1f}-{b:.1f} s] {s:+.5f}" for a, b, s in ex["top_windows"])
        )
    lines.append(f"- figures: [tokens]({ex['fig_tokens']})" + (f", [audio]({ex['fig_audio']})" if ex["fig_audio"] else ""))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if "test" in Path(args.manifest).name.lower() and not args.allow_test:
        print(
            f"REFUSING to run XAI on {args.manifest!r} (filename contains 'test') without --allow-test (A6).",
            file=sys.stderr,
        )
        return 2

    seed_everything(args.seed)
    cfg = load_experiment_config(args.config)
    device = _resolve_device(args.device)
    text_only = cfg.model.name == "text_only"
    if cfg.model.name not in ("xattn", "text_only"):
        print(
            f"REFUSING: IG needs a text-embedding entry point; model {cfg.model.name!r} has none "
            "(supported: xattn via forward_from_embeds, text_only).",
            file=sys.stderr,
        )
        return 2

    # Same dataset/model construction as scripts/evaluate.py.
    if text_only:
        ds = MeldDataset(args.manifest, cfg, load_audio=False)
        collate_fn = make_collate_fn(ds.tokenizer)
    elif cfg.train.use_cache:
        ds = CachedAudioMeldDataset(args.manifest, cfg, use_cache=True)
        collate_fn = make_collate_fn_cached(ds.tokenizer)
    else:
        ds = MeldDataset(args.manifest, cfg, load_audio=True)
        collate_fn = make_collate_fn(ds.tokenizer)
    loader = make_dataloader(ds, cfg, shuffle=False, seed=args.seed, collate_fn=collate_fn)

    model = build_model(cfg, len(ds.tokenizer), [ds.cur_token_id, ds.end_cur_token_id]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    report_dir = Path(cfg.data.report_dir)
    fig_dir = report_dir / "figures"
    out_dir = Path(args.out_dir)
    meta = {
        "ckpt": args.ckpt,
        "config": args.config,
        "model": cfg.model.name,
        "manifest": args.manifest,
        "seed": args.seed,
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ"),
    }

    # ---------------- STAGE A: gates over the whole manifest ----------------
    print(f"STAGE A: collecting gates over {len(ds)} utterances ...")
    df = collect_gates(model, loader, device=device)
    summary = gate_summary(df)
    has_gate = bool(summary["has_gate"])
    written: list[str] = []
    if has_gate:
        report_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_dir / "gates.csv", index=False)
        (report_dir / "gate_analysis.md").write_text(_gate_markdown(summary, df, meta), encoding="utf-8")
        plot_gate_distribution(df, fig_dir / "gate_distribution.png")
        written += [str(report_dir / "gates.csv"), str(report_dir / "gate_analysis.md"), str(fig_dir / "gate_distribution.png")]
        o = summary["overall"]
        print(
            f"  g_audio mean={o['mean']:.4f} median={o['median']:.4f} std={o['std']:.4f}; "
            f"audio-dominant={summary['frac_audio_dominant']:.4f}; saturated(audio)={summary['frac_saturated_audio']:.4f} "
            f"saturated(text)={summary['frac_saturated_text']:.4f}"
        )
    else:
        print(
            f"NOTE: model {cfg.model.name!r} has NO GATE — skipping gate analysis. "
            "Running STAGE B text attribution ONLY."
        )

    # ---------------- STAGE B: IG on selected examples ----------------
    forced = [u.strip() for u in args.utt_ids.split(",") if u.strip()] if args.utt_ids else None
    chosen = select_examples(df, args.n_examples, args.seed, forced=forced, has_gate=has_gate)
    print(f"STAGE B: {len(chosen)} examples selected:")
    for u, why in chosen:
        print(f"  {u}: {', '.join(why)}")

    index_of = {u: i for i, u in enumerate(df["utt_id"])}
    examples: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for utt_id, why in chosen:
        item = ds[index_of[utt_id]]
        if item["meta"]["utt_id"] != utt_id:
            raise RuntimeError(f"dataset order mismatch: expected {utt_id}, got {item['meta']['utt_id']}")
        batch = collate_fn([item])
        text_ids = batch.text_ids[0]
        try:
            span = cur_span(text_ids, ds.cur_token_id, ds.end_cur_token_id)
        except ValueError as exc:
            skipped.append((utt_id, str(exc)))
            print(f"  SKIPPED {utt_id}: {exc}")
            continue

        t = attribute_text(model, batch, n_steps=args.n_steps, device=device)
        a = None if text_only else attribute_audio(model, batch, n_steps=args.n_steps, device=device)

        tokens = ds.tokenizer.convert_ids_to_tokens(text_ids.tolist())
        text_attr = t["attr"][0]
        pred = int(np.argmax(t["probs"][0]))
        label = int(batch.label[0])
        row = df.iloc[index_of[utt_id]]
        gate = (float(row["g_text"]), float(row["g_audio"])) if has_gate else None

        if a is not None:
            audio_attr = a["attr"][0]
            frame_times = frames_to_times(len(audio_attr))
            delta = (float(t["delta"][0]), float(a["delta"][0]))
            steps = (int(t["n_steps"][0]), int(a["n_steps"][0]))
        else:
            audio_attr = np.zeros(0, dtype=np.float32)
            frame_times = np.zeros(0)
            delta = (float(t["delta"][0]), float("nan"))
            steps = (int(t["n_steps"][0]), 0)

        save_attribution(
            attribution_path(out_dir, utt_id),
            {
                "text_attr": text_attr,
                "text_tokens": tokens,
                "cur_span": span,
                "audio_attr": audio_attr,
                "frame_times": frame_times,
                "pred": pred,
                "label": label,
                "probs": t["probs"][0],
                "gate": gate,
                "convergence_delta": delta,
                "ig_n_steps": steps,
            },
        )

        stem = utt_id.replace("/", "_")
        plot_token_heatmap(tokens, text_attr, span, ID_TO_EMOTION[pred], ID_TO_EMOTION[label], fig_dir / f"{stem}_tokens.png")
        fig_audio = None
        if a is not None:
            plot_audio_timeline(audio_attr, frame_times, ID_TO_EMOTION[pred], fig_dir / f"{stem}_audio.png")
            fig_audio = f"figures/{stem}_audio.png"

        examples.append(
            {
                "utt_id": utt_id,
                "reasons": why,
                "speaker": item["meta"]["speaker"],
                "text": item["meta"]["text"],
                "label": label,
                "pred": pred,
                "probs": t["probs"][0],
                "gate": gate,
                "delta_text": delta[0],
                "delta_audio": delta[1],
                "steps_text": steps[0],
                "steps_audio": steps[1],
                "top_tokens": _top_tokens(tokens, text_attr, span),
                "top_windows": _top_windows(audio_attr, frame_times) if a is not None else None,
                "fig_tokens": f"figures/{stem}_tokens.png",
                "fig_audio": fig_audio,
            }
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# IG examples",
        "",
        f"- checkpoint: `{args.ckpt}`  |  model: `{cfg.model.name}`  |  manifest: `{args.manifest}`  |  seed: {args.seed}",
        f"- generated (UTC): {meta['date']}  |  n_steps: {args.n_steps} (re-run at 64 if |delta|/|sum(attr)| > 0.05)",
        (
            "- IG target is the PREDICTED class's logit. Text baseline: pad embeddings; audio baseline: zeros, "
            "attributed at the pre-`proj_a` frame output (documented deviation from ARCHITECTURE.md's "
            "post-`proj_a` point)."
        ),
        (
            "- Text scores are raw signed sums over the embedding dimension; audio scores are raw sums over the "
            "hidden dimension. Positive = pushes the predicted-class logit up relative to the baseline."
        ),
        "",
    ]
    if text_only:
        md += ["_Text-only model: no gate and no audio attribution._", ""]
    for n, ex in enumerate(examples, 1):
        md.append(_example_markdown(n, ex))
    if skipped:
        md += ["## Skipped", ""] + [f"- `{u}`: {why}" for u, why in skipped] + [""]
    (report_dir / "xai_examples.md").write_text("\n".join(md), encoding="utf-8")
    written.append(str(report_dir / "xai_examples.md"))

    # ---------------- summary ----------------
    deltas = [abs(ex["delta_text"]) for ex in examples] + [
        abs(ex["delta_audio"]) for ex in examples if ex["top_windows"] is not None
    ]
    reran = sum(ex["steps_text"] == 64 for ex in examples) + sum(
        ex["steps_audio"] == 64 for ex in examples if ex["top_windows"] is not None
    )
    print("\n=== run_xai summary ===")
    print(f"examples attributed: {len(examples)} (skipped: {len(skipped)})")
    print(f"mean |convergence delta|: {np.mean(deltas):.3e}" if deltas else "mean |convergence delta|: n/a")
    print(f"IG passes that needed n_steps=64: {reran}")
    print(f"attributions (.npz): {out_dir}")
    print(f"figures: {fig_dir}")
    for w in written:
        print(f"written: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
