"""Faithfulness evaluation of IG attributions on a trained xattn checkpoint (CC-9; docs/EVALUATION.md section 5).

Usage:
    python scripts/run_faithfulness.py --ckpt artifacts/runs/<run>/best.pt --config configs/xattn.yaml \\
        [--manifest artifacts/manifests/dev.jsonl] [--n-examples 100] [--n-steps 64] [--seed 42] \\
        [--device auto] [--allow-test]

Takes a class-stratified, seeded sample of `--n-examples` utterances, computes
IG once per example per modality (text, audio), then comprehensiveness and
sufficiency curves for IG AND for a random-attribution control, for both
modalities. A parameter-randomisation sanity check (classifier head) runs on a
20-example subset. This is the slowest script in the repo; it prints progress.
Writes, under `cfg.data.report_dir` (artifacts/reports): `faithfulness.md`
(AUC tables, sanity result, figure), `faithfulness.csv` (curve summary),
`faithfulness_per_example.csv` and `figures/faithfulness_curves.png`.

Deletion operators match the IG baselines by construction (pad embedding for
text, zero pre-`proj_a` frame for audio), which mildly favours IG
(docs/KNOWN_LIMITATIONS.md 14). Like scripts/evaluate.py, refuses a manifest
whose filename contains "test" without --allow-test (A6).
"""

import argparse
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import MeldDataset, make_collate_fn, stratified_subset_indices
from emorec.models.registry import build_model
from emorec.utils.config import load_experiment_config
from emorec.utils.seed import seed_everything
from emorec.xai.faithfulness import (
    BROKEN_SPEARMAN_THRESHOLD,
    DEFAULT_FRACTIONS,
    combine_curves,
    comprehensiveness,
    faithfulness_auc,
    mean_ci,
    per_example_auc,
    random_baseline_curves,
    sanity_check_randomization,
    sufficiency,
)
from emorec.xai.ig import attribute_audio, attribute_text
from emorec.xai.viz import plot_faithfulness_curves

N_SANITY = 20
N_RANDOM_DRAWS = 5
MODALITIES = ("text", "audio")
KINDS = ("comprehensiveness", "sufficiency")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/dev.jsonl")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required if the manifest filename contains 'test' (A6 discipline).",
    )
    return parser.parse_args(argv)


def _curve_with_ci(curve: dict, seed: int) -> dict:
    """`curve` plus per-fraction bootstrap `lo`/`hi` over examples (for plotting and the CSV)."""
    cols = [mean_ci(curve["per_example"][:, j], n=1000, seed=seed) for j in range(curve["per_example"].shape[1])]
    return {**curve, "lo": np.array([c["lo"] for c in cols]), "hi": np.array([c["hi"] for c in cols])}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if "test" in Path(args.manifest).name.lower() and not args.allow_test:
        print(
            f"REFUSING to run faithfulness on {args.manifest!r} (filename contains 'test') without --allow-test (A6).",
            file=sys.stderr,
        )
        return 2

    seed_everything(args.seed)
    cfg = load_experiment_config(args.config)
    if cfg.model.name != "xattn":
        print(
            f"REFUSING: faithfulness needs both modalities' IG (xattn); config model is {cfg.model.name!r}.",
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

    model = build_model(cfg, len(ds.tokenizer), [ds.cur_token_id, ds.end_cur_token_id]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    n_examples = min(args.n_examples, len(ds))
    indices = stratified_subset_indices(ds, n_examples, args.seed)
    print(f"faithfulness: {len(indices)} stratified examples (seed {args.seed}), IG n_steps={args.n_steps}")

    # ---------------- per-example IG + curves ----------------
    store: dict = {(m, who, kind): [] for m in MODALITIES for who in ("ig", "random") for kind in KINDS}
    ig_attr: dict[int, dict] = {}
    utt_ids: list[str] = []
    deltas: list[float] = []
    reran = 0
    t0 = time.time()
    for n, idx in enumerate(indices, 1):
        item = ds[idx]
        batch = collate_fn([item])
        utt_ids.append(item["meta"]["utt_id"])

        ig = {"text": attribute_text(model, batch, n_steps=args.n_steps, device=device),
              "audio": attribute_audio(model, batch, n_steps=args.n_steps, device=device)}  # fmt: skip
        for m in MODALITIES:
            deltas.append(abs(float(ig[m]["delta"][0])))
            reran += int(ig[m]["n_steps"][0] > args.n_steps)
        ig_attr[idx] = {m: ig[m]["attr"] for m in MODALITIES}

        for m in MODALITIES:
            attr = ig[m]["attr"]
            store[(m, "ig", "comprehensiveness")].append(comprehensiveness(model, batch, attr, m))
            store[(m, "ig", "sufficiency")].append(sufficiency(model, batch, attr, m))
            rand = random_baseline_curves(model, batch, m, seed=args.seed + idx, n_draws=N_RANDOM_DRAWS)
            for kind in KINDS:
                store[(m, "random", kind)].append(rand[kind])

        elapsed = time.time() - t0
        eta = elapsed / n * (len(indices) - n)
        print(f"  [{n}/{len(indices)}] {item['meta']['utt_id']}  elapsed {elapsed:6.0f}s  eta {eta:6.0f}s", flush=True)

    curves = {key: combine_curves(v) for key, v in store.items()}

    # ---------------- randomisation sanity check ----------------
    sanity_idx = random.Random(args.seed).sample(list(indices), min(N_SANITY, len(indices)))
    print(f"randomisation sanity check on {len(sanity_idx)} examples ...", flush=True)
    sanity = {m: [] for m in MODALITIES}
    sanity_abs = {m: [] for m in MODALITIES}
    for k, idx in enumerate(sanity_idx, 1):
        batch = collate_fn([ds[idx]])
        res = sanity_check_randomization(model, batch, ig_attr[idx], seed=args.seed, n_steps=args.n_steps)
        for m in MODALITIES:
            sanity[m] += res["modalities"][m]["spearman"]
            sanity_abs[m] += res["modalities"][m]["spearman_abs"]
        print(f"  [{k}/{len(sanity_idx)}] done", flush=True)

    # ---------------- tables ----------------
    summary_rows, per_example_rows, auc_rows = [], [], []
    for m in MODALITIES:
        for kind in KINDS:
            c_ig, c_rand = curves[(m, "ig", kind)], curves[(m, "random", kind)]
            auc_ig, auc_rand = per_example_auc(c_ig), per_example_auc(c_rand)
            ci_ig, ci_rand = mean_ci(auc_ig, seed=args.seed), mean_ci(auc_rand, seed=args.seed)
            diff = mean_ci(auc_ig - auc_rand, seed=args.seed)  # paired by example
            better = diff["lo"] > 0 if kind == "comprehensiveness" else diff["hi"] < 0
            auc_rows.append(
                {"modality": m, "kind": kind, "auc_ig": faithfulness_auc(c_ig), "ig_lo": ci_ig["lo"], "ig_hi": ci_ig["hi"],
                 "auc_random": faithfulness_auc(c_rand), "rand_lo": ci_rand["lo"], "rand_hi": ci_rand["hi"],
                 "diff": diff["mean"], "diff_lo": diff["lo"], "diff_hi": diff["hi"], "ig_beats_random": bool(better)}
            )  # fmt: skip
            for who, c in (("ig", c_ig), ("random", c_rand)):
                cc = _curve_with_ci(c, args.seed)
                for j, f in enumerate(c["fractions"]):
                    summary_rows.append(
                        {"modality": m, "method": who, "kind": kind, "fraction": float(f), "mean": float(cc["mean"][j]),
                         "lo": float(cc["lo"][j]), "hi": float(cc["hi"][j]), "n": int(c["per_example"].shape[0])}
                    )  # fmt: skip
                for i, uid in enumerate(utt_ids):
                    for j, f in enumerate(c["fractions"]):
                        per_example_rows.append(
                            {"utt_id": uid, "modality": m, "method": who, "kind": kind, "fraction": float(f),
                             "drop": float(c["per_example"][i, j])}
                        )  # fmt: skip
    auc_df = pd.DataFrame(auc_rows)

    report_dir = Path(cfg.data.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(report_dir / "faithfulness.csv", index=False)
    pd.DataFrame(per_example_rows).to_csv(report_dir / "faithfulness_per_example.csv", index=False)

    plot_faithfulness_curves(
        {
            f"IG ({m})": {
                "comprehensiveness": _curve_with_ci(curves[(m, "ig", "comprehensiveness")], args.seed),
                "sufficiency": _curve_with_ci(curves[(m, "ig", "sufficiency")], args.seed),
                "random": {k: curves[(m, "random", k)] for k in KINDS},
            }
            for m in MODALITIES
        },
        report_dir / "figures" / "faithfulness_curves.png",
    )

    # ---------------- report ----------------
    fr = ", ".join(f"{f:g}" for f in DEFAULT_FRACTIONS)
    lines = [
        "# Faithfulness of IG attributions",
        "",
        f"- checkpoint: `{args.ckpt}`  |  config: `{args.config}`  |  manifest: `{args.manifest}`",
        (f"- {len(indices)} class-stratified examples (seed {args.seed})  |  IG n_steps={args.n_steps}  |  "
        f"generated (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}"),
        (f"- fractions: {fr}. Curves are the mean DROP in predicted-class probability; AUC is the unnormalised "
        "trapezoid area over those fractions."),
        ("- comprehensiveness: DELETE the top-k% attributed positions (higher = more faithful). sufficiency: RETAIN "
        "only the top-k% (the drop from the full input; lower = more faithful)."),
        ("- Deletion operators match the IG baselines by construction (pad embedding for text, zero pre-`proj_a` "
        "frame for audio), which mildly favours IG (docs/KNOWN_LIMITATIONS.md 14). Positions are ranked by signed "
        "attribution; all real tokens (including structural ones) and real frames are eligible."),
        f"- Random control: mean of {N_RANDOM_DRAWS} seeded random attributions per example.",
        (f"- IG convergence: mean |delta| = {np.mean(deltas):.3e}; {reran} of {len(deltas)} IG passes were re-run at 64 steps "
        "(re-runs only happen when --n-steps < 64)."),
        "",
        "## AUC (mean over examples, 95% bootstrap CI)",
        "",
        "| modality | curve | IG AUC [CI] | random AUC [CI] | IG − random [CI] | IG better than random? |",
        "|---|---|---|---|---|---|",
    ]
    for r in auc_rows:
        lines.append(
            f"| {r['modality']} | {r['kind']} | {r['auc_ig']:.4f} [{r['ig_lo']:.4f}, {r['ig_hi']:.4f}] | "
            f"{r['auc_random']:.4f} [{r['rand_lo']:.4f}, {r['rand_hi']:.4f}] | {r['diff']:+.4f} "
            f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] | {'yes' if r['ig_beats_random'] else 'no'} |"
        )
    lines += [
        "",
        ("`IG better than random` is `yes` when the paired (per-example) CI of IG − random excludes 0 in the "
        "faithful direction (positive for comprehensiveness, negative for sufficiency); otherwise the result is "
        "reported as no significant difference from chance."),
        "",
        "## Parameter-randomisation sanity check (Adebayo-style, classifier head)",
        "",
        (f"Re-initialised `model.mlp` on a copy (the real model is never modified), IG recomputed for the same "
        f"target, Spearman correlation with the original attributions per example ({len(sanity_idx)} examples). "
        f"A method is reported BROKEN for this model if the mean signed Spearman stays >= {BROKEN_SPEARMAN_THRESHOLD}."),
        "",
        "| modality | n | mean Spearman (signed) | mean Spearman (magnitudes) | verdict |",
        "|---|---|---|---|---|",
    ]
    for m in MODALITIES:
        vals = np.array(sanity[m], dtype=float)
        absv = np.array(sanity_abs[m], dtype=float)
        mean_s = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
        mean_a = float(np.nanmean(absv)) if not np.all(np.isnan(absv)) else float("nan")
        broken = (not np.isnan(mean_s)) and mean_s >= BROKEN_SPEARMAN_THRESHOLD
        lines.append(
            f"| {m} | {int(np.sum(~np.isnan(vals)))} | {mean_s:.4f} | {mean_a:.4f} | "
            f"{'BROKEN (attributions do not degrade)' if broken else 'attributions degrade'} |"
        )
    lines += ["", "![faithfulness curves](figures/faithfulness_curves.png)", "", "Dashed lines: random-attribution control.", ""]
    (report_dir / "faithfulness.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== faithfulness summary ===")
    print(auc_df.round(4).to_string(index=False))
    print(f"written: {report_dir / 'faithfulness.md'}, {report_dir / 'faithfulness.csv'}, "
          f"{report_dir / 'faithfulness_per_example.csv'}, {report_dir / 'figures' / 'faithfulness_curves.png'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
