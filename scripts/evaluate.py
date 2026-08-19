"""Evaluate a trained checkpoint on a manifest.

Usage:
    python scripts/evaluate.py --ckpt artifacts/runs/<run>/best.pt \\
        --config configs/text_only.yaml --manifest artifacts/manifests/dev.jsonl \\
        [--split dev] [--use-cache | --no-use-cache] [--device auto]

HARD GUARD (A6, test-split discipline): refuses to run on a manifest whose
FILENAME contains "test" unless `--allow-test` is passed. Dev drives every
decision; test is evaluated ONCE per final model, never for exploration —
"quick check on test" is exactly the mistake this guard exists to block.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import MeldDataset, make_collate_fn, make_dataloader
from emorec.data.labels import EMOTION_NAMES
from emorec.models.registry import build_model
from emorec.training.metrics import (
    classification_report_text,
    compute_metrics,
    save_confusion_matrix,
)
from emorec.training.trainer import evaluate as run_evaluate
from emorec.utils.config import load_experiment_config


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=None)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required to evaluate on a manifest whose filename contains 'test' (A6 discipline).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if "test" in Path(args.manifest).name.lower() and not args.allow_test:
        print(
            f"REFUSING to evaluate on {args.manifest!r} (filename contains 'test') without "
            "--allow-test.\n"
            "Test-split discipline (A6): dev drives every decision; test is evaluated ONCE "
            "per final model, never for exploration — no peeking, no 'quick check on test'. "
            "Pass --allow-test only for that one final, deliberate run.",
            file=sys.stderr,
        )
        return 2

    # load_experiment_config, not load_config: --config is a MINIMAL override
    # of configs/base.yaml (ED11), same reasoning as scripts/train.py.
    cfg = load_experiment_config(args.config)
    use_cache = args.use_cache if args.use_cache is not None else bool(cfg.train.use_cache)
    device = _resolve_device(args.device)

    if use_cache:
        ds = CachedAudioMeldDataset(args.manifest, cfg, use_cache=True)
        collate_fn = make_collate_fn_cached(ds.tokenizer)
    else:
        ds = MeldDataset(args.manifest, cfg, load_audio=True)
        collate_fn = make_collate_fn(ds.tokenizer)

    loader = make_dataloader(ds, cfg, shuffle=False, seed=cfg.run.seed, collate_fn=collate_fn)

    tokenizer_len = len(ds.tokenizer)
    new_token_ids = [ds.cur_token_id, ds.end_cur_token_id]
    model = build_model(cfg, tokenizer_len, new_token_ids).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    y_true, y_pred, _y_prob = run_evaluate(model, loader, device=device)
    metrics = compute_metrics(y_true, y_pred)

    out_dir = Path(args.ckpt).parent
    save_confusion_matrix(
        metrics["confusion_matrix"],
        out_dir / f"confusion_matrix_{args.split}.png",
        EMOTION_NAMES,
        f"{cfg.model.name} — {args.split} confusion matrix",
    )
    (out_dir / f"classification_report_{args.split}.txt").write_text(
        classification_report_text(y_true, y_pred), encoding="utf-8"
    )
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"{args.split}: accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(f"artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
