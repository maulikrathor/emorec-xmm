"""Train one model (B1/B2/B3/M, per cfg.model.name) on MELD.

Usage:
    python scripts/train.py --config configs/text_only.yaml \\
        --train-manifest artifacts/manifests/train.jsonl \\
        --dev-manifest artifacts/manifests/dev.jsonl [--subset-n N] \\
        [--seed 42] [--epochs N] [--device auto] [--use-cache | --no-use-cache] \\
        [dotlist.override=value ...]

Builds datasets (cached audio via CC-4's disk cache by default,
`cfg.train.use_cache`), the model via `emorec.models.registry.build_model`,
the loss with TRAIN-split-only class counts, then runs
`emorec.training.trainer.fit`. Writes
`artifacts/runs/<name>-<ts>/{metrics.json,confusion_matrix.png,
classification_report.txt,best.pt,last.pt,run_meta.json}`. Prints the
trivial all-neutral baseline metrics alongside the trained model's dev
result — the honesty anchor (docs/CLAUDE.md commandment #9).
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import (
    MeldDataset,
    make_collate_fn,
    make_dataloader,
    stratified_subset_indices,
)
from emorec.data.labels import EMOTION_NAMES, NUM_CLASSES
from emorec.models.registry import build_model
from emorec.training.losses import build_loss
from emorec.training.metrics import (
    classification_report_text,
    compute_metrics,
    save_confusion_matrix,
    trivial_baseline_metrics,
)
from emorec.training.trainer import evaluate, fit
from emorec.utils.config import load_experiment_config, run_dir
from emorec.utils.seed import seed_everything


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _class_counts(dataset, num_classes: int) -> list[int]:
    counts = Counter(dataset.labels())
    return [counts.get(i, 0) for i in range(num_classes)]


def _build_dataset(manifest: str, cfg, use_cache: bool, tokenizer=None, subset_indices=None):
    if use_cache:
        return CachedAudioMeldDataset(
            manifest, cfg, tokenizer=tokenizer, subset_indices=subset_indices, use_cache=True
        )
    return MeldDataset(manifest, cfg, tokenizer=tokenizer, subset_indices=subset_indices, load_audio=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--dev-manifest", default=None)
    parser.add_argument("--subset-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=None)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-4")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # load_experiment_config, not load_config: --config is a MINIMAL override
    # of configs/base.yaml (ED11) -- e.g. configs/text_only.yaml only sets
    # model.name, it has no run:/data:/train:/loss: sections of its own.
    cfg = load_experiment_config(args.config, overrides=args.overrides)

    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.epochs is not None:
        cfg.train.max_epochs = args.epochs
    use_cache = args.use_cache if args.use_cache is not None else bool(cfg.train.use_cache)

    seed_everything(cfg.run.seed)

    train_manifest = args.train_manifest or str(Path(cfg.data.manifest_dir) / "train.jsonl")
    dev_manifest = args.dev_manifest or str(Path(cfg.data.manifest_dir) / "dev.jsonl")

    device = _resolve_device(args.device)

    train_ds = _build_dataset(train_manifest, cfg, use_cache)
    if args.subset_n is not None:
        indices = stratified_subset_indices(train_ds, args.subset_n, cfg.run.seed)
        train_ds = _build_dataset(train_manifest, cfg, use_cache, tokenizer=train_ds.tokenizer, subset_indices=indices)

    dev_ds = _build_dataset(dev_manifest, cfg, use_cache, tokenizer=train_ds.tokenizer)

    collate_fn = make_collate_fn_cached(train_ds.tokenizer) if use_cache else make_collate_fn(train_ds.tokenizer)
    train_loader = make_dataloader(train_ds, cfg, shuffle=True, seed=cfg.run.seed, collate_fn=collate_fn)
    dev_loader = make_dataloader(dev_ds, cfg, shuffle=False, seed=cfg.run.seed, collate_fn=collate_fn)

    tokenizer_len = len(train_ds.tokenizer)
    new_token_ids = [train_ds.cur_token_id, train_ds.end_cur_token_id]
    model = build_model(cfg, tokenizer_len, new_token_ids).to(device)

    class_counts = _class_counts(train_ds, NUM_CLASSES)
    loss_fn = build_loss(cfg, class_counts)

    rdir = run_dir(cfg)  # creates the run dir, writes config.yaml + git_sha.txt
    print(f"run dir: {rdir}")
    print(f"train class counts (TRAIN split only): {class_counts}")

    fit(
        model,
        train_loader,
        dev_loader,
        loss_fn,
        cfg,
        rdir,
        split_names={"train": train_manifest, "dev": dev_manifest},
        subset_size=args.subset_n,
    )

    # Reload the BEST checkpoint before reporting: 'best.pt' is what
    # scripts/evaluate.py will load later, so the metrics/artifacts written
    # here must correspond to that checkpoint, not whatever the last epoch
    # happened to leave in memory.
    model.load_state_dict(torch.load(rdir / "best.pt", map_location=device))

    y_true, y_pred, _y_prob = evaluate(model, dev_loader, device=device)
    dev_metrics = compute_metrics(y_true, y_pred)
    trivial_metrics = trivial_baseline_metrics(y_true)

    save_confusion_matrix(
        dev_metrics["confusion_matrix"],
        rdir / "confusion_matrix.png",
        EMOTION_NAMES,
        f"{cfg.model.name} — dev confusion matrix (best checkpoint)",
    )
    (rdir / "classification_report.txt").write_text(
        classification_report_text(y_true, y_pred), encoding="utf-8"
    )

    print(f"\n=== {cfg.model.name} — dev (best checkpoint) ===")
    print(
        f"accuracy={dev_metrics['accuracy']:.4f}  macro_f1={dev_metrics['macro_f1']:.4f}  "
        f"weighted_f1={dev_metrics['weighted_f1']:.4f}"
    )
    print("=== trivial all-neutral baseline — dev (honesty anchor) ===")
    print(
        f"accuracy={trivial_metrics['accuracy']:.4f}  macro_f1={trivial_metrics['macro_f1']:.4f}  "
        f"weighted_f1={trivial_metrics['weighted_f1']:.4f}"
    )
    print(f"\nartifacts written to {rdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
