"""Build the fp16 on-disk audio feature cache for one manifest.

CONTRACT: only AUDIO features are cached here. Text is deliberately NEVER
cached — `context_k` (the dialogue-context window) is an experiment
variable (ablated in E2, changes the text every time it's touched), and
XAI (T-18+) needs gradients flowing through the live text embeddings, so
text is always tokenized+encoded online.

Usage:
    python scripts/build_cache.py --manifest artifacts/manifests/dev.jsonl \\
        --config configs/base.yaml [--subset-n N] [--seed 42] \\
        [--batch-size 16] [--device auto|cpu|cuda] [--force]

Resume-safe: an utterance whose `.npz` already exists (and is non-empty) is
skipped unless `--force`. Only the TRUE (unpadded) frame length of each
utterance's hidden states is ever written to disk — padding is never
persisted.
"""

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from emorec.data.dataset import MeldDataset, make_collate_fn, stratified_subset_indices
from emorec.models.encoders import AudioEncoder, cache_key, cache_path, save_cached_features
from emorec.utils.config import load_config, run_dir
from emorec.utils.log import JsonlLogger
from emorec.utils.seed import seed_everything


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to a {split}.jsonl manifest.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--subset-n", type=int, default=None, help="Restrict to a stratified subset of this size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true", help="Re-encode and overwrite even if cached.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(args.seed)

    rdir = run_dir(cfg)
    run_logger = JsonlLogger(rdir / "build_cache.log.jsonl")
    run_logger.log(
        {
            "event": "start",
            "manifest": args.manifest,
            "subset_n": args.subset_n,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "force": args.force,
        }
    )

    device = _resolve_device(args.device)

    ds = MeldDataset(args.manifest, cfg, load_audio=True)
    if args.subset_n is not None:
        indices = stratified_subset_indices(ds, args.subset_n, args.seed)
        ds = MeldDataset(
            args.manifest, cfg, tokenizer=ds.tokenizer, subset_indices=indices, load_audio=True
        )

    collate_fn = make_collate_fn(ds.tokenizer)
    # Not CC-3's make_dataloader: its batch size is cfg.train.batch_size,
    # which is a training-loop concern, not this script's own --batch-size
    # knob. Order doesn't matter for cache building (no shuffling needed).
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    encoder = AudioEncoder(cfg).to(device)
    encoder.eval()

    cache_root = Path(cfg.data.cache_dir) / cache_key(cfg.model.audio_ckpt)

    n_cached = 0
    n_skipped = 0
    start = time.time()
    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="build_cache"):
        utt_ids = [m["utt_id"] for m in batch.meta]
        targets = [cache_path(cache_root, uid) for uid in utt_ids]
        todo = [args.force or not (p.exists() and p.stat().st_size > 0) for p in targets]

        if not any(todo):
            n_skipped += len(utt_ids)
            continue

        batch = batch.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            hidden, frame_mask = encoder(batch.audio_values, batch.audio_mask)

        for i, uid in enumerate(utt_ids):
            if not todo[i]:
                n_skipped += 1
                continue
            true_len = int(frame_mask[i].sum().item())
            h = hidden[i, :true_len].to(torch.float16).cpu()
            m = frame_mask[i, :true_len].cpu()
            save_cached_features(targets[i], h, m)
            n_cached += 1

    elapsed = time.time() - start
    total_bytes = sum(p.stat().st_size for p in cache_root.rglob("*.npz")) if cache_root.exists() else 0

    print(
        f"build_cache: cached={n_cached} skipped={n_skipped} "
        f"total_bytes_on_disk={total_bytes} wall_time_s={elapsed:.1f} cache_root={cache_root}"
    )
    run_logger.log(
        {
            "event": "done",
            "cached": n_cached,
            "skipped": n_skipped,
            "total_bytes_on_disk": total_bytes,
            "wall_time_s": elapsed,
            "cache_root": str(cache_root),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
