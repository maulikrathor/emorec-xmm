"""Predict the emotion of ONE MELD utterance and print a human-readable explanation.

This is the live-demo script: its output must be clean and self-explaining
without narration. It reuses `emorec.data`/`emorec.models` exactly as
`scripts/train.py` does (same cached-audio path, same model construction) --
it never reimplements tokenization, context building, or audio loading.

Usage:
    python scripts/demo_predict.py --ckpt artifacts/runs/<run>/best.pt \\
        --config configs/text_only.yaml \\
        [--manifest artifacts/manifests/dev.jsonl] \\
        [--utt-id dev/dia0_utt0 | --index 0 | --random] \\
        [--seed 42] [--device auto] [--use-cache | --no-use-cache]

Exactly one of `--utt-id`/`--index`/`--random` should be given; if none are,
the first example in the dataset (`--index 0`) is used.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from emorec.data.cached_dataset import CachedAudioMeldDataset, make_collate_fn_cached
from emorec.data.dataset import MeldDataset, make_collate_fn
from emorec.data.labels import EMOTION_NAMES, ID_TO_EMOTION
from emorec.models.registry import build_model
from emorec.utils.config import load_experiment_config
from emorec.utils.seed import seed_everything

BAR_WIDTH = 40  # characters at probability 1.0


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_dataset(manifest: str, cfg, use_cache: bool):
    """Same construction pattern as scripts/train.py's `_build_dataset`."""
    if cfg.model.name == "text_only":
        # B1 never touches audio: no WAV reads, no audio-cache dependency.
        return MeldDataset(manifest, cfg, load_audio=False)
    if use_cache:
        return CachedAudioMeldDataset(manifest, cfg, use_cache=True)
    return MeldDataset(manifest, cfg, load_audio=True)


def _find_wav_path(manifest_path: str, utt_id: str) -> str | None:
    """Read `wav_path` for `utt_id` straight from the manifest.

    `MeldDataset.__getitem__`'s `meta` dict carries `utt_id`/`context`/
    `speaker`/`text`/`duration` but not `wav_path` (CC-3 contract). This is a
    plain JSONL field lookup, not a reimplementation of any dataset logic --
    tokenization/context-building/audio-loading all still go through
    `MeldDataset`/`CachedAudioMeldDataset` themselves.
    """
    with Path(manifest_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("utt_id") == utt_id:
                return row.get("wav_path")
    return None


def _select_item(dataset, args: argparse.Namespace) -> dict:
    """Return the one `dataset[i]` item selected by --utt-id / --random / --index (default 0)."""
    if args.utt_id:
        for i in range(len(dataset)):
            item = dataset[i]
            if item["meta"]["utt_id"] == args.utt_id:
                return item
        raise ValueError(f"utt_id {args.utt_id!r} not found in {args.manifest}")

    if args.random:
        rng = random.Random(args.seed)
        idx = rng.randrange(len(dataset))
        return dataset[idx]

    idx = args.index if args.index is not None else 0
    if not (0 <= idx < len(dataset)):
        raise ValueError(f"--index {idx} out of range for dataset of size {len(dataset)}")
    return dataset[idx]


def _bar(prob: float, width: int = BAR_WIDTH) -> str:
    return "|" * max(0, round(prob * width))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/dev.jsonl")
    parser.add_argument("--utt-id", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=None)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seed_everything(args.seed)

    cfg = load_experiment_config(args.config)
    use_cache = args.use_cache if args.use_cache is not None else bool(cfg.train.use_cache)
    if cfg.model.name == "text_only":
        use_cache = False  # plain MeldDataset(load_audio=False) + plain collate
    device = _resolve_device(args.device)

    dataset = _build_dataset(args.manifest, cfg, use_cache)
    item = _select_item(dataset, args)

    collate_fn = make_collate_fn_cached(dataset.tokenizer) if use_cache else make_collate_fn(dataset.tokenizer)
    batch = collate_fn([item]).to(device)

    tokenizer_len = len(dataset.tokenizer)
    new_token_ids = [dataset.cur_token_id, dataset.end_cur_token_id]
    model = build_model(cfg, tokenizer_len, new_token_ids).to(device)

    state_dict = torch.load(args.ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {args.ckpt!r} does not match the model built from "
            f"cfg.model.name={cfg.model.name!r} (state_dict keys/shapes mismatch). "
            "This almost always means the checkpoint was trained with a DIFFERENT "
            "--config (a different model.name) than the one given here. "
            f"Original error: {exc}"
        ) from exc
    model.eval()

    with torch.no_grad():
        output = model(batch)

    probs = torch.softmax(output.logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())
    true_id = int(batch.label[0].item())

    meta = item["meta"]
    wav_path = _find_wav_path(args.manifest, meta["utt_id"])

    print(f"UTT_ID:            {meta['utt_id']}")
    print(f"SPEAKER:           {meta['speaker']}")
    print(f"TEXT:              {meta['text']}")
    print(f"CONTEXT STRING:    {meta['context']}")
    print(f"AUDIO:             {wav_path or '(unknown)'}  ({meta['duration']:.2f}s)")
    print(f"TRUE EMOTION:      {ID_TO_EMOTION[true_id]} ({true_id})")
    print(f"PREDICTED EMOTION: {ID_TO_EMOTION[pred_id]} ({pred_id})")
    print(f"CORRECT?:          {'yes' if pred_id == true_id else 'no'}")

    print("PROBABILITIES:")
    ranked = sorted(range(len(EMOTION_NAMES)), key=lambda i: float(probs[i]), reverse=True)
    for i in ranked:
        p = float(probs[i])
        print(f"  {EMOTION_NAMES[i]:<10} {p:.2f} {_bar(p)}")

    print("MODALITY GATE:")
    if output.gate is None:
        print("  gate: n/a (unimodal baseline)")
    else:
        g_text = float(output.gate[0, 0].item())
        g_audio = float(output.gate[0, 1].item())
        print(f"  g_text={g_text:.3f}  g_audio={g_audio:.3f}")
        print(f"  -> model weighted text {g_text:.2f} / audio {g_audio:.2f} for this utterance")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
