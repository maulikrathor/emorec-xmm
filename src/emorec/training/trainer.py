"""The single training loop every model (baseline or fusion) in this repo uses.

CONTRACT: `fit`'s cosine-with-warmup schedule steps on OPTIMIZER steps
(`total_batches // grad_accum * epochs`), NEVER on raw micro-batches — the
classic gradient-accumulation bug is stepping the scheduler once per
micro-batch, which decays the LR `grad_accum`-times too fast
(docs/CLAUDE.md "Known dragon locations"). AMP order is
`unscale_ -> clip -> step` (scaler.step ordering with clipping).
"""

import json
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from emorec.training.metrics import compute_metrics
from emorec.utils.log import JsonlLogger


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> tuple[list[int], list[int], list[list[float]]]:
    """Run `model` over every batch in `loader` in `eval()`/`no_grad()`, aligned by example order.

    Returns `(y_true, y_pred, y_prob)` as plain Python lists/lists-of-lists
    (never tensors) so callers never need to know which device the model
    ran on. `device` defaults to the model's own device (its first
    parameter's `.device`) if not given.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            probs = torch.softmax(output.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            y_true.extend(batch.label.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    return y_true, y_pred, y_prob


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay to 0 over the remaining `total_steps`."""
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    loss_fn: nn.Module,
    cfg: DictConfig,
    run_dir: str | Path,
    *,
    split_names: dict[str, str],
    subset_size: int | None = None,
) -> dict:
    """Train `model` per `cfg.train.*`, evaluating on `dev_loader` every epoch; early-stops on dev weighted-F1.

    `loss_fn`/`split_names`/`subset_size` are NOT in the abbreviated
    5-argument signature from the ticket text, but are unavoidable: the
    trainer cannot invent a loss function (`emorec.training.losses.build_loss`
    needs TRAIN-split-only class counts only the caller has) or know which
    manifests/subset size produced `train_loader`/`dev_loader` (that's a
    fact about how the caller built the datasets, not derivable from `cfg`
    or the loaders themselves) — both are required to satisfy this
    function's OWN `run_meta.json` contract below.

    AdamW over `model.parameters()` with `requires_grad=True` only, lr/wd
    from `cfg.train`. Cosine-with-warmup schedule computed on OPTIMIZER
    steps (`len(train_loader) // grad_accum * max_epochs`), not raw
    micro-batches. Gradient accumulation (`cfg.train.grad_accum`), grad-norm
    clipping (`cfg.train.grad_clip`), AMP autocast + `GradScaler` when the
    model is on `cuda` and `cfg.train.amp` is true (order: `unscale_ ->
    clip -> step`). Early stop on dev weighted-F1 not improving for
    `cfg.train.early_stop_patience` epochs.

    Writes, into `run_dir`: `last.pt` (every epoch), `best.pt` (whenever dev
    weighted-F1 improves), `metrics.json` (one JSONL line per epoch, via
    `JsonlLogger`), and `run_meta.json` (split names, subset size, seed,
    model name, the FULL resolved config, git SHA — read back from
    `run_dir/git_sha.txt` if `emorec.utils.config.run_dir` wrote one there,
    else `"unknown"` — epoch count, and wall time).

    Returns `{"best_dev_weighted_f1": float, "epochs_run": int, "wall_time_s": float}`.
    """
    run_dir = Path(run_dir)
    device = next(model.parameters()).device
    loss_fn = loss_fn.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    grad_accum = max(1, int(cfg.train.grad_accum))
    max_epochs = int(cfg.train.max_epochs)
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = max(1, steps_per_epoch // grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * max_epochs
    warmup_steps = max(1, int(cfg.train.warmup_ratio * total_optimizer_steps))

    scheduler = LambdaLR(
        optimizer, lambda step: _lr_lambda(step, warmup_steps, total_optimizer_steps)
    )

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    epoch_logger = JsonlLogger(run_dir / "metrics.json")

    best_weighted_f1 = -1.0
    epochs_without_improvement = 0
    epochs_run = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()  # encoder submodules override train() to stay eval() (J-03) -- only the fusion head flips
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch)
                loss = loss_fn(output.logits, batch.label) / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss_sum += loss.item() * grad_accum
            epoch_loss_count += 1

            is_last_micro_batch = (i + 1) == steps_per_epoch
            if (i + 1) % grad_accum == 0 or is_last_micro_batch:
                if use_amp:
                    scaler.unscale_(optimizer)
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = epoch_loss_sum / max(1, epoch_loss_count)

        y_true, y_pred, _y_prob = evaluate(model, dev_loader, device=device)
        dev_metrics = compute_metrics(y_true, y_pred)

        print(
            f"epoch {epoch + 1}/{max_epochs}  train_loss={train_loss:.4f}  "
            f"dev_acc={dev_metrics['accuracy']:.4f}  dev_macro_f1={dev_metrics['macro_f1']:.4f}  "
            f"dev_weighted_f1={dev_metrics['weighted_f1']:.4f}"
        )
        epoch_logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_weighted_f1": dev_metrics["weighted_f1"],
            }
        )
        epochs_run = epoch + 1

        torch.save(model.state_dict(), run_dir / "last.pt")

        if dev_metrics["weighted_f1"] > best_weighted_f1:
            best_weighted_f1 = dev_metrics["weighted_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.train.early_stop_patience:
            break

    if not (run_dir / "best.pt").exists():
        # Defensive fallback: should be unreachable (epoch 1 always
        # "improves" from best_weighted_f1=-1.0), but never leave best.pt
        # missing.
        torch.save(model.state_dict(), run_dir / "best.pt")

    wall_time_s = time.time() - start_time

    git_sha_path = run_dir / "git_sha.txt"
    git_sha = git_sha_path.read_text(encoding="utf-8").strip() if git_sha_path.exists() else "unknown"

    run_meta = {
        "split_names": split_names,
        "subset_size": subset_size,
        "seed": cfg.run.seed,
        "model_name": cfg.model.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_sha": git_sha,
        "epoch_count": epochs_run,
        "wall_time_s": wall_time_s,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return {
        "best_dev_weighted_f1": best_weighted_f1,
        "epochs_run": epochs_run,
        "wall_time_s": wall_time_s,
    }
