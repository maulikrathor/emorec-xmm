"""Tests for emorec.utils: seeding, config loading, and JSONL logging."""

import json
from pathlib import Path

import torch
from torch import nn

from emorec.utils.config import load_config
from emorec.utils.log import JsonlLogger
from emorec.utils.seed import seed_everything


def _build_and_run_mlp() -> float:
    """Build a small 3-layer MLP, run one forward+backward pass on random data.

    Both the model's random initialization and the random input data are
    drawn from the global torch RNG, so calling `seed_everything(42)`
    immediately before this function makes the whole sequence deterministic.
    """
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(5, 16)
    y = torch.randint(0, 4, (5,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    return loss.item()


def test_seed_everything_gives_bitwise_equal_losses() -> None:
    seed_everything(42)
    loss_a = _build_and_run_mlp()

    seed_everything(42)
    loss_b = _build_and_run_mlp()

    assert loss_a == loss_b


def test_load_config_dotlist_override_changes_nested_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  lr: 2.0e-4\n  batch_size: 8\n", encoding="utf-8")

    cfg = load_config(str(config_path))
    assert cfg.train.lr == 2.0e-4

    overridden = load_config(str(config_path), overrides=["train.lr=1.0e-3"])
    assert overridden.train.lr == 1.0e-3
    assert overridden.train.batch_size == 8  # untouched keys survive the merge


def test_jsonl_logger_round_trips_rows_with_ts(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "log.jsonl"
    logger = JsonlLogger(log_path)

    logger.log({"step": 1, "loss": 0.5})
    logger.log({"step": 2, "loss": 0.4})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    rows = [json.loads(line) for line in lines]
    assert rows[0]["step"] == 1
    assert rows[1]["step"] == 2
    for row in rows:
        assert "ts" in row
        assert isinstance(row["ts"], str)
