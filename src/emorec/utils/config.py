"""OmegaConf-based config loading and run-directory bookkeeping.

CONTRACT: `configs/base.yaml` is the single source of default truth (per
CLAUDE.md). Every entrypoint loads it through `load_config` and layers
dotlist overrides on top — never hand-construct config dicts, never read
YAML with a bare `yaml.safe_load` elsewhere in the codebase.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load `path` as a config, merge in dotlist `overrides`, and resolve interpolations.

    `overrides` uses OmegaConf dotlist syntax, e.g. `["train.lr=1e-4",
    "data.context_k=0"]`. The returned config has all `${...}` interpolations
    resolved, so downstream code never needs to call `OmegaConf.resolve`
    itself.
    """
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _git_sha() -> str:
    """Return the current `git rev-parse HEAD`, or "unknown" if git/the repo is unavailable.

    Never raises: a missing git binary, a non-repo working directory, or any
    other failure all degrade to the literal string "unknown" rather than
    crashing a run.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def run_dir(cfg: DictConfig) -> Path:
    """Create and return `artifacts/runs/{cfg.run.name}-{UTC timestamp}/`.

    Side effects: creates the directory (with parents), writes the fully
    resolved config to `config.yaml`, and writes the output of `git rev-parse
    HEAD` (or "unknown") to `git_sha.txt` inside it. The timestamp is UTC and
    colon-free (`%Y%m%dT%H%M%SZ`) so the path is valid on all platforms,
    including Windows.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/runs") / f"{cfg.run.name}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, out_dir / "config.yaml")
    (out_dir / "git_sha.txt").write_text(_git_sha() + "\n", encoding="utf-8")

    return out_dir
