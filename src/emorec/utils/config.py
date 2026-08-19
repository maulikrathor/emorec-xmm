"""OmegaConf-based config loading and run-directory bookkeeping.

CONTRACT: `configs/base.yaml` is the single source of default truth (per
CLAUDE.md). Every entrypoint loads it through `load_config` (a single,
already-complete config file, e.g. `base.yaml` itself) or
`load_experiment_config` (a MINIMAL override file merged on top of
`base.yaml`, ED11) and layers dotlist overrides on top — never hand-construct
config dicts, never read YAML with a bare `yaml.safe_load` elsewhere in the
codebase.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load `path` as a SELF-CONTAINED config, merge in dotlist `overrides`, and resolve interpolations.

    `overrides` uses OmegaConf dotlist syntax, e.g. `["train.lr=1e-4",
    "data.context_k=0"]`. The returned config has all `${...}` interpolations
    resolved, so downstream code never needs to call `OmegaConf.resolve`
    itself.

    `path` must already contain every key the caller will read (this is
    `base.yaml`'s own loader). It does NOT merge with `base.yaml` — for a
    minimal per-experiment override file (e.g. `configs/text_only.yaml`,
    which only sets `model.name`), use `load_experiment_config` instead, or
    this raises `ConfigAttributeError` on the first key the override file
    doesn't repeat (e.g. `cfg.run.seed`).
    """
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def load_experiment_config(
    config_path: str,
    overrides: list[str] | None = None,
    base_path: str = "configs/base.yaml",
) -> DictConfig:
    """Load `config_path` as a MINIMAL override of `base_path`, merge in dotlist `overrides`, resolve.

    CONTRACT (ED11, "experiments are config diffs"): every experiment config
    under `configs/` (`text_only.yaml`, `audio_only.yaml`, `xattn.yaml`, and
    any future one) is a PARTIAL diff against `base.yaml` — it repeats only
    the keys it changes (typically just `model.name`), never the whole file.
    This function is what actually performs that composition: `base_path` is
    loaded first, `config_path` is merged on top of it, then dotlist
    `overrides` are merged last, then the result is resolved once. If
    `config_path` already resolves to `base_path` (the same file), the merge
    is a harmless no-op.

    Use this — never `load_config` directly — for any config file that isn't
    `base.yaml` itself; `load_config` has no notion of a base and will raise
    `ConfigAttributeError` on any key (e.g. `cfg.run.seed`) the override file
    doesn't repeat. This is the fix for the exact failure seen training B1 on
    Kaggle: `python scripts/train.py --config configs/text_only.yaml` ->
    `cfg.run.seed = args.seed` -> `ConfigAttributeError: Missing key run`,
    because `text_only.yaml` alone has no `run:` section at all.
    """
    base_cfg = OmegaConf.load(base_path)
    override_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, override_cfg)
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
