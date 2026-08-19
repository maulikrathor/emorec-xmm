"""Regression test for the Kaggle B1-training config bug.

`python scripts/train.py --config configs/text_only.yaml ...` failed on
Kaggle at `cfg.run.seed = args.seed` with
`omegaconf.errors.ConfigAttributeError: Missing key run` -- because
`configs/text_only.yaml` (and `audio_only.yaml`/`xattn.yaml`) are MINIMAL
overrides of `configs/base.yaml` (ED11) that only set `model.name`, and
`scripts/train.py`/`scripts/evaluate.py` were loading them with
`load_config`, which loads exactly one file with no notion of a base to
merge against. The fix is `emorec.utils.config.load_experiment_config`,
which merges `base.yaml` first, the override file on top, then dotlist
overrides. This test loads the REAL config files (not synthetic stand-ins)
so it fails again immediately if the composition regresses.
"""

from omegaconf import OmegaConf

from emorec.utils.config import load_config, load_experiment_config


def test_text_only_config_inherits_run_section_from_base() -> None:
    """The exact failure: cfg.run.seed must resolve for configs/text_only.yaml (B1)."""
    cfg = load_experiment_config("configs/text_only.yaml")
    assert cfg.run.seed == 42
    assert cfg.model.name == "text_only"
    # Inherited sections that text_only.yaml never repeats:
    assert cfg.data.sample_rate == 16000
    assert cfg.train.lr == 2.0e-4
    assert cfg.loss.name == "weighted_ce"


def test_audio_only_config_inherits_run_section_from_base() -> None:
    """Same failure mode, B2."""
    cfg = load_experiment_config("configs/audio_only.yaml")
    assert cfg.run.seed == 42
    assert cfg.model.name == "audio_only"


def test_xattn_config_inherits_run_section_from_base() -> None:
    """Same failure mode, M (also base.yaml's own default, so this doubles as a no-op-merge check)."""
    cfg = load_experiment_config("configs/xattn.yaml")
    assert cfg.run.seed == 42
    assert cfg.model.name == "xattn"


def test_experiment_config_overrides_base_on_conflicting_keys() -> None:
    """The override file's own keys must WIN over base.yaml's, not just fill gaps."""
    cfg = load_experiment_config("configs/text_only.yaml")
    base_only = load_config("configs/base.yaml")
    assert base_only.model.name == "xattn"  # base.yaml's own default
    assert cfg.model.name == "text_only"  # text_only.yaml's override wins


def test_experiment_config_seed_override_still_works_via_dotlist() -> None:
    """--seed is applied on top of scripts/train.py's cfg via a plain attribute
    set (cfg.run.seed = args.seed) after loading -- this only works at all if
    cfg.run exists in the first place, which is exactly what was missing."""
    cfg = load_experiment_config("configs/text_only.yaml")
    cfg.run.seed = 13
    assert cfg.run.seed == 13


def test_load_config_alone_does_not_inherit_from_base() -> None:
    """Documents WHY the bug happened: load_config (unlike load_experiment_config)
    loads exactly the given file, nothing more -- configs/text_only.yaml alone
    has no run: section at all."""
    cfg = OmegaConf.load("configs/text_only.yaml")
    assert "run" not in cfg
    assert "data" not in cfg
    assert "train" not in cfg


def test_load_experiment_config_matches_base_when_config_path_is_base_itself() -> None:
    """config_path == base_path must be a harmless no-op merge."""
    cfg = load_experiment_config("configs/base.yaml", base_path="configs/base.yaml")
    plain = load_config("configs/base.yaml")
    assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(plain, resolve=True)
