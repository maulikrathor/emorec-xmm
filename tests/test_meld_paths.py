"""Tests for emorec.data.meld_paths: root resolution and layout discovery."""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from emorec.data.meld_paths import (
    VIDEO_DIR_CANDIDATES,
    resolve_meld_layout,
    resolve_meld_root,
)


def test_resolve_meld_layout_finds_nested_video_dirs_over_their_parents(toy_meld_root: Path) -> None:
    layout = resolve_meld_layout(toy_meld_root)

    assert layout.splits["train"].csv == toy_meld_root / "train" / "train_sent_emo.csv"
    assert layout.splits["dev"].csv == toy_meld_root / "dev_sent_emo.csv"
    assert layout.splits["test"].csv == toy_meld_root / "test_sent_emo.csv"

    # Video dirs are nested one level down in the real mirror; the parent
    # directories (train/, dev/, test/) hold zero direct .mp4 files and must
    # never be chosen.
    assert layout.splits["train"].video_dir == toy_meld_root / "train" / "train_splits"
    assert layout.splits["dev"].video_dir == toy_meld_root / "dev" / "dev_splits_complete"
    assert layout.splits["test"].video_dir == toy_meld_root / "test" / "output_repeated_splits_test"


def test_video_dir_candidate_with_only_appledouble_files_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "train").mkdir(parents=True)
    (root / "train" / "train_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "dev_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "test_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")

    # First candidate "train/train_splits" contains ONLY AppleDouble files
    # -> must be rejected, not accepted as "has .mp4 files".
    appledouble_only = root / "train" / "train_splits"
    appledouble_only.mkdir(parents=True)
    (appledouble_only / "._dia0_utt0.mp4").write_bytes(b"\x00\x05\x16\x07")

    # Second candidate "train_splits" (direct child of root) has real media
    # -> should be picked instead.
    real_dir = root / "train_splits"
    real_dir.mkdir()
    (real_dir / "dia0_utt0.mp4").write_bytes(b"x")

    # Minimal real media so dev/test resolve too (this test only exercises
    # train's candidate-rejection behavior).
    (root / "dev" / "dev_splits_complete").mkdir(parents=True)
    (root / "dev" / "dev_splits_complete" / "dia0_utt0.mp4").write_bytes(b"x")
    (root / "test" / "output_repeated_splits_test").mkdir(parents=True)
    (root / "test" / "output_repeated_splits_test" / "dia0_utt0.mp4").write_bytes(b"x")

    layout = resolve_meld_layout(root)
    assert layout.splits["train"].video_dir == real_dir


def test_video_dir_candidate_with_zero_direct_mp4_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "train").mkdir(parents=True)
    (root / "train" / "train_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "dev_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "test_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")

    # "train/train_splits" exists but is empty -> rejected.
    (root / "train" / "train_splits").mkdir(parents=True)
    # "train_splits" exists but is empty too -> rejected.
    (root / "train_splits").mkdir()
    # "train" itself (third candidate) has zero DIRECT .mp4 (only nested
    # ones inside train_splits, which is non-recursive-checked) -> also
    # rejected since train_splits contains no real files. So the whole
    # split fails to resolve.
    with pytest.raises(FileNotFoundError, match="train"):
        resolve_meld_layout(root)


def test_unresolvable_video_dir_raises_with_directory_enumeration(tmp_path: Path) -> None:
    root = tmp_path / "weird_root"
    (root / "train").mkdir(parents=True)
    (root / "train" / "train_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "dev_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")
    (root / "test_sent_emo.csv").write_text("Utterance\nhi\n", encoding="utf-8")

    # Videos live under a directory name that matches none of the train
    # candidates ("train/train_splits", "train_splits", "train").
    weird_dir = root / "train" / "clips"
    weird_dir.mkdir()
    for i in range(3):
        (weird_dir / f"dia0_utt{i}.mp4").write_bytes(b"x")
    # Plus one AppleDouble file in the same directory, to prove the
    # enumeration reports total/appledouble/real separately.
    (weird_dir / "._dia0_utt9.mp4").write_bytes(b"\x00\x05\x16\x07")

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_meld_layout(root)

    message = str(excinfo.value)
    assert "train" in message
    assert "clips" in message
    assert "total=4" in message
    assert "appledouble=1" in message
    assert "real=3" in message


def test_resolve_meld_root_precedence_cli_over_env_over_cfg(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli_dir = tmp_path / "cli"
    env_dir = tmp_path / "env"
    cfg_dir = tmp_path / "cfg"
    for d in (cli_dir, env_dir, cfg_dir):
        d.mkdir()

    cfg = OmegaConf.create({"data": {"meld_root": str(cfg_dir)}})

    # All three set: CLI wins.
    monkeypatch.setenv("MELD_ROOT", str(env_dir))
    assert resolve_meld_root(str(cli_dir), cfg) == cli_dir

    # No CLI arg: env wins over cfg.
    assert resolve_meld_root(None, cfg) == env_dir

    # No CLI arg, no env: cfg wins.
    monkeypatch.delenv("MELD_ROOT", raising=False)
    assert resolve_meld_root(None, cfg) == cfg_dir

    # Nothing exists: raises, message names all three mechanisms.
    monkeypatch.delenv("MELD_ROOT", raising=False)
    empty_cfg = OmegaConf.create({"data": {"meld_root": None}})
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_meld_root(None, empty_cfg)
    message = str(excinfo.value)
    assert "--meld-root" in message
    assert "MELD_ROOT" in message
    assert "cfg.data.meld_root" in message


def test_empty_cli_arg_raises_distinct_unexpanded_shell_variable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MELD_ROOT", raising=False)
    cfg = OmegaConf.create({"data": {"meld_root": None}})

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_meld_root("", cfg)

    message = str(excinfo.value)
    assert "unexpanded shell variable" in message

    # Whitespace-only is treated the same way.
    with pytest.raises(FileNotFoundError, match="unexpanded shell variable"):
        resolve_meld_root("   ", cfg)


def test_empty_meld_root_env_var_raises_distinct_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MELD_ROOT", "")
    cfg = OmegaConf.create({"data": {"meld_root": None}})

    with pytest.raises(FileNotFoundError, match="unexpanded shell variable"):
        resolve_meld_root(None, cfg)


def test_not_found_message_contains_cwd_and_all_three_input_reprs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MELD_ROOT", raising=False)
    cfg = OmegaConf.create({"data": {"meld_root": None}})

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_meld_root("./nonexistent", cfg)

    message = str(excinfo.value)
    assert repr(os.getcwd()) in message or os.getcwd() in message
    assert repr("./nonexistent") in message
    assert repr(None) in message  # MELD_ROOT env var and cfg.data.meld_root are both None


def test_print_paths_exits_0_and_writes_no_files(toy_meld_root: Path) -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    before = set(toy_meld_root.rglob("*"))

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "inspect_meld.py"), "--meld-root", str(toy_meld_root), "--print-paths"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Resolved MELD root" in result.stdout
    for split in ("train", "dev", "test"):
        assert split in result.stdout

    after = set(toy_meld_root.rglob("*"))
    assert after == before  # nothing written under the toy MELD root


def test_video_dir_candidates_prefer_nested_paths_first() -> None:
    assert VIDEO_DIR_CANDIDATES["train"][0] == "train/train_splits"
    assert VIDEO_DIR_CANDIDATES["dev"][0] == "dev/dev_splits_complete"
    assert VIDEO_DIR_CANDIDATES["test"][0] == "test/output_repeated_splits_test"
