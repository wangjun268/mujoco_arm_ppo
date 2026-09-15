"""Tests for the central path helpers and the shared CLI defaults."""

import argparse
import os

import pytest

import cli
from paths import (
    ASSETS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    asset_path,
    default_model_path,
    default_out_dir,
    default_tb_dir,
    ensure_dir,
    resolve_model,
    results_path,
)


def test_paths_point_at_the_checkout():
    assert os.path.isdir(PROJECT_ROOT)
    assert ASSETS_DIR == os.path.join(PROJECT_ROOT, "assets")
    assert RESULTS_DIR == os.path.join(PROJECT_ROOT, "results")
    assert os.path.isfile(asset_path("two_joint_arm.xml"))
    assert results_path("x") == os.path.join(RESULTS_DIR, "x")


def test_default_workspace_paths():
    assert default_model_path("two_joint") == results_path("ppo_two_joint")
    assert default_tb_dir("two_joint") == results_path("tb_two_joint")
    assert default_out_dir("two_joint") == results_path("two_joint")


def test_ensure_dir_creates_missing_directories(tmp_path):
    target = tmp_path / "a" / "b"
    assert ensure_dir(str(target)) == str(target)
    assert target.is_dir()


def test_resolve_model_accepts_a_checkpoint(tmp_path):
    model = tmp_path / "ppo_x.zip"
    model.write_bytes(b"")
    assert resolve_model(str(model)) == str(model)
    # A bare name gets the .zip suffix appended.
    assert resolve_model(str(tmp_path / "ppo_x")) == str(model)


def test_resolve_model_rejects_directories(tmp_path):
    with pytest.raises(SystemExit):
        resolve_model(str(tmp_path))


def test_resolve_model_rejects_missing_files(tmp_path):
    with pytest.raises(SystemExit):
        resolve_model(str(tmp_path / "nope"))


def test_cli_fills_defaults_from_env():
    parser = argparse.ArgumentParser()
    cli.add_env_arg(parser)
    cli.add_model_arg(parser)
    cli.add_out_arg(parser)
    args = cli.apply_defaults(parser.parse_args(["--env", "three_joint"]))
    assert args.model == default_model_path("three_joint")
    assert args.out == default_out_dir("three_joint")


def test_cli_rejects_unknown_env():
    parser = argparse.ArgumentParser()
    cli.add_env_arg(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--env", "not_an_env"])
