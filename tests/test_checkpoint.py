"""Checkpoint storage, integrity, and restoration through the real component APIs."""

import json
import pickle

import pytest

from test_training_baseline import environment_factory as _environment_factory
from checkpoint_worker import exact
from autorec.agent import DDQN_ECM
from autorec import checkpoint
from autorec.runtime_state import rng_state_dict, restore_rng_state


environment_factory = _environment_factory


@pytest.fixture
def agent(tmp_path, environment_factory):
    saved = rng_state_dict()
    instance = DDQN_ECM(
        environment_factory(),
        save_dir=tmp_path / "outputs",
        num_trials=12,
        hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
        action_cap=2,
        batch_size=2,
        buffer_capacity=7,
        NN_sleep=3,
        train_frequency=3,
        update_target_frequency=5,
        save_start=100,
    )
    instance.train(additional_trials=0)
    yield instance
    restore_rng_state(saved)


def test_save_read_is_exact_and_rng_inert(agent, tmp_path):
    before = rng_state_dict()
    path = checkpoint.save_checkpoint(agent, tmp_path / "checkpoint")
    payloads = checkpoint._read_checkpoint(path)
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())
    assert (
        payloads["replay.npy"].tobytes()
        == agent._replay_buffer.state_dict()["buffer"].tobytes()
    )
    assert payloads["config.json"]["agent"]["num_trials"] == 12
    assert payloads["config.json"]["agent"]["beta_anneal_trials"] == pytest.approx(7.14)
    assert payloads["training.pkl"]["progress"]["completed_trials"] == 0
    assert payloads["manifest.json"]["series_id"] == agent._checkpoint_series_id
    with pytest.raises(FileExistsError):
        checkpoint.save_checkpoint(agent, path)


def test_reconstructible_performance_caches_are_not_checkpoint_state(agent):
    before = checkpoint._capture(agent)
    agent._flatten_z_matrices = {}
    agent._action_index_maps = {}
    agent._compiled_train_step_signature = ("derived",)
    agent._compiled_train_step = object()

    assert exact(checkpoint._capture(agent)) == exact(before)

    agent.unexpected_state = object()
    with pytest.raises(ValueError, match="unmodified DDQN_ECM"):
        checkpoint._capture(agent)


def test_invalid_checkpoints_are_rejected_before_unpickling(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpoint.pickle, "load", lambda *_: pytest.fail("Unpickled before validation")
    )

    for problem in ("missing", "corrupt", "version", "runtime", "incomplete"):
        path = checkpoint.save_checkpoint(agent, tmp_path / f"checkpoint-{problem}")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if problem == "missing":
            (path / "neural.pkl").unlink()
        elif problem == "corrupt":
            (path / "training.pkl").write_bytes(b"broken")
        elif problem == "version":
            manifest["format_version"] = 100
        elif problem == "runtime":
            manifest["runtime"]["python"] = "incompatible"
        else:
            manifest["files"].pop("replay.npy")
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError):
            checkpoint._read_checkpoint(path)


def test_interrupted_write_preserves_previous_checkpoint(agent, tmp_path, monkeypatch):
    previous = checkpoint.save_checkpoint(agent, tmp_path / "previous")
    original = (previous / "manifest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(checkpoint.pickle, "dump", fail)
    with pytest.raises(OSError, match="interrupted"):
        checkpoint.save_checkpoint(agent, tmp_path / "next")
    assert (previous / "manifest.json").read_bytes() == original
    assert not (tmp_path / "next").exists()
    assert not list(tmp_path.glob(".next-*"))


def test_find_latest_checkpoint_handles_missing_empty_and_valid_roots(agent, tmp_path):
    root = tmp_path / "checkpoints"
    assert checkpoint.find_latest_checkpoint(root, target_trials=10) is None
    root.mkdir()
    (root / "notes.txt").write_text("unrelated")
    (root / ".trial-00000010-interrupted").mkdir()
    assert checkpoint.find_latest_checkpoint(root, target_trials=10) is None

    first = checkpoint.save_checkpoint(agent, root / "trial-00000000")
    agent._training_state.completed_trials = 10
    latest = checkpoint.save_checkpoint(agent, root / "trial-00000010")

    assert checkpoint.find_latest_checkpoint(root, target_trials=10) == latest
    assert checkpoint.find_latest_checkpoint(root) == latest
    assert first.is_dir()


def test_checkpoint_series_is_preserved_and_mixed_roots_are_rejected(
    agent, tmp_path, environment_factory
):
    root = tmp_path / "checkpoints"
    agent._checkpoint_config_fingerprint = "a" * 64
    first = checkpoint.save_checkpoint(agent, root / "trial-00000000")
    first_manifest = json.loads((first / "manifest.json").read_text())
    series_id = first_manifest["series_id"]
    assert first_manifest["config_fingerprint"] == "a" * 64
    restored = checkpoint.load_checkpoint(first, save_dir=tmp_path / "restored")
    assert restored._checkpoint_config_fingerprint == "a" * 64
    restored._training_state.completed_trials = 1
    second = checkpoint.save_checkpoint(restored, root / "trial-00000001")

    assert json.loads((second / "manifest.json").read_text())["series_id"] == series_id
    assert checkpoint.find_latest_checkpoint(root) == second

    second_manifest_path = second / "manifest.json"
    second_manifest = json.loads(second_manifest_path.read_text())
    second_manifest["config_fingerprint"] = "b" * 64
    second_manifest_path.write_text(json.dumps(second_manifest))
    with pytest.raises(ValueError, match="inconsistent configuration fingerprints"):
        checkpoint.find_latest_checkpoint(root)
    second_manifest["config_fingerprint"] = "a" * 64
    second_manifest_path.write_text(json.dumps(second_manifest))

    other = DDQN_ECM(
        environment_factory(),
        save_dir=tmp_path / "other-output",
        num_trials=1,
        hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
    )
    other.train(additional_trials=0)
    other._training_state.completed_trials = 2
    checkpoint.save_checkpoint(other, root / "trial-00000002")
    with pytest.raises(ValueError, match="multiple training series"):
        checkpoint.find_latest_checkpoint(root)


def test_find_latest_checkpoint_rejects_unsafe_candidates(agent, tmp_path):
    for problem in ("malformed", "name-mismatch", "corrupt", "above-target"):
        root = tmp_path / problem
        root.mkdir()
        if problem == "malformed":
            (root / "trial-invalid").mkdir()
            match = "Malformed checkpoint"
        else:
            path = checkpoint.save_checkpoint(agent, root / "trial-00000000")
            if problem == "name-mismatch":
                path.rename(root / "trial-00000001")
                match = "trial counts differ"
            elif problem == "corrupt":
                (path / "replay.npy").write_bytes(b"corrupt")
                match = "corrupt checkpoint component"
            else:
                match = "exceeds target"

        target = -1 if problem == "above-target" else 10
        if problem == "above-target":
            target = 0
            agent._training_state.completed_trials = 1
            checkpoint.save_checkpoint(agent, root / "trial-00000001")
        with pytest.raises(ValueError, match=match):
            checkpoint.find_latest_checkpoint(root, target_trials=target)


@pytest.mark.parametrize("completed", [None, 4, 10])
def test_plan_training_recovery_is_cumulative_and_side_effect_free(agent, tmp_path, completed):
    root = tmp_path / "checkpoints"
    if completed is not None:
        agent._training_state.completed_trials = completed
        checkpoint.save_checkpoint(agent, root / f"trial-{completed:08d}")
    before = rng_state_dict()

    plan = checkpoint.plan_training_recovery(root, target_trials=10)

    expected_checkpoint = None if completed is None else root / f"trial-{completed:08d}"
    assert plan.checkpoint == expected_checkpoint
    assert plan.completed_trials == (0 if completed is None else completed)
    assert plan.additional_trials == 10 - (0 if completed is None else completed)
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())


@pytest.mark.parametrize(
    ("target", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_plan_training_recovery_rejects_invalid_targets(tmp_path, target, error):
    with pytest.raises(error, match="target_trials"):
        checkpoint.plan_training_recovery(tmp_path, target)


def test_unfinished_trial_rejected(agent, tmp_path):
    agent._training_state.at_trial_boundary = False
    with pytest.raises(ValueError, match="boundary"):
        checkpoint.save_checkpoint(agent, tmp_path / "checkpoint")


def test_restore_before_optimizer_slots_and_relocate(agent, tmp_path):
    path = checkpoint.save_checkpoint(agent, tmp_path / "checkpoint")
    restored = checkpoint.load_checkpoint(path, save_dir=tmp_path / "relocated", num_trials=3)
    assert restored.completed_trials == 0
    assert restored.training_env is restored.eval_env
    assert restored.optimizer.built is False
    assert restored.save_dir == tmp_path / "relocated"
    assert restored.num_trials == 3
    assert restored._continuation_mode is True
    assert exact(checkpoint._capture(agent)["training.pkl"]) == exact(
        checkpoint._capture(restored)["training.pkl"]
    )


def test_invalid_restored_session_lengths_are_rng_inert(agent, tmp_path):
    path = checkpoint.save_checkpoint(agent, tmp_path / "checkpoint")

    for value in (0, -1, True, 1.5, "3"):
        before = rng_state_dict()
        error = TypeError if value is True or not isinstance(value, int) else ValueError

        with pytest.raises(error, match="num_trials"):
            checkpoint.load_checkpoint(path, num_trials=value)

        assert pickle.dumps(before) == pickle.dumps(rng_state_dict())


def test_failed_reconstruction_restores_caller_rng(agent, tmp_path, monkeypatch):
    path = checkpoint.save_checkpoint(agent, tmp_path / "checkpoint")
    before = rng_state_dict()

    def fail(*args, **kwargs):
        import random
        import numpy as np

        random.random()
        np.random.random()
        raise ValueError("simulated neural restoration failure")

    monkeypatch.setattr(checkpoint, "restore_neural_state", fail)
    with pytest.raises(ValueError, match="simulated"):
        checkpoint.load_checkpoint(path)
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())


def test_invalid_checkpoint_intervals_do_not_advance(agent, tmp_path):
    for every in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            agent.train(additional_trials=1, checkpoint_dir=tmp_path, checkpoint_every=every)
        assert agent.completed_trials == 0


def test_checkpoint_interval_requires_directory(agent):
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        agent.train(checkpoint_every=1)
    assert agent.completed_trials == 0


def test_public_checkpoint_api_accepts_progress_beyond_session_length(agent, tmp_path):
    agent._training_state.completed_trials = agent.num_trials + 1
    path = agent.save_checkpoint(tmp_path / "checkpoint")
    restored = DDQN_ECM.from_checkpoint(path, save_dir=tmp_path / "resumed")
    before = exact(checkpoint._capture(restored))
    restored.train(additional_trials=0)
    assert exact(checkpoint._capture(restored)) == before
