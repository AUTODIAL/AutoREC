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
