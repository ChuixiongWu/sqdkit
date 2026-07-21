import json
from pathlib import Path

import pytest

from sqdkit.config import AdvancedConfig, RunConfig, derived_seeds


def test_run_config_round_trips_and_derives_stage_seeds() -> None:
    # Given: a valid run configuration with serializable options.
    config = RunConfig(
        input_path="inputs/h2.npz",
        stateprep_options={"reps": 2},
        backend_profile_options={"optimization_level": 1},
        shots=1_024,
        samples_per_batch=64,
        num_batches=4,
        max_iterations=8,
        seed=11,
    )

    # When: its serialized form is loaded into a new configuration.
    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    # Then: the configuration and deterministically derived seeds are retained.
    assert restored == config
    assert derived_seeds(config.seed) == {"transpiler": 11, "sampler": 12, "sqd": 13}


def test_advanced_config_round_trips_tuple_max_dim() -> None:
    # Given: advanced diagonalization controls containing a tuple max dimension.
    config = AdvancedConfig(max_dim=(100, 50), spin_sq=0.0, symmetrize_spin=True)

    # When: its JSON-compatible dictionary is restored.
    restored = AdvancedConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    # Then: the typed advanced configuration is retained.
    assert restored == config


def test_run_config_rejects_non_positive_shots() -> None:
    # Given: an otherwise valid configuration with zero shots.
    # When / Then: construction identifies the invalid field.
    with pytest.raises(ValueError, match="shots"):
        RunConfig(shots=0)


def test_run_config_rejects_unknown_backend_profile() -> None:
    # Given: an otherwise valid configuration with an unsupported profile.
    # When / Then: construction identifies the invalid field.
    with pytest.raises(ValueError, match="backend_profile"):
        RunConfig(backend_profile="unsupported")


def test_run_config_serializes_path_input() -> None:
    # Given: a configuration initialized with a Path.
    config = RunConfig(input_path=Path("inputs/h2.npz"))

    # When: it is represented as a dictionary.
    data = config.to_dict()

    # Then: the path is JSON-serializable.
    assert data["input_path"] == "inputs/h2.npz"
