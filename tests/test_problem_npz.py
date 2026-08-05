import json

import numpy as np
import pytest

from sqdkit.problem import ProblemData


def test_round_trip_preserves_problem_data_without_object_arrays(tmp_path):
    # Given
    h1 = np.array([[1.0, 0.25], [0.25, -0.5]], dtype=np.float64)
    h2 = np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2)
    problem = ProblemData(
        h1=h1,
        h2=h2,
        core_energy=1.25,
        norb=2,
        n_alpha=1,
        n_beta=1,
        integral_convention="chemist",
        orbital_ordering="spin_blocks:alpha[0:norb),beta[norb:2norb)",
        provenance={"source": "test", "active_orbitals": [0, 1]},
    )
    path = tmp_path / "problem.npz"

    # When
    problem.save_npz(path)
    loaded = ProblemData.load_npz(path)

    # Then
    assert np.array_equal(loaded.h1, h1)
    assert np.array_equal(loaded.h2, h2)
    assert loaded.core_energy == 1.25
    assert loaded.provenance == {"source": "test", "active_orbitals": [0, 1]}
    with np.load(path, allow_pickle=False) as archive:
        assert all(array.dtype != np.dtype(object) for array in archive.values())


def test_load_raises_when_h2_is_missing(tmp_path):
    # Given
    path = tmp_path / "missing-h2.npz"
    np.savez(
        path,
        h1=np.eye(1, dtype=np.float64),
        core_energy=np.array(0.0, dtype=np.float64),
        norb=np.array(1, dtype=np.int64),
        n_alpha=np.array(1, dtype=np.int64),
        n_beta=np.array(0, dtype=np.int64),
        integral_convention=np.array("chemist"),
        orbital_ordering=np.array("spin_blocks:alpha[0:norb),beta[norb:2norb)"),
        schema_version=np.array(1, dtype=np.int64),
        metadata_json=np.array(json.dumps({})),
    )

    # When / Then
    with pytest.raises((KeyError, ValueError)):
        ProblemData.load_npz(path)


def test_load_raises_for_physicist_integral_convention(tmp_path):
    # Given
    path = tmp_path / "physicist.npz"
    np.savez(
        path,
        h1=np.eye(1, dtype=np.float64),
        h2=np.zeros((1, 1, 1, 1), dtype=np.float64),
        core_energy=np.array(0.0, dtype=np.float64),
        norb=np.array(1, dtype=np.int64),
        n_alpha=np.array(1, dtype=np.int64),
        n_beta=np.array(0, dtype=np.int64),
        integral_convention=np.array("physicist"),
        orbital_ordering=np.array("spin_blocks:alpha[0:norb),beta[norb:2norb)"),
        schema_version=np.array(1, dtype=np.int64),
        metadata_json=np.array(json.dumps({})),
    )

    # When / Then
    with pytest.raises(ValueError, match="chemist"):
        ProblemData.load_npz(path)


def test_load_npz_rejects_object_array_archive(tmp_path):
    # Given: an NPZ whose metadata_json is an object array (pickle payload)
    path = tmp_path / "object-array.npz"
    np.savez(
        path,
        h1=np.eye(2, dtype=np.float64),
        h2=np.zeros((2, 2, 2, 2), dtype=np.float64),
        core_energy=np.array(0.0, dtype=np.float64),
        norb=np.array(2, dtype=np.int64),
        n_alpha=np.array(1, dtype=np.int64),
        n_beta=np.array(1, dtype=np.int64),
        integral_convention=np.array("chemist"),
        orbital_ordering=np.array("spin_blocks:alpha[0:norb),beta[norb:2norb)"),
        schema_version=np.array(1, dtype=np.int64),
        metadata_json=np.array([{"provenance": np.float64(1.0)}], dtype=object),
    )

    # When / Then: load_npz opens with allow_pickle=False, so numpy refuses
    with pytest.raises(ValueError):
        ProblemData.load_npz(path)
