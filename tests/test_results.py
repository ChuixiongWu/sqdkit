"""Persistence and validation tests for SQD results."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sqdkit.results import RunMetadata, SQDResult


def _result() -> SQDResult:
    return SQDResult(
        total_energy=-1.25,
        electronic_energy=-2.0,
        core_energy=0.75,
        converged=np.bool_(True),
        iterations=4,
        subspace_summary={"4": {"dimension": 12}},
        shots_total=1_000,
        valid_count=900,
        unique_count=80,
        stateprep_summary={"method": "lucj", "circuit_depth": 9},
        compiled_summary={"depth": 15},
        backend_profile_mode="ideal",
        backend_profile_fingerprint="abc123",
        effective_config={"norb": 4},
        seeds={"transpiler": 1, "sampler": 2, "sqd": 3},
        elapsed_seconds=1.5,
    )


def test_result_round_trips_energies_seeds_and_package_versions(tmp_path) -> None:
    # Given: a complete SQD result.
    result = _result()
    path = tmp_path / "run"

    # When: it is saved and loaded through the reproducible artifact format.
    result.save(path)
    loaded = SQDResult.load(path)

    # Then: persisted diagnostics and creation metadata are retained.
    assert loaded.total_energy == pytest.approx(-1.25)
    assert loaded.seeds == {"transpiler": 1, "sampler": 2, "sqd": 3}
    assert loaded.metadata.package_versions == result.metadata.package_versions


def test_load_rejects_pickle_backed_energy_array(tmp_path) -> None:
    path = tmp_path / "unsafe"
    np.savez(
        path.with_suffix(".npz"),
        total_energy=np.array(1.0, dtype=object),
        electronic_energy=np.array(-2.0, dtype=np.float64),
        core_energy=np.array(0.5, dtype=np.float64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "converged": True,
                "iterations": 1,
                "subspace_summary": {},
                "shots_total": 1,
                "valid_count": 1,
                "unique_count": 1,
                "stateprep_summary": {},
                "compiled_summary": {},
                "backend_profile_mode": "ideal",
                "backend_profile_fingerprint": "fingerprint",
                "effective_config": {},
                "seeds": {"transpiler": 1, "sampler": 2, "sqd": 3},
                "elapsed_seconds": 0.1,
                "metadata": {
                    "schema_version": 1,
                    "seeds": {"transpiler": 1, "sampler": 2, "sqd": 3},
                    "package_versions": {
                        "qiskit": "1",
                        "qiskit_addon_sqd": "1",
                        "ffsim": "1",
                        "qiskit_aer": "1",
                        "pyscf": "1",
                    },
                    "backend_profile_mode": "ideal",
                    "backend_profile_fingerprint": "fingerprint",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            }
        )
    )

    # When / Then: load refuses to unpickle the object array.
    with pytest.raises(ValueError, match="allow_pickle=False"):
        SQDResult.load(path)


def test_run_metadata_rejects_missing_seeds() -> None:
    # Given / When / Then: required top-level metadata cannot be omitted.
    with pytest.raises(ValueError, match="seeds"):
        RunMetadata(
            package_versions={
                "qiskit": "1",
                "qiskit_addon_sqd": "1",
                "ffsim": "1",
                "qiskit_aer": "1",
                "pyscf": "1",
            },
            backend_profile_mode="ideal",
            backend_profile_fingerprint="fingerprint",
            created_at="2026-01-01T00:00:00+00:00",
        )


def test_run_metadata_validates_all_required_keys() -> None:
    # Given: otherwise complete metadata with one missing nested seed.
    metadata = RunMetadata(
        seeds={"transpiler": 1, "sampler": 2, "sqd": 3},
        package_versions={
            "qiskit": "1",
            "qiskit_addon_sqd": "1",
            "ffsim": "1",
            "qiskit_aer": "1",
            "pyscf": "1",
        },
        backend_profile_mode="ideal",
        backend_profile_fingerprint="fingerprint",
        created_at="2026-01-01T00:00:00+00:00",
    )

    # When / Then: validation succeeds only while every required key exists.
    metadata.validate()


def test_result_round_trips_rdm1_and_occupancies_within_1e12(tmp_path) -> None:
    # Given: a complete result carrying a physically consistent 1-RDM
    # (symmetric, trace = active electron count) and occupancies.
    rdm1 = np.array([[1.5, -0.2], [-0.2, 0.5]], dtype=np.float64)
    occupancies = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = replace(_result(), rdm1=rdm1, orbital_occupancies=occupancies)
    path = tmp_path / "run"

    # When: it is saved and loaded through the reproducible artifact format.
    result.save(path)
    loaded = SQDResult.load(path)

    # Then: the density arrays survive bit-close, and the JSON records shapes.
    assert loaded.rdm1 is not None
    assert np.allclose(loaded.rdm1, rdm1, rtol=0.0, atol=1e-12)
    assert np.allclose(
        loaded.orbital_occupancies, occupancies, rtol=0.0, atol=1e-12
    )
    assert loaded.rdm2 is None
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["rdm1_shape"] == [2, 2]
    assert sidecar["orbital_occupancies_shape"] == [2, 2]
    assert sidecar["orbital_occupancies_summary"]["alpha_sum"] == pytest.approx(1.0)


def test_load_accepts_stale_artifact_without_density_fields(tmp_path) -> None:
    # Given: an artifact written by the pre-extension schema (NPZ with only
    # the three energy scalars, JSON without any rdm/occupancy keys).
    path = tmp_path / "stale"
    np.savez(
        path.with_suffix(".npz"),
        total_energy=np.array(-1.1373, dtype=np.float64),
        electronic_energy=np.array(-2.0, dtype=np.float64),
        core_energy=np.array(0.8627, dtype=np.float64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "total_energy": -1.1373,
                "electronic_energy": -2.0,
                "core_energy": 0.8627,
                "converged": True,
                "iterations": 3,
                "subspace_summary": {},
                "shots_total": 1000,
                "valid_count": 1000,
                "unique_count": 3,
                "stateprep_summary": {},
                "compiled_summary": {},
                "backend_profile_mode": "ideal",
                "backend_profile_fingerprint": "fingerprint",
                "effective_config": {},
                "seeds": {"transpiler": 1, "sampler": 2, "sqd": 3},
                "elapsed_seconds": 0.4,
                "metadata": {
                    "schema_version": 1,
                    "seeds": {"transpiler": 1, "sampler": 2, "sqd": 3},
                    "package_versions": {
                        "qiskit": "1",
                        "qiskit_addon_sqd": "1",
                        "ffsim": "1",
                        "qiskit_aer": "1",
                        "pyscf": "1",
                    },
                    "backend_profile_mode": "ideal",
                    "backend_profile_fingerprint": "fingerprint",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            }
        )
    )

    # When / Then: it loads, with every new density field defaulting to None.
    loaded = SQDResult.load(path)
    assert loaded.rdm1 is None
    assert loaded.rdm2 is None
    assert loaded.orbital_occupancies is None
    assert loaded.total_energy == pytest.approx(-1.1373)


def test_result_rejects_rdm1_with_wrong_electron_count() -> None:
    # Given: a symmetric 1-RDM whose trace (3.0) contradicts the occupancies
    # (2.0 active electrons).
    rdm1 = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    occupancies = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    # When / Then: the physics validation rejects the inconsistent density.
    with pytest.raises(ValueError, match="active electron count"):
        replace(_result(), rdm1=rdm1, orbital_occupancies=occupancies)


def test_result_rejects_non_symmetric_rdm1() -> None:
    # Given: a square 1-RDM that is not symmetric.
    rdm1 = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)

    # When / Then: validation rejects it.
    with pytest.raises(ValueError, match="symmetric"):
        replace(_result(), rdm1=rdm1)


def test_result_rejects_occupancies_outside_unit_interval() -> None:
    # Given: an orbital population above 1 (impossible for one spin channel).
    occupancies = np.array([[1.5, 0.0], [0.0, 1.0]], dtype=np.float64)

    # When / Then: validation rejects it.
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(_result(), orbital_occupancies=occupancies)


def test_result_rejects_non_square_rdm1() -> None:
    # Given: a 1-RDM whose shape is not (norb, norb).
    rdm1 = np.zeros((3, 2), dtype=np.float64)

    # When / Then: validation rejects it.
    with pytest.raises(ValueError, match="square"):
        replace(_result(), rdm1=rdm1)
