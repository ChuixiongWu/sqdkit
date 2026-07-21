"""Persistence and validation tests for SQD results."""

from __future__ import annotations

import json

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
    # Given: a result artifact whose energy array requires pickle deserialization.
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
