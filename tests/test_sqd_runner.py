"""End-to-end locks for the SQD orchestration pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sqdkit.config import RunConfig
from sqdkit.pyscf_driver import build_problem
from sqdkit.results import SQDResult
from sqdkit.sqd_runner import run
from sqdkit.stateprep import ReferenceProductStatePrep
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile
from sqdkit.validation import fci_reference

H2_STO3G = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]

H2_TODO5_TOTAL_ENERGY = -1.1373060357534004


@pytest.fixture(scope="module")
def h2_problem():
    """Provide a small correlated problem with CCSD amplitudes for LUCJ."""
    return build_problem(H2_STO3G, "sto-3g")


def _config(*, max_iterations: int = 3) -> RunConfig:
    return RunConfig(
        shots=1_000,
        samples_per_batch=100,
        num_batches=1,
        max_iterations=max_iterations,
        seed=42,
    )


def test_run_matches_h2_fci_with_lucj_state_preparation(h2_problem) -> None:
    # Given: an H2/STO-3G problem, a correlated LUCJ state, and ideal sampling.
    problem = h2_problem

    # When: the complete SQD pipeline runs.
    result = run(problem, LucjStatePrep(n_reps=1), BackendProfile.ideal(), _config())

    # Then: its total energy locks the integral and bit-ordering conventions to FCI.
    assert abs(result.total_energy - fci_reference(problem)) < 1e-3


def test_run_diagonalizes_without_lucj(h2_problem) -> None:
    # Given: the same problem but a Qiskit-only Hartree--Fock product state.
    problem = h2_problem
    stateprep = ReferenceProductStatePrep(norb=2, n_alpha=1, n_beta=1)

    # When: the complete SQD pipeline runs without LUCJ preparation.
    result = run(problem, stateprep, BackendProfile.ideal(), _config())

    # Then: SQD returns a meaningful energy independently of ffsim's LUCJ circuit.
    assert math.isfinite(result.total_energy)
    assert result.total_energy != 0.0


def test_run_surfaces_non_convergence_at_iteration_limit(h2_problem) -> None:
    # Given: a valid reference-state workflow restricted to one recovery iteration.
    problem = h2_problem
    stateprep = ReferenceProductStatePrep(norb=2, n_alpha=1, n_beta=1)

    # When: the iteration budget ends before a convergence comparison is possible.
    result = run(
        problem,
        stateprep,
        BackendProfile.ideal(),
        _config(max_iterations=1),
    )

    # Then: the result explicitly records that the run did not converge.
    assert result.converged is False


def test_h2_ideal_surfaces_rdm1_and_occupancies_energy_unchanged(
    h2_problem, tmp_path
) -> None:
    # Given: the todo-5 baseline H2 workflow (smoke-example config).
    problem = h2_problem
    config = RunConfig(
        shots=1000,
        samples_per_batch=50,
        num_batches=2,
        max_iterations=5,
        seed=42,
    )

    # When: the complete SQD pipeline runs.
    result = run(problem, LucjStatePrep(n_reps=1), BackendProfile.ideal(), config)

    # Then: the plumbing changed nothing numerically.
    assert (
        abs(result.total_energy - H2_TODO5_TOTAL_ENERGY) < 1e-8
    )

    # And: the upstream 1-RDM is surfaced in the active-space MO basis.
    assert result.rdm1 is not None
    assert result.rdm1.shape == (problem.norb, problem.norb)

    # And: it traces to the active electron count (the decisive physics check).
    assert (
        abs(float(np.trace(result.rdm1)) - (problem.n_alpha + problem.n_beta))
        < 1e-6
    )

    # And: it is symmetric.
    assert np.max(np.abs(result.rdm1 - result.rdm1.T)) < 1e-8

    # And: per-spin occupancies lie in [0, 1] and sum to the electron counts.
    assert result.orbital_occupancies is not None
    assert result.orbital_occupancies.shape == (2, problem.norb)
    assert np.all(result.orbital_occupancies >= -1e-8)
    assert np.all(result.orbital_occupancies <= 1.0 + 1e-8)
    assert (
        abs(float(result.orbital_occupancies[0].sum()) - problem.n_alpha) < 1e-6
    )
    assert (
        abs(float(result.orbital_occupancies[1].sum()) - problem.n_beta) < 1e-6
    )

    # And: the 2-RDM (the documented density-rebuild piece) is surfaced too.
    assert result.rdm2 is not None
    assert result.rdm2.shape == (problem.norb,) * 4

    # And: save/load round-trips the densities within 1e-12.
    result.save(tmp_path / "h2_ideal")
    loaded = SQDResult.load(tmp_path / "h2_ideal")
    assert np.allclose(loaded.rdm1, result.rdm1, rtol=0.0, atol=1e-12)
    assert np.allclose(
        loaded.orbital_occupancies, result.orbital_occupancies, rtol=0.0, atol=1e-12
    )
    assert np.allclose(loaded.rdm2, result.rdm2, rtol=0.0, atol=1e-12)
