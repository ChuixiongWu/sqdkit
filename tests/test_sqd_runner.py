"""End-to-end locks for the SQD orchestration pipeline."""

from __future__ import annotations

import math

import pytest

from sqdkit.config import RunConfig
from sqdkit.pyscf_driver import build_problem
from sqdkit.sqd_runner import run
from sqdkit.stateprep import ReferenceProductStatePrep
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile
from sqdkit.validation import fci_reference

H2_STO3G = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]


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
