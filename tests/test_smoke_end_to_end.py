"""Fast end-to-end coverage for reproducible local SQD workflows."""

from __future__ import annotations

import math

from sqdkit.config import RunConfig
from sqdkit.pyscf_driver import build_problem
from sqdkit.sqd_runner import run as sqd_run
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile

H2_STO3G = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]


def _fast_config() -> RunConfig:
    return RunConfig(
        shots=100,
        samples_per_batch=10,
        num_batches=1,
        max_iterations=2,
        seed=42,
    )


def test_h2_pipeline_is_exactly_reproducible_for_a_fixed_seed() -> None:
    # Given: a deterministic H2 workflow with ideal local simulation.
    problem = build_problem(H2_STO3G, "sto-3g")
    stateprep = LucjStatePrep(n_reps=1)
    profile = BackendProfile.ideal()

    # When: the complete pipeline runs twice with the same seed.
    first_result = sqd_run(problem, stateprep, profile, _fast_config())
    second_result = sqd_run(problem, stateprep, profile, _fast_config())

    # Then: the total energy is bit-for-bit reproducible.
    assert first_result.total_energy == second_result.total_energy


def test_h2_pipeline_returns_a_sane_energy_with_synthetic_noise() -> None:
    # Given: an H2 workflow with a simple topology-aware noise profile.
    problem = build_problem(H2_STO3G, "sto-3g")
    stateprep = LucjStatePrep(n_reps=1)
    profile = BackendProfile.synthetic_noise(
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
        ["rz", "sx", "x", "cx"],
        gate_error=0.001,
        readout_error=0.01,
        seed=42,
    )

    # When: the complete pipeline runs with finite-shot noise.
    result = sqd_run(problem, stateprep, profile, _fast_config())

    # Then: SQD returns a finite chemical-scale energy.
    assert math.isfinite(result.total_energy)
    assert -5.0 < result.total_energy < 5.0
