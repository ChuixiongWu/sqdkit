"""Slow active-space integration coverage for the N2 SQD workflow."""

from __future__ import annotations

import math

import pytest

from sqdkit.config import RunConfig
from sqdkit.pyscf_driver import build_problem
from sqdkit.sqd_runner import run as sqd_run
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile

N2_6_31G = [("N", (0, 0, 0)), ("N", (1.0, 0, 0))]


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_n2_active_space_pipeline_returns_a_sane_energy() -> None:
    # Given: a 10-electron, 8-orbital N2 active space and ideal local SQD.
    problem = build_problem(N2_6_31G, "6-31g", active_space=list(range(2, 10)))
    config = RunConfig(
        shots=10_000,
        samples_per_batch=1_000,
        num_batches=1,
        max_iterations=3,
        seed=42,
    )

    # When: LUCJ preparation and SQD run with the fixed seed.
    result = sqd_run(
        problem,
        LucjStatePrep(n_reps=2),
        BackendProfile.ideal(),
        config,
    )

    # Then: the finite active-space estimate remains in the N2 chemical range.
    assert math.isfinite(result.total_energy)
    assert -109.0 < result.total_energy < -107.0
