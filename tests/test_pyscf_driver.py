import numpy as np
import pytest

from sqdkit.problem import ProblemData
from sqdkit.pyscf_driver import build_problem

H2_STO3G = [["H", (0, 0, 0)], ["H", (0, 0, 0.735)]]


def test_build_problem_returns_round_trippable_h2_data(tmp_path):
    # Given
    output_path = tmp_path / "h2.npz"

    # When
    problem = build_problem(H2_STO3G, "sto-3g")
    problem.save_npz(output_path)
    loaded = ProblemData.load_npz(output_path)

    # Then
    assert problem.norb == 2
    assert (problem.n_alpha, problem.n_beta) == (1, 1)
    assert np.array_equal(problem.h1, problem.h1.T)
    assert problem.core_energy > 0
    assert problem.h2.shape == (2, 2, 2, 2)
    assert np.asarray(problem.provenance["t2"]).shape == (1, 1, 1, 1)
    assert np.array_equal(loaded.h1, problem.h1)
    assert np.array_equal(loaded.h2, problem.h2)


def test_build_problem_rejects_active_space_larger_than_orbitals():
    # Given
    active_space = [0, 1, 2]

    # When / Then
    with pytest.raises(ValueError, match="active_space"):
        build_problem(H2_STO3G, "sto-3g", active_space=active_space, with_ccsd=False)
