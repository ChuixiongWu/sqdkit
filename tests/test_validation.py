import math

import pytest

from sqdkit.pyscf_driver import build_problem
from sqdkit.validation import fci_reference


H2_STO3G = [["H", (0, 0, 0)], ["H", (0, 0, 0.735)]]


def test_fci_reference_returns_sane_h2_total_energy() -> None:
    # Given
    problem = build_problem(H2_STO3G, "sto-3g", with_ccsd=False)

    # When
    energy = fci_reference(problem)

    # Then
    assert math.isfinite(energy)
    assert -1.5 < energy < 0.0


def test_fci_reference_rejects_problem_above_orbital_limit() -> None:
    # Given
    problem = build_problem(H2_STO3G, "sto-3g", with_ccsd=False)

    # When / Then
    with pytest.raises(ValueError, match=r"1.*2"):
        fci_reference(problem, max_norb=1)
