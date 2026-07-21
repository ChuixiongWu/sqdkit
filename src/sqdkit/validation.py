"""Optional exact references for validating small chemistry problems."""

from pyscf import fci

from sqdkit.problem import ProblemData


def fci_reference(problem: ProblemData, *, max_norb: int = 8) -> float:
    """Return the total PySCF FCI energy for a small spatial-orbital problem."""
    if problem.norb > max_norb:
        raise ValueError(
            f"FCI reference is limited to {max_norb} orbitals; received {problem.norb}"
        )

    cisolver = fci.direct_spin1.FCI()
    electronic_energy, _ = cisolver.kernel(
        problem.h1,
        problem.h2,
        problem.norb,
        (problem.n_alpha, problem.n_beta),
    )
    return float(electronic_energy) + problem.core_energy
