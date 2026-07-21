"""PySCF-based construction of canonical spatial-orbital problems."""

from collections.abc import Sequence

import numpy as np
from pyscf import ao2mo, cc, gto, scf

from sqdkit.problem import ProblemData


def build_problem(
    atom: str | list[tuple[str, tuple[float, float, float]]],
    basis: str,
    charge: int = 0,
    spin: int = 0,
    active_space: list[int] | None = None,
    with_ccsd: bool = True,
) -> ProblemData:
    """Build a canonical CHEMIST-integral problem from a PySCF RHF calculation."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin)
    mf = scf.RHF(mol)
    mf.kernel()

    c_matrix = mf.mo_coeff
    h1_mo = c_matrix.T @ mf.get_hcore() @ c_matrix
    norb = mol.nao_nr()
    eri = mol.intor("int2e")
    h2_mo = ao2mo.incore.full(eri, c_matrix).reshape(norb, norb, norb, norb)
    n_alpha, n_beta = mol.nelec
    core_energy = float(mol.energy_nuc())

    selected_orbitals = _active_orbitals(active_space, norb)
    active_occupied: list[int] | None = None
    active_virtual: list[int] | None = None
    if selected_orbitals is not None:
        active_occupied = [
            orbital for orbital in selected_orbitals if orbital < n_alpha
        ]
        active_virtual = [
            orbital - n_alpha for orbital in selected_orbitals if orbital >= n_alpha
        ]
        frozen_orbitals = [
            orbital for orbital in range(n_alpha) if orbital not in selected_orbitals
        ]
        for frozen_orbital in frozen_orbitals:
            core_energy += 2.0 * h1_mo[frozen_orbital, frozen_orbital]
            for other_frozen_orbital in frozen_orbitals:
                core_energy += 2.0 * h2_mo[
                    frozen_orbital,
                    frozen_orbital,
                    other_frozen_orbital,
                    other_frozen_orbital,
                ] - h2_mo[
                    frozen_orbital,
                    other_frozen_orbital,
                    other_frozen_orbital,
                    frozen_orbital,
        ]
        h1_mo = h1_mo[np.ix_(selected_orbitals, selected_orbitals)]
        for frozen_orbital in frozen_orbitals:
            coulomb = h2_mo[
                np.ix_(
                    selected_orbitals,
                    selected_orbitals,
                    [frozen_orbital],
                    [frozen_orbital],
                )
            ][:, :, 0, 0]
            exchange = h2_mo[
                np.ix_(
                    selected_orbitals,
                    [frozen_orbital],
                    [frozen_orbital],
                    selected_orbitals,
                )
            ][:, 0, 0, :]
            h1_mo += 2.0 * coulomb - exchange
        h2_mo = h2_mo[
            np.ix_(
                selected_orbitals,
                selected_orbitals,
                selected_orbitals,
                selected_orbitals,
            )
        ]
        n_alpha = len(active_occupied)
        n_beta = sum(orbital < n_beta for orbital in selected_orbitals)
        norb = len(selected_orbitals)

    provenance: dict[str, str | int | bool | list[int] | list[list[float]]] = {
        "source": "pyscf",
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "active_space": selected_orbitals if selected_orbitals is not None else [],
        "with_ccsd": with_ccsd,
    }
    if with_ccsd:
        coupled_cluster = cc.CCSD(mf)
        coupled_cluster.kernel()
        t1 = coupled_cluster.t1
        t2 = coupled_cluster.t2
        if active_occupied is not None and active_virtual is not None:
            t1 = t1[np.ix_(active_occupied, active_virtual)]
            t2 = t2[
                np.ix_(
                    active_occupied,
                    active_occupied,
                    active_virtual,
                    active_virtual,
                )
            ]
        provenance["t1"] = t1.tolist()
        provenance["t2"] = t2.tolist()

    return ProblemData(
        h1=np.asarray((h1_mo + h1_mo.T) / 2, dtype=np.float64),
        h2=np.asarray(h2_mo, dtype=np.float64),
        core_energy=core_energy,
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        integral_convention="chemist",
        orbital_ordering="spin_blocks:alpha[0:norb),beta[norb:2norb)",
        provenance=provenance,
    )


def _active_orbitals(
    active_space: Sequence[int] | None, norb: int
) -> list[int] | None:
    """Validate and normalize an optional spatial-orbital selection."""
    if active_space is None:
        return None
    orbitals = list(active_space)
    if len(orbitals) > norb:
        raise ValueError("active_space cannot contain more orbitals than the molecule")
    if len(set(orbitals)) != len(orbitals):
        raise ValueError("active_space orbital indices must be unique")
    if any(orbital < 0 or orbital >= norb for orbital in orbitals):
        raise ValueError("active_space orbital indices must be within the MO range")
    return orbitals
