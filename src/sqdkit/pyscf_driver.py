"""PySCF-based construction of canonical spatial-orbital problems.

Closed shell (``spin=0``) runs RHF.  Open shell (``spin != 0``): PySCF's
``scf.RHF(mol)`` silently returns an ROHF object instead (MRO
``ROHF -> RHF -> SCF``), which is exactly what we want -- the single
``(nao, nmo)`` ROHF ``mo_coeff`` keeps the integral transform unchanged.
``cc.CCSD`` on that ROHF reference converts it to UHF (pyscf/cc/__init__.py:
100-110; the "RCCSD method does not support ROHF" warning is expected and
correct) and returns a UCCSD whose amplitudes are tuples
``t1 = (t1a, t1b)`` and ``t2 = (t2aa, t2ab, t2bb)``; those are sliced per
spin block and stored as nested lists.
"""

from collections.abc import Sequence

import numpy as np
from pyscf import ao2mo, cc, gto, scf

from sqdkit.problem import ProblemData


def build_problem(
    atom: str | list[tuple[str, tuple[float, float, float]]],
    basis: str,
    charge: int = 0,
    spin: int = 0,
    ecp: str | dict | None = None,
    active_space: list[int] | None = None,
    with_ccsd: bool = True,
) -> ProblemData:
    """Build a canonical CHEMIST-integral problem from a PySCF SCF calculation."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, ecp=ecp)
    mf = scf.RHF(mol)
    mf.kernel()

    c_matrix = mf.mo_coeff
    h1_mo = c_matrix.T @ mf.get_hcore() @ c_matrix
    norb = mol.nao_nr()
    eri = mol.intor("int2e")
    h2_mo = ao2mo.incore.full(eri, c_matrix).reshape(norb, norb, norb, norb)
    n_alpha, n_beta = mol.nelec
    # Per-spatial-orbital occupation: 2.0 = doubly occupied, 1.0 = singly
    # occupied (alpha, the ROHF convention), 0.0 = virtual.  This drives the
    # frozen-core energy so a singly occupied frozen orbital contributes
    # exactly one electron instead of the closed-shell 2.0 factor.
    mo_occ = np.asarray(mf.mo_occ, dtype=np.float64)
    core_energy = float(mol.energy_nuc())

    selected_orbitals = _active_orbitals(active_space, norb)
    active_occupied: list[int] | None = None
    active_virtual: list[int] | None = None
    # Beta-block indices differ from alpha for open shell (n_beta < n_alpha);
    # they slice the unrestricted CCSD amplitudes spin-by-spin below.  They
    # are built from the FULL mol.nelec before the active-space recount.
    beta_occupied: list[int] | None = None
    beta_virtual: list[int] | None = None
    if selected_orbitals is not None:
        active_occupied = [
            orbital for orbital in selected_orbitals if orbital < n_alpha
        ]
        active_virtual = [
            orbital - n_alpha for orbital in selected_orbitals if orbital >= n_alpha
        ]
        beta_occupied = [
            orbital for orbital in selected_orbitals if orbital < n_beta
        ]
        beta_virtual = [
            orbital - n_beta for orbital in selected_orbitals if orbital >= n_beta
        ]
        frozen_orbitals = [
            orbital for orbital in range(n_alpha) if orbital not in selected_orbitals
        ]
        # Occupation-aware frozen-core energy.  Each ordered pair (i, j)
        # including i == j contributes
        #   1/2 [ occ_i occ_j (ii|jj) - min(occ_i, occ_j) (ij|ji) ]
        # which for occ = 2.0 everywhere is bitwise identical to the
        # closed-shell 2.0 (ii|jj) - (ij|ji) form: powers of two scale
        # exactly in IEEE-754, so 0.5 * (4.0*J - 2.0*K) == 2.0*J - K.
        for frozen_orbital in frozen_orbitals:
            occ_frozen = mo_occ[frozen_orbital]
            core_energy += occ_frozen * h1_mo[frozen_orbital, frozen_orbital]
            for other_frozen_orbital in frozen_orbitals:
                occ_other = mo_occ[other_frozen_orbital]
                core_energy += 0.5 * (
                    occ_frozen
                    * occ_other
                    * h2_mo[
                        frozen_orbital,
                        frozen_orbital,
                        other_frozen_orbital,
                        other_frozen_orbital,
                    ]
                    - min(occ_frozen, occ_other)
                    * h2_mo[
                        frozen_orbital,
                        other_frozen_orbital,
                        other_frozen_orbital,
                        frozen_orbital,
                    ]
                )
        h1_mo = h1_mo[np.ix_(selected_orbitals, selected_orbitals)]
        # Frozen -> active embedding.  The spin-averaged one-body effective
        # potential of a frozen orbital f is
        #   occ_f (aa|ff) - occ_f/2 (af|fa)
        # (the exchange averaged over the two spin blocks).  It is exact
        # whenever every frozen orbital is doubly occupied -- the common
        # frozen-core case, including closed shell where it reduces to the
        # original 2 (aa|ff) - (af|fa) -- and a singly occupied frozen
        # orbital next to a singly occupied active orbital would be off by
        # half the exchange integral (documented limitation of the single
        # spin-averaged h1).
        for frozen_orbital in frozen_orbitals:
            occ_frozen = mo_occ[frozen_orbital]
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
            h1_mo += occ_frozen * coulomb - 0.5 * occ_frozen * exchange
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

    geometry = [
        [mol.atom_symbol(i), [float(x), float(y), float(z)]]
        for i, (x, y, z) in enumerate(mol.atom_coords(unit="Angstrom"))
    ]
    provenance: dict[
        str, str | int | bool | float | np.ndarray | list | dict
    ] = {
        "source": "pyscf",
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "geometry": geometry,
        "mo_coeff": np.asarray(c_matrix, dtype=np.float64),
        "mo_occ": mo_occ,
        "active_space": selected_orbitals if selected_orbitals is not None else [],
        "with_ccsd": with_ccsd,
    }
    if with_ccsd:
        coupled_cluster = cc.CCSD(mf)
        coupled_cluster.kernel()
        if isinstance(coupled_cluster.t1, tuple):
            # Unrestricted amplitudes from an ROHF reference (PySCF converts
            # to UHF and returns UCCSD): t1 = (t1a, t1b), t2 = (t2aa, t2ab,
            # t2bb).  Stored as nested lists in the exact shape the
            # spin-unbalanced LUCJ branch consumes:
            #   provenance["t1"] == [t1a, t1b]
            #   provenance["t2"] == [t2aa, t2ab, t2bb]
            t1a, t1b = coupled_cluster.t1
            t2aa, t2ab, t2bb = coupled_cluster.t2
            if active_occupied is not None and active_virtual is not None:
                t1a = t1a[np.ix_(active_occupied, active_virtual)]
                t1b = t1b[np.ix_(beta_occupied, beta_virtual)]
                t2aa = t2aa[
                    np.ix_(
                        active_occupied,
                        active_occupied,
                        active_virtual,
                        active_virtual,
                    )
                ]
                t2ab = t2ab[
                    np.ix_(
                        active_occupied,
                        beta_occupied,
                        active_virtual,
                        beta_virtual,
                    )
                ]
                t2bb = t2bb[
                    np.ix_(
                        beta_occupied,
                        beta_occupied,
                        beta_virtual,
                        beta_virtual,
                    )
                ]
            provenance["t1"] = [t1a.tolist(), t1b.tolist()]
            provenance["t2"] = [t2aa.tolist(), t2ab.tolist(), t2bb.tolist()]
        else:
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
