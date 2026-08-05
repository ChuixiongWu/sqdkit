import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from sqdkit.problem import ProblemData
from sqdkit.pyscf_driver import build_problem
from sqdkit.validation import fci_reference

H2_STO3G = [["H", (0, 0, 0)], ["H", (0, 0, 0.735)]]

NH_STO3G = "N 0 0 0; H 0 0 1.0"


def _rohf_per_spin_occupations(mf):
    """(occ_a, occ_b) per spatial orbital from an ROHF occupation vector.

    ``mf.mo_occ`` is 2.0 (doubly), 1.0 (singly, alpha) or 0.0 (virtual);
    ``mf.make_rdm1()`` on an ROHF reference builds the per-spin densities as
    ``C @ diag(occ_sigma) @ C.T`` (see pyscf/scf/rohf.py:264-274), so the
    diagonal occupations below ARE the MO-basis ROHF density.
    """
    mo_occ = np.asarray(mf.mo_occ, dtype=np.float64)
    occ_a = np.asarray(mo_occ >= 1.0, dtype=np.float64)
    occ_b = np.asarray(mo_occ >= 2.0, dtype=np.float64)
    return occ_a, occ_b


def _reconstruct_energy(problem, occ_a, occ_b):
    """Exact energy of a determinant with per-spin occupations occ_a/occ_b.

    E = core + Tr(h1 D) + 1/2 J(D) - 1/2 [K(D_a) + K(D_b)] with D = D_a + D_b,
    evaluated on the diagonal MO-basis densities; for the ROHF determinant this
    is exactly the energy PySCF reports (rohf.energy_elec uses the same
    density).  J uses the total density; K is spin-diagonal only, hence the
    per-spin split (the h2 term of a spin-blocked Hamiltonian builds the
    same-spin exchange automatically).
    """
    occ = occ_a + occ_b
    one_body = np.einsum("i,ii->", occ, problem.h1)
    coulomb = np.einsum("i,j,iijj->", occ, occ, problem.h2)
    exchange = np.einsum("i,j,ijji->", occ_a, occ_a, problem.h2)
    exchange += np.einsum("i,j,ijji->", occ_b, occ_b, problem.h2)
    return problem.core_energy + one_body + 0.5 * coulomb - 0.5 * exchange


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


def test_build_problem_persists_mo_coeff_occ_and_geometry(tmp_path):
    # Given
    atoms = "H 0 0 0; H 0 0 0.74"
    basis = "sto-3g"
    output_path = tmp_path / "h2-provenance.npz"

    # When
    problem = build_problem(atoms, basis)
    mo_coeff = problem.provenance["mo_coeff"]
    mo_occ = problem.provenance["mo_occ"]
    geometry = problem.provenance["geometry"]
    problem.save_npz(output_path)
    loaded = ProblemData.load_npz(output_path)

    # Then - the full (nao, nmo) MO coefficient matrix and length-nmo occupations
    mol = gto.M(atom=atoms, basis=basis)
    nao = mol.nao_nr()
    assert isinstance(mo_coeff, np.ndarray)
    assert mo_coeff.shape == (nao, nao)
    assert mo_occ.shape == (nao,)
    np.testing.assert_allclose(mo_occ, [2.0, 0.0], rtol=0, atol=1e-12)

    # The geometry reproduces the input atoms (normalized Angstrom atom spec)
    assert [atom[0] for atom in geometry] == ["H", "H"]
    np.testing.assert_allclose(
        np.asarray([atom[1] for atom in geometry]),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        rtol=0,
        atol=1e-8,
    )

    # Round-trip through the NPZ: recovered mo_coeff/mo_occ match to 1e-12 and
    # the archive contains no object arrays (pickle-free persistence contract)
    recovered = loaded.provenance["mo_coeff"]
    assert isinstance(recovered, np.ndarray)
    assert recovered.shape == mo_coeff.shape
    np.testing.assert_allclose(recovered, mo_coeff, rtol=0, atol=1e-12)
    np.testing.assert_allclose(loaded.provenance["mo_occ"], mo_occ, rtol=0, atol=1e-12)
    assert loaded.provenance["geometry"] == geometry
    with np.load(output_path, allow_pickle=False) as archive:
        assert all(array.dtype != np.dtype(object) for array in archive.values())
        np.testing.assert_allclose(archive["mo_coeff"], mo_coeff, rtol=0, atol=1e-12)

    # The stored matrix is the real MO basis: C.T @ S @ C == I within 1e-8
    overlap = mol.intor("int1e_ovlp")
    deviation = float(np.abs(recovered.T @ overlap @ recovered - np.eye(nao)).max())
    assert deviation < 1e-8

    # An active space shrinks the problem but must not truncate the persisted
    # matrix: mo_coeff stays the full (nao, nmo) coefficients
    reduced = build_problem(atoms, basis, active_space=[0], with_ccsd=False)
    assert reduced.norb == 1
    assert reduced.provenance["mo_coeff"].shape == (nao, nao)


def test_build_problem_open_shell_rohf_stores_unrestricted_amplitudes(tmp_path):
    # Given: an open-shell triplet NH.  scf.RHF(mol) silently returns an ROHF
    # object, and cc.CCSD(rohf) converts it to UHF (pyscf/cc/__init__.py:
    # 100-110; the "RCCSD method does not support ROHF" warning is expected),
    # so t1 is a 2-tuple and t2 a 3-tuple -- this used to crash with
    # AttributeError: 'tuple' object has no attribute 'tolist'.

    # When
    problem = build_problem(NH_STO3G, "sto-3g", spin=2, with_ccsd=True)

    # Then (a) no raise, (b) correct open-shell electron counts
    assert (problem.n_alpha, problem.n_beta) == (5, 3)

    # (c) the five amplitude components, stored as nested lists in the exact
    # shape the spin-unbalanced LUCJ branch consumes:
    #   t1 == [t1a, t1b]  t2 == [t2aa, t2ab, t2bb]
    t1 = problem.provenance["t1"]
    t2 = problem.provenance["t2"]
    assert isinstance(t1, list) and len(t1) == 2
    assert isinstance(t2, list) and len(t2) == 3
    assert [np.asarray(amplitude).shape for amplitude in t1] == [(5, 1), (3, 3)]
    assert [np.asarray(amplitude).shape for amplitude in t2] == [
        (5, 5, 1, 1),
        (5, 3, 1, 3),
        (3, 3, 3, 3),
    ]

    # The active-space slicing must be spin-aware too: beta occupied block is
    # only 2 orbitals when the N 1s (orbital 0) is frozen.
    sliced = build_problem(
        NH_STO3G, "sto-3g", spin=2, active_space=list(range(1, 6)), with_ccsd=True
    )
    sliced_t1 = sliced.provenance["t1"]
    sliced_t2 = sliced.provenance["t2"]
    assert [np.asarray(amplitude).shape for amplitude in sliced_t1] == [(4, 1), (2, 3)]
    assert [np.asarray(amplitude).shape for amplitude in sliced_t2] == [
        (4, 4, 1, 1),
        (4, 2, 1, 3),
        (2, 2, 3, 3),
    ]

    # The nested lists survive the pickle-free NPZ round-trip.
    output_path = tmp_path / "nh-open-shell.npz"
    problem.save_npz(output_path)
    loaded = ProblemData.load_npz(output_path)
    assert loaded.provenance["t1"] == t1
    assert loaded.provenance["t2"] == t2
    with np.load(output_path, allow_pickle=False) as archive:
        assert all(array.dtype != np.dtype(object) for array in archive.values())


def test_build_problem_open_shell_reconstructs_rohf_total_energy():
    # Given: the ROHF reference of NH triplet sto-3g
    mol = gto.M(atom=NH_STO3G, basis="sto-3g", spin=2)
    mf = scf.RHF(mol)
    mf.kernel()
    occ_a, occ_b = _rohf_per_spin_occupations(mf)

    # Full space: h1/h2/core_energy contracted against the ROHF density must
    # reproduce mf.e_tot (catches a wrong core_energy or electron count).
    full = build_problem(NH_STO3G, "sto-3g", spin=2, with_ccsd=False)
    assert abs(_reconstruct_energy(full, occ_a, occ_b) - mf.e_tot) < 1e-8

    # Frozen N 1s (doubly occupied): same check on the active problem.
    frozen_core = build_problem(
        NH_STO3G, "sto-3g", spin=2, active_space=list(range(1, 6)), with_ccsd=False
    )
    assert (frozen_core.n_alpha, frozen_core.n_beta) == (4, 2)
    assert abs(
        _reconstruct_energy(frozen_core, occ_a[1:6], occ_b[1:6]) - mf.e_tot
    ) < 1e-8

    # Occupation-aware frozen core: freeze the two SINGLY occupied alpha
    # orbitals too (active = [1, 2], frozen = [0, 3, 4]).  A singly occupied
    # frozen orbital must contribute one electron, not the closed-shell two.
    singly_frozen = build_problem(
        NH_STO3G, "sto-3g", spin=2, active_space=[1, 2], with_ccsd=False
    )
    assert (singly_frozen.n_alpha, singly_frozen.n_beta) == (2, 2)
    assert abs(
        _reconstruct_energy(singly_frozen, occ_a[[1, 2]], occ_b[[1, 2]]) - mf.e_tot
    ) < 1e-8


def test_build_problem_open_shell_frozen_core_matches_direct_casci():
    # Given: a direct CASCI in the same (6e, 5o) space, N 1s frozen.
    mol = gto.M(atom=NH_STO3G, basis="sto-3g", spin=2)
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcscf.CASCI(mf, 5, (4, 2), ncore=1)
    mc.kernel()
    # CASCI internally re-canonicalizes its MOs (mc.mo_coeff is not
    # mf.mo_coeff), so contracting CASCI densities against the ROHF-basis h1/h2
    # is basis-fragile.  Instead compare exact diagonalizations of the same
    # active-space Hamiltonian: FCI on the problem's integrals vs the direct
    # CASCI total -- both are basis-invariant ground states of that space.
    problem = build_problem(
        NH_STO3G, "sto-3g", spin=2, active_space=list(range(1, 6)), with_ccsd=False
    )
    assert abs(fci_reference(problem) - mc.e_tot) < 1e-8


def test_build_problem_closed_shell_bitwise_unchanged():
    # The frozen-core embedding must stay bit-for-bit the closed-shell
    # arithmetic: recompute the H2/sto-3g active_space=[0] problem from raw
    # integrals with the pre-open-shell formulas (2.0 * h1, 2*coulomb -
    # exchange, 2.0*h2 - h2 pairs) and require EXACT equality.
    atoms = "H 0 0 0; H 0 0 0.74"
    basis = "sto-3g"
    mol = gto.M(atom=atoms, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    c_matrix = mf.mo_coeff
    h1_mo = c_matrix.T @ mf.get_hcore() @ c_matrix
    norb = mol.nao_nr()
    h2_mo = ao2mo.incore.full(mol.intor("int2e"), c_matrix).reshape(
        norb, norb, norb, norb
    )
    n_alpha, _ = mol.nelec
    selected = [0]

    core_energy_ref = float(mol.energy_nuc())
    frozen = [orbital for orbital in range(n_alpha) if orbital not in selected]
    for frozen_orbital in frozen:
        core_energy_ref += 2.0 * h1_mo[frozen_orbital, frozen_orbital]
        for other_frozen_orbital in frozen:
            core_energy_ref += (
                2.0
                * h2_mo[
                    frozen_orbital,
                    frozen_orbital,
                    other_frozen_orbital,
                    other_frozen_orbital,
                ]
                - h2_mo[
                    frozen_orbital,
                    other_frozen_orbital,
                    other_frozen_orbital,
                    frozen_orbital,
                ]
            )
    h1_ref = h1_mo[np.ix_(selected, selected)].copy()
    for frozen_orbital in frozen:
        coulomb = h2_mo[
            np.ix_(selected, selected, [frozen_orbital], [frozen_orbital])
        ][:, :, 0, 0]
        exchange = h2_mo[
            np.ix_(selected, [frozen_orbital], [frozen_orbital], selected)
        ][:, 0, 0, :]
        h1_ref += 2.0 * coulomb - exchange
    h2_ref = h2_mo[np.ix_(selected, selected, selected, selected)]
    h1_ref = np.asarray((h1_ref + h1_ref.T) / 2, dtype=np.float64)

    problem = build_problem(atoms, basis, active_space=selected, with_ccsd=True)
    assert np.array_equal(problem.h1, h1_ref)
    assert np.array_equal(problem.h2, h2_ref)
    assert problem.core_energy == core_energy_ref
