import numpy as np
import pytest
from pyscf import gto

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
