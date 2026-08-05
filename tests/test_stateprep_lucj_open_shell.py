"""Open-shell (ROHF, n_alpha != n_beta) LUCJ state-preparation locks.

Todo 9 stores unrestricted UCCSD amplitudes as nested lists in the exact
tuple order ``UCJOpSpinUnbalanced.from_t_amplitudes`` consumes (``t1`` is
``[t1a, t1b]``, ``t2`` is ``[t2aa, t2ab, t2bb]``); todo 10's branch feeds them
through the spin-unbalanced LUCJ ansatz.  This reproduces the planning probe:
NH sto-3g spin=2, full space, sampled and diagonalized end to end.

NOTE ON THE SQD INPUT ENCODING: ``to_sqd_bit_array`` returns the layout
upstream ``qiskit_addon_sqd`` expects: blocks ordered ``[beta..., alpha...]``
with the columns within each block reversed, because
``bitstring_matrix_to_integers`` treats column 0 as the most significant bit
of the CI string while PySCF's ``kernel_fixed_space`` reads it
least-significant-bit-first.  H2's norb=2 ground state is symmetric under the
reversal, which is why the closed-shell tests cannot see a wrong intra-block
order; the variational bound below is the gate that catches the mismatch for
open-shell systems.  The ``LucjStatePrep`` ansatz branch itself is
order-independent.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from sqdkit.config import RunConfig, derived_seeds
from sqdkit.pyscf_driver import build_problem
from sqdkit.sampling import sample
from sqdkit.stateprep import to_sqd_bit_array
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.synthesis import transpile_to_target
from sqdkit.targets import BackendProfile
from sqdkit.validation import fci_reference

NH_STO3G = [("N", (0, 0, 0)), ("H", (0, 0, 1.0))]

# Todo-9 verified open-shell counts for NH sto-3g spin=2 (6 spatial orbitals).
NH_N_ALPHA = 5
NH_N_BETA = 3


def _nh_open_shell_problem():
    """Build the todo-9 open-shell problem and lock its occupation counts."""
    problem = build_problem(NH_STO3G, "sto-3g", spin=2, with_ccsd=True)
    assert problem.n_alpha == NH_N_ALPHA
    assert problem.n_beta == NH_N_BETA
    return problem


def _run_config() -> RunConfig:
    return RunConfig(
        shots=20_000,
        samples_per_batch=1_000,
        num_batches=1,
        max_iterations=5,
        seed=42,
    )


def _sample_sqd_array(problem, stateprep, config) -> np.ndarray:
    """Sample the compiled circuit with the kit's deterministic seeds."""
    profile = BackendProfile.ideal()
    seeds = derived_seeds(config.seed)
    prepared = stateprep.build(problem)
    circuit = prepared.circuit.copy()
    circuit.measure_all()
    compiled, _ = transpile_to_target(
        circuit,
        profile,
        pass_manager=stateprep.pass_manager(profile),
        seed=seeds["transpiler"],
    )
    bit_array, _ = sample(
        compiled,
        profile,
        shots=config.shots,
        seed=seeds["sampler"],
    )
    return to_sqd_bit_array(bit_array, problem.norb)


def test_open_shell_lucj_samples_exactly_the_spin_resolved_occupations() -> None:
    # Given: an open-shell NH problem and the spin-unbalanced LUCJ preparation.
    problem = _nh_open_shell_problem()
    stateprep = LucjStatePrep(n_reps=2)
    config = _run_config()

    # When: the prepared circuit is compiled and sampled.
    sqd_array = _sample_sqd_array(problem, stateprep, config)

    # Then: every sampled bitstring carries exactly (n_alpha, n_beta)
    # electrons.  After to_sqd_bit_array the layout is [beta..., alpha...].
    assert sqd_array.shape == (config.shots, 2 * problem.norb)
    alpha_counts = sqd_array[..., problem.norb:].sum(axis=-1)
    beta_counts = sqd_array[..., :problem.norb].sum(axis=-1)
    assert np.all(alpha_counts == NH_N_ALPHA)
    assert np.all(beta_counts == NH_N_BETA)


def test_open_shell_lucj_energy_is_variational_and_rdm1_traces_to_n() -> None:
    # Given: the same open-shell problem sampled through the LUCJ preparation.
    problem = _nh_open_shell_problem()
    stateprep = LucjStatePrep(n_reps=2)
    config = _run_config()
    seeds = derived_seeds(config.seed)
    sqd_array = _sample_sqd_array(problem, stateprep, config)

    # When: the upstream diagonalizer runs on the layout already produced by
    # the fixed to_sqd_bit_array, with the kit's fixed seed and
    # (n_alpha, n_beta) = (5, 3).
    sqd_input = BitArray.from_bool_array(sqd_array, order="big")
    sci_result = diagonalize_fermionic_hamiltonian(
        problem.h1,
        problem.h2,
        sqd_input,
        config.samples_per_batch,
        problem.norb,
        (problem.n_alpha, problem.n_beta),
        num_batches=config.num_batches,
        max_iterations=config.max_iterations,
        seed=seeds["sqd"],
        symmetrize_spin=False,
    )
    e_sqd = float(sci_result.energy + problem.core_energy)

    # Then: the total energy is variational between FCI and the ROHF reference
    # (both computed in this test on the same integrals / geometry).
    e_fci = fci_reference(problem)
    mol = gto.M(atom=NH_STO3G, basis="sto-3g", spin=2)
    e_rohf = scf.RHF(mol).kernel()
    assert e_fci - 1e-6 <= e_sqd <= e_rohf

    # And: the surfaced 1-RDM traces to the active electron count.
    assert sci_result.rdm1 is not None
    assert sci_result.rdm1.shape == (problem.norb, problem.norb)
    assert abs(float(np.trace(sci_result.rdm1)) - (NH_N_ALPHA + NH_N_BETA)) < 1e-6
