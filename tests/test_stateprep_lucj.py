from __future__ import annotations

import numpy as np
from qiskit.transpiler import PassManager

from sqdkit.problem import ProblemData
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile


def test_lucj_state_prep_builds_spin_blocked_circuit() -> None:
    # Given: a two-orbital CCSD problem with serialized amplitudes.
    norb = 2
    problem = ProblemData(
        h1=np.zeros((norb, norb), dtype=np.float64),
        h2=np.zeros((norb, norb, norb, norb), dtype=np.float64),
        core_energy=0.0,
        norb=norb,
        n_alpha=1,
        n_beta=1,
        integral_convention="chemist",
        orbital_ordering="alpha_beta",
        provenance={
            "t1": [[0.0]],
            "t2": [[[[0.1]]]],
        },
    )

    # When: LUCJ state preparation is constructed.
    result = LucjStatePrep(n_reps=1).build(problem)

    # Then: it uses one qubit per spin orbital.
    assert result.circuit.num_qubits == 2 * norb
    assert result.norb == norb
    assert result.metadata == {"method": "lucj"}


def test_lucj_state_prep_builds_pass_manager_after_circuit(monkeypatch) -> None:
    # Given: a constructed LUCJ preparation and local backend profile.
    state_prep = LucjStatePrep(n_reps=1)
    problem = ProblemData(
        h1=np.zeros((2, 2), dtype=np.float64),
        h2=np.zeros((2, 2, 2, 2), dtype=np.float64),
        core_energy=0.0,
        norb=2,
        n_alpha=1,
        n_beta=1,
        integral_convention="chemist",
        orbital_ordering="alpha_beta",
        provenance={"t2": [[[[0.1]]]]},
    )
    state_prep.build(problem)
    expected_pass_manager = PassManager()
    monkeypatch.setattr(
        "sqdkit.stateprep_lucj.ffsim.qiskit.generate_lucj_pass_manager",
        lambda **_: (expected_pass_manager, []),
    )

    # When: state-preparation-specific compilation passes are requested.
    pass_manager = state_prep.pass_manager(
        BackendProfile.topology_only(
            [(0, 1), (1, 0)], ["rz", "sx", "x", "cx"]
        )
    )

    # Then: ffsim supplies a Qiskit pass manager.
    assert pass_manager is not None
    assert isinstance(pass_manager, PassManager)


def test_lucj_state_prep_builds_topology_pass_manager() -> None:
    # Given: a constructed LUCJ preparation and a four-qubit line topology.
    state_prep = LucjStatePrep(n_reps=1)
    problem = ProblemData(
        h1=np.zeros((2, 2), dtype=np.float64),
        h2=np.zeros((2, 2, 2, 2), dtype=np.float64),
        core_energy=0.0,
        norb=2,
        n_alpha=1,
        n_beta=1,
        integral_convention="chemist",
        orbital_ordering="alpha_beta",
        provenance={"t2": [[[[0.1]]]]},
    )
    state_prep.build(problem)
    profile = BackendProfile.topology_only(
        [(0, 1), (1, 2), (2, 3)], ["rz", "sx", "x", "cx"]
    )

    # When: ffsim derives physical LUCJ compilation passes.
    pass_manager = state_prep.pass_manager(profile)

    # Then: the line topology produces a usable pass manager.
    assert isinstance(pass_manager, PassManager)
