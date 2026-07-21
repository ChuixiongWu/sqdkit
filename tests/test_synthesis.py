from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.exceptions import TranspilerError

from sqdkit.synthesis import transpile_to_target
from sqdkit.targets import BackendProfile


class TranspilingPassManager:
    """Minimal pass-manager stand-in that delegates to Qiskit transpilation."""

    def __init__(self, profile: BackendProfile) -> None:
        self.profile = profile
        self.ran = False

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.ran = True
        return transpile(circuit, target=self.profile.to_target(), optimization_level=1)


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _line_profile() -> BackendProfile:
    return BackendProfile.topology_only(
        [(0, 1), (1, 0)], ["rz", "sx", "x", "cx"]
    )


def test_transpile_to_target_preserves_profile_capabilities() -> None:
    # Given: a Bell state and a bidirectional two-qubit line target.
    profile = _line_profile()

    # When: the state preparation circuit is compiled for the target.
    compiled, _ = transpile_to_target(_bell_circuit(), profile, seed=7)

    # Then: every instruction is supported and two-qubit gates use the line edge.
    for instruction in compiled.data:
        assert instruction.operation.name in profile.basis_gates
        if instruction.operation.num_qubits == 2:
            edge = tuple(compiled.find_bit(qubit).index for qubit in instruction.qubits)
            assert edge in profile.coupling_map


def test_transpile_to_target_produces_deterministic_summary_for_seed() -> None:
    # Given: a circuit, target, and fixed transpiler seed.
    profile = _line_profile()

    # When: compilation runs twice with the same seed.
    _, first_summary = transpile_to_target(_bell_circuit(), profile, seed=11)
    _, second_summary = transpile_to_target(_bell_circuit(), profile, seed=11)

    # Then: the compilation diagnostics are reproducible.
    assert first_summary == second_summary


def test_transpile_to_target_rejects_circuit_wider_than_target() -> None:
    # Given: a three-qubit circuit and a two-qubit target.
    circuit = QuantumCircuit(3)

    # When / Then: target compilation rejects the incompatible circuit width.
    with pytest.raises(TranspilerError):
        transpile_to_target(circuit, _line_profile())


def test_transpile_to_target_uses_supplied_pass_manager() -> None:
    # Given: a supplied state-preparation compilation manager.
    profile = _line_profile()
    pass_manager = TranspilingPassManager(profile)

    # When: target synthesis receives the manager.
    compiled, summary = transpile_to_target(
        _bell_circuit(), profile, pass_manager=pass_manager
    )

    # Then: synthesis delegates to the manager and reports its compiled circuit.
    assert pass_manager.ran
    assert summary["depth"] == compiled.depth()
