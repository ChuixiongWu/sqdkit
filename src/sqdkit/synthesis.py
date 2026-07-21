"""Target-aware compilation for prepared Qiskit circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager

from sqdkit.targets import BackendProfile


def transpile_to_target(
    circuit: "QuantumCircuit",
    profile: BackendProfile,
    *,
    pass_manager: "PassManager | None" = None,
    seed: int | None = None,
) -> tuple["QuantumCircuit", dict[str, int | str | None]]:
    """Compile a prepared circuit against a portable backend profile."""
    compiled_circuit = (
        pass_manager.run(circuit)
        if pass_manager is not None
        else transpile(
            circuit,
            target=profile.to_target(),
            optimization_level=1,
            seed_transpiler=seed,
        )
    )
    layout = compiled_circuit.layout
    summary = {
        "depth": compiled_circuit.depth(),
        "num_qubits": compiled_circuit.num_qubits,
        "two_qubit_gate_count": sum(
            instruction.operation.name == "cx" for instruction in compiled_circuit.data
        ),
        "layout": str(layout) if layout is not None else None,
    }
    return compiled_circuit, summary
