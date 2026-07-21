"""Seeded local sampling for compiled quantum circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.primitives import BitArray
from qiskit_aer.primitives import SamplerV2

from sqdkit.targets import BackendProfile


def sample(
    compiled_circuit: QuantumCircuit,
    profile: BackendProfile,
    *,
    shots: int,
    seed: int,
) -> tuple[BitArray, dict[str, int]]:
    """Sample an already-compiled circuit reproducibly with an Aer profile."""
    simulator = profile.build_simulator()
    sampler = SamplerV2.from_backend(simulator, default_shots=shots, seed=seed)
    result = sampler.run([(compiled_circuit,)]).result()
    register_name = compiled_circuit.cregs[0].name if compiled_circuit.cregs else "meas"
    bit_array: BitArray = getattr(result[0].data, register_name)
    counts = bit_array.get_counts()
    return bit_array, {
        "total_shots": shots,
        "unique_bitstrings": len(counts),
        "valid_hamming": sum(counts.values()),
    }
