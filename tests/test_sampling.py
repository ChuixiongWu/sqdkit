from __future__ import annotations

from qiskit import QuantumCircuit, transpile

from sqdkit.sampling import sample
from sqdkit.targets import BackendProfile


def _compiled_bell_circuit(profile: BackendProfile) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return transpile(circuit, target=profile.to_target(), seed_transpiler=7)


def test_sample_returns_requested_shots_and_diagnostics() -> None:
    # Given: a compiled Bell circuit and an ideal sampling profile.
    profile = BackendProfile.ideal()
    compiled_circuit = _compiled_bell_circuit(profile)

    # When: the circuit is sampled with a fixed seed.
    bit_array, diagnostics = sample(compiled_circuit, profile, shots=64, seed=42)

    # Then: the result retains every shot and reports its summary.
    assert bit_array.num_shots == 64
    assert diagnostics["total_shots"] == 64
    assert diagnostics["unique_bitstrings"] == len(bit_array.get_counts())


def test_sample_is_deterministic_for_same_seed() -> None:
    # Given: an ideal compiled Bell circuit.
    profile = BackendProfile.ideal()
    compiled_circuit = _compiled_bell_circuit(profile)

    # When: sampling runs twice with the same seed.
    first, _ = sample(compiled_circuit, profile, shots=64, seed=42)
    second, _ = sample(compiled_circuit, profile, shots=64, seed=42)

    # Then: both runs produce identical outcomes.
    assert first.get_counts() == second.get_counts()


def test_sample_changes_outcomes_for_different_seeds() -> None:
    # Given: an ideal compiled Bell circuit.
    profile = BackendProfile.ideal()
    compiled_circuit = _compiled_bell_circuit(profile)

    # When: sampling runs with different seeds.
    first, _ = sample(compiled_circuit, profile, shots=64, seed=42)
    second, _ = sample(compiled_circuit, profile, shots=64, seed=43)

    # Then: the sampled counts differ.
    assert first.get_counts() != second.get_counts()
