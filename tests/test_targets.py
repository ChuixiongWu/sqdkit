from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Target

from sqdkit.targets import BackendProfile


def _assert_runnable(profile: BackendProfile) -> None:
    """Run a single-qubit circuit through a profile's simulator and target."""
    # Given: a profile with a simulator and compilation target.
    simulator = profile.build_simulator()
    target = profile.to_target()
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()

    # When: the circuit is compiled and sampled locally.
    compiled = transpile(circuit, target=target, seed_transpiler=profile.seed)
    result = simulator.run(compiled, shots=8).result()

    # Then: both profile artifacts are usable.
    assert isinstance(target, Target)
    assert result.get_counts(compiled)


@pytest.mark.parametrize(
    "profile",
    [
        BackendProfile.ideal(),
        BackendProfile.topology_only([(0, 1), (1, 0)], ["rz", "sx", "x", "cx"]),
        BackendProfile.synthetic_noise(
            [(0, 1), (1, 0)], ["rz", "sx", "x", "cx"], 0.001, 0.01
        ),
        BackendProfile.calibration_like("FakeVigo"),
    ],
)
def test_profile_mode_builds_runnable_simulator_and_target(
    profile: BackendProfile,
) -> None:
    _assert_runnable(profile)


def test_topology_target_preserves_line_connectivity() -> None:
    # Given: a directed line topology.
    coupling_map = CouplingMap.from_line(3, bidirectional=True)
    profile = BackendProfile.topology_only(
        coupling_map.get_edges(), ["rz", "sx", "x", "cx"]
    )

    # When: the profile becomes a transpiler target.
    target = profile.to_target()

    # Then: target connectivity remains the requested line, not all-to-all.
    assert target.build_coupling_map().get_edges() == coupling_map.get_edges()


def test_fingerprint_is_stable_for_identical_profiles() -> None:
    # Given: two independently constructed, equivalent profiles.
    first = BackendProfile.synthetic_noise(
        [(0, 1), (1, 0)], ["rz", "sx", "x", "cx"], 0.001, 0.01, seed=7
    )
    second = BackendProfile.synthetic_noise(
        [(0, 1), (1, 0)], ["rz", "sx", "x", "cx"], 0.001, 0.01, seed=7
    )

    # When: each profile is fingerprinted.
    first_fingerprint = first.fingerprint()
    second_fingerprint = second.fingerprint()

    # Then: the stable profile representation produces the same digest.
    assert first_fingerprint == second_fingerprint


@pytest.mark.parametrize("mode", ["synthetic_noise", "calibration"])
def test_noise_modes_require_a_coupling_map(mode: str) -> None:
    # Given: a noisy profile mode without topology information.
    # When / Then: construction rejects the incomplete capability contract.
    with pytest.raises(ValueError, match="coupling_map"):
        BackendProfile(mode=mode, coupling_map=None, basis_gates=["rz", "sx", "cx"])
