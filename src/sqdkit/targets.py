"""Portable backend-capability profiles for local Qiskit Aer sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Self

from qiskit.circuit import Measure
from qiskit.circuit.library import IGate, RZGate, SXGate, XGate
from qiskit.transpiler import CouplingMap, Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

_DEFAULT_BASIS_GATES = ["id", "rz", "sx", "x", "cx"]
_ONE_QUBIT_GATES = {"id": IGate, "rz": RZGate, "sx": SXGate, "x": XGate}


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Serializable target capabilities and local-simulator sampling configuration."""

    mode: str
    coupling_map: list[tuple[int, int]] | None
    basis_gates: list[str]
    gate_error: float = 0.0
    readout_error: float = 0.0
    seed: int = 0
    fake_backend_name: str | None = None
    _calibration_simulator: AerSimulator | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate profile invariants at the construction boundary."""
        match self.mode:
            case "ideal" | "topology":
                pass
            case "synthetic_noise" | "calibration":
                if self.coupling_map is None:
                    msg = f"{self.mode} mode requires a coupling_map"
                    raise ValueError(msg)
            case unsupported:
                msg = f"unsupported backend profile mode: {unsupported}"
                raise ValueError(msg)
        if not self.basis_gates:
            raise ValueError("basis_gates must not be empty")
        if self.gate_error < 0.0 or self.gate_error > 1.0:
            raise ValueError("gate_error must be between 0 and 1")
        if self.readout_error < 0.0 or self.readout_error > 1.0:
            raise ValueError("readout_error must be between 0 and 1")

    @classmethod
    def ideal(cls, *, basis_gates: list[str] | None = None, seed: int = 0) -> Self:
        """Create an all-to-all, noiseless sampling profile."""
        return cls("ideal", None, basis_gates or _DEFAULT_BASIS_GATES, seed=seed)

    @classmethod
    def topology_only(
        cls,
        coupling_map: list[tuple[int, int]],
        basis_gates: list[str],
        *,
        seed: int = 0,
    ) -> Self:
        """Create a noiseless profile constrained by a physical topology."""
        return cls("topology", list(coupling_map), list(basis_gates), seed=seed)

    @classmethod
    def synthetic_noise(
        cls,
        coupling_map: list[tuple[int, int]],
        basis_gates: list[str],
        gate_error: float,
        readout_error: float,
        *,
        seed: int = 0,
    ) -> Self:
        """Create a topology-constrained profile with depolarizing/readout noise."""
        return cls(
            "synthetic_noise",
            list(coupling_map),
            list(basis_gates),
            gate_error,
            readout_error,
            seed,
        )

    @classmethod
    def calibration_like(cls, fake_backend_name: str) -> Self:
        """Create an offline profile backed by an IBM Runtime fake backend snapshot."""
        from qiskit_ibm_runtime import fake_provider

        backend_class = getattr(fake_provider, fake_backend_name, None)
        if backend_class is None and not fake_backend_name.endswith("V2"):
            backend_class = getattr(fake_provider, f"{fake_backend_name}V2", None)
        if backend_class is None:
            raise ValueError(f"unknown fake backend: {fake_backend_name}")
        backend = backend_class()
        coupling_map = backend.coupling_map
        if coupling_map is None:
            raise ValueError(f"fake backend {fake_backend_name} has no coupling_map")
        profile = cls(
            "calibration",
            list(coupling_map.get_edges()),
            list(backend.operation_names),
            fake_backend_name=fake_backend_name,
        )
        simulator = AerSimulator.from_backend(backend)
        object.__setattr__(profile, "_calibration_simulator", simulator)
        return profile

    def fingerprint(self) -> str:
        """Return a stable digest of the persisted capability contract."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()

    def build_simulator(self) -> AerSimulator:
        """Build an Aer simulator matching this profile's sampling mode."""
        match self.mode:
            case "ideal":
                return AerSimulator(seed_simulator=self.seed)
            case "topology":
                return AerSimulator(
                    coupling_map=self.coupling_map,
                    basis_gates=self.basis_gates,
                    seed_simulator=self.seed,
                )
            case "synthetic_noise":
                return AerSimulator(
                    noise_model=self._noise_model(),
                    coupling_map=self.coupling_map,
                    basis_gates=self.basis_gates,
                    seed_simulator=self.seed,
                )
            case "calibration":
                if self._calibration_simulator is None:
                    return self._reload_calibration().build_simulator()
                return self._calibration_simulator
            case unsupported:
                raise AssertionError(f"validated unsupported mode: {unsupported}")

    def to_target(self) -> Target:
        """Build a transpiler target that preserves the profile's topology."""
        coupling_map = (
            None if self.coupling_map is None else CouplingMap(self.coupling_map)
        )
        target = Target.from_configuration(
            basis_gates=self.basis_gates,
            coupling_map=coupling_map,
        )
        if "measure" not in target.operation_names:
            target.add_instruction(Measure(), name="measure")
        return target

    def to_dict(
        self,
    ) -> dict[str, str | int | float | list[str] | list[list[int]] | None]:
        """Return a JSON-compatible representation of this profile."""
        return {
            "mode": self.mode,
            "coupling_map": (
                None
                if self.coupling_map is None
                else [list(edge) for edge in self.coupling_map]
            ),
            "basis_gates": self.basis_gates,
            "gate_error": self.gate_error,
            "readout_error": self.readout_error,
            "seed": self.seed,
            "fake_backend_name": self.fake_backend_name,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, str | int | float | list[str] | list[list[int]] | None]
    ) -> Self:
        """Restore a profile from its JSON-compatible representation."""
        mode = data["mode"]
        if not isinstance(mode, str):
            raise ValueError("mode must be a string")
        fake_backend_name = data.get("fake_backend_name")
        if mode == "calibration" and isinstance(fake_backend_name, str):
            return cls.calibration_like(fake_backend_name)
        raw_coupling_map = data.get("coupling_map")
        coupling_map = (
            None
            if raw_coupling_map is None
            else [tuple(edge) for edge in raw_coupling_map if isinstance(edge, list)]
        )
        basis_gates = data.get("basis_gates")
        if not isinstance(basis_gates, list) or not all(
            isinstance(gate, str) for gate in basis_gates
        ):
            raise ValueError("basis_gates must be a list of strings")
        return cls(
            mode,
            coupling_map,
            basis_gates,
            float(data.get("gate_error", 0.0)),
            float(data.get("readout_error", 0.0)),
            int(data.get("seed", 0)),
            fake_backend_name if isinstance(fake_backend_name, str) else None,
        )

    def save_json(self, path: Path) -> None:
        """Persist this profile as formatted JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> Self:
        """Load a profile persisted by :meth:`save_json`."""
        raw_data = json.loads(path.read_text())
        if not isinstance(raw_data, dict):
            raise ValueError("backend profile JSON must contain an object")
        return cls.from_dict(raw_data)

    def _noise_model(self) -> NoiseModel:
        """Construct the simplified synthetic error model."""
        noise_model = NoiseModel()
        one_qubit_gates = [
            gate for gate in self.basis_gates if gate in _ONE_QUBIT_GATES
        ]
        two_qubit_gates = [
            gate for gate in self.basis_gates if gate in {"cx", "ecr", "cz"}
        ]
        if one_qubit_gates and self.gate_error:
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(self.gate_error, 1), one_qubit_gates
            )
        if two_qubit_gates and self.gate_error:
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(self.gate_error, 2), two_qubit_gates
            )
        if self.readout_error:
            noise_model.add_all_qubit_readout_error(
                ReadoutError(
                    [
                        [1.0 - self.readout_error, self.readout_error],
                        [self.readout_error, 1.0 - self.readout_error],
                    ]
                )
            )
        return noise_model

    def _reload_calibration(self) -> Self:
        """Recreate a persisted calibration profile through its builder."""
        if self.fake_backend_name is None:
            raise ValueError("calibration profile requires fake_backend_name")
        return self.calibration_like(self.fake_backend_name)
