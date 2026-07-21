"""Small, serializable configuration objects for reproducible SQD runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_STATEPREP_METHODS = frozenset({"lucj"})
_BACKEND_PROFILES = frozenset(
    {"ideal", "topology", "synthetic_noise", "calibration"}
)


def derived_seeds(seed: int) -> dict[str, int]:
    """Derive the deterministic seeds used by each stage of an SQD run."""
    return {"transpiler": seed, "sampler": seed + 1, "sqd": seed + 2}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Common controls for one reproducible SQD run."""

    input_path: str | Path = "problem.npz"
    stateprep_method: str = "lucj"
    stateprep_options: dict[str, JSONValue] = field(default_factory=dict)
    backend_profile: str = "ideal"
    backend_profile_options: dict[str, JSONValue] = field(default_factory=dict)
    shots: int = 10_000
    samples_per_batch: int = 100
    num_batches: int = 10
    max_iterations: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject invalid run controls before execution begins."""
        for field_name, value in (
            ("shots", self.shots),
            ("samples_per_batch", self.samples_per_batch),
            ("num_batches", self.num_batches),
            ("max_iterations", self.max_iterations),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if self.stateprep_method not in _STATEPREP_METHODS:
            raise ValueError(f"unknown stateprep_method: {self.stateprep_method}")
        if self.backend_profile not in _BACKEND_PROFILES:
            raise ValueError(f"unknown backend_profile: {self.backend_profile}")

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable representation of this configuration."""
        return {
            "input_path": str(self.input_path),
            "stateprep_method": self.stateprep_method,
            "stateprep_options": self.stateprep_options.copy(),
            "backend_profile": self.backend_profile,
            "backend_profile_options": self.backend_profile_options.copy(),
            "shots": self.shots,
            "samples_per_batch": self.samples_per_batch,
            "num_batches": self.num_batches,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, values: dict[str, JSONValue]) -> RunConfig:
        """Build a validated run configuration from a serialized dictionary."""
        return cls(
            input_path=str(values.get("input_path", "problem.npz")),
            stateprep_method=str(values.get("stateprep_method", "lucj")),
            stateprep_options=dict(values.get("stateprep_options", {})),
            backend_profile=str(values.get("backend_profile", "ideal")),
            backend_profile_options=dict(values.get("backend_profile_options", {})),
            shots=int(values.get("shots", 10_000)),
            samples_per_batch=int(values.get("samples_per_batch", 100)),
            num_batches=int(values.get("num_batches", 10)),
            max_iterations=int(values.get("max_iterations", 10)),
            seed=int(values.get("seed", 0)),
        )


@dataclass(frozen=True, slots=True)
class AdvancedConfig:
    """Less common SQD diagonalization controls passed through unchanged."""

    energy_tol: float = 1e-8
    occupancies_tol: float = 1e-5
    carryover_threshold: float = 1e-4
    symmetrize_spin: bool = False
    max_dim: int | tuple[int, ...] | None = None
    spin_sq: float | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable representation of advanced controls."""
        max_dim: JSONValue = (
            list(self.max_dim) if isinstance(self.max_dim, tuple) else self.max_dim
        )
        return {
            "energy_tol": self.energy_tol,
            "occupancies_tol": self.occupancies_tol,
            "carryover_threshold": self.carryover_threshold,
            "symmetrize_spin": self.symmetrize_spin,
            "max_dim": max_dim,
            "spin_sq": self.spin_sq,
        }

    @classmethod
    def from_dict(cls, values: dict[str, JSONValue]) -> AdvancedConfig:
        """Build advanced controls from a serialized dictionary."""
        raw_max_dim = values.get("max_dim")
        max_dim = tuple(raw_max_dim) if isinstance(raw_max_dim, list) else raw_max_dim
        return cls(
            energy_tol=float(values.get("energy_tol", 1e-8)),
            occupancies_tol=float(values.get("occupancies_tol", 1e-5)),
            carryover_threshold=float(values.get("carryover_threshold", 1e-4)),
            symmetrize_spin=bool(values.get("symmetrize_spin", False)),
            max_dim=max_dim,
            spin_sq=(
                float(values["spin_sq"])
                if values.get("spin_sq") is not None
                else None
            ),
        )
