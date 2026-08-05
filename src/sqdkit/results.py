"""Reproducible, pickle-free persistence for SQD run results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

_RESULT_NPZ_FIELDS = frozenset(
    {"total_energy", "electronic_energy", "core_energy"}
)
_RESULT_JSON_FIELDS = frozenset(
    {
        "total_energy",
        "electronic_energy",
        "core_energy",
        "converged",
        "iterations",
        "subspace_summary",
        "shots_total",
        "valid_count",
        "unique_count",
        "stateprep_summary",
        "compiled_summary",
        "backend_profile_mode",
        "backend_profile_fingerprint",
        "effective_config",
        "seeds",
        "elapsed_seconds",
        "metadata",
    }
)
_SEED_KEYS = frozenset({"transpiler", "sampler", "sqd"})
_PACKAGE_VERSION_KEYS = frozenset(
    {"qiskit", "qiskit_addon_sqd", "ffsim", "qiskit_aer", "pyscf"}
)
_PACKAGE_DISTRIBUTIONS = {
    "qiskit": "qiskit",
    "qiskit_addon_sqd": "qiskit-addon-sqd",
    "ffsim": "ffsim",
    "qiskit_aer": "qiskit-aer",
    "pyscf": "pyscf",
}


def _package_versions() -> dict[str, str]:
    """Return installed distribution versions without importing scientific packages."""
    versions: dict[str, str] = {}
    for key, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    """Normalize an artifact stem into its NPZ and JSON sidecar paths."""
    artifact_path = Path(path)
    return artifact_path.with_suffix(".npz"), artifact_path.with_suffix(".json")


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Required provenance recorded alongside every persisted SQD result."""

    schema_version: int = 1
    seeds: dict = field(default_factory=dict)
    package_versions: dict = field(default_factory=dict)
    backend_profile_mode: str = ""
    backend_profile_fingerprint: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Ensure each reproducibility field and nested required key is present."""
        required_values = {
            "seeds": self.seeds,
            "package_versions": self.package_versions,
            "backend_profile_mode": self.backend_profile_mode,
            "backend_profile_fingerprint": self.backend_profile_fingerprint,
            "created_at": self.created_at,
        }
        for key, value in required_values.items():
            if not value:
                raise ValueError(f"missing required metadata key: {key}")

        missing_seeds = _SEED_KEYS.difference(self.seeds)
        if missing_seeds:
            raise ValueError(
                f"missing required metadata key: {sorted(missing_seeds)[0]}"
            )
        missing_versions = _PACKAGE_VERSION_KEYS.difference(self.package_versions)
        if missing_versions:
            raise ValueError(
                f"missing required metadata key: {sorted(missing_versions)[0]}"
            )
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        try:
            datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise ValueError("created_at must be ISO8601") from error


@dataclass(frozen=True, slots=True)
class SQDResult:
    """SQD energies, diagnostics, and provenance for one completed run."""

    total_energy: float
    electronic_energy: float
    core_energy: float
    converged: bool
    iterations: int
    subspace_summary: dict
    shots_total: int
    valid_count: int
    unique_count: int
    stateprep_summary: dict
    compiled_summary: dict
    backend_profile_mode: str
    backend_profile_fingerprint: str
    effective_config: dict
    seeds: dict
    elapsed_seconds: float
    metadata: RunMetadata | None = None
    rdm1: np.ndarray | None = None
    """Spin-summed 1-RDM in the active-space MO basis, shape ``(norb, norb)``.

    Copied verbatim from the upstream ``SCIResult.rdm1`` (pyscf
    ``make_rdm1`` on the converged SCI vector); its trace equals the active
    electron count ``n_alpha + n_beta``. ``None`` on pre-extension artifacts.
    """

    rdm2: np.ndarray | None = None
    """Spin-summed 2-RDM in the active-space MO basis, shape ``(norb,) * 4``.

    Copied verbatim from the upstream ``SCIResult.rdm2``. Chosen over the
    raw ``sci_state`` amplitudes/``ci_strs`` as the density-rebuild piece
    because upstream already computed it (zero recompute) and every
    rank-1/rank-2 density (spin-summed or spin-resolved, via diagonal
    extraction) is recoverable from ``(rdm1, rdm2, orbital_occupancies)``.
    ``None`` on pre-extension artifacts.
    """

    orbital_occupancies: np.ndarray | None = None
    """Per-spin orbital populations, shape ``(2, norb)``; row 0 = alpha,
    row 1 = beta (stacked from the upstream ``orbital_occupancies`` tuple).

    Each entry lies in ``[0, 1]`` per spin; row sums equal ``n_alpha`` and
    ``n_beta``. ``None`` on pre-extension artifacts.
    """

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(
                self,
                "metadata",
                RunMetadata(
                    seeds=self.seeds,
                    package_versions=_package_versions(),
                    backend_profile_mode=self.backend_profile_mode,
                    backend_profile_fingerprint=self.backend_profile_fingerprint,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        else:
            self.metadata.validate()
        self._validate_density_fields()

    def _validate_density_fields(self) -> None:
        rdm1 = self.rdm1
        if rdm1 is not None:
            rdm1 = np.asarray(rdm1, dtype=np.float64)
            object.__setattr__(self, "rdm1", rdm1)
            if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
                raise ValueError("rdm1 must be a square 2D array")
            norb = rdm1.shape[0]
            if not np.allclose(rdm1, rdm1.T, rtol=0.0, atol=1e-8):
                raise ValueError("rdm1 must be symmetric within 1e-8")
            trace = float(np.trace(rdm1))
            if not 0.0 - 1e-8 <= trace <= 2.0 * norb + 1e-8:
                raise ValueError(
                    "rdm1 trace must lie in [0, 2 * norb] for a 1-RDM"
                )

        rdm2 = self.rdm2
        if rdm2 is not None:
            rdm2 = np.asarray(rdm2, dtype=np.float64)
            object.__setattr__(self, "rdm2", rdm2)
            if (
                rdm2.ndim != 4
                or rdm2.shape[0] != rdm2.shape[1]
                or rdm2.shape[1] != rdm2.shape[2]
                or rdm2.shape[2] != rdm2.shape[3]
            ):
                raise ValueError("rdm2 must have shape (norb, norb, norb, norb)")

        occupancies = self.orbital_occupancies
        if occupancies is not None:
            occupancies = np.asarray(occupancies, dtype=np.float64)
            object.__setattr__(self, "orbital_occupancies", occupancies)
            if occupancies.ndim != 2 or occupancies.shape[0] != 2:
                raise ValueError(
                    "orbital_occupancies must have shape (2, norb) "
                    "(row 0 = alpha, row 1 = beta)"
                )
            if not np.all(occupancies >= -1e-8) or not np.all(
                occupancies <= 1.0 + 1e-8
            ):
                raise ValueError(
                    "orbital_occupancies must lie in [0, 1] within 1e-8"
                )

        if rdm1 is not None and occupancies is not None:
            if occupancies.shape[1] != rdm1.shape[0]:
                raise ValueError(
                    "orbital_occupancies norb must match rdm1 norb"
                )
            expected_electrons = float(
                occupancies[0].sum() + occupancies[1].sum()
            )
            if abs(float(np.trace(rdm1)) - expected_electrons) > 1e-6:
                raise ValueError(
                    "rdm1 must trace to the active electron count "
                    f"(trace={float(np.trace(rdm1))}, expected={expected_electrons})"
                )

    def save(self, path: str | Path) -> None:
        """Persist float64 energy scalars and a JSON-only metadata sidecar."""
        npz_path, json_path = _artifact_paths(path)
        npz_fields = {
            "total_energy": np.array(self.total_energy, dtype=np.float64),
            "electronic_energy": np.array(self.electronic_energy, dtype=np.float64),
            "core_energy": np.array(self.core_energy, dtype=np.float64),
        }
        if self.rdm1 is not None:
            npz_fields["rdm1"] = np.asarray(self.rdm1, dtype=np.float64)
        if self.rdm2 is not None:
            npz_fields["rdm2"] = np.asarray(self.rdm2, dtype=np.float64)
        if self.orbital_occupancies is not None:
            npz_fields["orbital_occupancies"] = np.asarray(
                self.orbital_occupancies, dtype=np.float64
            )
        np.savez(npz_path, **npz_fields)
        occupancies_summary = None
        if self.orbital_occupancies is not None:
            occupancies_summary = {
                "alpha_sum": float(self.orbital_occupancies[0].sum()),
                "beta_sum": float(self.orbital_occupancies[1].sum()),
                "min": float(self.orbital_occupancies.min()),
                "max": float(self.orbital_occupancies.max()),
            }
        json_path.write_text(
            json.dumps(
                {
                    "total_energy": self.total_energy,
                    "electronic_energy": self.electronic_energy,
                    "core_energy": self.core_energy,
                    "converged": bool(self.converged),
                    "iterations": self.iterations,
                    "subspace_summary": self.subspace_summary,
                    "shots_total": self.shots_total,
                    "valid_count": self.valid_count,
                    "unique_count": self.unique_count,
                    "stateprep_summary": self.stateprep_summary,
                    "compiled_summary": self.compiled_summary,
                    "backend_profile_mode": self.backend_profile_mode,
                    "backend_profile_fingerprint": self.backend_profile_fingerprint,
                    "effective_config": self.effective_config,
                    "seeds": self.seeds,
                    "elapsed_seconds": self.elapsed_seconds,
                    "metadata": asdict(self.metadata),
                    "rdm1_shape": (
                        None
                        if self.rdm1 is None
                        else [int(dim) for dim in self.rdm1.shape]
                    ),
                    "rdm2_shape": (
                        None
                        if self.rdm2 is None
                        else [int(dim) for dim in self.rdm2.shape]
                    ),
                    "orbital_occupancies_shape": (
                        None
                        if self.orbital_occupancies is None
                        else [int(dim) for dim in self.orbital_occupancies.shape]
                    ),
                    "orbital_occupancies_summary": occupancies_summary,
                },
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> SQDResult:
        """Load a result artifact while prohibiting pickle deserialization."""
        npz_path, json_path = _artifact_paths(path)
        with np.load(npz_path, allow_pickle=False) as archive:
            missing_fields = _RESULT_NPZ_FIELDS.difference(archive.files)
            if missing_fields:
                raise KeyError(
                    f"NPZ is missing required fields: {sorted(missing_fields)}"
                )
            energies = {name: archive[name] for name in _RESULT_NPZ_FIELDS}
            rdm1 = archive["rdm1"] if "rdm1" in archive.files else None
            rdm2 = archive["rdm2"] if "rdm2" in archive.files else None
            occupancies = (
                archive["orbital_occupancies"]
                if "orbital_occupancies" in archive.files
                else None
            )

        for name, value in energies.items():
            if value.shape != ():
                raise ValueError(f"{name} must be a zero-dimensional scalar")
            if value.dtype != np.dtype(np.float64):
                raise ValueError(f"{name} must have dtype float64")

        json_payload = json.loads(json_path.read_text())
        if not isinstance(json_payload, dict):
            raise ValueError("result JSON must encode an object")
        missing_json_fields = _RESULT_JSON_FIELDS.difference(json_payload)
        if missing_json_fields:
            raise KeyError(
                f"JSON is missing required fields: {sorted(missing_json_fields)}"
            )
        metadata_payload = json_payload["metadata"]
        if not isinstance(metadata_payload, dict):
            raise ValueError("metadata must encode an object")

        return cls(
            total_energy=float(energies["total_energy"].item()),
            electronic_energy=float(energies["electronic_energy"].item()),
            core_energy=float(energies["core_energy"].item()),
            converged=bool(json_payload["converged"]),
            iterations=int(json_payload["iterations"]),
            subspace_summary=json_payload["subspace_summary"],
            shots_total=int(json_payload["shots_total"]),
            valid_count=int(json_payload["valid_count"]),
            unique_count=int(json_payload["unique_count"]),
            stateprep_summary=json_payload["stateprep_summary"],
            compiled_summary=json_payload["compiled_summary"],
            backend_profile_mode=str(json_payload["backend_profile_mode"]),
            backend_profile_fingerprint=str(json_payload["backend_profile_fingerprint"]),
            effective_config=json_payload["effective_config"],
            seeds=json_payload["seeds"],
            elapsed_seconds=float(json_payload["elapsed_seconds"]),
            metadata=RunMetadata(**metadata_payload),
            rdm1=rdm1,
            rdm2=rdm2,
            orbital_occupancies=occupancies,
        )
