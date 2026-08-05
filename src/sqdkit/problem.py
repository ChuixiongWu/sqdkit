"""Versioned, pickle-free chemistry problem persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REQUIRED_NPZ_FIELDS = frozenset(
    {
        "h1",
        "h2",
        "core_energy",
        "norb",
        "n_alpha",
        "n_beta",
        "integral_convention",
        "orbital_ordering",
        "schema_version",
        "metadata_json",
    }
)


@dataclass(frozen=True, slots=True)
class ProblemData:
    """Spatial-orbital electronic integrals and their reproducibility metadata."""

    h1: np.ndarray
    h2: np.ndarray
    core_energy: float
    norb: int
    n_alpha: int
    n_beta: int
    integral_convention: str
    orbital_ordering: str
    schema_version: int = 1
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the in-memory schema before it can be persisted or consumed."""
        if self.h1.dtype != np.dtype(np.float64):
            raise ValueError("h1 must have dtype float64")
        if self.h2.dtype != np.dtype(np.float64):
            raise ValueError("h2 must have dtype float64")
        if self.h1.shape != (self.norb, self.norb):
            raise ValueError("h1 must have shape (norb, norb)")
        if not np.allclose(self.h1, self.h1.T):
            raise ValueError("h1 must be symmetric")
        if self.h2.shape != (self.norb,) * 4:
            raise ValueError("h2 must have shape (norb, norb, norb, norb)")
        if not 0 <= self.n_alpha <= self.norb:
            raise ValueError("n_alpha must be between zero and norb")
        if not 0 <= self.n_beta <= self.norb:
            raise ValueError("n_beta must be between zero and norb")
        if self.integral_convention != "chemist":
            raise ValueError("integral_convention must be 'chemist'")
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")

    def save_npz(self, path: str | Path) -> None:
        """Write the v1 schema using only numeric and Unicode NumPy arrays.

        ``provenance["mo_coeff"]`` and ``provenance["mo_occ"]`` (when present)
        are stored as plain float64 top-level arrays so the archive stays
        readable with ``allow_pickle=False``; everything else in ``provenance``
        (including the normalized geometry) is JSON inside ``metadata_json``.
        """
        provenance = dict(self.provenance)
        mo_coeff = provenance.pop("mo_coeff", None)
        mo_occ = provenance.pop("mo_occ", None)
        fields: dict[str, np.ndarray] = {
            "h1": self.h1,
            "h2": self.h2,
            "core_energy": np.array(self.core_energy, dtype=np.float64),
            "norb": np.array(self.norb, dtype=np.int64),
            "n_alpha": np.array(self.n_alpha, dtype=np.int64),
            "n_beta": np.array(self.n_beta, dtype=np.int64),
            "integral_convention": np.array(self.integral_convention, dtype=str),
            "orbital_ordering": np.array(self.orbital_ordering, dtype=str),
            "schema_version": np.array(self.schema_version, dtype=np.int64),
            "metadata_json": np.array(json.dumps(provenance), dtype=str),
        }
        if mo_coeff is not None:
            fields["mo_coeff"] = np.asarray(mo_coeff, dtype=np.float64)
        if mo_occ is not None:
            fields["mo_occ"] = np.asarray(mo_occ, dtype=np.float64)
        np.savez(path, **fields)

    @classmethod
    def load_npz(cls, path: str | Path) -> ProblemData:
        """Load and validate a v1 NPZ artifact without allowing pickle data."""
        with np.load(path, allow_pickle=False) as archive:
            missing_fields = _REQUIRED_NPZ_FIELDS.difference(archive.files)
            if missing_fields:
                raise KeyError(
                    f"NPZ is missing required fields: {sorted(missing_fields)}"
                )

            h1 = archive["h1"]
            h2 = archive["h2"]
            core_energy = archive["core_energy"]
            norb = archive["norb"]
            n_alpha = archive["n_alpha"]
            n_beta = archive["n_beta"]
            integral_convention = archive["integral_convention"]
            orbital_ordering = archive["orbital_ordering"]
            schema_version = archive["schema_version"]
            metadata_json = archive["metadata_json"]

            if h1.dtype != np.dtype(np.float64) or h2.dtype != np.dtype(np.float64):
                raise ValueError("h1 and h2 must have dtype float64")
            for name, value in (
                ("core_energy", core_energy),
                ("norb", norb),
                ("n_alpha", n_alpha),
                ("n_beta", n_beta),
                ("schema_version", schema_version),
                ("integral_convention", integral_convention),
                ("orbital_ordering", orbital_ordering),
                ("metadata_json", metadata_json),
            ):
                if value.shape != ():
                    raise ValueError(f"{name} must be a zero-dimensional scalar")
            if core_energy.dtype != np.dtype(np.float64):
                raise ValueError("core_energy must have dtype float64")
            for name, value in (
                ("norb", norb),
                ("n_alpha", n_alpha),
                ("n_beta", n_beta),
                ("schema_version", schema_version),
            ):
                if value.dtype != np.dtype(np.int64):
                    raise ValueError(f"{name} must have dtype int64")
            for name, value in (
                ("integral_convention", integral_convention),
                ("orbital_ordering", orbital_ordering),
                ("metadata_json", metadata_json),
            ):
                if value.dtype.kind != "U":
                    raise ValueError(f"{name} must have a Unicode dtype")

            convention = str(integral_convention.item())
            if convention != "chemist":
                raise ValueError("integral_convention must be 'chemist'")
            provenance = json.loads(str(metadata_json.item()))
            if not isinstance(provenance, dict):
                raise ValueError("metadata_json must encode a JSON object")
            # Optional additive fields; absent in v1 artifacts, which must keep loading.
            if "mo_coeff" in archive.files:
                provenance["mo_coeff"] = archive["mo_coeff"]
            if "mo_occ" in archive.files:
                provenance["mo_occ"] = archive["mo_occ"]

            return cls(
                h1=h1,
                h2=h2,
                core_energy=float(core_energy.item()),
                norb=int(norb.item()),
                n_alpha=int(n_alpha.item()),
                n_beta=int(n_beta.item()),
                integral_convention=convention,
                orbital_ordering=str(orbital_ordering.item()),
                schema_version=int(schema_version.item()),
                provenance=provenance,
            )
