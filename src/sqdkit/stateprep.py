"""Qiskit-only state preparation contracts and reference product states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BitArray
from qiskit.transpiler import PassManager, Target

from sqdkit.problem import ProblemData
from sqdkit.targets import BackendProfile


@dataclass(frozen=True, slots=True)
class StatePrepResult:
    """A prepared circuit and the orbital metadata required by SQD."""

    circuit: QuantumCircuit
    norb: int
    metadata: dict[str, str]

    def plot(self, fmt: str = "text", path: str | None = None) -> None:
        """Draw the state-preparation circuit.

        Parameters
        ----------
        fmt : str
            ``"text"`` prints ASCII art to stdout.
            ``"png"`` saves a matplotlib rendering to *path*.
        path : str or None
            Output file path (required when ``fmt="png"``).
        """
        if fmt not in {"text", "png"}:
            raise ValueError(f"unknown format {fmt!r}; expected 'text' or 'png'")

        if fmt == "text":
            from sqdkit.viz._circuit import circuit_to_text

            print(circuit_to_text(self.circuit))
        else:
            if path is None:
                raise ValueError("path is required for png format")
            from sqdkit.viz._circuit import circuit_to_png

            circuit_to_png(self.circuit, path)


@runtime_checkable
class StatePrep(Protocol):
    """Prepare an electronic state and optionally supply compilation passes."""

    def build(
        self, problem: ProblemData, target: Target | None = None
    ) -> StatePrepResult:
        """Build a state-preparation circuit for a problem."""

    def pass_manager(self, profile: BackendProfile) -> PassManager | None:
        """Return optional state-preparation-specific compilation passes."""


def to_sqd_bit_array(bit_array: BitArray, norb: int) -> np.ndarray:
    """Convert spin-blocked Qiskit samples to SQD's beta-then-alpha layout.

    Qiskit columns in little-endian order map directly to qubit indices.  The
    circuit convention is ``alpha[0:norb),beta[norb:2norb)``; SQD expects
    ``[beta...,alpha...]``.
    """
    if bit_array.num_bits != 2 * norb:
        msg = f"bit array width must equal 2*norb ({2 * norb})"
        raise ValueError(msg)
    qubit_ordered = bit_array.to_bool_array(order="little")
    return np.concatenate(
        (qubit_ordered[..., norb:], qubit_ordered[..., :norb]), axis=-1
    )


@dataclass(frozen=True, slots=True)
class ReferenceProductStatePrep:
    """Prepare the Hartree--Fock reference occupation with X gates."""

    norb: int
    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        """Validate the occupation counts at construction."""
        if self.norb < 0:
            raise ValueError("norb must be non-negative")
        if not 0 <= self.n_alpha <= self.norb:
            raise ValueError("n_alpha must be between zero and norb")
        if not 0 <= self.n_beta <= self.norb:
            raise ValueError("n_beta must be between zero and norb")

    def build(
        self, problem: ProblemData, target: Target | None = None
    ) -> StatePrepResult:
        """Build the spin-blocked Hartree--Fock product state."""
        del problem, target
        circuit = QuantumCircuit(2 * self.norb)
        for qubit in range(self.n_alpha):
            circuit.x(qubit)
        for qubit in range(self.norb, self.norb + self.n_beta):
            circuit.x(qubit)
        return StatePrepResult(
            circuit=circuit,
            norb=self.norb,
            metadata={"method": "reference_hf"},
        )

    def pass_manager(self, profile: BackendProfile) -> PassManager | None:
        """Use Qiskit's default transpilation for the product-state circuit."""
        del profile
        return None
