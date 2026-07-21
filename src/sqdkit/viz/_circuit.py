"""Circuit drawing utilities for StatePrepResult.plot()."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit

matplotlib.use("Agg")


def circuit_to_text(circuit: QuantumCircuit) -> str:
    """Return an ASCII line-drawing of a circuit."""
    drawing = circuit.draw(output="text", idle_wires=False)
    return drawing.single_string()


def circuit_to_png(circuit: QuantumCircuit, path: str | Path) -> None:
    """Save a circuit as a PNG via matplotlib.

    Uses a reduced scale and wider layout to accommodate long gate names
    from composite operations (e.g. ffsim's ``hartree_fock_jw``).
    """
    n_qubits = circuit.num_qubits
    depth = circuit.depth()
    width = max(10, depth * 0.35)
    height = max(2, n_qubits * 0.6)
    fig, ax = plt.subplots(figsize=(width, height))
    circuit.draw(output="mpl", ax=ax, idle_wires=False, scale=0.35, fold=-1)
    ax.set_title(f"qubits={n_qubits}, depth={depth}")
    fig.tight_layout()
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
