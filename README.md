# sqdkit

Research-oriented sample-based quantum diagonalization (SQD) framework.

[中文版 (Chinese)](README.zh.md)

For the implementation details and project architecture, see
[`docs/implementation-plan.md`](docs/implementation-plan.md).
API reference: [`docs/api.md`](docs/api.md).
LUCJ design notes: [`docs/lucj-design-notes.md`](docs/lucj-design-notes.md).

## Environment (from project metadata)

```bash
uv sync
uv run python -c "import sqdkit; print(sqdkit.__version__)"
```

## Quickstart

```bash
uv run python examples/h2_lucj_smoke.py
```

Runs H₂ LUCJ state preparation with ideal + noisy Aer sampling, performs SQD
diagonalization, and saves reproducible result artifacts.

## How to add your own state-preparation method

The framework's state preparation interface is **ffsim-free**. Write a class
implementing the `StatePrep` protocol:

```python
from sqdkit.stateprep import StatePrep, StatePrepResult
from qiskit import QuantumCircuit


class MyAdiabaticPrep:
    def __init__(self, evolution_time: float = 1.0):
        self.evolution_time = evolution_time

    def build(self, problem, target=None) -> StatePrepResult:
        circuit = QuantumCircuit(2 * problem.norb)
        # Build your custom state-preparation circuit here.
        # Qubit convention: alpha orbitals = qubits [0, norb), beta = [norb, 2*norb).
        # SQD receives measured bitstrings in [beta..., alpha...] order.
        ...
        return StatePrepResult(
            circuit=circuit,
            norb=problem.norb,
            metadata={"method": "my_method"},
        )

    def pass_manager(self, profile):
        return None  # Or return a custom Qiskit PassManager.
```

Pass your stateprep to `sqd_runner.run()` — it works without any change to
config, backend, or SQD.

## Testing

```bash
uv run pytest -q -m "not slow"  # fast suite (unit + H2 E2E)
uv run pytest -q -m slow        # integration (N2 active space)
```

## Package

```bash
sqdkit  # import name
```

## Dependencies

Qiskit 2.x, ffsim, PySCF, qiskit-aer, qiskit-addon-sqd, numpy.
