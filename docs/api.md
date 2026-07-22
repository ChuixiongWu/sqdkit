# sqdkit API Reference

## Core workflow

A typical SQD run needs five things: a problem, a state-preparation method,
a backend profile, a run config, and the runner itself.

```python
from sqdkit.pyscf_driver import build_problem
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile
from sqdkit.config import RunConfig
from sqdkit.sqd_runner import run as sqd_run

problem = build_problem("H 0 0 0; H 0 0 0.735", basis="sto-3g")
stateprep = LucjStatePrep(n_reps=1)
profile = BackendProfile.ideal()
config = RunConfig(shots=1000, seed=42)

result = sqd_run(problem, stateprep, profile, config)
print(result.total_energy)
```

### `sqdkit.sqd_runner.run(problem, stateprep, profile, config, advanced_config=None) → SQDResult`

The main entry point. Wires state preparation → target-aware compilation → sampling
→ SQD diagonalization → result collection. Returns an `SQDResult` with energies,
convergence diagnostics, and full provenance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem` | `ProblemData` | Chemistry problem with h1, h2 integrals |
| `stateprep` | `StatePrep` | State preparation object implementing the protocol |
| `profile` | `BackendProfile` | Target capabilities and noise model |
| `config` | `RunConfig` | Shots, batches, iterations, seed |
| `advanced_config` | `AdvancedConfig \| None` | Convergence tolerances, spin symmetrisation, max subspace dim |

### `sqdkit.pyscf_driver.build_problem(atom, basis, charge=0, spin=0, active_space=None, with_ccsd=True) → ProblemData`

Build a `ProblemData` from a PySCF RHF calculation. The molecule is described by
`atom` (PySCF format string or list of `(symbol, (x,y,z))` tuples) and a `basis`
string (e.g. `"sto-3g"`, `"6-31g"`).

- If `active_space` is a list of spatial orbital indices (0-based), the integrals
  are restricted to those orbitals and frozen-core energy is added to `core_energy`.
- If `with_ccsd=True` (default), CCSD t1/t2 amplitudes are computed and stored
  in `provenance` for state-preparation methods that need them (e.g. LUCJ).
- Currently RHF only. UHF/ROHF support is not in v0 scope.

Examples:
```python
# Full active space, with CCSD amplitudes
problem = build_problem("H 0 0 0; H 0 0 0.735", basis="sto-3g")

# Active space selection (orbitals 2–9 for N2/6-31G)
problem = build_problem(
    [("N", (0, 0, 0)), ("N", (1.0, 0, 0))],
    basis="6-31g",
    active_space=list(range(2, 10)),
)
```

### `sqdkit.stateprep_lucj.LucjStatePrep(n_reps=2, interaction_pairs=(None, None))`

LUCJ (local unitary cluster Jastrow) state preparation. The only module that
imports ffsim. Requires CCSD t1/t2 amplitudes in the problem's `provenance`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_reps` | `2` | Number of LUCJ repetitions |
| `interaction_pairs` | `(None, None)` | `(pairs_aa, pairs_ab)` or None for all-to-all |

Implements the `StatePrep` protocol:
- `build(problem, target=None)` → `StatePrepResult` — constructs the circuit
- `pass_manager(profile)` → `PassManager | None` — returns ffsim's LUCJ-aware pass manager for topology/calibration modes

```python
lucj = LucjStatePrep(n_reps=1)
prep_result = lucj.build(problem)
# prep_result.circuit      — QuantumCircuit on 2*norb qubits
# prep_result.norb         — spatial orbital count
# prep_result.metadata     — {"method": "lucj"}
```

### `sqdkit.stateprep.ReferenceProductStatePrep(norb, n_alpha, n_beta)`

Hartree-Fock reference product state using X gates. An ffsim-free state-preparation
method that exists to prove the pipeline does not depend on LUCJ. Useful as a
baseline and as a template for writing your own state-preparation class.

```python
hf_prep = ReferenceProductStatePrep(
    norb=problem.norb,
    n_alpha=problem.n_alpha,
    n_beta=problem.n_beta,
)
```

### `sqdkit.targets.BackendProfile`

Portable backend-capability contract. Four factory methods:

| Method | Mode | Noise | Requires |
|--------|------|-------|----------|
| `.ideal(basis_gates=None, seed=0)` | ideal | none | — |
| `.topology_only(coupling_map, basis_gates, seed=0)` | topology | none | coupling map |
| `.synthetic_noise(coupling_map, basis_gates, gate_error, readout_error, seed=0)` | synthetic_noise | depolarizing + readout | coupling map |
| `.calibration_like(fake_backend_name)` | calibration | from IBM fake backend | `qiskit-ibm-runtime` |

Key methods:
- `fingerprint()` → `str` — stable SHA-256 digest of the profile
- `build_simulator()` → `AerSimulator` — for sampling
- `to_target()` → `qiskit.transpiler.Target` — for compilation
- `save_json(path)` / `load_json(path)` — persistence

```python
# Ideal (all-to-all, noiseless)
profile = BackendProfile.ideal()

# Linear topology with synthetic noise
profile = BackendProfile.synthetic_noise(
    coupling_map=[(0, 1), (1, 2), (2, 3)],
    basis_gates=["cx", "rz", "sx", "x", "id"],
    gate_error=0.001,
    readout_error=0.01,
    seed=42,
)

# IBM calibration snapshot (requires qiskit-ibm-runtime)
profile = BackendProfile.calibration_like("FakeSherbrooke")
```

### `sqdkit.config.RunConfig(...)`

Serializable run configuration. All fields have sensible defaults.

| Field | Default | Description |
|-------|---------|-------------|
| `input_path` | `"problem.npz"` | Path to NPZ problem file |
| `stateprep_method` | `"lucj"` | State preparation method name |
| `stateprep_options` | `{}` | Method-specific options |
| `backend_profile` | `"ideal"` | Profile mode string |
| `shots` | `10000` | Sampling shots |
| `samples_per_batch` | `100` | SQD batch size |
| `num_batches` | `10` | Number of SQD batches |
| `max_iterations` | `10` | Maximum SQD iterations |
| `seed` | `0` | Master seed (derives transpiler/sampler/sqd seeds) |

Methods: `to_dict()` / `from_dict(d)` for serialization. Validates all fields
on construction.

### `sqdkit.config.AdvancedConfig(...)`

Less-common SQD diagonalization controls, passed through directly to
`diagonalize_fermionic_hamiltonian`.

| Field | Default | Description |
|-------|---------|-------------|
| `energy_tol` | `1e-8` | Energy convergence tolerance |
| `occupancies_tol` | `1e-5` | Orbital occupancy convergence tolerance |
| `carryover_threshold` | `1e-4` | Configuration carryover threshold |
| `symmetrize_spin` | `False` | Enforce spin symmetry |
| `max_dim` | `None` | Maximum SCI subspace dimension |
| `spin_sq` | `None` | Target S² value |

---

## Conventions

### Qubit ordering

The framework uses a fixed convention to bridge Qiskit/ffsim circuits
and SQD post-processing:

| Layer | Convention |
|-------|-----------|
| Circuit (ffsim) | qubits ``[0, norb)`` = alpha, ``[norb, 2*norb)`` = beta |
| Qiskit samples | little-endian: column ``i`` = qubit ``i`` (alpha first, then beta) |
| SQD input (`qiskit-addon-sqd`) | ``[beta₀ … beta_{norb-1}, alpha₀ … alpha_{norb-1}]`` |

``sqdkit.stateprep.to_sqd_bit_array(bit_array, norb)`` performs the conversion
from Qiskit samples to SQD input layout in one call.

**If you write a custom ``StatePrep`` that uses a different qubit layout,**
you must either (a) ensure ``build()`` produces a circuit compatible with the
convention above, or (b) provide your own conversion to the SQD ``[beta…, alpha…]``
format before calling ``diagonalize_fermionic_hamiltonian``.

The conversion is trivial when the circuit respects spin blocks —
just split at ``norb`` and swap the halves.  See the source of
``to_sqd_bit_array`` for the one-line implementation.

---

## Internal pipeline (for customisation)

These functions are called internally by `sqd_runner.run()`. You only need them
if you are writing a custom workflow, a custom state-preparation method, or
debugging.

### `sqdkit.stateprep.StatePrep` (Protocol)

The state-preparation contract. Any class implementing this protocol can be
passed to `sqd_runner.run()`.

```python
class MyPrep:
    def build(self, problem: ProblemData, target=None) -> StatePrepResult:
        """Build a circuit on 2*norb qubits (alpha[0:norb), beta[norb:2norb))."""
        ...

    def pass_manager(self, profile: BackendProfile) -> PassManager | None:
        """Optional compilation passes. Return None for default transpilation."""
        ...
```

### `sqdkit.stateprep.StatePrepResult(circuit, norb, metadata)`

Returned by `StatePrep.build()`.

| Field | Type | Description |
|-------|------|-------------|
| `circuit` | `QuantumCircuit` | Prepared circuit (without measurements) |
| `norb` | `int` | Number of spatial orbitals |
| `metadata` | `dict` | At minimum `{"method": "..."}` |

### `sqdkit.stateprep.to_sqd_bit_array(bit_array, norb) → np.ndarray`

Convert Qiskit's spin-blocked bitstrings (alpha[0:norb), beta[norb:2norb)) to
SQD's expected `[beta..., alpha...]` ordering. This is correctness-critical —
a wrong convention produces physically wrong but plausible-looking energies.

### `sqdkit.synthesis.transpile_to_target(circuit, profile, pass_manager=None, seed=None) → (QuantumCircuit, dict)`

Compile a circuit for a target profile. If a `pass_manager` is supplied (from
the state-preparation method), it is used; otherwise `qiskit.transpile` is
called with the profile's target at optimisation level 1.

### `sqdkit.sampling.sample(compiled_circuit, profile, shots, seed) → (BitArray, dict)`

Sample a compiled circuit with seeded `SamplerV2`. Returns the `BitArray` plus
diagnostics: `total_shots`, `unique_bitstrings`, `valid_hamming`.

---

## Persistence

### `sqdkit.problem.ProblemData`

Chemistry problem: spatial-orbital integrals plus metadata.

| Field | Type | Description |
|-------|------|-------------|
| `h1` | `np.ndarray (norb, norb)` | One-body MO integrals (chemist convention) |
| `h2` | `np.ndarray (norb, norb, norb, norb)` | Two-body MO integrals `(ij\|kl)` |
| `core_energy` | `float` | Nuclear repulsion + frozen-core energy |
| `norb` | `int` | Number of spatial orbitals |
| `n_alpha` | `int` | Alpha electrons |
| `n_beta` | `int` | Beta electrons |
| `integral_convention` | `str` | Always `"chemist"` |
| `orbital_ordering` | `str` | Convention string |
| `schema_version` | `int` | Schema version (currently 1) |
| `provenance` | `dict` | Source metadata (basis, geometry, t1/t2, ...) |

Methods:
- `save_npz(path)` — persist as NPZ (v1 schema, readable with `allow_pickle=False`)
- `ProblemData.load_npz(path)` — load and validate an NPZ artifact

```python
problem.save_npz("h2.npz")
reloaded = ProblemData.load_npz("h2.npz")
```

### `sqdkit.results.SQDResult`

Complete result of one SQD run. Key fields: `total_energy`, `electronic_energy`,
`core_energy`, `converged`, `iterations`, `subspace_summary`, `shots_total`,
`stateprep_summary`, `compiled_summary`, `backend_profile_fingerprint`.

Methods:
- `save(path)` → NPZ (energies) + JSON (metadata sidecar)
- `SQDResult.load(path)` → reload with `allow_pickle=False`

### `sqdkit.results.RunMetadata`

Provenance recorded alongside every result: `schema_version`, `seeds` (transpiler,
sampler, sqd), `package_versions`, `backend_profile_mode`, `backend_profile_fingerprint`,
`created_at` (ISO8601). Missing required keys raise `ValueError`.

---

## Validation

### `sqdkit.validation.fci_reference(problem, max_norb=8) → float`

Compute the exact FCI total energy via PySCF for a small problem. Returns total
energy (electronic + core_energy). Raises `ValueError` if `problem.norb > max_norb`.

```python
from sqdkit.validation import fci_reference
fci_total = fci_reference(problem)
error_mHa = abs(result.total_energy - fci_total) * 1000
```
