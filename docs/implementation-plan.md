# SQD Research Framework — v0 Implementation Plan

Status: **planning complete, awaiting go-ahead to implement**. This document is the deliverable of the first task (brief §"First OpenCode task"). No dependencies have been installed and no production source has been created. The machine-executable task breakdown lives in `.omo/plans/sqd-v0-framework.md`.

> NOTE: This is the reviewed, verbatim source for `docs/implementation-plan.md`. The planning agent (Prometheus) is restricted to writing under `.omo/`; the worker copies this file to `docs/implementation-plan.md` as the first action of `/start-work` (todo D0).

Package name: **`sqdkit`**.

---

## 1. Upstream API inspection (verified against current releases)

All signatures below were confirmed by reading current docs/source in July 2026. They drive the exact adapters chosen in §5.

### 1.1 `qiskit-addon-sqd` — SQD post-processing (reused, not rewritten)
- Version **0.12.x** (0.12.0 stable 2025-09-11; 0.12.1 pre-release 2026-01-16). Requires `qiskit>=1.2,<3`, Python `>=3.10`.
- **High-level entry point (the v0 path):**
  ```python
  qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian(
      one_body_tensor, two_body_tensor, bit_array,
      samples_per_batch, norb, nelec, *,
      num_batches=1, energy_tol=1e-8, occupancies_tol=1e-5, max_iterations=100,
      sci_solver=None, symmetrize_spin=False, max_dim=None,
      include_configurations=None, initial_occupancies=None,
      carryover_threshold=1e-4, callback=None, seed=None,
  ) -> SCIResult
  ```
  It runs the **entire configuration-recovery → subsample → SCI** loop internally — exactly the "reuse upstream recovery and SCI" mandate (AGENTS.md:14).
- **Two correctness-critical facts:**
  - It takes **no** nuclear/core-energy argument. `SCIResult.energy` is the **electronic** energy; we add `core_energy` ourselves.
  - `bit_array` must be ordered **`[beta_N…beta_0, alpha_N…alpha_0]`**, width `2*norb`.
- `SCIResult` = `energy`, `sci_state`, `orbital_occupancies`, optional `rdm1/rdm2`. Lower-level `solve_sci_batch` (default solver), `subsample`, `postselect_by_hamming_right_and_left` exist; `postselect_and_subsample`/`post_select_by_hamming_weight` are **deprecated** in 0.12.0. v0 uses only the high-level function.

### 1.2 `ffsim` — LUCJ state preparation (isolated dependency)
- Version **0.0.83** (2026-07-18). Requires **`qiskit>=2.0`**, `pyscf>=2.12`. *This is the constraint that pins the whole stack to the Qiskit 2.x line.*
- Ansatz: `ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, *, t1=None, n_reps=None, interaction_pairs=(pairs_aa, pairs_ab), ...)`.
- Circuit: `ffsim.qiskit.PrepareHartreeFockJW(norb, nelec)` + `ffsim.qiskit.UCJOpSpinBalancedJW(op)` on **`2*norb`** qubits, layout **alpha = qubits `[0, norb)`, beta = qubits `[norb, 2*norb)`**.
- Hardware-aware: `ffsim.qiskit.generate_lucj_pass_manager(backend, norb, connectivity, interaction_pairs, optimization_level)` → a Qiskit `PassManager`.

### 1.3 `qiskit-aer` + Qiskit `Target` — simulation, noise, capabilities
- `qiskit-aer` **0.17.1** (2025-06-04), `qiskit>=1.1` (supports 2.x).
- Sampling: `qiskit_aer.primitives.SamplerV2(seed=…)` → `pub_result.data.<creg>` is a `BitArray`; `seed` gives reproducibility.
- Noise: `qiskit_aer.noise.NoiseModel` with `depolarizing_error`/`thermal_relaxation_error`/`ReadoutError`; `NoiseModel.from_backend(backend)`; `AerSimulator.from_backend(backend)`.
- Capabilities: Qiskit `Target` exposes coupling map, native basis gates, gate durations, gate/readout errors, qubit `T1/T2`. Synthetic topology via `GenericBackendV2` + `CouplingMap.from_line/from_grid`; offline **calibration-derived** backends via `qiskit_ibm_runtime.fake_provider` (e.g. `FakeSherbrooke`).

### 1.4 PySCF — canonical chemistry integrals (no qiskit-nature)
- `gto.M` / `scf.RHF`; `h1 = C.T @ mf.get_hcore() @ C`; `h2 = ao2mo.incore.full(eri, C)` → full 4-index **chemist** tensor `(ij|kl)`; `core_energy = mol.energy_nuc()`; `norb = mol.nao_nr()`; `nelec = mol.nelec`. CCSD `t1/t2` via `pyscf.cc.CCSD`. Optional FCI via `pyscf.fci`. Active space via manual slice or `pyscf.mcscf.CASCI`.
- Decision: depend on **PySCF directly**. The brief wants electronic integrals (not a Pauli operator) as the canonical representation; qiskit-nature would add abstraction and physicist-notation conversion we do not need.

### 1.5 Reconciliation note
Two research passes disagreed on the SQD surface: one reproduced the older tutorial API (manual `recover_configurations` + `postselect_and_subsample` + `solve_fermion`). The authoritative source-read of v0.12.1 confirms the **single high-level `diagonalize_fermionic_hamiltonian`** is current and encapsulates recovery — this is the v0 choice.

---

## 2. Design principles

Minimal, single-researcher-maintainable, YAGNI (AGENTS.md:15). Reproduce first, don't redesign (AGENTS.md:11). Keep the five concerns conceptually distinct — **state-preparation specification → target-aware synthesis → physical compilation → sampling → SQD post-processing** — but implement them as flat modules, not a plugin framework. The single most important architectural property: **state preparation is the research entry point**, so its contract is dependency-free and swappable, and **SQD never depends on LUCJ** (brief:15).

---

## 3. Architecture & data flow

```
molecule spec ──▶ pyscf_driver ──▶ ProblemData ──▶ (NPZ on disk, allow_pickle=False)
                                        │
                                        ▼
        StatePrep.build(problem, target) ──▶ QuantumCircuit  (+ optional PassManager)
                                        │
                                        ▼
        synthesis.transpile_to_target(circuit, profile, pass_manager?) ──▶ compiled circuit
                                        │
                                        ▼
        sampling.sample(compiled, profile, shots, seed) ──▶ BitArray
                                        │
              to_sqd_bit_array (lock alpha/beta convention)
                                        ▼
   diagonalize_fermionic_hamiltonian(h1, h2, bit_array, …, seed) ──▶ SCIResult
                                        │            (+ core_energy)
                                        ▼
                              SQDResult ──▶ NPZ (numeric) + JSON (metadata sidecar)
```

`BackendProfile` (a small capability contract) is the only thing synthesis/sampling see — never a vendor backend object (brief:46).

---

## 4. Directory tree

```
lab/SQD/
├── pyproject.toml            # project metadata + pinned deps (Qiskit 2.x stack) + pytest/ruff config
├── uv.lock                   # committed lockfile
├── README.md                 # env-from-metadata, quickstart, "add your own state prep"
├── docs/
│   └── implementation-plan.md   # this document
├── src/
│   └── sqdkit/
│       ├── __init__.py          # __version__, public exports
│       ├── py.typed
│       ├── logging.py           # module-scoped logger helper (no import-time root config)
│       ├── config.py            # RunConfig + AdvancedConfig, seed derivation, validation
│       ├── problem.py           # ProblemData + versioned NPZ I/O (allow_pickle=False)
│       ├── pyscf_driver.py      # PySCF → ProblemData (+ optional CCSD, active space)
│       ├── stateprep.py         # ffsim-FREE StatePrep protocol + ordering helper + HF reference
│       ├── stateprep_lucj.py    # LUCJ via ffsim  ← ONLY module importing ffsim
│       ├── targets.py           # BackendProfile + 4 sampling modes  ← ONLY module importing qiskit_ibm_runtime
│       ├── synthesis.py         # ffsim-agnostic transpile + StatePrep-supplied pass manager
│       ├── sampling.py          # seeded SamplerV2 → BitArray + diagnostics
│       ├── sqd_runner.py        # orchestrate diagonalize_fermionic_hamiltonian (+ core_energy)
│       ├── results.py           # SQDResult + RunMetadata (required keys) + NPZ/JSON save
│       └── validation.py        # optional, guarded PySCF FCI reference
├── examples/
│   └── h2_lucj_smoke.py         # end-to-end ideal + noisy rerun
└── tests/
    ├── test_env_versions.py             # pinned-version fail-fast
    ├── test_config.py
    ├── test_problem_npz.py              # round-trip, allow_pickle=False, no object dtype
    ├── test_pyscf_driver.py
    ├── test_bit_ordering.py             # locks ffsim↔SQD alpha/beta convention
    ├── test_stateprep_lucj.py
    ├── test_no_ffsim_import_outside_lucj.py   # AST import guard
    ├── test_targets.py
    ├── test_synthesis.py
    ├── test_sampling.py
    ├── test_sqd_runner.py               # H2 energy≈FCI; ffsim-free independence proof
    ├── test_results.py
    ├── test_validation.py
    ├── test_smoke_end_to_end.py         # H2 fast (default lane)
    └── test_n2_active_space.py          # N2 (10e,8o), @pytest.mark.slow, timeout, nightly lane
```

`cli.py` is intentionally **absent** (YAGNI; the brief's acceptance list needs an example + tests, not a CLI).

---

## 5. Module responsibilities & key adapters

| Module | Responsibility | Upstream adapter |
| --- | --- | --- |
| `config.py` | Common params (input, method, profile, shots, `samples_per_batch`, `num_batches`, `max_iterations`, `seed`) + `AdvancedConfig` pass-through for less-common SQD knobs | feeds `diagonalize_fermionic_hamiltonian` kwargs |
| `problem.py` | `ProblemData` + versioned NPZ (§6) | numpy `savez`/`load(allow_pickle=False)` |
| `pyscf_driver.py` | Molecule → integrals/CCSD/active space | PySCF §1.4 |
| `stateprep.py` | `StatePrep` protocol, ordering contract, HF reference | pure Qiskit |
| `stateprep_lucj.py` | LUCJ ansatz + owns LUCJ pass manager | ffsim §1.2 |
| `targets.py` | `BackendProfile`, 4 sim modes, fingerprint | Aer + fake_provider §1.3 |
| `synthesis.py` | Target-aware transpile + physical compilation | `qiskit.transpile` (+ injected PassManager) |
| `sampling.py` | Seeded sampling → `BitArray` | `SamplerV2(seed=…)` |
| `sqd_runner.py` | Orchestration + `core_energy` add + convergence surfacing | `diagonalize_fermionic_hamiltonian` |
| `results.py` | `SQDResult` + required-key `RunMetadata` | numpy + json |
| `validation.py` | Guarded optional FCI | `pyscf.fci` |

---

## 6. Canonical NPZ schema (v1)

Loaded with `np.load(path, allow_pickle=False)`. **No `object` dtype anywhere** — optional provenance/active-space go into the `metadata_json` string.

| Key | dtype / shape | Notes |
| --- | --- | --- |
| `h1` | float64 `(norb,norb)` | one-body MO integrals |
| `h2` | float64 `(norb,norb,norb,norb)` | two-body MO integrals, **chemist** `(ij|kl)` |
| `core_energy` | float64 (0-d) | nuclear/core constant |
| `norb` | int64 (0-d) | spatial orbitals |
| `n_alpha`, `n_beta` | int64 (0-d) | electron counts |
| `integral_convention` | `<U…` | literal `{"chemist"}` |
| `orbital_ordering` | `<U…` | `"spin_blocks:alpha[0:norb),beta[norb:2norb)"` |
| `schema_version` | int64 (0-d) | `1` |
| `metadata_json` | `<U…` | JSON string: source, basis, geometry, active-space & fragment provenance, package versions |

---

## 7. State-preparation contract & the extension point

The contract is **ffsim-free** so future research can replace state preparation without touching the rest:

```python
class StatePrep(Protocol):
    def build(self, problem: ProblemData, target: BackendProfile | None) -> StatePrepResult: ...
    def pass_manager(self, profile: BackendProfile) -> PassManager | None: ...  # default None
```
`StatePrepResult = (circuit: QuantumCircuit over 2*norb qubits, norb, metadata)`.

- **Bit-ordering convention (correctness-critical):** qubit `i` = alpha spatial orbital `i` for `i<norb`; qubit `norb+i` = beta spatial orbital `i`. `to_sqd_bit_array()` maps a measured `BitArray` to SQD's `[beta…, alpha…]` expectation. A deterministic **known-state** test (prepare a fixed occupation, zero sampling noise) asserts the decoded per-half popcounts equal the known `n_alpha`/`n_beta`, tied to the exact convention string. This is the guard against a silent, physically-wrong-but-plausible energy.
- **`LucjStatePrep`** is the only ffsim importer; it also *owns* the LUCJ `PassManager` (built via `generate_lucj_pass_manager`) and hands it to `synthesis.py` as an opaque object, so `synthesis.py` stays ffsim-agnostic. An AST test enforces the isolation on every wave.
- **To add your own method:** implement `build()` returning a plain Qiskit circuit honoring the ordering convention (import Qiskit, **not** ffsim). Reuse the generic `synthesis.transpile_to_target()`; supply a custom `pass_manager()` only if you want specialized synthesis. SQD is unaffected — it consumes only the `BitArray`. A trivial ffsim-free `ReferenceProductStatePrep` (HF occupation via X gates) ships as the living proof of this path and as the independence test for "SQD does not depend on LUCJ".

---

## 8. Backend / target capability contract & the four sampling modes

`BackendProfile` carries coupling map, native basis gates, gate durations, gate/readout errors, coherence, and a seed, plus a stable `fingerprint()`. Builders:

| Mode | Construction |
| --- | --- |
| (a) ideal finite-shot | `AerSimulator()`, no noise |
| (b) topology-only | `GenericBackendV2`/`CouplingMap` (e.g. `from_line` → Rigetti-like, **not** all-to-all), no noise |
| (c) topology + synthetic noise | `NoiseModel` with depolarizing/thermal/readout on a coupling map |
| (d) calibration-derived backend-like | `qiskit_ibm_runtime.fake_provider` + `NoiseModel.from_backend` / `AerSimulator.from_backend` |

`qiskit_ibm_runtime` is imported **only** in `targets.py`; the rest of the pipeline consumes the vendor-neutral `BackendProfile` (honors brief:46 "no vendor-backend hard-wiring"). Requesting a noise/calibration mode **without** a coupling map raises a clear config error (no silent default topology).

---

## 9. SQD orchestration

`sqd_runner.run()` wires `stateprep → synthesis → sampling → to_sqd_bit_array → diagonalize_fermionic_hamiltonian(h1, h2, bit_array, samples_per_batch, norb, (n_alpha,n_beta), num_batches, max_iterations, energy_tol, seed=sqd_seed, callback)`. **Total energy = `SCIResult.energy + core_energy`.** The `callback` records per-iteration energy + subspace dimensions; non-convergence is surfaced as a queryable flag on `SQDResult`, never swallowed.

---

## 10. Result artifact & reproducibility contract

- **Seeds:** one `RunConfig.seed` → `transpiler_seed=seed`, `sampler_seed=seed+1`, `sqd_seed=seed+2`; all recorded. Identical seed+config ⇒ identical `BitArray` counts and identical energy; a different seed must change the counts (a "seed actually has effect" check).
- **`SQDResult`** persists total & electronic energy, `core_energy`, convergence + iteration/subspace summary, shot + valid/unique diagnostics, state-prep & compiled-circuit summaries, backend mode + fingerprint, effective config + seeds, package versions, timing.
- **`RunMetadata` required keys** (missing ⇒ `ValueError`): `schema_version`, `seeds{transpiler,sampler,sqd}`, `package_versions{qiskit,qiskit_addon_sqd,ffsim,qiskit_aer,pyscf}`, `backend_profile_mode`, `backend_profile_fingerprint`, `created_at`. Saved as NPZ (numeric) + JSON sidecar, `allow_pickle=False`.

---

## 11. Test strategy

Tests-after + pytest; every module has happy + failure paths (agent-executed). Correctness locks: bit-ordering known-state test; **H2/STO-3G total energy within 1e-3 Ha of FCI** (locks the integral/bit conventions end-to-end); ffsim-free HF-prep run through `sqd_runner` (independence proof); AST import guards for ffsim and qiskit-ibm-runtime; NPZ `allow_pickle=False` round-trip fidelity; version fail-fast. **H2** is the fast default lane; **N2 (10e,8o)** is `@pytest.mark.slow` with a `pytest-timeout` ceiling, a fixed seed, optional/skippable FCI, gated to a separate/nightly job (keeps expensive science out of the core runtime and the default CI, per AGENTS.md:21).

---

## 12. Dependencies & version constraints

Python **3.11**; `uv` + project-local `.venv` + committed `uv.lock`. Because `ffsim>=0.0.83` requires `qiskit>=2.0`, the whole stack sits on the **Qiskit 2.x** line — mutually compatible with `qiskit-addon-sqd 0.12.x` (`<3`) and `qiskit-aer 0.17.1`. Core deps: `qiskit>=2,<3`, `qiskit-aer>=0.17,<0.18`, `qiskit-addon-sqd>=0.12,<0.13`, `ffsim>=0.0.83`, `pyscf>=2.12`, `qiskit-ibm-runtime`, `numpy`. Dev: `pytest`, `pytest-timeout`, `ruff`. A W1 test asserts resolved versions fall in-range with a named failure, and `uv lock --check` guards against silent resolution drift. "Create env from metadata" (brief:112) = `uv sync` from `pyproject.toml` + `uv.lock`.

---

## 13. Assumptions, resolved decisions & open risks

**Resolved decisions:** package `sqdkit`; PySCF-direct integrals (no qiskit-nature); high-level `diagonalize_fermionic_hamiltonian` (reuse recovery); `SamplerV2` seeded sampling; NPZ+JSON results; flat module layout; `cli.py` dropped; include `qiskit-ibm-runtime` for mode (d), isolated to `targets.py`; H2 fast + N2 slow tests.

**Assumptions to keep visible:**
- The exact scaling/antisymmetrization of `two_body_tensor` expected by `diagonalize_fermionic_hamiltonian` is a cross-API bridge that is **not** independently confirmed in docs — it is *validated empirically* by the H2 energy-vs-FCI test. If that test fails, revisit the h2 convention in `pyscf_driver.py` before anything else.
- The ffsim JW layout ↔ SQD `[beta…,alpha…]` ordering is validated by the known-state test rather than assumed.
- The Qiskit 2.x window is narrow (three independently versioned add-ons); the lockfile + version test are the guardrails.

**Non-goals (v0):** rewriting recovery/SCI; plugin framework; every chemistry file format; generic Pauli workflows; mandatory FCI; production HPC; multiple speculative state-prep algorithms; ablation suites in the core runtime API (brief:97-106).

---

## 14. Execution overview

17 build todos across 6 waves + a 4-part final verification wave (plus todo D0 that materializes this document at `docs/implementation-plan.md`); full acceptance criteria, QA scenarios, dependency matrix, and commit plan are in `.omo/plans/sqd-v0-framework.md`. Waves: (1) environment & packaging, (2) problem + NPZ persistence, (3) state preparation, (4) targets/synthesis/sampling, (5) SQD orchestration/results/validation, (6) example/tests/docs. Implementation begins only when you start a worker session (e.g. `/start-work`); this task ends at the plan.
