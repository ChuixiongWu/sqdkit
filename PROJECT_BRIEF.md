# SQD Research Framework — Project Brief

## Goal

Create a minimal, reproducible framework that first reproduces a Qiskit-based sample-based quantum diagonalization (SQD) workflow on local simulators, while leaving a clean research entry point for replacing the state-preparation method later.

The project is not intended to reproduce IBM's tutorial structure verbatim or to build a general quantum-computing platform.

## Agreed scope for v0

1. Use Qiskit and `qiskit-addon-sqd` for the reference SQD implementation.
2. Run locally with finite-shot ideal simulation through Qiskit Aer.
3. Support an Aer noise path capable of using either synthetic noise or a real-backend-like target/noise snapshot.
4. Reuse upstream configuration recovery and SCI behavior. Expose only the main SQD parameters in the normal configuration surface; keep less common controls advanced.
5. Implement LUCJ as the first reference state-preparation method, while ensuring SQD post-processing does not depend on LUCJ.
6. Keep future state-preparation methods possible, especially adiabatic state preparation. Amplitude amplification is a later research option and requires an explicit oracle/target definition.
7. Use electronic integrals—not a Pauli operator—as the canonical chemistry representation.
8. Use a convenient, versioned NPZ file as the main persistent input format.
9. Provide one small end-to-end example and automated smoke test.

## Canonical chemistry input

The runtime problem representation should contain at least:

- one-body spatial-orbital integrals `h1`
- two-body spatial-orbital integrals `h2`
- constant/core/nuclear energy shift
- number of spatial orbitals
- `(n_alpha, n_beta)`
- integral convention
- orbital/bit ordering metadata
- optional active-space and fragment provenance

The main disk format should be a single versioned `.npz` file containing numeric arrays/scalars and a JSON metadata string. It must be readable with `allow_pickle=False`.

A direct PySCF producer/adapter is useful. FCIDUMP interoperability may be added later, but is not the main v0 format. Generic Pauli/spin-system SQD is out of scope for v0.

## State preparation and hardware co-design

Do not interpret “separation” as hardware ignorance. Use three conceptual stages:

1. **State-preparation specification** — what state transformation is desired, such as LUCJ or adiabatic evolution.
2. **Target-aware synthesis** — choose a realization using target capabilities such as connectivity, native entanglers, parallelism, analog blocks, ancillas, and reset.
3. **Physical compilation** — layout, routing, calibrated gates, and backend-specific optimization.

These stages may exchange explicit artifacts and target-capability metadata. Avoid hard-wiring a state-preparation implementation directly to one vendor backend object when a smaller capability contract is sufficient.

## Backend and noise model

A backend profile may include:

- coupling map/connectivity
- native basis gates or analog operations
- gate durations
- gate and readout errors
- coherence information
- compilation settings and seeds

Support the distinction between:

- ideal finite-shot sampling
- topology-only simulation
- topology plus synthetic noise
- calibration-derived backend-like noise

Connectivity is part of the target being simulated. For example, emulating a Rigetti-like device must not assume all-to-all connectivity.

## Normal user-facing parameters

Keep the first configuration surface small. Likely common parameters include:

- input path/problem
- state-preparation method and its main options
- backend/noise profile
- shots
- `samples_per_batch`
- `num_batches`
- `max_iterations`
- random seeds

Other upstream SQD and SCI controls can be grouped as advanced options rather than omitted permanently.

## Results

Persist enough information to reproduce and inspect a run:

- energy estimate and convergence status
- iteration/subspace summary
- shot count and valid/unique sample diagnostics
- state-preparation and compiled-circuit summary
- backend/noise profile
- effective configuration and seeds
- package versions and timing

FCI comparison is optional and should run only when explicitly requested for a sufficiently small problem.

## Non-goals for v0

- rewriting configuration recovery
- implementing a general plugin framework
- supporting every chemistry file format
- generic Pauli Hamiltonian workflows
- mandatory FCI references
- production HPC/distributed execution
- implementing multiple speculative state-preparation algorithms
- embedding scientific ablation suites into the core runtime API

## Acceptance criteria

A new user can:

1. create the environment from project metadata;
2. generate or load one small chemistry problem;
3. prepare and sample the reference LUCJ circuit with ideal finite shots;
4. run upstream SQD post-processing;
5. save a reproducible result artifact;
6. rerun with a simple Aer noise/backend profile;
7. execute automated tests and one end-to-end smoke example successfully.

## First OpenCode task

The first task is planning only. Inspect current upstream APIs, propose the smallest viable architecture and directory tree, identify assumptions, and write `docs/implementation-plan.md`. Do not install dependencies or create production source files until the plan is reviewed.
