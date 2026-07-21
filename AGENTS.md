# AGENTS.md

## Project intent

Build a small, research-oriented SQD framework. The first milestone is a reproducible local implementation based on Qiskit; future research will mainly modify state preparation rather than SQD recovery.

Read `PROJECT_BRIEF.md` before planning or coding.

## Working principles

- Start by reproducing the upstream Qiskit SQD workflow; do not redesign the algorithm prematurely.
- Keep state preparation, target-aware synthesis, physical compilation, sampling, and SQD post-processing conceptually distinct, while allowing hardware-aware co-design through explicit target capabilities.
- Treat LUCJ as the first reference implementation, not as the framework itself.
- Reuse `qiskit-addon-sqd` recovery and SCI behavior in v0. Do not fork or rewrite recovery unless explicitly requested.
- Prefer a minimal design that one researcher can understand and maintain. Apply YAGNI; avoid speculative plugin systems and deep class hierarchies.
- Expose a small set of common parameters first. Advanced upstream parameters may remain in an advanced configuration section or pass-through object.
- Use Python 3.11, `uv`, a project-local `.venv`, `pyproject.toml`, and a lockfile.
- Use a `src/` layout and pytest unless inspection reveals a strong reason not to.
- Keep persistent chemistry inputs simple and versioned. The proposed main format is NPZ with numeric arrays/scalars plus `metadata_json`; load with `allow_pickle=False`.
- FCI comparison is optional validation for small systems, never a mandatory runtime step.
- Ablations and expensive scientific benchmarks belong in tests/benchmarks or examples, not in the core runtime API.
- Preserve reproducibility: record package versions, configuration, seeds, backend/target metadata, and essential run diagnostics.

## Development discipline

- Inspect current official APIs before choosing exact adapters or signatures.
- Plan before implementing. State assumptions and unresolved choices.
- Add focused tests for input validation, reproducibility, and one end-to-end smoke workflow.
- Actually run tests and the smoke workflow; do not report success without real output.
- Do not silently broaden scope beyond the v0 acceptance criteria in `PROJECT_BRIEF.md`.
- Do not commit credentials, backend tokens, large generated artifacts, or virtual environments.
