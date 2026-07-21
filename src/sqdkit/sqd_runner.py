"""End-to-end orchestration for sample-based quantum diagonalization."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from time import perf_counter

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian

from sqdkit.config import AdvancedConfig, RunConfig, derived_seeds
from sqdkit.problem import ProblemData
from sqdkit.results import RunMetadata, SQDResult
from sqdkit.sampling import sample
from sqdkit.stateprep import StatePrep, to_sqd_bit_array
from sqdkit.synthesis import transpile_to_target
from sqdkit.targets import BackendProfile

_DISTRIBUTIONS = {
    "qiskit": "qiskit",
    "qiskit_addon_sqd": "qiskit-addon-sqd",
    "ffsim": "ffsim",
    "qiskit_aer": "qiskit-aer",
    "pyscf": "pyscf",
}


def _package_versions() -> dict[str, str]:
    """Collect installed dependency versions without importing optional packages."""
    versions: dict[str, str] = {}
    for key, distribution in _DISTRIBUTIONS.items():
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def run(
    problem: ProblemData,
    stateprep: StatePrep,
    profile: BackendProfile,
    config: RunConfig,
    advanced_config: AdvancedConfig | None = None,
) -> SQDResult:
    """Prepare, sample, and diagonalize a fermionic problem with SQD."""
    started_at = perf_counter()
    seeds = derived_seeds(config.seed)
    stateprep_result = stateprep.build(problem)
    pass_manager = stateprep.pass_manager(profile)
    prepared_circuit = stateprep_result.circuit.copy()
    prepared_circuit.measure_all()
    compiled, compiled_summary = transpile_to_target(
        prepared_circuit,
        profile,
        pass_manager=pass_manager,
        seed=seeds["transpiler"],
    )
    bit_array, sample_diagnostics = sample(
        compiled,
        profile,
        shots=config.shots,
        seed=seeds["sampler"],
    )
    sqd_input = BitArray.from_bool_array(
        to_sqd_bit_array(bit_array, problem.norb), order="big"
    )

    iterations = 0
    converged = False
    previous_energy: float | None = None
    previous_occupancies: np.ndarray | None = None
    subspace_summary: dict[str, dict[str, int | list[int]]] = {}
    controls = advanced_config or AdvancedConfig()

    def record_iteration(batch_results: list[SCIResult]) -> None:
        """Record SQD iteration diagnostics and its upstream convergence criterion."""
        nonlocal converged, iterations, previous_energy, previous_occupancies
        iterations += 1
        best_result = min(batch_results, key=lambda item: item.energy)
        subspace_summary[str(iterations)] = {
            "batch_count": len(batch_results),
            "dimensions": [
                int(result.sci_state.ci_strs_a.size * result.sci_state.ci_strs_b.size)
                for result in batch_results
            ],
        }
        occupancies = best_result.orbital_occupancies
        if previous_energy is not None and previous_occupancies is not None:
            converged = (
                abs(previous_energy - best_result.energy) < controls.energy_tol
                and np.linalg.norm(
                    np.ravel(previous_occupancies) - np.ravel(occupancies), ord=np.inf
                )
                < controls.occupancies_tol
            )
        previous_energy = best_result.energy
        previous_occupancies = occupancies

    sci_result = diagonalize_fermionic_hamiltonian(
        problem.h1,
        problem.h2,
        sqd_input,
        config.samples_per_batch,
        problem.norb,
        (problem.n_alpha, problem.n_beta),
        num_batches=config.num_batches,
        max_iterations=config.max_iterations,
        seed=seeds["sqd"],
        energy_tol=controls.energy_tol,
        occupancies_tol=controls.occupancies_tol,
        carryover_threshold=controls.carryover_threshold,
        symmetrize_spin=controls.symmetrize_spin,
        max_dim=controls.max_dim,
        callback=record_iteration,
    )

    stateprep_summary = {
        **stateprep_result.metadata,
        "norb": stateprep_result.norb,
        "num_qubits": stateprep_result.circuit.num_qubits,
        "depth": stateprep_result.circuit.depth(),
    }
    return SQDResult(
        total_energy=float(sci_result.energy + problem.core_energy),
        electronic_energy=float(sci_result.energy),
        core_energy=problem.core_energy,
        converged=converged,
        iterations=iterations,
        subspace_summary=subspace_summary,
        shots_total=sample_diagnostics["total_shots"],
        valid_count=sample_diagnostics["valid_hamming"],
        unique_count=sample_diagnostics["unique_bitstrings"],
        stateprep_summary=stateprep_summary,
        compiled_summary=compiled_summary,
        backend_profile_mode=profile.mode,
        backend_profile_fingerprint=profile.fingerprint(),
        effective_config=config.to_dict(),
        seeds=seeds,
        elapsed_seconds=perf_counter() - started_at,
        metadata=RunMetadata(
            seeds=seeds,
            package_versions=_package_versions(),
            backend_profile_mode=profile.mode,
            backend_profile_fingerprint=profile.fingerprint(),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
