#!/usr/bin/env python3
"""End-to-end H2 LUCJ SQD smoke example."""

from sqdkit.config import RunConfig
from sqdkit.pyscf_driver import build_problem
from sqdkit.sqd_runner import run as sqd_run
from sqdkit.stateprep_lucj import LucjStatePrep
from sqdkit.targets import BackendProfile


def main() -> None:
    """Prepare H2, run ideal and topology-constrained SQD, and save both results."""
    problem = build_problem(
        "H 0 0 0; H 0 0 0.735", basis="sto-3g", charge=0, spin=0
    )
    print(f"Problem: norb={problem.norb}, nelec=({problem.n_alpha},{problem.n_beta})")

    config = RunConfig(
        shots=1000,
        samples_per_batch=50,
        num_batches=2,
        max_iterations=5,
        seed=42,
    )
    lucj = LucjStatePrep(n_reps=1)

    print("Running LUCJ + ideal Aer...")
    ideal_result = sqd_run(problem, lucj, BackendProfile.ideal(), config)
    print(
        f"Ideal total energy: {ideal_result.total_energy:.6f} Ha, "
        f"converged={ideal_result.converged}"
    )
    ideal_result.save("h2_ideal_result")

    noisy_profile = BackendProfile.synthetic_noise(
        coupling_map=[(0, 1), (1, 2), (2, 3)],
        basis_gates=["cx", "rz", "sx", "x", "id"],
        gate_error=0.001,
        readout_error=0.01,
        seed=42,
    )
    print("Running LUCJ + synthetic_noise Aer...")
    noisy_result = sqd_run(problem, lucj, noisy_profile, config)
    print(
        f"Synthetic-noise total energy: {noisy_result.total_energy:.6f} Ha, "
        f"converged={noisy_result.converged}"
    )
    noisy_result.save("h2_noisy_result")

    print("Done! Artifacts saved.")


if __name__ == "__main__":
    main()
