"""ffsim-backed local unitary cluster Jastrow state preparation."""

from __future__ import annotations

import ffsim
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager, Target

from sqdkit.problem import ProblemData
from sqdkit.stateprep import StatePrepResult
from sqdkit.targets import BackendProfile


class LucjStatePrep:
    """Build LUCJ circuits and their ffsim-specific compilation passes."""

    def __init__(
        self,
        n_reps: int = 2,
        interaction_pairs: tuple[
            list[tuple[int, int]] | None, list[tuple[int, int]] | None
        ] = (None, None),
    ) -> None:
        self.n_reps = n_reps
        self.interaction_pairs = interaction_pairs
        self._norb: int | None = None
        self._unbalanced: bool = False

    def build(
        self, problem: ProblemData, target: Target | None = None
    ) -> StatePrepResult:
        """Construct the Hartree--Fock reference followed by the LUCJ ansatz."""
        del target
        if problem.n_alpha != problem.n_beta:
            self._unbalanced = True
            return self._build_spin_unbalanced(problem)
        self._unbalanced = False
        t2 = np.array(problem.provenance["t2"])
        raw_t1 = problem.provenance.get("t1")
        t1 = None if raw_t1 is None else np.array(raw_t1)
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=self.n_reps,
            interaction_pairs=self.interaction_pairs,
        )
        circuit = QuantumCircuit(2 * problem.norb)
        circuit.append(
            ffsim.qiskit.PrepareHartreeFockJW(
                problem.norb, (problem.n_alpha, problem.n_beta)
            ),
            range(2 * problem.norb),
        )
        circuit.append(
            ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), range(2 * problem.norb)
        )
        self._norb = problem.norb
        return StatePrepResult(circuit, norb=problem.norb, metadata={"method": "lucj"})

    def _build_spin_unbalanced(self, problem: ProblemData) -> StatePrepResult:
        """Open-shell branch: consume the UCCSD amplitude tuples from todo 9.

        The amplitudes are stored as nested JSON lists in exactly the tuple
        order ``UCJOpSpinUnbalanced.from_t_amplitudes`` expects: ``t1`` is
        ``[t1a, t1b]`` and ``t2`` is ``[t2aa, t2ab, t2bb]``.
        """
        t2 = tuple(
            np.asarray(block, dtype=np.float64) for block in problem.provenance["t2"]
        )
        raw_t1 = problem.provenance.get("t1")
        t1 = (
            None
            if raw_t1 is None
            else tuple(np.asarray(block, dtype=np.float64) for block in raw_t1)
        )
        ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=self.n_reps,
            interaction_pairs=self._unbalanced_interaction_pairs(),
        )
        circuit = QuantumCircuit(2 * problem.norb)
        circuit.append(
            ffsim.qiskit.PrepareHartreeFockJW(
                problem.norb, (problem.n_alpha, problem.n_beta)
            ),
            range(2 * problem.norb),
        )
        circuit.append(
            ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), range(2 * problem.norb)
        )
        self._norb = problem.norb
        return StatePrepResult(circuit, norb=problem.norb, metadata={"method": "lucj"})

    def _unbalanced_interaction_pairs(
        self,
    ) -> tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]:
        """Widen the balanced 2-list form onto ffsim's 3-list unbalanced form.

        Balanced pairs are ``(same-spin, opposite-spin)``; the unbalanced form
        is ``(aa, ab, bb)``, so the same-spin list covers both the alpha-alpha
        and beta-beta blocks.
        """
        if len(self.interaction_pairs) == 3:
            return self.interaction_pairs
        same_spin, opposite_spin = self.interaction_pairs
        return (same_spin, opposite_spin, same_spin)

    def pass_manager(self, profile: BackendProfile) -> PassManager | None:
        """Build ffsim's LUCJ-aware pass manager after circuit construction."""
        if self._norb is None:
            raise RuntimeError("build() must be called before pass_manager()")
        if profile.mode in {"ideal", "synthetic_noise"}:
            return None
        backend = profile.build_simulator()
        if profile.mode == "topology":
            backend = GenericBackendV2(
                max(max(edge) for edge in profile.coupling_map) + 1,
                profile.basis_gates,
                coupling_map=profile.coupling_map,
                seed=profile.seed,
                noise_info=False,
            )
        pass_manager, _ = ffsim.qiskit.generate_lucj_pass_manager(
            backend=backend,
            norb=self._norb,
            connectivity="square",
            interaction_pairs=self._resolved_interaction_pairs(),
            optimization_level=1,
        )
        return pass_manager

    def _resolved_interaction_pairs(
        self,
    ) -> (
        tuple[list[tuple[int, int]], list[tuple[int, int]] | None]
        | tuple[
            list[tuple[int, int]],
            list[tuple[int, int]] | None,
            list[tuple[int, int]],
        ]
    ):
        """Materialize interaction pairs into the form the pass manager accepts.

        A ``None`` same-spin list becomes the full upper-triangular pair list,
        and the spin-unbalanced build widens the balanced 2-list form to the
        3-list ``(aa, ab, bb)`` form the unbalanced ansatz consumes.
        """
        same_spin_pairs = self.interaction_pairs[0]
        if same_spin_pairs is None:
            same_spin_pairs = [
                (first, second)
                for first in range(self._norb)
                for second in range(first + 1, self._norb)
            ]
        if not self._unbalanced:
            return (same_spin_pairs, self.interaction_pairs[1])
        pairs_aa, pairs_ab, pairs_bb = self._unbalanced_interaction_pairs()
        if pairs_aa is None:
            pairs_aa = same_spin_pairs
        if pairs_bb is None:
            pairs_bb = same_spin_pairs
        return (pairs_aa, pairs_ab, pairs_bb)
