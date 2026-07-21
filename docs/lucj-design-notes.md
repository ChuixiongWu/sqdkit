# LUCJ design notes: what it really is

## Form

The local unitary cluster Jastrow (LUCJ) ansatz has the form [^1]

\[
\prod_{k=1}^{L} \mathcal{U}_k \, e^{i\mathcal{J}_k} \, \mathcal{U}_k^\dagger
\]

where each \(\mathcal{U}_k\) is an orbital rotation and
\(\mathcal{J}_k = \frac{1}{2}\sum_{ij,\sigma\tau} J^{(\sigma\tau)}_{ij} n_{i\sigma} n_{j\tau}\)
is a diagonal number-number (Coulomb) operator.
The circuit implementation decomposes \(\mathcal{U}_k\) into Givens rotation
networks (\(\text{XX+YY}\) gates) and \(e^{i\mathcal{J}_k}\) into controlled-phase
(\(\text{CP}\)) gates in the Jordan-Wigner encoding.

The form resembles a single Trotter step of unitary coupled cluster
\(e^{T - T^\dagger}\), but the parallel is structural only.

## Parameter source

The parameters \((U_k, J_k)\) are **not** obtained by Trotterising \(e^{T-T^\dagger}\)
and are **not** variationally optimised.
Instead, `UCJOpSpinBalanced.from_t_amplitudes(t2, t1)` performs a **double
factorisation** of the CCSD \(t_2\) amplitudes tensor [^2]:

\[
t_{ijab} = i \sum_{k} \sum_{pq}
    Z^{(k)}_{pq} \,
    U^{(k)}_{ap} U^{(k)*}_{ip} \,
    U^{(k)}_{bq} U^{(k)*}_{jq},
\]

a nested eigenvalue decomposition followed by truncation to \(L\) terms.
\(Z^{(k)}\) supplies the diagonal Coulomb strengths \(J_k\) and \(U^{(k)}\) supplies
the orbital rotation matrices.
\(t_1\) amplitudes, when available, contribute only to an optional final orbital
rotation and are typically negligible for closed-shell molecules near equilibrium
(Brillouin's theorem).

## What it means

**LUCJ is CCSD repackaged as a hardware-friendly unitary circuit.**

- The ansatz form is borrowed from UCCSD, but the parameters are *derived*
  from CCSD via matrix factorisation — bypassing variational optimisation
  entirely.
- The factorisation introduces redundant degrees of freedom (more
  \(U_k, J_k\) pairs than strictly necessary to represent \(t_2\)).
  This redundancy is what makes hardware adaptation possible: gates can
  be dropped to match a connectivity graph without losing the essential
  correlation physics — the remaining parameters absorb the slack.
- The approximation is in the *ansatz flexibility*, not in the Hamiltonian:
  SQD diagonalisation uses the full \(h_1, h_2\) integrals; LUCJ only
  restricts which subspace is sampled.

## Implications for future work

Any improvement to the state-preparation stage should likely start from the
same principle — a compact classical correlation model (CC, CI, MP2, DMET
embedding, …) whose parameters are mapped to a unitary circuit via a
structured decomposition, rather than via variational optimisation or
ad-hoc Trotterisation.  The key design decisions are (a) the choice of
classical model, (b) the ansatz form that can absorb its parameters, and
(c) the decomposition strategy.

---

[^1]: O'Brien, Anselmetti, et al. “Bridging physical intuition and hardware
      efficiency for correlated electronic states: the local unitary cluster
      Jastrow ansatz for electronic structure.” *Chem. Sci.*, 2023.
      DOI: `10.1039/d3sc02516k`.

[^2]: `ffsim.linalg.double_factorized_t2` —
      `ffsim/variational/ucj_spin_balanced.py`,
      `ffsim/linalg/double_factorized_decomposition.py`.
      (C) Copyright IBM 2024, Apache-2.0.
