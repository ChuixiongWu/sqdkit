# LUCJ 设计笔记：它到底是什么

## 形式

局域酉簇 Jastrow（LUCJ）拟设具有如下形式 [^1]

\\[
\\prod_{k=1}^{L} \\mathcal{U}_k \\, e^{i\\mathcal{J}_k} \\, \\mathcal{U}_k^\\dagger
\\]

其中每个 \\(\\mathcal{U}_k\\) 是一个轨道旋转算子，
\\(\\mathcal{J}_k = \\frac{1}{2}\\sum_{ij,\\sigma\\tau} J^{(\\sigma\\tau)}_{ij} n_{i\\sigma} n_{j\\tau}\\)
是一个对角化的数-数（库仑）算子。
该线路实现将 \\(\\mathcal{U}_k\\) 分解为 Givens 旋转网络（\\(\\text{XX+YY}\\) 门），
并将 \\(e^{i\\mathcal{J}_k}\\) 分解为 Jordan-Wigner 编码下的受控相位（\\(\\text{CP}\\)）门。

该形式类似于酉耦合簇 \\(e^{T - T^\\dagger}\\) 的单次 Trotter 步，
但这种类比仅仅是结构上的。

## 参数来源

参数 \\((U_k, J_k)\\) **并非**通过对 \\(e^{T-T^\\dagger}\\) 进行 Trotter 化得到，
也**不**经过变分优化。
相反，`UCJOpSpinBalanced.from_t_amplitudes(t2, t1)` 对 CCSD \\(t_2\\) 振幅张量执行**双重
分解** [^2]：

\\[
t_{ijab} = i \\sum_{k} \\sum_{pq}
    Z^{(k)}_{pq} \\,
    U^{(k)}_{ap} U^{(k)*}_{ip} \\,
    U^{(k)}_{bq} U^{(k)*}_{jq},
\\]

这是一个嵌套的特征值分解，然后截断至 \\(L\\) 项。
\\(Z^{(k)}\\) 提供对角化的库仑强度 \\(J_k\\)，而 \\(U^{(k)}\\) 提供
轨道旋转矩阵。
\\(t_1\\) 振幅（若存在）仅贡献于一个可选的最终轨道旋转，
对于处于平衡态附近的闭壳层分子而言通常可忽略不计
（Brillouin 定理）。

## 含义

**LUCJ 是将 CCSD 重新封装为适合硬件的酉电路。**

- 拟设形式借鉴自 UCCSD，但参数是通过矩阵分解从 CCSD *推导*而来——完全绕过了变分优化。
- 分解引入了冗余自由度（表征 \\(t_2\\) 所需的 \\(U_k, J_k\\) 对数量超过了严格必要的数目）。正是这种冗余使得硬件适配成为可能：可以在不损失核心关联物理的前提下，根据连通性图裁剪门——剩余的参数会吸收这部分余量。
- 近似存在于*拟设的灵活性*中，而非哈密顿量中：SQD 对角化使用完整的 \\(h_1, h_2\\) 积分；LUCJ 仅限制所采样的子空间。

## 对未来工作的启示

对态制备阶段的任何改进，都应遵循相同的原则——一个紧凑的经典关联模型（CC、CI、MP2、DMET 嵌入……），其参数通过结构化分解映射到酉电路，而非通过变分优化或特设的 Trotter 化。关键的设计决策是 (a) 经典模型的选择，(b) 能够吸收其参数的拟设形式，以及 (c) 分解策略。

---

[^1]: O'Brien, Anselmetti, et al. “Bridging physical intuition and hardware
      efficiency for correlated electronic states: the local unitary cluster
      Jastrow ansatz for electronic structure.” *Chem. Sci.*, 2023.
      DOI: `10.1039/d3sc02516k`.

[^2]: `ffsim.linalg.double_factorized_t2` —
      `ffsim/variational/ucj_spin_balanced.py`,
      `ffsim/linalg/double_factorized_decomposition.py`.
      (C) Copyright IBM 2024, Apache-2.0.
