# sqdkit API 参考

## 核心流程

一个典型的 SQD 运行需要五个要素：问题（problem）、态制备方法（state-preparation method）、后端配置文件（backend profile）、运行配置（run config）以及运行器本身（runner）。

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

主入口函数。将态制备 → 面向目标的编译 → 采样 → SQD 对角化 → 结果收集串联起来。返回一个包含能量、收敛诊断和完整来源信息的 `SQDResult`。

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `problem` | `ProblemData` | 包含 h1、h2 积分的化学问题 |
| `stateprep` | `StatePrep` | 实现该协议的态制备对象 |
| `profile` | `BackendProfile` | 目标能力与噪声模型 |
| `config` | `RunConfig` | 采样次数、批次、迭代次数、随机种子 |
| `advanced_config` | `AdvancedConfig \| None` | 收敛容差、自旋对称化、最大子空间维度 |

### `sqdkit.pyscf_driver.build_problem(atom, basis, charge=0, spin=0, active_space=None, with_ccsd=True) → ProblemData`

从 PySCF RHF 计算构建一个 `ProblemData`。分子由 `atom`（PySCF 格式字符串或 `(symbol, (x,y,z))` 元组列表）和 `basis` 字符串（例如 `"sto-3g"`、`"6-31g"`）描述。

- 如果 `active_space` 是一个空间轨道索引列表（从 0 开始），则积分被限制在这些轨道上，冻结核能量被加到 `core_energy` 中。
- 如果 `with_ccsd=True`（默认），则计算 CCSD t1/t2 振幅并存储在 `provenance` 中，供需要它们的态制备方法（例如 LUCJ）使用。
- 当前仅支持 RHF。UHF/ROHF 支持不在 v0 范围内。

示例：
```python
# 全活性空间，带 CCSD 振幅
problem = build_problem("H 0 0 0; H 0 0 0.735", basis="sto-3g")

# 活性空间选择（N2/6-31G 的轨道 2–9）
problem = build_problem(
    [("N", (0, 0, 0)), ("N", (1.0, 0, 0))],
    basis="6-31g",
    active_space=list(range(2, 10)),
)
```

### `sqdkit.stateprep_lucj.LucjStatePrep(n_reps=2, interaction_pairs=(None, None))`

LUCJ（局域酉簇 Jastrow）态制备。唯一导入 ffsim 的模块。需要在问题的 `provenance` 中包含 CCSD t1/t2 振幅。

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `n_reps` | `2` | LUCJ 重复次数 |
| `interaction_pairs` | `(None, None)` | `(pairs_aa, pairs_ab)` 或 None 表示全连接 |

实现 `StatePrep` 协议：
- `build(problem, target=None)` → `StatePrepResult` — 构建线路
- `pass_manager(profile)` → `PassManager | None` — 返回 ffsim 中支持 LUCJ 的 pass manager，用于拓扑/校准模式

```python
lucj = LucjStatePrep(n_reps=1)
prep_result = lucj.build(problem)
# prep_result.circuit      — 作用于 2*norb 量子比特的 QuantumCircuit
# prep_result.norb         — 空间轨道数
# prep_result.metadata     — {"method": "lucj"}
```

### `sqdkit.stateprep.ReferenceProductStatePrep(norb, n_alpha, n_beta)`

使用 X 门的 Hartree-Fock 参考乘积态。一种不依赖 ffsim 的态制备方法，用于证明整个管线不依赖于 LUCJ。可作为基准线以及编写自定义态制备类的模板。

```python
hf_prep = ReferenceProductStatePrep(
    norb=problem.norb,
    n_alpha=problem.n_alpha,
    n_beta=problem.n_beta,
)
```

### `sqdkit.targets.BackendProfile`

可移植的后端能力契约。四种工厂方法：

| 方法 | 模式 | 噪声 | 依赖条件 |
|--------|------|-------|----------|
| `.ideal(basis_gates=None, seed=0)` | ideal | 无 | — |
| `.topology_only(coupling_map, basis_gates, seed=0)` | topology | 无 | 耦合映射 |
| `.synthetic_noise(coupling_map, basis_gates, gate_error, readout_error, seed=0)` | synthetic_noise | 去极化 + 读取 | 耦合映射 |
| `.calibration_like(fake_backend_name)` | calibration | 来自 IBM 虚假后端 | `qiskit-ibm-runtime` |

关键方法：
- `fingerprint()` → `str` — 配置文件的稳定 SHA-256 摘要
- `build_simulator()` → `AerSimulator` — 用于采样
- `to_target()` → `qiskit.transpiler.Target` — 用于编译
- `save_json(path)` / `load_json(path)` — 持久化

```python
# 理想情况（全连接，无噪声）
profile = BackendProfile.ideal()

# 带合成噪声的线性拓扑
profile = BackendProfile.synthetic_noise(
    coupling_map=[(0, 1), (1, 2), (2, 3)],
    basis_gates=["cx", "rz", "sx", "x", "id"],
    gate_error=0.001,
    readout_error=0.01,
    seed=42,
)

# IBM 校准快照（需要 qiskit-ibm-runtime）
profile = BackendProfile.calibration_like("FakeSherbrooke")
```

### `sqdkit.config.RunConfig(...)`

可序列化的运行配置。所有字段都有合理的默认值。

| 字段 | 默认值 | 描述 |
|-------|---------|-------------|
| `input_path` | `"problem.npz"` | NPZ 问题文件路径 |
| `stateprep_method` | `"lucj"` | 态制备方法名称 |
| `stateprep_options` | `{}` | 方法特定选项 |
| `backend_profile` | `"ideal"` | 配置文件模式字符串 |
| `shots` | `10000` | 采样次数 |
| `samples_per_batch` | `100` | SQD 批次大小 |
| `num_batches` | `10` | SQD 批次数 |
| `max_iterations` | `10` | 最大 SQD 迭代次数 |
| `seed` | `0` | 主随机种子（派生编译/采样/SQD 种子） |

方法：`to_dict()` / `from_dict(d)` 用于序列化。构造时验证所有字段。

### `sqdkit.config.AdvancedConfig(...)`

较不常用的 SQD 对角化控制参数，直接传递给 `diagonalize_fermionic_hamiltonian`。

| 字段 | 默认值 | 描述 |
|-------|---------|-------------|
| `energy_tol` | `1e-8` | 能量收敛容差 |
| `occupancies_tol` | `1e-5` | 轨道占据数收敛容差 |
| `carryover_threshold` | `1e-4` | 构型继承阈值 |
| `symmetrize_spin` | `False` | 强制自旋对称性 |
| `max_dim` | `None` | 最大 SCI 子空间维度 |
| `spin_sq` | `None` | 目标 S² 值 |

---

## 内部管线（用于自定义）

以下函数由 `sqd_runner.run()` 内部调用。仅在编写自定义工作流、自定义态制备方法或进行调试时才需要它们。

### `sqdkit.stateprep.StatePrep`（协议）

态制备契约。任何实现该协议的类均可传递给 `sqd_runner.run()`。

```python
class MyPrep:
    def build(self, problem: ProblemData, target=None) -> StatePrepResult:
        """在 2*norb 量子比特（alpha[0:norb), beta[norb:2norb)）上构建线路。"""
        ...

    def pass_manager(self, profile: BackendProfile) -> PassManager | None:
        """可选的编译 passes。返回 None 则使用默认编译。"""
        ...
```

### `sqdkit.stateprep.StatePrepResult(circuit, norb, metadata)`

由 `StatePrep.build()` 返回。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `circuit` | `QuantumCircuit` | 制备好的线路（不含测量） |
| `norb` | `int` | 空间轨道数 |
| `metadata` | `dict` | 至少包含 `{"method": "..."}` |

### `sqdkit.stateprep.to_sqd_bit_array(bit_array, norb) → np.ndarray`

将 Qiskit 的自旋阻塞比特串（alpha[0:norb), beta[norb:2norb)）转换为 SQD 期望的 `[beta..., alpha...]` 顺序。这对正确性至关重要——错误的约定会产生物理上不合理但看似可信的能量值。

### `sqdkit.synthesis.transpile_to_target(circuit, profile, pass_manager=None, seed=None) → (QuantumCircuit, dict)`

为目标配置文件编译线路。如果提供了 `pass_manager`（来自态制备方法），则使用该管理器；否则以优化等级 1 调用 `qiskit.transpile` 并使用配置文件的 target。

### `sqdkit.sampling.sample(compiled_circuit, profile, shots, seed) → (BitArray, dict)`

使用带种子的 `SamplerV2` 对编译后的线路进行采样。返回 `BitArray` 以及诊断信息：`total_shots`、`unique_bitstrings`、`valid_hamming`。

---

## 持久化

### `sqdkit.problem.ProblemData`

化学问题：空间轨道积分及元数据。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `h1` | `np.ndarray (norb, norb)` | 单粒子 MO 积分（化学家约定） |
| `h2` | `np.ndarray (norb, norb, norb, norb)` | 双粒子 MO 积分 `(ij\|kl)` |
| `core_energy` | `float` | 核排斥能 + 冻结核能量 |
| `norb` | `int` | 空间轨道数 |
| `n_alpha` | `int` | Alpha 电子数 |
| `n_beta` | `int` | Beta 电子数 |
| `integral_convention` | `str` | 始终为 `"chemist"` |
| `orbital_ordering` | `str` | 约定字符串 |
| `schema_version` | `int` | 模式版本（当前为 1） |
| `provenance` | `dict` | 来源元数据（基组、几何构型、t1/t2 等） |

方法：
- `save_npz(path)` — 保存为 NPZ 格式（v1 模式，可用 `allow_pickle=False` 读取）
- `ProblemData.load_npz(path)` — 加载并验证 NPZ 工件

```python
problem.save_npz("h2.npz")
reloaded = ProblemData.load_npz("h2.npz")
```

### `sqdkit.results.SQDResult`

一次 SQD 运行的完整结果。关键字段：`total_energy`、`electronic_energy`、`core_energy`、`converged`、`iterations`、`subspace_summary`、`shots_total`、`stateprep_summary`、`compiled_summary`、`backend_profile_fingerprint`。

方法：
- `save(path)` → NPZ（能量）+ JSON（元数据侧车文件）
- `SQDResult.load(path)` → 用 `allow_pickle=False` 重新加载

### `sqdkit.results.RunMetadata`

每次结果附带记录来源信息：`schema_version`、`seeds`（编译、采样、SQD）、`package_versions`、`backend_profile_mode`、`backend_profile_fingerprint`、`created_at`（ISO8601）。缺少必要键将引发 `ValueError`。

---

## 验证

### `sqdkit.validation.fci_reference(problem, max_norb=8) → float`

通过 PySCF 计算小问题的精确 FCI 总能。返回总能（电子能量 + core_energy）。若 `problem.norb > max_norb` 则引发 `ValueError`。

```python
from sqdkit.validation import fci_reference
fci_total = fci_reference(problem)
error_mHa = abs(result.total_energy - fci_total) * 1000
```
