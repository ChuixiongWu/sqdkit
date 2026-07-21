# sqdkit

面向研究的基于采样的量子对角化（Sample-based Quantum Diagonalization, SQD）框架。

[English](README.md)

有关实现细节和项目架构，请参阅
[`docs/implementation-plan.md`](docs/implementation-plan.md)。
API 参考：[`docs/api.zh.md`](docs/api.zh.md)。
LUCJ 设计笔记：[`docs/lucj-design-notes.zh.md`](docs/lucj-design-notes.zh.md)。

## 环境配置（来自项目元数据）

```bash
uv sync
uv run python -c "import sqdkit; print(sqdkit.__version__)"
```

## 快速开始

```bash
uv run python examples/h2_lucj_smoke.py
```

运行 H₂ LUCJ 态制备（采用理想 + 带噪声 Aer 采样），执行 SQD
对角化，并保存可重现的结果工件。

## 如何添加自定义态制备方法

该框架的态制备接口**不依赖 ffsim**。编写一个实现 `StatePrep` 协议的类：

```python
from sqdkit.stateprep import StatePrep, StatePrepResult
from qiskit import QuantumCircuit


class MyAdiabaticPrep:
    def __init__(self, evolution_time: float = 1.0):
        self.evolution_time = evolution_time

    def build(self, problem, target=None) -> StatePrepResult:
        circuit = QuantumCircuit(2 * problem.norb)
        # 在此构建您的自定义态制备线路。
        # 量子比特约定：alpha 轨道 = qubits [0, norb)，beta = [norb, 2*norb)。
        # SQD 接收的测量比特串顺序为 [beta..., alpha...]。
        ...
        return StatePrepResult(
            circuit=circuit,
            norb=problem.norb,
            metadata={"method": "my_method"},
        )

    def pass_manager(self, profile):
        return None  # 或返回自定义的 Qiskit PassManager。
```

将您的态制备对象传递给 `sqd_runner.run()` — 无需对配置、后端或 SQD 做任何修改即可运行。

## 测试

```bash
uv run pytest -q -m "not slow"  # 快速测试套件（单元测试 + H2 端到端）
uv run pytest -q -m slow        # 集成测试（N2 活性空间）
```

## 包

```bash
sqdkit  # 导入名称
```

## 依赖项

Qiskit 2.x、ffsim、PySCF、qiskit-aer、qiskit-addon-sqd、numpy。
