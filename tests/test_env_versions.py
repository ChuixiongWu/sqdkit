"""Lock the environment's scientific-package compatibility ranges."""

from importlib.metadata import PackageNotFoundError, version

import pytest
from packaging.version import Version


def installed_version(package: str) -> str:
    """Return installed distribution metadata with a named pytest failure."""
    try:
        return version(package)
    except PackageNotFoundError:
        pytest.fail(f"{package} version not installed (expected a pinned compatible range)")


def test_resolved_package_versions_are_compatible() -> None:
    """Fail fast when project metadata resolves an incompatible package version."""
    qiskit = installed_version("qiskit")
    qiskit_aer = installed_version("qiskit_aer")
    qiskit_addon_sqd = installed_version("qiskit_addon_sqd")
    ffsim = installed_version("ffsim")
    pyscf = installed_version("pyscf")

    assert qiskit.startswith("2."), f"qiskit version {qiskit} not in range (expected 2.x)"
    assert qiskit_aer.startswith("0.17."), (
        f"qiskit_aer version {qiskit_aer} not in range (expected 0.17.x)"
    )
    assert qiskit_addon_sqd.startswith("0.12."), (
        "qiskit_addon_sqd version "
        f"{qiskit_addon_sqd} not in range (expected 0.12.x)"
    )
    assert Version(ffsim) >= Version("0.0.83"), (
        f"ffsim version {ffsim} not in range (expected >=0.0.83)"
    )
    assert Version(pyscf) >= Version("2.12"), (
        f"pyscf version {pyscf} not in range (expected >=2.12)"
    )


def test_environment_is_created_from_project_metadata() -> None:
    """Create this environment from pyproject.toml and uv.lock with ``uv sync``."""
