from __future__ import annotations

import ast
from pathlib import Path


def _imports_ffsim(path: Path) -> bool:
    """Return whether a module directly imports ffsim."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name == "ffsim" or alias.name.startswith("ffsim.")
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "ffsim" or node.module.startswith("ffsim."))
        )
        for node in ast.walk(tree)
    )


def test_only_lucj_state_prep_imports_ffsim() -> None:
    # Given: all sqdkit modules.
    package_dir = Path(__file__).parents[1] / "src" / "sqdkit"
    lucj_module = package_dir / "stateprep_lucj.py"
    other_modules = sorted(
        path for path in package_dir.glob("*.py") if path != lucj_module
    )

    # When: direct ffsim imports are inspected syntactically.
    offenders = [path.name for path in other_modules if _imports_ffsim(path)]

    # Then: LUCJ exclusively owns the dependency.
    assert _imports_ffsim(lucj_module)
    assert not offenders, f"ffsim imports outside stateprep_lucj.py: {offenders}"
