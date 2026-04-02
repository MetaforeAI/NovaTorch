"""Pytest configuration for NovaTorch tests."""
import sys
from pathlib import Path

# Add the Python package source and the build output to sys.path
# so that `import novatorch` resolves to our local package and finds _C.so.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "novatorch" / "python"))
sys.path.insert(0, str(_repo / "novatorch" / "build"))

import novatorch  # noqa: E402,F401  — registers the backend
