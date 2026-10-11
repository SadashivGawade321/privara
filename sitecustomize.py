"""sitecustomize – automatically imported by Python before any script.
Ensures the project's src/ directory is first on sys.path so imports like
`from data.dataset_loader import ...` resolve to src/data rather than the raw
root data folder, and binds src/compression to handle Python 3.14+ stdlib collisions.
"""
import os
import sys
import importlib.util

_root = os.path.abspath(os.path.dirname(__file__))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Ensure local 'compression' package is resolved even if Python 3.14+ stdlib compression exists
_comp_dir = os.path.join(_src, "compression")
_comp_init = os.path.join(_comp_dir, "__init__.py")
if os.path.exists(_comp_init) and ("compression" not in sys.modules or not getattr(sys.modules["compression"], "__file__", "").startswith(_src)):
    spec = importlib.util.spec_from_file_location(
        "compression",
        _comp_init,
        submodule_search_locations=[_comp_dir]
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["compression"] = mod
        spec.loader.exec_module(mod)
