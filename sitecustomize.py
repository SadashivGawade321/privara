"""sitecustomize – automatically imported by Python before any script.
Ensures the project's src/ directory is first on sys.path so imports like
`from data.dataset_loader import ...` resolve to src/data rather than the raw
root data folder.
"""
import os, sys
_root = os.path.abspath(os.path.dirname(__file__))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
