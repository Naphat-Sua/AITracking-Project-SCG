"""Shared helpers for the test suite.

The lesson folders use hyphenated names (e.g. ``Linear-Regression``) which are
not importable as packages, so ``load_module`` loads a module directly from
its file path.  The module's folder is temporarily added to ``sys.path`` so
that sibling imports (e.g. ``import Visualize``) keep working.
"""

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open GUI windows in tests

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_ROOT = REPO_ROOT / "lessons"

_loaded = {}


def load_module(relative_path, module_name=None):
    """Load a module from a repo-relative file path, e.g.
    ``load_module("Activation/Activation.py")`` (relative to ``lessons/``)."""
    path = LESSONS_ROOT / relative_path
    if module_name is None:
        module_name = "lesson_" + path.stem.replace("-", "_").lower()

    cache_key = str(path)
    if cache_key in _loaded:
        return _loaded[cache_key]

    folder = str(path.parent)
    sys.path.insert(0, folder)
    old_cwd = os.getcwd()
    try:
        os.chdir(folder)  # some lessons read data files relative to their folder
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path.remove(folder)

    _loaded[cache_key] = module
    return module
