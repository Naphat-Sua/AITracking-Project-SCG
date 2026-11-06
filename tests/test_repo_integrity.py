"""Guards against the two whole-repo problems this project has already hit once.

1. Binary corruption. Every tracked binary in this repository was once mangled
   by a UTF-8 text round trip (non-decodable bytes replaced with U+FFFD),
   destroying 57-70% of each file. Text survives such a round trip; binaries do
   not. These tests fail loudly if a binary with a known magic number is ever
   committed in a corrupted state again.

2. Import-time side effects. Several lessons used to train models, download
   datasets or open plot windows at import time, which makes them impossible to
   import, test or reuse. `test_modules_import_without_side_effects` keeps every
   module importable and quiet.
"""

import subprocess

import pytest

from conftest import REPO_ROOT, load_module

# Magic numbers (file signatures) for the binary formats this repo stores.
MAGIC_NUMBERS = {
    ".h5": (b"\x89HDF\r\n\x1a\n", "HDF5"),
    ".hdf5": (b"\x89HDF\r\n\x1a\n", "HDF5"),
    ".whl": (b"PK\x03\x04", "ZIP (wheel)"),
    ".zip": (b"PK\x03\x04", "ZIP"),
    ".xlsx": (b"PK\x03\x04", "XLSX (zip)"),
    ".png": (b"\x89PNG\r\n\x1a\n", "PNG"),
    ".jpg": (b"\xff\xd8\xff", "JPEG"),
    ".jpeg": (b"\xff\xd8\xff", "JPEG"),
    ".gif": (b"GIF8", "GIF"),
    ".pdf": (b"%PDF", "PDF"),
    ".mat": (b"MATLAB", "MATLAB .mat"),
}

# The UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER. Its presence in a binary
# file is the fingerprint of a lossy text round trip.
REPLACEMENT_CHAR = b"\xef\xbf\xbd"


def tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return [line for line in out.splitlines() if line]


def binary_files():
    return [f for f in tracked_files()
            if any(f.lower().endswith(ext) for ext in MAGIC_NUMBERS)]


def test_tracked_binaries_have_valid_magic_numbers():
    """A committed binary must start with its format's signature."""
    broken = []
    for name in binary_files():
        ext = "." + name.rsplit(".", 1)[-1].lower()
        expected, label = MAGIC_NUMBERS[ext]
        data = (REPO_ROOT / name).read_bytes()
        if not data.startswith(expected):
            broken.append(
                f"{name}: expected {label} header {expected.hex(' ')}, "
                f"found {data[:len(expected)].hex(' ')}"
            )
    assert not broken, "Corrupted binary file(s):\n  " + "\n  ".join(broken)


def test_tracked_binaries_contain_no_replacement_characters():
    """U+FFFD inside a binary means it was destroyed by a text round trip."""
    broken = []
    for name in binary_files():
        data = (REPO_ROOT / name).read_bytes()
        count = data.count(REPLACEMENT_CHAR)
        if count:
            pct = 100 * count * len(REPLACEMENT_CHAR) / len(data)
            broken.append(f"{name}: {count:,} U+FFFD sequences ({pct:.1f}% of bytes)")
    assert not broken, (
        "Binary file(s) mangled by a UTF-8 round trip:\n  " + "\n  ".join(broken)
    )


def test_no_tracked_file_is_an_empty_placeholder():
    """Catch assets that were reduced to a single newline."""
    tiny = []
    for name in binary_files():
        size = (REPO_ROOT / name).stat().st_size
        if size <= 1:
            tiny.append(f"{name}: {size} byte(s)")
    assert not tiny, "Placeholder file(s) with no content:\n  " + "\n  ".join(tiny)


# Modules that must import cleanly: importing them may not train a model,
# download a dataset, open a window or exit the interpreter.
IMPORTABLE_MODULES = [
    "Activation/Activation.py",
    "Generative-Algorithm/route_way.py",
    "Hash/CrowndStrike.py",
    "Linear-Regression/Feature.py",
    "Linear-Regression/Regression.py",
    "Linear-Regression/Visualize.py",
    "Log-Regression/Function.py",
    "Log-Regression/Visualize.py",
    "MNIST/Dataset.py",
    "Principal-Component-Analysis/PCA.py",
    "Text-Classification/thai_dataset.py",
]


@pytest.mark.parametrize("relative_path", IMPORTABLE_MODULES)
def test_modules_import_without_side_effects(relative_path, capsys):
    """Importing a lesson module must be silent, fast and harmless."""
    import matplotlib.pyplot as plt

    plt.close("all")
    module = load_module(relative_path)
    assert module is not None

    # Importing must not print a training log or dataset dump.
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"{relative_path} printed at import time:\n{captured.out[:500]}"
    )
    # ...nor build figures (which means it was plotting at import).
    open_figures = plt.get_fignums()
    plt.close("all")
    assert not open_figures, (
        f"{relative_path} created {len(open_figures)} matplotlib figure(s) at "
        f"import time; move the plotting into a function guarded by __main__"
    )


# Scripts whose demo body must sit behind `if __name__ == "__main__":` rather
# than running when the file is imported. Parsed statically so the test does not
# need TensorFlow, a GPU, or the large model downloads these scripts want.
GUARDED_SCRIPTS = [
    "Activation/Activation.py",
    "Dataset-Prepare/1st-Prepare.py",
    "Dataset-Prepare/2nd-Prepare.py",
    "Dataset-Prepare/Enumerate.py",
    "Dataset-Prepare/From-MNIST.py",
    "Encoder-Pack/Image-Encoder.py",
    "Encoder-Pack/Text-Encoder.py",
    "Generative-Adversarial/Generate-Image.py",
    "Generative-Algorithm/Get-Data.py",
    "Generative-Algorithm/Routing.py",
    "IMGNET-Pre-Trained/Image-Classifier.py",
    "Linear-Regression/Feature.py",
    "Linear-Regression/Regression.py",
    "Log-Regression/Function.py",
    "MNIST/Test.py",
    "MNIST/Visual.py",
    "Neural-Network/Class.py",
    "Neural-Network/Tensorflow.py",
    "Object-Oriented/Detection.py",
    "Recurrent-Neural-Network/Train.py",
    "Text-Classification/Text-Classify.py",
    "Text-Classification/thai_dataset.py",
]


@pytest.mark.parametrize("relative_path", GUARDED_SCRIPTS)
def test_scripts_do_their_work_behind_a_main_guard(relative_path):
    """No top-level statement may train, download, plot or exit.

    Only definitions, imports, constants and the `__main__` guard are allowed at
    module level.
    """
    import ast

    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    allowed = (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
        ast.If, ast.Try,
    )
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring or a block comment string
        if isinstance(node, ast.Expr):
            offenders.append(f"line {node.lineno}: bare call at module level")
            continue
        if isinstance(node, (ast.For, ast.While)):
            offenders.append(f"line {node.lineno}: loop at module level")
            continue
        if not isinstance(node, allowed):
            offenders.append(f"line {node.lineno}: {type(node).__name__} at module level")

    assert not offenders, (
        f"{relative_path} does work at import time:\n  " + "\n  ".join(offenders)
    )


def test_markdown_relative_links_resolve():
    """Every relative link in the documentation must point at a real file."""
    import re

    broken = []
    for name in tracked_files():
        if not name.endswith(".md"):
            continue
        path = REPO_ROOT / name
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)#:]+?)\)", text):
            target = match.group(1).strip()
            if target.startswith(("http", "mailto", "#")):
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{name} -> {target}")

    assert not broken, "Broken relative link(s):\n  " + "\n  ".join(broken)


def test_every_python_file_compiles():
    """No syntax errors anywhere, including vendored third-party code."""
    py_files = [f for f in tracked_files() if f.endswith(".py")]
    assert len(py_files) > 20, "expected to find the lesson scripts"

    result = subprocess.run(
        ["python", "-m", "py_compile", *py_files],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Syntax errors:\n{result.stderr}"
