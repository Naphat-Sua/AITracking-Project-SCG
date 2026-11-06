"""Tests for behaviour that the refactoring round changed or fixed."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_module


def test_activation_plotting_is_opt_in():
    """plot_activations() builds a figure only when explicitly called."""
    import matplotlib.pyplot as plt

    act = load_module("Activation/Activation.py")
    plt.close("all")
    assert plt.get_fignums() == []

    fig = act.plot_activations(x=np.linspace(-5, 5, 50))
    assert fig is not None
    assert len(fig.axes) == 5  # input + 4 activation functions
    plt.close("all")


def test_genetic_algorithm_seed_makes_runs_reproducible(tmp_path):
    rw = load_module("Generative-Algorithm/route_way.py")

    cities = ["A", "B", "C", "D", "E", "F"]
    rows, value = [], 100
    for i, c1 in enumerate(cities):
        for c2 in cities[i + 1:]:
            rows.append({"waypoint1": c1, "waypoint2": c2,
                         "distance_m": value, "duration_s": value * 2})
            value += 7
    csv_file = tmp_path / "waypoints.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    first = rw.run_genetic_algorithm(generations=30, population_size=10,
                                     csv_file=str(csv_file), seed=1234)
    second = rw.run_genetic_algorithm(generations=30, population_size=10,
                                      csv_file=str(csv_file), seed=1234)
    assert first == second, "same seed must produce the same routes"


def test_logistic_cost_is_finite_at_saturated_predictions():
    """mean_cost clips predictions, so log10(0) never produces -inf/NaN."""
    logreg = load_module("Log-Regression/Function.py")

    # sigmoid saturates to exactly 0.0 / 1.0 for large margins
    saturated = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([1, 0, 0, 1])
    cost = logreg.mean_cost(saturated, labels)
    assert np.isfinite(cost), "cost must stay finite at saturated predictions"
    assert cost > 0


def test_logistic_training_respects_max_steps():
    """A learning rate that never converges must still terminate."""
    logreg = load_module("Log-Regression/Function.py")

    rng = np.random.RandomState(0)
    data_X = pd.DataFrame({'X1': rng.uniform(0, 1, 20), 'X2': rng.uniform(0, 1, 20)})
    Y = (data_X['X1'].values > 0.5).astype(int).reshape(-1, 1)

    # a huge learning rate oscillates instead of converging
    C_List = logreg.train_method(data_X, Y, learningRate=50.0, max_steps=200)
    assert C_List.shape[1] == 3  # bias + 2 features, and it returned at all


def test_thai_frequency_csv_resolves_from_any_working_directory(tmp_path, monkeypatch):
    """The word list is found relative to the module, not the cwd."""
    thai = load_module("Text-Classification/thai_dataset.py")

    monkeypatch.chdir(tmp_path)  # somewhere with no data files
    thai._dict_word = None       # force a reload
    assert thai.get_index(['ที่']) == [1]


@pytest.mark.parametrize("relative_path,symbol", [
    ("Neural-Network/Tensorflow.py", "load_dataset"),
    ("Neural-Network/Tensorflow.py", "train"),
    ("Neural-Network/Class.py", "load_dataset"),
    ("Neural-Network/Class.py", "make_trainer"),
    ("Dataset-Prepare/Enumerate.py", "load_vgg"),
])
def test_refactored_entry_points_exist(relative_path, symbol):
    """The extracted helpers are importable without TensorFlow side effects.

    Parsed statically: these modules import tensorflow, which is optional here.
    """
    import ast
    from conftest import REPO_ROOT

    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert symbol in names, f"{relative_path} should define {symbol}()"


def test_neural_network_visualization_uses_its_own_model():
    """Regression test: update() used to read a module-level `model` global."""
    from conftest import REPO_ROOT

    source = (REPO_ROOT / "Neural-Network/Class.py").read_text(encoding="utf-8")
    assert "self.model.predict(" in source
    assert "\t\tZ = model.predict(" not in source, (
        "Visualization.update() must predict with self.model, not a global"
    )
