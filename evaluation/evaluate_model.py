
import os
import importlib.util
from pathlib import Path
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from evaluation.leaderboard_writer import update_leaderboard
from src.data_utils import LabelledTraitData


VARS = [
    'N.percent',
    'P.percent',
    'K.percent',
    'Ca.percent',
    'Mg.percent',
    'C.percent',
    'Amax',
    'Asat',
    'Area.cm2',
    'Dry.mass.g',
    'Fresh.mass.g',
    'Thickness.mm',
    'SLA.g.m2'
]


def load_model(model_path: Path):
    """Dynamically load a model from a Python file."""
    module_name = model_path.stem
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Models should implement a create_model() function
    return module.create_model()

def evaluate_model(model, dpath: str, var):
    """Evaluate a model on a specific dataset."""
    dataset = LabelledTraitData(dpath, var)

    X_train, y_train = dataset.train_data, dataset.train_labels
    X_test, y_test = dataset.test_data, dataset.test_labels

    X_train, y_train = model.configure_data(X_train, y_train)
    X_test, y_test = model.configure_data(X_test, y_test)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    metrics = {
        # Calculate metrics.
        "R_squared": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

    return metrics

def run_evaluation(model_path: str):
    """Run evaluation for a model on all datasets."""
    dpath = Path("data")
    print(dpath.parent)
    print(dpath.resolve())
    print(list(dpath.iterdir()))
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    model = load_model(model_path)
    results = {}

    for var in VARS:
        print(f"Evaluating {model.name} on {var}...")
        metrics = evaluate_model(model, dpath, var)
        results[var] = metrics

    results['mean_r2_score'] = sum([res['R_squared'] for res in results.values()]) / len(results)

    # Update leaderboard with results
    update_leaderboard(model.name, model_path, results)
    print(f"Leaderboard updated with results for '{model.name}'")

if __name__ == "__main__":
    # Run manually or via GitHub Actions
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model and update the leaderboard")
    parser.add_argument("--model_path", required=True, help="Path to the model implementation file")

    args = parser.parse_args()
    run_evaluation(args.model_path)
