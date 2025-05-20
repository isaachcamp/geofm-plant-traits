
# Seed not added to argument parser yet.

from warnings import filterwarnings
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from evaluation.leaderboard_writer import update_leaderboard
from src.data_utils import LabelledTraitData, train_val_test_split

filterwarnings("ignore", module="sklearn")

N_REPEATS = 1
VARS = [
    'N.Percent',
    'P.Percent',
    'K.Percent',
    'Ca.Percent',
    'Mg.Percent',
    'C.Percent',
    'Amax',
    'Asat',
    'Area.cm2',
    'Dry.mass.g',
    'Fresh.mass.g',
    'Thickness.mm',
    'SLA.g.m2'
]


def load_model(model_path: Path, seed: int = None, var: str = None):
    """Dynamically load a model from a Python file."""
    module_name = model_path.stem
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Models should implement a create_model() function
    return module.create_model(seed, var)

def evaluate_model(model, dataset: LabelledTraitData):
    """Evaluate a model on a specific dataset."""
    X = pd.concat([dataset.train_data, dataset.val_data, dataset.test_data])
    y = pd.concat([dataset.train_labels, dataset.val_labels, dataset.test_labels])

    # X_train, X_val, X_test = train_val_test_split(X, 0.7, 0.1)
    X_train, X_val, X_test = dataset.train_data, dataset.val_data, dataset.test_data
    y_train, y_val, y_test = y.loc[X_train.index], y.loc[X_val.index], y.loc[X_test.index]

    X_train, y_train = model.configure_data(X_train, y_train)
    X_val, y_val = model.configure_data(X_val, y_val)
    X_test, y_test = model.configure_data(X_test, y_test)

    # Train model.
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate model.
    y_pred = model.predict(X_test)

    y_pred, y_test = model.unstandardise(y_pred, y_test)

    # Calculate metrics.
    metrics = {
        "R_squared": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

    return metrics

def run_evaluation(model_path: str, dpath: str, seed: int = 42):
    """Run evaluation for a model on all datasets."""
    dpath = Path(dpath).resolve()
    model_path = Path(model_path)
    metadata_path = dpath / 'metadata'

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    results = {}
    with open(metadata_path / 'trait_stats.json', 'r') as f:
        trait_stats = json.load(f)

    for var in VARS:
        results[var] = {
            'R_squared': [],
            'RMSE': [],
            'MAE': [],
            'MAPE': []
        }
        dataset = LabelledTraitData(dpath, var)

        for i in range(N_REPEATS):
            model = load_model(model_path, seed, var)
            var_stats = trait_stats[var]
            model.set_stats(var_stats)

            if i == 0:
                print(f"Evaluating {model.name} on {var}...")

            metrics = evaluate_model(model, dataset)

            results[var]['R_squared'].append(metrics['R_squared'])
            results[var]['RMSE'].append(metrics['RMSE'])
            results[var]['MAE'].append(metrics['MAE'])
            results[var]['MAPE'].append(metrics['MAPE'])

        for metric in metrics:
            results[var][f'{metric}_mean'] = float(np.mean(results[var][metric]))
            results[var][f'{metric}_std'] = float(np.std(results[var][metric]))

    traits_mean = np.array([results[var]['R_squared'] for var in VARS]).mean(axis=0)

    results['mean_r2_score'] = float(traits_mean.mean())
    results['std_r2_score'] = float(traits_mean.std())

    # Update leaderboard with results
    update_leaderboard(model.name, model_path, results)
    print(f"Leaderboard updated with results for '{model.name}'")

if __name__ == "__main__":
    # Run manually or via GitHub Actions
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model and update the leaderboard")
    parser.add_argument("--model_path", required=True, help="Path to the model implementation file")
    parser.add_argument("--dpath", default="data", help="Path to the data directory")

    args = parser.parse_args()
    run_evaluation(args.model_path, args.dpath)
