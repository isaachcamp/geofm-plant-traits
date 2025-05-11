
# Seed not added to argument parser yet.

from warnings import filterwarnings
import importlib.util
from typing import List
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from evaluation.leaderboard_writer import update_leaderboard
from src.data_utils import LabelledTraitData

filterwarnings("ignore", module="sklearn")


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

def evaluate_model(model, dpath: str, var, preds_df: pd.DataFrame):
    """Evaluate a model on a specific dataset."""
    dataset = LabelledTraitData(dpath, var)

    X_train, y_train = dataset.train_data, dataset.train_labels
    X_val, y_val = dataset.val_data, dataset.val_labels
    X_test, y_test = dataset.test_data, dataset.test_labels

    # Concatenate all data for predictions, save DataFrame indexing ability.
    X = pd.concat([X_train, X_val, X_test])

    if not preds_df.empty:
        X_train = X_train.join(preds_df[X_train.index])
        X_val = X_val.join(preds_df[X_val.index])
        X_test = X_test.join(preds_df[X_test.index])

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

    # Predict trait value for all samples.
    X_prepped = model.configure_data(X, None)
    preds = pd.Series(model.predict(X_prepped), index=X.index, name=var)

    return metrics, preds

def run_evaluation(model_path: str, dpath: str, seed: int = 42):
    """Run evaluation for a model on all datasets."""
    dpath = Path(dpath).resolve()
    model_path = Path(model_path)
    metadata_path = dpath / 'metadata'

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    results = {}
    preds_df = pd.DataFrame()

    with open(metadata_path / 'trait_stats.json', 'r') as f:
        trait_stats = json.load(f)

    for var in VARS:
        model = load_model(model_path, seed, var)
        var_stats = trait_stats[var]
        model.set_stats(var_stats)

        print(f"Evaluating {model.name} on {var}...")
        metrics, preds = evaluate_model(model, dpath, var, preds_df)
        results[var] = metrics

        preds_df.join(preds, how='outer')

    results['mean_r2_score'] = sum([res['R_squared'] for res in results.values()]) / len(results)

    # Update leaderboard with results
    # update_leaderboard(model.name, model_path, results)
    # print(f"Leaderboard updated with results for '{model.name}'")

if __name__ == "__main__":
    # Run manually or via GitHub Actions
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model with auto-regressive predictions and update the leaderboard")
    parser.add_argument("--model_path", required=True, help="Path to model implementation file")
    parser.add_argument("--dpath", default="data", help="Path to data directory")

    args = parser.parse_args()
    run_evaluation(args.model_path, args.dpath)
