"""
CarDekho Used Car Price Prediction
Feature Engineering & Model Training Module

Builds the sklearn Pipeline, trains multiple regressors, performs
hyperparameter tuning via RandomizedSearchCV, and serialises the
best model + metadata for use by the Streamlit app.
"""

import logging
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                    OneHotEncoder, StandardScaler)
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = Path("models")

# ─── Feature lists ────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "car_age", "kms_driven", "engine_cc", "max_power_bhp",
    "mileage", "seats", "owner_no", "torque_nm",
    "length_mm", "width_mm", "height_mm",
]

CATEGORICAL_FEATURES = [
    "fuel_type", "body_type", "transmission", "oem",
    "city", "insurance_validity", "drive_type",
]

TARGET = "price"

# ─── Feature engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing columns."""
    df = df.copy()

    # Power-to-weight proxy
    df["power_per_cc"] = df["max_power_bhp"] / (df["engine_cc"] + 1)

    # Kms per year
    df["kms_per_year"] = df["kms_driven"] / (df["car_age"].clip(lower=1))

    # Volume proxy
    df["volume_cm3"] = (
        df["length_mm"].fillna(0)
        * df["width_mm"].fillna(0)
        * df["height_mm"].fillna(0)
        / 1e6
    )

    # Log-transform skewed numerics
    df["log_kms_driven"] = np.log1p(df["kms_driven"])
    df["log_engine_cc"]  = np.log1p(df["engine_cc"])

    return df


def get_feature_lists():
    """Return extended numeric + categorical lists after feature engineering."""
    num = NUMERIC_FEATURES + [
        "power_per_cc", "kms_per_year", "volume_cm3",
        "log_kms_driven", "log_engine_cc",
    ]
    cat = CATEGORICAL_FEATURES
    return num, cat


# ─── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor(num_features, cat_features):
    """Build ColumnTransformer with scaling + encoding."""
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ─── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=100.0, max_iter=5000),
    "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=4,
        n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42),
}

PARAM_GRIDS = {
    "RandomForest": {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [10, 15, 20, None],
        "model__min_samples_leaf": [2, 4, 6],
        "model__max_features":     ["sqrt", "log2"],
    },
    "GradientBoosting": {
        "model__n_estimators":  [200, 300, 400],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth":     [3, 5, 7],
        "model__subsample":     [0.7, 0.8, 0.9],
    },
}

# ─── Training utilities ───────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "MAE_lakh": mae / 1e5}


def train_all_models(df: pd.DataFrame, test_size: float = 0.2):
    """
    Train every model, collect metrics, return best pipeline + results.
    """
    df = engineer_features(df)
    num_feats, cat_feats = get_feature_lists()

    # Keep only columns that exist
    all_feats = [f for f in num_feats + cat_feats if f in df.columns]
    num_feats  = [f for f in num_feats  if f in df.columns]
    cat_feats  = [f for f in cat_feats  if f in df.columns]

    X = df[all_feats].copy()
    y = df[TARGET].copy()

    # Fill any residual NaN
    for c in num_feats:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_feats:
        X[c] = X[c].fillna("Unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = build_preprocessor(num_feats, cat_feats)
    results = {}
    trained_pipelines = {}

    for name, estimator in MODELS.items():
        logger.info(f"Training {name} …")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        trained_pipelines[name] = pipeline
        logger.info(f"  {name}: R²={metrics['R2']:.4f}  MAE=₹{metrics['MAE_lakh']:.2f}L  RMSE={metrics['RMSE']/1e5:.2f}L")

    return trained_pipelines, results, X_train, X_test, y_train, y_test, num_feats, cat_feats


def tune_best_model(pipeline, X_train, y_train, model_name: str):
    """RandomizedSearchCV on a named model if a param grid exists."""
    if model_name not in PARAM_GRIDS:
        logger.info(f"No param grid for {model_name}; skipping tuning.")
        return pipeline

    logger.info(f"Tuning {model_name} with RandomizedSearchCV …")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=20,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV R²: {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Save / Load helpers ──────────────────────────────────────────────────────

def save_model(pipeline, metadata: dict, path: str = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = path or str(MODEL_DIR / "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "metadata": metadata}, f)
    logger.info(f"Model saved to {path}")
    # also save metadata as JSON for quick inspection
    meta_path = Path(path).with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")


def load_model(path: str = None):
    path = path or str(MODEL_DIR / "best_model.pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["metadata"]


# ─── Main training run ────────────────────────────────────────────────────────

def run_training(data_dir: str = "data", model_path: str = None):
    from data_preprocessing import preprocess_pipeline  # local import

    df = preprocess_pipeline(data_dir=data_dir)
    trained, results, X_train, X_test, y_train, y_test, num_feats, cat_feats = \
        train_all_models(df)

    # Pick model with highest R² on test set
    best_name = max(results, key=lambda k: results[k]["R2"])
    logger.info(f"\n🏆 Best model: {best_name}  R²={results[best_name]['R2']:.4f}")

    # Tune best if possible
    best_pipeline = tune_best_model(trained[best_name], X_train, y_train, best_name)
    final_metrics = evaluate_model(best_pipeline, X_test, y_test)

    metadata = {
        "best_model": best_name,
        "all_results": {k: {m: float(v) for m, v in met.items()}
                        for k, met in results.items()},
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "unique_values": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in cat_feats if col in df.columns
        },
        "price_min": float(df["price"].min()),
        "price_max": float(df["price"].max()),
        "oem_list":  sorted(df["oem"].dropna().unique().tolist()),
        "model_map": (
            df[["oem", "model"]]
            .dropna()
            .drop_duplicates()
            .groupby("oem")["model"]
            .apply(lambda x: sorted(x.unique().tolist()))
            .to_dict()
        ),
    }

    save_model(best_pipeline, metadata, model_path)
    logger.info(f"\nFinal Test Metrics:\n"
                f"  MAE  = ₹{final_metrics['MAE_lakh']:.2f} Lakh\n"
                f"  RMSE = ₹{final_metrics['RMSE']/1e5:.2f} Lakh\n"
                f"  R²   = {final_metrics['R2']:.4f}")
    return best_pipeline, metadata


if __name__ == "__main__":
    run_training(data_dir="../data")
