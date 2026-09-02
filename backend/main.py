"""
FastAPI backend for the Credit Risk Prediction web app.

On startup the server:
  1.  Loads and preprocesses data/cs-training.csv using the existing
      preprocessing.py functions (no modifications to that file).
  2.  Trains *both* the from-scratch logistic-regression model and an
      sklearn LogisticRegression on the same train/test split.
  3.  Caches every artefact (models, scaling params, test data) to
      backend/cache/ via joblib so subsequent startups are instant.

Endpoints
---------
POST /predict   – score a single borrower with both models
GET  /metrics   – threshold-sweep table + scratch-vs-sklearn comparison
GET  /health    – simple liveness check
"""

import os
import sys
import pathlib

import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    SingleModelResult,
    MetricsResponse,
    ThresholdRow,
    ModelMetricsSummary,
)

# ---------------------------------------------------------------------------
# Resolve project root so we can import the original modules that live
# one level above backend/
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from the original, unmodified project files
from preprocessing import load_data, cleandata, scalefeature, train_test_split
from logistic_regression import logisitic_regressionScratch          # note: original typo preserved
from evaluate import confusematrix_compenents, compute_metrics       # note: original typo preserved
from sklearn.linear_model import LogisticRegression as SklearnLR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"
DEFAULT_THRESHOLD = 0.16

# Thresholds for the sweep — includes the README set plus intermediate
# values requested by the user for a smoother chart.
SWEEP_THRESHOLDS = sorted(
    [0.50, 0.40, 0.30, 0.25, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08],
    reverse=True,
)

# The 10 feature columns in the order they appear after preprocessing
# (i.e. after dropping 'Unnamed: 0' and 'SeriousDlqin2yrs').
FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ---------------------------------------------------------------------------
# In-memory store populated at startup
# ---------------------------------------------------------------------------
store: dict = {}


def _train_and_cache() -> dict:
    """
    Load data, preprocess, train both models, compute metrics,
    and persist everything to CACHE_DIR.
    """
    print("[startup] Loading data …")
    df = load_data(str(DATA_PATH))
    df = cleandata(df)

    y = df["SeriousDlqin2yrs"].values
    X = df.drop(columns=["SeriousDlqin2yrs"]).values

    X_scaled, mean, std = scalefeature(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

    # --- scratch model ---
    print("[startup] Training scratch model (1 000 iterations) …")
    scratch = logisitic_regressionScratch(learning_rate=0.01, n_iterations=1000)
    scratch.fit(X_train, y_train)

    # --- sklearn model ---
    print("[startup] Training sklearn model …")
    sklearn_model = SklearnLR(max_iter=1000)
    sklearn_model.fit(X_train, y_train)

    # --- threshold sweep (both models) ---
    print("[startup] Computing threshold sweep …")
    scratch_sweep = _sweep(scratch, X_test, y_test)
    sklearn_sweep = _sweep_sklearn(sklearn_model, X_test, y_test)

    bundle = {
        "scratch_model": scratch,
        "sklearn_model": sklearn_model,
        "mean": mean,
        "std": std,
        "X_test": X_test,
        "y_test": y_test,
        "scratch_sweep": scratch_sweep,
        "sklearn_sweep": sklearn_sweep,
    }

    # Persist to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CACHE_DIR / "models.joblib")
    print("[startup] Cached models to", CACHE_DIR / "models.joblib")

    return bundle


def _sweep(model, X_test, y_test) -> list[dict]:
    """Run threshold sweep for the scratch model."""
    rows = []
    for t in SWEEP_THRESHOLDS:
        preds = model.predict(X_test, threshold=t)
        TP, TN, FP, FN = confusematrix_compenents(y_test, preds)
        acc, prec, rec, f1 = compute_metrics(TP, TN, FP, FN)
        rows.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "is_optimal": t == DEFAULT_THRESHOLD,
        })
    return rows


def _sweep_sklearn(model, X_test, y_test) -> list[dict]:
    """Run threshold sweep for the sklearn model."""
    probas = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in SWEEP_THRESHOLDS:
        preds = (probas >= t).astype(int)
        TP, TN, FP, FN = confusematrix_compenents(y_test, preds)
        acc, prec, rec, f1 = compute_metrics(TP, TN, FP, FN)
        rows.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "is_optimal": t == DEFAULT_THRESHOLD,
        })
    return rows


# ---------------------------------------------------------------------------
# Lifespan: load or train models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cache_path = CACHE_DIR / "models.joblib"
    if cache_path.exists():
        print("[startup] Loading cached models …")
        bundle = joblib.load(cache_path)
    else:
        bundle = _train_and_cache()

    store.update(bundle)
    print("[startup] Ready.")
    yield
    # Shutdown — nothing to clean up


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Compares a from-scratch logistic regression with sklearn's "
        "LogisticRegression on the Give Me Some Credit dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite dev-server and common local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    """
    Accept 10 raw (unscaled) feature values, scale them using the
    training-set mean/std, and return predictions from both models.
    """
    # Build a 1-D numpy array in the same column order the model expects.
    # The Pydantic field names use underscores where the CSV has dashes,
    # so we map explicitly.
    raw = np.array([
        body.RevolvingUtilizationOfUnsecuredLines,
        body.age,
        body.NumberOfTime30_59DaysPastDueNotWorse,
        body.DebtRatio,
        body.MonthlyIncome,
        body.NumberOfOpenCreditLinesAndLoans,
        body.NumberOfTimes90DaysLate,
        body.NumberRealEstateLoansOrLines,
        body.NumberOfTime60_89DaysPastDueNotWorse,
        body.NumberOfDependents,
    ], dtype=float).reshape(1, -1)

    # Scale using the saved training-set statistics
    mean = store["mean"]
    std = store["std"]
    scaled = (raw - mean) / std

    # --- Scratch model ---
    scratch = store["scratch_model"]
    scratch_prob = float(scratch.predict_proba(scaled)[0])
    scratch_pred = int(scratch_prob >= DEFAULT_THRESHOLD)
    scratch_label = "Likely to default" if scratch_pred == 1 else "Low risk"

    # --- Sklearn model ---
    sklearn_model = store["sklearn_model"]
    sklearn_prob = float(sklearn_model.predict_proba(scaled)[0, 1])
    sklearn_pred = int(sklearn_prob >= DEFAULT_THRESHOLD)
    sklearn_label = "Likely to default" if sklearn_pred == 1 else "Low risk"

    return PredictionResponse(
        scratch_model=SingleModelResult(
            probability=round(scratch_prob, 6),
            prediction=scratch_pred,
            label=scratch_label,
        ),
        sklearn_model=SingleModelResult(
            probability=round(sklearn_prob, 6),
            prediction=sklearn_pred,
            label=sklearn_label,
        ),
        threshold_used=DEFAULT_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------
@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Return the full threshold-sweep table for both models, plus a
    head-to-head comparison at threshold 0.5 (the sklearn default).
    """
    scratch_sweep = store["scratch_sweep"]
    sklearn_sweep = store["sklearn_sweep"]

    # Find the t=0.5 row for each model to build the comparison summary
    scratch_05 = next(r for r in scratch_sweep if r["threshold"] == 0.5)
    sklearn_05 = next(r for r in sklearn_sweep if r["threshold"] == 0.5)

    comparison = [
        ModelMetricsSummary(
            model_name="Scratch (from scratch)",
            threshold=0.5,
            precision=scratch_05["precision"],
            recall=scratch_05["recall"],
            f1=scratch_05["f1"],
            accuracy=scratch_05["accuracy"],
        ),
        ModelMetricsSummary(
            model_name="sklearn LogisticRegression",
            threshold=0.5,
            precision=sklearn_05["precision"],
            recall=sklearn_05["recall"],
            f1=sklearn_05["f1"],
            accuracy=sklearn_05["accuracy"],
        ),
    ]

    return MetricsResponse(
        scratch_threshold_sweep=[ThresholdRow(**r) for r in scratch_sweep],
        sklearn_threshold_sweep=[ThresholdRow(**r) for r in sklearn_sweep],
        comparison_at_0_5=comparison,
        optimal_threshold=DEFAULT_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": bool(store)}
