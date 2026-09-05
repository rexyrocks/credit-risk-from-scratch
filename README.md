# RiskSight: Credit Risk Prediction from Scratch

A from-scratch implementation of logistic regression (no `sklearn.fit()` shortcuts) to predict loan default / credit risk, with the math — gradient descent, cost function, L1/L2 regularization — derived and coded manually. `sklearn` is used only afterward, as a benchmark to validate correctness.

## Motivation

Most credit risk projects import `sklearn.linear_model.LogisticRegression` and call `.fit()`. This project instead builds the model from first principles — deriving the sigmoid, cross-entropy cost, and gradient descent update rules by hand — to demonstrate a working understanding of the underlying math, not just the API.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) — ~150,000 anonymized borrower records. Target: `SeriousDlqin2yrs` — whether the borrower experienced 90+ days delinquency within 2 years. 10 features including revolving credit utilization, age, debt ratio, monthly income, and payment-history counts.

The dataset is imbalanced: only ~6.7% of borrowers are actual defaults.

## Project Structure

```
├── data/                    # Kaggle dataset (cs-training.csv, cs-test.csv, etc.)
├── results/                 # Saved outputs
├── logistic_regression.py   # Core model class — sigmoid, cost, gradient, fit, predict, L1/L2 regularization
├── preprocessing.py         # Missing-value cleaning, feature scaling, train-test split (all from scratch)
├── train.py                 # Trains the model on the processed data
├── evaluate.py              # Metrics, threshold tuning, sklearn comparison
├── backend/                 # FastAPI API — serves both models for predictions
│   ├── main.py              # App startup, /predict, /metrics endpoints
│   ├── schemas.py           # Pydantic request/response validation
│   └── requirements.txt
├── frontend/                # React + Vite + Tailwind CSS v4
│   ├── src/
│   │   ├── App.jsx          # Main layout with tabs
│   │   ├── api.js           # API client
│   │   └── components/      # PredictionForm, ResultsPanel, ModelPerformance, About
│   └── vite.config.js
├── docker-compose.yml       # One-command local demo
├── WEBAPP.md                # Web app setup instructions
└── README.md
```

## Implementation Details

**`logistic_regression.py`** implements, from scratch (NumPy only):
- Sigmoid activation
- Binary cross-entropy cost
- Gradient descent parameter updates
- L1 (Lasso) and L2 (Ridge) regularization, with correctly derived gradients (`sign(w)` for L1, `w` for L2 — bias term left unregularized)
- `predict_proba()` and `predict()` with a configurable decision threshold

**`preprocessing.py`** implements, from scratch:
- Missing-value imputation (median for `MonthlyIncome`, mode for `NumberOfDependents`)
- Feature standardization ((x − mean) / std)
- Train-test split with a fixed random seed for reproducibility

## Results

Evaluated on a held-out 20% test split (30,000 rows).

### Default threshold (0.5)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Scratch implementation (with balanced class weights) | 0.104 | 0.673 | 0.180 |
| sklearn `LogisticRegression` (default, no weights) | 0.524 | 0.039 | 0.072 |

By implementing `class_weight="balanced"` in our custom model, the recall dramatically improves from ~3.9% (Sklearn default) to ~67.3% at the standard 0.5 threshold, making it much better at actually catching defaults.

### Threshold tuning (Prioritizing Monetary Profit)

Because credit risk models are fundamentally about minimizing financial loss, we evaluated the model using a hypothetical profit matrix:
- **True Negative** (Correctly granting a loan): Earns **$1,000** in interest.
- **False Negative** (Granting a loan to a defaulter): Loses **$5,000** in principal.
- **True Positive / False Positive** (Denying a loan): Yields **$0**.

Sweeping the decision threshold reveals how optimizing for money differs from optimizing for pure F1-score:

| Threshold | Recall | False Positives | False Negatives | Expected Profit |
|---|---|---|---|---|
| 0.10 | 1.000 | 28,039 | 0 | $5,000 |
| 0.30 | 0.982 | 26,015 | 35 | $1,854,000 |
| 0.50 | 0.673 | 11,388 | 640 | $13,456,000 |
| 0.60 | 0.346 | 3,548 | 1,280 | $18,096,000 |
| **0.70** | **0.039** | **79** | **1,879** | **$18,570,000** |
| 0.80 | 0.015 | 29 | 1,927 | $18,380,000 |

Optimal expected profit occurs at **threshold = 0.70**. While it seems counter-intuitive to drop recall this low, the sheer volume of good customers (True Negatives) in the highly imbalanced dataset means that the extra interest earned from loosening the credit policy outweighs the additional default losses.

## Web App

An interactive web interface lets you submit borrower profiles and see predictions from both models side-by-side, along with charts visualising the threshold sweep and model comparison.

**🔗 Live demo:** [frontend-rose-eight-31.vercel.app](https://frontend-rose-eight-31.vercel.app)

> **Note:** The Vercel deployment hosts the frontend only. The FastAPI backend can be deployed manually on [Render](https://render.com/) as a free Web Service. To test locally, you can still run the backend — see [WEBAPP.md](WEBAPP.md) for setup instructions.

### Running locally

```bash
# Terminal 1 — Backend (from project root)
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
```

Or with Docker:

```bash
docker-compose up --build
```

## Setup (core pipeline only)

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

## Limitations & Next Steps
- A single linear decision boundary limits performance on this problem relative to ensemble methods (e.g. gradient boosting), which are standard in production credit scoring.
- Feature engineering (e.g. interaction terms) was not explored — the focus of this project was validating the from-scratch training pipeline, not maximizing leaderboard performance.

## Status

Core pipeline (preprocessing → training → evaluation → sklearn validation) complete. Full-stack web app deployed to Vercel.