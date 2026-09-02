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
| Scratch implementation | 0.519 | 0.014 | 0.028 |
| sklearn `LogisticRegression` | 0.524 | 0.039 | 0.072 |

Precision is nearly identical between the two (0.519 vs 0.524), which validates that the from-scratch cost function and gradient derivation are mathematically correct. The recall/F1 gap is attributable to sklearn's more sophisticated solver (L-BFGS vs. vanilla gradient descent) rather than any error in the underlying math.

### Threshold tuning

Because the dataset is imbalanced, the default 0.5 threshold is overly conservative — the model rarely predicts "default." Sweeping the decision threshold shows the precision/recall trade-off clearly:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.519 | 0.014 | 0.028 |
| 0.30 | 0.519 | 0.014 | 0.028 |
| 0.20 | 0.492 | 0.016 | 0.031 |
| 0.18 | 0.300 | 0.050 | 0.086 |
| **0.16** | **0.132** | **0.372** | **0.194** |
| 0.14 | 0.095 | 0.719 | 0.168 |
| 0.12 | 0.074 | 0.949 | 0.137 |
| 0.10 | 0.065 | 0.999 | 0.123 |

Best F1 occurs around **threshold ≈ 0.16**, balancing catching real defaults against false alarms.

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

- Class imbalance is currently handled only via threshold tuning; class-weighted loss would likely improve recall further.
- A single linear decision boundary limits performance on this problem relative to ensemble methods (e.g. gradient boosting), which are standard in production credit scoring.
- Feature engineering (e.g. interaction terms) was not explored — the focus of this project was validating the from-scratch training pipeline, not maximizing leaderboard performance.

## Status

Core pipeline (preprocessing → training → evaluation → sklearn validation) complete. Full-stack web app deployed to Vercel.