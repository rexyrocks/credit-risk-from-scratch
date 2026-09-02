# Web App — How to Run Locally

Interactive web frontend for the Credit Risk Prediction project.  
The backend (FastAPI) serves both the from-scratch logistic regression model and an sklearn benchmark; the frontend (React + Vite + Tailwind v4) lets you submit borrower profiles, view predictions, and explore model performance.

## Prerequisites

- **Python 3.10+** (tested on 3.13)
- **Node.js 18+** (tested on 22)
- The Kaggle dataset `cs-training.csv` in `data/` (should already be present from the base project)

## Quick Start

### 1. Backend

```bash
# From the project root:
pip install -r backend/requirements.txt

# Start the server (first run trains both models — ~30 s):
python -m uvicorn backend.main:app --reload
```

The API is now running at **http://localhost:8000**.  
On the very first run, both models are trained and cached to `backend/cache/`.  
Subsequent restarts load from cache instantly.

### 2. Frontend

```bash
# In a second terminal:
cd frontend
npm install
npm run dev
```

The app is now running at **http://localhost:5173**.

## Docker (alternative)

```bash
docker-compose up --build
```

This starts both services — backend on `:8000`, frontend on `:5173`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Score a borrower with both models. Accepts 10 raw feature values in JSON. |
| `GET` | `/metrics` | Threshold sweep table + scratch-vs-sklearn comparison. |
| `GET` | `/health` | Liveness check — confirms models are loaded. |

### Example — `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.3,
    "MonthlyIncome": 6000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 2,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }'
```

### Response

```json
{
  "scratch_model": {
    "probability": 0.150857,
    "prediction": 0,
    "label": "Low risk"
  },
  "sklearn_model": {
    "probability": 0.104524,
    "prediction": 0,
    "label": "Low risk"
  },
  "threshold_used": 0.16
}
```

## Project Structure (web app additions)

```
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — startup, /predict, /metrics
│   ├── schemas.py            # Pydantic request/response models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── cache/               # Auto-generated — cached trained models
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main layout with tabs
│   │   ├── api.js            # API helper (fetch wrapper)
│   │   ├── index.css         # Tailwind v4 + design tokens
│   │   ├── main.jsx          # React entry point
│   │   └── components/
│   │       ├── PredictionForm.jsx    # 10-feature input form
│   │       ├── ResultsPanel.jsx      # Side-by-side model results
│   │       ├── ModelPerformance.jsx  # Charts + tables
│   │       └── About.jsx            # Project description
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── WEBAPP.md                 # ← this file
```
