# Credit Risk Predictor — Logistic Regression from Scratch

A logistic regression model built from first principles (no sklearn for the core model) to predict loan default risk. Gradient descent and L1/L2 regularization derived and implemented manually, then validated against scikit-learn's implementation.

## Structure
- `logistic_regression.py` — core model class (fit, predict, gradient descent, regularization)
- `preprocessing.py` — data cleaning, feature scaling, train-test split
- `train.py` — trains the model on the dataset
- `evaluate.py` — metrics, precision-recall, ROC-AUC, comparison with scikit-learn
- `data/` — dataset
- `results/` — plots, logs, saved outputs

## Setup
\`\`\`bash
pip install -r requirements.txt
python train.py
\`\`\`

## Status
Work in progress.