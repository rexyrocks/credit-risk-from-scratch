"""
Pydantic schemas for request/response validation.

Each of the 10 credit-risk features has constraints based on reasonable
real-world ranges.  Some bounds are generous (e.g. DebtRatio up to 50 000)
because the raw Kaggle data contains outliers that exceed naive expectations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Request: the 10 raw (unscaled) feature values a user submits
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    JSON body for POST /predict.

    Field names use underscores for the dash-containing Kaggle columns
    (e.g. NumberOfTime30_59DaysPastDueNotWorse) so they are valid Python
    identifiers.  The backend maps them back to the original column order
    before scaling.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ...,
        ge=0,
        le=50000,
        description=(
            "Total balance on credit cards and personal lines of credit "
            "divided by the sum of credit limits.  Values > 1 indicate "
            "the borrower is over-limit."
        ),
    )
    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age of the borrower in years.",
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        le=98,
        description=(
            "Number of times the borrower has been 30–59 days past due "
            "(but not worse) in the last 2 years."
        ),
    )
    DebtRatio: float = Field(
        ...,
        ge=0,
        le=50000,
        description=(
            "Monthly debt payments, alimony, and living costs divided by "
            "monthly gross income."
        ),
    )
    MonthlyIncome: float = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Monthly gross income in dollars.",
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Number of open loans (e.g. car loan, mortgage) and lines of "
            "credit (e.g. credit cards)."
        ),
    )
    NumberOfTimes90DaysLate: int = Field(
        ...,
        ge=0,
        le=98,
        description="Number of times the borrower has been 90+ days late.",
    )
    NumberRealEstateLoansOrLines: int = Field(
        ...,
        ge=0,
        le=60,
        description=(
            "Number of mortgage and real-estate loans, including home "
            "equity lines of credit."
        ),
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        le=98,
        description=(
            "Number of times the borrower has been 60–89 days past due "
            "(but not worse) in the last 2 years."
        ),
    )
    NumberOfDependents: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of dependents in the household (excluding the borrower).",
    )


# ---------------------------------------------------------------------------
# Response: prediction results from both models
# ---------------------------------------------------------------------------

class SingleModelResult(BaseModel):
    """Result from a single model (scratch or sklearn)."""
    probability: float = Field(
        ..., description="Predicted probability of serious delinquency (0–1)."
    )
    prediction: int = Field(
        ..., description="Binary prediction (0 or 1) at the chosen threshold."
    )
    label: str = Field(
        ..., description='Human-readable label: "Likely to default" or "Low risk".'
    )


class PredictionResponse(BaseModel):
    """JSON response for POST /predict."""
    scratch_model: SingleModelResult
    sklearn_model: SingleModelResult
    threshold_used: float


# ---------------------------------------------------------------------------
# Response: model-performance metrics for GET /metrics
# ---------------------------------------------------------------------------

class ThresholdRow(BaseModel):
    """One row of the threshold-sweep table."""
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    is_optimal: bool = Field(
        False,
        description="True for the threshold that maximises F1 (≈ 0.16).",
    )


class ModelMetricsSummary(BaseModel):
    """Aggregate metrics for a single model at a single threshold."""
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float


class MetricsResponse(BaseModel):
    """JSON response for GET /metrics."""
    scratch_threshold_sweep: List[ThresholdRow]
    sklearn_threshold_sweep: List[ThresholdRow]
    comparison_at_0_5: List[ModelMetricsSummary]
    optimal_threshold: float
