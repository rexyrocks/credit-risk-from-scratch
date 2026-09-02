import { useState } from "react";
import { fetchPrediction } from "../api";

/* ---------------------------------------------------------------
   Feature definitions — label, key, helper text, and input config.
   The key uses underscores (matching the Pydantic model), while the
   description is pulled from the Kaggle data dictionary.
   --------------------------------------------------------------- */
const FEATURES = [
  {
    key: "RevolvingUtilizationOfUnsecuredLines",
    label: "Revolving Utilization",
    helper:
      "Total balance on credit cards and personal lines of credit divided by the sum of credit limits. Values > 1 mean the borrower is over-limit.",
    type: "slider",
    min: 0,
    max: 1.5,
    step: 0.01,
    default: 0.3,
  },
  {
    key: "age",
    label: "Age",
    helper: "Age of the borrower in years.",
    type: "slider",
    min: 18,
    max: 100,
    step: 1,
    default: 40,
  },
  {
    key: "NumberOfTime30_59DaysPastDueNotWorse",
    label: "30–59 Days Past Due",
    helper:
      "Number of times the borrower has been 30–59 days past due (but not worse) in the last 2 years.",
    type: "number",
    min: 0,
    max: 20,
    step: 1,
    default: 0,
  },
  {
    key: "DebtRatio",
    label: "Debt Ratio",
    helper:
      "Monthly debt payments, alimony, and living costs divided by monthly gross income.",
    type: "slider",
    min: 0,
    max: 2,
    step: 0.01,
    default: 0.3,
  },
  {
    key: "MonthlyIncome",
    label: "Monthly Income ($)",
    helper: "Monthly gross income in dollars.",
    type: "number",
    min: 0,
    max: 100000,
    step: 100,
    default: 5000,
  },
  {
    key: "NumberOfOpenCreditLinesAndLoans",
    label: "Open Credit Lines & Loans",
    helper:
      "Number of open loans (e.g. car loan, mortgage) and lines of credit (e.g. credit cards).",
    type: "number",
    min: 0,
    max: 60,
    step: 1,
    default: 8,
  },
  {
    key: "NumberOfTimes90DaysLate",
    label: "90+ Days Late",
    helper: "Number of times the borrower has been 90 or more days late.",
    type: "number",
    min: 0,
    max: 20,
    step: 1,
    default: 0,
  },
  {
    key: "NumberRealEstateLoansOrLines",
    label: "Real Estate Loans",
    helper:
      "Number of mortgage and real-estate loans, including home equity lines of credit.",
    type: "number",
    min: 0,
    max: 30,
    step: 1,
    default: 1,
  },
  {
    key: "NumberOfTime60_89DaysPastDueNotWorse",
    label: "60–89 Days Past Due",
    helper:
      "Number of times the borrower has been 60–89 days past due (but not worse) in the last 2 years.",
    type: "number",
    min: 0,
    max: 20,
    step: 1,
    default: 0,
  },
  {
    key: "NumberOfDependents",
    label: "Dependents",
    helper: "Number of dependents in the household (excluding the borrower).",
    type: "number",
    min: 0,
    max: 15,
    step: 1,
    default: 1,
  },
];

/* ---------------------------------------------------------------
   PredictionForm component
   --------------------------------------------------------------- */
export default function PredictionForm({ onResult }) {
  // Initialise form values from the feature defaults
  const [values, setValues] = useState(() =>
    Object.fromEntries(FEATURES.map((f) => [f.key, f.default]))
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (key, raw) => {
    const feat = FEATURES.find((f) => f.key === key);
    // Keep numbers as numbers, not strings
    const parsed = feat.step >= 1 ? parseInt(raw, 10) : parseFloat(raw);
    setValues((prev) => ({ ...prev, [key]: isNaN(parsed) ? "" : parsed }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await fetchPrediction(values);
      onResult(result);
    } catch (err) {
      setError(err.message || "Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      id="prediction-form"
      onSubmit={handleSubmit}
      className="space-y-5"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
        {FEATURES.map((feat) => (
          <div key={feat.key} className="space-y-1.5">
            {/* Label row */}
            <div className="flex items-baseline justify-between">
              <label
                htmlFor={feat.key}
                className="text-sm font-medium text-slate-700"
              >
                {feat.label}
              </label>

              {/* Live numeric readout for sliders */}
              {feat.type === "slider" && (
                <span className="text-xs font-mono text-slate-500 tabular-nums">
                  {typeof values[feat.key] === "number"
                    ? values[feat.key].toFixed(feat.step < 1 ? 2 : 0)
                    : "—"}
                </span>
              )}
            </div>

            {/* Input */}
            {feat.type === "slider" ? (
              <input
                id={feat.key}
                type="range"
                min={feat.min}
                max={feat.max}
                step={feat.step}
                value={values[feat.key]}
                onChange={(e) => handleChange(feat.key, e.target.value)}
                className="w-full accent-brand-600 cursor-pointer"
              />
            ) : (
              <input
                id={feat.key}
                type="number"
                min={feat.min}
                max={feat.max}
                step={feat.step}
                value={values[feat.key]}
                onChange={(e) => handleChange(feat.key, e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2
                           text-sm text-slate-800 shadow-sm
                           focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20
                           outline-none transition"
              />
            )}

            {/* Helper text */}
            <p className="text-xs text-slate-400 leading-relaxed">
              {feat.helper}
            </p>
          </div>
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div
          id="prediction-error"
          className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700"
        >
          <span className="font-medium">Error:</span> {error}
        </div>
      )}

      {/* Submit */}
      <button
        id="submit-prediction"
        type="submit"
        disabled={loading}
        className="w-full cursor-pointer rounded-lg bg-brand-600 px-5 py-3 text-sm font-semibold
                   text-white shadow-md hover:bg-brand-700 active:bg-brand-800
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transition-colors duration-150"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            {/* Simple spinner */}
            <svg
              className="animate-spin h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
              />
            </svg>
            Running models…
          </span>
        ) : (
          "Predict Default Risk"
        )}
      </button>
    </form>
  );
}
