/**
 * ResultsPanel — shows side-by-side prediction results from both models.
 *
 * Displays:
 *  - A probability bar (0–100 %) for each model
 *  - The prediction label ("Likely to default" / "Low risk")
 *  - A note about the 0.16 threshold choice
 */

export default function ResultsPanel({ result }) {
  if (!result) return null;

  const { scratch_model, sklearn_model, threshold_used } = result;

  return (
    <div id="results-panel" className="space-y-6 animate-in fade-in">
      <h2 className="text-lg font-semibold text-slate-800">
        Prediction Results
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <ModelCard
          title="Scratch Model"
          subtitle="From-scratch logistic regression"
          data={scratch_model}
        />
        <ModelCard
          title="Sklearn Model"
          subtitle="sklearn LogisticRegression"
          data={sklearn_model}
        />
      </div>

      {/* Threshold explainer */}
      <p className="text-xs text-slate-400 leading-relaxed bg-slate-100 rounded-lg px-4 py-3">
        <span className="font-medium text-slate-500">
          Threshold = {threshold_used}
        </span>{" "}
        — chosen to maximise F1 on this imbalanced dataset (~6.7 % default
        rate). The naïve 0.5 threshold barely predicts any defaults because
        the model's predicted probabilities are concentrated well below 0.5.
      </p>
    </div>
  );
}

/* ---------------------------------------------------------------
   Individual model result card
   --------------------------------------------------------------- */
function ModelCard({ title, subtitle, data }) {
  const isHighRisk = data.prediction === 1;
  const pct = (data.probability * 100).toFixed(1);

  return (
    <div
      className={`rounded-xl border p-5 space-y-4 transition-colors duration-300 ${
        isHighRisk
          ? "border-red-200 bg-risk-high-bg"
          : "border-green-200 bg-risk-low-bg"
      }`}
    >
      {/* Header */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        <p className="text-xs text-slate-400">{subtitle}</p>
      </div>

      {/* Probability bar */}
      <div className="space-y-1.5">
        <div className="flex items-baseline justify-between text-xs text-slate-500">
          <span>Default probability</span>
          <span className="font-mono font-semibold tabular-nums text-slate-700">
            {pct}%
          </span>
        </div>
        <div className="h-3 w-full rounded-full bg-slate-200 overflow-hidden">
          <div
            className="prob-bar"
            style={{
              width: `${Math.min(parseFloat(pct), 100)}%`,
              backgroundColor: isHighRisk ? "#ef4444" : "#22c55e",
            }}
          />
        </div>
      </div>

      {/* Label badge */}
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
            isHighRisk
              ? "bg-red-100 text-red-700"
              : "bg-green-100 text-green-700"
          }`}
        >
          {/* Dot indicator */}
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isHighRisk ? "bg-red-500" : "bg-green-500"
            }`}
          />
          {data.label}
        </span>
      </div>
    </div>
  );
}
