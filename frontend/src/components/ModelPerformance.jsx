import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { fetchMetrics } from "../api";

/**
 * ModelPerformance — fetches GET /metrics and renders:
 *  1. A line chart of the threshold sweep (F1 vs threshold) for both models
 *  2. A table of the full sweep with the optimal threshold highlighted
 *  3. A bar chart comparing scratch vs sklearn at threshold 0.5
 */
export default function ModelPerformance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400">
        <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
        Loading metrics…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
        <span className="font-medium">Error:</span> {error}
      </div>
    );
  }

  // -----------------------------------------------------------------
  // Prepare chart data
  // -----------------------------------------------------------------

  // Combined sweep data for the line chart (merge by threshold)
  const sweepChart = data.scratch_threshold_sweep.map((row, i) => ({
    threshold: row.threshold,
    "Scratch F1": row.f1,
    "Sklearn F1": data.sklearn_threshold_sweep[i].f1,
    "Scratch Recall": row.recall,
    "Sklearn Recall": data.sklearn_threshold_sweep[i].recall,
    "Scratch Precision": row.precision,
    "Sklearn Precision": data.sklearn_threshold_sweep[i].precision,
  }));

  // Comparison at threshold 0.5 for the bar chart
  const comparisonBar = [
    {
      metric: "Precision",
      Scratch: data.comparison_at_0_5[0].precision,
      Sklearn: data.comparison_at_0_5[1].precision,
    },
    {
      metric: "Recall",
      Scratch: data.comparison_at_0_5[0].recall,
      Sklearn: data.comparison_at_0_5[1].recall,
    },
    {
      metric: "F1",
      Scratch: data.comparison_at_0_5[0].f1,
      Sklearn: data.comparison_at_0_5[1].f1,
    },
  ];

  return (
    <div className="space-y-10">
      {/* ---- Section 1: F1 / Recall / Precision vs Threshold ---- */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">
            Threshold Sweep — F1, Precision &amp; Recall
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            How each metric changes as the decision threshold is lowered from
            0.50 → 0.08. The dashed line marks the F1-optimal threshold (0.16).
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={sweepChart} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="threshold"
                reversed
                tick={{ fontSize: 12 }}
                label={{ value: "Threshold", position: "insideBottom", offset: -4, fontSize: 12 }}
              />
              <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  fontSize: "12px",
                }}
              />
              <Legend verticalAlign="top" height={36} iconType="plainline" />
              <ReferenceLine
                x={data.optimal_threshold}
                stroke="#4f46e5"
                strokeDasharray="6 3"
                label={{ value: "Optimal", fill: "#4f46e5", fontSize: 11, position: "top" }}
              />
              {/* F1 curves */}
              <Line type="monotone" dataKey="Scratch F1" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="Sklearn F1" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} />
              {/* Recall curves */}
              <Line type="monotone" dataKey="Scratch Recall" stroke="#6366f1" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
              <Line type="monotone" dataKey="Sklearn Recall" stroke="#06b6d4" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ---- Section 2: Threshold sweep table ---- */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">
            Detailed Threshold Sweep
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Scratch model metrics at each threshold.
            Row <span className="font-medium text-brand-600">highlighted in indigo</span> is the
            F1-optimal threshold.
          </p>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-200 shadow-sm">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-slate-100 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                <th className="px-4 py-3">Threshold</th>
                <th className="px-4 py-3">Precision</th>
                <th className="px-4 py-3">Recall</th>
                <th className="px-4 py-3">F1</th>
                <th className="px-4 py-3">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {data.scratch_threshold_sweep.map((row) => (
                <tr
                  key={row.threshold}
                  className={row.is_optimal ? "row-optimal" : "hover:bg-slate-50"}
                >
                  <td className="px-4 py-2.5 font-mono tabular-nums">{row.threshold.toFixed(2)}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{row.precision.toFixed(4)}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{row.recall.toFixed(4)}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{row.f1.toFixed(4)}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{row.accuracy.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ---- Section 3: Scratch vs Sklearn at 0.5 ---- */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">
            Scratch vs Sklearn — Threshold 0.5
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            At the default 0.5 threshold, precision is nearly identical
            (validating the from-scratch math). The recall gap comes from
            sklearn's L-BFGS solver, not from any error in the cost function.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={comparisonBar} barGap={6} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="metric" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 12 }} domain={[0, 0.6]} />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  fontSize: "12px",
                }}
              />
              <Legend verticalAlign="top" height={36} />
              <Bar dataKey="Scratch" fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Sklearn" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
