import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import ResultsPanel from "./components/ResultsPanel";
import ModelPerformance from "./components/ModelPerformance";
import About from "./components/About";

/**
 * App — single-page layout with two tabs:
 *   1. Predict  (form + results)
 *   2. Model Performance  (charts + tables)
 * Plus an About section in the footer area.
 */
const TABS = [
  { id: "predict", label: "Predict" },
  { id: "performance", label: "Model Performance" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ───────────────────── Header ───────────────────── */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="mx-auto max-w-4xl px-4 sm:px-6">
          {/* Title row */}
          <div className="flex items-center justify-between py-4">
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight">
                Credit Risk Predictor
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Logistic regression from scratch — vs sklearn benchmark
              </p>
            </div>

            {/* Status badge */}
            <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-green-50 border border-green-200 px-3 py-1 text-xs font-medium text-green-700">
              <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
              Models loaded
            </span>
          </div>

          {/* Tab bar */}
          <nav className="flex gap-6 -mb-px" aria-label="Tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`pb-3 text-sm cursor-pointer transition-colors duration-150 ${
                  activeTab === tab.id ? "tab-active" : "tab-inactive"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ───────────────────── Main ───────────────────── */}
      <main className="flex-1 mx-auto w-full max-w-4xl px-4 sm:px-6 py-8">
        {activeTab === "predict" ? (
          <div className="space-y-8">
            {/* Prediction form card */}
            <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-slate-800">
                  Borrower Details
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  Enter the borrower's financial profile to predict
                  default risk. Both models run on the same input.
                </p>
              </div>
              <PredictionForm onResult={setResult} />
            </section>

            {/* Results */}
            {result && (
              <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <ResultsPanel result={result} />
              </section>
            )}

            {/* About */}
            <About />
          </div>
        ) : (
          <ModelPerformance />
        )}
      </main>

      {/* ───────────────────── Footer ───────────────────── */}
      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400">
          <span>
            Built from scratch — no sklearn for training, only for benchmarking.
          </span>
          <a
            href="https://github.com/rexyrocks/credit-risk-from-scratch"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-600 transition-colors"
          >
            github.com/rexyrocks/credit-risk-from-scratch
          </a>
        </div>
      </footer>
    </div>
  );
}
