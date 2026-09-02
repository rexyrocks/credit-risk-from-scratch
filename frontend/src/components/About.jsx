/**
 * About — brief section explaining the from-scratch approach
 * and linking back to the GitHub repository.
 */

export default function About() {
  return (
    <section id="about-section" className="space-y-4">
      <h2 className="text-lg font-semibold text-slate-800">
        About This Project
      </h2>

      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-600 leading-relaxed">
        <p>
          Most credit-risk projects import{" "}
          <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded font-mono">
            sklearn.linear_model.LogisticRegression
          </code>{" "}
          and call{" "}
          <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded font-mono">
            .fit()
          </code>
          . This project instead builds the model{" "}
          <span className="font-semibold text-slate-700">from first principles</span>{" "}
          — deriving the sigmoid, cross-entropy cost, and gradient-descent
          update rules by hand — to demonstrate a working understanding of
          the underlying math, not just the API.
        </p>

        <p>
          Sklearn is used <em>only</em> afterward as a benchmark to validate
          correctness. At the default 0.5 threshold, precision is nearly
          identical between the two implementations (0.519 vs 0.524),
          confirming the from-scratch math is correct.
        </p>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <a
            href="https://github.com/rexyrocks/credit-risk-from-scratch"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800 transition-colors"
          >
            {/* GitHub icon */}
            <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
            View on GitHub
          </a>

          <a
            href="https://www.kaggle.com/c/GiveMeSomeCredit"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
          >
            Kaggle Dataset ↗
          </a>
        </div>
      </div>
    </section>
  );
}
