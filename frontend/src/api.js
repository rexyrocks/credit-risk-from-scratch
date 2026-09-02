/**
 * API helper — centralises all fetch calls to the FastAPI backend.
 * Base URL defaults to localhost:8000 for local development.
 */

// In production (Vercel), set VITE_API_URL to the deployed backend URL.
// Locally it falls back to the FastAPI dev server.
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * POST /predict
 * @param {Object} features – the 10 raw feature values
 * @returns {Promise<Object>} – { scratch_model, sklearn_model, threshold_used }
 */
export async function fetchPrediction(features) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Server error (${res.status})`);
  }

  return res.json();
}

/**
 * GET /metrics
 * @returns {Promise<Object>} – threshold sweeps + comparison data
 */
export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);

  if (!res.ok) {
    throw new Error(`Failed to load metrics (${res.status})`);
  }

  return res.json();
}
