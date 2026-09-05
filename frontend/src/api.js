/**
 * API helper — centralises all fetch calls to the FastAPI backend.
 * Base URL defaults to localhost:8000 for local development.
 */

// In production (Vercel), set VITE_API_URL to the deployed backend URL.
// Locally it falls back to the FastAPI dev server.
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Helper to retry fetch requests. 
 * Render's free tier sleeps after 15 mins. Waking up can take time and cause 
 * 502/503/504 errors or network drops. We retry up to 5 times with a 5s delay.
 */
async function fetchWithRetry(url, options = {}, retries = 5, delay = 5000) {
  for (let i = 0; i < retries; i++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);
      
      // If the response is successful, or it's a 4xx error (client error), don't retry.
      if (res.ok || (res.status >= 400 && res.status < 500)) {
        return res;
      }
      // If it's a 5xx error, it might be Render waking up. Throw to trigger retry.
      if (res.status >= 500 && res.status < 600) {
        throw new Error(`Server error (${res.status})`);
      }
      return res; // Fallback
    } catch (err) {
      clearTimeout(timeoutId);
      if (i === retries - 1) throw err;
      console.warn(`Fetch failed (${err.name === 'AbortError' ? 'timeout' : err.message}). Retrying in ${delay}ms... (Attempt ${i + 1} of ${retries - 1})`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

/**
 * POST /predict
 * @param {Object} features – the 10 raw feature values
 * @returns {Promise<Object>} – { scratch_model, sklearn_model, threshold_used }
 */
export async function fetchPrediction(features) {
  const res = await fetchWithRetry(`${API_BASE}/predict`, {
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
  const res = await fetchWithRetry(`${API_BASE}/metrics`);

  if (!res.ok) {
    throw new Error(`Failed to load metrics (${res.status})`);
  }

  return res.json();
}
