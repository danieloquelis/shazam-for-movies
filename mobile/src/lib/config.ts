import Constants from 'expo-constants';

/**
 * Backend configuration. Values come from environment variables at build time
 * (loaded from `.env` via `app.config.ts`) so different builds — local dev,
 * device dev, staging, prod — can point at different backends without code
 * changes.
 *
 * See `.env.example` for the full list and `app.config.ts` for the wiring.
 */
type AppExtra = {
  apiUrl?: string;
  apiKey?: string;
};

const extra = (Constants.expoConfig?.extra ?? {}) as AppExtra;

export const API_URL = extra.apiUrl ?? 'http://localhost:8000';
export const API_KEY = extra.apiKey ?? 'dev-key-change-me';

/**
 * Length of each scan recording, in seconds. The matcher gets best results
 * with 5–10 s of footage.
 */
export const SCAN_DURATION_SEC = 5;
