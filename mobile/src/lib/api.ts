import { API_KEY, API_URL } from './config';

export type MatchSuccess = {
  status: 'match';
  movieId: number;
  title: string;
  timestampSec: number;
  timestampHuman: string;
  confidence: number;
  visualScore: number;
  matchDetails: Record<string, unknown>;
};

export type MatchMiss = {
  status: 'no_match';
};

export type MatchError = {
  status: 'error';
  message: string;
};

export type QueryResult = MatchSuccess | MatchMiss | MatchError;

/**
 * Upload a recorded clip to the backend's `/query` endpoint.
 *
 * Returns one of three shapes — caller pattern-matches on `status`. We
 * deliberately do NOT throw for 422 ("no_match") because that's an expected
 * outcome, not an error.
 */
export async function queryClip(localUri: string): Promise<QueryResult> {
  const form = new FormData();
  // React Native's FormData accepts the {uri, name, type} shape — TS thinks it
  // wants a Blob, hence the cast.
  form.append(
    'file',
    {
      uri: localUri,
      name: 'scan.mov',
      type: 'video/quicktime',
    } as unknown as Blob,
  );

  let response: Response;
  try {
    response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'x-api-key': API_KEY,
      },
      body: form,
    });
  } catch (e) {
    return { status: 'error', message: e instanceof Error ? e.message : 'network error' };
  }

  if (response.status === 422) {
    return { status: 'no_match' };
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore — keep the default detail string
    }
    return { status: 'error', message: detail };
  }

  const body = (await response.json()) as {
    movie_id: number;
    title: string;
    timestamp_sec: number;
    timestamp_human: string;
    confidence: number;
    visual_score: number;
    match_details: Record<string, unknown>;
  };

  return {
    status: 'match',
    movieId: body.movie_id,
    title: body.title,
    timestampSec: body.timestamp_sec,
    timestampHuman: body.timestamp_human,
    confidence: body.confidence,
    visualScore: body.visual_score,
    matchDetails: body.match_details,
  };
}
