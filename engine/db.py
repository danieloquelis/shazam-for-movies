"""
Database layer with abstract interface for easy backend swapping.

Current implementation: PostgreSQL (metadata + pHash) + FAISS (CLIP embeddings).
Two FAISS indexes: frame-level (keyframes) and scene-level (mean embeddings).

To port to another backend, implement the DatabaseBackend protocol.
"""

import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
import faiss
import psycopg2
from psycopg2.extras import execute_values
from engine.config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    CLIP_EMBEDDING_DIM, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH,
    FAISS_SCENE_INDEX_PATH, FAISS_SCENE_ID_MAP_PATH, DATA_DIR,
)


class DatabaseBackend(ABC):
    """Abstract interface — implement this to swap storage backends."""

    @abstractmethod
    def create_movie(self, title: str, file_path: str, duration_sec: float) -> int:
        ...

    @abstractmethod
    def get_movie(self, movie_id: int) -> dict | None:
        ...

    @abstractmethod
    def movie_exists(self, title: str) -> int | None:
        ...

    @abstractmethod
    def store_visual_fingerprints(self, movie_id: int,
                                   fingerprints: list[tuple[float, int, np.ndarray]],
                                   silent: bool = False):
        ...

    @abstractmethod
    def search_visual(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[int, float, float]]]:
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def close(self):
        ...

    def store_scene_descriptors(self, movie_id: int,
                                 scenes: list[tuple[float, float, int, np.ndarray]],
                                 silent: bool = False):
        """Store scene-level descriptors. Override in subclass."""
        pass

    def search_scenes(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple]]:
        """Search scene index. Override in subclass."""
        return [[] for _ in range(len(query_embeddings))]


class FaissIndex:
    """FAISS index wrapper with lazy loading.

    IMPORTANT: Loading FAISS index from disk before PyTorch causes a segfault
    (OMP thread conflict on macOS). Lazy loading ensures PyTorch/CLIP loads first.
    """

    def __init__(self, index_path: str = None, id_map_path: str = None):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._index_path = index_path or FAISS_INDEX_PATH
        self._id_map_path = id_map_path or FAISS_ID_MAP_PATH
        self._index = None
        self._id_map = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if os.path.exists(self._index_path) and os.path.exists(self._id_map_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._id_map_path, "rb") as f:
                self._id_map = pickle.load(f)
        else:
            self._index = faiss.IndexFlatIP(CLIP_EMBEDDING_DIM)
            self._id_map = []
        self._loaded = True

    @property
    def index(self):
        self._ensure_loaded()
        return self._index

    @property
    def id_map(self):
        self._ensure_loaded()
        return self._id_map

    def add(self, embeddings: np.ndarray, metadata: list):
        self.index.add(embeddings)
        self.id_map.extend(metadata)

    def search(self, query: np.ndarray, top_k: int) -> list[list[tuple]]:
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query))]

        query = np.ascontiguousarray(query, dtype=np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for i in range(len(query)):
            frame_results = []
            for j in range(top_k):
                idx = indices[i][j]
                if idx < 0:
                    continue
                meta = self.id_map[idx]
                frame_results.append((*meta, float(scores[i][j])))
            results.append(frame_results)
        return results

    def save(self):
        faiss.write_index(self.index, self._index_path)
        with open(self._id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def reset(self):
        """Clear the index (in-memory only, call save() to persist)."""
        self._index = faiss.IndexFlatIP(CLIP_EMBEDDING_DIM)
        self._id_map = []
        self._loaded = True


class PostgresDatabase(DatabaseBackend):
    """PostgreSQL + FAISS implementation."""

    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
        )
        self.conn.autocommit = True
        self._create_tables()
        self.faiss = FaissIndex(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)
        self.faiss_scenes = FaissIndex(FAISS_SCENE_INDEX_PATH, FAISS_SCENE_ID_MAP_PATH)

    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT,
                    duration_sec FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS visual_fingerprints (
                    id SERIAL PRIMARY KEY,
                    movie_id INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
                    timestamp_sec FLOAT NOT NULL,
                    phash TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_visual_phash
                ON visual_fingerprints (phash);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    id SERIAL PRIMARY KEY,
                    movie_id INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
                    start_sec FLOAT NOT NULL,
                    end_sec FLOAT NOT NULL,
                    n_keyframes INTEGER NOT NULL
                );
            """)

    def create_movie(self, title: str, file_path: str, duration_sec: float) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO movies (title, file_path, duration_sec) VALUES (%s, %s, %s) RETURNING movie_id",
                (title, file_path, duration_sec),
            )
            return cur.fetchone()[0]

    def get_movie(self, movie_id: int) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT movie_id, title, duration_sec FROM movies WHERE movie_id = %s", (movie_id,))
            row = cur.fetchone()
            if row:
                return {"movie_id": row[0], "title": row[1], "duration_sec": row[2]}
            return None

    def movie_exists(self, title: str) -> int | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT movie_id FROM movies WHERE title = %s", (title,))
            row = cur.fetchone()
            return row[0] if row else None

    def store_visual_fingerprints(self, movie_id: int,
                                   fingerprints: list[tuple[float, int, np.ndarray]],
                                   silent: bool = False):
        phash_rows = [(movie_id, ts, hex(ph)) for ts, ph, _ in fingerprints]
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO visual_fingerprints (movie_id, timestamp_sec, phash) VALUES %s",
                phash_rows,
                page_size=5000,
            )

        embeddings = np.array([emb for _, _, emb in fingerprints], dtype=np.float32)
        metadata = [(movie_id, ts) for ts, _, _ in fingerprints]
        self.faiss.add(embeddings, metadata)

        if not silent:
            print(f"  Visual DB: {len(fingerprints)} keyframes stored")

    def store_scene_descriptors(self, movie_id: int,
                                 scenes: list[tuple[float, float, int, np.ndarray]],
                                 silent: bool = False):
        """
        Store scene-level descriptors.
        Each scene: (start_sec, end_sec, n_keyframes, mean_embedding_512d)
        """
        if not scenes:
            return

        pg_rows = [(movie_id, s, e, n) for s, e, n, _ in scenes]
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO scenes (movie_id, start_sec, end_sec, n_keyframes) VALUES %s",
                pg_rows,
            )

        embeddings = np.array([emb for _, _, _, emb in scenes], dtype=np.float32)
        # Normalize scene embeddings (mean of normalized vectors isn't necessarily normalized)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        metadata = [(movie_id, s, e) for s, e, _, _ in scenes]
        self.faiss_scenes.add(embeddings, metadata)

        if not silent:
            print(f"  Scene DB: {len(scenes)} scenes stored")

    def search_visual(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[int, float, float]]]:
        return self.faiss.search(query_embeddings, top_k)

    def search_scenes(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple]]:
        """Search scene index. Returns per-query list of (movie_id, start_sec, end_sec, similarity)."""
        return self.faiss_scenes.search(query_embeddings, top_k)

    def save(self):
        self.faiss.save()
        self.faiss_scenes.save()

    def close(self):
        self.conn.close()

    def reset_indexes(self):
        """Reset all FAISS indexes and clear all tables for clean re-indexing."""
        self.faiss.reset()
        self.faiss_scenes.reset()
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE visual_fingerprints CASCADE")
            cur.execute("TRUNCATE scenes CASCADE")
            cur.execute("TRUNCATE movies CASCADE")
        print("  All indexes and tables reset")


# Default database factory
def Database() -> DatabaseBackend:
    return PostgresDatabase()
