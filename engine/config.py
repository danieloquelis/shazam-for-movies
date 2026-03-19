import os

# --- Visual fingerprinting ---
VISUAL_INDEX_FPS = 2            # Frames/sec when indexing
VISUAL_QUERY_FPS = 4            # Frames/sec when querying
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIM = 512
PHASH_SIZE = 8                  # 8x8 DCT -> 64-bit hash
FRAME_TARGET_SIZE = 224         # All frames normalized to this before CLIP

# --- Matching ---
OFFSET_BIN_WIDTH = 0.5          # Seconds for offset histogram binning
MIN_VISUAL_MATCHES = 4
OFFSET_STDDEV_THRESHOLD = 1.5   # Max stddev in winning cluster (sec)
CONFIDENCE_RATIO = 1.25         # Best must be >= this * second-best
VISUAL_TOP_K = 20               # Nearest neighbors per query frame

# --- Database ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "movie_fp")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fingerprint")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fingerprint")

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_visual.index")
FAISS_ID_MAP_PATH = os.path.join(DATA_DIR, "faiss_id_map.pkl")

# --- Frame processing ---
FRAME_BATCH_SIZE = 120          # Process this many frames at a time
