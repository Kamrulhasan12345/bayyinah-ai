import numpy as np
import faiss
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_PATH = os.getenv("EMBEDDINGS_CACHE_PATH", "data/verse_embeddings.npy")

_model = None
_faiss_index = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model

def _prepare_verse_texts(df) -> list[str]:
    texts = df['english'].fillna('').astype(str).str.strip().tolist()
    logger.info(f"Prepared {len(texts)} verse texts for embedding.")
    return texts

def load_or_compute_embeddings(df) -> np.ndarray:
    global _faiss_index

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        logger.info(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}")
        return np.load(EMBEDDINGS_CACHE_PATH)

    logger.info("No cached embeddings found. Computing embeddings...")
    model = get_model()
    texts = _prepare_verse_texts(df)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    logger.info(f"Embeddings computed and saved to {EMBEDDINGS_CACHE_PATH}")

    return embeddings

def get_faiss_index(df) -> faiss.Index:
    global _faiss_index

    if _faiss_index is not None:
        return _faiss_index

    embeddings = load_or_compute_embeddings(df)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    _faiss_index = index
    logger.info("FAISS index created and populated with embeddings.")
    return _faiss_index

def semantic_search(query: str, df, top_k: int = 50) -> list[tuple[int, float]]:
    model = get_model()
    faiss_idx = get_faiss_index(df)

    query_embedding = model.encode([query], convert_to_numpy=True)

    distances, indices = faiss_idx.search(query_embedding, k = min(top_k, len(df)))
    
    results = [(int(i), float(f)) for i, f in zip(indices[0], distances[0]) if i < len(df)]

    return results
