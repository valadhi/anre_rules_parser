from typing import List, Optional
import numpy as np

# Default lightweight model
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"

_MODEL_INSTANCE = None


def get_embedder_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Lazy singleton loader for the SentenceTransformer model.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL_INSTANCE = SentenceTransformer(model_name, device=device)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it via: pip install sentence-transformers"
            ) from e
    return _MODEL_INSTANCE


def encode_passages(
    passages: List[str], 
    model_name: str = DEFAULT_MODEL_NAME, 
    batch_size: int = 32,
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Generates normalized embedding vectors for a list of passages.
    Returns a 2D float32 numpy array of shape (len(passages), embedding_dim).
    """
    model = get_embedder_model(model_name)
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def encode_query(
    query_text: str, 
    model_name: str = DEFAULT_MODEL_NAME
) -> np.ndarray:
    """
    Encodes a single query string into a normalized 1D float32 embedding vector.
    """
    model = get_embedder_model(model_name)
    vector = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vector.astype(np.float32)
