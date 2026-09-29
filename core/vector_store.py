import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_PATH = os.path.join(PROJECT_DIR, "anre_rules.index")
DEFAULT_MAPPING_PATH = os.path.join(PROJECT_DIR, "anre_mapping.json")


class VectorStore:
    """
    Manages embedding vector indexing and nearest-neighbor search.
    Uses FAISS if available; otherwise falls back transparently to NumPy dot-product.
    """

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, mapping_path: str = DEFAULT_MAPPING_PATH):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.mapping: List[int] = []  # Maps row index -> SQLite node_id
        self.faiss_index = None
        self.numpy_matrix: Optional[np.ndarray] = None
        self.use_faiss = False

    def is_indexed(self) -> bool:
        """Returns True if the index and mapping files exist on disk."""
        return os.path.exists(self.index_path) and os.path.exists(self.mapping_path)

    def build_and_save(self, embeddings: np.ndarray, node_ids: List[int]) -> None:
        """
        Builds a vector index from normalized embeddings and saves it along with the node mapping.
        """
        self.mapping = list(node_ids)
        num_vectors, dim = embeddings.shape

        # Attempt to use FAISS
        try:
            import faiss
            index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine Similarity since vectors are normalized
            index.add(embeddings)
            faiss.write_index(index, self.index_path)
            self.faiss_index = index
            self.use_faiss = True
        except ImportError:
            # Fallback: Save as numpy matrix
            np.save(self.index_path, embeddings)
            self.numpy_matrix = embeddings
            self.use_faiss = False

        # Save mapping as JSON
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f)

    def load(self) -> bool:
        """
        Loads the index and mapping from disk into memory.
        """
        if not self.is_indexed():
            return False

        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        try:
            import faiss
            self.faiss_index = faiss.read_index(self.index_path)
            self.use_faiss = True
        except Exception:
            # Fallback to NumPy
            if self.index_path.endswith(".npy"):
                self.numpy_matrix = np.load(self.index_path)
            else:
                # If faiss wasn't available to read the .index binary, try numpy fallback if available
                return False
            self.use_faiss = False

        return True

    def search(self, query_vector: np.ndarray, top_k: int = 25) -> List[Tuple[int, float]]:
        """
        Searches the nearest neighbors for a normalized query vector.
        Returns a list of (node_id, cosine_similarity_score) sorted by relevance descending.
        """
        if not self.mapping:
            if not self.load():
                return []

        top_k = min(top_k, len(self.mapping))
        query_2d = query_vector.reshape(1, -1).astype(np.float32)

        if self.use_faiss and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(query_2d, top_k)
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.mapping) and idx >= 0:
                    results.append((self.mapping[idx], float(score)))
            return results
        
        elif self.numpy_matrix is not None:
            # Pure NumPy dot-product fallback
            scores = np.dot(self.numpy_matrix, query_vector)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(self.mapping[idx], float(scores[idx])) for idx in top_indices]
            return results

        return []
