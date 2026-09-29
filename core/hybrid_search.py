from typing import Dict, List, Optional, Tuple
from core.db import (
    check_exact_citation,
    fetch_nodes_by_ids,
    search_nodes_lexical
)
from core.chunker import format_query_for_embedding
from core.embedder import encode_query
from core.vector_store import VectorStore
from core.normalizer import to_ascii

# Standard RRF smoothing constant (Cormack et al., 2009)
RRF_K = 60


def hybrid_search(
    db_path: str,
    vector_store: VectorStore,
    query: str,
    alpha: float = 0.6,
    top_k: int = 20,
    short_circuit_exact: bool = True
) -> List[Dict]:
    """
    Executes a hybrid search combining Dense Vector Similarity and Sparse SQL Lexical Search.
    
    Args:
        db_path: Path to the SQLite database.
        vector_store: Initialized and loaded VectorStore instance.
        query: User search query in Romanian.
        alpha: Weight for dense vector search [0.0 = pure SQL, 1.0 = pure Semantic].
        top_k: Number of final ranked results to return.
        short_circuit_exact: Whether to pin exact citation matches (e.g. 'Art. 7') at Rank 1.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # 1. Exact Citation Check (Short-circuiting)
    exact_match_node = None
    if short_circuit_exact:
        exact_match_node = check_exact_citation(db_path, clean_query)

    # 2. Dense Vector Search
    vector_ranks: Dict[int, int] = {}
    vector_scores: Dict[int, float] = {}

    if alpha > 0.0 and vector_store.is_indexed():
        try:
            formatted_query = format_query_for_embedding(clean_query)
            query_vector = encode_query(formatted_query)
            vec_results = vector_store.search(query_vector, top_k=top_k * 2)

            for rank_idx, (node_id, score) in enumerate(vec_results, start=1):
                vector_ranks[node_id] = rank_idx
                vector_scores[node_id] = score
        except Exception as e:
            # If embedding model cannot be loaded, fallback smoothly to keyword search
            print(f"[Warning] Dense search skipped: {e}")

    # 3. Sparse Lexical Search (SQLite CTE / LIKE)
    keyword_ranks: Dict[int, int] = {}
    if alpha < 1.0 or not vector_ranks:
        # Search both original query and ASCII normalized variant
        lex_results = search_nodes_lexical(db_path, clean_query, limit=top_k * 2)
        ascii_query = to_ascii(clean_query)
        if ascii_query != clean_query:
            lex_results_ascii = search_nodes_lexical(db_path, ascii_query, limit=top_k * 2)
            # Merge preserving order
            seen_ids = {r["id"] for r in lex_results}
            for r in lex_results_ascii:
                if r["id"] not in seen_ids:
                    lex_results.append(r)
                    seen_ids.add(r["id"])

        for rank_idx, item in enumerate(lex_results, start=1):
            keyword_ranks[item["id"]] = rank_idx

    # 4. Reciprocal Rank Fusion (RRF)
    all_candidate_ids = set(vector_ranks.keys()).union(set(keyword_ranks.keys()))

    if not all_candidate_ids and exact_match_node:
        all_candidate_ids.add(exact_match_node["id"])

    rrf_scores: Dict[int, float] = {}
    for node_id in all_candidate_ids:
        r_dense = vector_ranks.get(node_id, float("inf"))
        r_sparse = keyword_ranks.get(node_id, float("inf"))

        dense_term = (alpha / (RRF_K + r_dense)) if r_dense != float("inf") else 0.0
        sparse_term = ((1.0 - alpha) / (RRF_K + r_sparse)) if r_sparse != float("inf") else 0.0

        rrf_scores[node_id] = dense_term + sparse_term

    # Sort candidates by combined RRF score descending
    sorted_ids = sorted(all_candidate_ids, key=lambda nid: rrf_scores[nid], reverse=True)

    # 5. Hydrate Nodes with Breadcrumb Hierarchy & Metadata
    hydrated_lookup = fetch_nodes_by_ids(db_path, sorted_ids[:top_k * 2])

    final_results = []
    # If exact match exists, prioritize it at Rank 1
    exact_id = exact_match_node["id"] if exact_match_node else None

    if exact_id and exact_id in hydrated_lookup:
        exact_item = hydrated_lookup[exact_id]
        exact_item["is_exact_match"] = True
        exact_item["rrf_score"] = 1.0  # Max score badge
        exact_item["vector_score"] = vector_scores.get(exact_id, None)
        exact_item["keyword_match"] = True
        final_results.append(exact_item)

    for nid in sorted_ids:
        if nid == exact_id:
            continue
        if nid in hydrated_lookup:
            node = hydrated_lookup[nid]
            node["is_exact_match"] = False
            node["rrf_score"] = rrf_scores.get(nid, 0.0)
            node["vector_score"] = vector_scores.get(nid, None)
            node["keyword_match"] = nid in keyword_ranks
            final_results.append(node)
            if len(final_results) >= top_k:
                break

    return final_results
