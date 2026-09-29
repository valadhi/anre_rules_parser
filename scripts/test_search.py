import os
import sys

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.db import resolve_db_path, fetch_article_context
from core.vector_store import VectorStore
from core.hybrid_search import hybrid_search


def run_tests():
    db_path = resolve_db_path()
    store = VectorStore()
    if not store.load():
        print("❌ Could not load vector store.")
        return

    print(f"✅ Vector store loaded: {len(store.mapping)} vectors.")

    # 1. Exact Citation Test
    print("\n--- TEST 1: Exact Citation: 'Art. 7' ---")
    results_exact = hybrid_search(db_path, store, "Art. 7", alpha=0.6, top_k=3)
    for r in results_exact:
        print(f"  • [Score: {r.get('rrf_score', 0):.4f}] Exact={r.get('is_exact_match')} | ID={r['id']} | Path: {r['breadcrumb_path'][:90]}")

    # 2. Semantic Query Test
    print("\n--- TEST 2: Semantic / Conceptual: 'In cat timp se transmite oferta de racordare?' ---")
    q_sem = "In cat timp se transmite oferta de racordare?"
    results_sem = hybrid_search(db_path, store, q_sem, alpha=0.6, top_k=3)
    for r in results_sem:
        sim = f"{int(r['vector_score'] * 100)}%" if r.get("vector_score") is not None else "N/A"
        print(f"  • [Score: {r.get('rrf_score', 0):.4f}] Sim={sim} | ID={r['id']} | Path: {r['breadcrumb_path'][:70]}")
        print(f"    Text: {r['title_or_text'][:90]}...")

    # 3. Top-Down Article Context Test
    print("\n--- TEST 3: Top-Down Article Context Expansion ---")
    if results_sem:
        top_id = results_sem[0]["id"]
        art, desc = fetch_article_context(db_path, top_id)
        if art:
            print(f"  • Parent Article: {art['node_identifier']} — {art['title_or_text'][:60]}...")
            print(f"  • Found {len(desc)} child clauses/paragraphs under this article.")
            for c in desc[:3]:
                print(f"    - {c['node_identifier']}: {c['title_or_text'][:70]}...")


if __name__ == "__main__":
    run_tests()
