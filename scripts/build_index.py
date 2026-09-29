import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.db import resolve_db_path, fetch_all_nodes_with_breadcrumbs
from core.chunker import prepare_nodes_for_indexing
from core.embedder import encode_passages, DEFAULT_MODEL_NAME
from core.vector_store import VectorStore, DEFAULT_INDEX_PATH, DEFAULT_MAPPING_PATH


def main():
    print("=" * 60)
    print("ANRE Regulatory Rules: Vector Index Generator")
    print("=" * 60)

    # 1. Locate database
    db_path = resolve_db_path()
    if not db_path:
        print("❌ Error: Could not locate SQLite database (anre_rules_db.db).")
        sys.exit(1)

    print(f"📁 Database found: {db_path}")

    # 2. Extract nodes with Recursive CTE breadcrumb paths
    t0 = time.time()
    print("⏳ Traversing regulatory hierarchy with Recursive SQL CTE...")
    nodes = fetch_all_nodes_with_breadcrumbs(db_path)
    cte_time = time.time() - t0
    print(f"✅ Extracted {len(nodes)} regulation nodes with full breadcrumbs in {cte_time:.2f}s.")

    # 3. Format passages for embedding
    passages, node_ids = prepare_nodes_for_indexing(nodes)
    print(f"📄 Sample enriched chunk:\n   '{passages[0][:130]}...'")

    # 4. Generate embeddings
    print(f"\n🧠 Generating normalized embeddings using '{DEFAULT_MODEL_NAME}' on CPU/GPU...")
    t1 = time.time()
    embeddings = encode_passages(passages, model_name=DEFAULT_MODEL_NAME, batch_size=32)
    embed_time = time.time() - t1
    print(f"✅ Generated {embeddings.shape[0]} embeddings (dim: {embeddings.shape[1]}) in {embed_time:.2f}s.")
    print(f"⚡ Throughput: {len(passages) / embed_time:.1f} nodes/second.")

    # 5. Build and save index
    print("\n💾 Saving index and ID mapping...")
    index_file = os.path.join(PROJECT_ROOT, DEFAULT_INDEX_PATH)
    mapping_file = os.path.join(PROJECT_ROOT, DEFAULT_MAPPING_PATH)

    vstore = VectorStore(index_path=index_file, mapping_path=mapping_file)
    vstore.build_and_save(embeddings, node_ids)

    index_size_kb = os.path.getsize(index_file) / 1024
    print(f"🎉 Successfully built index:")
    print(f"   • Vector Index:  {index_file} ({index_size_kb:.1f} KB)")
    print(f"   • Node Mapping:  {mapping_file} ({len(node_ids)} entries)")
    print(f"\nTotal elapsed time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
