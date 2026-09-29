import os
import re
import sqlite3
import time
from typing import Dict, List, Optional
import streamlit as st

from core.db import (
    resolve_db_path,
    get_db_connection,
    fetch_article_context
)
from core.vector_store import VectorStore, DEFAULT_INDEX_PATH, DEFAULT_MAPPING_PATH
from core.hybrid_search import hybrid_search
from core.normalizer import to_ascii

# ==============================================================================
# STREAMLIT RESOURCE CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_vector_store() -> VectorStore:
    """Loads the precomputed vector index into memory once across sessions."""
    store = VectorStore(index_path=DEFAULT_INDEX_PATH, mapping_path=DEFAULT_MAPPING_PATH)
    store.load()
    return store


def highlight_match(text: str, query: str) -> str:
    """Highlight matched query occurrences with markdown bold."""
    if not query.strip() or not text:
        return text
    # Strip diacritics for highlighting matching if needed, or match basic tokens
    tokens = [re.escape(t) for t in query.strip().split() if len(t) > 2]
    if not tokens:
        tokens = [re.escape(query.strip())]
    pattern = re.compile(f"({'|'.join(tokens)})", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


def render_full_article_modal(db_path: str, node_id: int, current_node_identifier: str):
    """Renders the parent article and all its child paragraphs in sequential hierarchy."""
    article_dict, descendants = fetch_article_context(db_path, node_id)
    if not article_dict:
        st.warning("Could not locate parent article container.")
        return

    st.markdown(f"#### 📜 `{article_dict['node_identifier']}` — {article_dict['title_or_text']}")
    st.markdown("---")

    if not descendants:
        st.info("This node has no additional child clauses.")
        return

    for child in descendants:
        indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * (child["depth"] - 1)
        is_current = (child["id"] == node_id)
        
        # Highlight currently retrieved item within the article flow
        border_style = "border-left: 3px solid #ff4b4b; padding-left: 8px; background-color: rgba(255, 75, 75, 0.05);" if is_current else ""
        badge = "👈 *(Matched)*" if is_current else ""
        
        st.markdown(
            f"<div style='margin-bottom: 6px; {border_style}'>"
            f"{indent}<strong>{child['node_identifier']}</strong>: {child['title_or_text']} {badge}"
            f"</div>",
            unsafe_allow_html=True
        )


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    st.set_page_config(
        page_title="ANRE Regulations Hybrid Explorer",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 1. Database & Vector Store Verification
    db_path = resolve_db_path()
    vector_store = load_vector_store()
    is_vector_ready = vector_store.is_indexed()

    # Sidebar: Index Status & Controls
    with st.sidebar:
        st.markdown("### ⚙️ Search Configuration")
        
        if is_vector_ready:
            st.success(f"⚡ Vector Index Active ({len(vector_store.mapping)} nodes)")
        else:
            st.warning("⚠️ Vector index not found.\nRunning in **Keyword / CTE Only** mode.")
            st.caption("Generate embeddings locally by running:\n`python scripts/build_index.py`")

        st.markdown("---")
        st.markdown("#### ⚖️ Search Balance (α)")
        
        alpha = st.slider(
            label="Adjust Semantic vs. Keyword weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help=(
                "0.0 = Pure SQL CTE (Exact keywords/numbers)\n"
                "0.6 = Balanced Hybrid (Recommended)\n"
                "1.0 = Pure AI Semantic (Concepts & meaning)"
            ),
            disabled=not is_vector_ready
        )

        if alpha == 0.0:
            st.caption("🔍 **Mode: Pure Exact Keyword (SQL CTE)**")
        elif alpha == 1.0:
            st.caption("🧠 **Mode: Pure AI Semantic (Dense Vectors)**")
        else:
            pct_sem = int(alpha * 100)
            pct_key = int((1.0 - alpha) * 100)
            st.caption(f"⚖️ **Hybrid Mode:** {pct_sem}% Semantic + {pct_key}% Keyword")

        short_circuit = st.checkbox(
            "Pin Exact Legal Citations (e.g. 'Art. 7')",
            value=True,
            help="If checked, queries matching an exact Article or Document identifier are automatically promoted to Rank 1."
        )

        top_k = st.slider("Max Results:", min_value=5, max_value=50, value=20, step=5)

        st.markdown("---")
        st.caption("Built for **ANRE** Regulatory Directives & Grid Connection Rules.")

    # Header Section
    st.title("📜 ANRE Regulatory Search & Hierarchy Explorer")
    st.markdown(
        "Search energy regulations, grid connection laws, and official directives. "
        "Powered by **Hybrid Search (Dense Multilingual Vectors + Recursive SQL CTE)**."
    )

    if not db_path:
        st.error("⚠️ Local SQLite database (`anre_rules_db.db`) not found in workspace.")
        st.stop()

    # Search Bar
    st.markdown("### 🔍 Search Regulations")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="Search by natural questions, concepts, or article citations:",
            placeholder="e.g. 'In cat timp primesc oferta de racordare?', 'Art. 7', 'obligatii operator retea'...",
            key="search_query"
        )
    with col_btn:
        st.write("")
        st.write("")
        do_search = st.button("🔎 Search", type="primary", use_container_width=True)

    # Search Execution
    if query and query.strip():
        t0 = time.time()
        with st.spinner("Searching regulatory hierarchy..."):
            results = hybrid_search(
                db_path=db_path,
                vector_store=vector_store,
                query=query,
                alpha=alpha if is_vector_ready else 0.0,
                top_k=top_k,
                short_circuit_exact=short_circuit
            )
        elapsed_ms = (time.time() - t0) * 1000

        # Results Metric Banner
        st.markdown(
            f"**Found {len(results)} matching node{'s' if len(results) != 1 else ''}** "
            f"for `'{query}'` *(retrieval time: {elapsed_ms:.1f} ms)*:"
        )

        if not results:
            st.info("No matching regulation nodes found. Try rephrasing your search or adjusting the balance slider.")
        else:
            for idx, item in enumerate(results):
                is_exact = item.get("is_exact_match", False)
                vec_score = item.get("vector_score")
                kw_match = item.get("keyword_match", False)

                # Card Header
                badge_prefix = "🎯 [EXACT CITATION MATCH] " if is_exact else ""
                expander_title = f"{badge_prefix}{item['breadcrumb_path']}"

                with st.expander(expander_title, expanded=(idx < 3)):
                    # Badges Row
                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns([2, 2, 2, 4])
                    with meta_col1:
                        st.caption(f"**Node ID:** `{item['id']}`")
                    with meta_col2:
                        st.caption(f"**Type:** `{item['node_type']}`")
                    with meta_col3:
                        st.caption(f"**Identifier:** `{item['node_identifier']}`")
                    with meta_col4:
                        badges = []
                        if is_exact:
                            badges.append("🎯 **Exact Match**")
                        if vec_score is not None:
                            similarity_pct = int(max(0.0, min(1.0, vec_score)) * 100)
                            badges.append(f"🧠 Sim: `{similarity_pct}%`")
                        if kw_match:
                            badges.append("🔤 Keyword Hit")
                        st.caption(" • ".join(badges) if badges else "")

                    st.markdown("---")

                    # Highlighted Content
                    highlighted_text = highlight_match(item["title_or_text"], query)
                    st.markdown(highlighted_text)

                    # Top-down Article Context Expansion
                    with st.expander("📖 Expand Full Article Context (Sibling Clauses)", expanded=False):
                        render_full_article_modal(db_path, item["id"], item["node_identifier"])

    else:
        st.info("💡 Type a query above to search or click one of the sample queries below.")

        st.markdown("**Sample queries to test Hybrid vs. Exact search:**")
        sample_queries = [
            "In cat timp se transmite oferta de racordare?",
            "Art. 7",
            "obligatii operator retea",
            "aviz tehnic de racordare",
            "ORDIN 123/2020",
            "tarife de racordare transparente"
        ]
        cols = st.columns(len(sample_queries))
        for i, sample in enumerate(sample_queries):
            if cols[i].button(sample, key=f"sample_{i}"):
                st.session_state["search_query"] = sample
                st.rerun()


if __name__ == "__main__":
    main()
