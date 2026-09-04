import os
import re
import sqlite3
from typing import Dict, List, Optional
import streamlit as st

# ==============================================================================
# SQL RECURSIVE COMMON TABLE EXPRESSION (CTE) QUERY
# ==============================================================================
# The SQL query below accomplishes two key tasks:
# 1. Matches nodes whose 'node_identifier' OR 'title_or_text' contains the search term.
# 2. Uses a Recursive CTE (Bottom-Up traversal) to walk up the 'parent_id' hierarchy
#    to the root node (where parent_id IS NULL), prepending ancestor labels to
#    reconstruct the full breadcrumb path (e.g. "ORDIN > CAPITOLUL VI > SECŢIUNEA 1 > Art. 7 > (1)").
#
# Raw SQL Definition:
# ------------------------------------------------------------------------------
# WITH RECURSIVE node_path_cte AS (
#     -- Anchor Member: Select initial matching nodes
#     SELECT 
#         id AS match_id,
#         parent_id,
#         node_type,
#         node_identifier,
#         title_or_text,
#         COALESCE(NULLIF(node_identifier, ''), node_type) AS full_path
#     FROM regulation_nodes
#     WHERE node_identifier LIKE ? COLLATE NOCASE
#        OR title_or_text LIKE ? COLLATE NOCASE
#
#     UNION ALL
#
#     -- Recursive Member: Traverse up the parent_id chain to the root
#     SELECT 
#         c.match_id,
#         p.parent_id,
#         c.node_type,
#         c.node_identifier,
#         c.title_or_text,
#         COALESCE(NULLIF(p.node_identifier, ''), p.node_type) || ' > ' || c.full_path AS full_path
#     FROM node_path_cte c
#     JOIN regulation_nodes p ON c.parent_id = p.id
# )
# -- Final Selection: Pick records where recursion reached the root ancestor (parent_id IS NULL)
# SELECT 
#     match_id AS id,
#     node_type,
#     node_identifier,
#     title_or_text,
#     full_path AS breadcrumb_path
# FROM node_path_cte
# WHERE parent_id IS NULL
# ORDER BY match_id;
# ==============================================================================

RECURSIVE_CTE_SEARCH_QUERY = """
WITH RECURSIVE node_path_cte AS (
    -- 1. Anchor Member: Find all nodes directly matching search criteria
    SELECT 
        id AS match_id,
        parent_id,
        node_type,
        node_identifier,
        title_or_text,
        COALESCE(NULLIF(node_identifier, ''), node_type) AS full_path
    FROM regulation_nodes
    WHERE node_identifier LIKE ? COLLATE NOCASE
       OR title_or_text LIKE ? COLLATE NOCASE

    UNION ALL

    -- 2. Recursive Member: Ascend the parent_id hierarchy until reaching the root
    SELECT 
        c.match_id,
        p.parent_id,
        c.node_type,
        c.node_identifier,
        c.title_or_text,
        COALESCE(NULLIF(p.node_identifier, ''), p.node_type) || ' > ' || c.full_path AS full_path
    FROM node_path_cte c
    JOIN regulation_nodes p ON c.parent_id = p.id
)
-- 3. Final Query: Keep only the fully assembled paths (root reached when parent_id IS NULL)
SELECT 
    match_id AS id,
    node_type,
    node_identifier,
    title_or_text,
    full_path AS breadcrumb_path
FROM node_path_cte
WHERE parent_id IS NULL
ORDER BY match_id;
"""

# Supported database file names in order of preference
DB_CANDIDATE_NAMES = [
    "anre_rules_db",
    "anre_rules_db.sqlite",
    "anre_rules_db.db",
    "anre_rules.db"
]


def resolve_db_path() -> Optional[str]:
    """Locate the SQLite database file in the workspace directory."""
    for filename in DB_CANDIDATE_NAMES:
        if os.path.exists(filename):
            return filename
    return None


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a connection with sqlite3."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_sample_database(db_path: str = "anre_rules_db") -> None:
    """Initialize a demo database populated with hierarchical ANRE regulation nodes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS regulation_nodes (
        id INTEGER PRIMARY KEY,
        parent_id INTEGER,
        node_type VARCHAR(50),
        node_identifier VARCHAR(100),
        title_or_text TEXT,
        FOREIGN KEY (parent_id) REFERENCES regulation_nodes(id)
    );
    """)

    # Sample hierarchical regulatory data based on Romanian ANRE regulations
    sample_nodes = [
        (1, None, "ORDIN", "ORDIN 123/2020", "Ordin pentru aprobarea Regulamentului de racordare a utilizatorilor la rețelele electrice de interes public"),
        (2, 1, "CAPITOL", "CAPITOLUL I", "Dispoziții generale"),
        (3, 2, "SECTIUNE", "SECŢIUNEA 1", "Scop și domeniu de aplicare"),
        (4, 3, "ARTICOL", "Art. 1", "Prezentul regulament stabilește etapele și procedurile pentru racordarea instalațiilor utilizatorilor la rețeaua electrică de interes public."),
        (5, 3, "ARTICOL", "Art. 2", "Prevederile se aplică operatorilor de rețea, solicitanților și utilizatorilor de rețea electrică."),
        (6, 1, "CAPITOL", "CAPITOLUL VI", "Racordarea utilizatorilor la rețele electrice"),
        (7, 6, "SECTIUNE", "SECŢIUNEA 1", "Procedura de racordare și emiterea avizului tehnic"),
        (8, 7, "ARTICOL", "Art. 7", "Condiții tehnice generale pentru avizul tehnic de racordare (ATR)"),
        (9, 8, "ALINEAT", "(1)", "Operatorii de rețea au obligația să asigure accesul nediscriminatoriu la rețeaua electrică pentru toți solicitanții."),
        (10, 8, "ALINEAT", "(2)", "Tarifele de racordare se stabilesc în mod transparent, pe baza metodologiei aprobate de către autoritatea competentă ANRE."),
        (11, 8, "ALINEAT", "(3)", "Termenul maxim pentru transmiterea ofertei de racordare este de 30 de zile calendaristice de la depunerea documentației complete."),
        (12, 6, "SECTIUNE", "SECŢIUNEA 2", "Contractul de racordare"),
        (13, 12, "ARTICOL", "Art. 8", "Drepturile și obligațiile părților"),
        (14, 13, "ALINEAT", "(1)", "Utilizatorul are dreptul de a solicita modificarea soluției de racordare în condițiile legii."),
        (15, 1, "CAPITOL", "CAPITOLUL VII", "Dispoziții tranzitorii și finale"),
        (16, 15, "ARTICOL", "Art. 9", "Intrarea în vigoare"),
        (17, 16, "ALINEAT", "(1)", "Prezentul ordin se publică în Monitorul Oficial al României, Partea I, și intră în vigoare la data publicării.")
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO regulation_nodes (id, parent_id, node_type, node_identifier, title_or_text) VALUES (?, ?, ?, ?, ?)",
        sample_nodes
    )
    conn.commit()
    conn.close()


def search_regulation_nodes(db_path: str, search_term: str) -> List[Dict]:
    """
    Executes the Recursive CTE query to search nodes and reconstruct their full hierarchy breadcrumb path.
    """
    wildcard_term = f"%{search_term.strip()}%"
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute(RECURSIVE_CTE_SEARCH_QUERY, (wildcard_term, wildcard_term))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "node_type": r["node_type"] or "",
            "node_identifier": r["node_identifier"] or "",
            "title_or_text": r["title_or_text"] or "",
            "breadcrumb_path": r["breadcrumb_path"] or ""
        })
    return results


def highlight_match(text: str, query: str) -> str:
    """Highlight matched query occurrences with markdown/HTML mark."""
    if not query.strip() or not text:
        return text
    escaped_query = re.escape(query.strip())
    pattern = re.compile(f"({escaped_query})", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


# ==============================================================================
# STREAMLIT UI IMPLEMENTATION
# ==============================================================================
def main():
    st.set_page_config(
        page_title="ANRE Regulations Search",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("📜 ANRE Regulatory Search & Hierarchy Explorer")
    st.markdown(
        "Search regulatory articles, paragraphs, and clauses from the adjacency list database. "
        "Each result reconstructs the complete hierarchical document path via **Recursive SQL CTE**."
    )

    # Database file check
    db_path = resolve_db_path()

    if not db_path:
        st.warning("⚠️ Local SQLite database (`anre_rules_db`) not found in the current directory.")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✨ Initialize Demo Database", type="primary"):
                create_sample_database("anre_rules_db")
                st.success("Sample database `anre_rules_db` initialized successfully!")
                st.rerun()
        st.stop()

    # Search Bar (Persistent input field at the top)
    st.markdown("### 🔍 Search Regulations")
    query = st.text_input(
        label="Search by keywords, concepts, or article identifiers (e.g. 'Art. 7', 'racordare', 'nediscriminatoriu', '(1)'):",
        placeholder="Enter search terms here...",
        key="search_query"
    )

    # Search Execution
    if query and query.strip():
        try:
            results = search_regulation_nodes(db_path, query)
            
            # Status / metric header
            st.markdown(f"**Found {len(results)} matching node{'s' if len(results) != 1 else ''}** for `'{query}'`:")

            if not results:
                st.info("No matching regulation nodes found. Try searching for a different keyword or article number.")
            else:
                for idx, item in enumerate(results):
                    # Breadcrumb path as the expander header
                    expander_title = f"{item['breadcrumb_path']}"
                    
                    # Open the first few expanders by default for immediate preview
                    with st.expander(expander_title, expanded=(idx < 3)):
                        # Node metadata badges
                        meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 4])
                        with meta_col1:
                            st.caption(f"**Node ID:** {item['id']}")
                        with meta_col2:
                            st.caption(f"**Type:** `{item['node_type']}`")
                        with meta_col3:
                            st.caption(f"**Identifier:** `{item['node_identifier']}`")

                        st.markdown("---")
                        
                        # Matched content
                        highlighted_text = highlight_match(item["title_or_text"], query)
                        st.markdown(highlighted_text)

        except sqlite3.OperationalError as e:
            st.error(f"Database query error: {e}")
            st.info("Ensure the `regulation_nodes` table exists with columns: `id`, `parent_id`, `node_type`, `node_identifier`, `title_or_text`.")
    else:
        st.info("💡 Type in the search box above to query the regulatory database.")
        
        # Quick sample suggestions
        st.markdown("**Example queries to try:**")
        sample_queries = ["nediscriminatoriu", "Art. 7", "racordare", "aviz", "(1)"]
        cols = st.columns(len(sample_queries))
        for i, sample in enumerate(sample_queries):
            if cols[i].button(sample, key=f"sample_{i}"):
                st.session_state["search_query"] = sample
                st.rerun()

    # Footer showing CTE details
    with st.expander("🛠️ View SQL Recursive CTE Details"):
        st.markdown("This query utilizes a **Recursive Common Table Expression** in SQLite to climb from any matching node up to the root:")
        st.code(RECURSIVE_CTE_SEARCH_QUERY, language="sql")


if __name__ == "__main__":
    main()
