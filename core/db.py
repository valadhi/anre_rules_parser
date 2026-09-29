import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from core.normalizer import to_ascii

DB_CANDIDATE_NAMES = [
    "anre_rules_db.db",
    "anre_rules_db.sqlite",
    "anre_rules_db",
    "anre_rules.db"
]

# Recursive CTE to build breadcrumbs for all nodes
ALL_NODES_WITH_BREADCRUMBS_QUERY = """
WITH RECURSIVE node_path_cte AS (
    -- Anchor: every node starts with its own identifier or type
    SELECT 
        id AS match_id,
        parent_id,
        node_type,
        node_identifier,
        title_or_text,
        COALESCE(NULLIF(node_identifier, ''), node_type) AS full_path
    FROM regulation_nodes

    UNION ALL

    -- Recursive: walk up parent_id to root
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
SELECT 
    match_id AS id,
    parent_id,
    node_type,
    node_identifier,
    title_or_text,
    full_path AS breadcrumb_path
FROM node_path_cte
WHERE parent_id IS NULL
ORDER BY match_id;
"""

# Recursive CTE to assemble breadcrumbs for specific node IDs
SPECIFIC_NODES_BREADCRUMB_QUERY = """
WITH RECURSIVE node_path_cte AS (
    SELECT 
        id AS match_id,
        parent_id,
        node_type,
        node_identifier,
        title_or_text,
        COALESCE(NULLIF(node_identifier, ''), node_type) AS full_path
    FROM regulation_nodes
    WHERE id IN ({placeholders})

    UNION ALL

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

# Recursive CTE for Lexical Search
LEXICAL_SEARCH_CTE_QUERY = """
WITH RECURSIVE node_path_cte AS (
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
SELECT 
    match_id AS id,
    node_type,
    node_identifier,
    title_or_text,
    full_path AS breadcrumb_path
FROM node_path_cte
WHERE parent_id IS NULL
ORDER BY match_id
LIMIT ?;
"""


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_db_path() -> Optional[str]:
    """Locate the SQLite database file in the workspace directory."""
    for filename in DB_CANDIDATE_NAMES:
        if os.path.exists(filename):
            return filename
        candidate = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a connection with sqlite3 and Row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all_nodes_with_breadcrumbs(db_path: str) -> List[Dict]:
    """
    Returns all regulatory nodes along with their complete reconstructed breadcrumbs.
    Used for precomputing embeddings offline.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(ALL_NODES_WITH_BREADCRUMBS_QUERY)
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


def fetch_nodes_by_ids(db_path: str, node_ids: List[int]) -> Dict[int, Dict]:
    """
    Fetch full breadcrumbs and content for a given list of node IDs using Recursive CTE.
    Returns a dictionary mapping node_id -> node_dict.
    """
    if not node_ids:
        return {}
    
    placeholders = ",".join(["?"] * len(node_ids))
    query = SPECIFIC_NODES_BREADCRUMB_QUERY.format(placeholders=placeholders)
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(query, node_ids)
    rows = cursor.fetchall()
    conn.close()

    lookup = {}
    for r in rows:
        lookup[r["id"]] = {
            "id": r["id"],
            "node_type": r["node_type"] or "",
            "node_identifier": r["node_identifier"] or "",
            "title_or_text": r["title_or_text"] or "",
            "breadcrumb_path": r["breadcrumb_path"] or ""
        }
    return lookup


def search_nodes_lexical(db_path: str, search_term: str, limit: int = 50) -> List[Dict]:
    """
    Executes the lexical search query using SQLite LIKE and recursive CTE.
    """
    clean_term = search_term.strip()
    if not clean_term:
        return []
    
    wildcard = f"%{clean_term}%"
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(LEXICAL_SEARCH_CTE_QUERY, (wildcard, wildcard, limit))
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


def check_exact_citation(db_path: str, query: str) -> Optional[Dict]:
    """
    Checks if the user query directly matches a legal article/document identifier.
    e.g., 'Art. 7', 'Art 7', 'ordin 123/2020', 'capitolul vi'.
    """
    cleaned = query.strip()
    if not cleaned:
        return None
    
    # Normalize common citation prefixes (e.g. 'art 7' -> 'art. 7')
    variations = [cleaned, to_ascii(cleaned)]
    if re.match(r"^art\s+\d+", cleaned, re.IGNORECASE):
        variations.append(re.sub(r"^art\s+", "Art. ", cleaned, flags=re.IGNORECASE))
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    for var in variations:
        cursor.execute("""
            SELECT id FROM regulation_nodes 
            WHERE node_identifier = ? COLLATE NOCASE
            LIMIT 1
        """, (var,))
        row = cursor.fetchone()
        if row:
            conn.close()
            # Fetch with full breadcrumb
            nodes = fetch_nodes_by_ids(db_path, [row["id"]])
            return nodes.get(row["id"])
    
    conn.close()
    return None


def fetch_article_context(db_path: str, node_id: int) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Finds the ancestor Article for a given node (or the node itself if it's an article),
    and retrieves all its child paragraphs and letters in order (Top-Down expansion).
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Walk up to find the closest ancestor ARTICLE (or DOCUMENT/CHAPTER if no article exists)
    FIND_ARTICLE_ANCESTOR_CTE = """
    WITH RECURSIVE ancestor_cte AS (
        SELECT id, parent_id, node_type, node_identifier, title_or_text
        FROM regulation_nodes
        WHERE id = ?

        UNION ALL

        SELECT p.id, p.parent_id, p.node_type, p.node_identifier, p.title_or_text
        FROM regulation_nodes p
        JOIN ancestor_cte a ON a.parent_id = p.id
    )
    SELECT id, parent_id, node_type, node_identifier, title_or_text
    FROM ancestor_cte
    ORDER BY CASE 
        WHEN node_type = 'ARTICLE' THEN 1 
        WHEN node_type = 'SECTION' THEN 2
        WHEN node_type = 'CHAPTER' THEN 3
        ELSE 4
    END
    LIMIT 1;
    """
    cursor.execute(FIND_ARTICLE_ANCESTOR_CTE, (node_id,))
    target_ancestor = cursor.fetchone()
    
    if not target_ancestor:
        conn.close()
        return None, []
    
    article_dict = {
        "id": target_ancestor["id"],
        "node_type": target_ancestor["node_type"],
        "node_identifier": target_ancestor["node_identifier"],
        "title_or_text": target_ancestor["title_or_text"]
    }
    
    # 2. Walk down to fetch all descendants of this article (Top-Down traversal)
    FETCH_DESCENDANTS_CTE = """
    WITH RECURSIVE descendant_cte AS (
        SELECT id, parent_id, node_type, node_identifier, title_or_text, 1 as depth
        FROM regulation_nodes
        WHERE parent_id = ?

        UNION ALL

        SELECT c.id, c.parent_id, c.node_type, c.node_identifier, c.title_or_text, d.depth + 1
        FROM regulation_nodes c
        JOIN descendant_cte d ON c.parent_id = d.id
    )
    SELECT id, parent_id, node_type, node_identifier, title_or_text, depth
    FROM descendant_cte
    ORDER BY id;
    """
    cursor.execute(FETCH_DESCENDANTS_CTE, (article_dict["id"],))
    descendants = [
        {
            "id": r["id"],
            "parent_id": r["parent_id"],
            "node_type": r["node_type"],
            "node_identifier": r["node_identifier"],
            "title_or_text": r["title_or_text"],
            "depth": r["depth"]
        }
        for r in cursor.fetchall()
    ]
    conn.close()
    return article_dict, descendants
