from typing import Dict, List, Tuple
from core.normalizer import to_ascii


def format_chunk_for_embedding(node: Dict) -> str:
    """
    Constructs a hierarchy-enriched textual passage for a regulation node.
    Format:
        passage: [Full Breadcrumb Path] > [Identifier]: [Text]
    Both the breadcrumb path and content are normalized to ASCII to ensure
    robust vector similarity regardless of Romanian diacritic presence in user queries.
    """
    breadcrumb = to_ascii(node.get("breadcrumb_path", ""))
    identifier = to_ascii(node.get("node_identifier", ""))
    content = to_ascii(node.get("title_or_text", ""))

    # If identifier is already at the end of the breadcrumb, avoid duplicating it
    if identifier and not breadcrumb.endswith(identifier):
        prefix = f"{breadcrumb} > {identifier}"
    else:
        prefix = breadcrumb

    if content:
        raw_text = f"{prefix}: {content}" if prefix else content
    else:
        raw_text = prefix

    # E5 models require 'passage: ' prefix for document indexing
    return f"passage: {raw_text.strip()}"


def format_query_for_embedding(query: str) -> str:
    """
    Prepares a user query for embedding:
    - Normalizes to ASCII (removing diacritics)
    - Prepends the 'query: ' prefix required by E5 models
    """
    clean_query = to_ascii(query)
    return f"query: {clean_query}"


def prepare_nodes_for_indexing(nodes: List[Dict]) -> Tuple[List[str], List[int]]:
    """
    Takes a list of node dictionaries (with breadcrumb paths) and returns:
    1. A list of formatted passage strings ready for embedding.
    2. A corresponding list of integer node IDs.
    """
    passages = []
    node_ids = []

    for node in nodes:
        passage = format_chunk_for_embedding(node)
        passages.append(passage)
        node_ids.append(node["id"])

    return passages, node_ids
