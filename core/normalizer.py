import re
import unicodedata

# Explicit mapping for Romanian diacritics (both standard comma and legacy cedilla forms)
ROMANIAN_DIACRITIC_MAP = {
    "ș": "s", "Ș": "S",
    "ț": "t", "Ț": "T",
    "ş": "s", "Ş": "S",
    "ţ": "t", "Ţ": "T",
    "ă": "a", "Ă": "A",
    "â": "a", "Â": "A",
    "î": "i", "Î": "I",
}


def to_ascii(text: str) -> str:
    """
    Normalizes Romanian text to pure ASCII by stripping accents and diacritics.
    
    1. Replaces Romanian specific characters (ș, ț, ă, â, î and cedilla variants).
    2. Decomposes remaining characters via NFKD and removes combining marks.
    3. Normalizes repeated whitespace.
    """
    if not text:
        return ""
    
    # 1. Map explicit Romanian letters first to guarantee exact phonetic conversion
    for char, replacement in ROMANIAN_DIACRITIC_MAP.items():
        text = text.replace(char, replacement)
    
    # 2. General unicode decomposition fallback
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_chars = [c for c in decomposed if not unicodedata.combining(c)]
    cleaned = "".join(ascii_chars)
    
    # 3. Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
