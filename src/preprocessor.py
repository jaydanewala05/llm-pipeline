"""
preprocessor.py — Clean raw text and split into LLM-sized chunks.

Cleaning steps:
  1. Fix common encoding artefacts (smart quotes, em-dashes, BOM, etc.)
  2. Collapse excessive whitespace / blank lines
  3. Remove obvious boilerplate patterns (cookie banners, nav menus…)

Chunking:
  Uses tiktoken (cl100k_base) for accurate token counts.
  Falls back to a simple word-based estimate if tiktoken is unavailable.
"""

import logging
import re
import unicodedata
from typing import Optional


# ── ENCODING FIXES ────────────────────────────────────────────────────────────

_REPLACEMENTS = [
    ("\u2018", "'"), ("\u2019", "'"),   # curly single quotes
    ("\u201c", '"'), ("\u201d", '"'),   # curly double quotes
    ("\u2013", "-"), ("\u2014", "--"),  # en/em dash
    ("\u00a0", " "),                    # non-breaking space
    ("\u200b", ""),                     # zero-width space
    ("\ufeff", ""),                     # BOM
    ("\r\n", "\n"), ("\r", "\n"),       # Windows / old-Mac line endings
]

_BOILERPLATE_PATTERNS = [
    r"accept(ing)? (all )?cookies.*",
    r"subscribe to (our )?newsletter.*",
    r"sign up for (our )?(free )?newsletter.*",
    r"©\s*\d{4}.*all rights reserved.*",
    r"privacy policy\s*[\|·]\s*terms.*",
    r"share (this )?(article|post|page).*",
    r"follow us on.*",
    r"advertisement",
    r"skip to (main )?(content|navigation).*",
    r"read (also|more)[:\-].*",
]

_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


def _clean_text(text: str) -> str:
    # Step 1: Unicode normalise to NFC
    text = unicodedata.normalize("NFC", text)

    # Step 2: Character replacements
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)

    # Step 3: Remove boilerplate lines
    lines = text.split("\n")
    clean_lines = [ln for ln in lines if not _BOILERPLATE_RE.search(ln.strip())]

    # Step 4: Collapse 3+ consecutive blank lines → 2
    text = "\n".join(clean_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 5: Trim each line
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))

    return text.strip()


# ── TOKEN COUNTING ────────────────────────────────────────────────────────────

def _get_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(enc.encode(t))
    except ImportError:
        # Rough fallback: ~0.75 tokens per word
        return lambda t: int(len(t.split()) * 1.33)


# ── CHUNKING ─────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_tokens: int, count_tokens) -> list[str]:
    """
    Splits text into chunks that fit within max_tokens.
    Tries to split on paragraph boundaries first, then sentence boundaries,
    then hard-truncates as a last resort.
    """
    # Split on paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            # Para is bigger than the entire window — split by sentences
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_tokens = [], 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > max_tokens:
                    if current_parts:
                        chunks.append(" ".join(current_parts))
                    current_parts = [sentence]
                    current_tokens = sent_tokens
                else:
                    current_parts.append(sentence)
                    current_tokens += sent_tokens

            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts, current_tokens = [], 0

        elif current_tokens + para_tokens > max_tokens:
            # Flush current window and start new one
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens

        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Filter empty
    return [c for c in chunks if c.strip()]


# ── PUBLIC INTERFACE ──────────────────────────────────────────────────────────

def preprocess_documents(
    documents: list[dict],
    max_tokens: int = 800,
    logger: Optional[logging.Logger] = None,
) -> list[dict]:
    """
    Clean and chunk every document.

    Returns a list of chunk dicts:
      {
        "source":      str,
        "source_type": str,
        "chunk_index": int,   # 0-based within this document
        "total_chunks": int,
        "text":        str,
      }
    """
    count_tokens = _get_token_counter()
    all_chunks = []

    for doc in documents:
        source = doc["source"]
        raw = doc["raw_text"]

        cleaned = _clean_text(raw)
        if logger:
            logger.debug(
                f"Cleaned '{source}': {len(raw):,} → {len(cleaned):,} chars"
            )

        chunks = _split_into_chunks(cleaned, max_tokens, count_tokens)

        if not chunks:
            if logger:
                logger.warning(f"No usable text chunks from '{source}' — skipping.")
            continue

        total = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "source":       source,
                "source_type":  doc["source_type"],
                "chunk_index":  idx,
                "total_chunks": total,
                "text":         chunk_text,
            })

        if logger:
            logger.info(
                f"'{source}' → {total} chunk(s) "
                f"(~{count_tokens(cleaned):,} tokens cleaned)"
            )

    return all_chunks
