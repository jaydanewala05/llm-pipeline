"""
ingestion.py — Ingest raw text from .txt/.pdf files and URLs.

Each returned document is a dict:
  {
    "source": str,          # filename or URL
    "source_type": str,     # "file_txt" | "file_pdf" | "url"
    "raw_text": str,        # full extracted text
  }

Bad inputs are logged and skipped; the function never raises.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


# ── FILE INGESTION ────────────────────────────────────────────────────────────

def ingest_file(file_path: str, logger: logging.Logger) -> Optional[dict]:
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _read_txt(path, logger)
    elif suffix == ".pdf":
        return _read_pdf(path, logger)
    else:
        logger.error(f"Unsupported file type '{suffix}' for {file_path}. Only .txt and .pdf accepted.")
        return None


def _read_txt(path: Path, logger: logging.Logger) -> Optional[dict]:
    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"{path.name}: UTF-8 decode failed — retrying with latin-1.")
            text = path.read_text(encoding="latin-1")

        logger.info(f"TXT ingested: {path.name} ({len(text):,} chars)")
        return {"source": str(path), "source_type": "file_txt", "raw_text": text}

    except Exception as exc:
        logger.error(f"Failed to read TXT {path}: {exc}", exc_info=True)
        return None


def _read_pdf(path: Path, logger: logging.Logger) -> Optional[dict]:
    try:
        import pypdf  # imported lazily so txt-only runs don't need it

        reader = pypdf.PdfReader(str(path))
        pages = []
        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            except Exception as page_exc:
                logger.warning(f"PDF {path.name} — page {page_num} failed: {page_exc}")

        text = "\n".join(pages)
        if not text.strip():
            logger.warning(f"PDF {path.name} yielded no extractable text (possibly scanned).")
            return None

        logger.info(f"PDF ingested: {path.name} ({len(reader.pages)} pages, {len(text):,} chars)")
        return {"source": str(path), "source_type": "file_pdf", "raw_text": text}

    except ImportError:
        logger.error("pypdf is not installed. Run: pip install pypdf")
        return None
    except Exception as exc:
        logger.error(f"Failed to read PDF {path}: {exc}", exc_info=True)
        return None


# ── URL INGESTION ─────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LLMPipeline/1.0; "
        "+https://github.com/your-org/llm-pipeline)"
    )
}
_TIMEOUT = 15   # seconds
_MAX_RETRIES = 3


def ingest_urls(urls: list[str], logger: logging.Logger) -> list[dict]:
    docs = []
    for url in urls:
        doc = _fetch_url(url, logger)
        if doc:
            docs.append(doc)
        else:
            logger.warning(f"Skipping URL: {url}")
    return docs


def _fetch_url(url: str, logger: logging.Logger) -> Optional[dict]:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type:
                text = _parse_html(resp.text, url, logger)
            else:
                text = resp.text  # plain text / JSON / etc.

            if not text.strip():
                logger.warning(f"URL {url} returned empty content.")
                return None

            logger.info(f"URL ingested: {url} ({len(text):,} chars)")
            return {"source": url, "source_type": "url", "raw_text": text}

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url} (attempt {attempt}/{_MAX_RETRIES})")
        except requests.exceptions.HTTPError as exc:
            logger.warning(f"HTTP {exc.response.status_code} for {url} (attempt {attempt}/{_MAX_RETRIES})")
        except requests.exceptions.ConnectionError as exc:
            logger.warning(f"Connection error for {url}: {exc} (attempt {attempt}/{_MAX_RETRIES})")
        except Exception as exc:
            logger.error(f"Unexpected error fetching {url}: {exc}", exc_info=True)
            return None   # non-retriable

        if attempt < _MAX_RETRIES:
            backoff = 2 ** attempt
            logger.debug(f"Retrying {url} in {backoff}s…")
            time.sleep(backoff)

    logger.error(f"All {_MAX_RETRIES} attempts failed for {url}.")
    return None


def _parse_html(html: str, url: str, logger: logging.Logger) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noisy tags
        for tag in soup(["script", "style", "noscript", "nav", "footer",
                          "header", "aside", "form", "iframe", "svg"]):
            tag.decompose()

        # Prefer article / main body; fall back to full body
        main = soup.find("article") or soup.find("main") or soup.find("body")
        if main is None:
            return soup.get_text(separator="\n", strip=True)

        return main.get_text(separator="\n", strip=True)

    except Exception as exc:
        logger.warning(f"HTML parsing failed for {url}: {exc} — using raw text.")
        return html
