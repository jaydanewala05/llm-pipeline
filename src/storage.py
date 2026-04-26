"""
storage.py — Persist pipeline results to JSON, CSV/Excel, and plain-text report.
"""

import json
import logging
from pathlib import Path
from typing import Optional


def save_json(results: list[dict], path: Path, logger: Optional[logging.Logger] = None) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if logger:
            logger.info(f"JSON saved: {path} ({len(results)} records)")
    except Exception as exc:
        if logger:
            logger.error(f"Failed to save JSON: {exc}", exc_info=True)
        raise


def save_csv(results: list[dict], path: Path, logger: Optional[logging.Logger] = None) -> None:
    try:
        import pandas as pd

        rows = []
        for r in results:
            ent = r.get("entities", {})
            sent = r.get("sentiment", {})
            qs = r.get("key_questions", [])
            rows.append({
                "chunk_id":          r.get("chunk_id"),
                "source":            r.get("source"),
                "chunk_index":       r.get("chunk_index"),
                "text_preview":      r.get("text_preview", "")[:300],
                "summary":           r.get("summary", ""),
                "people":            "; ".join(ent.get("people", [])),
                "places":            "; ".join(ent.get("places", [])),
                "organizations":     "; ".join(ent.get("organizations", [])),
                "sentiment_label":   sent.get("label", ""),
                "sentiment_confidence": sent.get("confidence", ""),
                "question_1":        qs[0] if len(qs) > 0 else "",
                "question_2":        qs[1] if len(qs) > 1 else "",
                "question_3":        qs[2] if len(qs) > 2 else "",
                "parse_error":       r.get("parse_error", False),
            })

        df = pd.DataFrame(rows)
        suffix = path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")

        if logger:
            logger.info(f"CSV saved: {path} ({len(rows)} rows)")

    except ImportError:
        if logger:
            logger.error("pandas is not installed. Run: pip install pandas")
        raise
    except Exception as exc:
        if logger:
            logger.error(f"Failed to save CSV: {exc}", exc_info=True)
        raise


def save_summary_report(text: str, path: Path, logger: Optional[logging.Logger] = None) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if logger:
            logger.info(f"Summary report saved: {path}")
    except Exception as exc:
        if logger:
            logger.error(f"Failed to save summary report: {exc}", exc_info=True)
        raise
