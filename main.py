"""
LLM Integration & Data Pipeline
AI Engineer Intern — Assignment 2
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

from src.logger import setup_logger
from src.ingestion import ingest_file, ingest_urls
from src.preprocessor import preprocess_documents
from src.llm_client import LLMClient
from src.storage import save_json, save_csv, save_summary_report
from src.reporter import generate_summary_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Data Pipeline: ingest text/PDF files and URLs, extract structured insights."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a .txt or .pdf file to ingest"
    )
    parser.add_argument(
        "--urls", nargs="+", default=[],
        help="One or more URLs to ingest (space-separated)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to write output files (default: outputs/)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Max tokens per chunk sent to the LLM (default: 800)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.file and not args.urls:
        print("Error: Provide at least one --file or one or more --urls.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("pipeline", log_path)
    logger.info("Pipeline started")
    logger.info(f"File input: {args.file}")
    logger.info(f"URL inputs: {args.urls}")

    # ── 1. INGESTION ──────────────────────────────────────────────────────────
    raw_documents = []

    if args.file:
        logger.info(f"Ingesting file: {args.file}")
        doc = ingest_file(args.file, logger)
        if doc:
            raw_documents.append(doc)
        else:
            logger.warning(f"Skipping file {args.file} — ingestion failed.")

    if args.urls:
        logger.info(f"Ingesting {len(args.urls)} URL(s)…")
        url_docs = ingest_urls(args.urls, logger)
        raw_documents.extend(url_docs)

    if not raw_documents:
        logger.error("No documents were successfully ingested. Exiting.")
        sys.exit(1)

    logger.info(f"Total documents ingested: {len(raw_documents)}")

    # ── 2. PREPROCESSING ─────────────────────────────────────────────────────
    logger.info("Preprocessing documents…")
    chunks = preprocess_documents(raw_documents, max_tokens=args.chunk_size, logger=logger)
    logger.info(f"Total chunks after preprocessing: {len(chunks)}")

    # ── 3. LLM EXTRACTION ────────────────────────────────────────────────────
    client = LLMClient(logger=logger)
    results = []
    failed = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)} — source: {chunk['source']}")
        extraction = client.extract(chunk["text"])
        if extraction is None:
            logger.warning(f"Chunk {i} failed extraction — skipping.")
            failed.append(chunk)
            continue

        results.append({
            "chunk_id": i,
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "text_preview": chunk["text"][:200],
            **extraction,
        })

    logger.info(f"Extraction complete: {len(results)} succeeded, {len(failed)} skipped.")

    if not results:
        logger.error("No results produced. Check API key and network.")
        sys.exit(1)

    # ── 4. STORAGE ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path  = output_dir / f"results_{ts}.json"
    csv_path   = output_dir / f"results_{ts}.csv"
    report_path = output_dir / f"summary_{ts}.txt"

    save_json(results, json_path, logger)
    save_csv(results, csv_path, logger)

    report_text = generate_summary_report(results, failed, raw_documents)
    save_summary_report(report_text, report_path, logger)

    logger.info("Pipeline finished successfully.")
    print(f"\n✅  Pipeline complete!")
    print(f"   JSON    → {json_path}")
    print(f"   CSV     → {csv_path}")
    print(f"   Report  → {report_path}")
    print(f"   Log     → {log_path}")
    if failed:
        print(f"   ⚠️  {len(failed)} chunk(s) failed — see log for details.")


if __name__ == "__main__":
    main()
