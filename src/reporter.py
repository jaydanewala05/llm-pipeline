"""
reporter.py — Build a human-readable aggregated summary report.
"""

from collections import Counter
from datetime import datetime
from typing import Optional


def generate_summary_report(
    results: list[dict],
    failed_chunks: list[dict],
    raw_documents: list[dict],
) -> str:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "=" * 70,
        "  LLM DATA PIPELINE — SUMMARY REPORT",
        f"  Generated: {ts}",
        "=" * 70,
        "",
    ]

    # ── Overview ──────────────────────────────────────────────────────────────
    lines += [
        "OVERVIEW",
        "-" * 40,
        f"  Total documents ingested : {len(raw_documents)}",
        f"  Total chunks processed   : {len(results) + len(failed_chunks)}",
        f"  Successful extractions   : {len(results)}",
        f"  Failed / skipped chunks  : {len(failed_chunks)}",
        "",
    ]

    # ── Per-source breakdown ──────────────────────────────────────────────────
    lines += ["SOURCES", "-" * 40]
    source_counts: Counter = Counter(r["source"] for r in results)
    for src, cnt in source_counts.items():
        lines.append(f"  • {src}  ({cnt} chunk(s))")
    lines.append("")

    # ── Sentiment aggregation ─────────────────────────────────────────────────
    sentiment_counts: Counter = Counter(
        r.get("sentiment", {}).get("label", "neutral") for r in results
    )
    total = len(results) or 1
    lines += [
        "SENTIMENT DISTRIBUTION",
        "-" * 40,
        f"  Positive : {sentiment_counts.get('positive', 0):>3}  "
        f"({100 * sentiment_counts.get('positive', 0) / total:.0f}%)",
        f"  Neutral  : {sentiment_counts.get('neutral', 0):>3}  "
        f"({100 * sentiment_counts.get('neutral', 0) / total:.0f}%)",
        f"  Negative : {sentiment_counts.get('negative', 0):>3}  "
        f"({100 * sentiment_counts.get('negative', 0) / total:.0f}%)",
        "",
    ]

    # Average confidence
    confidences = [
        r.get("sentiment", {}).get("confidence", 0.5) for r in results
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    lines += [
        f"  Average sentiment confidence: {avg_conf:.2f}",
        "",
    ]

    # ── Top entities ──────────────────────────────────────────────────────────
    people_ctr: Counter = Counter()
    places_ctr: Counter = Counter()
    orgs_ctr: Counter   = Counter()

    for r in results:
        ent = r.get("entities", {})
        people_ctr.update(ent.get("people", []))
        places_ctr.update(ent.get("places", []))
        orgs_ctr.update(ent.get("organizations", []))

    def _top(counter: Counter, n: int = 5) -> list[tuple]:
        return counter.most_common(n)

    lines += ["TOP ENTITIES", "-" * 40]

    lines.append("  People:")
    for name, cnt in _top(people_ctr):
        lines.append(f"    - {name} ({cnt}x)")
    if not people_ctr:
        lines.append("    (none found)")

    lines.append("  Places:")
    for name, cnt in _top(places_ctr):
        lines.append(f"    - {name} ({cnt}x)")
    if not places_ctr:
        lines.append("    (none found)")

    lines.append("  Organizations:")
    for name, cnt in _top(orgs_ctr):
        lines.append(f"    - {name} ({cnt}x)")
    if not orgs_ctr:
        lines.append("    (none found)")
    lines.append("")

    # ── Per-chunk summaries ───────────────────────────────────────────────────
    lines += ["CHUNK SUMMARIES", "-" * 40]
    for r in results:
        chunk_id    = r.get("chunk_id")
        source      = r.get("source", "unknown")
        chunk_index = r.get("chunk_index", 0)
        summary     = r.get("summary", "")
        sent_label  = r.get("sentiment", {}).get("label", "neutral")
        sent_conf   = r.get("sentiment", {}).get("confidence", 0.0)

        lines.append(f"\n  [Chunk {chunk_id}]  {source}  (part {chunk_index + 1})")
        lines.append(f"  Sentiment : {sent_label} (conf: {sent_conf:.2f})")
        lines.append(f"  Summary   : {summary}")

        qs = r.get("key_questions", [])
        if qs:
            lines.append("  Key Questions:")
            for q in qs:
                lines.append(f"    ? {q}")
    lines.append("")

    # ── Failures ─────────────────────────────────────────────────────────────
    if failed_chunks:
        lines += ["FAILED / SKIPPED CHUNKS", "-" * 40]
        for fc in failed_chunks:
            lines.append(
                f"  - Source: {fc.get('source')}  "
                f"Part {fc.get('chunk_index', 0) + 1}/{fc.get('total_chunks', '?')}"
            )
        lines.append("")

    lines += ["=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(lines)
