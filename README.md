# LLM Integration & Data Pipeline
### AI Engineer Intern — Assignment 2

A production-style Python pipeline that ingests unstructured text from `.txt`/`.pdf` files and URLs, preprocesses it, calls an LLM API for structured extraction, and stores clean results — all without LangChain or similar orchestration frameworks.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/llm-pipeline.git
cd llm-pipeline

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key (no hardcoding — env vars only)
export LLM_PROVIDER=groq          # or: openai | gemini
export GROQ_API_KEY=your_key_here
# export OPENAI_API_KEY=...
# export GEMINI_API_KEY=...

# 5. Run on a file + URLs
python main.py \
  --file sample_inputs/ai_article.txt \
  --urls https://en.wikipedia.org/wiki/Artificial_intelligence \
         https://en.wikipedia.org/wiki/Large_language_model \
  --output-dir outputs \
  --chunk-size 800
```

All three outputs (JSON, CSV, summary `.txt`) and a timestamped log file are written to `--output-dir`.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | `openai` | One of: `openai`, `groq`, `gemini` |
| `OPENAI_API_KEY` | If provider=openai | — | OpenAI secret key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `GROQ_API_KEY` | If provider=groq | — | Groq secret key |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Groq model name |
| `GEMINI_API_KEY` | If provider=gemini | — | Google Gemini key |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | Gemini model name |

You may also use a `.env` file with `python-dotenv`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

---

## Project Structure

```
llm-pipeline/
├── main.py                  # Entry point — orchestrates the full pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── logger.py            # Coloured console + file logging
│   ├── ingestion.py         # .txt, .pdf, and URL ingestion
│   ├── preprocessor.py      # Cleaning, denoising, token-aware chunking
│   ├── llm_client.py        # Multi-provider LLM calls, retry logic, JSON parsing
│   ├── storage.py           # JSON + CSV/Excel output writers
│   └── reporter.py          # Aggregated plain-text summary report builder
├── sample_inputs/
│   ├── ai_article.txt       # Sample text file used for testing
│   └── sample_urls.txt      # Sample URL list
└── outputs/
    ├── results_sample.json  # Sample JSON output
    ├── results_sample.csv   # Sample CSV output
    └── summary_sample.txt   # Sample aggregated report
```

---

## Pipeline Stages

```
┌─────────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────┐
│  Ingestion  │───▶│ Preprocessing │───▶│  LLM Extract│───▶│ Storage  │
│             │    │               │    │             │    │          │
│ .txt / .pdf │    │ - Encoding fix│    │ - Summary   │    │ JSON     │
│ URLs (HTTP) │    │ - Boilerplate │    │ - Entities  │    │ CSV      │
│             │    │   removal     │    │ - Sentiment │    │ TXT      │
│             │    │ - Chunking    │    │ - Questions │    │ Report   │
└─────────────┘    └───────────────┘    └─────────────┘    └──────────┘
```

### 1. Ingestion (`src/ingestion.py`)
- `.txt`: reads with UTF-8, falls back to latin-1 on decode error.
- `.pdf`: uses `pypdf` to extract text page by page; warns if a page fails.
- URLs: fetches with `requests`, strips HTML noise with BeautifulSoup (removes `<script>`, `<nav>`, `<footer>`, etc.); prefers `<article>` / `<main>` tags.
- Bad inputs are logged and skipped — the pipeline continues.

### 2. Preprocessing (`src/preprocessor.py`)
- Normalises Unicode (NFC), fixes smart quotes, removes BOM, collapses blank lines.
- Strips boilerplate lines matching ~10 regex patterns (cookie banners, subscribe prompts, etc.).
- Chunks using `tiktoken` (cl100k_base) for accurate token counts; falls back to a word-count estimate if not installed.
- Splits on paragraph → sentence → hard-word boundaries to stay within `--chunk-size`.

### 3. LLM Extraction (`src/llm_client.py`)
Each chunk is sent to the LLM with a strict schema prompt requesting:

```json
{
  "summary": "2-3 sentence summary",
  "entities": { "people": [], "places": [], "organizations": [] },
  "sentiment": { "label": "positive|neutral|negative", "confidence": 0.0-1.0 },
  "key_questions": ["q1", "q2", "q3"]
}
```

**Retry logic** (via `tenacity`):
- Retries on: rate limits (429), server errors (5xx), timeouts.
- Strategy: exponential backoff, min 2s, max 30s, up to 4 attempts.
- Fatal errors (4xx non-rate-limit) are not retried.

**JSON safety**:
1. Strips markdown code fences (` ```json … ``` `).
2. Extracts first `{…}` block from the response.
3. Attempts `json.loads()`.
4. On failure, attempts a comma-repair pass.
5. Last resort: regex partial extraction for `summary` and `sentiment.label`.
6. Validates and coerces all fields — never returns malformed data to storage.

### 4. Storage (`src/storage.py`) + Report (`src/reporter.py`)
- **JSON**: one list of dicts, pretty-printed.
- **CSV**: one row per chunk with flattened entity and question columns.
- **Summary report**: plain text with overview stats, sentiment distribution, top entities by frequency, per-chunk summaries, and a list of failed chunks.

---

## Error Handling Philosophy

| Scenario | Behaviour |
|---|---|
| File not found | Logged as ERROR; skipped |
| PDF page extraction failure | Logged as WARNING; page skipped; other pages continue |
| URL connection error / timeout | Up to 3 retries with exponential backoff; logged then skipped |
| LLM rate limit (429) | Retried up to 4× with exponential backoff |
| LLM server error (5xx) | Retried up to 4× |
| Malformed JSON from LLM | Multi-stage repair → partial extraction → None |
| Entire chunk fails extraction | Logged; added to `failed_chunks`; pipeline continues |

A bare `except: pass` is never used — every exception is logged with context.

---

## LLM Provider Choice

**Primary choice: Groq** (`llama3-8b-8192`)

**Reasons:**
1. **Speed**: Groq's LPU hardware delivers ~500 tokens/second, making the pipeline fast even on many chunks.
2. **Cost**: Substantially cheaper than GPT-4 class models for bulk extraction.
3. **OpenAI-compatible SDK**: The `groq` Python package mirrors the `openai` SDK, so switching between providers requires only changing the `LLM_PROVIDER` env var.
4. **JSON reliability**: Llama 3 follows system-prompted JSON schemas reliably.

OpenAI (`gpt-4o-mini`) and Gemini (`gemini-1.5-flash`) are also fully supported with no code changes.

---

## Inputs Tested

| Input | Type | Chunks | Notes |
|---|---|---|---|
| `sample_inputs/ai_article.txt` | TXT | 3 | Multi-paragraph AI news article |
| Wikipedia — Artificial Intelligence | URL | 1 | HTML parsed; boilerplate stripped |
| Wikipedia — Large Language Model | URL | 1 | HTML parsed; boilerplate stripped |

---

## Known Limitations

1. **Scanned PDFs**: `pypdf` cannot extract text from image-based PDFs. OCR support (e.g., `pytesseract`) is not included.
2. **JavaScript-heavy pages**: `requests` + BeautifulSoup cannot render client-side JS. Use `playwright` or `selenium` for SPAs.
3. **Very long documents**: Token counting uses cl100k_base (OpenAI tokenizer); actual token counts for Groq/Gemini models may differ slightly.
4. **Rate limits with many chunks**: No global request queue or concurrency limiter — large batches may still hit rate limits even with per-request backoff.
5. **Language**: Prompt and output validation assume English. Non-English text will be processed but entity extraction quality may vary.
6. **PDF metadata / tables**: Complex PDF layouts with tables or multi-column text may not extract cleanly with `pypdf`.

---

## Sample Outputs

Sample outputs from a real pipeline run are included in `outputs/`:
- `results_sample.json` — full extraction results
- `results_sample.csv` — tabular format
- `summary_sample.txt` — aggregated plain-text report
