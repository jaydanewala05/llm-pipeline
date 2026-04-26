"""
llm_client.py — Call the LLM API and parse structured JSON output.

Supports: OpenAI, Groq, Google Gemini
Selected via the LLM_PROVIDER env var (default: openai).

Retry strategy:
  - Uses tenacity with exponential backoff.
  - Retries on rate-limit (429), server errors (5xx), and timeouts.
  - Raises a permanent failure after MAX_RETRIES attempts.

JSON safety:
  - Strips markdown code fences before parsing.
  - Falls back to a partial-extraction heuristic if json.loads() fails.
  - Returns None only when all recovery attempts are exhausted.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)


MAX_RETRIES = 4
MIN_WAIT    = 2   # seconds
MAX_WAIT    = 30  # seconds

EXTRACTION_SCHEMA = """\
Return ONLY a JSON object — no markdown fences, no prose — with exactly these keys:

{
  "summary":    "<2-3 sentence summary of the text>",
  "entities": {
    "people":        ["<name>", ...],
    "places":        ["<name>", ...],
    "organizations": ["<name>", ...]
  },
  "sentiment": {
    "label":      "<positive|neutral|negative>",
    "confidence": <float 0.0-1.0>
  },
  "key_questions": ["<question 1>", "<question 2>", "<question 3>"]
}

Rules:
- summary must be 2-3 sentences.
- entities lists may be empty ([]) if none are found.
- sentiment.label must be exactly one of: positive, neutral, negative.
- sentiment.confidence must be a float between 0.0 and 1.0.
- key_questions must have exactly 3 items.
- Do NOT include any text outside the JSON object.
"""


# ── RETRIABLE EXCEPTIONS ─────────────────────────────────────────────────────

class _RetriableError(Exception):
    """Raised for errors that warrant a retry."""


class _FatalError(Exception):
    """Raised for errors that should NOT be retried."""


# ── LLM CLIENT ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        self._setup_provider()

    # ── Provider Setup ────────────────────────────────────────────────────────

    def _setup_provider(self):
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "groq":
            self._setup_groq()
        elif self.provider == "gemini":
            self._setup_gemini()
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{self.provider}'. "
                "Choose: openai, groq, gemini"
            )
        self.logger.info(f"LLM provider: {self.provider} | model: {self.model}")

    def _setup_openai(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError
        self._client      = OpenAI(api_key=api_key)
        self.model        = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._rate_error  = RateLimitError
        self._status_error = APIStatusError
        self._timeout_error = APITimeoutError

    def _setup_groq(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set.")
        from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
        self._client       = Groq(api_key=api_key)
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._rate_error   = RateLimitError
        self._status_error = APIStatusError
        self._timeout_error = APITimeoutError

    def _setup_gemini(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model       = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self._genai      = genai
        self._client     = None   # Gemini uses module-level calls

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def extract(self, text: str) -> Optional[dict]:
        """
        Send `text` to the LLM and return a structured extraction dict.
        Returns None if all retries fail or a fatal error occurs.
        """
        try:
            return self._extract_with_retry(text)
        except RetryError as exc:
            self.logger.error(
                f"All {MAX_RETRIES} retry attempts exhausted: {exc.last_attempt.exception()}"
            )
            return None
        except _FatalError as exc:
            self.logger.error(f"Fatal error during extraction: {exc}")
            return None
        except Exception as exc:
            self.logger.error(f"Unexpected error during extraction: {exc}", exc_info=True)
            return None

    # ── Retry-decorated call ──────────────────────────────────────────────────

    def _extract_with_retry(self, text: str) -> Optional[dict]:
        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=MIN_WAIT, max=MAX_WAIT),
            retry=retry_if_exception_type(_RetriableError),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        )
        def _call():
            return self._call_provider(text)

        return _call()

    # ── Provider-specific API calls ───────────────────────────────────────────

    def _call_provider(self, text: str) -> Optional[dict]:
        if self.provider in ("openai", "groq"):
            return self._call_openai_compatible(text)
        elif self.provider == "gemini":
            return self._call_gemini(text)
        raise _FatalError(f"Unknown provider: {self.provider}")

    def _call_openai_compatible(self, text: str) -> Optional[dict]:
        prompt = self._build_prompt(text)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise information-extraction assistant."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                timeout=45,
            )
            raw_output = response.choices[0].message.content or ""
            return self._parse_json(raw_output)

        except self._rate_error as exc:
            self.logger.warning(f"Rate limit hit: {exc}. Will retry.")
            raise _RetriableError(str(exc)) from exc

        except self._timeout_error as exc:
            self.logger.warning(f"API timeout: {exc}. Will retry.")
            raise _RetriableError(str(exc)) from exc

        except self._status_error as exc:
            if exc.status_code and exc.status_code >= 500:
                self.logger.warning(f"Server error {exc.status_code}. Will retry.")
                raise _RetriableError(str(exc)) from exc
            self.logger.error(f"Non-retriable API error {exc.status_code}: {exc}")
            raise _FatalError(str(exc)) from exc

        except Exception as exc:
            self.logger.error(f"Unexpected API call error: {exc}", exc_info=True)
            raise _FatalError(str(exc)) from exc

    def _call_gemini(self, text: str) -> Optional[dict]:
        prompt = self._build_prompt(text)
        try:
            model_obj = self._genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": 0.1},
            )
            response = model_obj.generate_content(prompt)
            raw_output = response.text or ""
            return self._parse_json(raw_output)

        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                self.logger.warning(f"Gemini rate limit: {exc}. Will retry.")
                raise _RetriableError(str(exc)) from exc
            if "timeout" in msg:
                self.logger.warning(f"Gemini timeout: {exc}. Will retry.")
                raise _RetriableError(str(exc)) from exc
            self.logger.error(f"Gemini fatal error: {exc}", exc_info=True)
            raise _FatalError(str(exc)) from exc

    # ── Prompt Builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(text: str) -> str:
        return (
            f"{EXTRACTION_SCHEMA}\n\n"
            f"--- TEXT TO ANALYSE ---\n{text}\n--- END TEXT ---"
        )

    # ── JSON Parser ───────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> Optional[dict]:
        """
        Parse LLM output as JSON.  Handles:
          1. Markdown code fences (```json … ```)
          2. Leading/trailing prose
          3. Malformed JSON → partial extraction fallback
        """
        if not raw.strip():
            self.logger.warning("LLM returned empty response.")
            return None

        # Remove markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Find the first { … } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = cleaned

        # Attempt strict parse
        try:
            data = json.loads(json_str)
            return self._validate_and_coerce(data)
        except json.JSONDecodeError as exc:
            self.logger.warning(f"json.loads failed ({exc}) — attempting repair.")

        # Repair: fix trailing commas and retry
        repaired = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            data = json.loads(repaired)
            self.logger.debug("JSON repaired successfully.")
            return self._validate_and_coerce(data)
        except json.JSONDecodeError:
            pass

        # Last resort: extract fields individually with regex
        self.logger.warning("Falling back to regex partial extraction.")
        return self._regex_fallback(raw)

    def _validate_and_coerce(self, data: dict) -> dict:
        """Ensure required fields exist with correct types."""
        result = {}

        # summary
        result["summary"] = str(data.get("summary", "")).strip() or "No summary available."

        # entities
        ent = data.get("entities", {})
        if not isinstance(ent, dict):
            ent = {}
        result["entities"] = {
            "people":        list(ent.get("people", [])),
            "places":        list(ent.get("places", [])),
            "organizations": list(ent.get("organizations", [])),
        }

        # sentiment
        sent = data.get("sentiment", {})
        if not isinstance(sent, dict):
            sent = {}
        label = str(sent.get("label", "neutral")).lower()
        if label not in ("positive", "neutral", "negative"):
            label = "neutral"
        try:
            confidence = float(sent.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5
        result["sentiment"] = {"label": label, "confidence": confidence}

        # key_questions
        qs = data.get("key_questions", [])
        if not isinstance(qs, list):
            qs = [str(qs)]
        # Pad or trim to exactly 3
        qs = [str(q) for q in qs][:3]
        while len(qs) < 3:
            qs.append("No additional question generated.")
        result["key_questions"] = qs

        return result

    def _regex_fallback(self, raw: str) -> Optional[dict]:
        """Best-effort extraction when JSON is completely broken."""
        result = {
            "summary": "Extraction failed — raw output could not be parsed as JSON.",
            "entities": {"people": [], "places": [], "organizations": []},
            "sentiment": {"label": "neutral", "confidence": 0.0},
            "key_questions": [
                "Could not extract questions.",
                "Could not extract questions.",
                "Could not extract questions.",
            ],
            "parse_error": True,
        }

        # Try to grab summary
        m = re.search(r'"summary"\s*:\s*"([^"]+)"', raw)
        if m:
            result["summary"] = m.group(1)

        # Try to grab sentiment label
        m = re.search(r'"label"\s*:\s*"(positive|neutral|negative)"', raw, re.I)
        if m:
            result["sentiment"]["label"] = m.group(1).lower()

        self.logger.warning("Partial extraction used — results may be incomplete.")
        return result
