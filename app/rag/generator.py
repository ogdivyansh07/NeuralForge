"""
Gemini answer generation: lazy client, strong prompts, safe error strings.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

NOT_FOUND_MESSAGE = "I could not find this in the uploaded PDF."
_CLIENT: Any = None
_MODEL_NAME = "gemini-2.0-flash"


def get_genai_client():
    """Lazy-init Gemini client; returns None if API key missing (no import-time crash)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    from google import genai

    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def build_student_qa_prompt(context: str, query: str) -> str:
    """
    Prompt template: concise, student-friendly, grounded, anti-hallucination.
    """
    return f"""You are a careful teaching assistant helping a student understand an uploaded PDF.

Rules (follow strictly):
- Use ONLY information supported by the Context below. Do not use outside knowledge.
- If the Context does not contain enough information to answer, reply with exactly:
  {NOT_FOUND_MESSAGE}
- Write in clear, simple English (short paragraphs, no fluff).
- Summarize and synthesize; do NOT paste long raw quotes or repeat the whole Context.
- If you mention facts, they must appear in the Context.

Context:
{context}

Question:
{query}

Answer:"""


def generate_answer(context: str, query: str) -> str:
    """
    Call Gemini to produce a grounded summary. Returns user-facing strings only (never raises).
    """
    client = get_genai_client()
    if client is None:
        return "⚠️ API key not configured. Set GEMINI_API_KEY in your environment or .env file."

    prompt = build_student_qa_prompt(context, query)
    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            return "⚠️ Gemini error: empty response"
        return text
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg:
            return "⚠️ API quota exceeded. Please try later."
        if "api key" in error_msg or "401" in error_msg or "403" in error_msg:
            return "⚠️ Invalid API key."
        return f"⚠️ Gemini error: {str(e)}"


def gemini_failed(result: str) -> bool:
    return result.strip().startswith("⚠️")
