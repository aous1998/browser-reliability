"""Success judges: deterministic string check (default) and model-based.

The evaluator decides s_{t,r} for each browser-agent run. `string_judge` is a
normalized-substring check against the reference answer -- deterministic, so
it adds zero noise to the measured run-to-run variance. `judge` is the
WebVoyager-style LLM verdict, kept for answers that cannot be string-matched;
it is itself an unreliable instrument when backed by a small model (it scored
a verbatim-correct answer FAILURE in runs_20260704T093033Z.jsonl).
"""
from __future__ import annotations

import re
import unicodedata

import config

# Standalone number words -> digits, so "two meanings" matches reference "2".
_NUM_WORDS = {w: str(i) for i, w in enumerate(
    ("zero", "one", "two", "three", "four", "five",
     "six", "seven", "eight", "nine", "ten"))}


def _normalize(text: str) -> str:
    """Casefold, strip accents, map number words, collapse non-word runs."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    tokens = re.split(r"[^\w]+", text, flags=re.UNICODE)
    tokens = [_NUM_WORDS.get(t, t) for t in tokens if t]
    return " ".join(tokens)


def string_judge(reference: str, answer: str, parts: list[str] | None = None) -> int:
    """Return 1 iff every required reference part appears in the answer.

    `parts` (from the task's optional "reference_parts") lets a reference like
    a date match order-insensitively; without it the whole reference string
    must appear. Matching is on normalized text (accents/case/punctuation
    ignored), so "efimero" matches "efímero".
    """
    required = parts if parts else [reference]
    norm_answer = _normalize(answer or "")
    if not norm_answer:
        return 0
    return int(all(_normalize(p) in norm_answer for p in required))

_PROMPT = """You are evaluating whether a web-browsing agent completed a task.

Task: {ques}
Reference answer (ground truth): {reference}
Agent's final answer: {answer}

Decide if the agent's final answer is correct and actually fulfils the task.
Be strict: a vague, missing, or wrong answer is a FAILURE. Minor wording
differences that still convey the correct fact are a SUCCESS.

Respond with exactly one word on the first line: SUCCESS or FAILURE.
"""


def judge(ques: str, reference: str, answer: str) -> int:
    """Return 1 if the run is judged a success, else 0."""
    prompt = _PROMPT.format(ques=ques, reference=reference, answer=answer or "(no answer)")

    if config.PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=config.API_KEY)
        resp = client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=16,
            temperature=config.JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = resp.content[0].text.strip().upper()
    elif config.PROVIDER == "ollama":
        import ollama
        client = ollama.Client(host=config.OLLAMA_HOST)
        resp = client.chat(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,  # top-level API param; inside options= it is ignored
            options={"temperature": config.JUDGE_TEMPERATURE},
        )
        verdict = resp["message"]["content"].strip().upper()
    else:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.API_KEY)
        resp = client.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=config.JUDGE_TEMPERATURE),
        )
        verdict = (resp.text or "").strip().upper()

    return 1 if verdict.startswith("SUCCESS") else 0
