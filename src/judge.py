"""Model-based success judge, in the spirit of WebVoyager's automatic judge.

Given a task question, a reference answer, and the agent's final answer, ask
the model for a binary SUCCESS / FAILURE verdict. This is the evaluator that
decides s_{t,r} for each browser-agent run.
"""
from __future__ import annotations

import config

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
