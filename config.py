"""Central configuration for the browser-agent reliability study.

All values can be overridden through environment variables (see .env.example),
so the harness is runnable without editing code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# --- Model / provider ------------------------------------------------------
# Which LLM backbone drives the agent and the judge: "anthropic", "gemini", or
# "ollama" (a local model served by Ollama -- free, no API tokens).
PROVIDER = os.getenv("PROVIDER", "anthropic").lower()

# Anthropic (default). Cheapest current Claude model is Haiku 4.5.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Gemini (the report's backbone). Set the exact id your key can access.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Ollama (local). Drives the agent and judge from a model on this machine.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Context window for the local model. Ollama's default (~4k tokens) silently
# truncates Browser Use's 10-20k-token prompts, which makes the agent lose the
# task entirely (see results/runs_20260630T202223Z.jsonl: every answer chases a
# phantom task from the truncated system prompt's worked example).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
# Browser Use's default per-call LLM timeout (75 s) is too tight for a local
# model whose KV cache spills to CPU at this context size; every call times out
# and the run aborts. Give local calls more headroom.
OLLAMA_LLM_TIMEOUT_S = int(os.getenv("OLLAMA_LLM_TIMEOUT_S", "300"))

# Resolved per-provider model id and credential used for both agent and judge.
if PROVIDER == "anthropic":
    AGENT_MODEL = ANTHROPIC_MODEL
    API_KEY = ANTHROPIC_API_KEY
elif PROVIDER == "ollama":
    AGENT_MODEL = OLLAMA_MODEL
    API_KEY = "local"  # unused, but keeps the "key is set" guard satisfied
else:
    AGENT_MODEL = GEMINI_MODEL
    API_KEY = GOOGLE_API_KEY
JUDGE_MODEL = os.getenv("JUDGE_MODEL", AGENT_MODEL)

# Decoding settings, held fixed across all runs (report, Experimental Setup).
# The agent temperature is non-zero by design: the run-to-run sampling variation
# it produces is exactly the phenomenon under study. The judge runs at 0 so the
# evaluator is deterministic and does not add noise to the measured variance.
AGENT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

# Recover actions that Haiku serialises into `thinking` instead of the required
# structured `action` field (see src/repair_llm.py). Anthropic only. Set
# REPAIR_TOOL_CALLS=false to measure the unmodified off-the-shelf stack.
REPAIR_TOOL_CALLS = os.getenv("REPAIR_TOOL_CALLS", "true").lower() == "true"

# --- Experiment ------------------------------------------------------------
TASKS_PATH = Path(os.getenv("TASKS_PATH", ROOT / "tasks" / "webvoyager_slice.jsonl"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))

# Number of independent runs per task (k in the report).
K = int(os.getenv("K", "3"))
# Reliability threshold tau: a task is "reliably solved" when p_hat >= TAU.
TAU = float(os.getenv("TAU", "0.9"))
# Per-task step budget handed to the Browser Use agent.
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
# Run the browser without a visible window.
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"

# On a transient infra error (429 rate limit, 503), retry the run this many
# times before marking it invalid (excluded from p_hat, not counted as failure).
INFRA_RETRIES = int(os.getenv("INFRA_RETRIES", "2"))
INFRA_BACKOFF_S = float(os.getenv("INFRA_BACKOFF_S", "30"))
