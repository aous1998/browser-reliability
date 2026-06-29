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
# Which LLM backbone drives the agent and the judge: "anthropic" or "gemini".
PROVIDER = os.getenv("PROVIDER", "anthropic").lower()

# Anthropic (default). Cheapest current Claude model is Haiku 4.5.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Gemini (the report's backbone). Set the exact id your key can access.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Resolved per-provider model id used for both agent and judge.
AGENT_MODEL = ANTHROPIC_MODEL if PROVIDER == "anthropic" else GEMINI_MODEL
JUDGE_MODEL = os.getenv("JUDGE_MODEL", AGENT_MODEL)
API_KEY = ANTHROPIC_API_KEY if PROVIDER == "anthropic" else GOOGLE_API_KEY

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
