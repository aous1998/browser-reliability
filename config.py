"""Central configuration for the seeded-defect web-QA study.

All values can be overridden through environment variables (see .env.example),
so the harness is runnable without editing code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# --- Model (local via Ollama) -----------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Context window for the local model. Ollama's default (~4k tokens) silently
# truncates Browser Use's 10-20k-token prompts and the agent loses the task.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
# Browser Use's default per-call LLM timeout (75 s) is too tight for a local
# model whose KV cache spills to CPU at this context size.
OLLAMA_LLM_TIMEOUT_S = int(os.getenv("OLLAMA_LLM_TIMEOUT_S", "300"))

# Decoding temperature, held fixed across all runs and non-zero by design:
# the run-to-run variation it produces is part of what the study measures.
AGENT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))

# --- Experiment ------------------------------------------------------------
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))

# Number of repeated audits per page (k in the report).
K = int(os.getenv("K", "5"))
# Step budget per audit run.
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
# Run the browser without a visible window.
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
