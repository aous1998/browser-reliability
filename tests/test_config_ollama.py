"""Config knob + infra-marker coverage for the Ollama path. No API key needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.run_experiment import is_infra_error


def test_ollama_num_ctx_default():
    assert config.OLLAMA_NUM_CTX == 16384


def test_ollama_connection_errors_are_infra():
    assert is_infra_error(Exception("[WinError 10061] connection refused by target"))
    assert is_infra_error(Exception("httpx.ConnectError: All connection attempts failed"))
    assert is_infra_error(Exception("httpx.ReadTimeout: timed out"))


def test_wrong_answer_is_not_infra():
    assert not is_infra_error(Exception("ValidationError: action field required"))


if __name__ == "__main__":
    test_ollama_num_ctx_default()
    test_ollama_connection_errors_are_infra()
    test_wrong_answer_is_not_infra()
    print("ok")
