"""ThinklessChatOllama must disable qwen3-style thinking at the API level.

Browser Use's ChatOllama forwards `ollama_options` as `options=` only, and
`think` is a top-level Ollama API parameter, not an option -- so putting
{"think": False} in options is silently ignored and the model reasons at
length on every step. The wrapper injects think=False into each chat() call.
No Ollama server is needed: we only verify the client's chat is wrapped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.local_llm import ThinklessChatOllama


def test_thinkless_client_wraps_chat():
    llm = ThinklessChatOllama(model="qwen3:8b", host="http://localhost:11434")
    client = llm.get_client()
    assert getattr(client.chat, "__thinkless__", False), "chat() not wrapped"


if __name__ == "__main__":
    test_thinkless_client_wraps_chat()
    print("ok")
