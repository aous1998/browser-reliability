"""Ollama chat model with qwen3-style thinking properly disabled.

Browser Use's ChatOllama forwards `ollama_options` to the Ollama API as
`options=` only. But `think` is a *top-level* chat() parameter, not an option,
so `{"think": False}` inside options is silently ignored: qwen3:8b then spends
minutes per agent step generating hidden chain-of-thought at local-inference
speeds, and steps blow through the LLM timeout. This wrapper injects
`think=False` into every chat() call at the level the API actually reads.
"""
from __future__ import annotations

from browser_use import ChatOllama


class ThinklessChatOllama(ChatOllama):
    """ChatOllama whose client always calls chat(..., think=False)."""

    def get_client(self):
        client = super().get_client()
        orig_chat = client.chat

        async def chat(*args, **kwargs):
            kwargs.setdefault("think", False)
            return await orig_chat(*args, **kwargs)

        chat.__thinkless__ = True
        client.chat = chat
        return client
