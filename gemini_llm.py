#!/usr/bin/env python3
"""
Simple wrapper around the Google Gemini API.
"""

import os
from typing import Any, Dict

import requests


class GeminiChatLLM:
    """
    A tiny LLM wrapper that talks to the Gemini endpoint.

    Parameters
    ----------
    api_key : str | None
        The `X-goog-api-key` header value.  If omitted it will be read from
        the ``GEMINI_API_KEY`` environment variable.
    model_name : str, default="gemini-2.0-flash"
        Which Gemini model to call.
    temperature : float, optional
        Sampling temperature (default 0.1).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key missing – set GEMINI_API_KEY env var"
            )
        # Prioritize passed arg, then env var, then default
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self.temperature = temperature

        # Endpoint (no trailing slash!)
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent"
        )

    def _request(self, prompt: str) -> Dict[str, Any]:
        """Internal helper – performs the HTTP request."""
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            # Optional: tweak temperature / topK etc. if you like
            "generationConfig": {"temperature": self.temperature},
        }

        resp = requests.post(self.endpoint, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Include the response text in the error message for debugging
            raise requests.exceptions.HTTPError(
                f"{e} - Response: {resp.text}", response=resp
            ) from e
        return resp.json()

    def predict(self, prompt: str) -> str:
        """Return the plain text produced by Gemini."""
        res = self._request(prompt)

        try:
            # Expected structure:
            # {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
            text = (
                res["candidates"][0]["content"]["parts"][0]["text"]
            )
        except Exception as exc:          # pragma: no cover
            raise RuntimeError(
                f"Unexpected Gemini response: {res}"
            ) from exc

        return text.strip()

    # ------------------------------------------------------------------
    # Compatibility helpers – LangChain expects a `__call__` that returns
    # an LLMResult.  For the simple RAG chain we only need predict().
    # ------------------------------------------------------------------

    def __call__(self, messages):
        """
        Dummy wrapper to satisfy any code that calls the model like a function.
        It concatenates all message texts and forwards them to `predict`.
        """
        prompt = "\n".join(msg["content"] for msg in messages)
        return self.predict(prompt)
