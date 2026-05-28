from __future__ import annotations

import logging
import shutil
import time
from collections import deque
from pathlib import Path

import tiktoken
from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


_encoder: tiktoken.Encoding | None = None


def count_tokens(text: str) -> int:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def cleanup_clone(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class RateLimiter:
    """Enforces RPM and TPM limits for Gemini free tier."""

    def __init__(self, rpm: int = 15, tpm: int = 1_000_000):
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: deque[float] = deque()
        self._token_usage: deque[tuple[float, int]] = deque()

    def wait_if_needed(self, estimated_tokens: int) -> None:
        now = time.time()

        # Clean up old entries (older than 60s)
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        while self._token_usage and now - self._token_usage[0][0] > 60:
            self._token_usage.popleft()

        # Check RPM
        if len(self._request_times) >= self.rpm:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logging.info(f"[dim]限频等待 {sleep_time:.1f}s (RPM)[/dim]")
                time.sleep(sleep_time)

        # Check TPM
        current_tokens = sum(t for _, t in self._token_usage)
        if current_tokens + estimated_tokens > self.tpm:
            sleep_time = 60 - (now - self._token_usage[0][0])
            if sleep_time > 0:
                logging.info(f"[dim]限频等待 {sleep_time:.1f}s (TPM)[/dim]")
                time.sleep(sleep_time)

    def record_request(self, tokens_used: int) -> None:
        now = time.time()
        self._request_times.append(now)
        self._token_usage.append((now, tokens_used))
