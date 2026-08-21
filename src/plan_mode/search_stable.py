"""Stable helpers for plan-space search."""
from __future__ import annotations

import hashlib
import random
from typing import Any


def stable_rng_for_text(text: str) -> random.Random:
    """Return a process-independent RNG seeded by the normalized input text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))
