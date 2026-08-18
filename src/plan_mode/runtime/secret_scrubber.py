"""Secret Scrubber for Output Streams, Error Logs, and Execution Traces."""

from __future__ import annotations

import math
import re
from typing import List, Pattern, Tuple


class SecretScrubber:
    """Detects and redacts high-entropy secrets and credential patterns from runtime output."""

    SECRET_PATTERNS: List[Tuple[str, Pattern]] = [
        ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("GITHUB_PAT", re.compile(r"\bghp_[0-9a-zA-Z]{20,255}\b")),
        ("GITHUB_FINE_GRAINED", re.compile(r"\bgithub_pat_[0-9a-zA-Z_]{20,255}\b")),
        ("BEARER_TOKEN", re.compile(r"(?i)Bearer\s+([a-zA-Z0-9_\-\.~+/]+=*)")),
        ("PRIVATE_KEY", re.compile(r"-----BEGIN\s+[A-Z ]+PRIVATE KEY-----[\s\S]*?-----END\s+[A-Z ]+PRIVATE KEY-----")),
        ("KEY_VALUE_SECRET", re.compile(r"(?i)(password|passwd|secret|api_key|token|auth_token)\s*[:=]\s*['\"]([^'\"]+)['\"]")),
    ]

    def __init__(self, entropy_threshold: float = 3.8, min_entropy_length: int = 16):
        self.entropy_threshold = entropy_threshold
        self.min_entropy_length = min_entropy_length

    @staticmethod
    def calculate_shannon_entropy(token: str) -> float:
        """Calculate Shannon entropy for character frequency distribution."""
        if not token:
            return 0.0
        length = len(token)
        counts: dict[str, int] = {}
        for char in token:
            counts[char] = counts.get(char, 0) + 1
        entropy = 0.0
        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy

    def scrub_text(self, text: str) -> str:
        """Apply pattern scrubbing and entropy scanning to replace secrets with redaction markers."""
        if not text:
            return text

        scrubbed = text

        # 1. Apply structural regex patterns
        for name, pattern in self.SECRET_PATTERNS:
            if name == "KEY_VALUE_SECRET":
                def replace_kv(m: re.Match) -> str:
                    key = m.group(1)
                    return f"{key} = '[REDACTED_SECRET:{name}]'"
                scrubbed = pattern.sub(replace_kv, scrubbed)
            elif name == "BEARER_TOKEN":
                scrubbed = pattern.sub("Bearer [REDACTED_SECRET:BEARER_TOKEN]", scrubbed)
            else:
                scrubbed = pattern.sub(f"[REDACTED_SECRET:{name}]", scrubbed)

        # 2. Scan remaining tokens for high-entropy strings
        words = scrubbed.split()
        for word in words:
            # Strip trailing punctuation
            clean_word = word.strip(".,;:()[]{}'\"")
            if len(clean_word) >= self.min_entropy_length and clean_word.isalnum():
                if not clean_word.startswith("[REDACTED_SECRET"):
                    if self.calculate_shannon_entropy(clean_word) >= self.entropy_threshold:
                        scrubbed = scrubbed.replace(clean_word, "[REDACTED_SECRET:HIGH_ENTROPY]")

        return scrubbed
