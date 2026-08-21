"""Final low-level integrity hardening from the code-by-code audit."""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def patch(ns: dict[str, Any]) -> None:
    from . import cordis as cordis_mod
    from . import fact_identity as fact_identity_mod
    from . import memory_distiller as memory_mod
    from .runtime import secret_scrubber as scrubber_mod

    # ---------------------------------------------------------
    # RoT memory: same-directory temp + fsync + atomic replace. Corruption is
    # an explicit integrity failure rather than silently deleting memory.
    # ---------------------------------------------------------
    def rot_save(self):
        if not self.storage_path:
            return
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            rule_id: {
                "rule_id": rule.rule_id,
                "trigger_condition": rule.trigger_condition,
                "forbidden_pattern": rule.forbidden_pattern,
                "remedy": rule.remedy,
                "source_flaw_type": rule.source_flaw_type,
                "predicate": rule.predicate,
                "affected_resource": rule.affected_resource,
                "perspective": rule.perspective,
                "hit_count": rule.hit_count,
                "success_count": rule.success_count,
                "failure_count": rule.failure_count,
                "confidence": rule.confidence,
                "created_at": rule.created_at,
            }
            for rule_id, rule in self.rules.items()
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp.",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            # Persist the directory entry as well where POSIX supports it.
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def rot_load(self):
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("RoT memory root must be an object")
            loaded = {}
            for rule_id, payload in data.items():
                if not isinstance(payload, dict):
                    raise ValueError(f"rule {rule_id!r} is not an object")
                loaded[rule_id] = memory_mod.RoTRule(**payload)
        except Exception as exc:
            raise ValueError(
                f"corrupt or invalid RoT memory at {self.storage_path}: {exc}"
            ) from exc
        self.rules = loaded

    memory_mod.RoTRuleBase._save = rot_save
    memory_mod.RoTRuleBase._load = rot_load

    # ---------------------------------------------------------
    # Secret scrubber: cover common bare provider key formats that evade the
    # alphanumeric entropy scan because they contain '-' or '_'.
    # ---------------------------------------------------------
    additional_patterns = [
        (
            "OPENAI_STYLE_KEY",
            re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
        ),
        (
            "GOOGLE_API_KEY",
            re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
        ),
    ]
    existing_names = {name for name, _ in scrubber_mod.SecretScrubber.SECRET_PATTERNS}
    scrubber_mod.SecretScrubber.SECRET_PATTERNS = [
        *[item for item in additional_patterns if item[0] not in existing_names],
        *scrubber_mod.SecretScrubber.SECRET_PATTERNS,
    ]

    # ---------------------------------------------------------
    # Fact identity: arbitrary object repr frequently embeds memory addresses.
    # Accept only explicit deterministic/serializable structures.
    # ---------------------------------------------------------
    raw_typed_value = fact_identity_mod._typed_value

    def typed_value(value):
        if (
            value is None
            or type(value) in (bool, int, float, str)
            or isinstance(value, (list, tuple, dict))
        ):
            return raw_typed_value(value)
        if hasattr(value, "model_dump"):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": typed_value(value.model_dump(mode="json")),
            }
        raise TypeError(
            f"unsupported fact argument type {type(value).__module__}.{type(value).__qualname__}; "
            "deterministic typed identity requires JSON-compatible values"
        )

    fact_identity_mod._typed_value = typed_value

    # ---------------------------------------------------------
    # Sync disposal cannot truthfully claim completion while an event loop is
    # active: async inverses would merely be scheduled. Require async_dispose.
    # ---------------------------------------------------------
    raw_dispose = cordis_mod.Context.dispose

    def dispose(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return raw_dispose(self)
        raise RuntimeError(
            "Context.dispose() cannot guarantee rollback completion inside a running event loop; "
            "use await context.async_dispose()"
        )

    cordis_mod.Context.dispose = dispose
