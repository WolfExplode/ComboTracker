"""Persistence adapters for ComboTracker state.

The engine depends on the small StateStore interface. The runtime uses an atomic
JSON file adapter; tests use the in-memory adapter and never touch user data.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


class StateStore(Protocol):
    def load(self) -> dict[str, Any] | None: ...

    def save(self, payload: dict[str, Any]) -> None: ...


class MemoryStateStore:
    """Isolated adapter for tests and ephemeral engine instances."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._payload = copy.deepcopy(initial)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._payload)

    def save(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payload = copy.deepcopy(payload)


class JsonStateStore:
    """Atomic JSON adapter with one validated, known-good backup."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self._lock = threading.Lock()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"State root must be an object: {path}")
        return value

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.path.exists():
                if self.backup_path.exists():
                    return self._read_json(self.backup_path)
                return None
            try:
                return self._read_json(self.path)
            except Exception:
                if not self.backup_path.exists():
                    raise
                logger.warning(
                    "Primary state file is unreadable; loading backup %s",
                    self.backup_path,
                    exc_info=True,
                )
                return self._read_json(self.backup_path)

    def _atomic_write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove state temp file %s", temporary, exc_info=True)

    def save(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        with self._lock:
            # Rotate only valid primary data into the known-good backup. A corrupt
            # primary must never replace a usable backup.
            if self.path.exists():
                try:
                    current = self.path.read_bytes()
                    parsed = json.loads(current.decode("utf-8"))
                    if isinstance(parsed, dict):
                        self._atomic_write(self.backup_path, current)
                except Exception:
                    logger.warning(
                        "Existing state is not valid JSON; preserving prior backup",
                        exc_info=True,
                    )
            self._atomic_write(self.path, encoded)
