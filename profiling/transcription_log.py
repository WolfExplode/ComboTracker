"""Persistent raw-event artifacts for completed transcription sessions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class TranscriptionLogWriter:
    """Write replayable transcription evidence without affecting input capture."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def write(self, recording: dict) -> Path | None:
        payload = dict(recording)
        payload["captured_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_path = self.directory / f"transcription-{stamp}-{uuid4().hex[:8]}.json"
            self._atomic_write(run_path, payload)
            self._atomic_write(self.directory / "latest.json", payload)
            return run_path
        except Exception:
            logger.warning("Could not write transcription recording", exc_info=True)
            return None

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
