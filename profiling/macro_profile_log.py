"""Persistent JSON artifacts for completed macro timing profiles."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from profiling.macro_timing import MacroTimingProfile

logger = logging.getLogger(__name__)


class MacroProfileLogWriter:
    """Writes post-run timing artifacts; failures never affect macro playback."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def write(
        self,
        profile: MacroTimingProfile,
        *,
        outcome: str,
        combo_name: str | None,
        plan_duration_ms: float,
    ) -> Path | None:
        captured_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        payload = {
            "schema_version": 1,
            "captured_at": captured_at,
            "outcome": outcome,
            "combo_name": combo_name,
            "plan_duration_ms": round(float(plan_duration_ms), 3),
            "profile": profile.to_dict(),
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_path = self.directory / f"macro-{stamp}-{uuid4().hex[:8]}.json"
            self._atomic_write(run_path, payload)
            self._atomic_write(self.directory / "latest.json", payload)
            return run_path
        except Exception:
            logger.warning("Could not write macro timing profile", exc_info=True)
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
