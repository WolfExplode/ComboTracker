"""
UI adapter layer for ComboTracker.

Purpose:
- Keep UI policy (dedupe, signature tracking, when to emit timeline updates) out of the core engine.
- The engine calls small hooks on input/tick; the adapter decides what UI messages to emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import combo_engine_ui as ui


EmitFunc = Callable[[dict[str, Any]], None]


@dataclass
class UIAdapter:
    """
    Tracks small pieces of UI-only state:
    - nested wait animation dedupe signature
    - active-step signature for timeline_update-on-change behavior
    """

    wait_anim_sig: tuple | None = None
    active_step_sig: Any | None = None

    def reset(self) -> None:
        self.wait_anim_sig = None
        self.active_step_sig = None

    def sync_wait_animation(self, engine, *, emit: EmitFunc) -> None:
        """
        Emit wait_begin/wait_end for nested waits so the frontend can animate progress via RAF
        without requiring tick-spam timeline updates.
        """
        try:
            step = engine._active_runtime_step()
            new_sig, events = ui.wait_animation_events(engine, active_step=step, prev_sig=self.wait_anim_sig)
            self.wait_anim_sig = new_sig
            for ev in events:
                emit(ev)
        except Exception:
            return

    def set_active_step_signature(self, engine) -> None:
        """Align stored signature with the current active runtime step."""
        try:
            self.active_step_sig = ui.ui_signature_for_step(engine._active_runtime_step())
        except Exception:
            self.active_step_sig = None

    def maybe_emit_timeline_update_on_change(self, engine, *, emit: EmitFunc) -> None:
        """
        If the active step's user-visible completion state changes without a top-level advance
        (e.g. a nested wait finishes inside a SequenceState), emit a single timeline_update.
        """
        try:
            step = engine._active_runtime_step()
            if step is None:
                return
            sig = ui.ui_signature_for_step(step)
            if self.active_step_sig is None:
                self.active_step_sig = sig
                return
            if sig != self.active_step_sig:
                self.active_step_sig = sig
                emit({"type": "timeline_update", "steps": ui.timeline_steps(engine)})
        except Exception:
            return

