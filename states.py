# states.py
"""
Runtime state objects for combo steps.
States know how to update themselves; the engine coordinates and decides what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Union

from parser import (
    GroupNode,
    HoldNode,
    PressNode,
    SequenceNode,
    StepNode,
    WaitNode,
    _is_composite_mandatory,
    _is_press_wait,
)


# ---------------------------------------------------------------------------
# Result types (what happened, engine decides what to do)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcceptResult:
    """Input was accepted."""
    record_hit: bool = True  # Engine should record timing
    advance: bool = False    # Engine should advance to next step


@dataclass(frozen=True)
class IgnoreResult:
    """Input ignored (e.g. during wait, wrong key in group)."""
    pass


@dataclass(frozen=True)
class FailResult:
    """Combo should drop."""
    reason: str


@dataclass(frozen=True)
class CompleteResult:
    """Step completed, engine should advance."""
    auto: bool = False  # True if completed without input (timer expired)


ProcessResult = Union[AcceptResult, IgnoreResult, FailResult, CompleteResult]


# ---------------------------------------------------------------------------
# State protocol (common interface)
# ---------------------------------------------------------------------------

class StepState(Protocol):
    """Protocol for all runtime states."""

    def process_press(self, key: str, now: float) -> ProcessResult:
        ...

    def process_release(self, key: str, now: float) -> ProcessResult:
        ...

    def tick(self, now: float) -> ProcessResult:
        """Called periodically for time-based advancement."""
        ...

    def to_dict(self) -> dict:
        """For UI rendering."""
        ...

    def reset(self) -> None:
        """Reset for new attempt."""
        ...

    def skip_and_advance(self, now: float) -> bool:
        """No-fail mode: skip current (failed) and advance. Returns True if this step is now complete (engine should advance)."""
        ...


# ---------------------------------------------------------------------------
# Simple states (Press, Hold, Wait)
# ---------------------------------------------------------------------------

@dataclass
class PressState:
    expected: str
    optional: bool = False
    completed: bool = False
    was_skipped: bool = False

    def process_press(self, key: str, now: float) -> ProcessResult:
        if key == self.expected:
            self.completed = True
            return AcceptResult(advance=True)
        return IgnoreResult()

    def process_release(self, key: str, now: float) -> ProcessResult:
        return IgnoreResult()

    def tick(self, now: float) -> ProcessResult:
        return IgnoreResult()

    def to_dict(self) -> dict:
        return {
            "type": "press",
            "input": self.expected,
            "optional": self.optional,
            "completed": self.completed,
            "was_skipped": self.was_skipped,
        }

    def reset(self) -> None:
        self.completed = False
        self.was_skipped = False

    def skip_and_advance(self, now: float) -> bool:
        return True


@dataclass
class HoldState:
    expected: str
    required_ms: int

    # Runtime mutable state
    in_progress: bool = False
    started_at: float = 0.0
    completed: bool = False

    def process_press(self, key: str, now: float) -> ProcessResult:
        if key != self.expected:
            # Forgiving hold: ignore other keys.
            # The engine decides whether a key is an "attempt to move on" (e.g., next expected combo input).
            return IgnoreResult()

        if not self.in_progress:
            # Start hold
            self.in_progress = True
            self.started_at = now
            return AcceptResult(advance=False)  # Don't advance, wait for release or timeout

        # Already holding, ignore repeat press
        return IgnoreResult()

    def process_release(self, key: str, now: float) -> ProcessResult:
        if key != self.expected or not self.in_progress:
            return IgnoreResult()

        held_ms = (now - self.started_at) * 1000
        if held_ms >= self.required_ms:
            self.complete()
            return AcceptResult(advance=True)
        self.in_progress = False
        # Forgiving hold: releasing too early does NOT fail immediately.
        # Failure happens only if the user attempts to move on (presses another key).
        return IgnoreResult()

    def tick(self, now: float) -> ProcessResult:
        return IgnoreResult()

    def check_complete(self, now: float) -> bool:
        """Check if hold duration satisfied (for engine to call)."""
        if not self.in_progress:
            return False
        return (now - self.started_at) * 1000 >= self.required_ms

    def complete(self) -> None:
        """Mark this hold as successfully completed. Call when engine has determined hold duration was satisfied."""
        self.in_progress = False
        self.completed = True

    def to_dict(self) -> dict:
        return {
            "type": "hold",
            "input": self.expected,
            "duration": self.required_ms,
            "in_progress": self.in_progress,
            "completed": self.completed,
        }

    def reset(self) -> None:
        self.in_progress = False
        self.started_at = 0.0
        self.completed = False

    def skip_and_advance(self, now: float) -> bool:
        return True


@dataclass
class WaitState:
    required_ms: int
    mode: Literal["soft", "hard", "mandatory"]
    wait_for: str | None = None  # for mandatory waits

    # Runtime mutable state
    in_progress: bool = False
    started_at: float = 0.0
    completed: bool = False

    def start(self, now: float) -> None:
        """Engine calls this when entering wait step."""
        self.in_progress = True
        self.started_at = now

    def process_press(self, key: str, now: float) -> ProcessResult:
        if not self.in_progress:
            if self.completed:
                return IgnoreResult()
            return IgnoreResult()

        # Check if wait completed
        if self.check_complete(now):
            self.completed = True
            self.in_progress = False
            return CompleteResult()

        # Wait still in progress
        if self.mode == "mandatory":
            return IgnoreResult()

        if self.mode == "hard":
            return FailResult("early press during hard wait")

        return IgnoreResult()

    def process_release(self, key: str, now: float) -> ProcessResult:
        return IgnoreResult()

    def tick(self, now: float) -> ProcessResult:
        if not self.in_progress or self.completed:
            return IgnoreResult()

        if self.check_complete(now):
            self.completed = True
            self.in_progress = False
            return CompleteResult()

        return IgnoreResult()

    def check_complete(self, now: float) -> bool:
        if not self.in_progress:
            return self.completed
        return (now - self.started_at) * 1000 >= self.required_ms

    def to_dict(self) -> dict:
        return {
            "type": "wait",
            "mode": self.mode,
            "wait_for": self.wait_for,
            "duration": self.required_ms,
            "in_progress": self.in_progress,
            "completed": self.completed,
        }

    def reset(self) -> None:
        self.in_progress = False
        self.started_at = 0.0
        self.completed = False

    def skip_and_advance(self, now: float) -> bool:
        return True


# ---------------------------------------------------------------------------
# Complex states (Sequence, Group)
# ---------------------------------------------------------------------------

@dataclass
class SequenceState:
    """Sequential execution: {} or press+wait pairs. optional=True means the whole sequence can be skipped."""
    steps: list[StepState] = field(default_factory=list)
    is_mandatory_wait: bool = False  # True if this is wait(r,t) composite
    optional: bool = False
    was_skipped: bool = False

    # Runtime mutable state
    current_index: int = 0
    started: bool = False

    def _advance_inner(self, now: float) -> bool:
        """
        Advance to the next inner step. If that next step is a WaitState, start it.
        Returns True if the sequence has reached/passed the end.
        """
        self.current_index += 1
        if self.current_index >= len(self.steps):
            return True
        next_step = self.steps[self.current_index]
        if isinstance(next_step, WaitState):
            next_step.start(now)
        return False

    def process_press(self, key: str, now: float) -> ProcessResult:
        """
        Important: inner waits can complete due to time *during a press*.

        If a nested WaitState returns CompleteResult here and we bubble it up unchanged,
        the engine may misinterpret it as "the whole top-level step completed" and
        skip ahead (especially when this SequenceState lives inside a GroupState).

        So we consume CompleteResult here by advancing the inner index, then
        re-processing the same key against the next inner step (because the key
        was not actually consumed by the wait completion).
        """
        if not self.started:
            self.started = True

        while True:
            if self.current_index >= len(self.steps):
                return IgnoreResult()

            current = self.steps[self.current_index]
            result = current.process_press(key, now)

            # A wait finished; advance inner state without consuming this key.
            if isinstance(result, CompleteResult):
                if self._advance_inner(now):
                    # Sequence completed without consuming the key:
                    # let the parent re-process it (e.g., GroupState can try other items,
                    # or the engine can apply it to the next top-level step).
                    return CompleteResult(auto=result.auto)
                # Continue loop to try to apply the same key to the next step.
                continue

            if isinstance(result, AcceptResult) and result.advance:
                if self._advance_inner(now):
                    return AcceptResult(advance=True, record_hit=result.record_hit)
                return AcceptResult(advance=False, record_hit=result.record_hit)

            if isinstance(result, FailResult):
                return result

            return result

    def process_release(self, key: str, now: float) -> ProcessResult:
        if not self.started or self.current_index >= len(self.steps):
            return IgnoreResult()

        current = self.steps[self.current_index]
        result = current.process_release(key, now)

        if isinstance(result, AcceptResult) and result.advance:
            if self._advance_inner(now):
                return AcceptResult(advance=True)
            return AcceptResult(advance=False)

        return result

    def tick(self, now: float) -> ProcessResult:
        if not self.started or self.current_index >= len(self.steps):
            return IgnoreResult()

        current = self.steps[self.current_index]
        result = current.tick(now)

        if isinstance(result, CompleteResult):
            if self._advance_inner(now):
                return AcceptResult(advance=True)
            return AcceptResult(advance=False)

        return result

    def to_dict(self) -> dict:
        d = {
            "type": "sequence",
            "started": self.started,
            "current": self.current_index,
            "total": len(self.steps),
            "items": [s.to_dict() for s in self.steps],
        }
        if self.optional:
            d["optional"] = True
            d["was_skipped"] = self.was_skipped
        return d

    def reset(self) -> None:
        self.current_index = 0
        self.started = False
        self.was_skipped = False
        for s in self.steps:
            s.reset()

    def skip_and_advance(self, now: float) -> bool:
        """Skip current inner step and advance. Returns True if sequence is now complete."""
        return self._advance_inner(now)


@dataclass
class GroupItemTracker:
    """Tracks one item in a group (press, hold, or sequence)."""
    state: StepState
    kind: Literal["press", "hold", "press_wait", "anim_wait", "sequence"]
    signature: str = ""
    completed_count: int = 0
    required_count: int = 1


@dataclass
class GroupState:
    """Any-order group: [q, e, hold(r,0.3), ...]"""
    items: list[GroupItemTracker] = field(default_factory=list)

    # Runtime mutable state
    active_item: GroupItemTracker | None = None
    wait_active: bool = False  # For animation lock waits

    def process_press(self, key: str, now: float) -> ProcessResult:
        # Check if we're in an animation lock wait
        if self.wait_active:
            for item in self.items:
                if item.kind == "anim_wait" and isinstance(item.state, WaitState):
                    result = item.state.process_press(key, now)
                    if isinstance(result, CompleteResult):
                        self.wait_active = False
                        item.completed_count = 1
                        if self._is_complete():
                            return AcceptResult(advance=True)
                        return IgnoreResult()
                    return IgnoreResult()
            return IgnoreResult()

        # We may need to apply the same key multiple times if an active sequence
        # completes "for free" due to an inner wait finishing (CompleteResult).
        # In that case, the key was not consumed and should be tried against other
        # group items in the same press event.
        while True:
            # Check if we have an active item in progress (hold or sequence)
            if self.active_item is not None:
                result = self.active_item.state.process_press(key, now)

                if isinstance(result, FailResult):
                    self.active_item = None
                    return result

                if isinstance(result, CompleteResult):
                    # Active item completed without consuming key (e.g., sequence ended on a wait).
                    self.active_item.completed_count += 1
                    self.active_item = None
                    if self._is_complete():
                        # Group completed without consuming key; let engine re-process it on next step.
                        return CompleteResult(auto=result.auto)
                    # Try to use the same key for another item.
                    continue

                if isinstance(result, AcceptResult):
                    if result.advance:
                        self.active_item.completed_count += 1
                        self.active_item = None
                        if self._is_complete():
                            return AcceptResult(advance=True, record_hit=result.record_hit)
                    return AcceptResult(advance=result.advance, record_hit=result.record_hit)

                return result

            # Try to start a new item (including starting anim_wait with the wait_for key)
            progressed_without_consuming_key = False
            for item in self.items:
                if self._is_item_satisfied(item):
                    continue

                # anim_wait: pressing wait_for key starts the mandatory wait
                if (
                    item.kind == "anim_wait"
                    and isinstance(item.state, WaitState)
                    and item.state.wait_for == key
                    and not item.state.in_progress
                    and not self.wait_active
                ):
                    item.state.start(now)
                    self.wait_active = True
                    return AcceptResult(advance=False)

                result = item.state.process_press(key, now)

                if isinstance(result, CompleteResult):
                    # Item completed without consuming key; mark it and keep trying same key.
                    item.completed_count += 1
                    if self._is_complete():
                        return CompleteResult(auto=result.auto)
                    progressed_without_consuming_key = True
                    break

                if isinstance(result, AcceptResult):
                    if not result.advance:
                        self.active_item = item
                        return AcceptResult(advance=False, record_hit=result.record_hit)
                    item.completed_count += 1
                    if self._is_complete():
                        return AcceptResult(advance=True, record_hit=result.record_hit)
                    return AcceptResult(advance=False, record_hit=result.record_hit)

                if isinstance(result, FailResult):
                    return result

            if progressed_without_consuming_key:
                continue

            return IgnoreResult()

    def process_release(self, key: str, now: float) -> ProcessResult:
        if self.active_item is not None:
            result = self.active_item.state.process_release(key, now)

            if isinstance(result, FailResult):
                self.active_item = None
                return result

            if isinstance(result, AcceptResult) and result.advance:
                self.active_item.completed_count += 1
                self.active_item = None
                if self._is_complete():
                    return AcceptResult(advance=True)
                return AcceptResult(advance=False)

            return result

        return IgnoreResult()

    def tick(self, now: float) -> ProcessResult:
        if self.wait_active:
            for item in self.items:
                if item.kind == "anim_wait" and isinstance(item.state, WaitState):
                    result = item.state.tick(now)
                    if isinstance(result, CompleteResult):
                        self.wait_active = False
                        item.completed_count = 1
                        if self._is_complete():
                            return AcceptResult(advance=True)
                        return AcceptResult(advance=False)
                    break
            return IgnoreResult()

        if self.active_item is not None:
            result = self.active_item.state.tick(now)

            if isinstance(result, CompleteResult) or (
                isinstance(result, AcceptResult) and result.advance
            ):
                self.active_item.completed_count += 1
                self.active_item = None
                if self._is_complete():
                    return AcceptResult(advance=True)
                return AcceptResult(advance=False)

            return result

        return IgnoreResult()

    def _is_item_satisfied(self, item: GroupItemTracker) -> bool:
        return item.completed_count >= item.required_count

    def _is_complete(self) -> bool:
        return all(self._is_item_satisfied(item) for item in self.items)

    def to_dict(self) -> dict:
        return {
            "type": "group",
            "items": [
                {
                    "kind": item.kind,
                    "signature": item.signature,
                    "completed": item.completed_count,
                    "required": item.required_count,
                    "state": item.state.to_dict(),
                }
                for item in self.items
            ],
            "complete": self._is_complete(),
        }

    def reset(self) -> None:
        self.active_item = None
        self.wait_active = False
        for item in self.items:
            item.completed_count = 0
            item.state.reset()

    def skip_and_advance(self, now: float) -> bool:
        """Skip current active item (count as done) and advance. Returns True if group is now complete."""
        if self.active_item is not None:
            self.active_item.completed_count += 1
            self.active_item = None
        return self._is_complete()


# ---------------------------------------------------------------------------
# Factory: AST -> Runtime State
# ---------------------------------------------------------------------------

def build_runtime_state(node: StepNode) -> StepState:
    """Convert AST node to runtime state object."""
    match node:
        case PressNode(key=key, optional=opt):
            return PressState(expected=key, optional=opt)

        case HoldNode(key=key, duration_ms=ms):
            return HoldState(expected=key, required_ms=ms)

        case WaitNode(duration_ms=ms, mode=mode, wait_for=wf):
            return WaitState(required_ms=ms, mode=mode, wait_for=wf)

        case SequenceNode(steps=steps):
            is_mandatory = _is_composite_mandatory(node)
            step_states = [build_runtime_state(s) for s in steps]
            opt = getattr(node, "optional", False)
            return SequenceState(
                steps=step_states,
                is_mandatory_wait=is_mandatory,
                optional=opt,
            )

        case GroupNode(items=items):
            trackers: list[GroupItemTracker] = []

            for item in items:
                if isinstance(item, PressNode):
                    trackers.append(
                        GroupItemTracker(
                            state=PressState(expected=item.key, optional=getattr(item, "optional", False)),
                            kind="press",
                            signature=item.key,
                        )
                    )

                elif isinstance(item, HoldNode):
                    sig = f"{item.key}:{item.duration_ms}"
                    trackers.append(
                        GroupItemTracker(
                            state=HoldState(expected=item.key, required_ms=item.duration_ms),
                            kind="hold",
                            signature=sig,
                        )
                    )

                elif isinstance(item, SequenceNode):
                    if _is_composite_mandatory(item):
                        press_node = item.steps[0]
                        wait_node = item.steps[1]
                        trackers.append(
                            GroupItemTracker(
                                state=WaitState(
                                    required_ms=wait_node.duration_ms,
                                    mode="mandatory",
                                    wait_for=(
                                        press_node.key
                                        if isinstance(press_node, PressNode)
                                        else None
                                    ),
                                ),
                                kind="anim_wait",
                                signature=(
                                    f"anim:{press_node.key if isinstance(press_node, PressNode) else ''}"
                                ),
                            )
                        )
                    elif _is_press_wait(item):
                        press_node = item.steps[0]
                        wait_node = item.steps[1]
                        sig = f"{press_node.key}:{wait_node.duration_ms}:{wait_node.mode}"
                        trackers.append(
                            GroupItemTracker(
                                state=SequenceState(
                                    steps=[
                                        PressState(expected=press_node.key, optional=getattr(press_node, "optional", False)),
                                        WaitState(
                                            required_ms=wait_node.duration_ms,
                                            mode=wait_node.mode,
                                        ),
                                    ],
                                    is_mandatory_wait=False,
                                ),
                                kind="press_wait",
                                signature=sig,
                            )
                        )
                    else:
                        seq_state = build_runtime_state(item)
                        assert isinstance(seq_state, SequenceState)
                        first_key = ""
                        if item.steps and isinstance(item.steps[0], PressNode):
                            first_key = item.steps[0].key
                        trackers.append(
                            GroupItemTracker(
                                state=seq_state,
                                kind="sequence",
                                signature=f"seq:{first_key or '...'}",
                            )
                        )

            return GroupState(items=trackers)

        case _:
            raise TypeError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def steps_from_ast(ast_steps: list[StepNode]) -> list[StepState]:
    """Build runtime states from AST nodes."""
    return [build_runtime_state(node) for node in ast_steps]
