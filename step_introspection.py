"""
Pure functions for introspecting step state (labels, start keys, input acceptance).
Used by combo_engine and combo_analytics; no engine state required.
"""

from __future__ import annotations

from typing import Any

from states import GroupState, HoldState, HoldWithBodyState, PressState, SequenceState, SpamState, WaitState


def expected_label_for_step(step: Any) -> str:
    """Return display label for a StepState (used by UI and fail reporting)."""
    if step is None:
        return "—"
    if isinstance(step, PressState):
        inp = (step.expected or "").strip().lower() or "—"
        if getattr(step, "optional", False):
            return f"{inp}?"
        return inp
    if isinstance(step, HoldState):
        h = step.required_ms
        inp = (step.expected or "").strip().lower()
        return f"hold({inp},≥{h}ms)" if inp else f"hold(≥{h}ms)"
    if isinstance(step, HoldWithBodyState):
        h = step.required_ms
        inp = (step.expected or "").strip().lower()
        return f"hold({inp},≥{h}ms,body)" if inp else f"hold(≥{h}ms,body)"
    if isinstance(step, SpamState):
        inp = (step.expected or "").strip().lower()
        return f"spam({inp},{step.required_ms}ms)"
    if isinstance(step, WaitState):
        w = step.required_ms
        if step.mode == "hard":
            return f"wait-hard(≥{w}ms)"
        if step.mode == "mandatory":
            k = (step.wait_for or "").strip().lower()
            return f"anim-wait({k},≥{w}ms)" if k else f"anim-wait(≥{w}ms)"
        return f"wait(≥{w}ms)"
    if isinstance(step, SequenceState):
        labels = []
        for s in step.steps:
            if isinstance(s, WaitState):
                w = s.required_ms
                if s.mode == "hard":
                    labels.append(f"wait-hard({w}ms)")
                elif s.mode == "mandatory":
                    k = (s.wait_for or "").strip().lower()
                    labels.append(f"{k}+wait({w}ms)" if k else f"wait({w}ms)")
                else:
                    labels.append(f"wait({w}ms)")
            elif isinstance(s, HoldState):
                labels.append(f"hold({s.expected},{s.required_ms}ms)")
            else:
                labels.append((getattr(s, "expected", "") or "").strip().lower() or "?")
        return f"seq({' → '.join(labels)})" if labels else "seq(—)"
    if isinstance(step, GroupState):
        opts = []
        for item in step.items:
            if item.kind == "press" and isinstance(item.state, PressState):
                opts.append((item.state.expected or "").strip().lower())
            elif item.kind == "hold" and isinstance(item.state, HoldState):
                opts.append((item.state.expected or "").strip().lower())
            elif item.kind == "press_wait" and isinstance(item.state, SequenceState) and len(item.state.steps) >= 2 and isinstance(item.state.steps[0], PressState):
                opts.append((item.state.steps[0].expected or "").strip().lower())
            elif item.kind == "anim_wait" and isinstance(item.state, WaitState) and item.state.wait_for:
                opts.append((item.state.wait_for or "").strip().lower())
            elif item.kind == "sequence" and isinstance(item.state, SequenceState) and item.state.steps:
                first = item.state.steps[0]
                if isinstance(first, PressState):
                    opts.append((first.expected or "").strip().lower())
                elif isinstance(first, WaitState) and first.wait_for:
                    opts.append((first.wait_for or "").strip().lower())
        opts = [o for o in opts if o]
        return f"any-order({'|'.join(opts)})" if opts else "any-order(—)"
    return "—"


def start_keys_for_step(step: Any) -> set[str]:
    """
    Return the set of input keys that can *start* the given step.
    Used for forgiving holds: only treat a key as an "attempt to move on" if it matches
    the next expected action, not random keys like movement.
    """
    out: set[str] = set()
    try:
        if step is None:
            return out
        if isinstance(step, PressState):
            k = str(step.expected or "").strip().lower()
            if k:
                out.add(k)
            return out
        if isinstance(step, (HoldState, HoldWithBodyState, SpamState)):
            k = str(step.expected or "").strip().lower()
            if k:
                out.add(k)
            return out
        if isinstance(step, WaitState):
            return out
        if isinstance(step, SequenceState):
            if getattr(step, "steps", None):
                return start_keys_for_step(step.steps[0])
            return out
        if isinstance(step, GroupState):
            for item in getattr(step, "items", []) or []:
                try:
                    if item.kind == "press" and isinstance(item.state, PressState):
                        k = str(item.state.expected or "").strip().lower()
                        if k:
                            out.add(k)
                    elif item.kind == "hold" and isinstance(item.state, HoldState):
                        k = str(item.state.expected or "").strip().lower()
                        if k:
                            out.add(k)
                    elif item.kind == "anim_wait" and isinstance(item.state, WaitState):
                        k = str(item.state.wait_for or "").strip().lower()
                        if k:
                            out.add(k)
                    elif item.kind in ("press_wait", "sequence") and isinstance(item.state, SequenceState):
                        out |= start_keys_for_step(item.state)
                except Exception:
                    continue
            return out
    except Exception:
        pass
    return out


def optional_step_key(step: Any) -> str | None:
    """
    Return the key that would complete this optional step (the key valid in its "slot").
    Returns None if step is not an optional PressState or optional SequenceState.
    Used for optional-key grace: that key is also valid one or two steps forward.
    """
    if step is None:
        return None
    if isinstance(step, PressState) and getattr(step, "optional", False):
        k = str(step.expected or "").strip().lower()
        return k if k else None
    if isinstance(step, SequenceState) and getattr(step, "optional", False):
        keys = start_keys_for_step(step)
        return next(iter(keys), None) if keys else None
    return None


def step_accepts_input(step: Any, input_name: str) -> bool:
    """True if this step could accept input_name (for find next/prev)."""
    input_name = (input_name or "").strip().lower()
    if not input_name:
        return False
    if isinstance(step, WaitState):
        return False
    if isinstance(step, PressState):
        return (step.expected or "").strip().lower() == input_name
    if isinstance(step, SpamState):
        return (step.expected or "").strip().lower() == input_name
    if isinstance(step, (HoldState, HoldWithBodyState)):
        return (step.expected or "").strip().lower() == input_name
    if isinstance(step, GroupState):
        for item in step.items:
            if item.kind == "press" and isinstance(item.state, PressState) and (item.state.expected or "").strip().lower() == input_name:
                return True
            if item.kind == "hold" and isinstance(item.state, HoldState) and (item.state.expected or "").strip().lower() == input_name:
                return True
            if item.kind == "press_wait" and isinstance(item.state, SequenceState) and len(item.state.steps) >= 2 and isinstance(item.state.steps[0], PressState) and (item.state.steps[0].expected or "").strip().lower() == input_name:
                return True
            if item.kind == "anim_wait" and isinstance(item.state, WaitState) and (item.state.wait_for or "").strip().lower() == input_name:
                return True
            if item.kind == "sequence" and isinstance(item.state, SequenceState) and item.state.steps:
                first = item.state.steps[0]
                if isinstance(first, PressState) and (first.expected or "").strip().lower() == input_name:
                    return True
                if isinstance(first, WaitState) and (first.wait_for or "").strip().lower() == input_name:
                    return True
        return False
    if isinstance(step, SequenceState) and step.steps:
        first = step.steps[0]
        if isinstance(first, PressState) and (first.expected or "").strip().lower() == input_name:
            return True
        if isinstance(first, HoldState) and (first.expected or "").strip().lower() == input_name:
            return True
    return False
