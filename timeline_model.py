"""
Timeline view-model schema (backend -> frontend).

The goal of this module is to make the timeline payload shape explicit and stable.
`combo_engine_ui.timeline_steps()` should return a list of these step dicts.

Compatibility note:
- This project targets Python 3.10+, so we avoid `typing.NotRequired` (3.11+).
  Optional keys are expressed via `TypedDict(total=False)` mixins instead.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


Mark = Literal["ok", "early", "missed", "wrong", "success", "fail"]
WaitMode = Literal["soft", "hard", "mandatory"]


class _WithMark(TypedDict, total=False):
    mark: str


class _WithProgress(TypedDict, total=False):
    progress: int  # 0..100


class _WithOptionalInput(TypedDict, total=False):
    # Some legacy payloads include input: None for waits.
    input: str | None


class _WithSequenceId(TypedDict, total=False):
    sequence_id: str


class TimelineBase(TypedDict):
    type: str
    active: bool
    completed: bool
    duration: int


class TimelinePress(TimelineBase, _WithMark):
    type: Literal["press"]
    input: str


class TimelineHold(TimelineBase, _WithMark):
    type: Literal["hold"]
    input: str


class TimelineWait(TimelineBase, _WithMark, _WithProgress, _WithOptionalInput):
    type: Literal["wait"]
    mode: WaitMode
    wait_for: str


class TimelinePressWait(TimelineBase, _WithMark, _WithProgress):
    type: Literal["press_wait"]
    input: str
    mode: WaitMode  # soft|hard only in practice


class TimelineSequence(TypedDict, _WithMark, _WithSequenceId):
    type: Literal["sequence"]
    active: bool
    completed: bool
    items: list["TimelineItem"]
    progress: dict[str, int]  # {done,total}


class TimelineGroup(TypedDict, _WithMark):
    type: Literal["group"]
    active: bool
    completed: bool
    items: list["TimelineItem"]
    progress: dict[str, int]  # {done,total}


TimelineItem = TimelinePress | TimelineHold | TimelineWait | TimelinePressWait | TimelineSequence | TimelineGroup
TimelineSteps = list[TimelineItem]

