"""
Combo syntax parser: pure AST, no runtime state.

Parses tokens into immutable AST nodes. Use build_state() to convert
AST to dict format for the existing engine (compatibility layer).
"""

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4


# ---------------------------------------------------------------------------
# AST nodes (immutable, no runtime state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressNode:
    key: str
    optional: bool = False


@dataclass(frozen=True)
class HoldNode:
    key: str
    duration_ms: int
    anim_lock_total_ms: int | None = None  # total lock from key down; expands to hold + wait(wait_ms, mandatory, key)


@dataclass(frozen=True)
class HoldWithBodyNode:
    """Hold key for minimum duration while executing an inner sequence of presses/waits.
    Syntax: hold(lmb, 1.75s, {wait:0.15s, q, wait:0.15s, e, wait:0.35s})
    Body may contain PressNode, SpamNode, and WaitNode only (no nested holds/groups).
    Fail immediately on early release of the holder key."""
    key: str
    duration_ms: int
    body: tuple["StepNode", ...]  # PressNode, SpamNode, and WaitNode only


@dataclass(frozen=True)
class SpamNode:
    """Repeated taps of one key over a fixed span. Syntax: spam(lmb, 4.4s)."""
    key: str
    duration_ms: int


@dataclass(frozen=True)
class WaitNode:
    duration_ms: int
    mode: Literal["soft", "hard", "mandatory"]
    wait_for: str | None = None  # for mandatory waits


@dataclass(frozen=True)
class SequenceNode:
    """Ordered sequence of steps. Used for: {}, press+wait, wait(r,t). optional=True means the whole sequence can be skipped."""
    steps: tuple["StepNode", ...]
    optional: bool = False


# Group item: press, hold, or sequence (press+wait, {}, wait(r,t))
GroupItemNode = PressNode | HoldNode | SequenceNode


@dataclass(frozen=True)
class GroupNode:
    """Any-order group. Items can be completed in any order."""
    items: tuple[GroupItemNode, ...]


StepNode = PressNode | HoldNode | HoldWithBodyNode | SpamNode | WaitNode | SequenceNode | GroupNode


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def split_inputs(keys_str: str) -> list[str]:
    """
    Split a user-entered Inputs string into top-level comma-separated tokens.

    Shallow parser: avoids splitting commas inside (), {}, [].
    """
    s = keys_str or ""
    out: list[str] = []
    buf: list[str] = []
    paren = brace = bracket = 0

    for ch in s:
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)

        if ch == "," and paren == 0 and brace == 0 and bracket == 0:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            continue
        buf.append(ch)

    token = "".join(buf).strip()
    if token:
        out.append(token)
    return out


def _parse_duration(raw: str) -> int | None:
    token = (raw or "").lower().strip()
    if not token:
        return None

    if token.endswith("ms"):
        token = token[:-2].strip()
        multiplier = 1
    elif token.endswith("s"):
        token = token[:-1].strip()
        multiplier = 1000
    else:
        multiplier = 1000 if "." in token else 1

    try:
        value = float(token)
    except ValueError:
        return None

    millis = value * multiplier
    if millis <= 0:
        return None
    return int(millis)


# ---------------------------------------------------------------------------
# Parser (pure: token -> AST)
# ---------------------------------------------------------------------------

def parse_step(token: str) -> StepNode | None:
    """
    Parse a single token into an AST node.

    Returns None for empty/invalid tokens.
    """
    t = (token or "").strip()
    if not t:
        return None

    tl = t.lower()

    # Animation-locked (mandatory) wait: wait(r, 1.5) -> [press r, mandatory wait]
    if tl.startswith("wait(") and tl.endswith(")") and len(tl) >= 6:
        inner = tl[len("wait("):-1].strip()
        parts = [p.strip() for p in split_inputs(inner) if p.strip()]
        if len(parts) == 2:
            key = parts[0].strip().lower()
            wait_ms = _parse_duration(parts[1])
            if key and wait_ms is not None:
                return SequenceNode((
                    PressNode(key),
                    WaitNode(wait_ms, "mandatory", key),
                ))

    # Sequential subgroup: {step1, step2, ...}
    if tl.startswith("{") and tl.endswith("}") and len(tl) >= 3:
        inner = tl[1:-1].strip()
        parts = [p.strip() for p in split_inputs(inner) if p.strip()]
        if len(parts) >= 1:
            seq_steps: list[StepNode] = []
            for p in parts:
                node = parse_step(p)
                if node is None:
                    break
                if isinstance(node, SequenceNode) and _is_composite_mandatory(node):
                    seq_steps.extend(node.steps)
                else:
                    seq_steps.append(node)
            else:
                if len(seq_steps) >= 1:
                    return SequenceNode(tuple(seq_steps))

    # Any-order group: [q, e, hold(r,0.3), ...]
    if tl.startswith("[") and tl.endswith("]") and len(tl) >= 3:
        inner = tl[1:-1].strip()
        parts = [p.strip() for p in split_inputs(inner) if p.strip()]
        if len(parts) >= 2:
            items: list[GroupItemNode] = []
            ok = True
            j = 0

            while j < len(parts):
                p = parts[j]
                node = parse_step(p)
                if node is None:
                    ok = False
                    break

                # wait(r, t) as one group item
                if isinstance(node, SequenceNode) and _is_composite_mandatory(node):
                    key = node.steps[0].key if isinstance(node.steps[0], PressNode) else ""
                    wait_node = node.steps[1]
                    if isinstance(wait_node, WaitNode) and wait_node.mode == "mandatory":
                        items.append(SequenceNode(node.steps))
                        j += 1
                        continue

                # Hold
                if isinstance(node, HoldNode):
                    if node.duration_ms <= 0:
                        ok = False
                        break
                    items.append(node)
                    j += 1
                    continue

                # Sequence subgroup {}
                if isinstance(node, SequenceNode):
                    items.append(node)
                    j += 1
                    continue

                # Standalone wait is invalid in group
                if isinstance(node, WaitNode):
                    ok = False
                    break

                # Press
                if isinstance(node, PressNode):
                    inp = node.key
                    # Check for press+wait pair
                    if j + 1 < len(parts):
                        nxt = parse_step(parts[j + 1])
                        if isinstance(nxt, WaitNode) and nxt.mode in ("soft", "hard"):
                            wms = nxt.duration_ms
                            if wms > 0:
                                items.append(SequenceNode((node, nxt)))
                                j += 2
                                continue
                    items.append(node)
                    j += 1
                    continue

                ok = False
                break

            if ok and len(items) >= 2:
                # Validate: no duplicate key in mandatory_wait + press
                has_anim_wait = any(
                    isinstance(it, SequenceNode) and _is_composite_mandatory(it)
                    for it in items
                )
                if has_anim_wait:
                    mw_key = None
                    for it in items:
                        if isinstance(it, SequenceNode) and _is_composite_mandatory(it):
                            mw_key = it.steps[0].key if isinstance(it.steps[0], PressNode) else None
                            break
                    if mw_key:
                        for it in items:
                            if isinstance(it, PressNode) and it.key == mw_key:
                                ok = False
                                break

                if ok:
                    return GroupNode(tuple(items))

    # Wait gate: wait:0.1 (only form; use after a key e.g. f, wait:0.23s)
    if tl.startswith("wait:"):
        dur = tl[len("wait:"):].strip()
        wait_ms = _parse_duration(dur)
        if wait_ms is not None:
            return WaitNode(wait_ms, "soft", None)

    # Repeated taps over a duration: spam(lmb, 4.4s)
    if tl.startswith("spam(") and tl.endswith(")"):
        inner = tl[len("spam("):-1]
        parts = [p.strip() for p in split_inputs(inner) if p.strip()]
        if len(parts) == 2 and parts[0]:
            spam_ms = _parse_duration(parts[1])
            if spam_ms is not None:
                return SpamNode(parts[0].strip().lower(), spam_ms)

    # Hold: hold(e, 0.35) or hold(e, 0.5s, 2s) for hold + animation lock
    #       or hold(e, 1.75s, {wait:0.15s, q, ...}) for hold-with-body
    if tl.startswith("hold(") and tl.endswith(")"):
        inner = tl[len("hold("):-1]
        parts = [p.strip() for p in split_inputs(inner) if p.strip()]
        if len(parts) >= 2 and parts[0]:
            hold_ms = _parse_duration(parts[1])
            if hold_ms is not None:
                if len(parts) == 3 and parts[2].startswith("{") and parts[2].endswith("}"):
                    # hold(key, time, {body}) — new hold-with-body form
                    body_inner = parts[2][1:-1].strip()
                    body_parts = [bp.strip() for bp in split_inputs(body_inner) if bp.strip()]
                    body_nodes: list[StepNode] = []
                    ok = True
                    for bp in body_parts:
                        bn = parse_step(bp)
                        if bn is None or not isinstance(bn, (PressNode, SpamNode, WaitNode)):
                            ok = False
                            break
                        body_nodes.append(bn)
                    if ok and body_nodes:
                        return HoldWithBodyNode(parts[0].strip().lower(), hold_ms, tuple(body_nodes))
                    # Fall through to plain press if body parse fails
                elif len(parts) >= 3:
                    # hold(key, hold_ms, total_lock_ms) — existing anim-lock form
                    anim_lock_total_ms: int | None = None
                    total_ms = _parse_duration(parts[2])
                    if total_ms is not None and total_ms > hold_ms:
                        anim_lock_total_ms = total_ms
                    return HoldNode(parts[0].strip().lower(), hold_ms, anim_lock_total_ms)
                else:
                    return HoldNode(parts[0].strip().lower(), hold_ms, None)

    # Optional press: -key
    if tl.startswith("-") and len(tl) > 1:
        key = tl[1:].strip().lower()
        if key:
            return PressNode(key=key, optional=True)

    # Plain press
    return PressNode(tl.strip().lower())


def _is_composite_mandatory(node: SequenceNode) -> bool:
    """True if sequence is wait(r,t) -> [press, mandatory wait]."""
    if len(node.steps) != 2:
        return False
    a, b = node.steps
    return isinstance(a, PressNode) and isinstance(b, WaitNode) and b.mode == "mandatory"


# ---------------------------------------------------------------------------
# Compatibility layer: AST -> dict (for existing engine)
# ---------------------------------------------------------------------------

def build_state(node: StepNode) -> dict[str, Any]:
    """
    Convert AST node to runtime dict format expected by ComboTrackerEngine.

    For SequenceNode that represents wait(r,t), returns {"composite_steps": [...]}
    so the caller can expand it. All other nodes return a single step dict.
    """
    match node:
        case PressNode(key=key):
            opt = getattr(node, "optional", False)
            return {"input": key, "hold_ms": None, "wait_ms": None, "optional": opt}

        case HoldNode(key=key, duration_ms=ms, anim_lock_total_ms=anim_lock):
            if anim_lock is not None:
                wait_ms = max(0, anim_lock - ms)
                return {
                    "composite_steps": [
                        {"input": key, "hold_ms": ms, "wait_ms": None},
                        {
                            "input": None,
                            "hold_ms": None,
                            "wait_ms": int(wait_ms),
                            "wait_mode": "mandatory",
                            "wait_for": key or "",
                        },
                    ]
                }
            return {"input": key, "hold_ms": ms, "wait_ms": None}

        case HoldWithBodyNode(key=key, duration_ms=ms, body=body_steps):
            return {
                "input": key,
                "hold_ms": ms,
                "wait_ms": None,
                "hold_with_body": True,
                "body_steps": [build_state(s) for s in body_steps],
            }

        case SpamNode(key=key, duration_ms=ms):
            return {
                "input": key,
                "hold_ms": None,
                "wait_ms": None,
                "spam_ms": ms,
            }

        case WaitNode(duration_ms=ms, mode=mode, wait_for=wf):
            return {
                "input": None,
                "hold_ms": None,
                "wait_ms": int(ms),
                "wait_mode": mode,
                "wait_for": wf or "",
            }

        case SequenceNode(steps=steps):
            if _is_composite_mandatory(node):
                return {
                    "composite_steps": [
                        build_state(steps[0]),
                        build_state(steps[1]),
                    ]
                }
            seq_id = uuid4().hex[:8]
            return {
                "input": None,
                "hold_ms": None,
                "wait_ms": None,
                "is_sequence": True,
                "sequence_id": seq_id,
                "sequence_steps": [build_state(s) for s in steps],
                "sequence_index": 0,
                "sequence_started": False,
            }

        case GroupNode(items=items):
            return _build_group_state(items)

        case _:
            raise TypeError(f"Unknown node type: {type(node)}")


def _is_press_wait(node: SequenceNode) -> bool:
    """True if sequence is press+wait (soft/hard)."""
    if len(node.steps) != 2:
        return False
    a, b = node.steps
    return isinstance(a, PressNode) and isinstance(b, WaitNode) and b.mode in ("soft", "hard")


def _build_group_state(items: tuple[GroupItemNode, ...]) -> dict[str, Any]:
    """Build the group step dict from AST items."""
    pw_need_counts: dict[str, int] = {}
    pw_meta: dict[str, dict[str, Any]] = {}
    pw_order_sigs: list[str] = []
    mandatory_wait: dict[str, Any] | None = None
    order: list[dict[str, Any]] = []
    press_need_counts: dict[str, int] = {}
    hold_need_counts: dict[str, int] = {}
    hold_meta: dict[str, dict[str, Any]] = {}
    hold_order_sigs: list[str] = []
    seq_need_counts: dict[str, int] = {}
    seq_meta: dict[str, dict[str, Any]] = {}
    seq_order_ids: list[str] = []

    for it in items:
        if isinstance(it, PressNode):
            inp = it.key
            press_need_counts[inp] = press_need_counts.get(inp, 0) + 1
            order.append({"kind": "press", "input": inp})

        elif isinstance(it, HoldNode):
            sig = f"{it.key}:{it.duration_ms}"
            hold_need_counts[sig] = hold_need_counts.get(sig, 0) + 1
            hold_meta[sig] = {"input": it.key, "hold_ms": it.duration_ms}
            if sig not in hold_order_sigs:
                hold_order_sigs.append(sig)
            order.append({"kind": "hold", "sig": sig, "input": it.key, "hold_ms": it.duration_ms})

        elif isinstance(it, SequenceNode):
            if _is_composite_mandatory(it):
                key = it.steps[0].key if isinstance(it.steps[0], PressNode) else ""
                w = it.steps[1]
                if isinstance(w, WaitNode):
                    mandatory_wait = {"wait_for": key, "wait_ms": w.duration_ms}
                    order.append({"kind": "anim_wait", "wait_for": key, "wait_ms": w.duration_ms})
            elif _is_press_wait(it):
                a, b = it.steps
                sig = f"{a.key}:{b.duration_ms}:{b.mode}"
                pw_need_counts[sig] = pw_need_counts.get(sig, 0) + 1
                pw_meta[sig] = {"input": a.key, "wait_ms": b.duration_ms, "wait_mode": b.mode}
                if sig not in pw_order_sigs:
                    pw_order_sigs.append(sig)
                order.append({"kind": "press_wait", "sig": sig, "input": a.key, "wait_ms": b.duration_ms, "wait_mode": b.mode})
            else:
                seq_id = uuid4().hex[:8]
                seq_steps = [build_state(s) for s in it.steps]
                seq_need_counts[seq_id] = 1
                seq_meta[seq_id] = {
                    "sequence_id": seq_id,
                    "sequence_steps": seq_steps,
                    "sequence_index": 0,
                    "sequence_started": False,
                }
                seq_order_ids.append(seq_id)
                order.append({"kind": "sequence", "sequence_id": seq_id, "sequence_steps": seq_steps})

    uniq_presses = list(press_need_counts.keys())

    return {
        "input": None,
        "hold_ms": None,
        "wait_ms": None,
        "group_presses": uniq_presses,
        "group_press_need_counts": press_need_counts,
        "group_pw_need_counts": pw_need_counts,
        "group_pw_done_counts": {},
        "group_pw_meta": pw_meta,
        "group_pw_order_sigs": pw_order_sigs,
        "group_done_counts": {},
        "group_hold_need_counts": hold_need_counts,
        "group_hold_done_counts": {},
        "group_hold_meta": hold_meta,
        "group_hold_order_sigs": hold_order_sigs,
        "group_hold_active": False,
        "group_hold_sig": "",
        "group_hold_for": "",
        "group_hold_started_at": 0.0,
        "group_hold_required_ms": 0,
        "group_mandatory_wait": mandatory_wait,
        "group_order": order,
        "group_wait_active": False,
        "group_wait_done": False,
        "group_wait_started_at": 0.0,
        "group_wait_until": 0.0,
        "group_pw_active": False,
        "group_pw_sig": "",
        "group_pw_until": 0.0,
        "group_seq_need_counts": seq_need_counts,
        "group_seq_done_counts": {},
        "group_seq_meta": seq_meta,
        "group_seq_order_ids": seq_order_ids,
        "group_seq_active": False,
        "group_seq_id": "",
        "group_seq_index": 0,
    }


def expanded_ast_from_tokens(tokens: list[str]) -> list[StepNode]:
    """
    Parse tokens and return a flat list of AST nodes.
    Expands composite (e.g. wait(r,1.5) -> [PressNode, WaitNode]) inline.
    Key + wait attachment: a press followed by a soft/hard wait is always merged into
    one SequenceNode (one composite step = key with interruptible animation). Optional
    press (-e) + wait becomes SequenceNode(..., optional=True).
    Use with states.build_runtime_state to get runtime steps.
    """
    ast_list: list[StepNode] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        node = parse_step(t)
        if node is None:
            i += 1
            continue
        # Press + following soft/hard wait -> one sequence (key + interruptible animation)
        if isinstance(node, PressNode) and i + 1 < len(tokens):
            nxt = parse_step(tokens[i + 1])
            if isinstance(nxt, WaitNode) and nxt.mode in ("soft", "hard") and (nxt.duration_ms or 0) > 0:
                press_only = PressNode(key=node.key)  # drop optional on inner press
                opt = getattr(node, "optional", False)
                ast_list.append(SequenceNode((press_only, nxt), optional=opt))
                i += 2
                continue
        if isinstance(node, SequenceNode) and _is_composite_mandatory(node):
            ast_list.extend(node.steps)
        elif isinstance(node, HoldNode) and getattr(node, "anim_lock_total_ms", None) is not None:
            # hold(key, hold_ms, total_lock_ms) -> hold + mandatory wait
            wait_ms = max(0, node.anim_lock_total_ms - node.duration_ms)
            ast_list.append(HoldNode(node.key, node.duration_ms))
            ast_list.append(WaitNode(wait_ms, "mandatory", node.key))
        elif isinstance(node, HoldWithBodyNode):
            # hold-with-body is a single step; no expansion
            ast_list.append(node)
        elif isinstance(node, SpamNode):
            ast_list.append(node)
        else:
            ast_list.append(node)
        i += 1
    return ast_list


def runtime_source_token_indices_from_tokens(tokens: list[str]) -> list[list[int]]:
    """
    Return source token indices for each runtime AST step produced by expanded_ast_from_tokens().

    Each entry in the returned list corresponds to one runtime step index and contains
    one or more source token indices from the original token list.
    """
    source_map: list[list[int]] = []
    i = 0
    while i < len(tokens):
        node = parse_step(tokens[i])
        if node is None:
            i += 1
            continue

        # Press + following soft/hard wait collapses into one runtime SequenceNode.
        if isinstance(node, PressNode) and i + 1 < len(tokens):
            nxt = parse_step(tokens[i + 1])
            if isinstance(nxt, WaitNode) and nxt.mode in ("soft", "hard") and (nxt.duration_ms or 0) > 0:
                source_map.append([i, i + 1])
                i += 2
                continue

        if isinstance(node, SequenceNode) and _is_composite_mandatory(node):
            # wait(key, t) expands to two runtime steps (press + mandatory wait),
            # both originating from the same source token.
            source_map.append([i])
            source_map.append([i])
        elif isinstance(node, HoldNode) and getattr(node, "anim_lock_total_ms", None) is not None:
            # hold(key, hold_ms, total_lock_ms) expands to hold + mandatory wait,
            # both originating from the same source token.
            source_map.append([i])
            source_map.append([i])
        elif isinstance(node, HoldWithBodyNode):
            # hold-with-body is a single runtime step
            source_map.append([i])
        elif isinstance(node, SpamNode):
            source_map.append([i])
        else:
            source_map.append([i])
        i += 1
    return source_map


def steps_from_tokens(tokens: list[str]) -> list[dict[str, Any]]:
    """
    Parse tokens and build flat list of step dicts.
    Expands composite_steps (e.g. wait(r,1.5)) inline.
    Mirrors set_active_combo step-building logic.
    """
    steps: list[dict[str, Any]] = []
    for t in tokens:
        node = parse_step(t)
        if node is None:
            continue
        d = build_state(node)
        if d.get("composite_steps") is not None:
            for sub in d.get("composite_steps") or []:
                if isinstance(sub, dict) and sub:
                    steps.append(sub)
        else:
            steps.append(d)
    return steps
