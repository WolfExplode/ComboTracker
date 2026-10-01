"""
Input normalization: pynput key/mouse events to internal tokens.
"""

from __future__ import annotations


def normalize_key(key) -> str:
    """
    Normalize pynput keyboard events to our internal string tokens.
    KeyCode has .char; Key has .name. Must never throw (listener callbacks).
    """
    try:
        ch = getattr(key, "char", None)
        if isinstance(ch, str) and ch:
            return ch.lower()

        name = getattr(key, "name", None)
        if isinstance(name, str) and name:
            return name.lower()

        s = str(key)
        s = s.replace("Key.", "").strip().strip("'").strip('"')
        return s.lower()
    except Exception:
        return ""


def normalize_mouse(button) -> str:
    from pynput import mouse

    if button == mouse.Button.left:
        return "lmb"
    if button == mouse.Button.right:
        return "rmb"
    if button == mouse.Button.middle:
        return "mmb"
    return "mouse_extra"
