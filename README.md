## ComboTracker

A small local web UI + Python backend that listens to your keyboard/mouse and tracks whether you performed a defined “combo” correctly, including **wait** and **hold** timing steps.

### Features
- **Practice combos**: see live status + a step timeline.
- **Wait + hold steps**:
  - `wait` = minimum delay gate (pressing later is OK).
  - `hold` = finger commitment (must hold long enough).
- **Combo enders**: define which “wrong” inputs should drop the combo.
- **Stats**: success/fail, best time, hardest steps, fail reasons.
- **Difficulty scoring** (simple + tunable):
  - Practical APM (uses your expected execution time)
  - Theoretical max APM (uses fastest-possible time)
  - Difficulty out of 10 (keys + timing + simple timing-variation rule)

---

## Getting started

### Requirements
- Python 3.10+ recommended

Install dependencies:

```bash
cd ComboTracker
python -m pip install -r requirements.txt
```

### Run

```bash
cd ComboTracker
python ui_server.py
```

Then open the UI:
- `http://localhost:8080`

Notes:
- The backend also runs a WebSocket server at `ws://localhost:8765`.
- The app listens to global keyboard/mouse via `pynput` (you may need accessibility permissions on some OSes).

---

## Combo format

Combos are written as comma-separated tokens, for example:

`f, wait:0.2, e, wait:0.05, lmb, 2, hold(e, 0.35), r`

### Press steps
- Most tokens are treated as presses: `f`, `e`, `2`, `r`, etc.
- Mouse buttons:
  - `lmb` = left mouse button
  - `rmb` = right mouse button
  - `mmb` = middle mouse button

### Any-order groups (order interchangeable)
Use brackets to indicate that multiple inputs can be pressed in **any order**, but all must be completed before the combo continues:

- Example: `3, r, wait:1.5, [q, e], 2, lmb, rmb`

This means after the wait gate you must press **both** `q` and `e`, in either order (`q` then `e` OR `e` then `q`).

You can also include an animation lock inside the group:
- Example: `e, 3, [wait(r, 1.5), q, e], 2`

This means press `r`, `q`, and `e` in any order — and if/when you press `r`, an animation lock of ≥ 1.5s is enforced before the group can finish (inputs during the lock are ignored).

### Hold steps
Two equivalent syntaxes:
- `hold(e, 0.35)` (seconds)
- `e{350ms}`

The hold step is complete when you’ve held the key for at least the required duration.

### Wait steps (soft vs hard)
Wait steps are “minimum delay gates”:
- **Soft wait (default)**: early presses are ignored; once the time has passed, the next correct input counts.
  - `wait:0.2`
  - `wait_soft:0.2`
- **Hard wait**: pressing too early can drop the combo (useful for games where early input consumes/cancels and desyncs you).
  - `wait_hard:0.2`

### Mandatory animation waits (inputs ignored)
Some abilities have an animation lock where **extra inputs do nothing in-game** (and should not end the combo).

Use:
- `wait(r, 1.5)` to mean: press `r`, then **mandatory** wait ≥ 1.5s (inputs during the lock are ignored).

Example:
`e, 3, wait(r, 1.5), [q, e], 2`

Durations accept:
- Seconds: `0.2`, `0.2s`
- Milliseconds: `200ms`

---

## Combo enders

“Combo enders” are keys/buttons that, if pressed at the wrong time, should drop the combo (instead of being ignored).

In the UI, enter them like:
- `q:0.2, e:0.2, lmb, 1, 2, 3`

Where `key:seconds` is a small grace window (useful if a key is legitimately pressed at the end of a combo and you don’t want immediate re-presses to drop it).

Data is stored in `ComboTracker/combos.json`.

---

## Difficulty + APM

### Expected time
In the editor, set **Expected time** (e.g. `1.05s` or `1050ms`).

This is used for:
- **Practical APM**: \( 60000 / T_{expected} * N_{press} \)

### Theoretical max APM
Computed from **fastest possible combo time** (sum of waits + holds):
- **Theoretical max APM**: \( 60000 / T_{min} * N_{press} \)

### Difficulty (/10)
Difficulty is a simple blend of:
- **Keys**: Practical APM + combo length
- **Timing**:
  - wait difficulty (triangle-shaped; peak ~350ms; fades to 0 by 600ms)
  - hold difficulty (monotonic; weighted higher than waits)
  - a simple “timing variation points” rule (distinct non-micro waits + distinct hold durations + single micro-wait gotcha)

This is intentionally simple so it’s easy to tune.

---

## Troubleshooting

### I get a 404 when opening `http://localhost:8080/`
Make sure you’re running `ui_server.py` from `ComboTracker/`, or just run it normally—recent versions serve `static/` using an absolute path so it works regardless of working directory.

### Nothing is being detected
- Make sure the window is not blocking global hooks (some apps/games run with elevated privileges).
- Try running the Python process as admin if needed.

