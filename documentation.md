## ComboTracker Documentation

This file contains the longer-form docs that don’t need to live in `README.md`:
- Detailed **combo syntax**
- **Combo enders** behavior and cooldown rules
- **Difficulty/APM** definitions
- **Troubleshooting**
- Developer notes and a quick **architecture** map of the Python modules

---

### Quick links

- **Getting started**: see `README.md`
- **Combo format**: [Combo format](#combo-format)
- **Combo enders**: [Combo enders](#combo-enders)
- **Difficulty + APM**: [Difficulty--apm](#difficulty--apm)
- **Troubleshooting**: [Troubleshooting](#troubleshooting)
- **Developer / architecture**: [Architecture](#architecture)

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

The hold step is complete when you’ve held the key for at least the required duration. You do **not** need to release the hold key before pressing the next step: pressing the next key while still holding is accepted (e.g. `hold(lmb, 0.1s), 1` — hold LMB for 0.1s then press 1 without releasing LMB). If a wait step immediately precedes a hold and you are already holding that key when the wait ends, the hold is started automatically (buffered hold).

### Wait steps vs Animation Locks

There are two ways to define waits, depending on whether you want to **enforce timing** or **ignore inputs** (spam-safe).

#### 1. Delay Gates (`wait:...`) — Enforce Timing

Use this when you want to learn *not* to press the next button too fast.
- The tracker **monitors** your inputs during the wait.
- If you press the *next* correct key too early, it is flagged as a **Timing Error** (marked "Early").
- **Hard request**: `wait_hard:0.2` will **fail/drop** the combo if you press early.
- **Soft request**: `wait:0.2` (default) will just warn you.

Syntax:
- `r, wait:0.5` (Press `r`, then wait 0.5s before next input)

#### 2. Animation Locks (`wait(key, duration)`) — Spam Safe

Use this when the game locks your character in an animation (e.g. ultimates).
- The tracker **ignores all inputs** during the wait.
- You can spam buttons safely; the combo **will not fail** even if you press the wrong keys or press early.

Syntax:
- `wait(r, 3.65s)` (Press `r`, then strictly ignore everything for 3.65s)

### Comparison

- **`r, wait:3.65`** = You *must* wait. Pressing the next key early is a mistake.
- **`wait(r, 3.65)`** = You *are forced* to wait. Pressing anything early is ignored (safe).

Durations accept:
- Seconds: `0.2`, `0.2s`
- Milliseconds: `200ms`

---

## Combo enders

Combo enders are inputs that will end the combo when pressed **only if that ability is off cooldown**.

**Notation:** `key:cooldown` e.g. `q:1.05s` (use explicit seconds with an `s` suffix).

The cooldown associated with the combo ender key is meant to mimic in-game ability cooldowns. When you press a button, that ability goes on cooldown and pressing it again will not cast that ability. Therefore that input will not end your combo if the ability is still on cooldown.

In the UI:
- `1:3s, 2:3s, 3:3s, e:2s, q:2s, r:2s, space:2s` (cooldown in seconds)
- Keys without a number have no cooldown: `lmb, rmb`

### Cooldown rule (how it works)

- When you press a key that is **the correct next key in the combo** and that key is a combo ender, the engine **starts that key’s cooldown timer**.
- Each ender key has its own cooldown (e.g. pressing `e` correctly starts `e`’s cooldown; pressing `q` later starts `q`’s cooldown).
- If you press a combo ender key while **that key is still on cooldown**, the press is **ignored** (the combo does not drop). This applies **everywhere**: during waits, during holds, or when the current step expects a different key.
- If you press a combo ender key when **that key is off cooldown**, the combo drops (unless the attempt hasn’t started yet).

Example:
- Combo: `f, e, q, lmb`
- Enders: `e:3s, q:3s`
- You press `f, e, q, e` quickly. The second `e` is wrong (expected `lmb`), but `e` is still on cooldown from when you correctly pressed `e` earlier → the combo does **not** drop.
- Same combo; you press `f, e, q`, wait until `e`’s cooldown has expired, then press `e` → the combo **drops** (ender `e` off cooldown).

### When enders drop the combo

- **Before the attempt starts**: stray keys are ignored.
- **During the attempt**: pressing an ender key drops the combo only if that key is **off cooldown**.
- **Key repeat (held key) does not drop**: only a **new key-down** can end the combo. If you keep a key held (e.g. hold `r` through a wait), repeat key events from the OS are ignored for ender logic—so the combo will not drop until you release and press that key again (a genuine new press).

Data is stored in `combos.json`.

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

---

## Architecture

ComboTracker is a small local web UI + Python backend. Roughly:

- **UI server + input hooks**: `ui_server.py`
  - Runs HTTP (`localhost:8080`) serving `static/`
  - Runs WebSocket (`localhost:8765`)
  - Starts global keyboard/mouse hooks via `pynput`
  - Calls into `ComboTrackerEngine`

- **Engine (orchestration + coordination)**: `combo_engine.py`
  - Owns persisted data (combos, enders, stats)
  - Coordinates the state machine and decides when to advance / fail / emit UI events

- **Step state machine (self-contained step logic)**: `states.py`
  - `PressState`, `HoldState`, `WaitState`, `SequenceState`, `GroupState`
  - Engine calls `process_press`, `process_release`, and `tick`

- **Analytics (read-only derived metrics)**: `combo_analytics.py`
  - Practical/theoretical APM, difficulty, min-time, fail summaries

- **UI view-model / payload building**: `combo_engine_ui.py`
  - Builds timeline payloads, editor payloads, status text, etc.

- **UI animation adapter**: `combo_engine_ui_adapter.py`
  - Tracks signatures and emits wait animation events only when needed

- **Persistence (schema + load/save/migrations)**: `persistence.py`
  - Owns JSON schema compatibility and sanitization

- **Commands (internal engine-only helpers)**: `_combo_commands.py`
  - Apply/save/delete/new/clear commands that mutate engine data (called under the engine lock)

- **Utilities extracted for maintainability**:
  - `step_introspection.py`: step labeling and key-introspection helpers
  - `format_utils.py`: duration parsing and ms formatting
  - `stats_recording.py`: success/fail stat mutation and fail event recording

- **Game-specific state / helpers**: `Game_Wuthering_Waves.py`
  - Stores WW-specific metadata (teams, target game, active character slot)
  - Provides WW ender policy used by the engine

### Running tests

```bash
python -m pytest tests\ -v
```

### Logging

Set `COMBOTRACKER_LOG_LEVEL` (e.g. `DEBUG`, `INFO`, `WARNING`) to control app logging.

