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
- **Auto transcription**: [Accurate auto transcription](#accurate-auto-transcription)
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

### Optional press steps

Prefix a key with `-` to make it optional: `-e` means you may skip that key. If you press the **next** step’s key instead, the optional step is skipped (no hit recorded, combo continues). If you press the optional key, it behaves as a normal press. Any other key (e.g. a wrong key or ender) still drops the combo as usual.

- Example: `f, wait:0.3s, -e, wait:0.80s, r, wait:1.8s` — you can do `f`, wait, then `r` (skipping `e`) or `f`, wait, `e`, wait, `r`. In the timeline, optional steps use a dashed border; skipped ones show a subtle “skipped” style, pressed ones show “was-pressed”.

### Any-order groups (order interchangeable)

Use brackets to indicate that multiple inputs can be pressed in **any order**, but all must be completed before the combo continues:

- Example: `3, r, wait:1.5, [q, e], 2, lmb, rmb`

This means after the wait gate you must press **both** `q` and `e`, in either order (`q` then `e` OR `e` then `q`).

You can also include an animation lock inside the group:
- Example: `e, 3, [wait(r, 1.5), q, e], 2`

This means press `r`, `q`, and `e` in any order — and if/when you press `r`, an animation lock of ≥ 1.5s is enforced before the group can finish (inputs during the lock are ignored).

### Hold steps

Syntax: `hold(e, 0.35)` (seconds).

The hold step is complete when you’ve held the key for at least the required duration. You do **not** need to release the hold key before pressing the next step: pressing the next key while still holding is accepted (e.g. `hold(lmb, 0.1s), 1` — hold LMB for 0.1s then press 1 without releasing LMB). If a wait step immediately precedes a hold and you are already holding that key when the wait ends, the hold is started automatically (buffered hold).

#### Hold + animation lock: `hold(key, hold_time, total_lock_time)`

Reasoning: sometimes abilities require you to hold for some minimum amount of time but also play a animation during which no inputs are read by the game.
For abilities that require a hold followed by an animation lock (no extra input; animation plays automatically after the hold), use the three-argument form:

- **Syntax:** `hold(lmb, 0.5s, 2s)` — hold for ≥ 0.5s, then a mandatory wait for the remaining time (2s − 0.5s = 1.5s). The third value is the **total** lock time from key down until you can act again.
- In the combo steps this appears as two tiles: **hold(e, 0.5s)** then **wait(e, 1.5s)**.
- The hold completes when the hold duration is satisfied — **even if you never release the key**. The mandatory wait then runs automatically (inputs ignored until it finishes).

#### Hold + inner sequence: `hold(key, hold_time, {body})`

Use when a key must remain held while you press one or more other buttons inside the hold.

- **Syntax:** `hold(lmb, 1.75s, {wait:0.15s, q, wait:0.15s, e, wait:0.35s})`
- The `hold_time` is the **minimum** duration the holder key must be held from key-down (same as plain `hold`).
- The `{body}` is a sequence of **presses** (`q`, `e`, …) and **`wait:` delay gates** that must be completed while the holder key is down. Waits in the body encode the relative timing from hold-start (useful for replay / macro playback).
- **Both** conditions must be true before the step advances: holder held for ≥ `hold_time` AND all body steps completed.
- Releasing the holder key **before** either condition is met **immediately fails** the combo (no forgiving hold).
- Body steps may only be plain presses, `spam(...)`, or `wait:` delays — no nested `hold(...)`, groups, or optional steps.
- **This notation is mutually exclusive with the anim-lock form:** `hold(key, a, b, {…})` (four arguments) is not valid.

### Repeated buffered input: `spam(key, duration)`

`spam(lmb, 4.4s)` repeatedly taps LMB over a 4.4-second span. Macro playback uses
the configured **Spam keys delay in ms** cadence and includes a tap at the end of
the span. Auto transcription emits this compact form for three or more consecutive
same-key taps whose start times are no more than 200ms apart. Two taps remain
expanded. This is a transcription convenience: it records the observed buffering
region and does not claim to know when the game accepted an input or ended an
animation lock.

`spam(...)` is also valid inside a hold body, for example
`hold(lmb, 1.5s, {wait:0.3s, spam(r, 0.8s)})`.

**Replay / macro:** the inner body steps are executed in order while the holder is held; any remaining `hold_time` after the body finishes is slept before the holder is released.

### Wait steps vs Animation Locks

There are two ways to define waits, depending on whether you want to **enforce timing** or **ignore inputs** (spam-safe).

#### Key + wait: one composite step (interruptible animation)

**A wait that immediately follows a key is attached to that key.** The combo treats it as **one action**: press the key, then an interruptible animation (the wait). In the parser and engine this is a single composite step (a sequence of key then wait).

- **Do not interrupt:** If you press another key before the wait completes, you **interrupt** the animation. That can drop the combo (or be marked "Early" for soft waits).
- **Syntax:** `e, wait:0.80s` = press `e`, then wait 0.80s; the wait is the animation tied to `e`. One composite step.

So: **a wait following a key denotes that the key has an associated interruptible animation. if you interrupt it, you drop the combo** (or get flagged early for soft).

#### 1. Delay Gates (`wait:...`) — Enforce Timing

Use this when you want to learn *not* to press the next button too fast.
- The tracker **monitors** your inputs during the wait.
- If you press the *next* correct key too early, it is flagged as a **Timing Error** (marked "Early").

Syntax (key + wait = one step):
- `r, wait:0.5` or `f, wait:0.23s` (Press the key, then wait; do not interrupt the wait)

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

**Soft enders** (`~key:cooldown`): Same as above, but **do not** end the combo when pressed during a **hold** step. Use for keys you might press while holding (e.g. movement/dodge). During a wait (delay gate) or when pressing the wrong key, soft enders still end the combo if off cooldown. Example: `1:1s, e:2s, ~space:2s, ~q:2s`.

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

## Accurate auto transcription

Auto transcription records timestamped key-down and key-up edges before creating
combo syntax. Distinct rapid clicks are preserved, keyboard repeat-down messages
are ignored, and each key is tracked independently so inputs performed inside a
hold can become `hold(key, duration, {body})`.

Waits and holds are formatted at millisecond precision. **Strip waits under** is
an optional post-recording simplification policy: omitted waits remain present in
the raw recording even when they are not included in the generated combo text.

Starting transcription records the real start-key press and release. Stopping
with Esc closes any still-held inputs at the Esc timestamp.

Every stopped recording is saved beside `combos.json`:

```text
logs/transcriptions/latest.json
logs/transcriptions/transcription-YYYYMMDD-HHMMSS-xxxxxxxx.json
```

The files contain the generated transcript, compiler settings, diagnostics, and
the complete raw event timeline. They are useful for investigating dropped or
misclassified inputs without re-recording the combo.

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

- **Input recording and transcription**: `transcriber.py`
  - Captures raw multi-key input edges under a lock
  - Compiles taps, waits, holds, and contained hold bodies after recording
  - Preserves raw evidence through `profiling/transcription_log.py`

- **Game-specific state / helpers**: `Game_Wuthering_Waves.py`
  - Stores WW-specific metadata (teams, target game, active character slot)
  - Provides WW ender policy used by the engine

### Macro timing profiler

Every completed macro run records a low-overhead timing profile and logs a one-line
summary at `INFO` level. The profile measures the part ComboTracker can observe:

- start request to playback-clock startup
- planned deadline to output-call start (dispatch lateness)
- time spent inside the `pynput` output call
- planned deadline to output-call completion
- error between planned and actual event intervals
- scheduler lateness using only the first dispatch at each deadline
- same-deadline serialization and output duration grouped by input

Repeated same-key spam keeps the configured cadence, but uses a pulse up to 5ms
shorter than that cadence. At the default 30ms interval this means 25ms pressed
and 5ms released, preventing each release from sharing a deadline with the next
press.

The complete most-recent trace is available from
`macro_player.last_profile()`. Pass `include_events=False` for only the summary.
Per-event records include key-down and key-up events. Set
`COMBOTRACKER_LOG_LEVEL=DEBUG` before starting the app to print those records;
the default `INFO` level prints only the compact run summary.

Each completed, stopped, or failed playback is also written beside `combos.json`:

```text
logs/macro_profiles/latest.json
logs/macro_profiles/macro-YYYYMMDD-HHMMSS-xxxxxxxx.json
```

`latest.json` is replaced after every run; timestamped files retain the complete
run history. These runtime diagnostics are ignored by Git.

These measurements end when the operating-system injection call returns. They do
not prove when the game consumes the input; measuring that would require telemetry
from the game or an external visual/audio response measurement.

### Running tests

```bash
python -m pytest tests\ -v
```

### Logging

Set `COMBOTRACKER_LOG_LEVEL` (e.g. `DEBUG`, `INFO`, `WARNING`) to control app logging.
