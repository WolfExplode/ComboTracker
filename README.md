## ComboTracker
![ComboTracker_AutoScroll](https://github.com/user-attachments/assets/b011d51d-d7c3-45e8-ae52-5e643f254e16)

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

## Documentation

Detailed docs live in [`documentation.md`](documentation.md):

- **Combo format**: [`documentation.md#combo-format`](documentation.md#combo-format)
- **Combo enders**: [`documentation.md#combo-enders`](documentation.md#combo-enders)
- **Difficulty + APM**: [`documentation.md#difficulty--apm`](documentation.md#difficulty--apm)
- **Troubleshooting**: [`documentation.md#troubleshooting`](documentation.md#troubleshooting)
- **Architecture / module map**: [`documentation.md#architecture`](documentation.md#architecture)

### Data

Combos and stats are stored locally in `combos.json`.

### Tests

```bash
python -m pytest tests\ -v
```
