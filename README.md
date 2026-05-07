## ComboTracker
![ComboTracker_AutoScroll](https://github.com/user-attachments/assets/b011d51d-d7c3-45e8-ae52-5e643f254e16)

A small local web UI + Python backend that listens to your keyboard/mouse and tracks whether you performed a defined “combo” correctly, including **wait** and **hold** timing steps.

![Combo Input](https://github.com/user-attachments/assets/30466399-8db6-474d-8508-b90143143ab7)

## Info
This combo tracker is meant to track real time, timing data. not in game timing data.
For example, shorekeeper's Liberation is 2.9s in real time, but has a in game freeze time of 2.83s. Therefore it has a In game time of only 0.07s but the combo tracker will display as the full 2.9s


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

### Building a standalone .exe (Windows)
You can package ComboTracker as a single executable so others can run it without installing Python.

1. Install build tooling (once):
   ```bash
   python -m pip install -r requirements-build.txt
   ```
2. From the project root, build:
   ```bash
   pyinstaller ComboTracker.spec
   ```
3. Release artifact: **`dist/ComboTracker.exe`**. Double-click or run from a terminal. Windows will prompt for **Administrator** approval once per launch; this matches elevated games so global input capture works in-game.
4. Open **`http://localhost:8080`** in your browser. A console window stays open with the server URL (close it or press Ctrl+C to stop).
5. **`combos.json`** (saved combos and settings) is written **next to the .exe**. Move both together if you relocate the app; otherwise a new `combos.json` is created on first run.

**Publishing a release:** ship `dist/ComboTracker.exe` (optionally zip it for GitHub Releases). Build on a **normal (non-elevated)** terminal; PyInstaller may warn if you run it from an elevated shell.

---



## OBS overlay

You can show the **Combo Steps** timeline in OBS as a separate overlay (e.g. for streaming or recording).

https://github.com/user-attachments/assets/0639590a-1de7-43b0-9ec8-b7eda59a833e

1. In the main UI, open the **Combo Steps** section and click **Copy Overlay URL**.
2. In OBS, add a **Browser Source**, paste the URL (e.g. `http://localhost:8080/?view=timeline`), and set the width/height you want. The overlay stays in sync with the app via WebSocket.
3. **Open in new window** is also available if you prefer OBS **Window Capture** instead of Browser Source.

**Browser Source vs your browser:** The OBS Browser Source is a separate embedded browser. Toggles (Auto scroll, Images, Show fail count) are per-instance: use **Interact** on the Browser Source in OBS (right‑click the source → Interact) to open a window where you can click those controls for the overlay. Timeline content and progress sync for all clients; only the toggle state is local to each instance.

**Wide layout:** In timeline-only view (`?view=timeline`), the section stretches to fill the width of the Browser Source, so you can set a wide source in OBS and use the space.

For a walkthrough of common OBS Browser Source gotchas (separate instance, interact window, scrolling, sizing), see: [OBS Browser Source demo](https://youtu.be/lgGtxO_He4Y?t=145) (video, ~2:25).

you can set custom CSS to zoom in
`body {zoom : 150%;}`

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




## Troubleshooting
If keystrokes are not registering when in game, launch the code in elevated command prompt, with administrator.