## ComboTracker for Wuthering Waves
![ComboTracker_AutoScroll](https://github.com/user-attachments/assets/b011d51d-d7c3-45e8-ae52-5e643f254e16)

A combo trainer built for **Wuthering Waves**. A small local web UI + Python backend listens to your keyboard/mouse and tracks whether you performed a rotation correctly, including **wait** and **hold** timing steps.

Every combo is tied to a team of three Resonators: the timeline shows each character's portrait on swaps (`1`/`2`/`3`) and their own Basic Attack, Resonance Skill (`e`), Echo (`q`) and Liberation (`r`) icons. A fresh install comes with WuWa default combo enders (swaps, `e`, `r`, jump) and auto-transcribe keys.

![Combo Input](https://github.com/user-attachments/assets/30466399-8db6-474d-8508-b90143143ab7)

## Info
This combo tracker is meant to track real time, timing data. not in game timing data.
For example, shorekeeper's Liberation is 2.9s in real time, but has a in game freeze time of 2.83s. Therefore it has a In game time of only 0.07s but the combo tracker will display as the full 2.9s

### Features
- **Practice combos**: see live status + a step timeline.
- **One-click characters**: in **Team & Characters**, type a name (e.g. `Phrolova`) and press **Add from wuthering.gg** to download their portrait and skill icons. **Refresh all icons** re-downloads icons for every saved character. Needs an internet connection.
- **Move names**: each step is labeled with the move it performs, like "Zani Basic 2", "Phoebe Intro" or "Tune Break", using your team slots. A leading `f` is the fight-start prompt; any later `f` is Tune Break. Toggle with **Move names** above the timeline.
- **Wait + hold steps**:
  - `wait` = minimum delay gate (pressing later is OK).
  - `hold` = finger commitment (must hold long enough).
- **Combo enders**: define which “wrong” inputs should drop the combo.
- **Stats**: success/fail, best time, hardest steps, fail reasons.
- **Difficulty scoring** (simple + tunable):
  - Practical APM (uses your expected execution time)
  - Theoretical max APM (uses fastest-possible time)
  - Difficulty out of 10 (keys + timing + simple timing-variation rule)

#### Easy to edit

https://github.com/user-attachments/assets/e32ac3e1-9fc6-40bf-ac14-98cb3224156d

---


## Demo Video:
<details>
  <summary>▶ Click to view in game demo video: </summary>
  <br>
  <a href="https://youtu.be/goTBFZBsBTo">
    <img src="https://img.youtube.com/vi/goTBFZBsBTo/maxresdefault.jpg" alt="Watch the video" style="width:100%;">
  </a>
</details>

## Getting started
If you downloaded a packaged release, keep the release folder intact and run
`ComboTracker.exe` from inside it. To run from source, use Python as described
below.

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
npm run serve
```

(`npm run serve` just runs `python ui_server.py`, which also works without Node.)

Then open the UI:
- `http://localhost:8737`

Notes:
- The backend also runs a WebSocket server at `ws://localhost:8765`.
- The app listens to global keyboard/mouse via `pynput` (you may need accessibility permissions on some OSes).

### Building the Windows release
You can package ComboTracker so others can run it without installing Python.

1. Install build tooling (once):
   ```bash
   python -m pip install -r requirements-build.txt
   ```
2. From the project root, build:
   ```bash
   python -m PyInstaller --noconfirm --clean ComboTracker.spec
   ```
3. Release artifact: **`dist/ComboTracker/`**. Keep the entire folder together and launch **`dist/ComboTracker/ComboTracker.exe`**. Windows will prompt for **Administrator** approval once per launch; this matches elevated games so global input capture works in-game.
4. Open **`http://localhost:8737`** in your browser. A console window stays open with the server URL (close it or press Ctrl+C to stop).
5. **`combos.json`** (saved combos and settings) is written **next to the .exe**. Keep it in the release folder if you want saved data to move with the app; otherwise a new `combos.json` is created on first run. Updating the app never touches it: the source download ships `combos.example.json`, which only seeds `combos.json` the first time.

**Without a local build:** every push runs the **CI** workflow on GitHub Actions, which runs the tests and builds the Windows release. Download it from the workflow run's **Artifacts** section (`ComboTracker-windows`).

**Publishing a release:** zip and ship the complete `dist/ComboTracker/` folder. The one-folder, non-UPX layout avoids the self-extracting executable pattern that commonly triggers antivirus heuristics. Build on a **normal (non-elevated)** terminal; PyInstaller may warn if you run it from an elevated shell. Code-signing the executable is recommended for public releases.

---



## OBS overlay

You can show the **Combo Steps** timeline in OBS as a separate overlay (e.g. for streaming or recording).

https://github.com/user-attachments/assets/0639590a-1de7-43b0-9ec8-b7eda59a833e

1. In the main UI, open the **Combo Steps** section and click **Copy Overlay URL**.
2. In OBS, add a **Browser Source**, paste the URL (e.g. `http://localhost:8737/?view=timeline`), and set the width/height you want. The overlay stays in sync with the app via WebSocket, and reconnects on its own if you restart ComboTracker.
3. **Open in new window** is also available if you prefer OBS **Window Capture** instead of Browser Source.

**Browser Source vs your browser:** The OBS Browser Source is a separate embedded browser. Toggles (Auto scroll, Images, Show fail count) are per-instance: use **Interact** on the Browser Source in OBS (right‑click the source → Interact) to open a window where you can click those controls for the overlay. Timeline content and progress sync for all clients; only the toggle state is local to each instance.

**Wide layout:** In timeline-only view (`?view=timeline`), the section stretches to fill the width of the Browser Source, so you can set a wide source in OBS and use the space.

For a walkthrough of common OBS Browser Source gotchas (separate instance, interact window, scrolling, sizing), see: [OBS Browser Source demo](https://youtu.be/lgGtxO_He4Y?t=145) (video, ~2:25).

you can set custom CSS to zoom in
`body {zoom : 150%;}`

## Live key overlay

A built-in, NohBoard-style key display for streams and recordings: WASD, Shift, Space, 1/2/3, Q/E/R/F and both mouse buttons light up as you press them. The swap keys show your selected team's character names and portraits, and macro playback shows up too.

- Open it with the keyboard button above **Combo Steps**, or add `http://localhost:8737/keys.html` as an OBS **Browser Source**. The background is transparent.
- It reconnects on its own if you restart ComboTracker.
- Only these keys are ever sent to the page, so other typing stays private.

## Characters: moves and team rotations

`http://localhost:8737/characters.html` (the book button above **Combo Steps**, or **Moves & rotations** under Team & Characters) lists every Wuthering Waves character with:

- **Moves**: each skill's text and damage multipliers at any skill level, with the key it's on (LMB, E, R...). From the [encore.moe](https://encore.moe) API.
- **Team rotations**: the community's team setups for that character with DPS, author, video and calc sheet, from [AntoCrasher's calc compilation](https://docs.google.com/spreadsheets/d/1mdl9J08N-0_j-U2zNP5OTHGBKprwmJEmz_Iy4IOJfPk/edit). **Combo** opens the move-by-move rotation transcript, converts it to tracker inputs (swaps use the team's slot order; no timings), and **Save as combo** adds it to your combos with the video attached.

Data is downloaded the first time you open a page and cached in `ww_library_cache/` next to `combos.json`, so it works offline afterwards. **Refresh data** downloads it again (cached copies older than a week refresh on their own).

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
