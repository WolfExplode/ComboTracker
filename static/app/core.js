// Combo Tracker main page, split by page. Load order (see index.html):
//   ../ws.js, ../ww_moves.js, core.js, timeline.js, practice.js, combos.js, teams.js, shell.js
// Every file is a classic script sharing one global scope; shell.js opens the backend connection last.

const getEl = (id) => document.getElementById(id);

/** Waits with duration <= this (ms) get class "short-wait" and a duller yellow border in CSS. Search for "short-wait" in style.css. */
const SHORT_WAIT_MS = 150;

/** Per-browser display preference; storage can be unavailable (private windows, OBS), so fall back quietly. */
function readStoredFlag(key, fallback) {
    try {
        const v = localStorage.getItem(key);
        return v === null ? fallback : v === '1';
    } catch (_) {
        return fallback;
    }
}

function writeStoredFlag(key, value) {
    try { localStorage.setItem(key, value ? '1' : '0'); } catch (_) { /* ignore */ }
}

/** Character names for team slots '1'/'2'/'3' of the selected team. */
function wwSlotNames() {
    const out = {};
    WW_SLOTS.forEach((slot, i) => {
        const key = (appState.wwTeamSlots[i] || '').toString();
        const ch = key ? appState.wwCharacters[key] : null;
        if (ch && ch.name) out[slot] = ch.name;
    });
    return out;
}

/** Backend connection; set by shell.js once every page's code has loaded. */
let tracker = null;

function sendMessage(type, payload = {}) {
    if (tracker) tracker.send(type, payload);
}

// Single app state (replaces scattered globals)
const appState = {
    stepDisplayMode: 'images',
    keyImages: {},
    lastTimelineSteps: null,
    lastFailByStep: {},
    showFailCount: false,
    collapseChainedPresses: true,
    showMoveNames: readStoredFlag('showMoveNames', true),
    autoScrollEnabled: false,
    stepEditMode: true,
    editStepsUndoStack: [],
    targetGame: 'wuthering_waves',
    wwAbilityImages: { "1": {}, "2": {}, "3": {} },
    wwSwapImages: { "1": "", "2": "", "3": "" },
    wwLmbImages: { "1": "", "2": "", "3": "" },
    wwDashImage: "",
    wwTeams: [],
    wwTeamId: '',
    wwTeamSlots: ["", "", ""],
    wwCharacters: {},
    wwCurrentChar: null,
    teamEdit: { id: '', name: '', slots: ['', '', ''], syncedSnap: '', pendingName: '' },
    pendingToast: '',
    batchQueue: [],
    isProcessingBatch: false,
    avgStepMsByPosition: [],
    // Combo list: one row per combo (team, steps) from the backend's "overview".
    comboNames: [],
    activeCombo: '',
    overview: [],
    connected: false,
};

// Two-click confirm pattern
function attachTwoClickConfirm(btn, opts) {
    let armed = false;
    let timer = null;
    const origText = btn.textContent;
    btn.addEventListener('click', () => {
        if (armed) {
            armed = false;
            clearTimeout(timer);
            btn.textContent = origText;
            if (opts.onConfirm) opts.onConfirm();
        } else {
            armed = true;
            btn.textContent = opts.confirmText || 'Confirm?';
            timer = setTimeout(() => {
                armed = false;
                btn.textContent = origText;
            }, 3000);
        }
    });
}

function scrollToBottom(el) {
    const target = (typeof el === 'string') ? getEl(el) : el;
    if (!target) return;
    target.scrollTop = target.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
