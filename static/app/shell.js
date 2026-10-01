// App shell: page navigation, the left rail and its status card,
// the combo list, Settings, toasts, and the backend message dispatch. Loads last.

const PAGES = {
    practice: { title: 'Practice', overline: 'Practice, edit and record' },
    teams: { title: 'Teams', overline: 'Your teams' },
    characters: { title: 'Characters', overline: 'Moves and rotations' },
    settings: { title: 'Settings', overline: 'App' },
};
const PAGE_STORAGE_KEY = 'ctPage';
const isTimelineView = document.body.classList.contains('timeline-window-view');

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

function showPage(name, opts) {
    // Combos used to be its own page; old #combos links land on Practice.
    const page = PAGES[name] ? name : 'practice';
    document.querySelectorAll('.page').forEach(el => { el.hidden = el.dataset.page !== page; });
    document.querySelectorAll('.navbtn[data-page]').forEach(btn => {
        const on = btn.dataset.page === page;
        btn.classList.toggle('on', on);
        if (on) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
    });
    getEl('pageTitle').textContent = PAGES[page].title;
    getEl('pageOverline').textContent = PAGES[page].overline;
    closePickers();

    if (page === 'characters') {
        const frame = getEl('charactersFrame');
        if (frame && !frame.getAttribute('src')) frame.setAttribute('src', frame.dataset.src);
    }
    if (page === 'practice' && appState.autoScrollEnabled) applyAutoScroll();

    try { localStorage.setItem(PAGE_STORAGE_KEY, page); } catch (_) { /* ignore */ }
    if (!(opts && opts.fromHash) && location.hash.slice(1) !== page) {
        history.replaceState(null, '', '#' + page);
    }
}

document.addEventListener('click', (e) => {
    const nav = e.target.closest('.navbtn[data-page], [data-page-link]');
    if (!nav) return;
    e.preventDefault();
    showPage(nav.dataset.page || nav.dataset.pageLink);
});
window.addEventListener('hashchange', () => showPage(location.hash.slice(1), { fromHash: true }));

// Collapse the rail to icons (remembered per browser)
const appEl = getEl('app');
appEl.classList.toggle('compact', readStoredFlag('ctRailCompact', false));
getEl('railToggle')?.addEventListener('click', () => {
    const compact = !appEl.classList.contains('compact');
    appEl.classList.toggle('compact', compact);
    writeStoredFlag('ctRailCompact', compact);
});

// Drag the rail's right edge to resize it (remembered per browser); double-click resets.
const RAIL_WIDTH_KEY = 'ctRailWidth';
const RAIL_MIN = 200, RAIL_MAX = 480, RAIL_DEFAULT = 256;
function setRailWidth(px) {
    const w = Math.round(Math.min(RAIL_MAX, Math.max(RAIL_MIN, px)));
    appEl.style.setProperty('--rail-w', `${w}px`);
    return w;
}
try {
    const saved = Number(localStorage.getItem(RAIL_WIDTH_KEY));
    if (saved) setRailWidth(saved);
} catch (_) { /* ignore */ }
const railResizer = getEl('railResizer');
railResizer?.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    railResizer.setPointerCapture(e.pointerId);
    document.body.classList.add('resizing');
    let width = Math.round(document.querySelector('.rail').getBoundingClientRect().width);
    const move = (ev) => { width = setRailWidth(ev.clientX); };
    const up = () => {
        railResizer.removeEventListener('pointermove', move);
        document.body.classList.remove('resizing');
        try { localStorage.setItem(RAIL_WIDTH_KEY, String(width)); } catch (_) { /* ignore */ }
    };
    railResizer.addEventListener('pointermove', move);
    railResizer.addEventListener('pointerup', up, { once: true });
    railResizer.addEventListener('pointercancel', up, { once: true });
});
railResizer?.addEventListener('dblclick', () => {
    setRailWidth(RAIL_DEFAULT);
    try { localStorage.removeItem(RAIL_WIDTH_KEY); } catch (_) { /* ignore */ }
});

// The combo list lives in the rail; without a rail (phones) it moves to the top of the Practice page.
const comboListCard = getEl('comboListCard');
const railSpacer = document.querySelector('.rail-spacer');
const phoneLayout = window.matchMedia('(max-width: 900px)');
function placeComboList() {
    if (!comboListCard) return;
    if (phoneLayout.matches) getEl('practiceEditor')?.prepend(comboListCard);
    else railSpacer?.before(comboListCard);
}
phoneLayout.addEventListener('change', placeComboList);
placeComboList();

// Picking a combo from the rail on another page goes to Practice.
comboListCard?.addEventListener('click', (e) => {
    if (e.target.closest('.combo-item') && !getEl('practiceEditor')?.closest('.page:not([hidden])')) showPage('practice');
});

// The Edit combo panel under the timeline remembers whether it was open
const comboEditDetails = getEl('comboEditDetails');
if (comboEditDetails) {
    comboEditDetails.open = readStoredFlag('ctComboEditOpen', true);
    comboEditDetails.addEventListener('toggle', () => writeStoredFlag('ctComboEditOpen', comboEditDetails.open));
}

// ---------------------------------------------------------------------------
// Toast (replaces the old status text for saves)
// ---------------------------------------------------------------------------

let toastTimer = null;
function showToast(text) {
    const el = getEl('toast');
    if (!el || !text) return;
    el.textContent = text;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2200);
}

function flushPendingToast() {
    if (!appState.pendingToast) return;
    showToast(appState.pendingToast);
    appState.pendingToast = '';
}

// ---------------------------------------------------------------------------
// Rail status card: Connected / Recording / Macro armed
// ---------------------------------------------------------------------------

function renderRailStatus() {
    const recording = !!getEl('transcribeModeToggle')?.checked;
    const macro = !!getEl('macroModeToggle')?.checked;
    let kind, title, detail;
    if (!appState.connected) {
        [kind, title, detail] = ['danger', 'Disconnected', 'Reconnecting to the tracker'];
    } else if (recording) {
        const start = (getEl('transcribeStartKey')?.value || 'f').trim().toUpperCase();
        [kind, title, detail] = ['danger', 'Recording', `${start} starts, Esc stops`];
    } else if (macro) {
        const start = (getEl('macroStartKey')?.value || '').trim().toUpperCase() || 'Start key';
        const stop = (getEl('macroStopKey')?.value || '').trim().toUpperCase() || 'Esc';
        [kind, title, detail] = ['warning', 'Macro armed', `${start} starts, ${stop} stops`];
    } else {
        [kind, title, detail] = ['success', 'Connected', 'Listening for your inputs'];
    }
    const dot = getEl('railStatusDot');
    if (dot) dot.dataset.kind = kind;
    getEl('railStatusTitle').textContent = title;
    getEl('railStatusDetail').textContent = detail;
    getEl('railStatus').title = `${title}: ${detail}`;
}

['transcribeStartKey', 'macroStartKey', 'macroStopKey'].forEach(id => {
    getEl(id)?.addEventListener('input', renderRailStatus);
});

// ---------------------------------------------------------------------------
// Combo list (grouped by team, in the rail)
// ---------------------------------------------------------------------------

function setComboList(names, active, overview) {
    appState.comboNames = Array.isArray(names) ? names.slice() : [];
    appState.activeCombo = active || '';
    if (Array.isArray(overview)) appState.overview = overview;
    renderComboViews();
}

function teamById(id) {
    return appState.wwTeams.find(t => t.id === id) || null;
}

function overviewRow(name) {
    return appState.overview.find(r => r.name === name) || null;
}

function comboTeamName(name) {
    const row = overviewRow(name);
    return (row && teamById(row.team_id)?.name) || '';
}

function selectCombo(name) {
    if (!name) return;
    if (name !== appState.activeCombo) sendMessage('select_combo', { name });
}

function renderComboViews() {
    renderComboList();
    renderTeamPicker();
}

// Dropdown pickers (the Teams page's team picker): close on outside click or Esc.
function closePickers() {
    document.querySelectorAll('.picker-menu:not([hidden])').forEach(menu => {
        menu.hidden = true;
        menu.closest('.picker')?.querySelector('.picker-btn')?.setAttribute('aria-expanded', 'false');
    });
}
document.addEventListener('click', (e) => {
    if (!e.target.closest('.picker')) closePickers();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePickers();
});

// Combos grouped by team: each team row (portraits, name, count) opens to its combos.
// Teams with no combos are left out. The active combo's team opens when it becomes active; searching opens every match.
const NO_TEAM = '';
const openTeams = new Set();
let openedForCombo = null; // the active combo whose team was last opened automatically

function comboGroups(names) {
    const groups = new Map(); // team id -> combo names, in wwTeams order, "No team" last
    appState.wwTeams.forEach(t => groups.set(t.id, []));
    names.forEach(name => {
        const id = overviewRow(name)?.team_id || NO_TEAM;
        if (!groups.has(id)) groups.set(id, []);
        groups.get(id).push(name);
    });
    const none = groups.get(NO_TEAM);
    if (none) { groups.delete(NO_TEAM); groups.set(NO_TEAM, none); }
    return [...groups].filter(([, combos]) => combos.length > 0);
}

function comboItem(name) {
    const row = overviewRow(name);
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'combo-item' + (name === appState.activeCombo ? ' on' : '');
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', String(name === appState.activeCombo));
    const label = document.createElement('span');
    label.className = 'combo-item-text';
    label.textContent = name;
    const steps = document.createElement('span');
    steps.className = 'combo-item-steps';
    steps.textContent = row && row.steps ? `${row.steps} steps` : '';
    item.append(label, steps);
    item.title = steps.textContent ? `${name} (${steps.textContent})` : name;
    item.addEventListener('click', () => selectCombo(name));
    return item;
}

function renderComboList() {
    const list = getEl('comboList');
    if (!list) return;
    const q = (getEl('comboSearch')?.value || '').trim().toLowerCase();
    list.replaceChildren();
    const names = appState.comboNames.filter(n => !q || n.toLowerCase().includes(q) || comboTeamName(n).toLowerCase().includes(q));
    if (names.length === 0) {
        const p = document.createElement('p');
        p.className = 'muted small menu-empty';
        p.textContent = appState.comboNames.length ? 'No combos match.' : 'No combos yet. Use + to make one.';
        list.appendChild(p);
        return;
    }
    const activeTeam = overviewRow(appState.activeCombo)?.team_id ?? null;
    if (activeTeam !== null && openedForCombo !== appState.activeCombo) {
        openTeams.add(activeTeam);
        openedForCombo = appState.activeCombo;
    }

    comboGroups(names).forEach(([teamId, combos]) => {
        const team = teamById(teamId);
        const open = !!q || openTeams.has(teamId);
        const group = document.createElement('div');
        group.className = 'combo-group' + (open ? ' open' : '') + (teamId === activeTeam ? ' has-active' : '');

        const head = document.createElement('button');
        head.type = 'button';
        head.className = 'combo-group-head';
        head.setAttribute('aria-expanded', String(open));
        const left = document.createElement('span');
        left.className = 'team-item-left';
        const title = document.createElement('span');
        title.textContent = team ? team.name : 'No team';
        left.append(teamAvatars(team ? [team.slot1, team.slot2, team.slot3] : ['', '', '']), title);
        const count = document.createElement('span');
        count.className = 'combo-group-count';
        count.textContent = String(combos.length);
        head.append(left, count);
        head.title = `${title.textContent}: ${combos.length === 1 ? '1 combo' : `${combos.length} combos`}`;
        head.addEventListener('click', () => {
            if (openTeams.has(teamId)) openTeams.delete(teamId);
            else openTeams.add(teamId);
            renderComboList();
        });
        group.appendChild(head);

        if (open) {
            const body = document.createElement('div');
            body.className = 'combo-group-body';
            body.setAttribute('role', 'listbox');
            body.setAttribute('aria-label', `${title.textContent} combos`);
            combos.forEach(name => body.appendChild(comboItem(name)));
            group.appendChild(body);
        }
        list.appendChild(group);
    });
}
getEl('comboSearch')?.addEventListener('input', renderComboList);

// ---------------------------------------------------------------------------
// Settings: overlay URLs and data actions
// ---------------------------------------------------------------------------

function keysOverlayUrl() {
    return window.location.origin + '/keys.html';
}

getEl('timelineOverlayUrl').textContent = getTimelineUrl().replace(/^https?:\/\//, '').replace(/#.*$/, '');
getEl('keysOverlayUrl').textContent = keysOverlayUrl().replace(/^https?:\/\//, '');
getEl('keyOverlayLink')?.setAttribute('target', '_blank');

function copyText(text, what) {
    const done = () => showToast(`Copied the ${what} URL`);
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(done, () => showToast(text));
    } else {
        showToast(text);
    }
}
getEl('copyOverlayUrlBtn')?.addEventListener('click', () => copyText(getTimelineUrl().replace(/#.*$/, ''), 'timeline overlay'));
getEl('copyKeysUrlBtn')?.addEventListener('click', () => copyText(keysOverlayUrl(), 'key overlay'));

// Reload combos.json from disk (same folder as the app / exe)
getEl('loadJsonBtn')?.addEventListener('click', () => {
    if (!window.confirm('Reload all combos and settings from combos.json? Unsaved changes in the editor will be lost.')) {
        return;
    }
    sendMessage('load_combos_from_json', {});
});

// Clear ALL history (every combo)
const clearAllBtn = getEl('clearAllBtn');
if (clearAllBtn) {
    attachTwoClickConfirm(clearAllBtn, {
        confirmText: 'Click again to wipe every combo',
        onConfirm: () => sendMessage('clear_history_all'),
    });
}

// ---------------------------------------------------------------------------
// Backend messages, batched per animation frame (keeps UI smooth with lots of hits)
// ---------------------------------------------------------------------------

const MESSAGE_HANDLERS = {
    init: (msg) => {
        initializeUI(msg);
        renderComboViews();
        flushPendingToast();
    },
    combo_list: (msg) => setComboList(msg.combos, msg.active, msg.overview),
    combo_data: (msg) => setEditorFields(msg),
    min_time: (msg) => updateMinTime(msg.text),
    difficulty_update: (msg) => {
        updateDifficulty(msg.text);
        setDifficultyColor(getEl('difficultyDisplay'), msg.value);
    },
    user_difficulty_update: (msg) => {
        updateUserDifficulty(msg.text);
        setDifficultyColor(getEl('userDifficultyDisplay'), msg.value);
    },
    apm_update: (msg) => updateAPM(msg.text),
    apm_max_update: (msg) => updateAPMMax(msg.text),
    hold_begin: (msg) => startHoldAnimation(msg.required_ms),
    hold_end: () => stopHoldAnimation(),
    wait_begin: (msg) => startWaitAnimation(msg.required_ms),
    wait_end: () => stopWaitAnimation(),
    spam_begin: (msg) => startSpamAnimation(msg.required_ms),
    spam_end: () => stopSpamAnimation(),
    hit: (msg) => addResultRow(msg),
    combo_dropped: (msg) => {
        stopWaitAnimation();
        stopSpamAnimation();
        stopHoldAnimation();
        updateStatus(msg.input, msg.color || 'fail');
        addResultRow(msg);
    },
    clear_results: () => clearAttemptLog(),
    status: (msg) => {
        updateStatus(msg.text, msg.color);
        if (msg.color === 'fail') {
            appState.pendingToast = '';
            if (!getEl('app').querySelector('.page[data-page="practice"]:not([hidden])')) showToast(msg.text);
        }
    },
    alert_notice: (msg) => {
        const t = (msg.text || '').toString();
        if (t) window.alert(t);
    },
    stat_update: (msg) => updateStats(msg.stats),
    attempt_start: (msg) => addAttemptSeparator(msg.name, msg.attempt),
    timeline_update: (msg) => {
        updateTimeline(msg.steps, { focusLatest: !!msg.focus_latest });
    },
    fail_update: (msg) => {
        appState.lastFailByStep = msg.fail_by_step || {};
        refreshTimelineIfLoaded();
    },
    transcription_result: (msg) => {
        const inputsEl = getEl('comboInputs');
        if (inputsEl) {
            inputsEl.value = msg.inputs || '';
            if (typeof updateComboInputHighlight === 'function') updateComboInputHighlight();
        }
    },
};

function handleMessage(msg) {
    const fn = MESSAGE_HANDLERS[msg.type];
    if (fn) fn(msg);
}

function processBatch() {
    if (appState.batchQueue.length === 0) {
        appState.isProcessingBatch = false;
        return;
    }
    appState.isProcessingBatch = true;

    requestAnimationFrame(() => {
        const batch = appState.batchQueue.splice(0, appState.batchQueue.length);
        batch.forEach(handleMessage);
        appState.isProcessingBatch = false;
        if (appState.batchQueue.length > 0) processBatch();
    });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

let startPage = location.hash.slice(1);
if (!PAGES[startPage]) {
    try { startPage = localStorage.getItem(PAGE_STORAGE_KEY) || 'practice'; } catch (_) { startPage = 'practice'; }
}
showPage(isTimelineView ? 'practice' : startPage, { fromHash: isTimelineView });
renderComboViews();
renderRailStatus();

// Per-character move rules (chain length, hold behavior, follow-ups) for move names on the timeline.
fetch('data/ww_characters.json')
    .then(res => (res.ok ? res.json() : null))
    .then(doc => {
        if (!doc) return;
        setWwCharacterData(doc);
        refreshTimelineIfLoaded();
    })
    .catch(() => { /* move names fall back to generic rules */ });

// WuwaLAB's per-ability Concerto, so the timeline can tell when a character's bar is full.
fetch('data/ww_timings.json')
    .then(res => (res.ok ? res.json() : null))
    .then(doc => {
        if (!doc) return;
        setWwTimingData(doc);
        refreshTimelineIfLoaded();
    })
    .catch(() => { /* no Concerto tracking */ });

tracker = connectTracker({
    onOpen: () => {
        console.log('Connected to WuWa Combo Tracker backend');
        appState.connected = true;
        updateStatus('Status: Ready', 'neutral');
        renderRailStatus();
    },
    onClose: (retryMs) => {
        console.warn(`Backend connection lost; retrying in ${retryMs}ms`);
        appState.connected = false;
        updateStatus('Backend disconnected. Reconnecting…', 'fail');
        renderRailStatus();
    },
    onMessage: (msg) => {
        appState.batchQueue.push(msg);
        if (!appState.isProcessingBatch) processBatch();
    },
});
