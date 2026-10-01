// App shell: page navigation, the left rail and its status card, the header combo picker,
// the Combos list, History, Settings, toasts, and the backend message dispatch. Loads last.

const PAGES = {
    practice: { title: 'Practice', overline: 'Practice, edit and record' },
    teams: { title: 'Teams', overline: 'Your teams' },
    characters: { title: 'Characters', overline: 'Moves and rotations' },
    history: { title: 'History', overline: 'Every combo' },
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
    closeComboMenu();

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
// Combo list, header picker, Combos page list, History
// ---------------------------------------------------------------------------

function setComboList(names, active, overview) {
    appState.comboNames = Array.isArray(names) ? names.slice() : [];
    appState.activeCombo = active || '';
    if (Array.isArray(overview)) appState.overview = overview;
    renderComboViews();
}

function setOverview(overview) {
    if (!Array.isArray(overview)) return;
    appState.overview = overview;
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

function formatSeconds(ms) {
    const n = Number(ms);
    return Number.isFinite(n) && n > 0 ? `${(n / 1000).toFixed(2)}s` : '—';
}

function selectCombo(name) {
    if (!name) return;
    closeComboMenu();
    if (name !== appState.activeCombo) sendMessage('select_combo', { name });
}

function renderComboViews() {
    renderPicker();
    renderComboMenu();
    renderComboList();
    renderHistory();
    renderTeamCards();
}

function renderPicker() {
    const avatars = getEl('pickerAvatars');
    if (!avatars) return;
    avatars.replaceChildren();
    const hasCombo = !!appState.activeCombo;
    const slots = hasCombo ? appState.wwTeamSlots : ['', '', ''];
    slots.forEach(key => avatars.appendChild(makeAvatar(key, 'avatar-sm')));
    const team = hasCombo ? teamById(appState.wwTeamId) : null;
    getEl('pickerTeam').textContent = team ? team.name : (hasCombo ? 'No team' : `${appState.comboNames.length} combos`);
    getEl('pickerName').textContent = appState.activeCombo || 'Select a combo';
}

function renderComboMenu() {
    const menu = getEl('comboMenu');
    if (!menu) return;
    menu.replaceChildren();
    if (appState.comboNames.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'muted small menu-empty';
        empty.textContent = 'No combos yet. Make one on the Combos page.';
        menu.appendChild(empty);
        return;
    }
    appState.comboNames.forEach(name => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'menu-item' + (name === appState.activeCombo ? ' on' : '');
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', String(name === appState.activeCombo));
        const label = document.createElement('span');
        label.textContent = name;
        const team = document.createElement('span');
        team.className = 'muted small';
        team.textContent = comboTeamName(name);
        item.append(label, team);
        item.addEventListener('click', () => selectCombo(name));
        menu.appendChild(item);
    });
}

function openComboMenu() {
    const menu = getEl('comboMenu');
    if (!menu) return;
    menu.hidden = false;
    getEl('comboPicker').setAttribute('aria-expanded', 'true');
    menu.querySelector('.menu-item.on, .menu-item')?.focus();
}

function closeComboMenu() {
    const menu = getEl('comboMenu');
    if (!menu || menu.hidden) return;
    menu.hidden = true;
    getEl('comboPicker')?.setAttribute('aria-expanded', 'false');
}

getEl('comboPicker')?.addEventListener('click', () => {
    if (getEl('comboMenu').hidden) openComboMenu();
    else closeComboMenu();
});
document.addEventListener('click', (e) => {
    if (!e.target.closest('.picker')) closeComboMenu();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeComboMenu();
});

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
    names.forEach(name => {
        const row = overviewRow(name);
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'combo-item' + (name === appState.activeCombo ? ' on' : '');
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', String(name === appState.activeCombo));
        const text = document.createElement('span');
        text.className = 'combo-item-text';
        const label = document.createElement('span');
        label.textContent = name;
        const team = document.createElement('span');
        team.className = 'muted small';
        team.textContent = comboTeamName(name) || 'No team';
        text.append(label, team);
        const steps = document.createElement('span');
        steps.className = 'combo-item-steps';
        steps.textContent = row && row.steps ? `${row.steps} steps` : '';
        item.append(text, steps);
        item.addEventListener('click', () => selectCombo(name));
        list.appendChild(item);
    });
}
getEl('comboSearch')?.addEventListener('input', renderComboList);

function renderHistory() {
    const body = getEl('historyBody');
    if (!body) return;
    body.replaceChildren();
    const rows = appState.overview || [];
    rows.forEach(r => {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'history-row' + (r.name === appState.activeCombo ? ' on' : '');
        row.title = `Practice ${r.name}`;
        const cells = [
            [r.name, 'history-name'],
            [`${r.success} / ${r.fail}`, ''],
            [formatSeconds(r.best_ms), ''],
            [formatSeconds(r.avg_ms), 'hide-small'],
            [formatSeconds(r.target_ms), 'hide-small'],
        ];
        cells.forEach(([text, cls]) => {
            const span = document.createElement('span');
            if (cls) span.className = cls;
            span.textContent = text;
            row.appendChild(span);
        });
        row.addEventListener('click', () => {
            selectCombo(r.name);
            showPage('practice');
        });
        body.appendChild(row);
    });
    const note = getEl('historyNote');
    if (note) {
        const attempts = rows.reduce((n, r) => n + r.success + r.fail, 0);
        note.textContent = rows.length === 0 ? 'No combos saved yet.'
            : attempts === 0 ? 'No attempts logged yet, so every row is empty.'
            : 'Pick a combo to practice it.';
    }
}

// ---------------------------------------------------------------------------
// "Reads as": the saved combo's steps named the way the timeline names them
// ---------------------------------------------------------------------------

const READS_AS_PREVIEW = 12;

function renderReadsAs() {
    const wrap = getEl('readsAs');
    const items = getEl('readsAsItems');
    if (!wrap || !items) return;
    items.replaceChildren();
    const steps = appState.lastTimelineSteps || [];
    const label = createWwMoveLabeler(wwSlotNames());
    let slot = '1';
    const named = [];
    steps.forEach(step => {
        const key = wwStepKey(step);
        if (WW_SLOTS.includes(key)) slot = key;
        const text = label(step, slot);
        if (key && text) named.push([key, text]);
    });
    // Long rotations would bury the form, so show the opening moves and let the rest expand.
    const shown = appState.readsAsExpanded ? named : named.slice(0, READS_AS_PREVIEW);
    shown.forEach(([key, text]) => {
        const chip = document.createElement('span');
        chip.className = 'reads-as-item';
        const cap = document.createElement('span');
        cap.className = 'keycap keycap-sm';
        cap.textContent = key.toUpperCase();
        chip.append(cap, document.createTextNode(text));
        items.appendChild(chip);
    });
    if (named.length > READS_AS_PREVIEW) {
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'btn ghost reads-as-more';
        more.textContent = appState.readsAsExpanded ? 'Show less' : `+${named.length - READS_AS_PREVIEW} more`;
        more.addEventListener('click', () => {
            appState.readsAsExpanded = !appState.readsAsExpanded;
            renderReadsAs();
        });
        items.appendChild(more);
    }
    wrap.hidden = named.length === 0;
}

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
        renderReadsAs();
        flushPendingToast();
    },
    combo_list: (msg) => setComboList(msg.combos, msg.active, msg.overview),
    combo_data: (msg) => {
        setEditorFields(msg);
        renderPicker();
    },
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
    stat_update: (msg) => {
        updateStats(msg.stats);
        setOverview(msg.overview);
    },
    attempt_start: (msg) => addAttemptSeparator(msg.name, msg.attempt),
    timeline_update: (msg) => {
        updateTimeline(msg.steps, { focusLatest: !!msg.focus_latest });
        if (!isTimelineView) renderReadsAs();
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
