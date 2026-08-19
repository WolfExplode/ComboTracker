// WebSocket connection
const ws = new WebSocket('ws://localhost:8765');

const getEl = (id) => document.getElementById(id);

/** Waits with duration <= this (ms) get class "short-wait" and a duller yellow border in CSS. Search for "short-wait" in style.css. */
const SHORT_WAIT_MS = 150;

// Timeline-only view for OBS (Browser Source or Window Capture)
if (new URLSearchParams(window.location.search).get('view') === 'timeline') {
    document.title = 'ComboTracker – Timeline';
    document.body.classList.add('timeline-window-view');
}

function getTimelineUrl() {
    const path = window.location.pathname || '/';
    return window.location.origin + path + (path.includes('?') ? '&' : '?') + 'view=timeline';
}

ws.onopen = () => {
    console.log('Connected to Combo Trainer backend');
};

ws.onclose = () => {
    console.error('Connection lost. Please restart the application.');
    updateStatus('ERROR: Backend disconnected', 'fail');
};

function sendMessage(type, payload = {}) {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type, ...payload }));
}

// Single app state (replaces scattered globals)
const appState = {
    stepDisplayMode: 'images',
    keyImages: {},
    lastTimelineSteps: null,
    lastFailByStep: {},
    showFailCount: false,
    collapseChainedPresses: true,
    autoScrollEnabled: false,
    stepEditMode: false,
    targetGame: 'generic',
    wwAbilityImages: { "1": {}, "2": {}, "3": {} },
    wwSwapImages: { "1": "", "2": "", "3": "" },
    wwLmbImages: { "1": "", "2": "", "3": "" },
    wwDashImage: "",
    wwTeams: [],
    wwTeamId: '',
    wwTeamSlots: ["", "", ""],
    wwCharacters: {},
    wwCurrentChar: null,
    batchQueue: [],
    isProcessingBatch: false,
    avgStepMsByPosition: [],
};

// UI Initialization
function initializeUI(data) {
    const selector = getEl('comboSelector');
    selector.innerHTML = '<option value="">— Select Combo —</option>';
    data.combos.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        selector.appendChild(opt);
    });
    if (data.active_combo) {
        selector.value = data.active_combo;
    }

    // Clear live tables on init (fresh UI state)
    getEl('resultsBody').innerHTML = '';

    if (data.fail_by_step) appState.lastFailByStep = data.fail_by_step;

    const isNewCombo = data.editor && (data.editor.name || '').toString().trim() === '';
    const preserved = isNewCombo ? {
        targetGame: appState.targetGame,
        wwTeamId: appState.wwTeamId,
        stepDisplayMode: appState.stepDisplayMode,
        noFailMode: !!getEl('noFailMode')?.checked,
        stepEditMode: !!getEl('stepEditToggle')?.checked,
        collapseChainedPresses: !!getEl('collapseChainsToggle')?.checked,
        keyImages: { ...appState.keyImages },
    } : null;

    if (data.editor) setEditorFields(data.editor);
    if (preserved) {
        appState.targetGame = preserved.targetGame;
        appState.wwTeamId = preserved.wwTeamId;
        appState.stepDisplayMode = preserved.stepDisplayMode;
        const gameSelect = getEl('targetGameSelect');
        if (gameSelect) gameSelect.value = appState.targetGame;
        const teamSelect = getEl('wwTeamSelect');
        if (teamSelect) teamSelect.value = appState.wwTeamId;
        const stepToggle = getEl('stepDisplayToggle');
        if (stepToggle) stepToggle.checked = (appState.stepDisplayMode === 'images');
        const noFailEl = getEl('noFailMode');
        if (noFailEl) noFailEl.checked = preserved.noFailMode;
        appState.stepEditMode = preserved.stepEditMode;
        const stepEditToggle = getEl('stepEditToggle');
        if (stepEditToggle) stepEditToggle.checked = preserved.stepEditMode;
        appState.collapseChainedPresses = preserved.collapseChainedPresses;
        const collapseChainsToggle = getEl('collapseChainsToggle');
        if (collapseChainsToggle) collapseChainsToggle.checked = preserved.collapseChainedPresses;
        appState.keyImages = preserved.keyImages;
        syncGameUIVisibility();
        refreshTimelineIfLoaded();
    }
    if (data.status) updateStatus(data.status.text, data.status.color);
    if (data.stats !== undefined) updateStats(data.stats);
    if (data.min_time !== undefined) updateMinTime(data.min_time);
    if (data.difficulty !== undefined) updateDifficulty(data.difficulty);
    if (data.user_difficulty !== undefined) updateUserDifficulty(data.user_difficulty);
    if (data.apm !== undefined) updateAPM(data.apm);
    if (data.apm_max !== undefined) updateAPMMax(data.apm_max);
    setDifficultyColor(getEl('difficultyDisplay'), data.difficulty_value);
    setDifficultyColor(getEl('userDifficultyDisplay'), data.user_difficulty_value);
    if (data.timeline) updateTimeline(data.timeline);

    const noFailEl = getEl('noFailMode');
    if (noFailEl && !preserved) noFailEl.checked = !!data.no_fail_mode;

    const transcribeValidKeysEl = getEl('transcribeValidKeys');
    if (transcribeValidKeysEl && data.transcribe_valid_keys !== undefined) transcribeValidKeysEl.value = data.transcribe_valid_keys || '';
    const transcribeStartKeyEl = getEl('transcribeStartKey');
    if (transcribeStartKeyEl && data.transcribe_start_key !== undefined) transcribeStartKeyEl.value = data.transcribe_start_key || '';

    const stripToggle = getEl('transcribeStripWaitToggle');
    if (stripToggle && data.transcribe_strip_wait_under_enabled !== undefined) {
        stripToggle.checked = !!data.transcribe_strip_wait_under_enabled;
    }
    const stripMs = getEl('transcribeStripWaitMs');
    if (stripMs && data.transcribe_strip_wait_under_ms !== undefined) {
        stripMs.value = data.transcribe_strip_wait_under_ms || '';
    }

    const macroStartKeyEl = getEl('macroStartKey');
    if (macroStartKeyEl && data.macro_start_key !== undefined) macroStartKeyEl.value = data.macro_start_key || '';
    const macroStopKeyEl = getEl('macroStopKey');
    if (macroStopKeyEl && data.macro_stop_key !== undefined) macroStopKeyEl.value = data.macro_stop_key || '';
    const macroSpamIntervalEl = getEl('macroSpamIntervalMs');
    if (macroSpamIntervalEl && data.macro_spam_interval_ms !== undefined) {
        macroSpamIntervalEl.value = String(data.macro_spam_interval_ms || '');
    }
}

function normalizeTargetGame(v) {
    const g = (v || '').toString().trim().toLowerCase();
    return (g === 'wuthering_waves') ? 'wuthering_waves' : 'generic';
}

function ensureWwAbilityShape(obj) {
    const out = { "1": {}, "2": {}, "3": {} };
    if (!obj || typeof obj !== 'object') return out;
    ['1', '2', '3'].forEach(c => {
        if (obj[c] && typeof obj[c] === 'object') {
            ['e', 'q', 'r'].forEach(a => {
                const url = (obj[c][a] || '').toString().trim();
                if (url) out[c][a] = url;
            });
        }
    });
    return out;
}

function ensureWwSlotShape(obj) {
    const out = { "1": "", "2": "", "3": "" };
    if (!obj || typeof obj !== 'object') return out;
    ['1', '2', '3'].forEach(k => {
        const url = (obj[k] || '').toString().trim();
        if (url) out[k] = url;
    });
    return out;
}

function syncGameUIVisibility() {
    const isWW = appState.targetGame === 'wuthering_waves';

    const wwPanels = getEl('wwPanels');
    if (wwPanels) wwPanels.classList.toggle('hidden', !isWW);

    const keyDetails = getEl('keyImagesDetails');
    if (keyDetails) keyDetails.classList.toggle('hidden', isWW);

    if (isWW) {
        renderWwTeamEditor();
        renderWwCharacterEditor();
        renderWwDashPreview();
    }

    // Re-render key images editor (generic mode only).
    renderKeyImagesEditor();
}

// ----- WW helper: image preview -----
function wwSetPreview(el, val) {
    const v = (val || '').toString().trim();
    if (!v) { el.innerHTML = ''; el.style.display = 'none'; return; }
    el.style.display = 'flex';
    el.style.alignItems = 'center';
    el.style.justifyContent = 'center';
    if (/^https?:\/\//i.test(v)) {
        el.innerHTML = `<img class="key-step-image" src="${escapeHtml(v)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:32px;height:32px;object-fit:contain;" />`;
    } else {
        el.innerHTML = `<span class="key-step-emoji">${escapeHtml(v)}</span>`;
    }
}

function wwSetSlotPortrait(el, char) {
    el.replaceChildren();
    const value = (char?.swap_image || '').toString().trim();
    el.classList.toggle('ww-slot-portrait-empty', !value);
    el.title = char?.name ? `${char.name} portrait` : 'Empty character slot';

    if (!value) return;
    if (/^https?:\/\//i.test(value)) {
        const img = document.createElement('img');
        img.src = value;
        img.alt = char?.name ? `${char.name} portrait` : 'Character portrait';
        img.loading = 'lazy';
        img.referrerPolicy = 'no-referrer';
        img.draggable = false;
        el.appendChild(img);
    } else {
        const emoji = document.createElement('span');
        emoji.textContent = value;
        el.appendChild(emoji);
    }
}

/** Team display names that still reference this character (by name_key, case-insensitive). */
function wwTeamNamesReferencingCharacter(nameKey) {
    const key = (nameKey || '').toString().trim().toLowerCase();
    if (!key) return [];
    const names = [];
    (appState.wwTeams || []).forEach(t => {
        if (!t || typeof t !== 'object') return;
        const slots = [t.slot1, t.slot2, t.slot3].map(s => (s || '').toString().trim().toLowerCase());
        if (slots.includes(key)) names.push((t.name || t.id || '').toString() || 'Team');
    });
    return names;
}

// ----- WW Dash preview -----
function renderWwDashPreview() {
    const input = getEl('wwDashImageInput');
    const preview = getEl('wwDashPreview');
    if (input && preview) {
        input.value = (appState.wwDashImage || '').toString();
        wwSetPreview(preview, input.value);
    }
}

// ----- WW Teams editor -----
function renderWwTeamEditor() {
    // Populate team select dropdown
    const teamSelect = getEl('wwTeamSelect');
    if (teamSelect) {
        const prev = teamSelect.value;
        teamSelect.innerHTML = '<option value="">— New Team —</option>';
        appState.wwTeams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.name;
            teamSelect.appendChild(opt);
        });
        teamSelect.value = prev || appState.wwTeamId;
    }

    // Build slot rows
    const slotsContainer = getEl('wwTeamSlots');
    if (!slotsContainer) return;
    slotsContainer.innerHTML = '';

    const charOptions = () => {
        const empty = '<option value="">— (empty) —</option>';
        const chars = Object.values(appState.wwCharacters)
            .filter(c => c && c.name)
            .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        return empty + chars.map(c => `<option value="${escapeHtml(c.name_key || c.name.toLowerCase())}">${escapeHtml(c.name)}</option>`).join('');
    };

    let dragSrcIdx = null;

    appState.wwTeamSlots.forEach((charKey, idx) => {
        const row = document.createElement('div');
        row.className = 'ww-slot-row';
        row.draggable = true;
        row.dataset.idx = idx;

        const handle = document.createElement('span');
        handle.className = 'ww-slot-handle';
        handle.textContent = '⠿';
        handle.title = 'Drag to reorder';

        const label = document.createElement('span');
        label.className = 'ww-slot-label';
        label.textContent = `Slot ${idx + 1}`;

        const portrait = document.createElement('div');
        portrait.className = 'ww-slot-portrait';
        wwSetSlotPortrait(portrait, charKey ? appState.wwCharacters[charKey] : null);

        const sel = document.createElement('select');
        sel.className = 'ww-slot-char-select';
        sel.innerHTML = charOptions();
        sel.value = charKey || '';

        sel.addEventListener('change', () => {
            appState.wwTeamSlots[idx] = sel.value;
            wwSetSlotPortrait(portrait, sel.value ? appState.wwCharacters[sel.value] : null);
            _resolveTeamImagesToState();
            refreshTimelineIfLoaded();
        });

        row.appendChild(handle);
        row.appendChild(label);
        row.appendChild(portrait);
        row.appendChild(sel);
        slotsContainer.appendChild(row);

        row.addEventListener('dragstart', e => {
            dragSrcIdx = idx;
            e.dataTransfer.effectAllowed = 'move';
            setTimeout(() => row.classList.add('ww-drag-active'), 0);
        });
        row.addEventListener('dragend', () => {
            dragSrcIdx = null;
            row.classList.remove('ww-drag-active');
            slotsContainer.querySelectorAll('.ww-slot-row').forEach(r => r.classList.remove('ww-drag-over'));
        });
        row.addEventListener('dragover', e => {
            if (dragSrcIdx !== null && dragSrcIdx !== idx) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                slotsContainer.querySelectorAll('.ww-slot-row').forEach(r => r.classList.remove('ww-drag-over'));
                row.classList.add('ww-drag-over');
            }
        });
        row.addEventListener('dragleave', e => {
            if (!row.contains(e.relatedTarget)) row.classList.remove('ww-drag-over');
        });
        row.addEventListener('drop', e => {
            e.preventDefault();
            const src = dragSrcIdx;
            if (src !== null && src !== idx) {
                const tmp = appState.wwTeamSlots[src];
                appState.wwTeamSlots[src] = appState.wwTeamSlots[idx];
                appState.wwTeamSlots[idx] = tmp;
                renderWwTeamEditor();
                _resolveTeamImagesToState();
                refreshTimelineIfLoaded();
            }
        });
    });
}

// Resolve current team slots → appState.wwSwapImages / wwLmbImages / wwAbilityImages for timeline
function _resolveTeamImagesToState() {
    const swap = { "1": "", "2": "", "3": "" };
    const lmb = { "1": "", "2": "", "3": "" };
    const ability = { "1": {}, "2": {}, "3": {} };
    appState.wwTeamSlots.forEach((charKey, idx) => {
        const sk = String(idx + 1);
        const char = charKey ? appState.wwCharacters[charKey] : null;
        if (!char) return;
        if (char.swap_image) swap[sk] = char.swap_image;
        if (char.lmb_image) lmb[sk] = char.lmb_image;
        if (char.ability_images && typeof char.ability_images === 'object') {
            ability[sk] = { ...char.ability_images };
        }
    });
    appState.wwSwapImages = swap;
    appState.wwLmbImages = lmb;
    appState.wwAbilityImages = ability;
}

// ----- WW Character editor -----
function _buildCharRow(labelText, inputValue, onInput) {
    const row = document.createElement('div');
    row.className = 'ww-ability-row';
    const label = document.createElement('span');
    label.className = 'ww-ability-label';
    label.textContent = labelText;
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'https://... or emoji';
    input.value = inputValue || '';
    const preview = document.createElement('div');
    preview.className = 'ww-ability-preview';
    wwSetPreview(preview, input.value);
    input.addEventListener('input', () => {
        onInput(input.value.trim());
        wwSetPreview(preview, input.value);
    });
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(preview);
    return row;
}

function renderWwCharacterPicker(chars) {
    const picker = getEl('wwCharPicker');
    if (!picker) return;
    picker.replaceChildren();

    const selectedKey = appState.wwCurrentChar || '';
    const selectedChar = selectedKey ? appState.wwCharacters[selectedKey] : null;
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'ww-char-picker-trigger';
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');

    const triggerPortrait = document.createElement('span');
    triggerPortrait.className = 'ww-char-picker-portrait';
    wwSetSlotPortrait(triggerPortrait, selectedChar);
    const triggerLabel = document.createElement('span');
    triggerLabel.className = 'ww-char-picker-label';
    triggerLabel.textContent = selectedChar?.name || '— New Character —';
    const chevron = document.createElement('span');
    chevron.className = 'ww-char-picker-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    trigger.append(triggerPortrait, triggerLabel, chevron);

    const menu = document.createElement('div');
    menu.className = 'ww-char-picker-menu';
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;

    const choose = (key) => {
        appState.wwCurrentChar = key || null;
        renderWwCharacterEditor();
    };
    const addOption = (char) => {
        const key = char ? (char.name_key || char.name.toLowerCase()) : '';
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'ww-char-picker-option';
        option.setAttribute('role', 'option');
        option.setAttribute('aria-selected', String(key === selectedKey));

        const portrait = document.createElement('span');
        portrait.className = 'ww-char-picker-portrait';
        wwSetSlotPortrait(portrait, char);
        const label = document.createElement('span');
        label.textContent = char?.name || '— New Character —';
        option.append(portrait, label);
        option.addEventListener('click', () => choose(key));
        menu.appendChild(option);
    };

    addOption(null);
    chars.forEach(addOption);
    trigger.addEventListener('click', () => {
        const isOpen = menu.hidden;
        menu.hidden = !isOpen;
        trigger.setAttribute('aria-expanded', String(isOpen));
    });
    trigger.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            menu.hidden = true;
            trigger.setAttribute('aria-expanded', 'false');
        }
    });

    picker.append(trigger, menu);
}

function renderWwCharacterEditor() {
    const chars = Object.values(appState.wwCharacters)
        .filter(c => c && c.name)
        .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    renderWwCharacterPicker(chars);

    const container = getEl('wwCharEditor');
    if (!container) return;
    container.innerHTML = '';

    const loaded = appState.wwCurrentChar ? appState.wwCharacters[appState.wwCurrentChar] : null;

    // Name row
    const nameRow = document.createElement('div');
    nameRow.className = 'ww-char-name-row';
    const nameLabel = document.createElement('span');
    nameLabel.className = 'ww-ability-label';
    nameLabel.textContent = 'Name :';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.id = 'wwCharNameInput';
    nameInput.placeholder = 'Character name';
    nameInput.value = (loaded && loaded.name) ? loaded.name : '';
    nameRow.appendChild(nameLabel);
    nameRow.appendChild(nameInput);
    container.appendChild(nameRow);

    // Character (slot swap) key icon, LMB, Q, E, R rows
    let swapVal = (loaded && loaded.swap_image) || '';
    let lmbVal = (loaded && loaded.lmb_image) || '';
    let abilVals = { q: '', e: '', r: '' };
    if (loaded && loaded.ability_images) {
        abilVals.q = loaded.ability_images.q || '';
        abilVals.e = loaded.ability_images.e || '';
        abilVals.r = loaded.ability_images.r || '';
    }

    container.appendChild(_buildCharRow('Character :', swapVal, v => { swapVal = v; }));
    container.appendChild(_buildCharRow('LMB :', lmbVal, v => { lmbVal = v; }));
    container.appendChild(_buildCharRow('Q :', abilVals.q, v => { abilVals.q = v; }));
    container.appendChild(_buildCharRow('E :', abilVals.e, v => { abilVals.e = v; }));
    container.appendChild(_buildCharRow('R :', abilVals.r, v => { abilVals.r = v; }));

    // Save & Delete buttons
    const btnRow = document.createElement('div');
    btnRow.className = 'ww-char-btn-row';

    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.textContent = 'Save';
    saveBtn.addEventListener('click', () => {
        const name = (nameInput.value || '').trim();
        if (!name) { updateStatus('Please enter a character name.', 'fail'); return; }
        const nameKey = name.toLowerCase();
        const isNew = appState.wwCurrentChar === null;
        const overwritingDifferent = !isNew && nameKey !== appState.wwCurrentChar && nameKey in appState.wwCharacters;
        const overwritingExisting = isNew && nameKey in appState.wwCharacters;
        const doSave = () => sendMessage('save_character', {
            name,
            swap_image: swapVal,
            lmb_image: lmbVal,
            ability_images: { q: abilVals.q, e: abilVals.e, r: abilVals.r },
        });
        if (overwritingDifferent || overwritingExisting) {
            if (!confirm(`"${name}" already exists. Overwrite?`)) return;
        }
        doSave();
    });

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'danger';
    deleteBtn.disabled = !loaded;
    deleteBtn.addEventListener('click', () => {
        if (!appState.wwCurrentChar) return;
        const charName = (loaded && loaded.name) || appState.wwCurrentChar;
        const blockingTeams = wwTeamNamesReferencingCharacter(appState.wwCurrentChar);
        if (blockingTeams.length) {
            alert(`Remove from these teams first: ${blockingTeams.join(', ')}`);
            return;
        }
        if (!confirm(`Delete character "${charName}"?`)) return;
        sendMessage('delete_character', { name: charName });
    });

    btnRow.appendChild(saveBtn);
    btnRow.appendChild(deleteBtn);
    container.appendChild(btnRow);
}

// Split a combo string into top-level tokens (respects (), {}, []).
function splitTopLevelTokens(str) {
    const out = [];
    let buf = '';
    let paren = 0, brace = 0, bracket = 0;
    for (const ch of (str || '')) {
        if (ch === '(') paren++;
        else if (ch === ')') paren = Math.max(0, paren - 1);
        else if (ch === '{') brace++;
        else if (ch === '}') brace = Math.max(0, brace - 1);
        else if (ch === '[') bracket++;
        else if (ch === ']') bracket = Math.max(0, bracket - 1);
        if (ch === ',' && paren === 0 && brace === 0 && bracket === 0) {
            const t = buf.trim();
            if (t) out.push(t);
            buf = '';
        } else {
            buf += ch;
        }
    }
    const t = buf.trim();
    if (t) out.push(t);
    return out;
}

// Extract the hold key from a hold(key, ...) token.
function _extractHoldKey(part) {
    const tl = part.toLowerCase();
    if (!tl.startsWith('hold(') || !tl.endsWith(')')) return null;
    const inner = part.slice('hold('.length, -1);
    const args = splitTopLevelTokens(inner);
    return args.length >= 1 ? args[0].trim().toLowerCase() : null;
}

// Extract keys from inputs text
function extractKeysFromInputs() {
    const txt = (getEl('comboInputs')?.value || '').toString();
    if (!txt.trim()) return [];

    const parts = splitTopLevelTokens(txt).map(x => x.toLowerCase());
    const keys = new Set();

    parts.forEach(part => {
        // hold(key, time) or hold(key, time, {body}) — new or existing form
        if (/^hold\s*\(/i.test(part)) {
            const hk = _extractHoldKey(part);
            if (hk) keys.add(hk);
            // Also extract keys from {body} if present
            const bodyM = part.match(/\{([^}]*)\}\s*\)$/);
            if (bodyM) {
                splitTopLevelTokens(bodyM[1]).forEach(bi => {
                    const bit = bi.trim().toLowerCase();
                    if (!/^wait[_a-z]*[:(\s]/i.test(bit)) keys.add(bit);
                });
            }
            return;
        }
        // key{time} shorthand
        let m = part.match(/^([^{]+)\{/);
        if (m) {
            keys.add(m[1].trim());
            return;
        }
        // wait(...) -> skip
        if (/^wait[_a-z]*[:(\s]/i.test(part)) return;
        // [group] -> extract items using top-level split
        m = part.match(/^\[(.+)\]$/s);
        if (m) {
            splitTopLevelTokens(m[1]).forEach(gi => {
                const gg = gi.trim().toLowerCase();
                if (/^hold\s*\(/i.test(gg)) {
                    const hk = _extractHoldKey(gg);
                    if (hk) keys.add(hk);
                    return;
                }
                let mm = gg.match(/^([^{]+)\{/);
                if (mm) { keys.add(mm[1].trim()); return; }
                mm = gg.match(/^wait\(\s*([^,]+)\s*,/i);
                if (mm) { keys.add(mm[1].trim()); return; }
                if (!/^wait[_a-z]*[:(\s]/i.test(gg)) keys.add(gg);
            });
            return;
        }
        // Plain key
        keys.add(part);
    });

    return Array.from(keys).sort();
}

function readKeyImagesFromUI() {
    const container = getEl('keyImagesEditor');
    if (!container) return;
    const inputs = container.querySelectorAll('input[data-key]');
    const next = {};
    inputs.forEach(inp => {
        const k = (inp.getAttribute('data-key') || '').trim().toLowerCase();
        const url = (inp.value || '').toString().trim();
        if (k && url) next[k] = url;
    });
    appState.keyImages = next;
}

function renderKeyImagesEditor() {
    // Generic mode only
    if (appState.targetGame === 'wuthering_waves') return;

    readKeyImagesFromUI();
    const container = getEl('keyImagesEditor');
    if (!container) return;
    container.innerHTML = '';

    const keys = extractKeysFromInputs();
    if (keys.length === 0) {
        container.innerHTML = '<div class="help-text">Enter inputs above to see key image fields.</div>';
        return;
    }

    keys.forEach(k => {
        const row = document.createElement('div');
        row.className = 'key-image-row';

        const label = document.createElement('span');
        label.className = 'key-image-label';
        label.textContent = k.toUpperCase();

        const input = document.createElement('input');
        input.type = 'text';
        input.setAttribute('data-key', k);
        input.placeholder = 'https://... or emoji';
        input.value = (appState.keyImages[k] || '').toString();

        const preview = document.createElement('div');
        preview.className = 'key-image-preview';
        const v = (appState.keyImages[k] || '').toString().trim();
        if (v) {
            preview.style.display = 'flex';
            if (/^https?:\/\//i.test(v)) {
                preview.innerHTML = `<img src="${escapeHtml(v)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:24px;height:24px;object-fit:contain;" />`;
            } else {
                preview.innerHTML = `<span>${escapeHtml(v)}</span>`;
            }
        } else {
            preview.style.display = 'none';
        }

        input.addEventListener('input', () => {
            const url = input.value.trim();
            if (url) {
                appState.keyImages[k] = url;
                preview.style.display = 'flex';
                if (/^https?:\/\//i.test(url)) {
                    preview.innerHTML = `<img src="${escapeHtml(url)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:24px;height:24px;object-fit:contain;" />`;
                } else {
                    preview.innerHTML = `<span>${escapeHtml(url)}</span>`;
                }
            } else {
                delete appState.keyImages[k];
                preview.style.display = 'none';
            }
        });

        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(preview);
        container.appendChild(row);
    });
}

// Demo video: normalize YouTube link to embed URL
function getYouTubeEmbedUrl(url) {
    const s = (url || '').toString().trim();
    if (!s) return null;
    try {
        // youtu.be/VIDEO_ID
        const short = s.match(/youtu\.be\/([a-zA-Z0-9_-]{10,})/);
        if (short) return 'https://www.youtube.com/embed/' + short[1];
        // youtube.com/watch?v=VIDEO_ID or youtube.com/embed/VIDEO_ID
        const u = new URL(s.startsWith('http') ? s : 'https://' + s);
        if (u.hostname.replace(/^www\./, '') === 'youtube.com') {
            const v = u.searchParams.get('v') || (u.pathname || '').split('/').pop();
            if (v && /^[a-zA-Z0-9_-]{10,}$/.test(v)) return 'https://www.youtube.com/embed/' + v;
        }
    } catch (_) {}
    return null;
}

function updateDemoVideoEmbed(url) {
    const wrap = getEl('demoVideoEmbedWrap');
    const iframe = getEl('demoVideoEmbed');
    if (!wrap || !iframe) return;
    const embedUrl = getYouTubeEmbedUrl(url);
    if (embedUrl) {
        iframe.src = embedUrl;
        wrap.classList.remove('hidden');
    } else {
        iframe.src = '';
        wrap.classList.add('hidden');
    }
}

// Editor fields update (from backend)
function setEditorFields(data) {
    getEl('comboName').value = data.name || '';
    const inputsEl = getEl('comboInputs');
    if (inputsEl) {
        inputsEl.value = data.inputs || '';
        if (typeof updateComboInputHighlight === 'function') updateComboInputHighlight();
    }
    getEl('comboEnders').value = data.enders || '';
    getEl('comboExpectedTime').value = data.expected_time || '';
    getEl('comboUserDifficulty').value = data.user_difficulty || '';

    const demoVideoEl = getEl('comboDemoVideo');
    if (demoVideoEl) {
        demoVideoEl.value = (data.demo_video || '').toString().trim();
        updateDemoVideoEmbed(demoVideoEl.value);
    }

    appState.stepDisplayMode = (data.step_display_mode || 'images').toString().trim().toLowerCase();
    if (!['icons', 'images'].includes(appState.stepDisplayMode)) appState.stepDisplayMode = 'images';
    const toggle = getEl('stepDisplayToggle');
    if (toggle) toggle.checked = (appState.stepDisplayMode === 'images');

    appState.keyImages = (typeof data.key_images === 'object' && data.key_images !== null) ? { ...data.key_images } : {};

    // Target game & WW data
    appState.targetGame = normalizeTargetGame(data.target_game || 'generic');
    const gameSelect = getEl('targetGameSelect');
    if (gameSelect) gameSelect.value = appState.targetGame;

    // WW character library
    const charsList = Array.isArray(data.ww_characters) ? data.ww_characters : [];
    appState.wwCharacters = {};
    charsList.forEach(c => {
        if (c && c.name_key) appState.wwCharacters[c.name_key] = c;
    });

    // WW teams list (now includes slot1/slot2/slot3)
    appState.wwTeams = Array.isArray(data.ww_teams) ? [...data.ww_teams] : [];

    // Selected team
    appState.wwTeamId = (data.ww_team_id || '').toString().trim();
    const teamNameEl = getEl('wwTeamName');
    if (teamNameEl) teamNameEl.value = (data.ww_team_name || '').toString();

    // Team slots
    const slots = data.ww_team_slots || {};
    appState.wwTeamSlots = [
        (slots.slot1 || '').toString(),
        (slots.slot2 || '').toString(),
        (slots.slot3 || '').toString(),
    ];

    // Global dash
    appState.wwDashImage = (data.ww_dash_image || data.ww_team_dash_image || '').toString().trim();

    // Resolved images for timeline rendering
    appState.wwSwapImages = ensureWwSlotShape(data.ww_team_swap_images);
    appState.wwLmbImages = ensureWwSlotShape(data.ww_team_lmb_images);
    appState.wwAbilityImages = ensureWwAbilityShape(data.ww_team_ability_images);

    // Keep the character editor selection across WW refreshes (e.g. changing team sends
    // combo_data). Only clear if that character no longer exists in the library.
    const prevCharKey = appState.wwCurrentChar;
    if (prevCharKey && !appState.wwCharacters[prevCharKey]) {
        appState.wwCurrentChar = null;
    }

    syncGameUIVisibility();
}

// Status display
function updateStatus(text, color) {
    const el = getEl('statusDisplay');
    if (!el) return;
    el.textContent = text || 'Status: Ready';
    el.className = 'status-' + (color || 'neutral');
}

// Stats
function updateStats(text) {
    const el = getEl('statsDisplay');
    if (el) el.textContent = text || 'Stats: —';
}

function updateMinTime(text) {
    const el = getEl('minTimeDisplay');
    if (el) el.textContent = text || 'Fastest possible: —';
}

function updateDifficulty(text) {
    const el = getEl('difficultyDisplay');
    if (el) el.textContent = text || 'Difficulty: —';
}

function updateUserDifficulty(text) {
    const el = getEl('userDifficultyDisplay');
    if (el) el.textContent = text || 'Your difficulty: —';
}

function updateAPM(text) {
    const el = getEl('apmDisplay');
    if (el) el.textContent = text || 'Practical APM: —';
}

function updateAPMMax(text) {
    const el = getEl('apmMaxDisplay');
    if (el) el.textContent = text || 'Theoretical max APM: —';
}

function setDifficultyColor(el, value) {
    if (!el) return;
    el.classList.remove('diff-easy', 'diff-med', 'diff-hard', 'diff-insane');
    const v = Number(value);
    if (!Number.isFinite(v)) return;
    if (v < 3) el.classList.add('diff-easy');
    else if (v < 6) el.classList.add('diff-med');
    else if (v < 8) el.classList.add('diff-hard');
    else el.classList.add('diff-insane');
}

// Attempt log
const LOG_ATTEMPTS_STORAGE_KEY = 'logAttemptsEnabled';
let logAttemptsEnabled = false;

function isLogAttemptsEnabled() {
    return logAttemptsEnabled;
}

function setLogAttemptsEnabled(enabled) {
    logAttemptsEnabled = !!enabled;
    try {
        localStorage.setItem(LOG_ATTEMPTS_STORAGE_KEY, logAttemptsEnabled ? '1' : '0');
    } catch (_) {
        // localStorage unavailable; runtime-only is fine.
    }
    const resultsTable = getEl('resultsTable');
    if (resultsTable) resultsTable.classList.toggle('hidden', !logAttemptsEnabled);
    renderAvgSplitsOnTimeline();
}

function clearAttemptLog() {
    getEl('resultsBody').innerHTML = '';
    appState.avgStepMsByPosition = [];
    renderAvgSplitsOnTimeline();
}

function escapeMarkdownCell(text) {
    return (text || '').toString().replace(/\|/g, '\\|');
}

function buildAttemptMarkdownTable(separatorRow) {
    if (!separatorRow) return '';
    const lines = [
        '| Input | Step Time (ms) | Total (ms) | Avg Step Time (ms) |',
        '| ----- | ---------- | ---------- | -------------- |',
    ];

    let cur = separatorRow.nextElementSibling;
    while (cur) {
        if (cur.classList.contains('separator')) break;
        if (cur.classList.contains('result-row')) {
            const cells = cur.querySelectorAll('span');
            if (cells.length >= 4) {
                const input = escapeMarkdownCell(cells[0].textContent?.trim() || '');
                const split = escapeMarkdownCell(cells[1].textContent?.trim() || '—');
                const total = escapeMarkdownCell(cells[2].textContent?.trim() || '—');
                const avgSplit = escapeMarkdownCell(cells[3].textContent?.trim() || '—');
                lines.push(`| ${input} | ${split} | ${total} | ${avgSplit} |`);
            }
        }
        cur = cur.nextElementSibling;
    }

    return lines.length > 2 ? lines.join('\n') : '';
}

async function copyAttemptToClipboard(separatorRow, copyBtn) {
    const markdown = buildAttemptMarkdownTable(separatorRow);
    if (!markdown) return;
    try {
        await navigator.clipboard.writeText(markdown);
    } catch (_) {
        const ta = document.createElement('textarea');
        ta.value = markdown;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
    if (copyBtn) {
        const originalTitle = copyBtn.title;
        copyBtn.title = 'Copied!';
        setTimeout(() => {
            copyBtn.title = originalTitle;
        }, 1200);
    }
}

function recalcAttemptAvgSplits() {
    const body = getEl('resultsBody');
    if (!body) return;

    // Group result rows by attempt (separated by .separator divs)
    const attempts = [];
    let current = null;
    let el = body.firstElementChild;
    while (el) {
        if (el.classList.contains('separator')) {
            current = [];
            attempts.push(current);
        } else if (el.classList.contains('result-row') && current !== null) {
            current.push(el);
        }
        el = el.nextElementSibling;
    }

    const maxLen = attempts.reduce((m, a) => Math.max(m, a.length), 0);
    const avgByPos = [];

    for (let pos = 0; pos < maxLen; pos++) {
        const splits = [];
        for (const attempt of attempts) {
            if (pos < attempt.length) {
                const cells = attempt[pos].querySelectorAll('span');
                const val = parseFloat(cells[1]?.textContent?.trim());
                if (Number.isFinite(val)) splits.push(val);
            }
        }
        const avg = splits.length > 0
            ? splits.reduce((s, v) => s + v, 0) / splits.length
            : null;
        avgByPos.push(avg);

        const avgText = avg !== null ? avg.toFixed(1) : '—';
        for (const attempt of attempts) {
            if (pos < attempt.length) {
                const avgCell = attempt[pos].querySelector('.result-avg-split');
                if (avgCell) avgCell.textContent = avgText;
            }
        }
    }

    appState.avgStepMsByPosition = avgByPos;
    renderAvgSplitsOnTimeline();
}

function renderAvgSplitsOnTimeline() {
    const container = getEl('comboTimeline');
    if (!container) return;

    // Clear all existing avg overlays
    container.querySelectorAll('.step-avg-split').forEach(e => e.remove());

    if (!appState.showFailCount) return;

    const avgs = appState.avgStepMsByPosition || [];
    if (avgs.length === 0) return;

    // Top-level tiles in DOM order (skip chain-dash connectors)
    const tiles = [...container.children].filter(
        e => !e.classList.contains('timeline-chain-dash')
    );

    const parseStepIndices = (tile, fallbackIdx) => {
        const raw = (tile.dataset.stepIndices || '').trim();
        if (!raw) return [fallbackIdx];
        const parsed = raw.split(',')
            .map(v => Number.parseInt(v, 10))
            .filter(v => Number.isFinite(v) && v >= 0);
        return parsed.length > 0 ? parsed : [fallbackIdx];
    };

    tiles.forEach((tile, idx) => {
        const indices = parseStepIndices(tile, idx);
        const values = indices
            .map(i => avgs[i])
            .filter(v => Number.isFinite(v));
        if (values.length === 0) return;
        const tileAvg = values.reduce((s, v) => s + v, 0);
        const span = document.createElement('span');
        span.className = 'step-avg-split';
        span.textContent = `avg step ${tileAvg.toFixed(1)}ms`;
        tile.appendChild(span);
    });
}

function deleteAttemptBlock(separatorRow) {
    if (!separatorRow || !separatorRow.parentElement) return;
    const toRemove = [separatorRow];
    let cur = separatorRow.nextElementSibling;
    while (cur) {
        if (cur.classList.contains('separator')) break;
        toRemove.push(cur);
        cur = cur.nextElementSibling;
    }
    toRemove.forEach((el) => el.remove());
    recalcAttemptAvgSplits();
}

function addAttemptSeparator(name, attempt) {
    if (!isLogAttemptsEnabled()) return;
    const body = getEl('resultsBody');
    if (!body) return;
    const row = document.createElement('div');
    row.className = 'result-row separator';

    const label = document.createElement('span');
    label.className = 'attempt-separator-label';
    label.textContent = `—— ${name} | Attempt ${attempt} ——`;

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'attempt-copy-btn subtle icon-btn';
    copyBtn.title = 'Copy this attempt as table';
    copyBtn.setAttribute('aria-label', 'Copy this attempt as table');
    copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    copyBtn.addEventListener('click', () => copyAttemptToClipboard(row, copyBtn));

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'attempt-delete-btn danger subtle icon-btn';
    deleteBtn.title = 'Delete this attempt from log';
    deleteBtn.setAttribute('aria-label', 'Delete this attempt from log');
    deleteBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"></polyline><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>';
    deleteBtn.addEventListener('click', () => deleteAttemptBlock(row));

    row.appendChild(label);
    row.appendChild(copyBtn);
    row.appendChild(deleteBtn);
    body.appendChild(row);
    scrollToBottom('resultsTable');
}

function addResultRow(data) {
    if (!isLogAttemptsEnabled()) return;
    const body = getEl('resultsBody');
    if (!body) return;
    const row = document.createElement('div');
    row.className = 'result-row';

    const stepMs = (data.step_ms != null) ? data.step_ms : data.split_ms;
    if (data.fail === true || stepMs === 'FAIL' || data.total_ms === 'FAIL') {
        row.classList.add('fail');
    } else {
        row.classList.add('success');
    }

    row.innerHTML = `
        <span>${escapeHtml(data.input || '')}</span>
        <span>${stepMs != null ? stepMs : '—'}</span>
        <span>${data.total_ms != null ? data.total_ms : '—'}</span>
        <span class="result-avg-split">—</span>
    `;

    body.appendChild(row);
    recalcAttemptAvgSplits();
    scrollToBottom('resultsTable');
}

// Hold / Wait animations (global UI state, not per-combo)
let holdAnim = { active: false, requiredMs: 0, startedAt: 0 };
let holdRafId = null;

let waitAnim = { active: false, requiredMs: 0, startedAt: 0 };
let waitRafId = null;

function stopHoldAnimation() {
    holdAnim.active = false;
    if (holdRafId !== null) {
        cancelAnimationFrame(holdRafId);
        holdRafId = null;
    }
}

function tickHoldAnimation() {
    if (!holdAnim.active) {
        holdRafId = null;
        return;
    }
    const stepEl = document.querySelector('.step.hold.active, .step.hold-with-body.active, .step.hold-with_body.active');
    if (!stepEl) {
        // Timeline might be re-rendering; try again next frame.
        holdRafId = requestAnimationFrame(tickHoldAnimation);
        return;
    }

    const elapsed = performance.now() - holdAnim.startedAt;
    const req = Math.max(1, holdAnim.requiredMs || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    stepEl.style.setProperty('--hold-pct', `${pct}%`);
    updateStatus(`Holding... ${Math.round(pct)}%`, 'recording');

    if (pct >= 100) {
        holdRafId = null;
        return;
    }
    holdRafId = requestAnimationFrame(tickHoldAnimation);
}

function startHoldAnimation(requiredMs) {
    stopHoldAnimation();
    holdAnim.active = true;
    holdAnim.requiredMs = Math.max(1, Number(requiredMs) || 1);
    holdAnim.startedAt = performance.now();
    holdRafId = requestAnimationFrame(tickHoldAnimation);
}

function stopWaitAnimation() {
    waitAnim.active = false;
    if (waitRafId !== null) {
        cancelAnimationFrame(waitRafId);
        waitRafId = null;
    }
}

function tickWaitAnimation() {
    if (!waitAnim.active) {
        waitRafId = null;
        return;
    }
    // Select active wait steps (including press_wait which has .press-wait class)
    const stepEl = document.querySelector('.step.wait.active, .step.press-wait.active');
    if (!stepEl) {
        waitRafId = requestAnimationFrame(tickWaitAnimation);
        return;
    }

    const elapsed = performance.now() - waitAnim.startedAt;
    const req = Math.max(1, Number(waitAnim.requiredMs) || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    stepEl.style.setProperty('--wait-pct', `${pct}%`);
    updateStatus(`Waiting... ${Math.round(pct)}%`, 'wait');

    if (pct >= 100) {
        waitRafId = null;
        return;
    }
    waitRafId = requestAnimationFrame(tickWaitAnimation);
}

function startWaitAnimation(requiredMs) {
    stopWaitAnimation();
    waitAnim.active = true;
    waitAnim.requiredMs = Math.max(1, Number(requiredMs) || 1);
    waitAnim.startedAt = performance.now();
    waitRafId = requestAnimationFrame(tickWaitAnimation);
}

// ---------------------------------------------------------------------------
// Step inline editing helpers
// ---------------------------------------------------------------------------

/**
 * JS port of the Python split_inputs: splits a comma-separated inputs string
 * into top-level tokens, respecting nested (, {, [ delimiters.
 */
function splitInputsTokens(str) {
    const out = [];
    let buf = '';
    let paren = 0, brace = 0, bracket = 0;
    for (const ch of (str || '')) {
        if (ch === '(') paren++;
        else if (ch === ')') paren = Math.max(0, paren - 1);
        else if (ch === '{') brace++;
        else if (ch === '}') brace = Math.max(0, brace - 1);
        else if (ch === '[') bracket++;
        else if (ch === ']') bracket = Math.max(0, bracket - 1);

        if (ch === ',' && paren === 0 && brace === 0 && bracket === 0) {
            const tok = buf.trim();
            if (tok) out.push(tok);
            buf = '';
            continue;
        }
        buf += ch;
    }
    const last = buf.trim();
    if (last) out.push(last);
    return out;
}

/**
 * Format a duration in milliseconds back to the shortest token string.
 * Uses "s" suffix if divisible cleanly, otherwise "ms".
 */
function formatDurationToken(ms) {
    const n = Number(ms);
    if (!Number.isFinite(n) || n <= 0) return `${ms}ms`;
    if (n % 1000 === 0) return `${n / 1000}s`;
    if (n % 100 === 0) return `${(n / 1000).toFixed(1)}s`;
    if (n % 10 === 0) return `${(n / 1000).toFixed(2)}s`;
    return `${Math.round(n)}ms`;
}

/**
 * Parse a user-entered duration string (e.g. "0.23s", "230ms", "230") to ms.
 * Returns null if unparseable.
 */
function parseDurationToMs(raw) {
    let s = (raw || '').trim().toLowerCase();
    if (!s) return null;
    // Strip trailing UI markers (e.g. hold-with-body status checkmark)
    s = s.replace(/\s*✓\s*$/u, '').trim();
    if (s.endsWith('ms')) {
        const v = parseFloat(s.slice(0, -2));
        return Number.isFinite(v) && v > 0 ? Math.round(v) : null;
    }
    if (s.endsWith('s')) {
        const v = parseFloat(s.slice(0, -1));
        return Number.isFinite(v) && v > 0 ? Math.round(v * 1000) : null;
    }
    const v = parseFloat(s);
    if (Number.isFinite(v) && v > 0) {
        // Treat as seconds if it looks fractional, ms otherwise.
        return s.includes('.') ? Math.round(v * 1000) : Math.round(v);
    }
    return null;
}

/**
 * Reconstruct source token(s) for a step after an inline field edit.
 * Returns an array of 1 or 2 token strings, or null on invalid input.
 *
 * `field` is either 'key' or 'duration'.
 * `newValue` is the raw string the user typed.
 * `s` is the step data dict from the backend.
 * `oldSourceToken` — original combo-input token when rebuilding complex holds (hold-with-body).
 */
function extractHoldWithBodyParts(oldSourceToken) {
    const t = (oldSourceToken || '').trim();
    if (!t.toLowerCase().startsWith('hold(') || !t.endsWith(')')) return null;
    const inner = t.slice(5, -1);
    const braceIdx = inner.indexOf('{');
    if (braceIdx === -1) return null;
    let head = inner.slice(0, braceIdx).replace(/,\s*$/, '').trim();
    const bodyPart = inner.slice(braceIdx).trim();
    const commaIdx = head.indexOf(',');
    if (commaIdx === -1) return null;
    const key = head.slice(0, commaIdx).trim();
    const durPart = head.slice(commaIdx + 1).trim();
    return { key, durPart, bodyPart };
}

function reconstructTokensForEdit(s, field, newValue, oldSourceToken) {
    const val = (newValue || '').trim().toLowerCase();
    if (!val) return null;

    if (s.type === 'press') {
        // Only 'key' editable
        return [val];
    }

    if (s.type === 'press_wait') {
        // Two source tokens: key token + wait:Xs token
        const key = field === 'key' ? val : (s.input || '').toLowerCase();
        if (!key) return null;
        const durMs = field === 'duration' ? parseDurationToMs(val) : s.duration;
        if (!durMs) return null;
        return [key, `wait:${formatDurationToken(durMs)}`];
    }

    if (s.type === 'hold') {
        // One token: hold(key, durMs)
        const key = field === 'key' ? val : (s.input || '').toLowerCase();
        if (!key) return null;
        const durMs = field === 'duration' ? parseDurationToMs(val) : s.duration;
        if (!durMs) return null;
        return [`hold(${key}, ${formatDurationToken(durMs)})`];
    }

    if (s.type === 'hold_with_body') {
        const parts = extractHoldWithBodyParts(oldSourceToken);
        if (!parts) return null;
        let key = parts.key.toLowerCase();
        let durMs = field === 'duration' ? parseDurationToMs(val) : Number(s.duration || 0);
        if (field === 'key') key = val;
        if (field === 'duration') {
            if (!durMs || !Number.isFinite(durMs)) return null;
        } else if (!durMs || !Number.isFinite(durMs)) {
            durMs = parseDurationToMs(parts.durPart);
            if (!durMs) return null;
        }
        const durTok = formatDurationToken(durMs);
        return [`hold(${key}, ${durTok}, ${parts.bodyPart})`];
    }

    if (s.type === 'wait' && s.mode === 'mandatory') {
        // One token: wait(key, durMs)
        const key = field === 'key' ? val : (s.wait_for || '').toLowerCase();
        if (!key) return null;
        const durMs = field === 'duration' ? parseDurationToMs(val) : s.duration;
        if (!durMs) return null;
        return [`wait(${key}, ${formatDurationToken(durMs)})`];
    }

    if (s.type === 'wait') {
        // Standalone soft/hard wait: one token wait:Xs
        const durMs = parseDurationToMs(val);
        if (!durMs) return null;
        return [`wait:${formatDurationToken(durMs)}`];
    }

    return null;
}

/**
 * Commit an inline step field edit:
 *  1. Locate source token(s) via runtime_source_token_indices (runtime index → source indices).
 *  2. Splice new token(s) into the inputs textarea.
 *  3. Send save_combo with the updated inputs string.
 */
function commitStepFieldEdit(runtimeIdx, s, field, newValue) {
    const inputsEl = getEl('comboInputs');
    if (!inputsEl) return false;

    const raw = inputsEl.value || '';
    const currentTokens = splitInputsTokens(raw);
    if (!currentTokens.length) return false;

    const srcMap = buildRuntimeToSourceMap(currentTokens);
    if (!srcMap || runtimeIdx >= srcMap.length) return false;

    const srcIndices = srcMap[runtimeIdx];
    if (!srcIndices || srcIndices.length === 0) return false;

    const minSrc = Math.min(...srcIndices);
    const oldSourceToken = currentTokens[minSrc];
    const newTokens = reconstructTokensForEdit(s, field, newValue, oldSourceToken);
    if (!newTokens) return false;

    // Splice: replace source token(s) at srcIndices with newTokens.
    const result = [];
    let i = 0;
    while (i < currentTokens.length) {
        if (srcIndices.includes(i)) {
            if (i === minSrc) {
                newTokens.forEach(t => result.push(t));
            }
            // skip the rest of the source group
        } else {
            result.push(currentTokens[i]);
        }
        i++;
    }
    if (result.length === 0) return false;

    const newInputs = result.join(', ');
    inputsEl.value = newInputs;
    if (typeof updateComboInputHighlight === 'function') updateComboInputHighlight();

    // Save via the same path as the Save/Update button.
    readKeyImagesFromUI();
    const toggle = getEl('stepDisplayToggle');
    sendMessage('save_combo', {
        name: (getEl('comboName')?.value || '').toString(),
        inputs: newInputs,
        enders: (getEl('comboEnders')?.value || '').toString(),
        expected_time: (getEl('comboExpectedTime')?.value || '').toString(),
        user_difficulty: (getEl('comboUserDifficulty')?.value || '').toString(),
        step_display_mode: toggle?.checked ? 'images' : 'icons',
        key_images: appState.keyImages,
        demo_video: (getEl('comboDemoVideo')?.value || '').toString().trim(),
        target_game: appState.targetGame,
        ww_team_id: appState.wwTeamId || '',
    });
    return true;
}

/**
 * JS port of runtime_source_token_indices_from_tokens from parser.py.
 * Returns an array where entry[runtimeIdx] = [srcTokenIdx, ...].
 */
function buildRuntimeToSourceMap(tokens) {
    const srcMap = [];
    let i = 0;
    while (i < tokens.length) {
        const tok = tokens[i].trim().toLowerCase();
        if (!tok) { i++; continue; }

        // press + following soft/hard wait -> one runtime SequenceNode (press_wait tile)
        if (!tok.startsWith('wait') && !tok.startsWith('hold(') && !tok.startsWith('[') && !tok.startsWith('{')) {
            // Could be a plain press followed by wait:Xs
            if (i + 1 < tokens.length) {
                const nxt = tokens[i + 1].trim().toLowerCase();
                if (nxt.startsWith('wait:')) {
                    srcMap.push([i, i + 1]);
                    i += 2;
                    continue;
                }
            }
        }

        // wait(key, t) -> two runtime steps (press + mandatory wait) from the same source token
        if (tok.startsWith('wait(') && tok.endsWith(')')) {
            srcMap.push([i]);
            srcMap.push([i]);
            i++;
            continue;
        }

        // hold(key, dur, {body}) -> one runtime step (matches parser HoldWithBodyNode).
        // hold(key, dur, total_ms) anim-lock -> two runtime steps from the same source token.
        if (tok.startsWith('hold(') && tok.endsWith(')')) {
            const raw = tokens[i].trim();
            const inner = raw.slice(5, -1);
            if (inner.indexOf('{') !== -1) {
                srcMap.push([i]);
                i++;
                continue;
            }
            const parts = inner.split(',').map(p => p.trim());
            if (parts.length >= 3) {
                srcMap.push([i]);
                srcMap.push([i]);
                i++;
                continue;
            }
        }

        srcMap.push([i]);
        i++;
    }
    return srcMap;
}

/**
 * Make a span inline-editable on double-click when edit mode is active.
 * `s` = step dict, `field` = 'key' | 'duration', `runtimeIdx` = first runtime index.
 */
function attachInlineEdit(span, s, field, runtimeIdx) {
    if (!appState.stepEditMode) return;
    span.classList.add('step-field-editable');
    span.title = 'Double-click to edit';

    span.addEventListener('dblclick', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (span.querySelector('input')) return; // already editing

        const original = span.textContent;
        // For duration fields strip the leading label text so user edits just the value
        let editValue = original;
        if (field === 'duration') {
            // "hold 300ms" -> "300ms", "Wait 500ms" -> "500ms", "230ms" -> "230ms"
            editValue = original.replace(/^(hold\s+|Wait\s+)/i, '').trim();
            editValue = editValue.replace(/\s*✓\s*$/u, '').trim();
        }

        const input = document.createElement('input');
        input.type = 'text';
        input.value = editValue;
        input.className = 'step-field-input';
        input.size = Math.max(4, editValue.length + 1);
        span.textContent = '';
        span.appendChild(input);
        input.focus();
        input.select();

        const commit = () => {
            const newVal = input.value.trim();
            span.textContent = original;
            span.classList.remove('step-field-editing');
            if (newVal && newVal !== editValue) {
                const ok = commitStepFieldEdit(runtimeIdx, s, field, newVal);
                if (!ok) {
                    span.title = 'Invalid value — double-click to try again';
                }
            }
        };

        const cancel = () => {
            span.textContent = original;
            span.classList.remove('step-field-editing');
        };

        span.classList.add('step-field-editing');
        input.addEventListener('blur', commit);
        input.addEventListener('keydown', (ke) => {
            if (ke.key === 'Enter') { ke.preventDefault(); input.blur(); }
            if (ke.key === 'Escape') { ke.preventDefault(); input.removeEventListener('blur', commit); cancel(); }
        });
    });
}

// Timeline rendering
function refreshTimelineIfLoaded() {
    if (appState.lastTimelineSteps) updateTimeline(appState.lastTimelineSteps);
}

function updateTimeline(steps, opts) {
    opts = opts || {};
    const scrollOpts = { focusLatest: !!opts.focusLatest };
    appState.lastTimelineSteps = steps;
    const container = getEl('comboTimeline');
    if (!container) return;
    container.innerHTML = '';

    const ctx = {
        failByStep: appState.lastFailByStep,
        stepDisplayMode: appState.stepDisplayMode,
        keyImages: appState.keyImages,
        targetGame: appState.targetGame,
        wwSwapImages: appState.wwSwapImages,
        wwDashImage: appState.wwDashImage,
        wwLmbImages: appState.wwLmbImages,
        wwAbilityImages: appState.wwAbilityImages,
        showFailCount: appState.showFailCount,
    };

    const viewport = getEl('comboTimelineViewport');
    const isAutoScroll = viewport?.classList.contains('auto-scroll-on');
    let baseStepWidthPx = 90;
    if (isAutoScroll) {
        const probe = document.createElement('div');
        probe.style.cssText = 'position:absolute;visibility:hidden;min-width:var(--auto-scroll-step-min-width)';
        document.body.appendChild(probe);
        const computedPx = getComputedStyle(probe).minWidth;
        document.body.removeChild(probe);
        const parsed = parseFloat(computedPx, 10);
        if (Number.isFinite(parsed) && parsed > 0) baseStepWidthPx = parsed;
    }
    const DURATION_WIDTH_DIVISOR = 350;
    const applyHoldWidth = (el, durationMs) => {
        const ms = Number(durationMs);
        const mult = (Number.isFinite(ms) && ms > 0) ? (ms / DURATION_WIDTH_DIVISOR) : 1;
        const w = Math.max(baseStepWidthPx, baseStepWidthPx * mult);
        el.style.minWidth = `${baseStepWidthPx}px`;
        el.style.width = `${w}px`;
    };
    const applyWaitWidth = (el, durationMs) => {
        const ms = Number(durationMs);
        const mult = (Number.isFinite(ms) && ms > 0) ? (ms / DURATION_WIDTH_DIVISOR) : 1;
        const w = Math.max(baseStepWidthPx, baseStepWidthPx * mult);
        el.style.minWidth = `${baseStepWidthPx}px`;
        el.style.width = `${w}px`;
    };
    const applyBaseWidth = (el) => {
        el.style.minWidth = `${baseStepWidthPx}px`;
        el.style.width = `${baseStepWidthPx}px`;
    };
    const addCornerKey = (el, key, s, runtimeIdx) => {
        if (ctx.stepDisplayMode !== 'images') return;
        const k = (key || '').toString().trim();
        if (!k) return;
        const span = document.createElement('span');
        span.className = 'corner-key';
        span.textContent = k.toUpperCase();
        el.appendChild(span);
        if (s && runtimeIdx != null) attachInlineEdit(span, s, 'key', runtimeIdx);
    };
    const parseStepIndices = (stepIndices) => {
        if (!Array.isArray(stepIndices)) return [];
        return stepIndices
            .map(v => Number.parseInt(v, 10))
            .filter(v => Number.isFinite(v) && v >= 0);
    };
    const attachStepDeleteControl = (el, stepIndices) => {
        const indices = parseStepIndices(stepIndices);
        if (!appState.stepEditMode || indices.length === 0) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'step-delete-btn';
        btn.title = 'Delete this step';
        btn.setAttribute('aria-label', 'Delete this step');
        btn.textContent = '🗑';
        btn.addEventListener('click', (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            sendMessage('delete_timeline_step', { step_indices: indices });
        });
        el.appendChild(btn);
    };

    // Drag-to-reorder: the first runtime index stored in step_indices is used as the drag handle identifier.
    const attachStepDragControl = (el, stepIndices) => {
        const indices = parseStepIndices(stepIndices);
        if (!appState.stepEditMode || indices.length === 0) return;
        const fromRuntimeIdx = indices[0];
        el.setAttribute('draggable', 'true');
        el.dataset.runtimeIdx = String(fromRuntimeIdx);

        el.addEventListener('dragstart', (ev) => {
            ev.dataTransfer.effectAllowed = 'move';
            ev.dataTransfer.setData('text/plain', String(fromRuntimeIdx));
            el.classList.add('step-dragging');
        });

        el.addEventListener('dragend', () => {
            el.classList.remove('step-dragging');
            container.querySelectorAll('.step-drag-over-before, .step-drag-over-after').forEach(t => {
                t.classList.remove('step-drag-over-before', 'step-drag-over-after');
            });
        });

        el.addEventListener('dragover', (ev) => {
            ev.preventDefault();
            ev.dataTransfer.dropEffect = 'move';
            const draggingIdx = ev.dataTransfer.getData('text/plain');
            if (draggingIdx === String(fromRuntimeIdx)) return;
            const rect = el.getBoundingClientRect();
            const midX = rect.left + rect.width / 2;
            container.querySelectorAll('.step-drag-over-before, .step-drag-over-after').forEach(t => {
                t.classList.remove('step-drag-over-before', 'step-drag-over-after');
            });
            if (ev.clientX < midX) {
                el.classList.add('step-drag-over-before');
            } else {
                el.classList.add('step-drag-over-after');
            }
        });

        el.addEventListener('dragleave', (ev) => {
            if (!el.contains(ev.relatedTarget)) {
                el.classList.remove('step-drag-over-before', 'step-drag-over-after');
            }
        });

        el.addEventListener('drop', (ev) => {
            ev.preventDefault();
            const draggedRuntimeIdx = Number.parseInt(ev.dataTransfer.getData('text/plain'), 10);
            el.classList.remove('step-drag-over-before', 'step-drag-over-after');
            if (!Number.isFinite(draggedRuntimeIdx) || draggedRuntimeIdx === fromRuntimeIdx) return;
            const rect = el.getBoundingClientRect();
            const midX = rect.left + rect.width / 2;
            if (ev.clientX < midX) {
                // Drop before this tile
                sendMessage('reorder_timeline_step', {
                    from_step_index: draggedRuntimeIdx,
                    before_step_index: fromRuntimeIdx,
                });
            } else {
                // Drop after this tile: find the next sibling's runtime index, or null to append.
                const allTiles = Array.from(container.querySelectorAll('[data-runtime-idx]'));
                const selfIdx = allTiles.indexOf(el);
                const nextTile = allTiles[selfIdx + 1] || null;
                const beforeIdx = nextTile ? Number.parseInt(nextTile.dataset.runtimeIdx, 10) : null;
                sendMessage('reorder_timeline_step', {
                    from_step_index: draggedRuntimeIdx,
                    before_step_index: Number.isFinite(beforeIdx) ? beforeIdx : null,
                });
            }
        });
    };

    function createGroupItemTile(it, characterId) {
        const el = document.createElement('div');
        el.className = 'step group-item';

        if (it.type === 'wait') {
            el.classList.add('wait');
            if (it.duration <= SHORT_WAIT_MS) el.classList.add('short-wait');
            const pct = (it.progress !== undefined && it.progress !== null) ? it.progress : (it.completed ? 100 : 0);
            el.style.setProperty('--wait-pct', `${pct}%`);
            applyWaitWidth(el, it.duration);
        } else if (it.type === 'press_wait') {
            el.classList.add('press-wait');
            if (it.duration <= SHORT_WAIT_MS) el.classList.add('short-wait');
            const pct = (it.progress !== undefined && it.progress !== null) ? it.progress : (it.completed ? 100 : 0);
            el.style.setProperty('--wait-pct', `${pct}%`);
            applyWaitWidth(el, it.duration);
        } else if (it.type === 'hold') {
            el.classList.add('hold');
            applyHoldWidth(el, it.duration);
            el.style.setProperty('--hold-pct', it.completed ? '100%' : '0%');
        } else if (isAutoScroll) {
            applyBaseWidth(el);
        }
        if (it.optional) el.classList.add('optional');
        if (it.optional && it.completed && !it.was_skipped) el.classList.add('was-pressed');

        if (it.active) el.classList.add('active');
        if (it.completed) el.classList.add('completed');

        appendStepContent(el, it, characterId, ctx);

        let keyForCorner = '';
        if (it.type === 'wait' && it.wait_for) keyForCorner = it.wait_for;
        else if (it.input) keyForCorner = it.input;
        if (keyForCorner) addCornerKey(el, keyForCorner);

        return el;
    }

    function renderGroupStep(s, idx, activeChar) {
        const indices = Array.isArray(s.step_indices) ? s.step_indices : [idx];
        const failCount = indices.reduce((n, i) => n + (ctx.failByStep[String(i)] || ctx.failByStep[i] || 0), 0);
        const showFailForStep = ctx.showFailCount && failCount > 0;

        const tile = document.createElement('div');
        tile.className = 'step-group';
        if (s.active) tile.classList.add('active');
        if (s.completed) tile.classList.add('completed');
        if (s.mark) {
            const m = String(s.mark).toLowerCase();
            if (m === 'ok') tile.classList.add('mark-ok');
            else if (m === 'early') tile.classList.add('mark-early');
            else if (m === 'missed') tile.classList.add('mark-missed');
            else if (m === 'wrong') tile.classList.add('mark-wrong');
        }
        if (showFailForStep) {
            tile.classList.add('mark-missed');
            const badge = document.createElement('span');
            badge.className = 'step-fail-count';
            badge.textContent = String(failCount);
            tile.appendChild(badge);
        }

        const items = document.createElement('div');
        items.className = 'step-group-items';
        let nextChar = activeChar;

        (s.items || []).forEach(it => {
            const itInp = (it.input || '').toString().toLowerCase();
            const itWait = (it.wait_for || '').toString().toLowerCase();

            if (it.type === 'sequence') {
                const seqEl = document.createElement('div');
                seqEl.className = 'step group-item group-item-sequence';
                if (it.active) seqEl.classList.add('active');
                if (it.completed) seqEl.classList.add('completed');

                const seqItems = document.createElement('div');
                seqItems.className = 'mini-sequence-items';
                (it.items || []).forEach(seqIt => {
                    const seqItInp = (seqIt.input || '').toString().toLowerCase();
                    const seqItWait = (seqIt.wait_for || '').toString().toLowerCase();
                    if (['1', '2', '3'].includes(seqItInp)) nextChar = seqItInp;
                    else if (['1', '2', '3'].includes(seqItWait)) nextChar = seqItWait;
                    seqItems.appendChild(createGroupItemTile(seqIt, nextChar));
                });
                seqEl.appendChild(seqItems);
                items.appendChild(seqEl);
            } else {
                if (['1', '2', '3'].includes(itInp)) nextChar = itInp;
                else if (['1', '2', '3'].includes(itWait)) nextChar = itWait;
                items.appendChild(createGroupItemTile(it, nextChar));
            }
        });
        tile.appendChild(items);
        attachStepDeleteControl(tile, s.step_indices);
        attachStepDragControl(tile, s.step_indices);
        return { tile, nextActiveChar: nextChar };
    }

    function renderSequenceStep(s, idx, activeChar) {
        const indices = Array.isArray(s.step_indices) ? s.step_indices : [idx];
        const failCount = indices.reduce((n, i) => n + (ctx.failByStep[String(i)] || ctx.failByStep[i] || 0), 0);
        const showFailForStep = ctx.showFailCount && failCount > 0;

        const tile = document.createElement('div');
        tile.className = 'step-sequence';
        if (s.active) tile.classList.add('active');
        if (s.completed) tile.classList.add('completed');
        if (showFailForStep) {
            tile.classList.add('mark-missed');
            const badge = document.createElement('span');
            badge.className = 'step-fail-count';
            badge.textContent = String(failCount);
            tile.appendChild(badge);
        }

        const items = document.createElement('div');
        items.className = 'sequence-items';
        let nextChar = activeChar;

        (s.items || []).forEach(it => {
            const itInp = (it.input || '').toString().toLowerCase();
            const itWait = (it.wait_for || '').toString().toLowerCase();
            if (['1', '2', '3'].includes(itInp)) nextChar = itInp;
            else if (['1', '2', '3'].includes(itWait)) nextChar = itWait;

            const itEl = document.createElement('div');
            itEl.className = 'step sequence-item';
            if (it.optional) itEl.classList.add('optional');
            if (it.optional && it.completed && !it.was_skipped) itEl.classList.add('was-pressed');
            if (it.active) itEl.classList.add('active');
            if (it.completed) itEl.classList.add('completed');
            appendStepContent(itEl, it, nextChar, ctx);
            items.appendChild(itEl);
        });
        tile.appendChild(items);
        attachStepDeleteControl(tile, s.step_indices);
        attachStepDragControl(tile, s.step_indices);
        return { tile, nextActiveChar: nextChar };
    }

    function renderNormalStep(s, idx, activeChar) {
        const indices = Array.isArray(s.step_indices) ? s.step_indices : [idx];
        const failCount = indices.reduce((n, i) => n + (ctx.failByStep[String(i)] || ctx.failByStep[i] || 0), 0);
        const showFailForStep = ctx.showFailCount && failCount > 0;

        const sInp = (s.input || '').toString().toLowerCase();
        const sWait = (s.wait_for || '').toString().toLowerCase();
        let nextChar = activeChar;
        if (['1', '2', '3'].includes(sInp)) nextChar = sInp;
        else if (['1', '2', '3'].includes(sWait)) nextChar = sWait;

        const tile = document.createElement('div');
        tile.className = 'step';
        if (s.active) tile.classList.add('active');
        if (s.completed) tile.classList.add('completed');
        if (s.mark === 'success') tile.classList.add('mark-ok');
        if (s.mark === 'fail' || s.mark === 'wrong') tile.classList.add('mark-wrong');
        if (s.mark === 'missed' || showFailForStep) tile.classList.add('mark-missed');
        if (s.mark === 'early') tile.classList.add('mark-early');
        if (showFailForStep) {
            const badge = document.createElement('span');
            badge.className = 'step-fail-count';
            badge.textContent = String(failCount);
            tile.appendChild(badge);
        }

        if (s.type) tile.classList.add(s.type.replace(/_/g, '-'));
        if (s.optional) tile.classList.add('optional');
        if (s.optional && s.completed && !s.was_skipped) tile.classList.add('was-pressed');
        let pct = (s.progress !== undefined) ? s.progress : (s.completed ? 100 : 0);
        if (s.type === 'wait' || s.type === 'press_wait') {
            tile.style.setProperty('--wait-pct', `${pct}%`);
            if (s.duration <= SHORT_WAIT_MS) tile.classList.add('short-wait');
            if (s.duration) applyWaitWidth(tile, s.duration);
        } else if (s.type === 'hold' || s.type === 'hold_with_body') {
            tile.style.setProperty('--hold-pct', `${pct}%`);
            if (s.duration) applyHoldWidth(tile, s.duration);
        } else if (isAutoScroll) {
            applyBaseWidth(tile);
        }

        const runtimeIdxNormal = Array.isArray(s.step_indices) ? s.step_indices[0] : idx;
        let keyForCorner = '';
        if (s.type === 'wait' && s.wait_for) keyForCorner = s.wait_for;
        else if (s.input) keyForCorner = s.input;
        if (keyForCorner && s.type !== 'hold_with_body') addCornerKey(tile, keyForCorner, s, runtimeIdxNormal);

        appendStepContent(tile, s, nextChar, ctx, runtimeIdxNormal);
        attachStepDeleteControl(tile, s.step_indices);
        attachStepDragControl(tile, s.step_indices);
        return { tile, nextActiveChar: nextChar };
    }

    function renderStep(s, idx, activeChar) {
        const out = (s.type === 'group')
            ? renderGroupStep(s, idx, activeChar)
            : (s.type === 'sequence')
                ? renderSequenceStep(s, idx, activeChar)
                : renderNormalStep(s, idx, activeChar);
        const indices = Array.isArray(s.step_indices) ? s.step_indices : [idx];
        out.tile.dataset.stepIndices = indices.join(',');
        return out;
    }

    const buildCollapsedLmbPressWaitStep = (chainSteps) => {
        const flattenedIndices = chainSteps.flatMap((it) =>
            Array.isArray(it.step_indices) ? it.step_indices : []
        );
        const totalDuration = chainSteps.reduce((sum, it) => sum + (Number(it.duration) || 0), 0);
        const progressTotal = chainSteps.reduce((sum, it) => {
            const pct = (it.progress !== undefined && it.progress !== null) ? Number(it.progress) : (it.completed ? 100 : 0);
            return sum + (Number.isFinite(pct) ? pct : 0);
        }, 0);
        const avgProgress = chainSteps.length > 0 ? (progressTotal / chainSteps.length) : 0;
        const anyMark = (name) => chainSteps.some((it) => (it.mark || '') === name);
        let collapsedMark = '';
        if (anyMark('wrong') || anyMark('fail')) collapsedMark = 'wrong';
        else if (anyMark('missed')) collapsedMark = 'missed';
        else if (anyMark('early')) collapsedMark = 'early';
        else if (anyMark('success')) collapsedMark = 'success';

        return {
            ...chainSteps[0],
            type: 'press_wait',
            input: 'lmb',
            duration: totalDuration,
            progress: avgProgress,
            chain_count: chainSteps.length,
            chain_collapsed: true,
            active: chainSteps.some((it) => !!it.active),
            completed: chainSteps.every((it) => !!it.completed),
            step_indices: flattenedIndices,
            mark: collapsedMark,
        };
    };

    if (!steps || steps.length === 0) {
        container.innerHTML = '<div class="help-text">No combo selected</div>';
        return;
    }

    const isLmbPressWait = (step) =>
        !!step
        && step.type === 'press_wait'
        && ((step.input || '').toString().toLowerCase() === 'lmb');

    const collapseChains = !!appState.collapseChainedPresses;
    let activeChar = '1';
    for (let idx = 0; idx < steps.length; idx += 1) {
        const s = steps[idx];

        if (collapseChains && isLmbPressWait(s)) {
            let chainEnd = idx + 1;
            while (chainEnd < steps.length && isLmbPressWait(steps[chainEnd])) chainEnd += 1;
            const chainLen = chainEnd - idx;
            if (chainLen > 1) {
                const collapsed = buildCollapsedLmbPressWaitStep(steps.slice(idx, chainEnd));
                const { tile, nextActiveChar } = renderStep(collapsed, idx, activeChar);
                tile.classList.add('chain-collapsed');
                activeChar = nextActiveChar;
                container.appendChild(tile);
                idx = chainEnd - 1;
                continue;
            }
        }

        const { tile, nextActiveChar } = renderStep(s, idx, activeChar);
        activeChar = nextActiveChar;
        container.appendChild(tile);
        if (!collapseChains && isLmbPressWait(s) && isLmbPressWait(steps[idx + 1])) {
            const dash = document.createElement('div');
            dash.className = 'timeline-chain-dash';
            dash.setAttribute('aria-hidden', 'true');
            container.appendChild(dash);
        }
    }

    renderAvgSplitsOnTimeline();

    if (viewport?.classList.contains('auto-scroll-on')) {
        requestAnimationFrame(() => {
            normalizeStepHeightsInAutoScroll(container);
            applyAutoScroll(scrollOpts);
        });
    } else if (appState.autoScrollEnabled) {
        requestAnimationFrame(() => applyAutoScroll(scrollOpts));
    }
}

function normalizeStepHeightsInAutoScroll(timelineEl) {
    if (!timelineEl) return;
    const stepTiles = timelineEl.querySelectorAll(':scope > .step, :scope .step.group-item');
    if (stepTiles.length === 0) return;
    stepTiles.forEach(el => { el.style.minHeight = ''; });
    const heights = Array.from(stepTiles).map(el => el.getBoundingClientRect().height);
    const maxH = Math.max(...heights, 0);
    if (maxH > 0) stepTiles.forEach(el => { el.style.minHeight = `${maxH}px`; });
}

function renderStepLabel(s) {
    if (s.type === 'sequence') {
        const parts = (s.items || []).map(it => renderStepLabel(it));
        return parts.length > 0 ? `Seq: ${parts.join('→')}` : 'Seq';
    }
    const inp = (s.input || '').toString().toUpperCase();
    const dur = s.duration || 0;
    if (s.type === 'wait') {
        if (s.mode === 'mandatory' && s.wait_for) {
            return `${s.wait_for.toUpperCase()} (${dur}ms)`;
        }
        return `Wait ${dur}ms`;
    }
    if (s.type === 'hold') return `Hold ${inp} ${dur}ms`;
    if (s.type === 'hold_with_body') return `Hold ${inp} ${dur}ms (+body)`;
    if (s.type === 'press_wait') return `${inp} + ${dur}ms`;
    return inp;
}

function getMouseIconSvg(type) {
    const t = type.toLowerCase();
    if (t === 'lmb') {
        return `<svg viewBox="0 0 64 64" role="img" focusable="false"><rect x="18" y="6" width="28" height="52" rx="14" ry="14" fill="none" stroke="currentColor" stroke-width="3"></rect><line x1="32" y1="6" x2="32" y2="26" stroke="currentColor" stroke-width="3" opacity="0.55"></line><path d="M18 20 C18 12, 24 6, 32 6 L32 26 L18 26 Z" fill="currentColor" opacity="0.35"></path></svg>`;
    }
    if (t === 'rmb') {
        return `<svg viewBox="0 0 64 64" role="img" focusable="false"><rect x="18" y="6" width="28" height="52" rx="14" ry="14" fill="none" stroke="currentColor" stroke-width="3"></rect><line x1="32" y1="6" x2="32" y2="26" stroke="currentColor" stroke-width="3" opacity="0.55"></line><path d="M46 20 C46 12, 40 6, 32 6 L32 26 L46 26 Z" fill="currentColor" opacity="0.35"></path></svg>`;
    }
    if (t === 'mmb') {
        return `<svg viewBox="0 0 64 64" role="img" focusable="false"><rect x="18" y="6" width="28" height="52" rx="14" ry="14" fill="none" stroke="currentColor" stroke-width="3"></rect><line x1="32" y1="6" x2="32" y2="26" stroke="currentColor" stroke-width="3" opacity="0.55"></line><rect x="28" y="10" width="8" height="12" rx="4" ry="4" fill="currentColor" opacity="0.35"></rect></svg>`;
    }
    return '';
}

function appendStepContent(parent, s, characterId, ctx, runtimeIdx) {
    const useImages = (ctx && ctx.stepDisplayMode === 'images') || !!getEl('stepDisplayToggle')?.checked;
    const inp = (s.input || '').toString().toLowerCase();
    const label = (s.input || '').toString().toUpperCase();
    const charId = characterId || '1';

    // Helper to decide content (icon or text when no image)
    const appendIconOrText = (key, fallbackText, target = parent) => {
        const svg = getMouseIconSvg(key);
        if (svg) {
            const icon = document.createElement('span');
            icon.className = 'mouse-icon';
            icon.innerHTML = svg;
            target.appendChild(icon);
            return;
        }
        const span = document.createElement('span');
        span.className = 'step-primary';
        if (key === 'space') {
            span.textContent = '⎵';
            span.classList.add('space-icon');
        } else {
            span.textContent = fallbackText;
        }
        target.appendChild(span);
    };

    // Resolve image URL based on current game/mode (WW vs generic); returns null when no image.
    const resolveImage = (key) => {
        if (!useImages) return null;
        if (ctx && ctx.targetGame === 'wuthering_waves') {
            return getWwImage(key, charId, ctx);
        }
        return (ctx && ctx.keyImages[key]) || null;
    };

    // Append primary content: image if available, else icon/text.
    const appendPrimary = (key, fallbackText, target = parent) => {
        const imgUrl = resolveImage(key);
        if (imgUrl) {
            target.appendChild(createImageElement(imgUrl));
        } else {
            appendIconOrText(key, fallbackText, target);
        }
    };

    // Append duration/secondary text; attach inline edit when runtimeIdx is provided.
    const appendDuration = (text, target = parent) => {
        const dur = document.createElement('span');
        dur.className = 'step-secondary';
        dur.textContent = text;
        target.appendChild(dur);
        if (runtimeIdx != null) attachInlineEdit(dur, s, 'duration', runtimeIdx);
    };

    // Single unified logic — no duplication across WW / generic / icons.
    if (s.type === 'wait' && s.mode === 'mandatory' && s.wait_for) {
        const key = s.wait_for.toLowerCase();
        appendPrimary(key, key.toUpperCase());
        appendDuration(`${s.duration}ms`);
    } else if (s.type === 'hold_with_body') {
        const shell = document.createElement('div');
        shell.className = 'hold-body-layout';

        const anchor = document.createElement('div');
        anchor.className = 'hold-anchor';
        appendPrimary(inp, label, anchor);

        const keyTag = document.createElement('span');
        keyTag.className = 'corner-key hold-anchor-key';
        keyTag.textContent = label;
        anchor.appendChild(keyTag);
        if (runtimeIdx != null) attachInlineEdit(keyTag, s, 'key', runtimeIdx);

        const right = document.createElement('div');
        right.className = 'hold-content';
        const bodyDoneLabel = s.body_done ? ' ✓' : '';
        appendDuration(`hold ${s.duration}ms${bodyDoneLabel}`, right);

        const bodyRow = document.createElement('div');
        bodyRow.className = 'hold-body-row';
        const bodyItems = (s.body && Array.isArray(s.body.items)) ? s.body.items : [];
        bodyItems.forEach((it) => {
            const chip = document.createElement('span');
            const isActive = !!(it && it.active);
            const isCompleted = !!(it && it.completed);
            const itemType = (it && it.type ? String(it.type) : '').toLowerCase();
            if (itemType === 'wait') {
                chip.className = 'hold-body-chip wait';
                chip.textContent = `${Number(it.duration || 0)}ms`;
                const waitPct = Number(it.progress);
                if (Number.isFinite(waitPct)) {
                    chip.style.setProperty('--chip-wait-pct', `${Math.max(0, Math.min(100, waitPct))}%`);
                }
            } else {
                chip.className = 'hold-body-chip press';
                const key = (it && it.input ? it.input : '').toString().toLowerCase();
                if (key) {
                    const imgUrl = resolveImage(key);
                    if (imgUrl) chip.appendChild(createImageElement(imgUrl));
                    else appendIconOrText(key, key.toUpperCase(), chip);
                } else {
                    chip.textContent = '?';
                }
                // Body sequences often arrive as press_wait (collapsed press + following wait).
                // Render the wait timing inline so all hold-body waits stay visible.
                if (itemType === 'press_wait') {
                    chip.classList.add('press-wait');
                    const inlineWait = document.createElement('span');
                    inlineWait.className = 'hold-body-inline-wait';
                    inlineWait.textContent = `${Number(it.duration || 0)}ms`;
                    const waitPct = Number(it.progress);
                    if (Number.isFinite(waitPct)) {
                        inlineWait.style.setProperty('--chip-wait-pct', `${Math.max(0, Math.min(100, waitPct))}%`);
                    }
                    chip.appendChild(inlineWait);
                }
            }
            if (isActive) chip.classList.add('active');
            if (isCompleted) chip.classList.add('completed');
            bodyRow.appendChild(chip);
        });
        right.appendChild(bodyRow);

        shell.appendChild(anchor);
        shell.appendChild(right);
        parent.appendChild(shell);
    } else if (s.type === 'hold') {
        appendPrimary(inp, label);
        appendDuration(`hold ${s.duration}ms`);
    } else if (s.type === 'press_wait') {
        const chainCount = Number(s.chain_count || 0);
        if (chainCount > 1) {
            const primaryRow = document.createElement('div');
            primaryRow.className = 'step-chain-primary';
            appendPrimary(inp, label, primaryRow);
            const count = document.createElement('span');
            count.className = 'step-chain-count';
            count.textContent = `x${chainCount}`;
            primaryRow.appendChild(count);
            parent.appendChild(primaryRow);
        } else {
            appendPrimary(inp, label);
        }
        appendDuration(`${s.duration}ms`);
    } else if (s.type === 'wait') {
        appendDuration(`Wait ${s.duration}ms`);
    } else {
        appendPrimary(inp, label);
    }
}

function getWwImage(key, characterId, ctx) {
    if (!ctx) return null;
    const k = key.toLowerCase();
    const cid = characterId || '1';

    // Check if it's a swap key (1/2/3) - Return the swap icon for that character regardless of who is active
    if (['1', '2', '3'].includes(k)) {
        return ctx.wwSwapImages[k] || null;
    }
    // Check if it's RMB (dash) - shared dash image
    if (k === 'rmb') {
        return ctx.wwDashImage || null;
    }

    // Check if it's LMB - use active character
    if (k === 'lmb') {
        return ctx.wwLmbImages[cid] || null;
    }

    // Check if it's an ability (e/q/r) - use active character
    if (['e', 'q', 'r'].includes(k)) {
        if (ctx.wwAbilityImages[cid] && ctx.wwAbilityImages[cid][k]) {
            return ctx.wwAbilityImages[cid][k];
        }
        // Fallback: search all characters if not found for specific one (legacy behavior, optional)
        for (const c of ['1', '2', '3']) {
            if (ctx.wwAbilityImages[c] && ctx.wwAbilityImages[c][k]) {
                return ctx.wwAbilityImages[c][k];
            }
        }
    }
    return null;
}

function createImageElement(url) {
    const img = document.createElement('span');
    img.className = 'key-img-wrap'; // Matches CSS .key-img-wrap
    if (/^https?:\/\//i.test(url)) {
        img.innerHTML = `<img class="key-step-image" src="${escapeHtml(url)}" alt="" loading="lazy" referrerpolicy="no-referrer" />`;
    } else {
        img.innerHTML = `<span class="step-emoji">${escapeHtml(url)}</span>`;
    }
    return img;
}

function setAutoScrollEnabled(enabled) {
    appState.autoScrollEnabled = !!enabled;
    const vp = getEl('comboTimelineViewport');
    const timeline = getEl('comboTimeline');
    if (vp) {
        if (appState.autoScrollEnabled) {
            vp.classList.add('auto-scroll-on');
            refreshTimelineIfLoaded();
        } else {
            vp.classList.remove('auto-scroll-on');
            if (timeline) timeline.style.transform = 'none';
            refreshTimelineIfLoaded();
        }
    }
}

/**
 * Element to center when auto-scroll is on.
 * Prefers the last top-level tile when `focusLatest` is set (server: transcription updates) or when
 * transcribe checkbox is on (main UI). Otherwise uses `.step.active` (combo practice / playback).
 */
function autoScrollTimelineTargetEl(timeline, scrollOpts) {
    scrollOpts = scrollOpts || {};
    if (!timeline) return null;
    const preferLatest =
        !!scrollOpts.focusLatest
        || !!getEl('transcribeModeToggle')?.checked;
    if (preferLatest) {
        const tops = [...timeline.children].filter(
            (c) =>
                c.classList.contains('step-group')
                || c.classList.contains('step-sequence')
                || (c.classList.contains('step')
                    && !c.classList.contains('group-item')
                    && !c.classList.contains('sequence-item')),
        );
        if (tops.length) return tops[tops.length - 1];
    }
    return timeline.querySelector('.step.active');
}

function applyAutoScroll(scrollOpts) {
    if (!appState.autoScrollEnabled) return;
    const viewport = getEl('comboTimelineViewport');
    const timeline = getEl('comboTimeline');
    if (!viewport || !timeline) return;

    const target = autoScrollTimelineTargetEl(timeline, scrollOpts);
    if (!target) return;

    const vpRect = viewport.getBoundingClientRect();
    const activeRect = target.getBoundingClientRect();

    const vpCenter = vpRect.left + (vpRect.width / 2);
    const activeCenter = activeRect.left + (activeRect.width / 2);
    const offset = activeCenter - vpCenter;

    // We must use transform for positioning as per layout
    const style = window.getComputedStyle(timeline);
    let currentX = 0;
    if (style.transform && style.transform !== 'none') {
        try {
            // Logic to parse matrix(1, 0, 0, 1, x, y)
            const matrix = new DOMMatrix(style.transform);
            currentX = matrix.m41;
        } catch (e) {
            console.error('AutoScroll matrix parse error', e);
        }
    }

    // Shift to left to compensate positive offset (right-side target)
    const newX = currentX - offset;
    timeline.style.transform = `translateX(${newX}px)`;
}

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

// Combo selector
const comboSelector = getEl('comboSelector');
if (comboSelector) {
    comboSelector.addEventListener('change', () => {
        const name = comboSelector.value;
        if (name) {
            sendMessage('select_combo', { name });
        }
    });
}

const noFailModeEl = getEl('noFailMode');
if (noFailModeEl) {
    noFailModeEl.addEventListener('change', () => {
        sendMessage('set_no_fail', { enabled: noFailModeEl.checked });
    });
}

// Save/Update button
const saveBtn = getEl('saveBtn');
if (saveBtn) {
    saveBtn.addEventListener('click', () => {
        readKeyImagesFromUI();

        const name = (getEl('comboName')?.value || '').toString();
        const inputs = (getEl('comboInputs')?.value || '').toString();
        const enders = (getEl('comboEnders')?.value || '').toString();
        const expectedTime = (getEl('comboExpectedTime')?.value || '').toString();
        const userDifficulty = (getEl('comboUserDifficulty')?.value || '').toString();
        const toggle = getEl('stepDisplayToggle');
        const mode = toggle?.checked ? 'images' : 'icons';

        const demoVideo = (getEl('comboDemoVideo')?.value || '').toString().trim();
        sendMessage('save_combo', {
            name,
            inputs,
            enders,
            expected_time: expectedTime,
            user_difficulty: userDifficulty,
            step_display_mode: mode,
            key_images: appState.keyImages,
            demo_video: demoVideo,
            target_game: appState.targetGame,
            ww_team_id: appState.wwTeamId || ''
        });
    });
}

// Demo video input: update embed preview on change
const comboDemoVideoEl = getEl('comboDemoVideo');
if (comboDemoVideoEl) {
    comboDemoVideoEl.addEventListener('input', () => updateDemoVideoEmbed(comboDemoVideoEl.value));
    comboDemoVideoEl.addEventListener('change', () => updateDemoVideoEmbed(comboDemoVideoEl.value));
}

// New combo button
const newBtn = getEl('newBtn');
if (newBtn) {
    newBtn.addEventListener('click', () => {
        sendMessage('new_combo');
    });
}

// Delete combo button
const deleteBtn = getEl('deleteBtn');
if (deleteBtn) {
    attachTwoClickConfirm(deleteBtn, {
        confirmText: 'Confirm delete',
        onConfirm: () => {
            const name = (getEl('comboName')?.value || '').toString();
            if (name) {
                sendMessage('delete_combo', { name });
            }
        }
    });
}

// Reload combos.json from disk (same folder as the app / exe)
const loadJsonBtn = getEl('loadJsonBtn');
if (loadJsonBtn) {
    loadJsonBtn.addEventListener('click', () => {
        if (!window.confirm('Reload all combos and settings from combos.json? Unsaved changes in the editor will be lost.')) {
            return;
        }
        sendMessage('load_combos_from_json', {});
    });
}

// Clear history button
const clearBtn = getEl('clearBtn');
if (clearBtn) {
    attachTwoClickConfirm(clearBtn, {
        confirmText: 'Clear all',
        onConfirm: () => {
            sendMessage('clear_history');
        }
    });
}

// Clear ALL history (every combo) button
const clearAllBtn = getEl('clearAllBtn');
if (clearAllBtn) {
    attachTwoClickConfirm(clearAllBtn, {
        confirmText: 'Wipe every combo?',
        onConfirm: () => {
            sendMessage('clear_history_all');
        }
    });
}

// Log attempts toggle: when off (default), suppress new entries in the attempt log.
const logAttemptsToggleEl = getEl('logAttemptsToggle');
if (logAttemptsToggleEl) {
    let initial = false;
    try {
        initial = localStorage.getItem(LOG_ATTEMPTS_STORAGE_KEY) === '1';
    } catch (_) {
        initial = false;
    }
    logAttemptsToggleEl.checked = initial;
    setLogAttemptsEnabled(initial);
    logAttemptsToggleEl.addEventListener('change', () => {
        setLogAttemptsEnabled(logAttemptsToggleEl.checked);
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

// Wire up editor UI events
const stepToggleEl = getEl('stepDisplayToggle');
if (stepToggleEl) {
    stepToggleEl.addEventListener('change', () => {
        appState.stepDisplayMode = stepToggleEl.checked ? 'images' : 'icons';
        syncGameUIVisibility();
        refreshTimelineIfLoaded();
    });
}

const autoScrollToggleEl = getEl('autoScrollToggle');
if (autoScrollToggleEl) {
    if (document.body.classList.contains('timeline-window-view')) {
        autoScrollToggleEl.checked = true;
        setAutoScrollEnabled(true);
    } else {
        setAutoScrollEnabled(autoScrollToggleEl.checked);
    }
    autoScrollToggleEl.addEventListener('change', () => {
        setAutoScrollEnabled(autoScrollToggleEl.checked);
        refreshTimelineIfLoaded();
    });
}

const showFailCountEl = getEl('showFailCount');
if (showFailCountEl) {
    showFailCountEl.addEventListener('change', () => {
        appState.showFailCount = showFailCountEl.checked;
        refreshTimelineIfLoaded();
    });
}

const stepEditToggleEl = getEl('stepEditToggle');
if (stepEditToggleEl) {
    stepEditToggleEl.addEventListener('change', () => {
        appState.stepEditMode = stepEditToggleEl.checked;
        refreshTimelineIfLoaded();
    });
}

const collapseChainsToggleEl = getEl('collapseChainsToggle');
if (collapseChainsToggleEl) {
    collapseChainsToggleEl.checked = !!appState.collapseChainedPresses;
    collapseChainsToggleEl.addEventListener('change', () => {
        appState.collapseChainedPresses = collapseChainsToggleEl.checked;
        refreshTimelineIfLoaded();
    });
}

const copyOverlayUrlBtn = getEl('copyOverlayUrlBtn');
if (copyOverlayUrlBtn) {
    copyOverlayUrlBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(getTimelineUrl()).then(() => {
            const prevTitle = copyOverlayUrlBtn.title;
            copyOverlayUrlBtn.title = 'Copied!';
            setTimeout(() => { copyOverlayUrlBtn.title = prevTitle; }, 1500);
        });
    });
}

const openTimelineWindowBtn = getEl('openTimelineWindowBtn');
if (openTimelineWindowBtn) {
    openTimelineWindowBtn.addEventListener('click', () => {
        window.open(getTimelineUrl(), 'combo-tracker-timeline', 'width=900,height=400,menubar=no,toolbar=no');
    });
}

function sendTranscribeMode() {
    const toggle = getEl('transcribeModeToggle');
    const validInput = getEl('transcribeValidKeys');
    const startInput = getEl('transcribeStartKey');
    const stripToggle = getEl('transcribeStripWaitToggle');
    const stripMsEl = getEl('transcribeStripWaitMs');
    sendMessage('set_transcribe_mode', {
        enabled: !!(toggle && toggle.checked),
        valid_keys: (validInput && validInput.value.trim()) || '',
        start_key: (startInput && startInput.value.trim()) || '',
        strip_wait_under_enabled: !!(stripToggle && stripToggle.checked),
        strip_wait_under_ms: (stripMsEl && stripMsEl.value.trim()) || ''
    });
}

const transcribeModeToggle = getEl('transcribeModeToggle');
const transcribeValidKeysWrap = getEl('transcribeValidKeysWrap');
const transcribeValidKeysInput = getEl('transcribeValidKeys');
const transcribeStartKeyInput = getEl('transcribeStartKey');
const transcribeStripWaitToggle = getEl('transcribeStripWaitToggle');
const transcribeStripWaitMs = getEl('transcribeStripWaitMs');

function transcribePersistIfOn() {
    if (transcribeModeToggle && transcribeModeToggle.checked) sendTranscribeMode();
}
if (transcribeModeToggle) {
    transcribeModeToggle.addEventListener('change', () => {
        if (transcribeModeToggle.checked && macroModeToggle && macroModeToggle.checked) {
            macroModeToggle.checked = false;
            if (macroSettingsWrap) macroSettingsWrap.classList.add('hidden');
            sendMacroMode();
        }
        if (transcribeValidKeysWrap) transcribeValidKeysWrap.classList.toggle('hidden', !transcribeModeToggle.checked);
        sendTranscribeMode();
    });
}
if (transcribeValidKeysInput) {
    transcribeValidKeysInput.addEventListener('blur', transcribePersistIfOn);
    transcribeValidKeysInput.addEventListener('input', transcribePersistIfOn);
}
if (transcribeStartKeyInput) {
    transcribeStartKeyInput.addEventListener('blur', transcribePersistIfOn);
    transcribeStartKeyInput.addEventListener('input', transcribePersistIfOn);
}
if (transcribeStripWaitToggle) {
    transcribeStripWaitToggle.addEventListener('change', transcribePersistIfOn);
}
if (transcribeStripWaitMs) {
    transcribeStripWaitMs.addEventListener('blur', transcribePersistIfOn);
    transcribeStripWaitMs.addEventListener('input', transcribePersistIfOn);
}
if (transcribeValidKeysWrap && transcribeModeToggle) {
    transcribeValidKeysWrap.classList.toggle('hidden', !transcribeModeToggle.checked);
}

// --- Macro replay mode ---
function sendMacroMode() {
    const toggle = getEl('macroModeToggle');
    const startInput = getEl('macroStartKey');
    const stopInput = getEl('macroStopKey');
    const spamIntervalInput = getEl('macroSpamIntervalMs');
    sendMessage('set_macro_mode', {
        enabled: !!(toggle && toggle.checked),
        start_key: (startInput && startInput.value.trim()) || '',
        stop_key: (stopInput && stopInput.value.trim()) || '',
        spam_interval_ms: (spamIntervalInput && spamIntervalInput.value.trim()) || '',
    });
}

function macroPersistIfOn() {
    if (getEl('macroModeToggle')?.checked) sendMacroMode();
}

const macroModeToggle = getEl('macroModeToggle');
const macroSettingsWrap = getEl('macroSettingsWrap');
const macroStartKeyInput = getEl('macroStartKey');
const macroStopKeyInput = getEl('macroStopKey');
const macroSpamIntervalInput = getEl('macroSpamIntervalMs');

if (macroModeToggle) {
    macroModeToggle.addEventListener('change', () => {
        if (macroModeToggle.checked && transcribeModeToggle && transcribeModeToggle.checked) {
            transcribeModeToggle.checked = false;
            if (transcribeValidKeysWrap) transcribeValidKeysWrap.classList.add('hidden');
            sendTranscribeMode();
        }
        if (macroSettingsWrap) macroSettingsWrap.classList.toggle('hidden', !macroModeToggle.checked);
        sendMacroMode();
    });
}
if (macroStartKeyInput) {
    macroStartKeyInput.addEventListener('blur', macroPersistIfOn);
    macroStartKeyInput.addEventListener('input', macroPersistIfOn);
}
if (macroStopKeyInput) {
    macroStopKeyInput.addEventListener('blur', macroPersistIfOn);
    macroStopKeyInput.addEventListener('input', macroPersistIfOn);
}
if (macroSpamIntervalInput) {
    macroSpamIntervalInput.addEventListener('blur', macroPersistIfOn);
    macroSpamIntervalInput.addEventListener('input', macroPersistIfOn);
}
if (macroSettingsWrap && macroModeToggle) {
    macroSettingsWrap.classList.toggle('hidden', !macroModeToggle.checked);
}

window.addEventListener('resize', () => {
    if (appState.autoScrollEnabled) applyAutoScroll();
});

/**
 * Tokenize combo input string for syntax highlighting. Returns HTML with spans.
 */
function tokenizeComboInput(text) {
    const escape = (s) => String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    const tokens = [];
    let i = 0;
    const len = text.length;
    while (i < len) {
        // wait(key, duration) - animation lock
        if (text.slice(i).match(/^wait\s*\(/)) {
            const start = i;
            i += text.slice(i).match(/^wait\s*\(/)[0].length;
            let depth = 1;
            while (i < len && depth) {
                if (text[i] === '(') depth++;
                else if (text[i] === ')') depth--;
                i++;
            }
            tokens.push({ type: 'anim-wait', text: text.slice(start, i) });
            continue;
        }
        // hold(key, duration)
        if (text.slice(i).match(/^hold\s*\(/)) {
            const start = i;
            i += text.slice(i).match(/^hold\s*\(/)[0].length;
            let depth = 1;
            while (i < len && depth) {
                if (text[i] === '(') depth++;
                else if (text[i] === ')') depth--;
                i++;
            }
            tokens.push({ type: 'hold', text: text.slice(start, i) });
            continue;
        }
        // wait:duration (soft wait)
        if (text.slice(i).match(/^wait\s*:/)) {
            const start = i;
            i += text.slice(i).match(/^wait\s*:/)[0].length;
            while (i < len && /[^\s,\[\]\{\}]/.test(text[i])) i++;
            tokens.push({ type: 'wait', text: text.slice(start, i) });
            continue;
        }
        // optional -key
        if (text[i] === '-' && i + 1 < len && /[a-zA-Z0-9]/.test(text[i + 1])) {
            const start = i;
            i++;
            while (i < len && /[a-zA-Z0-9]/.test(text[i])) i++;
            tokens.push({ type: 'optional', text: text.slice(start, i) });
            continue;
        }
        // brackets and braces
        if (text[i] === '[' || text[i] === ']') {
            tokens.push({ type: 'bracket', text: text[i] });
            i++;
            continue;
        }
        if (text[i] === '{' || text[i] === '}') {
            tokens.push({ type: 'sequence', text: text[i] });
            i++;
            continue;
        }
        if (text[i] === ',') {
            tokens.push({ type: 'punct', text: ',' });
            i++;
            continue;
        }
        // default: one character (preserve newlines/spaces)
        const ch = text[i];
        tokens.push({ type: 'default', text: ch });
        i++;
    }
    return tokens.map(t => {
        const escaped = escape(t.text);
        if (t.type === 'default') return escaped;
        return `<span class="combo-hl-${t.type}">${escaped}</span>`;
    }).join('');
}

function updateComboInputHighlight() {
    const ta = getEl('comboInputs');
    const mirror = getEl('comboInputHighlight');
    if (!ta || !mirror) return;
    const raw = (ta.value || '');
    mirror.innerHTML = raw ? tokenizeComboInput(raw) : '';
    mirror.scrollTop = ta.scrollTop;
    mirror.scrollLeft = ta.scrollLeft;
}

const inputsEl = getEl('comboInputs');
if (inputsEl) {
    let t = null;
    inputsEl.addEventListener('input', () => {
        updateComboInputHighlight();
        if (t) clearTimeout(t);
        t = setTimeout(() => {
            renderKeyImagesEditor();
        }, 150);
    });
    inputsEl.addEventListener('scroll', () => {
        const mirror = getEl('comboInputHighlight');
        if (mirror) {
            mirror.scrollTop = inputsEl.scrollTop;
            mirror.scrollLeft = inputsEl.scrollLeft;
        }
    });
    // Initial highlight if textarea already has content (e.g. restored state)
    updateComboInputHighlight();
}

const keyImagesDetails = getEl('keyImagesDetails');
if (keyImagesDetails) {
    keyImagesDetails.addEventListener('toggle', () => {
        // Ensure UI is up-to-date when opening/closing
        renderKeyImagesEditor();
    });
}

const targetGameEl = getEl('targetGameSelect');
if (targetGameEl) {
    targetGameEl.addEventListener('change', () => {
        appState.targetGame = normalizeTargetGame(targetGameEl.value);
        // Send target game change to backend immediately for stateless operation
        sendMessage('update_target_game', { target_game: appState.targetGame });
        syncGameUIVisibility();
        refreshTimelineIfLoaded();
    });
}

// WW: team select dropdown
document.addEventListener('change', e => {
    if (e.target && e.target.id === 'wwTeamSelect') {
        appState.wwTeamId = (e.target.value || '').toString();
        sendMessage('select_team', { team_id: appState.wwTeamId, target_game: appState.targetGame });
    }
});

// WW: save team button
document.addEventListener('click', e => {
    if (e.target && e.target.id === 'saveTeamBtn') {
        const name = (getEl('wwTeamName')?.value || '').toString().trim();
        if (!name) { updateStatus('Please enter a team name.', 'fail'); return; }
        sendMessage('save_team', {
            team_id: appState.wwTeamId || '',
            team_name: name,
            slot1: appState.wwTeamSlots[0] || '',
            slot2: appState.wwTeamSlots[1] || '',
            slot3: appState.wwTeamSlots[2] || '',
        });
    }
    if (e.target && e.target.id === 'newTeamBtn') {
        appState.wwTeamId = '';
        appState.wwTeamSlots = ['', '', ''];
        const teamSel = getEl('wwTeamSelect');
        if (teamSel) teamSel.value = '';
        const nameEl = getEl('wwTeamName');
        if (nameEl) nameEl.value = '';
        renderWwTeamEditor();
        _resolveTeamImagesToState();
        refreshTimelineIfLoaded();
    }
});

// WW: delete team (two-click confirm wired after DOM is ready)
const deleteTeamBtn = getEl('deleteTeamBtn');
if (deleteTeamBtn) {
    attachTwoClickConfirm(deleteTeamBtn, {
        confirmText: 'Confirm delete',
        onConfirm: () => {
            if (!appState.wwTeamId) return;
            sendMessage('delete_team', { team_id: appState.wwTeamId });
        }
    });
}

// WW: dash input
document.addEventListener('input', e => {
    if (e.target && e.target.id === 'wwDashImageInput') {
        appState.wwDashImage = e.target.value.trim();
        const preview = getEl('wwDashPreview');
        if (preview) wwSetPreview(preview, e.target.value);
        sendMessage('update_ww_dash', { dash_image: appState.wwDashImage });
        refreshTimelineIfLoaded();
    }
});

// Batched message handling (keeps UI smooth with lots of hits)

function handleComboList(msg) {
    const selector = getEl('comboSelector');
    const active = msg.active || '';
    selector.innerHTML = '<option value="">— Select Combo —</option>';
    (msg.combos || []).forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        selector.appendChild(opt);
    });
    selector.value = active;
}

const MESSAGE_HANDLERS = {
    init: (msg) => initializeUI(msg),
    combo_list: (msg) => handleComboList(msg),
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
    hit: (msg) => addResultRow(msg),
    combo_dropped: (msg) => {
        stopWaitAnimation();
        stopHoldAnimation();
        updateStatus(msg.input, msg.color || 'fail');
        addResultRow(msg);
    },
    clear_results: () => clearAttemptLog(),
    status: (msg) => updateStatus(msg.text, msg.color),
    alert_notice: (msg) => {
        const t = (msg.text || '').toString();
        if (t) window.alert(t);
    },
    stat_update: (msg) => updateStats(msg.stats),
    attempt_start: (msg) => addAttemptSeparator(msg.name, msg.attempt),
    timeline_update: (msg) => updateTimeline(msg.steps, { focusLatest: !!msg.focus_latest }),
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

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    appState.batchQueue.push(msg);
    if (!appState.isProcessingBatch) processBatch();
};
