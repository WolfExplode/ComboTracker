// Combo editor (Practice page): the editor fields, the inputs highlighter, and record (transcribe) / replay (macro) modes.

// UI Initialization
function initializeUI(data) {
    setComboList(data.combos || [], data.active_combo || '', data.overview);

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
        renderWwPanels();
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
        appState.savedInputs = inputsEl.value;
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

    renderWwPanels();
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
        const name = (getEl('comboName')?.value || '').toString();
        const inputs = (getEl('comboInputs')?.value || '').toString();
        const enders = (getEl('comboEnders')?.value || '').toString();
        const expectedTime = (getEl('comboExpectedTime')?.value || '').toString();
        const userDifficulty = (getEl('comboUserDifficulty')?.value || '').toString();
        const toggle = getEl('stepDisplayToggle');
        const mode = toggle?.checked ? 'images' : 'icons';

        const demoVideo = (getEl('comboDemoVideo')?.value || '').toString().trim();
        if (name.trim()) appState.pendingToast = `Saved ${name.trim()}`;
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

// Ctrl/Cmd+S: save the current combo instead of triggering the browser's save-page dialog
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 's') {
        e.preventDefault();
        saveBtn?.click();
    }
});

// Ctrl/Cmd+Z: undo the last Edit Steps mutation (delete, reorder, or inline field edit)
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'z') {
        if (!appState.stepEditMode || appState.editStepsUndoStack.length === 0) return;
        e.preventDefault();
        undoLastEditStep();
    }
});

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
        confirmText: 'Click again to delete',
        onConfirm: () => {
            const name = (getEl('comboName')?.value || '').toString();
            if (name) {
                sendMessage('delete_combo', { name });
            }
        }
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
        renderRailStatus();
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
        renderRailStatus();
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
        // wait:duration (soft wait), -wait:duration (optional wait)
        if (text.slice(i).match(/^-?wait\s*:/)) {
            const start = i;
            i += text.slice(i).match(/^-?wait\s*:/)[0].length;
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
    inputsEl.addEventListener('input', updateComboInputHighlight);
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

// ---------------------------------------------------------------------------
// Editing focus: the Combo Steps tile for the input under the text cursor gets outlined,
// so you can see which step you're editing. The timeline shows the saved combo, so this
// only follows tokens that still line up with it (edits after the cursor are fine).
// ---------------------------------------------------------------------------

/** Top-level tokens with their [start, end) offsets in the text (same split as splitInputsTokens). */
function splitInputsTokenSpans(str) {
    const spans = [];
    let depth = 0;
    let start = 0;
    const push = (end) => {
        const raw = str.slice(start, end);
        const lead = raw.length - raw.trimStart().length;
        const text = raw.trim();
        if (text) spans.push({ text, start: start + lead, end: start + lead + text.length });
    };
    for (let i = 0; i < str.length; i++) {
        const ch = str[i];
        if (ch === '(' || ch === '{' || ch === '[') depth++;
        else if (ch === ')' || ch === '}' || ch === ']') depth = Math.max(0, depth - 1);
        else if (ch === ',' && depth === 0) { push(i); start = i + 1; }
    }
    push(str.length);
    return spans;
}

/** Runtime step indices for the token under the cursor, or [] when it can't be matched to the timeline. */
function runtimeIndicesAtCursor() {
    const ta = getEl('comboInputs');
    if (!ta || document.activeElement !== ta) return [];
    const text = ta.value || '';
    const spans = splitInputsTokenSpans(text);
    if (spans.length === 0) return [];
    const caret = ta.selectionStart || 0;
    // The token whose text (or trailing comma/space) holds the caret.
    let tokIdx = spans.findIndex((sp, i) => caret <= sp.end || i === spans.length - 1 || caret < spans[i + 1].start);
    if (tokIdx < 0) tokIdx = spans.length - 1;

    // Only trust the mapping while every token up to the cursor matches the saved combo.
    const saved = splitInputsTokens(appState.savedInputs || '');
    for (let i = 0; i <= tokIdx; i++) {
        if ((saved[i] || '').toLowerCase() !== spans[i].text.toLowerCase()) return [];
    }
    const srcMap = buildRuntimeToSourceMap(saved);
    const out = [];
    srcMap.forEach((src, runtimeIdx) => { if (src.includes(tokIdx)) out.push(runtimeIdx); });
    return out;
}

function applyEditFocus() {
    const timeline = getEl('comboTimeline');
    if (!timeline) return;
    const wanted = new Set(runtimeIndicesAtCursor());
    timeline.querySelectorAll('.step-edit-focus').forEach(el => el.classList.remove('step-edit-focus'));
    if (wanted.size === 0) return;
    [...timeline.children].forEach(tile => {
        const raw = (tile.dataset.stepIndices || '').trim();
        if (!raw) return;
        if (raw.split(',').some(v => wanted.has(Number.parseInt(v, 10)))) tile.classList.add('step-edit-focus');
    });
}

if (inputsEl) {
    ['click', 'keyup', 'focus', 'input', 'select'].forEach(ev => inputsEl.addEventListener(ev, applyEditFocus));
    inputsEl.addEventListener('blur', applyEditFocus);
    // Re-apply after every timeline re-render (live attempts, toggles, saves).
    const timeline = getEl('comboTimeline');
    if (timeline) new MutationObserver(() => { if (document.activeElement === inputsEl) applyEditFocus(); }).observe(timeline, { childList: true });
}
