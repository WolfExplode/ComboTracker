// WebSocket connection
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log('Connected to Combo Trainer backend');
};

ws.onclose = () => {
    console.error('Connection lost. Please restart the application.');
    updateStatus('ERROR: Backend disconnected', 'fail');
};

// UI Initialization
function initializeUI(data) {
    const selector = document.getElementById('comboSelector');
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
    document.getElementById('resultsBody').innerHTML = '';
    document.getElementById('failBody').innerHTML = '';

    if (data.editor) setEditorFields(data.editor);
    if (data.status) updateStatus(data.status.text, data.status.color);
    if (data.stats !== undefined) updateStats(data.stats);
    if (data.min_time !== undefined) updateMinTime(data.min_time);
    if (data.difficulty !== undefined) updateDifficulty(data.difficulty);
    if (data.user_difficulty !== undefined) updateUserDifficulty(data.user_difficulty);
    if (data.apm !== undefined) updateAPM(data.apm);
    if (data.apm_max !== undefined) updateAPMMax(data.apm_max);
    setDifficultyColor(document.getElementById('difficultyDisplay'), data.difficulty_value);
    setDifficultyColor(document.getElementById('userDifficultyDisplay'), data.user_difficulty_value);
    if (data.timeline) updateTimeline(data.timeline);
    if (data.failures) updateFailures(data.failures);
}

// Step display config (per-combo, loaded from backend editor payload)
let currentStepDisplayMode = 'icons'; // "icons" | "images"
let currentKeyImages = {}; // key -> url
let lastTimelineSteps = null;

// Per-combo game config
let currentTargetGame = 'generic'; // "generic" | "wuthering_waves"
let currentWwAbilityImages = { "1": {}, "2": {}, "3": {} }; // char -> {e/q/r -> url}
let currentWwSwapImages = { "1": "", "2": "", "3": "" }; // swap key images for 1/2/3 (from team)
let currentWwLmbImages = { "1": "", "2": "", "3": "" }; // per-character LMB images (from team)
let currentWwDashImage = ""; // shared RMB/dash image (from team)
let currentWwTeams = []; // [{id,name}]
let currentWwTeamId = ''; // selected team id (combo assignment / active team)

function normalizeTargetGame(v) {
    const g = (v || '').toString().trim().toLowerCase();
    return (g === 'wuthering_waves') ? 'wuthering_waves' : 'generic';
}

function ensureWwAbilityShape(obj) {
    const out = { "1": {}, "2": {}, "3": {} };
    if (!obj || typeof obj !== 'object') return out;
    ['1','2','3'].forEach(c => {
        const m = obj[c];
        if (m && typeof m === 'object') {
            ['e','q','r'].forEach(a => {
                const url = (m[a] || '').toString().trim();
                if (url) out[c][a] = url;
            });
        }
    });
    return out;
}

function ensureWwSwapShape(obj) {
    const out = { "1": "", "2": "", "3": "" };
    if (!obj || typeof obj !== 'object') return out;
    ['1','2','3'].forEach(k => {
        const url = (obj[k] || '').toString().trim();
        if (url) out[k] = url;
    });
    return out;
}

function ensureWwLmbShape(obj) {
    const out = { "1": "", "2": "", "3": "" };
    if (!obj || typeof obj !== 'object') return out;
    ['1','2','3'].forEach(k => {
        const url = (obj[k] || '').toString().trim();
        if (url) out[k] = url;
    });
    return out;
}

function syncGameUIVisibility() {
    const imagesOn = currentStepDisplayMode === 'images' || !!document.getElementById('stepDisplayToggle')?.checked;
    const wwDetails = document.getElementById('wwAbilityDetails');
    if (wwDetails) {
        // Only show WW editor when WW mode AND images are enabled.
        wwDetails.classList.toggle('hidden', !(currentTargetGame === 'wuthering_waves' && imagesOn));
    }
    const keyDetails = document.getElementById('keyImagesDetails');
    if (keyDetails) {
        // In WW mode, the WW panel replaces key images completely.
        keyDetails.classList.toggle('hidden', currentTargetGame === 'wuthering_waves');
    }

    const teamLabel = document.getElementById('wwTeamLabel');
    const teamControls = document.getElementById('wwTeamControls');
    if (teamLabel) teamLabel.classList.toggle('hidden', currentTargetGame !== 'wuthering_waves');
    if (teamControls) teamControls.classList.toggle('hidden', currentTargetGame !== 'wuthering_waves');

    // Re-render key images editor (generic mode only).
    renderKeyImagesEditor();
}

function readWwAbilityFromUI() {
    const container = document.getElementById('wwAbilityEditor');
    if (!container) return;
    const inputs = container.querySelectorAll('input[data-char][data-ability]');
    const next = { "1": {}, "2": {}, "3": {} };
    inputs.forEach(inp => {
        const c = (inp.getAttribute('data-char') || '').trim();
        const a = (inp.getAttribute('data-ability') || '').trim().toLowerCase();
        const url = (inp.value || '').toString().trim();
        if (!['1','2','3'].includes(c)) return;
        if (!['e','q','r'].includes(a)) return;
        if (url) next[c][a] = url;
    });
    currentWwAbilityImages = next;
}

function readWwSwapFromUI() {
    const container = document.getElementById('wwAbilityEditor');
    if (!container) return;
    const inputs = container.querySelectorAll('input[data-swap]');
    const next = { "1": "", "2": "", "3": "" };
    inputs.forEach(inp => {
        const c = (inp.getAttribute('data-swap') || '').trim();
        if (!['1','2','3'].includes(c)) return;
        const url = (inp.value || '').toString().trim();
        next[c] = url;
    });
    currentWwSwapImages = next;
}

function readWwLmbFromUI() {
    const container = document.getElementById('wwAbilityEditor');
    if (!container) return;
    const inputs = container.querySelectorAll('input[data-lmb]');
    const next = { "1": "", "2": "", "3": "" };
    inputs.forEach(inp => {
        const c = (inp.getAttribute('data-lmb') || '').trim();
        if (!['1','2','3'].includes(c)) return;
        next[c] = (inp.value || '').toString().trim();
    });
    currentWwLmbImages = next;
}

function readWwDashFromUI() {
    const el = document.getElementById('wwDashImageInput');
    if (!el) return;
    currentWwDashImage = (el.value || '').toString().trim();
}

function renderWwAbilityEditor({ preserveEdits = true } = {}) {
    // In most re-renders we want to preserve in-progress edits by reading from the UI first.
    // But when selecting a team (loading from JSON), we must NOT overwrite loaded state with old DOM values.
    if (preserveEdits) {
        readWwAbilityFromUI();
        readWwSwapFromUI();
        readWwLmbFromUI();
        readWwDashFromUI();
    }
    const container = document.getElementById('wwAbilityEditor');
    if (!container) return;
    container.innerHTML = '';

    const setPreview = (imgEl, val) => {
        const v = (val || '').toString().trim();
        if (!v) {
            imgEl.innerHTML = '';
            imgEl.style.display = 'none';
            return;
        }
        imgEl.style.display = 'flex';
        imgEl.style.alignItems = 'center';
        imgEl.style.justifyContent = 'center';
        if (/^https?:\/\//i.test(v)) {
            imgEl.innerHTML = `<img class="key-step-image" src="${escapeHtml(v)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:32px;height:32px;object-fit:contain;" />`;
        } else {
            imgEl.innerHTML = `<span class="key-step-emoji">${escapeHtml(v)}</span>`;
        }
    };

    // Dash (RMB) icon (shared)
    const dashBox = document.createElement('div');
    dashBox.className = 'ww-ability-char';

    const dashTitle = document.createElement('div');
    dashTitle.className = 'ww-ability-key';
    dashTitle.textContent = 'Dash';

    const dashControls = document.createElement('div');
    dashControls.className = 'ww-ability-controls';

    const dashInput = document.createElement('input');
    dashInput.type = 'text';
    dashInput.placeholder = 'https://... or 💨';
    dashInput.id = 'wwDashImageInput';
    dashInput.value = (currentWwDashImage || '').toString();

    const dashPreview = document.createElement('div');
    dashPreview.className = 'ww-ability-preview';
    setPreview(dashPreview, dashInput.value);

    dashInput.addEventListener('input', () => {
        currentWwDashImage = (dashInput.value || '').toString().trim();
        setPreview(dashPreview, currentWwDashImage);
        if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
    });

    dashControls.appendChild(dashInput);
    dashControls.appendChild(dashPreview);
    dashBox.appendChild(dashTitle);
    dashBox.appendChild(dashControls);
    container.appendChild(dashBox);

    const abilities = ['e','q','r'];
    ['1','2','3'].forEach(c => {
        const box = document.createElement('div');
        box.className = 'ww-ability-char';

        // Character swap image (stored in currentKeyImages[c])
        const swapRow = document.createElement('div');
        swapRow.className = 'ww-ability-row';

        const swapKey = document.createElement('div');
        swapKey.className = 'ww-ability-key';
        swapKey.textContent = `Character ${c}`;

        const swapControls = document.createElement('div');
        swapControls.className = 'ww-ability-controls';

        const swapInput = document.createElement('input');
        swapInput.type = 'text';
        swapInput.placeholder = 'https://... or 😀';
        swapInput.setAttribute('data-swap', c);
        swapInput.value = (currentWwSwapImages && currentWwSwapImages[c]) ? String(currentWwSwapImages[c]) : '';

        const swapPreview = document.createElement('div');
        swapPreview.className = 'ww-ability-preview';
        setPreview(swapPreview, swapInput.value);

        swapInput.addEventListener('input', () => {
            const val = (swapInput.value || '').toString().trim();
            currentWwSwapImages[c] = val;
            setPreview(swapPreview, val);
            if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
        });

        swapControls.appendChild(swapInput);
        swapControls.appendChild(swapPreview);
        swapRow.appendChild(swapKey);
        swapRow.appendChild(swapControls);

        // LMB (normal attack) image (per character)
        const lmbRow = document.createElement('div');
        lmbRow.className = 'ww-ability-row';

        const lmbKey = document.createElement('div');
        lmbKey.className = 'ww-ability-key';
        lmbKey.textContent = 'LMB';

        const lmbControls = document.createElement('div');
        lmbControls.className = 'ww-ability-controls';

        const lmbInput = document.createElement('input');
        lmbInput.type = 'text';
        lmbInput.placeholder = 'https://... or 😀';
        lmbInput.setAttribute('data-lmb', c);
        lmbInput.value = (currentWwLmbImages && currentWwLmbImages[c]) ? String(currentWwLmbImages[c]) : '';

        const lmbPreview = document.createElement('div');
        lmbPreview.className = 'ww-ability-preview';
        setPreview(lmbPreview, lmbInput.value);

        lmbInput.addEventListener('input', () => {
            currentWwLmbImages[c] = (lmbInput.value || '').toString().trim();
            setPreview(lmbPreview, currentWwLmbImages[c]);
            if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
        });

        lmbControls.appendChild(lmbInput);
        lmbControls.appendChild(lmbPreview);
        lmbRow.appendChild(lmbKey);
        lmbRow.appendChild(lmbControls);

        abilities.forEach(a => {
            const row = document.createElement('div');
            row.className = 'ww-ability-row';

            const k = document.createElement('div');
            k.className = 'ww-ability-key';
            k.textContent = a.toUpperCase();

            const controls = document.createElement('div');
            controls.className = 'ww-ability-controls';

            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = 'https://... or 😀';
            input.setAttribute('data-char', c);
            input.setAttribute('data-ability', a);
            input.value = (currentWwAbilityImages?.[c]?.[a]) ? String(currentWwAbilityImages[c][a]) : '';

            const preview = document.createElement('div');
            preview.className = 'ww-ability-preview';
            setPreview(preview, input.value);

            input.addEventListener('input', () => {
                const val = (input.value || '').toString().trim();
                if (!currentWwAbilityImages[c]) currentWwAbilityImages[c] = {};
                if (val) currentWwAbilityImages[c][a] = val;
                else if (currentWwAbilityImages[c]) delete currentWwAbilityImages[c][a];
                setPreview(preview, val);
                if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
            });

            controls.appendChild(input);
            controls.appendChild(preview);
            row.appendChild(k);
            row.appendChild(controls);
            box.appendChild(row);
        });

        box.appendChild(swapRow);
        box.appendChild(lmbRow);
        container.appendChild(box);
    });
}

// Split by commas, but do NOT split commas that are inside (), {}, or [].
function splitInputsSafe(s) {
    const str = (s || '').toString();
    const out = [];
    let buf = '';
    let paren = 0;
    let brace = 0;
    let bracket = 0;

    for (const ch of str) {
        if (ch === '(') paren++;
        else if (ch === ')') paren = Math.max(0, paren - 1);
        else if (ch === '{') brace++;
        else if (ch === '}') brace = Math.max(0, brace - 1);
        else if (ch === '[') bracket++;
        else if (ch === ']') bracket = Math.max(0, bracket - 1);

        if (ch === ',' && paren === 0 && brace === 0 && bracket === 0) {
            const token = buf.trim();
            if (token) out.push(token);
            buf = '';
            continue;
        }
        buf += ch;
    }
    const token = buf.trim();
    if (token) out.push(token);
    return out;
}

function extractKeysFromToken(token) {
    const t = (token || '').toString().trim();
    if (!t) return [];
    const tl = t.toLowerCase().trim();

    // wait(r, 1.5) -> key "r"
    if (tl.startsWith('wait(') && tl.endsWith(')') && tl.length >= 6) {
        const inner = t.slice(5, -1);
        const parts = splitInputsSafe(inner);
        const k = (parts[0] || '').toString().trim().toLowerCase();
        return k ? [k] : [];
    }

    // [q, e] or [wait(r, 1.5), q, e]
    if (tl.startsWith('[') && tl.endsWith(']') && tl.length >= 3) {
        const inner = t.slice(1, -1);
        const parts = splitInputsSafe(inner);
        const out = [];
        parts.forEach(p => out.push(...extractKeysFromToken(p)));
        return out;
    }

    // wait:0.2 / wait_soft:0.2 / wait_hard:0.2 are timing gates, not keys
    if (tl.startsWith('wait:') || tl.startsWith('wait_soft:') || tl.startsWith('wait_hard:')) {
        return [];
    }

    // hold(space, 0.3) -> key "space"
    if (tl.startsWith('hold(') && tl.endsWith(')') && tl.length >= 7) {
        const inner = t.slice(5, -1);
        const parts = splitInputsSafe(inner);
        const k = (parts[0] || '').toString().trim().toLowerCase();
        return k ? [k] : [];
    }

    // space{200ms} -> key "space"
    if (tl.includes('{') && tl.endsWith('}')) {
        const base = t.split('{', 1)[0].trim().toLowerCase();
        return base ? [base] : [];
    }

    // Plain key token
    return [tl];
}

function uniqueKeysFromInputsText(inputsText) {
    const keys = new Set();
    const tokens = splitInputsSafe(inputsText || '');
    tokens.forEach(tok => {
        extractKeysFromToken(tok).forEach(k => {
            const key = (k || '').toString().trim().toLowerCase();
            if (key) keys.add(key);
        });
    });
    return keys;
}

function readKeyImagesFromUI() {
    const container = document.getElementById('keyImagesEditor');
    if (!container) return;
    const rows = container.querySelectorAll('.key-image-row');
    const next = {};
    rows.forEach(row => {
        const k = (row.getAttribute('data-key') || '').toString().trim().toLowerCase();
        const input = row.querySelector('input.key-image-url');
        if (!k || !input) return;
        const url = (input.value || '').toString().trim();
        if (url) next[k] = url;
    });
    currentKeyImages = next;
}

function renderKeyImagesEditor() {
    // Preserve any in-progress edits before re-render
    readKeyImagesFromUI();

    const container = document.getElementById('keyImagesEditor');
    if (!container) return;
    container.innerHTML = '';

    if (currentTargetGame === 'wuthering_waves') {
        // In WW mode, we hide this panel anyway; keep the editor empty.
        return;
    }

    const inputsText = document.getElementById('comboInputs')?.value || '';
    const unique = uniqueKeysFromInputsText(inputsText);

    const allKeys = new Set([...Object.keys(currentKeyImages || {}), ...unique]);
    const sorted = Array.from(allKeys).sort((a, b) => a.localeCompare(b));

    if (sorted.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'help-text';
        empty.textContent = 'No keys detected yet. Add Inputs above to generate key fields.';
        container.appendChild(empty);
        return;
    }

    sorted.forEach(key => {
        const row = document.createElement('div');
        row.className = 'key-image-row';
        row.setAttribute('data-key', key);
        if (!unique.has(key)) row.classList.add('unused');

        const keyEl = document.createElement('div');
        keyEl.className = 'key-image-key';
        keyEl.textContent = key.toUpperCase();

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'key-image-url';
        input.placeholder = 'https://...';
        input.value = (currentKeyImages && currentKeyImages[key]) ? String(currentKeyImages[key]) : '';

        const preview = document.createElement('div');
        preview.className = 'key-image-preview';

        const updatePreview = () => {
            const val = (input.value || '').toString().trim();
            if (val) {
                if (/^https?:\/\//i.test(val)) {
                    preview.innerHTML = `<img class="key-image-preview" alt="${escapeHtml(key)}" referrerpolicy="no-referrer" src="${escapeHtml(val)}" style="display:block; width:32px; height:32px; object-fit:contain; border:0; background:transparent;" />`;
                } else {
                    preview.innerHTML = `<span class="key-step-emoji">${escapeHtml(val)}</span>`;
                }
                preview.style.display = 'flex';
                preview.style.alignItems = 'center';
                preview.style.justifyContent = 'center';
            } else {
                preview.innerHTML = '';
                preview.style.display = 'none';
            }
        };
        updatePreview();

        input.addEventListener('input', () => {
            const val = (input.value || '').toString().trim();
            if (val) currentKeyImages[key] = val;
            else delete currentKeyImages[key];
            updatePreview();
            if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
        });

        row.appendChild(keyEl);
        row.appendChild(input);
        row.appendChild(preview);
        container.appendChild(row);
    });
}

// Hold progress (client-side animation)
let holdAnim = {
    active: false,
    requiredMs: 0,
    startedAt: 0
};
let holdRafId = null;

// Wait progress (client-side animation)
let waitAnim = {
    active: false,
    requiredMs: 0,
    startedAt: 0
};
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
    const stepEl = document.querySelector('.step.hold.active');
    if (!stepEl) {
        // Timeline might be re-rendering; try again next frame.
        holdRafId = requestAnimationFrame(tickHoldAnimation);
        return;
    }

    const elapsed = performance.now() - holdAnim.startedAt;
    const req = Math.max(1, holdAnim.requiredMs || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    stepEl.style.setProperty('--hold-pct', `${pct}%`);

    if (pct >= 100) {
        // Time's up; stop animating to save cycles (bar stays full).
        holdRafId = null;
        return;
    }

    holdRafId = requestAnimationFrame(tickHoldAnimation);
}

function startHoldAnimation(requiredMs) {
    holdAnim.active = true;
    holdAnim.requiredMs = Math.max(1, Number(requiredMs) || 1);
    holdAnim.startedAt = performance.now();
    if (holdRafId === null) {
        holdRafId = requestAnimationFrame(tickHoldAnimation);
    }
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
    const stepEl = document.querySelector('.step.wait.active');
    if (!stepEl) {
        // Timeline might be re-rendering; try again next frame.
        waitRafId = requestAnimationFrame(tickWaitAnimation);
        return;
    }

    const elapsed = performance.now() - waitAnim.startedAt;
    const req = Math.max(1, Number(waitAnim.requiredMs) || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    stepEl.style.setProperty('--wait-pct', `${pct}%`);

    if (pct >= 100) {
        // Time's up; stop animating to save cycles (bar stays full).
        waitRafId = null;
        return;
    }

    waitRafId = requestAnimationFrame(tickWaitAnimation);
}

function startWaitAnimation(requiredMs) {
    waitAnim.active = true;
    waitAnim.requiredMs = Math.max(1, Number(requiredMs) || 1);
    waitAnim.startedAt = performance.now();
    if (waitRafId === null) {
        waitRafId = requestAnimationFrame(tickWaitAnimation);
    }
}

// Status updates
function updateStatus(text, color) {
    const statusEl = document.getElementById('statusDisplay');
    statusEl.textContent = text;
    // Expected: ready|recording|success|fail|wait|neutral
    statusEl.className = `status-${color || 'neutral'}`;
}

// Stats display
function updateStats(statsText) {
    document.getElementById('statsDisplay').textContent = statsText;
}

function updateMinTime(text) {
    document.getElementById('minTimeDisplay').textContent = text;
}

function updateDifficulty(text) {
    document.getElementById('difficultyDisplay').textContent = text;
}

function updateUserDifficulty(text) {
    document.getElementById('userDifficultyDisplay').textContent = text;
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

function updateAPM(text) {
    document.getElementById('apmDisplay').textContent = text;
}

function updateAPMMax(text) {
    document.getElementById('apmMaxDisplay').textContent = text;
}

function setEditorFields(editor) {
    document.getElementById('comboName').value = editor.name || '';
    document.getElementById('comboInputs').value = editor.inputs || '';
    document.getElementById('comboEnders').value = editor.enders || '';
    document.getElementById('comboExpectedTime').value = editor.expected_time || '';
    document.getElementById('comboUserDifficulty').value = editor.user_difficulty || '';

    // New: step display configuration
    const toggleEl = document.getElementById('stepDisplayToggle');
    if (toggleEl) {
        const m = (editor.step_display_mode || 'icons').toString().trim().toLowerCase();
        toggleEl.checked = (m === 'images');
        currentStepDisplayMode = toggleEl.checked ? 'images' : 'icons';
    }
    currentKeyImages = (editor.key_images && typeof editor.key_images === 'object') ? {...editor.key_images} : {};
    renderKeyImagesEditor();

    // New: target game
    currentTargetGame = normalizeTargetGame(editor.target_game || 'generic');
    const gameEl = document.getElementById('targetGameSelect');
    if (gameEl) gameEl.value = currentTargetGame;

    // WW-specific config
    currentWwTeams = Array.isArray(editor.ww_teams) ? editor.ww_teams : [];
    currentWwTeamId = (editor.ww_team_id || '').toString();
    const teamSel = document.getElementById('wwTeamSelect');
    if (teamSel) {
        teamSel.innerHTML = '<option value="">— New Team —</option>';
        (currentWwTeams || []).forEach(t => {
            const id = (t.id || '').toString();
            const name = (t.name || '').toString();
            if (!id) return;
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = name || id;
            teamSel.appendChild(opt);
        });
        teamSel.value = currentWwTeamId || '';
    }
    const teamNameEl = document.getElementById('wwTeamName');
    if (teamNameEl) teamNameEl.value = (editor.ww_team_name || '').toString();

    currentWwSwapImages = ensureWwSwapShape(editor.ww_team_swap_images || {});
    currentWwLmbImages = ensureWwLmbShape(editor.ww_team_lmb_images || {});
    currentWwDashImage = (editor.ww_team_dash_image || '').toString();
    currentWwAbilityImages = ensureWwAbilityShape(editor.ww_team_ability_images || {});
    renderWwAbilityEditor({ preserveEdits: false });
    syncGameUIVisibility();
}

// Timeline visualization
function updateTimeline(steps) {
    lastTimelineSteps = steps;
    const timeline = document.getElementById('comboTimeline');
    timeline.innerHTML = '';
    const BASE_STEP_WIDTH_PX = 80; // 100ms hold == 1x base width

    const ctx = {
        game: currentTargetGame,
        char: '1',
    };

    const isSwapKey = (k) => (k === '1' || k === '2' || k === '3');
    const maybeSwapChar = (k) => {
        if (ctx.game !== 'wuthering_waves') return;
        if (isSwapKey(k)) ctx.char = k;
    };

    const escapeAttr = (text) => {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    };

    const isProbablyUrl = (val) => {
        const s = (val || '').toString().trim();
        return /^https?:\/\//i.test(s);
    };

    const resolveIconValue = (key) => {
        if (currentStepDisplayMode !== 'images') return null;
        const k = (key || '').toString().trim().toLowerCase();
        if (!k) return null;

        // WW: E/Q/R are character-specific.
        if (ctx.game === 'wuthering_waves' && (k === 'e' || k === 'q' || k === 'r')) {
            const c = (ctx.char || '1').toString();
            const url = currentWwAbilityImages?.[c]?.[k];
            if (url) return url;
        }

        // WW: swap keys 1/2/3 can have images (team preset)
        if (ctx.game === 'wuthering_waves' && (k === '1' || k === '2' || k === '3')) {
            const url = (currentWwSwapImages && currentWwSwapImages[k]) ? currentWwSwapImages[k] : '';
            if (url) return url;
        }

        // WW: LMB (normal attack) is per character
        if (ctx.game === 'wuthering_waves' && k === 'lmb') {
            const c = (ctx.char || '1').toString();
            const url = (currentWwLmbImages && currentWwLmbImages[c]) ? currentWwLmbImages[c] : '';
            if (url) return url;
        }

        // WW: RMB is always dash icon (shared)
        if (ctx.game === 'wuthering_waves' && k === 'rmb') {
            const url = (currentWwDashImage || '').toString().trim();
            if (url) return url;
        }

        // Fallback: generic per-key mapping
        const url = currentKeyImages ? currentKeyImages[k] : null;
        return url || null;
    };

    const keyImageHtml = (key) => {
        const k = (key || '').toString().trim().toLowerCase();
        const val = resolveIconValue(k);
        if (!val) return null;
        if (isProbablyUrl(val)) {
            return `<span class="key-img-wrap" aria-label="${escapeHtml(k)}" title="${escapeHtml(k)}"><img class="key-step-image" src="${escapeAttr(val)}" alt="${escapeAttr(k)}" loading="lazy" referrerpolicy="no-referrer"/></span>`;
        }
        // Emoji / text icon
        return `<span class="key-img-wrap" aria-label="${escapeHtml(k)}" title="${escapeHtml(k)}"><span class="key-step-emoji">${escapeHtml(val)}</span></span>`;
    };

    const cornerKeyHtml = (key) => {
        // Only show the corner key label in image mode (helps identify what the icon represents).
        if (currentStepDisplayMode !== 'images') return '';
        const k = (key || '').toString().trim();
        if (!k) return '';
        return `<span class="corner-key">${escapeHtml(k)}</span>`;
    };

    const mouseIconHtml = (key) => {
        const k = (key || '').toString().trim().toLowerCase();
        if (k !== 'lmb' && k !== 'rmb' && k !== 'mmb') return null;
        const isLeft = k === 'lmb';
        const isRight = k === 'rmb';
        const isMid = k === 'mmb';
        const label = isLeft ? 'LMB' : (isRight ? 'RMB' : 'MMB');
        // Inline SVG so this works offline with no external assets.
        // We highlight the clicked button area.
        return `
            <span class="mouse-icon" aria-label="${label}" title="${label}">
                <svg viewBox="0 0 64 64" role="img" focusable="false">
                    <rect x="18" y="6" width="28" height="52" rx="14" ry="14" fill="none" stroke="currentColor" stroke-width="3"/>
                    <line x1="32" y1="6" x2="32" y2="26" stroke="currentColor" stroke-width="3" opacity="0.55"/>
                    ${isLeft ? `<path d="M18 20 C18 12, 24 6, 32 6 L32 26 L18 26 Z" fill="currentColor" opacity="0.35"/>` : ``}
                    ${isRight ? `<path d="M46 20 C46 12, 40 6, 32 6 L32 26 L46 26 Z" fill="currentColor" opacity="0.35"/>` : ``}
                    ${isMid ? `<rect x="29" y="12" width="6" height="10" rx="3" fill="currentColor" opacity="0.4"/>` : ``}
                </svg>
            </span>
        `;
    };

    const primaryHtml = (raw) => {
        const t = (raw || '').toString().trim();
        const img = keyImageHtml(t);
        if (img) return img;
        const icon = mouseIconHtml(t);
        if (icon) return icon;
        // Single-character keys (letters/numbers) get a larger glyph.
        if (t.length === 1) return `<span class="step-primary">${escapeHtml(t.toUpperCase())}</span>`;
        // Fallback: normal text.
        return `<span class="step-primary step-primary-small">${escapeHtml(t)}</span>`;
    };

    const secondaryHtml = (raw) => {
        const t = (raw || '').toString();
        if (!t) return '';
        return `<span class="step-secondary">${escapeHtml(t)}</span>`;
    };

    const applyHoldWidth = (el, durationMs) => {
        const ms = Number(durationMs);
        const mult = (Number.isFinite(ms) && ms > 0) ? (ms / 100.0) : 1;
        const w = Math.max(BASE_STEP_WIDTH_PX, BASE_STEP_WIDTH_PX * mult);
        el.style.width = `${w}px`;
    };

    steps.forEach((step) => {

        // Group container (renders multiple step tiles inside one grouped block)
        if (step.type === 'group') {
            const group = document.createElement('div');
            group.className = 'step-group';

            // Per-step performance coloring (sent by backend as step.mark)
            if (step.mark) {
                const m = String(step.mark).toLowerCase();
                if (m === 'ok') group.classList.add('mark-ok');
                if (m === 'early') group.classList.add('mark-early');
                if (m === 'missed') group.classList.add('mark-missed');
                if (m === 'wrong') group.classList.add('mark-wrong');
            }
            if (step.active) group.classList.add('active');
            if (step.completed) group.classList.add('completed');

            const itemsWrap = document.createElement('div');
            itemsWrap.className = 'step-group-items';

            (step.items || []).forEach(item => {
                const child = document.createElement('div');
                child.className = 'step group-item';

                // Reuse existing step classes/labels
                if (item.type === 'wait') {
                    child.classList.add('wait');
                    const mode = (item.mode || 'soft').toLowerCase();
                    if (mode === 'mandatory') {
                        const k = (item.wait_for || '').toString().trim();
                        // Display as:
                        // R
                        // animation time 1500ms
                        child.innerHTML = `${primaryHtml(k || ' ')}${secondaryHtml(`animation time ${item.duration}ms`)}${cornerKeyHtml(k)}`;
                        maybeSwapChar((k || '').toString().trim().toLowerCase());
                    } else {
                        let label = 'wait';
                        if (mode === 'hard') label = 'wait hard';
                        child.innerHTML = `${primaryHtml(label)}${secondaryHtml(`≥${item.duration}ms`)}`;
                    }
                    child.style.setProperty('--wait-pct', item.completed ? '100%' : '0%');
                } else {
                    const k = (item.input || '').toString().trim().toLowerCase();
                    child.innerHTML = `${primaryHtml(k)}${cornerKeyHtml(k)}`;
                    maybeSwapChar(k);
                }

                if (item.active) child.classList.add('active');
                if (item.completed) child.classList.add('completed');

                itemsWrap.appendChild(child);
            });

            group.appendChild(itemsWrap);
            timeline.appendChild(group);
            return;
        }

        const div = document.createElement('div');
        div.className = 'step';

        // Per-step performance coloring (sent by backend as step.mark)
        if (step.mark) {
            const m = String(step.mark).toLowerCase();
            if (m === 'ok') div.classList.add('mark-ok');
            if (m === 'early') div.classList.add('mark-early');
            if (m === 'missed') div.classList.add('mark-missed');
            if (m === 'wrong') div.classList.add('mark-wrong');
        }
        
        if (step.type === 'press_wait') {
            // A key immediately followed by a wait gate, rendered as one tile:
            // key (big) + "100ms" (small) + wait progress fill.
            div.classList.add('press-wait');
            const k = (step.input || '').toString().trim().toLowerCase();
            div.innerHTML = `${primaryHtml(k)}${secondaryHtml(`${step.duration}ms`)}${cornerKeyHtml(k)}`;
            div.style.setProperty('--wait-pct', step.completed ? '100%' : '0%');
            maybeSwapChar(k);
        } else if (step.type === 'wait') {
            div.classList.add('wait');
            const mode = (step.mode || 'soft').toLowerCase();
            if (mode === 'mandatory') {
                const k = (step.wait_for || '').toString().trim();
                div.innerHTML = `${primaryHtml(k || ' ')}${secondaryHtml(`animation time ${step.duration}ms`)}${cornerKeyHtml(k)}`;
                maybeSwapChar((k || '').toString().trim().toLowerCase());
            } else {
                let label = 'wait';
                if (mode === 'hard') label = 'wait hard';
                div.innerHTML = `${primaryHtml(label)}${secondaryHtml(`≥${step.duration}ms`)}`;
            }
            // Whole-tile fill progress (default 0%, completed = 100%)
            div.style.setProperty('--wait-pct', step.completed ? '100%' : '0%');
        } else if (step.type === 'hold') {
            div.classList.add('hold');
            const k = (step.input || '').toString().trim().toLowerCase();
            div.innerHTML = `${primaryHtml(k)}${secondaryHtml(`hold ${step.duration}ms`)}${cornerKeyHtml(k)}`;
            applyHoldWidth(div, step.duration);
            // Whole-tile fill progress (default 0%, completed = 100%)
            div.style.setProperty('--hold-pct', step.completed ? '100%' : '0%');
            maybeSwapChar(k);
        } else {
            const k = (step.input || '').toString().trim().toLowerCase();
            div.innerHTML = `${primaryHtml(k)}${cornerKeyHtml(k)}`;
            maybeSwapChar(k);
        }
        
        if (step.active) div.classList.add('active');
        if (step.completed) div.classList.add('completed');
        
        timeline.appendChild(div);
    });
}

// Results table
function addAttemptSeparator(name, attempt) {
    const body = document.getElementById('resultsBody');
    const row = document.createElement('div');
    row.className = 'result-row separator';
    row.textContent = `—— ${name} | Attempt ${attempt} ——`;
    body.appendChild(row);
    scrollToBottom('resultsTable');
}

function addResultRow(data) {
    const body = document.getElementById('resultsBody');
    const row = document.createElement('div');
    row.className = 'result-row';
    
    if (data.split_ms === 'FAIL') {
        row.classList.add('fail');
    } else {
        row.classList.add('success');
    }
    
    row.innerHTML = `
        <span>${escapeHtml(data.input)}</span>
        <span>${data.split_ms}</span>
        <span>${data.total_ms}</span>
    `;
    
    body.appendChild(row);
    scrollToBottom('resultsTable');
}

function clearAttemptLog() {
    const body = document.getElementById('resultsBody');
    if (body) {
        body.innerHTML = '';
    }
}

// Failure analysis
function updateFailures(failures) {
    const body = document.getElementById('failBody');
    body.innerHTML = '';
    
    const labels = {
        'pressed too fast': 'pressed too fast (during wait)',
        'missed input': 'missed input (skipped step)',
        'out of order': 'out of order (jumped ahead)',
        'already passed': 'already passed (pressed old step)',
        'wrong input': 'wrong input',
        'too early': 'too early',
        'hold too short': 'hold too short'
    };

    const entries = Object.entries(failures || {})
        .map(([reason, count]) => [String(reason || ''), Number(count) || 0])
        .filter(([, count]) => count > 0)
        .sort((a, b) => b[1] - a[1]);

    entries.forEach(([reason, count]) => {
        const row = document.createElement('div');
        row.className = 'fail-row';
        const label = labels[reason.toLowerCase()] || reason;
        row.innerHTML = `
            <span>${escapeHtml(label)}</span>
            <span>${count}</span>
        `;
        body.appendChild(row);
    });
}

// Button handlers (send commands to Python)
document.getElementById('saveBtn').onclick = () => {
    readKeyImagesFromUI();
    readWwAbilityFromUI();
    readWwSwapFromUI();
    const useImages = !!document.getElementById('stepDisplayToggle')?.checked;
    ws.send(JSON.stringify({
        type: 'save_combo',
        name: document.getElementById('comboName').value,
        inputs: document.getElementById('comboInputs').value,
        enders: document.getElementById('comboEnders').value,
        expected_time: document.getElementById('comboExpectedTime').value,
        user_difficulty: document.getElementById('comboUserDifficulty').value,
        step_display_mode: useImages ? 'images' : 'icons',
        key_images: (currentTargetGame === 'wuthering_waves') ? {} : (currentKeyImages || {}),
        target_game: currentTargetGame || 'generic',
        ww_team_id: (currentTargetGame === 'wuthering_waves') ? (currentWwTeamId || '') : ''
    }));
};

document.getElementById('newBtn').onclick = () => {
    ws.send(JSON.stringify({type: 'new_combo'}));
};

document.getElementById('deleteBtn').onclick = () => {
    const name = document.getElementById('comboSelector').value;
    if (name) ws.send(JSON.stringify({type: 'delete_combo', name}));
};

document.getElementById('clearBtn').onclick = () => {
    ws.send(JSON.stringify({type: 'clear_history'}));
};

document.getElementById('comboSelector').onchange = (e) => {
    ws.send(JSON.stringify({type: 'select_combo', name: e.target.value}));
};

// Helper functions
function scrollToBottom(elementId) {
    const el = document.getElementById(elementId);
    el.scrollTop = el.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Wire up editor UI events
const stepToggleEl = document.getElementById('stepDisplayToggle');
if (stepToggleEl) {
    stepToggleEl.addEventListener('change', () => {
        currentStepDisplayMode = stepToggleEl.checked ? 'images' : 'icons';
        syncGameUIVisibility();
        // Re-render timeline immediately if we have one
        if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
    });
}

const inputsEl = document.getElementById('comboInputs');
if (inputsEl) {
    let t = null;
    inputsEl.addEventListener('input', () => {
        if (t) clearTimeout(t);
        t = setTimeout(() => {
            renderKeyImagesEditor();
        }, 150);
    });
}

const keyImagesDetails = document.getElementById('keyImagesDetails');
if (keyImagesDetails) {
    keyImagesDetails.addEventListener('toggle', () => {
        // Ensure UI is up-to-date when opening/closing
        renderKeyImagesEditor();
    });
}

const targetGameEl = document.getElementById('targetGameSelect');
if (targetGameEl) {
    targetGameEl.addEventListener('change', () => {
        currentTargetGame = normalizeTargetGame(targetGameEl.value);
        syncGameUIVisibility();
        renderWwAbilityEditor({ preserveEdits: true });
        if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
    });
}

const wwTeamSelectEl = document.getElementById('wwTeamSelect');
if (wwTeamSelectEl) {
    wwTeamSelectEl.addEventListener('change', () => {
        currentWwTeamId = (wwTeamSelectEl.value || '').toString();
        ws.send(JSON.stringify({ type: 'select_team', team_id: currentWwTeamId }));
    });
}

const saveTeamBtn = document.getElementById('saveTeamBtn');
if (saveTeamBtn) {
    saveTeamBtn.addEventListener('click', () => {
        readWwAbilityFromUI();
        readWwSwapFromUI();
        readWwLmbFromUI();
        readWwDashFromUI();
        const name = (document.getElementById('wwTeamName')?.value || '').toString();
        ws.send(JSON.stringify({
            type: 'save_team',
            team_id: currentWwTeamId || '',
            team_name: name,
            dash_image: currentWwDashImage || '',
            swap_images: currentWwSwapImages || {},
            lmb_images: currentWwLmbImages || {},
            ability_images: currentWwAbilityImages || {}
        }));
    });
}

const newTeamBtn = document.getElementById('newTeamBtn');
if (newTeamBtn) {
    newTeamBtn.addEventListener('click', () => {
        currentWwTeamId = '';
        const teamSel = document.getElementById('wwTeamSelect');
        if (teamSel) teamSel.value = '';
        const nameEl = document.getElementById('wwTeamName');
        if (nameEl) nameEl.value = '';
        currentWwSwapImages = { "1": "", "2": "", "3": "" };
        currentWwLmbImages = { "1": "", "2": "", "3": "" };
        currentWwDashImage = "";
        currentWwAbilityImages = { "1": {}, "2": {}, "3": {} };
        renderWwAbilityEditor({ preserveEdits: false });
    });
}

const deleteTeamBtn = document.getElementById('deleteTeamBtn');
if (deleteTeamBtn) {
    deleteTeamBtn.addEventListener('click', () => {
        if (!currentWwTeamId) return;
        ws.send(JSON.stringify({ type: 'delete_team', team_id: currentWwTeamId }));
    });
}

const wwDetails = document.getElementById('wwAbilityDetails');
if (wwDetails) {
    wwDetails.addEventListener('toggle', () => {
        renderWwAbilityEditor({ preserveEdits: true });
    });
}

// Batched message handling (keeps UI smooth with lots of hits)
let batchQueue = [];
let isProcessingBatch = false;

function handleMessage(msg) {
    switch (msg.type) {
        case 'init':
            initializeUI(msg);
            break;
        case 'combo_list': {
            const selector = document.getElementById('comboSelector');
            const active = msg.active || '';
            selector.innerHTML = '<option value="">— Select Combo —</option>';
            (msg.combos || []).forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                selector.appendChild(opt);
            });
            selector.value = active;
            break;
        }
        case 'combo_data':
            setEditorFields(msg);
            break;
        case 'min_time':
            updateMinTime(msg.text);
            break;
        case 'difficulty_update':
            updateDifficulty(msg.text);
            setDifficultyColor(document.getElementById('difficultyDisplay'), msg.value);
            break;
        case 'user_difficulty_update':
            updateUserDifficulty(msg.text);
            setDifficultyColor(document.getElementById('userDifficultyDisplay'), msg.value);
            break;
        case 'apm_update':
            updateAPM(msg.text);
            break;
        case 'apm_max_update':
            updateAPMMax(msg.text);
            break;
        case 'hold_begin':
            startHoldAnimation(msg.required_ms);
            break;
        case 'hold_end':
            stopHoldAnimation();
            break;
        case 'wait_begin':
            startWaitAnimation(msg.required_ms);
            break;
        case 'wait_end':
            stopWaitAnimation();
            break;
        case 'hit':
            addResultRow(msg);
            break;
        case 'clear_results':
            clearAttemptLog();
            break;
        case 'status':
            updateStatus(msg.text, msg.color);
            break;
        case 'stat_update':
            updateStats(msg.stats);
            break;
        case 'attempt_start':
            addAttemptSeparator(msg.name, msg.attempt);
            break;
        case 'timeline_update':
            updateTimeline(msg.steps);
            break;
        case 'fail_update':
            updateFailures(msg.failures);
            break;
    }
}

function processBatch() {
    if (batchQueue.length === 0) {
        isProcessingBatch = false;
        return;
    }
    isProcessingBatch = true;

    requestAnimationFrame(() => {
        const batch = batchQueue.splice(0, batchQueue.length);
        batch.forEach(handleMessage);
        isProcessingBatch = false;
        if (batchQueue.length > 0) processBatch();
    });
}

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    batchQueue.push(msg);
    if (!isProcessingBatch) processBatch();
};