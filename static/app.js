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
let autoScrollEnabled = false;

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

function readWwDataFromUI() {
    const container = document.getElementById('wwAbilityEditor');
    if (!container) return;

    currentWwAbilityImages = { "1": {}, "2": {}, "3": {} };
    currentWwSwapImages = { "1": "", "2": "", "3": "" };
    currentWwLmbImages = { "1": "", "2": "", "3": "" };

    const isValidChar = c => ['1', '2', '3'].includes(c);

    container.querySelectorAll('input[data-char][data-ability]').forEach(inp => {
        const c = inp.getAttribute('data-char')?.trim();
        const a = inp.getAttribute('data-ability')?.trim().toLowerCase();
        if (!isValidChar(c) || !['e', 'q', 'r'].includes(a)) return;
        const url = (inp.value || '').toString().trim();
        if (url) currentWwAbilityImages[c][a] = url;
    });

    const readFlat = (attr, target) => {
        container.querySelectorAll(`input[data-${attr}]`).forEach(inp => {
            const c = inp.getAttribute(`data-${attr}`)?.trim();
            if (!isValidChar(c)) return;
            target[c] = (inp.value || '').toString().trim();
        });
    };

    readFlat('swap', currentWwSwapImages);
    readFlat('lmb', currentWwLmbImages);

    currentWwDashImage = (document.getElementById('wwDashImageInput')?.value || '').toString().trim();
}

function renderWwAbilityEditor({ preserveEdits = true } = {}) {
    // In most re-renders we want to preserve in-progress edits by reading from the UI first.
    // But when selecting a team (loading from JSON), we must NOT overwrite loaded state with old DOM values.
    if (preserveEdits) {
        readWwDataFromUI();
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
        currentWwDashImage = dashInput.value.trim();
        setPreview(dashPreview, dashInput.value);
    });

    dashControls.appendChild(dashInput);
    dashControls.appendChild(dashPreview);
    dashBox.appendChild(dashTitle);
    dashBox.appendChild(dashControls);
    container.appendChild(dashBox);

    // For each character slot (1/2/3)
    ['1', '2', '3'].forEach(c => {
        const charBox = document.createElement('div');
        charBox.className = 'ww-ability-char';

        const charTitle = document.createElement('div');
        charTitle.className = 'ww-ability-key';
        charTitle.textContent = `Char ${c}`;

        charBox.appendChild(charTitle);

        // LMB (basic attack)
        const lmbRow = document.createElement('div');
        lmbRow.className = 'ww-ability-row';
        const lmbLabel = document.createElement('span');
        lmbLabel.textContent = 'LMB';
        lmbLabel.className = 'ww-ability-label';

        const lmbInput = document.createElement('input');
        lmbInput.type = 'text';
        lmbInput.setAttribute('data-lmb', c);
        lmbInput.placeholder = 'https://... or ⚔️';
        lmbInput.value = (currentWwLmbImages[c] || '').toString();

        const lmbPreview = document.createElement('div');
        lmbPreview.className = 'ww-ability-preview';
        setPreview(lmbPreview, lmbInput.value);

        lmbInput.addEventListener('input', () => {
            currentWwLmbImages[c] = lmbInput.value.trim();
            setPreview(lmbPreview, lmbInput.value);
        });

        lmbRow.appendChild(lmbLabel);
        lmbRow.appendChild(lmbInput);
        lmbRow.appendChild(lmbPreview);
        charBox.appendChild(lmbRow);

        // Swap icon (1/2/3 keys)
        const swapRow = document.createElement('div');
        swapRow.className = 'ww-ability-row';
        const swapLabel = document.createElement('span');
        swapLabel.textContent = c;
        swapLabel.className = 'ww-ability-label';

        const swapInput = document.createElement('input');
        swapInput.type = 'text';
        swapInput.setAttribute('data-swap', c);
        swapInput.placeholder = 'https://... or 🔄';
        swapInput.value = (currentWwSwapImages[c] || '').toString();

        const swapPreview = document.createElement('div');
        swapPreview.className = 'ww-ability-preview';
        setPreview(swapPreview, swapInput.value);

        swapInput.addEventListener('input', () => {
            currentWwSwapImages[c] = swapInput.value.trim();
            setPreview(swapPreview, swapInput.value);
        });

        swapRow.appendChild(swapLabel);
        swapRow.appendChild(swapInput);
        swapRow.appendChild(swapPreview);
        charBox.appendChild(swapRow);

        // Abilities (E/Q/R)
        ['E', 'Q', 'R'].forEach(a => {
            const row = document.createElement('div');
            row.className = 'ww-ability-row';

            const label = document.createElement('span');
            label.textContent = a;
            label.className = 'ww-ability-label';

            const input = document.createElement('input');
            input.type = 'text';
            input.setAttribute('data-char', c);
            input.setAttribute('data-ability', a.toLowerCase());
            input.placeholder = 'https://... or emoji';
            const key = a.toLowerCase();
            input.value = ((currentWwAbilityImages[c] || {})[key] || '').toString();

            const preview = document.createElement('div');
            preview.className = 'ww-ability-preview';
            setPreview(preview, input.value);

            input.addEventListener('input', () => {
                if (!currentWwAbilityImages[c]) currentWwAbilityImages[c] = {};
                const v = input.value.trim();
                if (v) {
                    currentWwAbilityImages[c][key] = v;
                } else {
                    delete currentWwAbilityImages[c][key];
                }
                setPreview(preview, input.value);
            });

            row.appendChild(label);
            row.appendChild(input);
            row.appendChild(preview);
            charBox.appendChild(row);
        });

        container.appendChild(charBox);
    });
}

// Extract keys from inputs text
function extractKeysFromInputs() {
    const txt = (document.getElementById('comboInputs')?.value || '').toString();
    if (!txt.trim()) return [];

    const parts = txt.split(',').map(x => x.trim().toLowerCase()).filter(Boolean);
    const keys = new Set();

    parts.forEach(part => {
        // hold(key,time) or key{time}
        let m = part.match(/^hold\(\s*([^,]+)\s*,/i);
        if (m) {
            keys.add(m[1].trim());
            return;
        }
        m = part.match(/^([^{]+)\{/);
        if (m) {
            keys.add(m[1].trim());
            return;
        }
        // wait(...) -> skip
        if (/^wait[_a-z]*[:(\s]/i.test(part)) return;
        // [group] -> extract items
        m = part.match(/^\[(.+)\]$/);
        if (m) {
            const group = m[1];
            group.split(',').forEach(gi => {
                const gg = gi.trim();
                let mm = gg.match(/^hold\(\s*([^,]+)\s*,/i);
                if (mm) {
                    keys.add(mm[1].trim());
                    return;
                }
                mm = gg.match(/^([^{]+)\{/);
                if (mm) {
                    keys.add(mm[1].trim());
                    return;
                }
                mm = gg.match(/^wait\(\s*([^,]+)\s*,/i);
                if (mm) {
                    keys.add(mm[1].trim());
                    return;
                }
                if (!/^wait[_a-z]*[:(\s]/i.test(gg)) {
                    keys.add(gg);
                }
            });
            return;
        }
        // Plain key
        keys.add(part);
    });

    return Array.from(keys).sort();
}

function readKeyImagesFromUI() {
    const container = document.getElementById('keyImagesEditor');
    if (!container) return;
    const inputs = container.querySelectorAll('input[data-key]');
    const next = {};
    inputs.forEach(inp => {
        const k = (inp.getAttribute('data-key') || '').trim().toLowerCase();
        const url = (inp.value || '').toString().trim();
        if (k && url) next[k] = url;
    });
    currentKeyImages = next;
}

function renderKeyImagesEditor() {
    // Generic mode only
    if (currentTargetGame === 'wuthering_waves') return;

    readKeyImagesFromUI();
    const container = document.getElementById('keyImagesEditor');
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
        input.value = (currentKeyImages[k] || '').toString();

        const preview = document.createElement('div');
        preview.className = 'key-image-preview';
        const v = (currentKeyImages[k] || '').toString().trim();
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
                currentKeyImages[k] = url;
                preview.style.display = 'flex';
                if (/^https?:\/\//i.test(url)) {
                    preview.innerHTML = `<img src="${escapeHtml(url)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:24px;height:24px;object-fit:contain;" />`;
                } else {
                    preview.innerHTML = `<span>${escapeHtml(url)}</span>`;
                }
            } else {
                delete currentKeyImages[k];
                preview.style.display = 'none';
            }
        });

        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(preview);
        container.appendChild(row);
    });
}

// Editor fields update (from backend)
function setEditorFields(data) {
    document.getElementById('comboName').value = data.name || '';
    document.getElementById('comboInputs').value = data.inputs || '';
    document.getElementById('comboEnders').value = data.enders || '';
    document.getElementById('comboExpectedTime').value = data.expected_time || '';
    document.getElementById('comboUserDifficulty').value = data.user_difficulty || '';

    currentStepDisplayMode = (data.step_display_mode || 'icons').toString().trim().toLowerCase();
    if (!['icons', 'images'].includes(currentStepDisplayMode)) currentStepDisplayMode = 'icons';
    const toggle = document.getElementById('stepDisplayToggle');
    if (toggle) toggle.checked = (currentStepDisplayMode === 'images');

    currentKeyImages = (typeof data.key_images === 'object' && data.key_images !== null) ? { ...data.key_images } : {};

    // Target game & WW data
    currentTargetGame = normalizeTargetGame(data.target_game || 'generic');
    const gameSelect = document.getElementById('targetGameSelect');
    if (gameSelect) gameSelect.value = currentTargetGame;

    // WW teams list
    currentWwTeams = Array.isArray(data.ww_teams) ? [...data.ww_teams] : [];
    const teamSelect = document.getElementById('wwTeamSelect');
    if (teamSelect) {
        teamSelect.innerHTML = '<option value="">— New Team —</option>';
        currentWwTeams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.name;
            teamSelect.appendChild(opt);
        });
    }

    // Selected team (from combo assignment or active team)
    currentWwTeamId = (data.ww_team_id || '').toString().trim();
    if (teamSelect) teamSelect.value = currentWwTeamId;

    const teamNameEl = document.getElementById('wwTeamName');
    if (teamNameEl) teamNameEl.value = (data.ww_team_name || '').toString();

    currentWwDashImage = (data.ww_team_dash_image || '').toString().trim();
    currentWwSwapImages = ensureWwSlotShape(data.ww_team_swap_images);
    currentWwLmbImages = ensureWwSlotShape(data.ww_team_lmb_images);
    currentWwAbilityImages = ensureWwAbilityShape(data.ww_team_ability_images);

    syncGameUIVisibility();
    renderWwAbilityEditor({ preserveEdits: false });
}

// Status display
function updateStatus(text, color) {
    const el = document.getElementById('statusDisplay');
    if (!el) return;
    el.textContent = text || 'Status: Ready';
    el.className = 'status-' + (color || 'neutral');
}

// Stats
function updateStats(text) {
    const el = document.getElementById('statsDisplay');
    if (el) el.textContent = text || 'Stats: —';
}

function updateMinTime(text) {
    const el = document.getElementById('minTimeDisplay');
    if (el) el.textContent = text || 'Fastest possible: —';
}

function updateDifficulty(text) {
    const el = document.getElementById('difficultyDisplay');
    if (el) el.textContent = text || 'Difficulty: —';
}

function updateUserDifficulty(text) {
    const el = document.getElementById('userDifficultyDisplay');
    if (el) el.textContent = text || 'Your difficulty: —';
}

function updateAPM(text) {
    const el = document.getElementById('apmDisplay');
    if (el) el.textContent = text || 'Practical APM: —';
}

function updateAPMMax(text) {
    const el = document.getElementById('apmMaxDisplay');
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
function clearAttemptLog() {
    document.getElementById('resultsBody').innerHTML = '';
}

function addAttemptSeparator(name, attempt) {
    const body = document.getElementById('resultsBody');
    if (!body) return;
    const row = document.createElement('div');
    row.className = 'result-row separator';
    row.textContent = `—— ${name} | Attempt ${attempt} ——`;
    body.appendChild(row);
    scrollToBottom('resultsTable');
}

function addResultRow(data) {
    const body = document.getElementById('resultsBody');
    if (!body) return;
    const row = document.createElement('div');
    row.className = 'result-row';

    if (data.split_ms === 'FAIL' || data.total_ms === 'FAIL') {
        row.classList.add('fail');
    } else {
        row.classList.add('success');
    }

    row.innerHTML = `
        <span>${escapeHtml(data.input || '')}</span>
        <span>${data.split_ms != null ? data.split_ms : '—'}</span>
        <span>${data.total_ms != null ? data.total_ms : '—'}</span>
    `;

    body.appendChild(row);
    scrollToBottom('resultsTable');
}

// Failure analysis
function updateFailures(failures) {
    const body = document.getElementById('failBody');
    if (!body) return;
    body.innerHTML = '';

    if (!failures || typeof failures !== 'object') return;
    const entries = Object.entries(failures).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) return;

    entries.forEach(([reason, count]) => {
        const row = document.createElement('div');
        row.className = 'fail-row';

        const cellReason = document.createElement('span');
        cellReason.textContent = reason;
        row.appendChild(cellReason);

        const cellCount = document.createElement('span');
        cellCount.textContent = count.toString();
        row.appendChild(cellCount);

        body.appendChild(row);
    });
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

// Timeline rendering
function updateTimeline(steps) {
    lastTimelineSteps = steps;
    const container = document.getElementById('comboTimeline');
    if (!container) return;
    container.innerHTML = '';

    const BASE_STEP_WIDTH_PX = 90;
    // Duration (ms) → width scale; larger divisor = narrower tiles (e.g. 250 → 400ms ≈ 128px).
    const DURATION_WIDTH_DIVISOR = 350;
    const applyHoldWidth = (el, durationMs) => {
        const ms = Number(durationMs);
        const mult = (Number.isFinite(ms) && ms > 0) ? (ms / DURATION_WIDTH_DIVISOR) : 1;
        const w = Math.max(BASE_STEP_WIDTH_PX, BASE_STEP_WIDTH_PX * mult);
        el.style.minWidth = `${BASE_STEP_WIDTH_PX}px`;
        el.style.width = `${w}px`;
    };
    const applyWaitWidth = (el, durationMs) => {
        const ms = Number(durationMs);
        const mult = (Number.isFinite(ms) && ms > 0) ? (ms / DURATION_WIDTH_DIVISOR) : 1;
        const w = Math.max(BASE_STEP_WIDTH_PX, BASE_STEP_WIDTH_PX * mult);
        el.style.minWidth = `${BASE_STEP_WIDTH_PX}px`;
        el.style.width = `${w}px`;
    };

    const addCornerKey = (el, key) => {
        if (currentStepDisplayMode !== 'images') return;
        const k = (key || '').toString().trim();
        if (!k) return;
        const span = document.createElement('span');
        span.className = 'corner-key';
        span.textContent = k;
        el.appendChild(span);
    };

    // Shared renderer for "step group-item" tiles (used by normal group items and mini-sequences)
    // so they always look identical (images, durations, corner key, progress bars, etc.).
    const createGroupItemTile = (it, characterId) => {
        const el = document.createElement('div');
        el.className = 'step group-item';

        if (it.type === 'wait') {
            el.classList.add('wait');
            if (it.duration <= 150) el.classList.add('short-wait');
            const pct = (it.progress !== undefined && it.progress !== null) ? it.progress : (it.completed ? 100 : 0);
            el.style.setProperty('--wait-pct', `${pct}%`);
            applyWaitWidth(el, it.duration);
        } else if (it.type === 'press_wait') {
            el.classList.add('press-wait');
            if (it.duration <= 150) el.classList.add('short-wait');
            const pct = (it.progress !== undefined && it.progress !== null) ? it.progress : (it.completed ? 100 : 0);
            el.style.setProperty('--wait-pct', `${pct}%`);
            applyWaitWidth(el, it.duration);
        } else if (it.type === 'hold') {
            el.classList.add('hold');
            applyHoldWidth(el, it.duration);
            el.style.setProperty('--hold-pct', it.completed ? '100%' : '0%');
        }

        if (it.active) el.classList.add('active');
        if (it.completed) el.classList.add('completed');

        appendStepContent(el, it, characterId);

        let keyForCorner = '';
        if (it.type === 'wait' && it.wait_for) keyForCorner = it.wait_for;
        else if (it.input) keyForCorner = it.input;
        if (keyForCorner) addCornerKey(el, keyForCorner);

        return el;
    };

    if (!steps || steps.length === 0) {
        container.innerHTML = '<div class="help-text">No combo selected</div>';
        return;
    }

    let activeChar = '1';

    steps.forEach((s, idx) => {
        const tile = document.createElement('div');

        // Handle class assignment based on type
        if (s.type === 'group') {
            tile.className = 'step-group';
            if (s.active) tile.classList.add('active');
            if (s.completed) tile.classList.add('completed');
            // Add performance marks if applicable to group? Usually on items.
            if (s.mark) {
                const m = String(s.mark).toLowerCase();
                if (m === 'ok') tile.classList.add('mark-ok');
                else if (m === 'early') tile.classList.add('mark-early');
                else if (m === 'missed') tile.classList.add('mark-missed');
                else if (m === 'wrong') tile.classList.add('mark-wrong');
            }

            const items = document.createElement('div');
            items.className = 'step-group-items';

            (s.items || []).forEach(it => {
                const itInp = (it.input || '').toString().toLowerCase();
                const itWait = (it.wait_for || '').toString().toLowerCase();

                if (it.type === 'sequence') {
                    // Render a mini sequence within the group
                    const seqEl = document.createElement('div');
                    seqEl.className = 'step group-item group-item-sequence';
                    if (it.active) seqEl.classList.add('active');
                    if (it.completed) seqEl.classList.add('completed');

                    const seqItems = document.createElement('div');
                    seqItems.className = 'mini-sequence-items';
                    const seqArr = (it.items || []);
                    for (let seqIdx = 0; seqIdx < seqArr.length; seqIdx++) {
                        const seqIt = seqArr[seqIdx];

                        const seqItInp = (seqIt.input || '').toString().toLowerCase();
                        const seqItWait = (seqIt.wait_for || '').toString().toLowerCase();

                        if (['1', '2', '3'].includes(seqItInp)) activeChar = seqItInp;
                        else if (['1', '2', '3'].includes(seqItWait)) activeChar = seqItWait;

                        // Backend is the single source of truth for merge/collapse rules.
                        // (No frontend-side sequence merging here.)
                        seqItems.appendChild(createGroupItemTile(seqIt, activeChar));
                    }
                    seqEl.appendChild(seqItems);
                    items.appendChild(seqEl);
                } else {
                    // Regular group item
                    if (['1', '2', '3'].includes(itInp)) activeChar = itInp;
                    else if (['1', '2', '3'].includes(itWait)) activeChar = itWait;

                    items.appendChild(createGroupItemTile(it, activeChar));
                }
            });
            tile.appendChild(items);

        } else if (s.type === 'sequence') {
            // Sequential subgroup
            tile.className = 'step-sequence';
            if (s.active) tile.classList.add('active');
            if (s.completed) tile.classList.add('completed');

            const items = document.createElement('div');
            items.className = 'sequence-items';

            (s.items || []).forEach((it, seqIdx) => {
                const itInp = (it.input || '').toString().toLowerCase();
                const itWait = (it.wait_for || '').toString().toLowerCase();

                if (['1', '2', '3'].includes(itInp)) activeChar = itInp;
                else if (['1', '2', '3'].includes(itWait)) activeChar = itWait;

                const itEl = document.createElement('div');
                itEl.className = 'step sequence-item';
                if (it.active) itEl.classList.add('active');
                if (it.completed) itEl.classList.add('completed');

                // Render sequence item label
                appendStepContent(itEl, it, activeChar);

                items.appendChild(itEl);
            });
            tile.appendChild(items);

        } else {
            // Normal Step
            const sInp = (s.input || '').toString().toLowerCase();
            const sWait = (s.wait_for || '').toString().toLowerCase();

            if (['1', '2', '3'].includes(sInp)) activeChar = sInp;
            else if (['1', '2', '3'].includes(sWait)) activeChar = sWait;

            tile.className = 'step';
            if (s.active) tile.classList.add('active');
            if (s.completed) tile.classList.add('completed');
            if (s.mark === 'success') tile.classList.add('mark-ok');
            if (s.mark === 'fail') tile.classList.add('mark-wrong');
            if (s.mark === 'missed') tile.classList.add('mark-missed');
            if (s.mark === 'early') tile.classList.add('mark-early');

            // Add type class (wait, hold, press-wait)
            if (s.type) {
                tile.classList.add(s.type.replace('_', '-'));
            }

            // check if wait/hold/press-wait logic for pct/width
            let pct = (s.progress !== undefined) ? s.progress : (s.completed ? 100 : 0);
            if (s.type === 'wait' || s.type === 'press_wait') {
                tile.style.setProperty('--wait-pct', `${pct}%`);
                if (s.duration <= 150) tile.classList.add('short-wait');
                if (s.duration) applyWaitWidth(tile, s.duration);
            } else if (s.type === 'hold') {
                tile.style.setProperty('--hold-pct', `${pct}%`);
                if (s.duration) {
                    applyHoldWidth(tile, s.duration);
                }
            }

            // Add corner key (except group/seq handled above)
            let keyForCorner = '';
            if (s.type === 'wait' && s.wait_for) keyForCorner = s.wait_for;
            else if (s.input) keyForCorner = s.input;
            if (keyForCorner) addCornerKey(tile, keyForCorner);

            appendStepContent(tile, s, activeChar);
        }

        container.appendChild(tile);
    });

    if (autoScrollEnabled) {
        requestAnimationFrame(() => applyAutoScroll());
    }
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

function appendStepContent(parent, s, characterId) {
    const useImages = (currentStepDisplayMode === 'images') || !!document.getElementById('stepDisplayToggle')?.checked;
    const inp = (s.input || '').toString().toLowerCase();
    const label = (s.input || '').toString().toUpperCase();
    const charId = characterId || '1';

    // Helper to decide content (icon or text when no image)
    const appendIconOrText = (key, fallbackText) => {
        const svg = getMouseIconSvg(key);
        if (svg) {
            const icon = document.createElement('span');
            icon.className = 'mouse-icon';
            icon.innerHTML = svg;
            parent.appendChild(icon);
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
        parent.appendChild(span);
    };

    // Resolve image URL based on current game/mode (WW vs generic); returns null when no image.
    const resolveImage = (key) => {
        if (!useImages) return null;
        if (currentTargetGame === 'wuthering_waves') {
            return getWwImage(key, charId);
        }
        return currentKeyImages[key] || null;
    };

    // Append primary content: image if available, else icon/text.
    const appendPrimary = (key, fallbackText) => {
        const imgUrl = resolveImage(key);
        if (imgUrl) {
            parent.appendChild(createImageElement(imgUrl));
        } else {
            appendIconOrText(key, fallbackText);
        }
    };

    // Append duration/secondary text.
    const appendDuration = (text) => {
        const dur = document.createElement('span');
        dur.className = 'step-secondary';
        dur.textContent = text;
        parent.appendChild(dur);
    };

    // Single unified logic — no duplication across WW / generic / icons.
    if (s.type === 'wait' && s.mode === 'mandatory' && s.wait_for) {
        const key = s.wait_for.toLowerCase();
        appendPrimary(key, key.toUpperCase());
        appendDuration(`${s.duration}ms`);
    } else if (s.type === 'hold') {
        appendPrimary(inp, label);
        appendDuration(`hold ${s.duration}ms`);
    } else if (s.type === 'press_wait') {
        appendPrimary(inp, label);
        appendDuration(`${s.duration}ms`);
    } else if (s.type === 'wait') {
        appendDuration(`Wait ${s.duration}ms`);
    } else {
        appendPrimary(inp, label);
    }
}

function getWwImage(key, characterId) {
    const k = key.toLowerCase();
    const cid = characterId || '1';

    // Check if it's a swap key (1/2/3) - Return the swap icon for that character regardless of who is active
    if (['1', '2', '3'].includes(k)) {
        return currentWwSwapImages[k] || null;
    }
    // Check if it's RMB (dash) - shared dash image
    if (k === 'rmb') {
        return currentWwDashImage || null;
    }

    // Check if it's LMB - use active character
    if (k === 'lmb') {
        return currentWwLmbImages[cid] || null;
    }

    // Check if it's an ability (e/q/r) - use active character
    if (['e', 'q', 'r'].includes(k)) {
        if (currentWwAbilityImages[cid] && currentWwAbilityImages[cid][k]) {
            return currentWwAbilityImages[cid][k];
        }
        // Fallback: search all characters if not found for specific one (legacy behavior, optional)
        for (const c of ['1', '2', '3']) {
            if (currentWwAbilityImages[c] && currentWwAbilityImages[c][k]) {
                return currentWwAbilityImages[c][k];
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
    autoScrollEnabled = !!enabled;
    const vp = document.getElementById('comboTimelineViewport');
    const timeline = document.getElementById('comboTimeline');
    if (vp) {
        if (autoScrollEnabled) {
            vp.classList.add('auto-scroll-on');
        } else {
            vp.classList.remove('auto-scroll-on');
            if (timeline) timeline.style.transform = 'none';
        }
    }
}

function applyAutoScroll() {
    if (!autoScrollEnabled) return;
    const viewport = document.getElementById('comboTimelineViewport');
    const timeline = document.getElementById('comboTimeline');
    if (!viewport || !timeline) return;

    // Target the specific active step (item), not the group container
    const active = timeline.querySelector('.step.active');
    if (!active) return;

    const vpRect = viewport.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();

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
const comboSelector = document.getElementById('comboSelector');
if (comboSelector) {
    comboSelector.addEventListener('change', () => {
        const name = comboSelector.value;
        if (name) {
            ws.send(JSON.stringify({ type: 'select_combo', name }));
        }
    });
}

// Save/Update button
const saveBtn = document.getElementById('saveBtn');
if (saveBtn) {
    saveBtn.addEventListener('click', () => {
        readKeyImagesFromUI();
        readWwDataFromUI();

        const name = (document.getElementById('comboName')?.value || '').toString();
        const inputs = (document.getElementById('comboInputs')?.value || '').toString();
        const enders = (document.getElementById('comboEnders')?.value || '').toString();
        const expectedTime = (document.getElementById('comboExpectedTime')?.value || '').toString();
        const userDifficulty = (document.getElementById('comboUserDifficulty')?.value || '').toString();
        const toggle = document.getElementById('stepDisplayToggle');
        const mode = toggle?.checked ? 'images' : 'icons';

        ws.send(JSON.stringify({
            type: 'save_combo',
            name,
            inputs,
            enders,
            expected_time: expectedTime,
            user_difficulty: userDifficulty,
            step_display_mode: mode,
            key_images: currentKeyImages,
            target_game: currentTargetGame,
            ww_team_id: currentWwTeamId || ''
        }));
    });
}

// New combo button
const newBtn = document.getElementById('newBtn');
if (newBtn) {
    newBtn.addEventListener('click', () => {
        ws.send(JSON.stringify({ type: 'new_combo' }));
    });
}

// Delete combo button
const deleteBtn = document.getElementById('deleteBtn');
if (deleteBtn) {
    attachTwoClickConfirm(deleteBtn, {
        confirmText: 'Confirm delete',
        onConfirm: () => {
            const name = (document.getElementById('comboName')?.value || '').toString();
            if (name) {
                ws.send(JSON.stringify({ type: 'delete_combo', name }));
            }
        }
    });
}

// Clear history button
const clearBtn = document.getElementById('clearBtn');
if (clearBtn) {
    attachTwoClickConfirm(clearBtn, {
        confirmText: 'Clear all',
        onConfirm: () => {
            ws.send(JSON.stringify({ type: 'clear_history' }));
        }
    });
}

function scrollToBottom(el) {
    if (!el) return;
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

const autoScrollToggleEl = document.getElementById('autoScrollToggle');
if (autoScrollToggleEl) {
    setAutoScrollEnabled(autoScrollToggleEl.checked);
    autoScrollToggleEl.addEventListener('change', () => {
        setAutoScrollEnabled(autoScrollToggleEl.checked);
        if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
    });
}

window.addEventListener('resize', () => {
    if (autoScrollEnabled) applyAutoScroll();
});

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
        // Send target game change to backend immediately for stateless operation
        ws.send(JSON.stringify({
            type: 'update_target_game',
            target_game: currentTargetGame
        }));
        syncGameUIVisibility();
        renderWwAbilityEditor({ preserveEdits: true });
        if (lastTimelineSteps) updateTimeline(lastTimelineSteps);
    });
}

const wwTeamSelectEl = document.getElementById('wwTeamSelect');
if (wwTeamSelectEl) {
    wwTeamSelectEl.addEventListener('change', () => {
        currentWwTeamId = (wwTeamSelectEl.value || '').toString();
        // Send both team_id AND target_game for stateless operation
        ws.send(JSON.stringify({
            type: 'select_team',
            team_id: currentWwTeamId,
            target_game: currentTargetGame
        }));
    });
}

const saveTeamBtn = document.getElementById('saveTeamBtn');
if (saveTeamBtn) {
    saveTeamBtn.addEventListener('click', () => {
        readWwDataFromUI();
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
    attachTwoClickConfirm(deleteTeamBtn, {
        confirmText: 'Confirm delete',
        onConfirm: () => {
            if (!currentWwTeamId) return;
            ws.send(JSON.stringify({ type: 'delete_team', team_id: currentWwTeamId }));
        }
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