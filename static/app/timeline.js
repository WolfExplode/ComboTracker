// Timeline-only view for OBS (Browser Source or Window Capture)
if (new URLSearchParams(window.location.search).get('view') === 'timeline') {
    document.title = 'WuWa Combo Tracker – Timeline';
    document.body.classList.add('timeline-window-view');
}

function getTimelineUrl() {
    const path = window.location.pathname || '/';
    return window.location.origin + path + (path.includes('?') ? '&' : '?') + 'view=timeline';
}

// Hold / Wait animations (global UI state, not per-combo)
let holdAnim = { active: false, requiredMs: 0, startedAt: 0 };
let holdRafId = null;

let waitAnim = { active: false, requiredMs: 0, startedAt: 0 };
let waitRafId = null;

let spamAnim = { active: false, requiredMs: 0, startedAt: 0 };
let spamRafId = null;

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
    // Select active wait steps, including hold-body wait chips.
    const stepEl = document.querySelector(
        '.step.wait.active, .step.press-wait.active, ' +
        '.hold-body-chip.wait.active, .hold-body-chip.press-wait.active .hold-body-inline-wait'
    );
    if (!stepEl) {
        waitRafId = requestAnimationFrame(tickWaitAnimation);
        return;
    }

    const elapsed = performance.now() - waitAnim.startedAt;
    const req = Math.max(1, Number(waitAnim.requiredMs) || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    const progressProperty = stepEl.classList.contains('hold-body-chip')
        || stepEl.classList.contains('hold-body-inline-wait')
        ? '--chip-wait-pct'
        : '--wait-pct';
    stepEl.style.setProperty(progressProperty, `${pct}%`);
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

function stopSpamAnimation() {
    spamAnim.active = false;
    if (spamRafId !== null) {
        cancelAnimationFrame(spamRafId);
        spamRafId = null;
    }
}

function tickSpamAnimation() {
    if (!spamAnim.active) {
        spamRafId = null;
        return;
    }
    const stepEl = document.querySelector(
        '.step.spam.active, .hold-body-chip.spam.active .hold-body-inline-wait'
    );
    if (!stepEl) {
        spamRafId = requestAnimationFrame(tickSpamAnimation);
        return;
    }

    const elapsed = performance.now() - spamAnim.startedAt;
    const req = Math.max(1, Number(spamAnim.requiredMs) || 1);
    const pct = Math.max(0, Math.min(100, (elapsed / req) * 100));
    const progressProperty = stepEl.classList.contains('hold-body-inline-wait')
        ? '--chip-wait-pct'
        : '--wait-pct';
    stepEl.style.setProperty(progressProperty, `${pct}%`);
    updateStatus(`Spamming... ${Math.round(pct)}%`, 'wait');

    if (pct >= 100) {
        spamRafId = null;
        return;
    }
    spamRafId = requestAnimationFrame(tickSpamAnimation);
}

function startSpamAnimation(requiredMs) {
    stopSpamAnimation();
    spamAnim.active = true;
    spamAnim.requiredMs = Math.max(1, Number(requiredMs) || 1);
    spamAnim.startedAt = performance.now();
    spamRafId = requestAnimationFrame(tickSpamAnimation);
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
    let quoted = false; // inside a "Move Name" (it may hold commas)
    for (const ch of (str || '')) {
        if (ch === '"') { quoted = !quoted; buf += ch; continue; }
        if (quoted) { buf += ch; continue; }
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
// A step can carry the move it casts, picked from its right-click menu:
// lmb "Basic: Origin Calculus 2 (Dodge Counter)". Same as split_move_name in parser.py.
function splitMoveName(token) {
    const m = String(token || '').match(/\s*"([^"]*)"\s*$/);
    if (!m) return [String(token || ''), null];
    return [String(token).slice(0, m.index), m[1]];
}

function withMoveName(token, name) {
    const base = splitMoveName(token)[0].trim();
    return name ? `${base} "${name.replace(/"/g, '')}"` : base;
}

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

// -wait:Xs (alone or after a key): the next press may cut the wait short.
function markOptionalWait(el, s) {
    const optionalWait = s.wait_optional || (s.type === 'wait' && s.optional);
    if (!optionalWait) return;
    el.classList.add('wait-optional');
    el.title = 'Optional wait: pressing the next key early is fine';
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
        return [`${s.optional ? '-' : ''}${key}`, `${s.wait_optional ? '-' : ''}wait:${formatDurationToken(durMs)}`];
    }

    if (s.type === 'hold') {
        // One token: hold(key, durMs)
        const key = field === 'key' ? val : (s.input || '').toLowerCase();
        if (!key) return null;
        const durMs = field === 'duration' ? parseDurationToMs(val) : s.duration;
        if (!durMs) return null;
        return [`hold(${key}, ${formatDurationToken(durMs)})`];
    }

    if (s.type === 'spam') {
        const key = field === 'key' ? val : (s.input || '').toLowerCase();
        if (!key) return null;
        const durMs = field === 'duration' ? parseDurationToMs(val) : s.duration;
        if (!durMs) return null;
        return [`spam(${key}, ${formatDurationToken(durMs)})`];
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
        return [`${s.optional ? '-' : ''}wait:${formatDurationToken(durMs)}`];
    }

    return null;
}

/** Snapshot of the fields save_combo persists, used to restore prior state on undo. */
function captureComboSnapshot() {
    return {
        name: (getEl('comboName')?.value || '').toString(),
        inputs: (getEl('comboInputs')?.value || '').toString(),
        enders: (getEl('comboEnders')?.value || '').toString(),
        expected_time: (getEl('comboExpectedTime')?.value || '').toString(),
        user_difficulty: (getEl('comboUserDifficulty')?.value || '').toString(),
        step_display_mode: getEl('stepDisplayToggle')?.checked ? 'images' : 'icons',
        key_images: { ...appState.keyImages },
        demo_video: (getEl('comboDemoVideo')?.value || '').toString().trim(),
        target_game: appState.targetGame,
        ww_team_id: appState.wwTeamId || '',
    };
}

const EDIT_STEPS_UNDO_LIMIT = 50;

/** Call before applying an Edit Steps mutation (delete/reorder/inline edit) so it can be undone. */
function pushEditStepsUndoSnapshot() {
    appState.editStepsUndoStack.push(captureComboSnapshot());
    if (appState.editStepsUndoStack.length > EDIT_STEPS_UNDO_LIMIT) appState.editStepsUndoStack.shift();
}

/** Ctrl/Cmd+Z while Edit Steps is on: restore the state captured before the last mutation. */
function undoLastEditStep() {
    const snapshot = appState.editStepsUndoStack.pop();
    if (!snapshot) return false;
    sendMessage('save_combo', snapshot);
    updateStatus('Undid last step edit.', 'neutral');
    return true;
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
    const [oldSourceToken, moveName] = splitMoveName(currentTokens[minSrc]);
    const newTokens = reconstructTokensForEdit(s, field, newValue, oldSourceToken.trim());
    if (!newTokens) return false;
    if (moveName) newTokens[0] = withMoveName(newTokens[0], moveName); // keep a picked move

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

    pushEditStepsUndoSnapshot();
    saveComboInputs(result.join(', '));
    return true;
}

/** Put new inputs in the Inputs box and save them, the same way as the Save button. */
function saveComboInputs(newInputs) {
    const inputsEl = getEl('comboInputs');
    if (!inputsEl) return;
    inputsEl.value = newInputs;
    if (typeof updateComboInputHighlight === 'function') updateComboInputHighlight();
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
}

/**
 * Pick (or clear, with name null) the move for the timeline step at runtimeIdx: writes
 * lmb "Basic: X 1" into its input token and saves. Returns false if it can't be mapped.
 */
function setStepMoveName(runtimeIdx, name) {
    const inputsEl = getEl('comboInputs');
    if (!inputsEl) return false;
    const tokens = splitInputsTokens(inputsEl.value || '');
    const src = buildRuntimeToSourceMap(tokens)[runtimeIdx];
    if (!src || !src.length) return false;
    const i = Math.min(...src);
    tokens[i] = withMoveName(tokens[i], name);
    pushEditStepsUndoSnapshot();
    saveComboInputs(tokens.join(', '));
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
        const tok = splitMoveName(tokens[i])[0].trim().toLowerCase();
        if (!tok) { i++; continue; }

        // press + following soft/hard wait -> one runtime SequenceNode (press_wait tile)
        if (!tok.startsWith('wait') && !tok.startsWith('-wait') && !tok.startsWith('hold(') && !tok.startsWith('spam(') && !tok.startsWith('[') && !tok.startsWith('{')) {
            // Could be a plain press followed by wait:Xs
            if (i + 1 < tokens.length) {
                const nxt = splitMoveName(tokens[i + 1])[0].trim().toLowerCase();
                if (nxt.startsWith('wait:') || nxt.startsWith('-wait:')) {
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
 * Make a span inline-editable on double-click.
 * `s` = step dict, `field` = 'key' | 'duration', `runtimeIdx` = first runtime index.
 */
function attachInlineEdit(span, s, field, runtimeIdx) {
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

    // Names each step's move ("Zani Basic 2"); steps must be labeled in timeline order.
    // A later step can rename an earlier one (E, E, E on Augusta becomes Strike, Leap, Plunge).
    // It also runs with names hidden, for the right-click "which move is this" list.
    const showNames = !!appState.showMoveNames;
    const labelMove = createWwMoveLabeler(wwSlotNames());
    const moveLabelEls = [];
    // Moves picked from the right-click menu live in the saved inputs: lmb "Basic: X 1".
    const savedTokens = splitInputsTokens(appState.savedInputs || '');
    const savedSrcMap = buildRuntimeToSourceMap(savedTokens);
    // The one input token behind a tile, or -1 (a collapsed chain or group spans several).
    const tokenForStep = (step) => {
        const idx = Array.isArray(step && step.step_indices) ? step.step_indices : [];
        const toks = new Set(idx.map((r) => (savedSrcMap[r] && savedSrcMap[r].length ? Math.min(...savedSrcMap[r]) : -1)));
        return toks.size === 1 ? [...toks][0] : -1;
    };
    if (showNames) {
        labelMove.onRevise = (index, text) => {
            const el = moveLabelEls[index];
            if (el) { el.title = el.title.replace(el.textContent, text); el.textContent = text; }
        };
    }
    function appendMoveLabel(tile, step, slot) {
        const tokIdx = tokenForStep(step);
        const chosen = tokIdx >= 0 ? splitMoveName(savedTokens[tokIdx])[1] : null;
        const text = labelMove(step, slot, chosen);
        if (tokIdx >= 0 && text && labelMove.choices.length > 1) {
            tile._moveChoice = { runtimeIdx: step.step_indices[0], choices: labelMove.choices, inputs: labelMove.choiceInputs, chosen };
        }
        if (!text || !showNames) return;
        const el = document.createElement('span');
        el.className = 'step-move';
        el.textContent = text;
        const conc = slot ? labelMove.concerto(slot) : 0;
        el.title = conc > 0 ? `${text}\nConcerto after this ≈ ${Math.round(conc)}/100` : text;
        tile.appendChild(el);
        moveLabelEls.push(el);
    }

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
    // Drag-to-reorder: the first runtime index stored in step_indices is used as the drag handle identifier.
    const attachStepDragControl = (el, stepIndices) => {
        const indices = parseStepIndices(stepIndices);
        if (indices.length === 0) return;
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
            pushEditStepsUndoSnapshot();
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
        } else if (it.type === 'press_wait' || it.type === 'spam') {
            el.classList.add('press-wait');
            if (it.type === 'spam') el.classList.add('spam');
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
        markOptionalWait(el, it);

        if (it.active) el.classList.add('active');
        if (it.completed) el.classList.add('completed');

        appendStepContent(el, it, characterId, ctx);
        appendMoveLabel(el, it, characterId);

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
            markOptionalWait(itEl, it);
            if (it.active) itEl.classList.add('active');
            if (it.completed) itEl.classList.add('completed');
            appendStepContent(itEl, it, nextChar, ctx);
            appendMoveLabel(itEl, it, nextChar);
            items.appendChild(itEl);
        });
        tile.appendChild(items);
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
        markOptionalWait(tile, s);
        let pct = (s.progress !== undefined) ? s.progress : (s.completed ? 100 : 0);
        if (s.type === 'wait' || s.type === 'press_wait' || s.type === 'spam') {
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
        appendMoveLabel(tile, s, nextChar);
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
    if (s.type === 'spam') return `Spam ${inp} ${dur}ms`;
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
                if (itemType === 'press_wait' || itemType === 'spam') {
                    chip.classList.add('press-wait');
                    if (itemType === 'spam') chip.classList.add('spam');
                    const inlineWait = document.createElement('span');
                    inlineWait.className = 'hold-body-inline-wait';
                    inlineWait.textContent = itemType === 'spam'
                        ? `spam ${Number(it.duration || 0)}ms`
                        : `${Number(it.duration || 0)}ms`;
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
    } else if (s.type === 'spam') {
        appendPrimary(inp, label);
        appendDuration(`spam ${s.duration}ms`);
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
        img.innerHTML = `<img class="key-step-image" src="${escapeHtml(url)}" alt="" loading="lazy" referrerpolicy="no-referrer" draggable="false" />`;
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


// ---------------------------------------------------------------------------
// Right-click menu on Combo Steps tiles (replaces the browser's menu there only; text boxes
// keep theirs). Add entries in tileMenuItems.
// ---------------------------------------------------------------------------

// Entries for a tile: { label, run, danger?, checked?, separator? }.
function tileMenuItems(tile, indices) {
    const items = [];
    // Which move this step is, when the keys alone could mean more than one (dodge into A1 vs
    // Dodge Counter, Heavy 1 vs Heavy 2, ...). The pick is saved in the inputs.
    const mc = tile._moveChoice;
    if (mc) {
        items.push({ heading: 'Which move is this?' });
        mc.choices.forEach((name, i) => items.push({
            label: wwShortMoveName(name),
            hint: mc.inputs[i],
            title: name,
            checked: name === mc.chosen,
            run: () => setStepMoveName(mc.runtimeIdx, name === mc.chosen ? null : name),
        }));
        if (mc.chosen) items.push({ label: 'Back to the guess', run: () => setStepMoveName(mc.runtimeIdx, null) });
        items.push({ separator: true });
    }
    items.push({
        label: 'Delete step',
        danger: true,
        run: () => {
            pushEditStepsUndoSnapshot();
            sendMessage('delete_timeline_step', { step_indices: indices });
        },
    });
    return items;
}

let tileMenuEl = null;

function closeTileMenu() {
    if (!tileMenuEl) return;
    tileMenuEl.remove();
    tileMenuEl = null;
}

function openTileMenu(x, y, tile, indices) {
    closeTileMenu();
    const menu = document.createElement('div');
    menu.className = 'ctx-menu';
    menu.setAttribute('role', 'menu');
    tileMenuItems(tile, indices).forEach((item) => {
        if (item.separator) {
            const hr = document.createElement('div');
            hr.className = 'ctx-menu-sep';
            menu.appendChild(hr);
            return;
        }
        if (item.heading) {
            const h = document.createElement('div');
            h.className = 'ctx-menu-heading';
            h.textContent = item.heading;
            menu.appendChild(h);
            return;
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `ctx-menu-item${item.danger ? ' danger' : ''}${item.checked ? ' checked' : ''}`;
        btn.setAttribute('role', item.checked !== undefined ? 'menuitemradio' : 'menuitem');
        if (item.checked !== undefined) btn.setAttribute('aria-checked', String(!!item.checked));
        if (item.title) btn.title = item.title;
        btn.textContent = item.label;
        if (item.hint) {
            const hint = document.createElement('code');
            hint.className = 'ctx-menu-hint';
            hint.textContent = item.hint;
            btn.appendChild(hint);
        }
        btn.addEventListener('click', () => { closeTileMenu(); item.run(); });
        menu.appendChild(btn);
    });
    document.body.appendChild(menu);
    // Keep it on screen.
    const r = menu.getBoundingClientRect();
    menu.style.left = `${Math.max(4, Math.min(x, window.innerWidth - r.width - 4))}px`;
    menu.style.top = `${Math.max(4, Math.min(y, window.innerHeight - r.height - 4))}px`;
    tileMenuEl = menu;
    menu.querySelector('button')?.focus();
}

{
    const timeline = getEl('comboTimeline');
    if (timeline) {
        timeline.addEventListener('contextmenu', (ev) => {
            const tile = [...timeline.children].find((c) => c.contains(ev.target));
            const indices = (tile?.dataset.stepIndices || '')
                .split(',').map((v) => Number.parseInt(v, 10)).filter((v) => Number.isFinite(v) && v >= 0);
            if (!indices.length) return;
            ev.preventDefault();
            openTileMenu(ev.clientX, ev.clientY, tile, indices);
        });
    }
    document.addEventListener('mousedown', (ev) => { if (tileMenuEl && !tileMenuEl.contains(ev.target)) closeTileMenu(); }, true);
    document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') closeTileMenu(); });
    window.addEventListener('blur', closeTileMenu);
    window.addEventListener('resize', closeTileMenu);
    // wheel, not scroll: the timeline scrolls itself on updates, which shouldn't close the menu.
    document.addEventListener('wheel', closeTileMenu, { passive: true });
}
