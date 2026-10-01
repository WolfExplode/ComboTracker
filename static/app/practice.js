// Practice page: stat cards, status line, attempt log and the Combo Steps toggles.

// Status display
function updateStatus(text, color) {
    const el = getEl('statusDisplay');
    if (!el) return;
    el.textContent = text || 'Status: Ready';
    el.className = 'status-' + (color || 'neutral');
}

// Stats: the backend sends "Label: value" texts; cards show just the value.
function setStatValue(id, text) {
    const el = getEl(id)?.querySelector('.stat-value');
    if (!el) return;
    const raw = (text || '').toString();
    const i = raw.indexOf(':');
    const value = (i >= 0 ? raw.slice(i + 1) : raw).trim();
    el.textContent = value || '—';
    el.title = raw;
}

// e.g. "Stats: 3 success / 1 fail (75%) | Best: 12.3s | Avg: 13.0s"
function updateStats(text) {
    const parts = (text || '').toString().replace(/^Stats:\s*/, '').split('|').map(s => s.trim());
    const result = (parts[0] || '').replace(/ success \/ /, ' / ').replace(/ fail/, '');
    setStatValue('statsResult', /^0 \/ 0\b/.test(result) ? '' : result); // no attempts yet
    setStatValue('statsBest', parts[1] || '');
    setStatValue('statsAvg', parts[2] || '');
}

function updateMinTime(text) {
    // Drop the "(53700ms)" repeat of the value to keep the card short.
    setStatValue('minTimeDisplay', (text || '').toString().replace(/\s*\(\d+ms\)\s*$/, ''));
}

function updateDifficulty(text) {
    setStatValue('difficultyDisplay', text);
}

function updateUserDifficulty(text) {
    setStatValue('userDifficultyDisplay', text);
}

function updateAPM(text) {
    setStatValue('apmDisplay', text);
}

function updateAPMMax(text) {
    setStatValue('apmMaxDisplay', text);
}

function setDifficultyColor(el, value) {
    if (!el) return;
    el.classList.remove('diff-easy', 'diff-med', 'diff-hard', 'diff-insane');
    if (value === null || value === undefined || value === '') return; // unrated, not "easy"
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
    getEl('attemptLogOffNote')?.classList.toggle('hidden', logAttemptsEnabled);
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

// Clear history button
const clearBtn = getEl('clearBtn');
if (clearBtn) {
    attachTwoClickConfirm(clearBtn, {
        confirmText: 'Click again to clear',
        onConfirm: () => {
            sendMessage('clear_history');
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

const moveNamesToggleEl = getEl('moveNamesToggle');
if (moveNamesToggleEl) {
    moveNamesToggleEl.checked = !!appState.showMoveNames;
    moveNamesToggleEl.addEventListener('change', () => {
        appState.showMoveNames = moveNamesToggleEl.checked;
        writeStoredFlag('showMoveNames', appState.showMoveNames);
        refreshTimelineIfLoaded();
    });
}

const stepEditToggleEl = getEl('stepEditToggle');
if (stepEditToggleEl) {
    stepEditToggleEl.checked = !!appState.stepEditMode;
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

const legendBtn = getEl('legendBtn');
if (legendBtn) {
    legendBtn.addEventListener('click', () => {
        const panel = getEl('legendPanel');
        if (!panel) return;
        const open = panel.classList.toggle('hidden') === false;
        legendBtn.setAttribute('aria-expanded', String(open));
    });
}

const openTimelineWindowBtn = getEl('openTimelineWindowBtn');
if (openTimelineWindowBtn) {
    openTimelineWindowBtn.addEventListener('click', () => {
        window.open(getTimelineUrl(), 'combo-tracker-timeline', 'width=900,height=400,menubar=no,toolbar=no');
    });
}

window.addEventListener('resize', () => {
    if (appState.autoScrollEnabled) applyAutoScroll();
});
