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

// Hold progress (client-side animation)
let holdAnim = {
    active: false,
    requiredMs: 0,
    startedAt: 0
};
let holdRafId = null;

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
}

// Timeline visualization
function updateTimeline(steps) {
    const timeline = document.getElementById('comboTimeline');
    timeline.innerHTML = '';
    
    steps.forEach((step, idx) => {
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
        
        if (step.type === 'wait') {
            div.classList.add('wait');
            const mode = (step.mode || 'soft').toLowerCase();
            const label = (mode === 'hard') ? 'wait hard' : 'wait';
            div.textContent = `${label}\n≥${step.duration}ms`;
        } else if (step.type === 'hold') {
            div.classList.add('hold');
            div.textContent = `${step.input}\nhold ${step.duration}ms`;
            // Whole-tile fill progress (default 0%, completed = 100%)
            div.style.setProperty('--hold-pct', step.completed ? '100%' : '0%');
        } else {
            div.textContent = step.input;
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
    ws.send(JSON.stringify({
        type: 'save_combo',
        name: document.getElementById('comboName').value,
        inputs: document.getElementById('comboInputs').value,
        enders: document.getElementById('comboEnders').value,
        expected_time: document.getElementById('comboExpectedTime').value,
        user_difficulty: document.getElementById('comboUserDifficulty').value
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
        case 'hit':
            addResultRow(msg);
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