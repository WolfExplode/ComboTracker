// Characters page: every character's moves (encore.moe) and community team rotations
// (AntoCrasher's calc compilation + rotation transcripts), served by ui_server.py under /api/ww/.

const WS_URL = 'ws://localhost:8765';
const ELEMENTS = ['Aero', 'Electro', 'Fusion', 'Glacio', 'Havoc', 'Spectro'];
const MAX_SKILL_LEVEL = 10;

// Abbreviations from AntoCrasher's rotation hub; each rotation's own list (if any) wins.
const DEFAULT_GLOSSARY = {
    ba: 'basic attack', fba: 'forte basic attack', uba: 'ultimate basic attack', ha: 'heavy attack',
    fha: 'forte heavy attack', skill: 'resonance skill', eskill: 'enhanced skill', fskill: 'forte skill',
    lib: 'liberation', echo: 'echo skill', tbs: 'tune break skill', nf: 'nightfall (Zani)',
    dash: 'dodge', swap: 'swap to the next character', outro: 'outro skill (swap out)',
};

const state = {
    roster: [],
    rotations: [],          // [{character, thoughts, teams: [...]}]
    rotationsSource: '',
    mine: [],               // nameTokens() of characters saved in the tracker
    element: '',
    query: '',
    selectedId: null,
    tab: 'moves',
    level: MAX_SKILL_LEVEL,
    kits: new Map(),
    transcripts: new Map(),
    socket: null,
};

const $ = (id) => document.getElementById(id);

function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// "Rover: Spectro" / "Rover" / "rover" all compare as "rover".
function nameKey(name) {
    return String(name || '').toLowerCase().split(/[:(]/)[0].trim();
}

// "Spectro_Rover" -> "rover spectro", "Yangyang: Xuanling" -> "xuanling yangyang" (word order ignored).
function nameTokens(name) {
    return String(name || '').toLowerCase().split(/[^a-z]+/).filter(Boolean).sort().join(' ');
}

// Saved tracker names are typed by hand ("Pheobe", "Agusta"), so allow one typo or swapped pair.
function closeEnough(a, b) {
    if (a === b) return true;
    if (Math.min(a.length, b.length) < 5 || Math.abs(a.length - b.length) > 1) return false;
    const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
    for (let j = 1; j <= b.length; j++) d[0][j] = j;
    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
        }
    }
    return d[a.length][b.length] <= 1;
}

function isMine(c) {
    const t = nameTokens(c.name);
    return state.mine.some((m) => closeEnough(m, t));
}

function safeUrl(u) {
    return /^https:\/\//i.test(String(u || '')) ? String(u) : '';
}

function fmtNum(n) {
    return n == null ? '—' : Math.round(n).toLocaleString();
}

function showNotice(text, kind = 'info') {
    const el = $('notice');
    if (!text) {
        el.classList.add('hidden');
        return;
    }
    el.textContent = text;
    el.className = `notice ${kind}`;
}

async function api(path, refresh = false) {
    const url = `/api/ww/${path}${refresh ? (path.includes('?') ? '&' : '?') + 'refresh=1' : ''}`;
    const res = await fetch(url);
    const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    if (data.stale) showNotice(`Showing saved data; couldn't refresh (${data.error}).`, 'warn');
    return data;
}

// ---------------------------------------------------------------------------
// Tracker connection (optional): marks your characters and saves rotations as combos
// ---------------------------------------------------------------------------

function connectTracker() {
    let socket;
    try {
        socket = new WebSocket(WS_URL);
    } catch {
        return;
    }
    socket.onmessage = (ev) => {
        let msg;
        try {
            msg = JSON.parse(ev.data);
        } catch {
            return;
        }
        // The character list rides along in the editor payload (init and later editor updates).
        const chars = (msg.editor && msg.editor.ww_characters) || msg.ww_characters;
        if (Array.isArray(chars)) {
            state.mine = chars.map((c) => nameTokens(c.name || c.name_key));
            renderRoster();
        }
        if (msg.type === 'status' && msg.text && msg.color === 'fail') {
            showNotice(msg.text, 'warn');
        }
    };
    socket.onopen = () => (state.socket = socket);
    socket.onclose = () => {
        state.socket = null;
        setTimeout(connectTracker, 3000);
    };
}

// ---------------------------------------------------------------------------
// Roster
// ---------------------------------------------------------------------------

function renderFilters() {
    const chips = ['', ...ELEMENTS].map((el) => {
        const label = el || 'All';
        const on = state.element === el ? ' on' : '';
        const dot = el ? `<span class="dot el-${el.toLowerCase()}"></span>` : '';
        return `<button type="button" class="chip${on}" data-el="${esc(el)}">${dot}${label}</button>`;
    });
    $('elementFilter').innerHTML = chips.join('');
}

function rosterCard(c) {
    const sel = c.id === state.selectedId ? ' selected' : '';
    const icon = safeUrl(c.icon);
    return `<button type="button" class="char${sel}" data-id="${c.id}" title="${esc(c.name)} · ${esc(c.element)} ${esc(c.weapon)}">
        ${icon ? `<img src="${esc(icon)}" alt="" loading="lazy">` : '<span class="noimg"></span>'}
        <span class="char-name"><span class="dot el-${esc(String(c.element).toLowerCase())}"></span>${esc(c.name)}</span>
    </button>`;
}

function renderRoster() {
    const q = state.query.toLowerCase();
    const list = state.roster.filter(
        (c) => (!state.element || c.element === state.element) && (!q || c.name.toLowerCase().includes(q))
    );
    const mine = list.filter(isMine);
    const rest = list.filter((c) => !isMine(c));
    let html = '';
    if (mine.length) html += `<h2>Your characters</h2><div class="grid">${mine.map(rosterCard).join('')}</div>`;
    if (rest.length) html += `${mine.length ? '<h2>Everyone else</h2>' : ''}<div class="grid">${rest.map(rosterCard).join('')}</div>`;
    $('roster').innerHTML = html || '<p class="muted">No characters match.</p>';
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

function rotationsFor(name) {
    const key = nameKey(name);
    const main = state.rotations.find((g) => nameKey(g.character) === key);
    const featured = [];
    for (const g of state.rotations) {
        if (g === main) continue;
        for (const t of g.teams) {
            if (t.members.slice(1).some((m) => nameKey(m) === key)) featured.push({ ...t, main: g.character });
        }
    }
    featured.sort((a, b) => (b.dps || 0) - (a.dps || 0));
    return { main, featured };
}

async function selectCharacter(id, { pushHash = true } = {}) {
    state.selectedId = id;
    if (pushHash) history.replaceState(null, '', `#${id}`);
    renderRoster();
    const c = state.roster.find((x) => x.id === id);
    if (!c) return;
    const detail = $('detail');
    detail.innerHTML = `${detailHeader(c)}<div class="loading">Loading ${esc(c.name)}'s kit…</div>`;
    if (window.innerWidth < 900) detail.scrollIntoView({ block: 'start', behavior: 'smooth' });
    try {
        if (!state.kits.has(id)) state.kits.set(id, await api(`kit/${id}`));
    } catch (e) {
        if (state.selectedId === id) detail.innerHTML = `${detailHeader(c)}<div class="error">Couldn't load the kit: ${esc(e.message)}</div>`;
        return;
    }
    if (state.selectedId === id) renderDetail();
}

function detailHeader(c) {
    const icon = safeUrl(c.icon);
    const { main, featured } = rotationsFor(c.name);
    const count = (main ? main.teams.length : 0) + featured.length;
    return `<div class="detail-head">
        ${icon ? `<img src="${esc(icon)}" alt="">` : ''}
        <div>
            <h2>${esc(c.name)}</h2>
            <div class="muted"><span class="dot el-${esc(String(c.element).toLowerCase())}"></span>${esc(c.element)} · ${esc(c.weapon)} · ${'★'.repeat(c.rarity || 0)}</div>
        </div>
    </div>
    <div class="tabs" role="tablist">
        <button type="button" role="tab" data-tab="moves" class="${state.tab === 'moves' ? 'on' : ''}">Moves</button>
        <button type="button" role="tab" data-tab="rotations" class="${state.tab === 'rotations' ? 'on' : ''}">Team rotations <span class="count">${count}</span></button>
    </div>`;
}

function renderDetail() {
    const c = state.roster.find((x) => x.id === state.selectedId);
    if (!c) return;
    const body = state.tab === 'moves' ? renderMoves(state.kits.get(c.id)) : renderRotations(c);
    $('detail').innerHTML = detailHeader(c) + body;
}

// --- Moves ------------------------------------------------------------------

function formatDescription(text) {
    return String(text || '')
        .split(/\n+/)
        .map((line) => {
            const heading = line.match(/^\*\*(.+)\*\*$/);
            if (heading) return `<h5>${esc(heading[1])}</h5>`;
            return `<p>${esc(line).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</p>`;
        })
        .join('');
}

function renderMoves(kit) {
    if (!kit) return '';
    const levels = Array.from({ length: MAX_SKILL_LEVEL }, (_, i) => i + 1)
        .map((l) => `<option value="${l}"${l === state.level ? ' selected' : ''}>Lv ${l}</option>`)
        .join('');
    const cards = kit.skills.map((s) => {
        const icon = safeUrl(s.icon);
        const rows = s.multipliers
            .map((m) => {
                const v = m.values[Math.min(state.level, m.values.length) - 1] ?? m.values[0] ?? '';
                return `<tr><td>${esc(m.name)}</td><td>${esc(v)}</td></tr>`;
            })
            .join('');
        const open = s.type === 'Normal Attack' || s.type === 'Resonance Skill' || s.type === 'Resonance Liberation';
        return `<details class="skill"${open ? ' open' : ''}>
            <summary>
                ${icon ? `<img src="${esc(icon)}" alt="" loading="lazy">` : '<span class="noimg"></span>'}
                <span class="skill-title"><span class="skill-type">${esc(s.type)}</span><span class="skill-name">${esc(s.name)}</span></span>
                ${s.key ? `<span class="keycap">${esc(s.key)}</span>` : ''}
            </summary>
            <div class="skill-body">
                <div class="desc">${formatDescription(s.description)}</div>
                ${rows ? `<table class="mult"><thead><tr><th>Hit</th><th>Lv ${state.level}</th></tr></thead><tbody>${rows}</tbody></table>` : ''}
            </div>
        </details>`;
    });
    return `<div class="toolbar">
            <label>Skill level <select id="levelSelect">${levels}</select></label>
            <a class="src" href="${esc(safeUrl(kit.source))}" target="_blank" rel="noopener">Source: encore.moe</a>
        </div>
        ${cards.join('') || '<p class="muted">No skills listed for this character.</p>'}`;
}

// --- Rotations ----------------------------------------------------------------

function teamRow(t, idx, showMain) {
    const video = safeUrl(t.video);
    const calc = safeUrl(t.calc_sheet);
    const hasTranscript = !!safeUrl(t.transcript);
    const best = teamRow.best || 1;
    const pct = Math.max(4, Math.round(((t.dps || 0) / best) * 100));
    return `<div class="team" data-idx="${idx}">
        <div class="team-main">
            <div class="team-name">${esc(showMain ? t.team : t.team.replace(/\s*\([^)]*\)\s*$/, ''))}</div>
            <div class="team-meta">
                ${t.style ? `<span class="tag">${esc(t.style)}</span>` : ''}
                ${t.setup ? `<span class="tag subtle">${esc(t.setup)}</span>` : ''}
                ${t.author ? `<span class="muted">by ${esc(t.author)}</span>` : ''}
            </div>
            <div class="dps-bar" title="${fmtNum(t.dps)} DPS over a ${t.rotation_time || '?'}s fight"><span style="width:${pct}%"></span></div>
        </div>
        <div class="team-num"><strong>${fmtNum(t.dps)}</strong><span class="muted">DPS</span></div>
        <div class="team-actions">
            ${hasTranscript ? `<button type="button" class="subtle" data-act="transcript">Combo</button>` : ''}
            ${video ? `<a class="subtle btn" href="${esc(video)}" target="_blank" rel="noopener">Video</a>` : ''}
            ${calc ? `<a class="subtle btn" href="${esc(calc)}" target="_blank" rel="noopener">Calc</a>` : ''}
        </div>
        <div class="transcript hidden"></div>
    </div>`;
}

function renderRotations(c) {
    const { main, featured } = rotationsFor(c.name);
    state.visibleTeams = [...(main ? main.teams : []), ...featured];
    const all = state.visibleTeams;
    teamRow.best = Math.max(1, ...all.map((t) => t.dps || 0));
    let html = '';
    if (main) {
        html += `<h3>${esc(c.name)} teams</h3>`;
        if (main.thoughts) html += `<blockquote class="thoughts">${esc(main.thoughts)}</blockquote>`;
        html += main.teams.map((t, i) => teamRow(t, i, false)).join('');
    }
    if (featured.length) {
        const offset = main ? main.teams.length : 0;
        html += `<h3>Supporting in other teams</h3>`;
        html += featured.map((t, i) => teamRow(t, offset + i, true)).join('');
    }
    if (!html) html = `<p class="muted">The rotation sheet has no teams with ${esc(c.name)} yet.</p>`;
    const src = safeUrl(state.rotationsSource);
    return `${html}<p class="src-line">Teams, DPS and rotation transcripts from ${src ? `<a href="${esc(src)}" target="_blank" rel="noopener">AntoCrasher's calc compilation</a>` : "AntoCrasher's calc compilation"} (2-minute fight). The "Combo" button shows the move-by-move rotation.</p>`;
}

function renderTranscript(t, tr) {
    const gloss = { ...DEFAULT_GLOSSARY, ...(tr.glossary || {}) };
    const conv = rotationToInputs(tr, t.members);
    const sections = tr.sections
        .map((sec) => {
            const steps = sec.steps
                .map((st) => {
                    const moves = st.moves
                        .map((m) => {
                            const base = m.toLowerCase().replace(/\d+$/, '');
                            const tip = gloss[m.toLowerCase()] || gloss[base] || '';
                            const swap = /^(swap|outro|intro)$/i.test(m) ? ' swap' : '';
                            return `<span class="move${swap}"${tip ? ` title="${esc(tip)}"` : ''}>${esc(m)}</span>`;
                        })
                        .join('<span class="arrow">›</span>');
                    return `<div class="step"><span class="who">${esc(st.character)}</span><span class="moves">${moves}</span></div>`;
                })
                .join('');
            return `${sec.name ? `<h4>${esc(sec.name)}</h4>` : ''}${steps}`;
        })
        .join('');
    const glossary = Object.entries(gloss)
        .map(([k, v]) => `<span><code>${esc(k)}</code> ${esc(v)}</span>`)
        .join('');
    const unmapped = conv.unmapped.length
        ? `<p class="muted small">Not converted (no single key): ${conv.unmapped.map((u) => `<code>${esc(u)}</code>`).join(' ')}</p>`
        : '';
    return `${sections || '<p class="muted">This transcript had no move lines.</p>'}
        ${glossary ? `<details class="glossary"><summary>Abbreviations</summary><div>${glossary}</div></details>` : ''}
        <div class="as-inputs">
            <label>As tracker inputs (slot order ${t.members.map((m, i) => `${i + 1} = ${esc(m)}`).join(', ')}; no timings):</label>
            <textarea rows="3" spellcheck="false" readonly>${esc(conv.inputs)}</textarea>
            ${unmapped}
            <div class="row">
                <button type="button" data-act="copy">Copy inputs</button>
                <button type="button" data-act="save" ${state.socket ? '' : 'disabled title="Start ComboTracker to save combos"'}>Save as combo</button>
                <a class="subtle btn" href="${esc(safeUrl(tr.source))}" target="_blank" rel="noopener">Open transcript</a>
            </div>
        </div>`;
}

async function toggleTranscript(teamEl) {
    const t = state.visibleTeams[Number(teamEl.dataset.idx)];
    const box = teamEl.querySelector('.transcript');
    if (!box.classList.contains('hidden')) {
        box.classList.add('hidden');
        return;
    }
    box.classList.remove('hidden');
    if (!state.transcripts.has(t.transcript)) {
        box.innerHTML = '<div class="loading">Loading rotation…</div>';
        try {
            state.transcripts.set(t.transcript, await api(`transcript?url=${encodeURIComponent(t.transcript)}`));
        } catch (e) {
            box.innerHTML = `<div class="error">Couldn't load the transcript: ${esc(e.message)}</div>`;
            return;
        }
    }
    box.innerHTML = renderTranscript(t, state.transcripts.get(t.transcript));
}

function saveAsCombo(teamEl) {
    const t = state.visibleTeams[Number(teamEl.dataset.idx)];
    const inputs = teamEl.querySelector('.as-inputs textarea').value;
    const suggested = `${t.team}${t.author ? ` (${t.author})` : ''}`;
    const name = prompt('Save this rotation as a combo named:', suggested);
    if (!name || !state.socket) return;
    state.socket.send(JSON.stringify({
        type: 'save_combo',
        name,
        inputs,
        demo_video: safeUrl(t.video),
        target_game: 'wuthering_waves',
        as_new: true,
    }));
    showNotice(`Saved "${name}". Open the Combo Tracker to practice it and add timings.`, 'ok');
}

// ---------------------------------------------------------------------------
// Events + boot
// ---------------------------------------------------------------------------

function bindEvents() {
    $('search').addEventListener('input', (e) => {
        state.query = e.target.value.trim();
        renderRoster();
    });
    $('elementFilter').addEventListener('click', (e) => {
        const b = e.target.closest('.chip');
        if (!b) return;
        state.element = b.dataset.el;
        renderFilters();
        renderRoster();
    });
    $('roster').addEventListener('click', (e) => {
        const b = e.target.closest('.char');
        if (b) selectCharacter(Number(b.dataset.id));
    });
    $('detail').addEventListener('click', (e) => {
        const tab = e.target.closest('[data-tab]');
        if (tab) {
            state.tab = tab.dataset.tab;
            renderDetail();
            return;
        }
        const act = e.target.closest('[data-act]');
        if (!act) return;
        const teamEl = act.closest('.team');
        if (act.dataset.act === 'transcript') toggleTranscript(teamEl);
        else if (act.dataset.act === 'copy') {
            const ta = teamEl.querySelector('.as-inputs textarea');
            navigator.clipboard?.writeText(ta.value).then(
                () => showNotice('Inputs copied. Paste them into a combo in the tracker.', 'ok'),
                () => { ta.select(); document.execCommand('copy'); }
            );
        } else if (act.dataset.act === 'save') saveAsCombo(teamEl);
    });
    $('detail').addEventListener('change', (e) => {
        if (e.target.id === 'levelSelect') {
            state.level = Number(e.target.value);
            renderDetail();
        }
    });
    $('refreshBtn').addEventListener('click', () => load(true));
    window.addEventListener('hashchange', () => {
        const id = Number(location.hash.slice(1));
        if (id && id !== state.selectedId) selectCharacter(id, { pushHash: false });
    });
}

async function load(refresh = false) {
    showNotice(refresh ? 'Downloading fresh data…' : '');
    if (refresh) {
        state.kits.clear();
        state.transcripts.clear();
    }
    const [roster, rotations] = await Promise.allSettled([api('roster', refresh), api('rotations', refresh)]);
    if (roster.status === 'rejected') {
        showNotice(`Couldn't load the character list: ${roster.reason.message}`, 'warn');
        return;
    }
    state.roster = roster.value.characters;
    if (rotations.status === 'fulfilled') {
        state.rotations = rotations.value.groups;
        state.rotationsSource = rotations.value.source;
    } else {
        showNotice(`Couldn't load team rotations: ${rotations.reason.message}`, 'warn');
    }
    if (refresh && roster.status === 'fulfilled' && rotations.status === 'fulfilled' && !roster.value.stale) {
        showNotice('Data refreshed.', 'ok');
    }
    renderRoster();
    const id = Number(location.hash.slice(1)) || state.selectedId;
    if (id) selectCharacter(id, { pushHash: false });
}

renderFilters();
bindEvents();
connectTracker();
load();
