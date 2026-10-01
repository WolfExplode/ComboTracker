// Characters page: each character's key-input moves (static/data/ww_characters.json, drafted from
// encore.moe and checked by hand) and community team rotations (AntoCrasher's compilation +
// rotation transcripts, served by ui_server.py under /api/ww/).

// Shown inside the main app's Characters page: hide this page's own back link and title.
if (new URLSearchParams(location.search).has('embed')) document.documentElement.classList.add('embed');

const ELEMENTS = ['Aero', 'Electro', 'Fusion', 'Glacio', 'Havoc', 'Spectro'];
const MOVES_URL = 'data/ww_characters.json';

// Abbreviations from AntoCrasher's rotation hub; each rotation's own list (if any) wins.
const DEFAULT_GLOSSARY = {
    ba: 'basic attack', fba: 'forte basic attack', uba: 'ultimate basic attack', ha: 'heavy attack',
    fha: 'forte heavy attack', skill: 'resonance skill', eskill: 'enhanced skill', fskill: 'forte skill',
    lib: 'liberation', echo: 'echo skill', tbs: 'tune break skill', nf: 'nightfall (Zani)',
    dash: 'dodge', swap: 'swap to the next character', outro: 'outro skill (swap out)',
};

const state = {
    roster: [],
    rotations: [],          // [{character, teams: [...]}]
    rotationsSource: '',
    mine: [],               // nameTokens() of characters saved in the tracker
    element: '',
    query: '',
    selectedId: null,
    tab: 'moves',
    moves: new Map(),       // character id -> entry from ww_characters.json
    transcripts: new Map(),
    tracker: null,
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

function watchTracker() {
    state.tracker = connectTracker({
        onMessage: (msg) => {
            // The character list rides along in the editor payload (init and later editor updates).
            const chars = (msg.editor && msg.editor.ww_characters) || msg.ww_characters;
            if (Array.isArray(chars)) {
                state.mine = chars.map((c) => nameTokens(c.name || c.name_key));
                renderRoster();
            }
            if (msg.type === 'status' && msg.text && msg.color === 'fail') {
                showNotice(msg.text, 'warn');
            }
        },
    });
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
    return { main, featured };
}

function selectCharacter(id, { pushHash = true } = {}) {
    state.selectedId = id;
    if (pushHash) history.replaceState(null, '', `#${id}`);
    renderRoster();
    if (!state.roster.find((x) => x.id === id)) return;
    renderDetail();
    if (window.innerWidth < 900) $('detail').scrollIntoView({ block: 'start', behavior: 'smooth' });
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
    const body = state.tab === 'moves' ? renderMoves(c) : renderRotations(c);
    $('detail').innerHTML = detailHeader(c) + body;
}

// --- Moves (key inputs only; from static/data/ww_characters.json) --------------

const CHAIN_AFTER = {
    intro: 'the Intro (swap in)', skill: 'the Resonance Skill (E)', liberation: 'the Liberation (R)',
    tune_break: 'a Tune Break (F)', heavy: 'a Heavy Attack', dodge: 'a Dodge Counter', midair: 'a Mid-air Attack',
};

function keycap(input) {
    if (!input) return '';
    return input.split(/,\s*/).map((k) => `<span class="keycap">${esc(k.toUpperCase())}</span>`).join('<span class="plus">then</span>');
}

function renderMoves(c) {
    const m = state.moves.get(c.id);
    if (!m) return '<p class="muted">No move data for this character yet.</p>';
    const b = m.basic || {};
    const hits = (b.hit_names || []).map((name, i) =>
        `<li><span class="keycap hit">A${i + 1}</span><span>${esc(name)}</span></li>`).join('');
    const hold = b.hold_lmb === 'chain'
        ? 'keeps the Basic chain going'
        : `${esc(b.hold_label || 'Heavy')} (Heavy Attack)`;
    const entries = Object.entries(m.chain_entry || {})
        .filter(([k]) => CHAIN_AFTER[k])
        .map(([k, stage]) => `<li>LMB right after ${esc(CHAIN_AFTER[k])} starts at <strong>A${stage}</strong></li>`).join('');

    const withInput = (m.moves || []).filter((x) => x.input);
    const other = (m.moves || []).filter((x) => !x.input);
    const moveRows = withInput.map((x) =>
        `<tr><td class="inp">${keycap(x.input)}</td><td>${esc(x.name)}${x.type ? ` <span class="muted small">${esc(x.type)}</span>` : ''}</td></tr>`).join('');
    const otherByType = {};
    other.forEach((x) => { (otherByType[x.type || 'Other'] ||= []).push(x.name); });
    const otherHtml = Object.entries(otherByType).map(([t, names]) =>
        `<p><span class="muted">${esc(t)}:</span> ${names.map(esc).join(', ')}</p>`).join('');
    const follow = (m.followups || []).filter((f) => !f.basic_stage).map((f) =>
        `<li>${keycap(f.press)} right after <strong>${esc(f.after)}</strong> casts <strong>${esc(f.gives)}</strong></li>`).join('');

    return `<div class="toolbar">
            <span class="${m.reviewed ? 'badge ok' : 'badge'}">${m.reviewed ? 'Checked by hand' : 'Auto-drafted, not checked yet'}</span>
            <span class="muted small">From encore.moe, trimmed to key inputs</span>
        </div>
        <section class="move-card">
            <h3>${esc(b.name || 'Basic Attack')} <span class="muted small">${b.hits ? `${b.hits}-hit chain` : 'chain length unknown'}</span></h3>
            ${hits ? `<ol class="hits">${hits}</ol>` : ''}
            <p><span class="keycap">HOLD LMB</span> ${hold}</p>
            ${entries ? `<ul class="entries">${entries}</ul>` : ''}
        </section>
        ${m.notes ? `<section class="move-card notes"><h3>Notes</h3><p>${esc(m.notes)}</p></section>` : ''}
        <section class="move-card">
            <h3>Moves</h3>
            <table class="moves"><tbody>${moveRows}</tbody></table>
            ${otherHtml ? `<div class="other-moves"><h4>Other named moves</h4>${otherHtml}</div>` : ''}
        </section>
        ${follow ? `<section class="move-card"><h3>Follow-ups</h3><ul class="follow">${follow}</ul></section>` : ''}`;
}

// --- Rotations ----------------------------------------------------------------

function teamRow(t, idx, showMain) {
    const video = safeUrl(t.video);
    const hasTranscript = !!safeUrl(t.transcript);
    return `<div class="team" data-idx="${idx}">
        <div class="team-main">
            <div class="team-name">${esc(showMain ? t.team : t.team.replace(/\s*\([^)]*\)\s*$/, ''))}</div>
            <div class="team-meta">
                ${t.style ? `<span class="tag">${esc(t.style)}</span>` : ''}
                ${t.setup ? `<span class="tag subtle">${esc(t.setup)}</span>` : ''}
                ${t.author ? `<span class="muted">by ${esc(t.author)}</span>` : ''}
            </div>
        </div>
        <div class="team-actions">
            ${hasTranscript ? `<button type="button" class="subtle" data-act="transcript">Combo</button>` : ''}
            ${video ? `<a class="subtle btn" href="${esc(video)}" target="_blank" rel="noopener">Video</a>` : ''}
        </div>
        <div class="transcript hidden"></div>
    </div>`;
}

function renderRotations(c) {
    const { main, featured } = rotationsFor(c.name);
    state.visibleTeams = [...(main ? main.teams : []), ...featured];
    const all = state.visibleTeams;
    let html = '';
    if (main) {
        html += `<h3>${esc(c.name)} teams</h3>`;
        html += main.teams.map((t, i) => teamRow(t, i, false)).join('');
    }
    if (featured.length) {
        const offset = main ? main.teams.length : 0;
        html += `<h3>Supporting in other teams</h3>`;
        html += featured.map((t, i) => teamRow(t, offset + i, true)).join('');
    }
    if (!html) html = `<p class="muted">The rotation sheet has no teams with ${esc(c.name)} yet.</p>`;
    const src = safeUrl(state.rotationsSource);
    return `${html}<p class="src-line">Teams and rotation transcripts from ${src ? `<a href="${esc(src)}" target="_blank" rel="noopener">AntoCrasher's rotation compilation</a>` : "AntoCrasher's rotation compilation"}. The "Combo" button shows the move-by-move rotation.</p>`;
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
                <button type="button" data-act="save" ${state.tracker && state.tracker.isOpen() ? '' : 'disabled title="Start ComboTracker to save combos"'}>Save as combo</button>
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
    if (!name || !state.tracker) return;
    const sent = state.tracker.send('save_combo', {
        name,
        inputs,
        demo_video: safeUrl(t.video),
        target_game: 'wuthering_waves',
        as_new: true,
    });
    if (!sent) {
        showNotice("Couldn't reach the tracker, so nothing was saved. Is ComboTracker running?", 'warn');
        return;
    }
    showNotice(`Saved "${name}". Open the Combos page to practice it and add timings.`, 'ok');
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
    $('rawBtn').addEventListener('click', downloadRaw);
    window.addEventListener('hashchange', () => {
        const id = Number(location.hash.slice(1));
        if (id && id !== state.selectedId) selectCharacter(id, { pushHash: false });
    });
}

async function load() {
    showNotice('');
    let moves;
    try {
        const res = await fetch(MOVES_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        moves = await res.json();
    } catch (e) {
        showNotice(`Couldn't load the character list: ${e.message}`, 'warn');
        return;
    }
    const chars = Object.values(moves.characters || {});
    chars.sort((a, b) => a.name.localeCompare(b.name));
    state.roster = chars;
    state.moves = new Map(chars.map((c) => [c.id, c]));
    renderRoster();
    const id = Number(location.hash.slice(1)) || state.selectedId;
    if (id) selectCharacter(id, { pushHash: false });

    try {
        const rot = await api('rotations');
        state.rotations = rot.groups;
        state.rotationsSource = rot.source;
        if (state.selectedId) renderDetail();
    } catch (e) {
        showNotice(`Couldn't load team rotations: ${e.message}`, 'warn');
    }
}

// Saves encore.moe's unedited data to data/encore_raw/ next to the app, for re-drafting the move file
// with tools/ww_build_moves.py. It doesn't change what this page shows.
async function downloadRaw() {
    if (!confirm('Download the raw encore.moe data for every character? It is saved as-is in data/encore_raw/ and does not change the moves shown here.')) return;
    const btn = $('rawBtn');
    btn.disabled = true;
    showNotice('Downloading raw data for every character…');
    try {
        const res = await fetch(`/api/ww/scrape-raw`, { method: 'POST' });
        const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        const failed = data.failed && data.failed.length ? ` ${data.failed.length} failed.` : '';
        showNotice(`Saved raw data for ${data.saved} characters in ${data.folder}.${failed} Run tools/ww_build_moves.py to draft new characters from it.`, failed ? 'warn' : 'ok');
    } catch (e) {
        showNotice(`Couldn't download the raw data: ${e.message}`, 'warn');
    } finally {
        btn.disabled = false;
    }
}

renderFilters();
bindEvents();
watchTracker();
load();
