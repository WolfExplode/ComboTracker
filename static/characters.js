// Characters page: each character's key-input moves (static/data/ww_characters.json, drafted from
// encore.moe and checked by hand), ability frame timings (static/data/ww_timings.json, from
// WuwaLAB) and community team rotations (AntoCrasher's compilation + rotation transcripts,
// served by ui_server.py under /api/ww/).

// Shown inside the main app's Characters page: hide this page's own back link and title.
if (new URLSearchParams(location.search).has('embed')) document.documentElement.classList.add('embed');

const ELEMENTS = ['Aero', 'Electro', 'Fusion', 'Glacio', 'Havoc', 'Spectro'];
const MOVES_URL = 'data/ww_characters.json';
const TIMINGS_URL = 'data/ww_timings.json';

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
    timings: null,          // ww_timings.json: {fps, fetched_at, characters: {id: {url, abilities}}}
    timingFilter: '',
    timingUnit: readPref('ww-timing-unit', 'f'),   // 'f' frames or 's' seconds
    openTimings: new Set(),  // section|name of Timings rows whose frame strip is open
    transcripts: new Map(),
    tracker: null,
    notes: null,            // your notes per character id, saved in the tracker (null = not loaded yet)
};

const $ = (id) => document.getElementById(id);

function readPref(key, fallback) {
    try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}

function writePref(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage blocked: keep it for this visit only */ }
}

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
            const notes = msg.type === 'ww_character_notes' ? msg.notes
                : ((msg.editor && msg.editor.ww_character_notes) || msg.ww_character_notes);
            if (notes && typeof notes === 'object') {
                state.notes = { ...notes };
                // Don't swap the text out from under someone typing in it.
                if (state.tab === 'moves' && document.activeElement?.id !== 'charNotes') renderDetail();
            }
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
    const t = timingsFor(c);
    return `<div class="detail-head">
        ${icon ? `<img src="${esc(icon)}" alt="">` : ''}
        <div>
            <h2>${esc(c.name)}</h2>
            <div class="muted"><span class="dot el-${esc(String(c.element).toLowerCase())}"></span>${esc(c.element)} · ${esc(c.weapon)} · ${'★'.repeat(c.rarity || 0)}</div>
        </div>
    </div>
    <div class="tabs" role="tablist">
        <button type="button" role="tab" data-tab="moves" class="${state.tab === 'moves' ? 'on' : ''}">Moves</button>
        <button type="button" role="tab" data-tab="timings" class="${state.tab === 'timings' ? 'on' : ''}">Timings${t ? ` <span class="count">${t.abilities.length}</span>` : ''}</button>
        <button type="button" role="tab" data-tab="rotations" class="${state.tab === 'rotations' ? 'on' : ''}">Team rotations <span class="count">${count}</span></button>
    </div>`;
}

function renderDetail() {
    const c = state.roster.find((x) => x.id === state.selectedId);
    if (!c) return;
    const body = state.tab === 'moves' ? renderMoves(c)
        : state.tab === 'timings' ? renderTimings(c)
        : renderRotations(c);
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
            ${(m.skill_chain || []).length > 1 ? `<p><span class="keycap">E</span><span class="plus">in a row</span> ${m.skill_chain.map(esc).join(' → ')}</p>` : ''}
        </section>
        ${renderNotes(c, m)}
        ${follow ? `<section class="move-card"><h3>Follow-ups</h3><ul class="follow">${follow}</ul></section>` : ''}`;
}

// Notes: your own, saved in the tracker when you stop typing or leave the box. Until you write
// some, the box starts with the shipped notes from ww_characters.json.
function renderNotes(c, m) {
    const mine = state.notes && Object.prototype.hasOwnProperty.call(state.notes, String(c.id));
    const text = mine ? state.notes[String(c.id)] : (m.notes || '');
    const live = state.tracker && state.tracker.isOpen();
    return `<section class="move-card notes">
        <h3>Notes <span class="muted small" id="notesStatus">${live ? '' : 'Start ComboTracker to save notes'}</span></h3>
        <textarea id="charNotes" rows="4" placeholder="Your notes on ${esc(c.name)}: rotations, cancels, things to remember…">${esc(text)}</textarea>
    </section>`;
}

let notesTimer = null;
function saveNotes() {
    clearTimeout(notesTimer);
    const ta = $('charNotes');
    if (!ta || state.selectedId == null) return;
    const id = String(state.selectedId);
    const text = ta.value.trim();
    const status = $('notesStatus');
    const current = state.notes && Object.prototype.hasOwnProperty.call(state.notes, id) ? state.notes[id] : null;
    if (current === text) return;
    const sent = state.tracker && state.tracker.isOpen() && state.tracker.send('save_character_note', { id, text });
    if (sent === false || !state.tracker || !state.tracker.isOpen()) {
        if (status) status.textContent = 'Not saved: ComboTracker isn\'t running';
        return;
    }
    state.notes = { ...(state.notes || {}), [id]: text };
    if (status) status.textContent = 'Saved';
}

// --- Timings (frame data from WuwaLAB; static/data/ww_timings.json) ------------

// Columns in WuwaLAB's order, timing columns only (no damage).
const TIMING_COLS = [
    { key: 'hits', label: 'Hits', tip: 'Number of hits', frames: false },
    { key: 'frames', label: 'Frames', tip: 'Full animation length', frames: true },
    { key: 'cancel', label: 'Cancel', tip: 'Earliest frame the next action can cancel this one', frames: true },
    { key: 'noswap', label: 'No swap', tip: "Frames before you can swap out", frames: true },
    { key: 'tstop', label: 'T.stop', tip: 'Time stop: the whole field freezes', frames: true },
    { key: 'mstop', label: 'M.stop', tip: 'Motion stop: hit-stop on the character', frames: true },
    { key: 'concerto', label: 'Concerto', tip: 'Concerto Energy gained (full = 100). A full bar lets the Outro fire on swap and unlocks moves like Iuno\'s Absolute Fullness', concerto: true },
    { key: 'cd', label: 'CD', tip: 'Cooldown', frames: true },
];

// WuwaLAB counts Concerto in hundredths (1,000 = 10 points, -10,000 = the Outro emptying the bar).
function fmtConcerto(v) {
    if (v == null) return '<span class="zero">—</span>';
    const pts = v / 100;
    const text = Number.isInteger(pts) ? String(pts) : pts.toFixed(2).replace(/0$/, '');
    if (v === 0) return `<span class="zero">${text}</span>`;
    return `<span class="${v < 0 ? 'neg' : 'conc'}">${text}</span>`;
}

function timingsFor(c) {
    return state.timings && state.timings.characters ? state.timings.characters[String(c.id)] : null;
}

// 73 -> "73f" or "1.22s", per the unit toggle. Zero and missing values are dimmed.
function fmtFrames(n) {
    if (n == null) return '<span class="zero">—</span>';
    const fps = (state.timings && state.timings.fps) || 60;
    const text = state.timingUnit === 's' ? `${(n / fps).toFixed(2)}s` : `${n}f`;
    return n === 0 ? `<span class="zero">${text}</span>` : text;
}

// The keys that cast a WuwaLAB move, in combo-input notation: lmb1, lmb2, hold(lmb), rmb, lmb,
// e, r, f... Worked out from the move's type and name; '' for passives and follow-on effects.
function abilityInput(a) {
    const name = String(a.name || '');
    const midAir = /mid-air/i.test(name);
    const stage = (name.match(/\s(\d+)(\s*\([^)]*\))*$/) || [])[1];
    const held = /\bhold\b|\bheld\b|charged/i.test(name);
    switch (a.genre) {
    case 'BASIC': return midAir ? 'space, lmb' : `lmb${stage || ''}`;
    case 'COUNTER': return midAir ? 'space, rmb, lmb' : 'rmb, lmb';
    case 'HEAVY': return midAir ? 'space, hold(lmb)' : 'hold(lmb)';
    case 'SKILL': return held ? 'hold(e)' : 'e';
    case 'LIBERATION': return 'r';
    case 'INTRO': return 'swap in';
    case 'OUTRO': return 'swap out';
    case 'TUNEBREAK': return 'f';
    case 'DODGE': case 'DASH': return 'rmb';
    default: return '';
    }
}

function timingCell(a, col) {
    const v = a[col.key];
    if (col.concerto) return fmtConcerto(v);
    if (!col.frames) return v ? String(v) : `<span class="zero">${v ?? '—'}</span>`;
    return fmtFrames(v);
}

const timingKey = (a) => `${a.section}|${a.name}`;

// WuwaLAB's frame strip: the animation as a bar, a tick per hit, and the part after the cancel
// frame shaded (from there the next action can cut it short).
function timingStrip(a) {
    const total = a.frames || 0;
    if (!total) return '<p class="muted small">No animation frames listed for this ability.</p>';
    const pct = (f) => `${Math.min(100, Math.max(0, (f / total) * 100)).toFixed(3)}%`;
    const hits = (a.hit_frames || []).filter((f) => f <= total);
    const cancel = a.cancel > 0 && a.cancel < total ? a.cancel : null;
    // Axis labels, most useful first; a label too close to one already placed is left off
    // (its tick keeps a tooltip).
    const want = [[0, '', fmtFrames(0)], [total, 'end', fmtFrames(total)]];
    if (cancel != null) want.push([cancel, 'cancel', `C ${fmtFrames(cancel)}`]);
    hits.forEach((f) => want.push([f, '', fmtFrames(f)]));
    const placed = [];
    for (const [f, cls, text] of want) {
        if (placed.some((p) => Math.abs(p[0] - f) / total < 0.06)) continue;
        placed.push([f, cls, text]);
    }
    const marks = placed.map(([f, cls, text]) =>
        `<span class="tick-lbl${f === 0 ? ' start' : ''}${cls ? ` ${cls}` : ''}" style="left:${pct(f)}">${text}</span>`);
    const zones = (a.zones || []).filter((z) => z.to > z.from).map((z) =>
        `<div class="zone ${z.kind}" style="left:${pct(z.from)};width:calc(${pct(z.to)} - ${pct(z.from)})" title="${z.kind === 'ts' ? 'Time stop' : 'Motion stop'} ${z.from}-${z.to}f">${z.kind.toUpperCase()}</div>`).join('');
    return `<div class="fstrip" role="img" aria-label="${esc(`${a.name}: ${total} frames, hits at ${hits.join(', ') || 'none'}${cancel != null ? `, cancel at ${cancel}` : ''}`)}">
            ${cancel != null ? `<div class="after-cancel" style="left:${pct(cancel)}" title="After the cancel frame"></div><div class="cancel-line" style="left:${pct(cancel)}"></div>` : ''}
            ${zones}
            ${hits.map((f, i) => `<div class="hit" style="left:${pct(f)}" title="Hit ${i + 1}: ${f}f"></div>`).join('')}
        </div>
        <div class="fstrip-axis">${marks.join('')}</div>`;
}

function timingRow(a) {
    const tags = (a.tags || []).map((t) => `<span class="ttag">${esc(t)}</span>`).join('');
    const cells = TIMING_COLS.map((col) =>
        `<td class="num${col.key === 'frames' ? ' strong' : ''}">${timingCell(a, col)}</td>`).join('');
    const hitFrames = (a.hit_frames || []).map((f) => fmtFrames(f)).join(' <span class="sep">·</span> ');
    const key = timingKey(a);
    const open = state.openTimings.has(key);
    return `<tr class="trow${open ? ' open' : ''}" data-tkey="${esc(key)}" aria-expanded="${open}" title="Show the frame strip">
        <td class="tname"><div><span class="caret">▸</span>${esc(a.name)}</div>${tags ? `<div class="ttags">${tags}</div>` : ''}</td>
        <td class="tinput">${abilityInput(a) ? `<code>${esc(abilityInput(a))}</code>` : '<span class="zero">—</span>'}</td>
        ${cells}
        <td class="hitf">${hitFrames || '<span class="zero">—</span>'}</td>
    </tr>${open ? `<tr class="tstrip"><td colspan="${TIMING_COLS.length + 3}">${timingStrip(a)}</td></tr>` : ''}`;
}

function timingRows(t) {
    const q = state.timingFilter.toLowerCase();
    const list = t.abilities.filter((a) => !q || a.name.toLowerCase().includes(q) || a.section.toLowerCase().includes(q));
    const span = TIMING_COLS.length + 3;
    let html = '';
    let section = null;
    for (const a of list) {
        if (a.section !== section) {
            section = a.section;
            html += `<tr class="tsection"><td colspan="${span}">${esc(section)}</td></tr>`;
        }
        html += timingRow(a);
    }
    return html || `<tr><td colspan="${span}" class="muted">No abilities match.</td></tr>`;
}

function renderTimings(c) {
    const t = timingsFor(c);
    if (!state.timings) return '<p class="muted">Loading timings…</p>';
    if (!t) return `<p class="muted">WuwaLAB has no frame data for ${esc(c.name)} yet.</p>`;
    const fps = state.timings.fps || 60;
    const head = TIMING_COLS.map((col) =>
        `<th class="num${col.key === 'frames' ? ' strong' : ''}" title="${esc(col.tip)}">${esc(col.label)}</th>`).join('');
    const unit = (u, label) =>
        `<button type="button" data-unit="${u}" class="${state.timingUnit === u ? 'on' : ''}">${label}</button>`;
    const url = safeUrl(t.url);
    return `<div class="toolbar">
            <input type="search" id="timingFilter" placeholder="Filter abilities…" value="${esc(state.timingFilter)}" autocomplete="off">
            <div class="seg" role="group" aria-label="Units">${unit('f', 'Frames')}${unit('s', 'Seconds')}</div>
            <span class="muted small">${fps} fps · 1f = ${(1000 / fps).toFixed(2)} ms</span>
        </div>
        <div class="timings-wrap">
            <table class="timings">
                <thead><tr><th>Name</th><th title="Keys that cast it, written like combo inputs (lmb2 = the 2nd LMB of the Basic chain)">Input</th>${head}<th title="Frame each hit lands on">Hit frames</th></tr></thead>
                <tbody id="timingRows">${timingRows(t)}</tbody>
            </table>
        </div>
        <p class="src-line">Frame data from ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">WuwaLAB</a>` : 'WuwaLAB'}${state.timings.fetched_at ? `, copied ${esc(state.timings.fetched_at.split(' ')[0])}` : ''}. Timing columns and Concerto only (Concerto is the total a move gives, in-game points). Hover a column name for what it means.</p>`;
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

// Drag the bar between the roster and the detail panel to resize the roster (remembered per
// browser); double-click resets it.
const ROSTER_WIDTH_KEY = 'ww-roster-width';
const ROSTER_MIN = 200, ROSTER_MAX = 900;

function setRosterWidth(px) {
    const w = Math.round(Math.min(ROSTER_MAX, Math.max(ROSTER_MIN, px)));
    document.documentElement.style.setProperty('--roster-w', `${w}px`);
    return w;
}

function bindResizer() {
    const saved = Number(readPref(ROSTER_WIDTH_KEY, 0));
    if (saved) setRosterWidth(saved);
    let bar = $('rosterResizer');
    if (!bar) {
        // A stale cached characters.html may predate the bar; add it so dragging still works.
        bar = document.createElement('div');
        bar.id = 'rosterResizer';
        bar.className = 'splitter';
        bar.setAttribute('role', 'separator');
        bar.setAttribute('aria-orientation', 'vertical');
        bar.setAttribute('aria-label', 'Resize character list');
        bar.title = 'Drag to resize. Double-click to reset.';
        $('roster').after(bar);
    }
    bar.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        bar.setPointerCapture(e.pointerId);
        document.body.classList.add('resizing');
        const startX = e.clientX;
        const startW = $('roster').getBoundingClientRect().width;
        let width = startW;
        const move = (ev) => { width = setRosterWidth(startW + ev.clientX - startX); };
        const up = () => {
            bar.removeEventListener('pointermove', move);
            document.body.classList.remove('resizing');
            writePref(ROSTER_WIDTH_KEY, String(width));
        };
        bar.addEventListener('pointermove', move);
        bar.addEventListener('pointerup', up, { once: true });
        bar.addEventListener('pointercancel', up, { once: true });
    });
    bar.addEventListener('dblclick', () => {
        document.documentElement.style.removeProperty('--roster-w');
        try { localStorage.removeItem(ROSTER_WIDTH_KEY); } catch { /* ignore */ }
    });
}

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
        const unit = e.target.closest('[data-unit]');
        if (unit) {
            state.timingUnit = unit.dataset.unit;
            writePref('ww-timing-unit', state.timingUnit);
            renderDetail();
            return;
        }
        const trow = e.target.closest('tr.trow');
        if (trow) {
            const key = trow.dataset.tkey;
            if (state.openTimings.has(key)) state.openTimings.delete(key);
            else state.openTimings.add(key);
            const c = state.roster.find((x) => x.id === state.selectedId);
            const t = c && timingsFor(c);
            if (t) $('timingRows').innerHTML = timingRows(t);
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
    // Filtering re-renders only the table body so the filter box keeps focus.
    $('detail').addEventListener('input', (e) => {
        if (e.target.id === 'charNotes') {
            const status = $('notesStatus');
            if (status) status.textContent = 'Editing…';
            clearTimeout(notesTimer);
            notesTimer = setTimeout(saveNotes, 1000);
            return;
        }
        if (e.target.id !== 'timingFilter') return;
        state.timingFilter = e.target.value.trim();
        const c = state.roster.find((x) => x.id === state.selectedId);
        const t = c && timingsFor(c);
        if (t) $('timingRows').innerHTML = timingRows(t);
    });
    $('detail').addEventListener('focusout', (e) => { if (e.target.id === 'charNotes') saveNotes(); });
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
        const res = await fetch(TIMINGS_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.timings = await res.json();
    } catch (e) {
        state.timings = { characters: {} };
        showNotice(`Couldn't load ability timings: ${e.message}`, 'warn');
    }
    if (state.selectedId) renderDetail();

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
bindResizer();
watchTracker();
load();
