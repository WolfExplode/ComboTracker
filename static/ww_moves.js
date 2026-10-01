// Wuthering Waves move names for timeline steps.
//
// Inputs are decoded by rule, walking the timeline in order:
//   leading f = start the fight (ToA prompt), any later f = Tune Break
//   1/2/3 = swap in that slot (Intro)
//   lmb = Basic 1, 2, 3... (reset by any other action or a wait of 1s+; wraps at the chain length)
//   lmb right after rmb = Dodge Counter?, right after space = Mid-air Attack
//   hold(lmb) = Heavy, or another Basic hit for characters whose hold continues the chain
//   e / hold(e) = Skill / Held Skill, q = Echo, r = Liberation, rmb = Dodge, space = Jump, shift = Sprint
//   e pressed 2+ times in a row (only waits between) = that character's skill_chain, if they have one
//   (Augusta: Strike, Leap, Plunge). The first E is renamed once the second arrives; see onRevise.
//
// Moves with their own name (Skill, Liberation, Intro, a named Heavy) show just that name, e.g.
// "Warrior's Blade" rather than "Augusta Skill": the tile's icon and key already say what kind of
// move it is. Unnamed moves keep the character ("Augusta Basic 2"), except Echo, which is just "Echo".
//
// Per-character rules (chain length, what hold(lmb) does, where LMB picks the chain back up after
// another move) come from static/data/ww_characters.json via setWwCharacterData().
//
// Concerto Energy is tracked per character as the timeline goes, from WuwaLAB's per-ability
// Concerto (static/data/ww_timings.json via setWwTimingData()). Each move adds its base version's
// Concerto (an estimate: forms like Iuno's Moonbow or Enhanced moves give more), everyone starts
// at 0, and swapping out with a full bar fires the Outro and empties it. A character whose
// basic.hold_full_concerto is set (Iuno: "Absolute Fullness") gets that Heavy instead of the usual
// one while their bar is full.

const WW_SLOTS = ['1', '2', '3'];

// basic = hits in the Basic chain before it restarts (0 = unknown, never wraps); hold = what hold(lmb)
// is called (null when holding just continues the Basic chain); entry = { intro|skill|liberation|
// tune_break|heavy|dodge|midair: stage } for "LMB right after that move is Basic Stage N";
// skillChain = names for E pressed again and again in a row ([] = every E is just "Skill");
// names = { skill, liberation, intro }: the moves' own names ('' = unknown, say "<character> Skill").
// holdFull = what hold(lmb) is called while Concerto is full (null = same as hold); key = name tokens.
const WW_DEFAULT_RULE = { key: '', basic: 0, hold: 'Heavy', holdFull: null, entry: {}, skillChain: [], names: {} };

// Concerto in WuwaLAB's units: 100 = 1 point, a full bar is 100 points.
const WW_CONCERTO_FULL = 10000;
let wwConcertoTables = {}; // name tokens -> [{ name, genre, concerto }] from ww_timings.json

// Own name of the move cast by `input` (ww_characters.json moves list), or ''.
function wwMoveName(c, input, type) {
    const m = (c.moves || []).find((x) => x.input === input && (!type || x.type === type));
    return m ? m.name : '';
}
let wwCharacterRules = {}; // normalized name -> rule

// The in-game Basic chain resets after a pause this long.
const WW_BASIC_RESET_MS = 1000;

/** "Rover: Spectro" / "Yangyang Xuanling" -> "rover spectro" / "xuanling yangyang" (word order ignored). */
function wwNameTokens(name) {
    return String(name || '').toLowerCase().split(/[^a-z]+/).filter(Boolean).sort().join(' ');
}

/** Load ww_characters.json ({characters: {key: entry}}) as labeling rules. */
function setWwCharacterData(doc) {
    const rules = {};
    Object.values((doc && doc.characters) || {}).forEach((c) => {
        const b = c.basic || {};
        rules[wwNameTokens(c.name)] = {
            key: wwNameTokens(c.name),
            basic: Number(b.hits) || 0,
            hold: b.hold_lmb === 'chain' ? null : (b.hold_label || 'Heavy'),
            holdFull: b.hold_full_concerto || null,
            entry: c.chain_entry || {},
            skillChain: Array.isArray(c.skill_chain) ? c.skill_chain : [],
            names: {
                skill: wwMoveName(c, 'e', 'Resonance Skill'),
                liberation: wwMoveName(c, 'r', 'Resonance Liberation'),
                intro: wwMoveName(c, 'swap in', 'Intro Skill'),
            },
        };
    });
    wwCharacterRules = rules;
}

/** Load ww_timings.json ({characters: {id: {name, abilities}}}) for Concerto tracking. */
function setWwTimingData(doc) {
    const tables = {};
    Object.values((doc && doc.characters) || {}).forEach((c) => {
        tables[wwNameTokens(c.name)] = (c.abilities || [])
            .filter((a) => typeof a.concerto === 'number')
            .map((a) => ({ name: String(a.name || ''), genre: String(a.genre || ''), concerto: a.concerto }));
    });
    wwConcertoTables = tables;
}

/**
 * Concerto a move gives, in WuwaLAB units (0 when unknown). move = { kind, stage, name }:
 * kind is intro / basic / midair / counter / heavy / skill / liberation / tune_break.
 * Picks the first (base-form) WuwaLAB row of that type, or the one with the move's own name.
 */
function wwConcertoGain(rule, move) {
    const rows = wwConcertoTables[rule && rule.key];
    if (!rows || !rows.length) return 0;
    const of = (genre) => rows.filter((r) => r.genre === genre);
    const named = (list) => {
        const n = String(move.name || '').toLowerCase();
        return (n && list.find((r) => r.name.toLowerCase().includes(n))) || list[0];
    };
    let row = null;
    if (move.kind === 'intro') row = of('INTRO')[0];
    else if (move.kind === 'basic') {
        const re = new RegExp(`^basic[^(]*\\s${Number(move.stage) || 1}$`, 'i');
        row = of('BASIC').find((r) => re.test(r.name));
    } else if (move.kind === 'midair') row = of('BASIC').find((r) => /mid-air/i.test(r.name));
    else if (move.kind === 'counter') row = of('COUNTER')[0];
    else if (move.kind === 'heavy') row = named(of('HEAVY'));
    else if (move.kind === 'skill') row = named(of('SKILL'));
    else if (move.kind === 'liberation') row = of('LIBERATION')[0];
    else if (move.kind === 'tune_break') row = of('TUNEBREAK')[0];
    return row ? Math.max(0, row.concerto) : 0;
}

// Team names are typed by hand ("Pheobe", "Agusta", "Rover"), so allow one typo, a missing word,
// or swapped letters before falling back to the default rule.
function wwRuleFor(name) {
    const t = wwNameTokens(name);
    if (!t) return WW_DEFAULT_RULE;
    if (wwCharacterRules[t]) return wwCharacterRules[t];
    const keys = Object.keys(wwCharacterRules);
    const squash = (s) => s.replace(/ /g, '');
    const close = (a, b) => {
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
    };
    const hit = keys.find((k) => close(squash(k), squash(t)))
        || keys.find((k) => k.split(' ').some((w) => close(w, squash(t))));
    return hit ? wwCharacterRules[hit] : WW_DEFAULT_RULE;
}

/** Key a timeline step acts on, or '' for a plain wait. */
function wwStepKey(step) {
    if (!step) return '';
    if (step.type === 'wait') {
        return step.mode === 'mandatory' ? (step.wait_for || '').toString().toLowerCase() : '';
    }
    return (step.input || '').toString().toLowerCase();
}

/**
 * Returns label(step, slot) that names each step's move. Call it once per step, in timeline order.
 * slotNames maps '1'/'2'/'3' to character names; slot is the active character's slot.
 *
 * A name can depend on what comes next (the first E of Augusta's Strike, Leap, Plunge only becomes
 * "Strike" once a second E follows). Then label.onRevise(index, text) is called, where index counts
 * the non-empty names returned so far (0 = the first), so the caller can update that earlier label.
 */
function createWwMoveLabeler(slotNames) {
    let fightStarted = false;
    let basicCount = 0;
    let lastKey = '';
    let after = ''; // the move LMB would follow (intro, skill, heavy, ...) for chain_entry
    let named = 0; // non-empty names returned so far, for onRevise
    let skillRun = null; // { slot, count, firstIndex }: E presses in a row
    const concerto = {}; // slot -> Concerto (WuwaLAB units)
    let onField = null; // slot of the character on the field, for the Outro on swap
    const gain = (slot, move) => {
        const v = (concerto[slot] || 0) + wwConcertoGain(rulesFor(slot), move);
        concerto[slot] = Math.min(WW_CONCERTO_FULL, v);
    };
    const isFull = (slot) => (concerto[slot] || 0) >= WW_CONCERTO_FULL;
    const who = (slot) => (slotNames && slotNames[slot]) || `Slot ${slot}`;
    const rulesFor = (slot) => wwRuleFor(who(slot));

    // Advance the Basic chain by `hits`, wrapping at the character's chain length.
    const basicLabel = (n, rule, hits, slot) => {
        const stage = (i) => (rule.basic > 0 ? ((i - 1) % rule.basic) + 1 : i);
        const first = stage(basicCount + 1);
        for (let i = 1; i <= hits; i++) gain(slot, { kind: 'basic', stage: stage(basicCount + i) });
        basicCount += hits;
        const last = stage(basicCount);
        return hits > 1 ? `${n} Basic ${first}-${last}` : `${n} Basic ${first}`;
    };

    const label = function (step, slot) {
        const text = name(step, slot);
        if (text) named += 1;
        return text;
    };
    label.onRevise = null;
    /** Concerto of the character in `slot` after the steps labeled so far, in points (0-100). */
    label.concerto = (slot) => (concerto[slot] || 0) / 100;

    // A move's own name, or "<character> <generic>" when the data doesn't have one.
    const own = (s, which, generic) => rulesFor(s).names[which] || `${who(s)} ${generic}`;

    // E pressed again right after E: name it from the character's skill_chain, renaming the first one.
    const skillLabel = (s) => {
        const chain = rulesFor(s).skillChain;
        if (skillRun && skillRun.slot === s) skillRun.count += 1;
        else skillRun = { slot: s, count: 1, firstIndex: named };
        const i = skillRun.count - 1;
        const useChain = !(chain.length < 2 || i === 0 || i >= chain.length);
        gain(s, { kind: 'skill', name: useChain ? chain[i] : rulesFor(s).names.skill });
        if (!useChain) return own(s, 'skill', 'Skill');
        if (i === 1 && typeof label.onRevise === 'function') label.onRevise(skillRun.firstIndex, chain[0]);
        return chain[i];
    };

    function name(step, slot) {
        const key = wwStepKey(step);
        if (!key) {
            if (step && step.type === 'wait' && Number(step.duration) >= WW_BASIC_RESET_MS) {
                basicCount = 0;
                after = '';
            }
            return '';
        }

        const isHold = step.type === 'hold' || step.type === 'hold_with_body';
        const s = slot || '1';
        const n = who(s);
        if (!onField && !WW_SLOTS.includes(key)) onField = s;
        const prevKey = lastKey;
        const wasStarted = fightStarted;
        const prevAfter = after;
        const tapE = key === 'e' && !isHold;
        if (!(tapE && prevKey === 'e')) skillRun = null; // any other move ends an E run
        lastKey = key;
        fightStarted = true;
        after = '';

        if (key === 'lmb') {
            const rule = rulesFor(s);
            if (step.type === 'spam') { basicCount = 0; return `${n} Basic spam`; }
            if (isHold && rule.hold) {
                basicCount = 0;
                after = 'heavy';
                if (rule.holdFull && isFull(s)) { gain(s, { kind: 'heavy', name: rule.holdFull }); return rule.holdFull; }
                gain(s, { kind: 'heavy', name: rule.hold === 'Heavy' ? '' : rule.hold });
                return rule.hold === 'Heavy' ? `${n} Heavy` : rule.hold; // "Steelclash" is named; plain Heavy isn't
            }
            if (!isHold && prevKey === 'space') {
                basicCount = 0;
                after = 'midair';
                gain(s, { kind: 'midair' });
                return `${n} Mid-air Attack`;
            }
            // A dodge can't be told from a perfect dodge by keys alone, hence the "?".
            const hits = Math.max(1, Number(step.chain_count) || 1);
            if (!isHold && prevKey === 'rmb') {
                basicCount = 0;
                after = 'dodge';
                gain(s, { kind: 'counter' });
                for (let i = 2; i <= hits; i++) gain(s, { kind: 'basic', stage: i });
                return hits > 1 ? `${n} Dodge Counter? +${hits - 1}` : `${n} Dodge Counter?`;
            }
            // Picking the chain back up after another move (e.g. LMB after Zani's Skill is Basic 3).
            const entry = prevAfter && rule.entry ? Number(rule.entry[prevAfter]) : 0;
            if (entry > 0) basicCount = entry - 1;
            const text = basicLabel(n, rule, hits, s);
            return isHold ? `${text} (held)` : text;
        }

        basicCount = 0;
        if (key === 'f') {
            if (!wasStarted) return 'Start fight';
            after = 'tune_break';
            gain(s, { kind: 'tune_break' });
            return 'Tune Break';
        }
        if (WW_SLOTS.includes(key)) {
            // Swapping out with a full bar fires the Outro, which empties it.
            if (onField && onField !== key && isFull(onField)) concerto[onField] = 0;
            onField = key;
            gain(key, { kind: 'intro' });
            after = 'intro';
            return own(key, 'intro', 'Intro');
        }
        if (key === 'e') {
            after = 'skill';
            if (!isHold) return skillLabel(s);
            gain(s, { kind: 'skill', name: rulesFor(s).names.skill });
            const skill = rulesFor(s).names.skill;
            return skill ? `${skill} (held)` : `${n} Held Skill`;
        }
        if (key === 'q') return 'Echo';
        if (key === 'r') { after = 'liberation'; gain(s, { kind: 'liberation' }); return own(s, 'liberation', 'Liberation'); }
        if (key === 'rmb') return 'Dodge';
        if (key === 'space') return 'Jump';
        if (key === 'shift') return 'Sprint';
        return '';
    }

    return label;
}

if (typeof module !== 'undefined') module.exports = { createWwMoveLabeler, wwStepKey, setWwCharacterData, setWwTimingData, wwRuleFor, wwConcertoGain };
