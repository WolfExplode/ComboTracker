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
// Per-character rules (chain length, what hold(lmb) does, where LMB picks the chain back up after
// another move) come from static/data/ww_characters.json via setWwCharacterData().

const WW_SLOTS = ['1', '2', '3'];

// basic = hits in the Basic chain before it restarts (0 = unknown, never wraps); hold = what hold(lmb)
// is called (null when holding just continues the Basic chain); entry = { intro|skill|liberation|
// tune_break|heavy|dodge|midair: stage } for "LMB right after that move is Basic Stage N";
// skillChain = names for E pressed again and again in a row ([] = every E is just "Skill").
const WW_DEFAULT_RULE = { basic: 0, hold: 'Heavy', entry: {}, skillChain: [] };
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
            basic: Number(b.hits) || 0,
            hold: b.hold_lmb === 'chain' ? null : (b.hold_label || 'Heavy'),
            entry: c.chain_entry || {},
            skillChain: Array.isArray(c.skill_chain) ? c.skill_chain : [],
        };
    });
    wwCharacterRules = rules;
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
    const who = (slot) => (slotNames && slotNames[slot]) || `Slot ${slot}`;
    const rulesFor = (slot) => wwRuleFor(who(slot));

    // Advance the Basic chain by `hits`, wrapping at the character's chain length.
    const basicLabel = (n, rule, hits) => {
        const stage = (i) => (rule.basic > 0 ? ((i - 1) % rule.basic) + 1 : i);
        const first = stage(basicCount + 1);
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

    // E pressed again right after E: name it from the character's skill_chain, renaming the first one.
    const skillLabel = (n, s) => {
        const chain = rulesFor(s).skillChain;
        if (skillRun && skillRun.slot === s) skillRun.count += 1;
        else skillRun = { slot: s, count: 1, firstIndex: named };
        const i = skillRun.count - 1;
        if (chain.length < 2 || i === 0 || i >= chain.length) return `${n} Skill`;
        if (i === 1 && typeof label.onRevise === 'function') label.onRevise(skillRun.firstIndex, `${n} ${chain[0]}`);
        return `${n} ${chain[i]}`;
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
            if (isHold && rule.hold) { basicCount = 0; after = 'heavy'; return `${n} ${rule.hold}`; }
            if (!isHold && prevKey === 'space') { basicCount = 0; after = 'midair'; return `${n} Mid-air Attack`; }
            // A dodge can't be told from a perfect dodge by keys alone, hence the "?".
            const hits = Math.max(1, Number(step.chain_count) || 1);
            if (!isHold && prevKey === 'rmb') {
                basicCount = 0;
                after = 'dodge';
                return hits > 1 ? `${n} Dodge Counter? +${hits - 1}` : `${n} Dodge Counter?`;
            }
            // Picking the chain back up after another move (e.g. LMB after Zani's Skill is Basic 3).
            const entry = prevAfter && rule.entry ? Number(rule.entry[prevAfter]) : 0;
            if (entry > 0) basicCount = entry - 1;
            const text = basicLabel(n, rule, hits);
            return isHold ? `${text} (held)` : text;
        }

        basicCount = 0;
        if (key === 'f') {
            if (!wasStarted) return 'Start fight';
            after = 'tune_break';
            return 'Tune Break';
        }
        if (WW_SLOTS.includes(key)) { after = 'intro'; return `${who(key)} Intro`; }
        if (key === 'e') { after = 'skill'; return isHold ? `${n} Held Skill` : skillLabel(n, s); }
        if (key === 'q') return `${n} Echo`;
        if (key === 'r') { after = 'liberation'; return `${n} Liberation`; }
        if (key === 'rmb') return 'Dodge';
        if (key === 'space') return 'Jump';
        if (key === 'shift') return 'Sprint';
        return '';
    }

    return label;
}

if (typeof module !== 'undefined') module.exports = { createWwMoveLabeler, wwStepKey, setWwCharacterData, wwRuleFor };
