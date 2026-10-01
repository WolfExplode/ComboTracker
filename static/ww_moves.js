// Wuthering Waves move names for timeline steps.
//
// Inputs are decoded by rule, walking the timeline in order:
//   leading f = start the fight (ToA prompt), any later f = Tune Break
//   1/2/3 = swap in that slot (Intro)
//   lmb = Basic 1, 2, 3... (reset by any other action or a wait of 1s+; wraps at the chain length)
//   lmb right after rmb = Dodge Counter?, right after space = Mid-air Attack
//   hold(lmb) = Heavy, or another Basic hit for characters whose hold continues the chain
//   e / hold(e) = Skill / Held Skill, q = Echo, r = Liberation, rmb = Dodge, space = Jump, shift = Sprint

const WW_SLOTS = ['1', '2', '3'];

// Per-character Normal Attack rules from each kit's text on encore.moe (see wuwa-research/move-rules.md).
// basic = hits in the Basic chain before it restarts; hold = what hold(lmb) does:
// a Heavy Attack name, or null when holding just continues the Basic chain.
const WW_CHARACTER_RULES = {
    zani: { basic: 4, hold: 'Heavy' },
    phoebe: { basic: 3, hold: 'Heavy' },
    pheobe: { basic: 3, hold: 'Heavy' },
    aemeath: { basic: 4, hold: 'Heavy Charged' },
    lynae: { basic: 3, hold: null },
    mornye: { basic: 4, hold: 'Heavy' },
    augusta: { basic: 4, hold: 'Steelclash' },
    iuno: { basic: 3, hold: 'Heavy' },
    shorekeeper: { basic: 4, hold: 'Heavy' },
    hiyuki: { basic: 3, hold: null },
    lucilla: { basic: 3, hold: 'Heavy' },
    suisui: { basic: 4, hold: null },
};
const WW_DEFAULT_RULE = { basic: 0, hold: 'Heavy' }; // basic 0 = unknown chain length, never wraps

// The in-game Basic chain resets after a pause this long.
const WW_BASIC_RESET_MS = 1000;

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
 */
function createWwMoveLabeler(slotNames) {
    let fightStarted = false;
    let basicCount = 0;
    let lastKey = '';
    const who = (slot) => (slotNames && slotNames[slot]) || `Slot ${slot}`;
    const rulesFor = (slot) => WW_CHARACTER_RULES[who(slot).toLowerCase()] || WW_DEFAULT_RULE;

    // Advance the Basic chain by `hits`, wrapping at the character's chain length.
    const basicLabel = (n, rule, hits) => {
        const stage = (i) => (rule.basic > 0 ? ((i - 1) % rule.basic) + 1 : i);
        const first = stage(basicCount + 1);
        basicCount += hits;
        const last = stage(basicCount);
        return hits > 1 ? `${n} Basic ${first}-${last}` : `${n} Basic ${first}`;
    };

    return function label(step, slot) {
        const key = wwStepKey(step);
        if (!key) {
            if (step && step.type === 'wait' && Number(step.duration) >= WW_BASIC_RESET_MS) basicCount = 0;
            return '';
        }

        const isHold = step.type === 'hold' || step.type === 'hold_with_body';
        const s = slot || '1';
        const n = who(s);
        const prevKey = lastKey;
        const wasStarted = fightStarted;
        lastKey = key;
        fightStarted = true;

        if (key === 'lmb') {
            const rule = rulesFor(s);
            if (step.type === 'spam') { basicCount = 0; return `${n} Basic spam`; }
            if (isHold && rule.hold) { basicCount = 0; return `${n} ${rule.hold}`; }
            if (!isHold && prevKey === 'space') { basicCount = 0; return `${n} Mid-air Attack`; }
            // A dodge can't be told from a perfect dodge by keys alone, hence the "?".
            const hits = Math.max(1, Number(step.chain_count) || 1);
            if (!isHold && prevKey === 'rmb') {
                basicCount = 0;
                return hits > 1 ? `${n} Dodge Counter? +${hits - 1}` : `${n} Dodge Counter?`;
            }
            const text = basicLabel(n, rule, hits);
            return isHold ? `${text} (held)` : text;
        }

        basicCount = 0;
        if (key === 'f') return wasStarted ? 'Tune Break' : 'Start fight';
        if (WW_SLOTS.includes(key)) return `${who(key)} Intro`;
        if (key === 'e') return isHold ? `${n} Held Skill` : `${n} Skill`;
        if (key === 'q') return `${n} Echo`;
        if (key === 'r') return `${n} Liberation`;
        if (key === 'rmb') return 'Dodge';
        if (key === 'space') return 'Jump';
        if (key === 'shift') return 'Sprint';
        return '';
    };
}

if (typeof module !== 'undefined') module.exports = { createWwMoveLabeler, wwStepKey };
