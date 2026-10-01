// Turn a community rotation transcript into ComboTracker inputs.
//
// Transcripts (AntoCrasher's rotation hub) read like
//   phoebe: eskill > lib > skill > echo > dash > ha > swap
//   rover:  ba23 > ha23 > echo > lib > fskill > swap
// Moves map to Wolf's keys: ba = lmb per stage digit, ha = hold(lmb), (e/f)skill = e, lib = r,
// echo = q, dash = rmb, jump = space, tbs = f, and swap/outro = the next character's slot key.
// Timings aren't in the transcripts, so the result has no waits; anything unknown is skipped and reported.

const WW_HOLD_LMB = 'hold(lmb, 0.5)';

// [pattern, keys for one occurrence]; a trailing digit run like "ba123" repeats the keys once per digit.
const WW_ROTATION_MOVES = [
    [/^[a-z]?ba$/, ['lmb']],            // ba, fba (forte), uba (ultimate), eba...
    [/^(n|b)\d?$/, ['lmb']],            // n1 / b2 style stage notation
    [/^[a-z]?ha$/, [WW_HOLD_LMB]],      // ha, fha
    [/^(heavy|charged)$/, [WW_HOLD_LMB]],
    [/^[a-z]?skill$/, ['e']],           // skill, eskill, fskill, skill2
    [/^(e|res)$/, ['e']],
    [/^(lib|ult|liberation|r)$/, ['r']],
    [/^(echo|q)$/, ['q']],
    [/^(dash|dodge)$/, ['rmb']],
    [/^(jump)$/, ['space']],
    [/^(plunge|mid-?air|midair)$/, ['space', 'lmb']],
    [/^(dc|dodgecounter|dodge counter)$/, ['rmb', 'lmb']],
    [/^(tbs|tunebreak|tune break|tb)$/, ['f']],
];

function wwNormalizeMove(move) {
    return String(move || '')
        .toLowerCase()
        .replace(/\(.*?\)/g, '')      // "(optional)" notes
        .replace(/\s*x\s*\d+$/, '')   // "ba x2" is handled below via count
        .trim();
}

// "ba123" -> { base: "ba", count: 3 }, "skill2" -> { base: "skill", count: 1 }, "ba x2" -> count 2
function wwSplitMove(move) {
    const raw = String(move || '').toLowerCase();
    const times = raw.match(/\s*x\s*(\d+)\s*$/);
    const norm = wwNormalizeMove(move);
    const m = norm.match(/^(.*?)(\d*)$/);
    let base = m ? m[1].trim() : norm;
    const digits = m ? m[2] : '';
    let count = 1;
    // Stage digits only repeat attacks; "skill2" means the second skill press, one press.
    if (digits && /^[a-z]?(ba|ha)$/.test(base)) count = digits.length;
    if (times) count *= parseInt(times[1], 10);
    return { base, count };
}

function wwMoveKeys(move) {
    const { base, count } = wwSplitMove(move);
    for (const [re, keys] of WW_ROTATION_MOVES) {
        if (re.test(base)) {
            const out = [];
            for (let i = 0; i < count; i++) out.push(...keys);
            return out;
        }
    }
    return null;
}

// Team member names -> slot key lookup ("Rover: Spectro" and "rover" both match "Rover").
function wwSlotFinder(members) {
    const slots = (members || []).map((m, i) => ({
        key: String(i + 1),
        name: String(m || '').toLowerCase().split(/[:\s]/)[0],
    }));
    return (character) => {
        const c = String(character || '').toLowerCase().split(/[:\s]/)[0];
        const hit = slots.find((s) => s.name && c && (s.name.startsWith(c) || c.startsWith(s.name)));
        return hit ? hit.key : '';
    };
}

// transcript: {sections: [{name, steps: [{character, moves: [..]}]}]}; members: team order (slot 1..3)
function rotationToInputs(transcript, members) {
    const slotOf = wwSlotFinder(members);
    const steps = [];
    for (const sec of (transcript && transcript.sections) || []) {
        for (const st of sec.steps || []) steps.push(st);
    }
    const keys = [];
    const unmapped = new Set();
    let active = slotOf(members && members[0]) || '1';

    const swapTo = (character) => {
        const slot = slotOf(character);
        if (!slot) {
            unmapped.add(`swap to ${character}`);
            return;
        }
        if (slot !== active) {
            keys.push(slot);
            active = slot;
        }
    };

    for (const st of steps) {
        swapTo(st.character); // covers the opener and transcripts that omit "swap"
        for (const move of st.moves || []) {
            const base = wwNormalizeMove(move);
            if (base === 'swap' || base === 'outro' || base === 'intro') continue; // done by the next step's swapTo
            const k = wwMoveKeys(move);
            if (k) keys.push(...k);
            else unmapped.add(move);
        }
    }
    return { inputs: keys.join(', '), unmapped: [...unmapped] };
}

if (typeof module !== 'undefined') module.exports = { rotationToInputs, wwMoveKeys, wwSlotFinder };
