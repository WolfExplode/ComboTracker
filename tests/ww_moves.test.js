// Run with: node --test tests/ww_moves.test.js
const test = require('node:test');
const assert = require('node:assert');
const { createWwMoveLabeler, setWwCharacterData, wwRuleFor } = require('../static/ww_moves.js');

// Rules come from the hand-checked move data the app loads.
setWwCharacterData(require('../static/data/ww_characters.json'));

const names = { '1': 'Zani', '2': 'Phoebe', '3': 'Rover' };
const press = (input) => ({ type: 'press', input });

test('leading f starts the fight, later f is Tune Break', () => {
    const label = createWwMoveLabeler(names);
    assert.strictEqual(label(press('f'), '1'), 'Start fight');
    assert.strictEqual(label(press('e'), '1'), 'Restless Watch');
    assert.strictEqual(label(press('f'), '1'), 'Tune Break');
});

test('basic attacks count up and reset on other actions and long waits', () => {
    const label = createWwMoveLabeler(names);
    assert.strictEqual(label({ type: 'press_wait', input: 'lmb' }, '1'), 'Zani Basic 1');
    assert.strictEqual(label({ type: 'wait', mode: 'soft', duration: 200 }, '1'), '');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Basic 2');
    assert.strictEqual(label({ type: 'press_wait', input: 'lmb', chain_count: 2 }, '1'), 'Zani Basic 3-4');
    assert.strictEqual(label(press('rmb'), '1'), 'Dodge');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Dodge Counter?');
    assert.strictEqual(label({ type: 'wait', mode: 'soft', duration: 1200 }, '1'), '');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Basic 1');
});

test('basic chain wraps at the character\'s chain length', () => {
    const label = createWwMoveLabeler(names); // Zani: 4 hits
    assert.strictEqual(label({ type: 'press_wait', input: 'lmb', chain_count: 5 }, '1'), 'Zani Basic 1-1');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Basic 2');
});

test('hold lmb continues the chain for Hiyuki, is a Heavy for others', () => {
    const label = createWwMoveLabeler({ '1': 'Hiyuki', '2': 'Augusta' });
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 500 }, '1'), 'Hiyuki Basic 1 (held)');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 500 }, '1'), 'Hiyuki Basic 2 (held)');
    assert.strictEqual(label(press('2'), '2'), 'Stride of Goldenflare');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 500 }, '2'), 'Steelclash');
});

test('lmb after a jump is a mid-air attack', () => {
    const label = createWwMoveLabeler(names);
    label(press('space'), '1');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Mid-air Attack');
});

test('swaps name the incoming character and later moves use them', () => {
    const label = createWwMoveLabeler(names);
    assert.strictEqual(label(press('2'), '2'), 'Golden Grace');
    assert.strictEqual(label({ type: 'hold', input: 'e', duration: 800 }, '2'), 'To Where Light Shines (held)');
    assert.strictEqual(label({ type: 'wait', mode: 'mandatory', wait_for: 'r', duration: 3950 }, '2'), 'Dawn of Enlightenment');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 2300 }, '2'), 'Phoebe Heavy');
    assert.strictEqual(label(press('q'), '2'), 'Phoebe Echo');
});

test('missing team falls back to slot numbers', () => {
    const label = createWwMoveLabeler({});
    assert.strictEqual(label(press('3'), '3'), 'Slot 3 Intro');
});

test('lmb after another move picks the chain up where the kit says', () => {
    const label = createWwMoveLabeler({ '1': 'Zani', '2': 'Aemeath' });
    assert.strictEqual(label(press('e'), '1'), 'Restless Watch');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Basic 3'); // Zani's Skill -> Basic Stage 3
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Basic 4');
    assert.strictEqual(label(press('2'), '2'), 'Overture of Departure');
    assert.strictEqual(label(press('lmb'), '2'), 'Aemeath Basic 3'); // Aemeath's Intro -> Stage 3
    assert.strictEqual(label(press('q'), '2'), 'Aemeath Echo');
    assert.strictEqual(label(press('lmb'), '2'), 'Aemeath Basic 1'); // Echo has no follow-up
});

test('hand-typed team names still find their character', () => {
    assert.strictEqual(wwRuleFor('Agusta').hold, 'Steelclash');
    assert.strictEqual(wwRuleFor('Pheobe').basic, 3);
    assert.strictEqual(wwRuleFor('Yangyang Xuanling').hold, null);
    assert.strictEqual(wwRuleFor('Nobody').basic, 0);
});

test('Lucilla holds through her Basic chain instead of a Heavy', () => {
    const label = createWwMoveLabeler({ '1': 'Lucilla' });
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 400 }, '1'), 'Lucilla Basic 1 (held)');
});

test('E pressed again in a row follows the skill chain (Augusta: Strike, Leap, Plunge)', () => {
    const label = createWwMoveLabeler({ '1': 'Agusta', '2': 'Iuno' });
    const revised = [];
    label.onRevise = (index, text) => revised.push([index, text]);
    assert.strictEqual(label(press('f'), '1'), 'Start fight');
    assert.strictEqual(label(press('e'), '1'), "Warrior's Blade");
    assert.strictEqual(label({ type: 'wait', mode: 'soft', duration: 800 }, '1'), '');
    assert.strictEqual(label(press('e'), '1'), 'Leap');
    assert.deepStrictEqual(revised, [[1, 'Strike']]);
    assert.strictEqual(label(press('e'), '1'), 'Plunge');
    // Past the end of the chain, and after any other move, E is a plain Skill again.
    assert.strictEqual(label(press('e'), '1'), "Warrior's Blade");
    assert.strictEqual(label(press('lmb'), '1'), 'Agusta Basic 1');
    assert.strictEqual(label(press('e'), '1'), "Warrior's Blade");
    assert.strictEqual(revised.length, 1);
});

test('characters without a skill chain name every E by its own skill name', () => {
    const label = createWwMoveLabeler(names);
    label.onRevise = () => assert.fail('nothing to revise');
    assert.strictEqual(label(press('e'), '1'), 'Restless Watch');
    assert.strictEqual(label(press('e'), '1'), 'Restless Watch');
});

test('named moves drop the character; unnamed ones keep it', () => {
    const label = createWwMoveLabeler({ '1': 'Augusta', '2': 'Somebody New' });
    assert.strictEqual(label({ type: 'wait', mode: 'mandatory', wait_for: 'r', duration: 1600 }, '1'), 'Sunward Conquest');
    assert.strictEqual(label(press('q'), '1'), 'Augusta Echo');
    assert.strictEqual(label(press('2'), '2'), 'Somebody New Intro');
    assert.strictEqual(label(press('e'), '2'), 'Somebody New Skill');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 500 }, '2'), 'Somebody New Heavy');
});
