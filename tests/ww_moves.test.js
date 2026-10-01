// Run with: node --test tests/ww_moves.test.js
const test = require('node:test');
const assert = require('node:assert');
const { createWwMoveLabeler } = require('../static/ww_moves.js');

const names = { '1': 'Zani', '2': 'Phoebe', '3': 'Rover' };
const press = (input) => ({ type: 'press', input });

test('leading f starts the fight, later f is Tune Break', () => {
    const label = createWwMoveLabeler(names);
    assert.strictEqual(label(press('f'), '1'), 'Start fight');
    assert.strictEqual(label(press('e'), '1'), 'Zani Skill');
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
    assert.strictEqual(label(press('2'), '2'), 'Augusta Intro');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 500 }, '2'), 'Augusta Steelclash');
});

test('lmb after a jump is a mid-air attack', () => {
    const label = createWwMoveLabeler(names);
    label(press('space'), '1');
    assert.strictEqual(label(press('lmb'), '1'), 'Zani Mid-air Attack');
});

test('swaps name the incoming character and later moves use them', () => {
    const label = createWwMoveLabeler(names);
    assert.strictEqual(label(press('2'), '2'), 'Phoebe Intro');
    assert.strictEqual(label({ type: 'hold', input: 'e', duration: 800 }, '2'), 'Phoebe Held Skill');
    assert.strictEqual(label({ type: 'wait', mode: 'mandatory', wait_for: 'r', duration: 3950 }, '2'), 'Phoebe Liberation');
    assert.strictEqual(label({ type: 'hold', input: 'lmb', duration: 2300 }, '2'), 'Phoebe Heavy');
    assert.strictEqual(label(press('q'), '2'), 'Phoebe Echo');
});

test('missing team falls back to slot numbers', () => {
    const label = createWwMoveLabeler({});
    assert.strictEqual(label(press('3'), '3'), 'Slot 3 Intro');
});
