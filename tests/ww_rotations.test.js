// Run with: node --test tests/ww_rotations.test.js
const test = require('node:test');
const assert = require('node:assert');
const { rotationToInputs, wwMoveKeys } = require('../static/ww_rotations.js');

const steps = (lines) => ({
    sections: [{ name: 'Opener', steps: lines.map(([character, moves]) => ({ character, moves: moves.split(' > ') })) }],
});

test('moves map to keys, stage digits repeat attacks', () => {
    assert.deepStrictEqual(wwMoveKeys('ba123'), ['lmb', 'lmb', 'lmb']);
    assert.deepStrictEqual(wwMoveKeys('uba12'), ['lmb', 'lmb']);
    assert.deepStrictEqual(wwMoveKeys('fha'), ['hold(lmb, 0.5)']);
    assert.deepStrictEqual(wwMoveKeys('skill2'), ['e']);
    assert.deepStrictEqual(wwMoveKeys('eskill'), ['e']);
    assert.deepStrictEqual(wwMoveKeys('lib2'), ['r']);
    assert.deepStrictEqual(wwMoveKeys('tbs'), ['f']);
    assert.deepStrictEqual(wwMoveKeys('dash'), ['rmb']);
    assert.strictEqual(wwMoveKeys('nf'), null);
});

test('swaps use the team slot order and the opener swaps in the first character', () => {
    const t = steps([
        ['phoebe', 'eskill > lib > swap'],
        ['zani', 'skill > ba3 > nf > swap'],
        ['rover', 'echo > outro'],
        ['phoebe', 'outro'],
    ]);
    const out = rotationToInputs(t, ['Zani', 'Phoebe', 'Rover']);
    assert.strictEqual(out.inputs, '2, e, r, 1, e, lmb, 3, q, 2');
    assert.deepStrictEqual(out.unmapped, ['nf']);
});

test('unknown characters are reported instead of guessed', () => {
    const out = rotationToInputs(steps([['zani', 'skill > swap'], ['verina', 'lib']]), ['Zani', 'Phoebe', 'Rover']);
    assert.strictEqual(out.inputs, 'e, r');
    assert.deepStrictEqual(out.unmapped, ['swap to verina']);
});
