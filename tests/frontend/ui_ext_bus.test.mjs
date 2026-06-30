// Unit tests for the UI extension event bus (#2038, ticket 02).
import test from 'node:test';
import assert from 'node:assert/strict';

const bus = (await import('../../kestrel_sovereign/static/js/ui-ext/bus.js')).default;

test('emit delivers the payload to every subscriber', () => {
    bus._reset();
    const seen = [];
    bus.on('e', (p) => seen.push(['a', p]));
    bus.on('e', (p) => seen.push(['b', p]));
    bus.emit('e', { x: 1 });
    assert.deepEqual(seen, [['a', { x: 1 }], ['b', { x: 1 }]]);
});

test('off removes a single handler', () => {
    bus._reset();
    let calls = 0;
    const fn = () => { calls++; };
    bus.on('e', fn);
    bus.emit('e');
    bus.off('e', fn);
    bus.emit('e');
    assert.equal(calls, 1);
});

test('on returns an unsubscribe function', () => {
    bus._reset();
    let calls = 0;
    const unsub = bus.on('e', () => { calls++; });
    bus.emit('e');
    unsub();
    bus.emit('e');
    assert.equal(calls, 1);
});

test('emit on an event with no subscribers is a no-op', () => {
    bus._reset();
    assert.doesNotThrow(() => bus.emit('nobody', { a: 1 }));
});

test('a throwing handler is isolated; the others still run', () => {
    bus._reset();
    let ran = 0;
    bus.on('e', () => { throw new Error('boom'); });
    bus.on('e', () => { ran++; });
    assert.doesNotThrow(() => bus.emit('e'));
    assert.equal(ran, 1);
});

test('unsubscribing during dispatch does not skip a sibling handler', () => {
    bus._reset();
    const order = [];
    const a = () => { order.push('a'); bus.off('e', b); };
    const b = () => { order.push('b'); };
    bus.on('e', a);
    bus.on('e', b);
    bus.emit('e');
    // Snapshot-on-dispatch means b still runs this round despite a removing it.
    assert.deepEqual(order, ['a', 'b']);
});
