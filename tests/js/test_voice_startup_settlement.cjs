const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { test } = require('node:test');

async function factory() {
  const source = await fs.readFile(path.resolve(__dirname, '../../kestrel_sovereign/static/js/voice/startup.js'), 'utf8');
  return (await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).createStartupSettlement;
}

test('null public withdrawal remains distinct from late native success', async () => {
  const settlement = (await factory())();
  let finish;
  const pending = new Promise(resolve => { finish = resolve; });
  const caller = settlement.run(() => pending);
  const rejected = caller.then(() => false, reason => reason === null);
  settlement.close(null);
  assert.equal(await rejected, true);
  finish('actual late acquisition');
  const [record] = await settlement.whenSettled();
  assert.equal(record.status, 'fulfilled');
  assert.equal(record.value, 'actual late acquisition');
  assert.equal(record.publicStatus, 'rejected', 'null withdrawal must not look like public fulfillment');
  assert.equal(record.publicReason, null);
});

for (const original of [new Error('original public playback failure'), null, false, 0, '']) {
test(`public withdrawal ${String(original)} and real late native rejection retain separate identities`, async () => {
  const settlement = (await factory())();
  const native = new Error('later native abort');
  let finish;
  const operation = new Promise((_, reject) => { finish = reject; });
  const caller = settlement.run(() => operation);
  const publicOutcome = caller.then(value => ({ fulfilled: true, value }), error => ({ fulfilled: false, error }));
  settlement.close(original);
  const repeatedClose = settlement.close(new Error('later close must not overwrite'));
  const publicResult = await publicOutcome;
  let joined = false;
  const completion = settlement.whenSettled().then(records => { joined = true; return records; });
  await Promise.resolve();
  const prematurelyJoined = joined;
  finish(native);
  const [record] = await completion;
  assert.equal(publicResult.fulfilled, false);
  assert.equal(publicResult.error, original);
  assert.equal(repeatedClose, original);
  assert.equal(prematurelyJoined, false, 'public rejection is not native completion');
  assert.equal(record.status, 'rejected');
  assert.equal(record.reason, native, 'actual native failure must not be relabeled');
  assert.equal(record.publicReason, original, 'original public failure must not be overwritten');
});
}

test('ordinary native rejection remains the same public and native failure', async () => {
  const settlement = (await factory())();
  const original = new Error('native acquisition failure');
  await assert.rejects(settlement.run(async () => { throw original; }), error => error === original);
  const [record] = await settlement.whenSettled();
  assert.equal(record.reason, original);
  assert.equal(record.publicReason, original);
});

for (const rejected of [false, true]) {
test(`actual public race outcome survives later close observer rejected=${rejected}`, async () => {
  const settlement = (await factory())();
  const native = new Error('actual native race winner');
  const later = new Error('subsequent observer close');
  let finish;
  const operation = new Promise((resolve, reject) => { finish = rejected ? reject : resolve; });
  const caller = settlement.run(() => operation);
  const publicOutcome = caller.then(value => ({ value }), error => ({ error }));
  // Let nativeStartup adopt the actual operation before adding this observer.
  await Promise.resolve();
  await Promise.resolve();
  operation.then(() => settlement.close(later), () => settlement.close(later));
  finish(native);
  const actualPublic = await publicOutcome;
  const [record] = await settlement.whenSettled();
  assert.equal(rejected ? actualPublic.error : actualPublic.value, native);
  assert.equal(record.status, rejected ? 'rejected' : 'fulfilled');
  assert.equal(record.publicReason, rejected ? native : null,
    'public classification must be the actual race outcome, not later close state');
});
}
