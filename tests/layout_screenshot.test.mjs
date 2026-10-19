import test from 'node:test';
import assert from 'node:assert/strict';
import {capturePhoneFigure} from '../scripts/layout_screenshot.mjs';

function pageFor(hasFigure, failScreenshot = false) {
  const calls = [];
  return {
    calls,
    async $(selector) {
      assert.equal(selector, '.diagram-figure, figure');
      return hasFigure ? {
        async evaluate(fn) { fn({scrollIntoView: () => calls.push('scroll')}); },
        async dispose() { calls.push('dispose'); },
      } : null;
    },
    async screenshot() {
      calls.push('screenshot');
      if (failScreenshot) throw new Error('capture failed');
    },
  };
}

test('figure-free preface continues to a later illustrated chapter', async () => {
  const pages = [pageFor(false), pageFor(true), pageFor(true)];
  let taken = false;
  for (const page of pages) {
    if (!taken) taken = await capturePhoneFigure(page, 'phone.png');
  }
  assert.equal(taken, true);
  assert.deepEqual(pages[0].calls, []);
  assert.deepEqual(pages[1].calls, ['scroll', 'screenshot', 'dispose']);
  assert.deepEqual(pages[2].calls, []);
});
test('static-figure preface is eligible without a Mermaid figure', async () => {
  const page = pageFor(true);
  assert.equal(await capturePhoneFigure(page, 'phone.png'), true);
  assert.deepEqual(page.calls, ['scroll', 'screenshot', 'dispose']);
});
test('real screenshot errors are propagated, not silently skipped', async () => {
  const page = pageFor(true, true);
  await assert.rejects(capturePhoneFigure(page, 'phone.png'), /capture failed/);
  assert.deepEqual(page.calls, ['scroll', 'screenshot', 'dispose']);
});
