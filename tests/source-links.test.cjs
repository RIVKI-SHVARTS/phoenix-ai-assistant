const test = require('node:test');
const assert = require('node:assert/strict');
const { sourceParts } = require('../static/app.js');
test('PDF, Hebrew label and surrounding text', () => {
  assert.deepEqual(sourceParts('מורידים את [טופס התביעה](https://www.fnx.co.il/form.pdf) ושולחים'), [
    {text:'מורידים את '}, {text:'טופס התביעה',href:'https://www.fnx.co.il/form.pdf'}, {text:' ושולחים'}]);
});
test('Email and phone links remain actionable', () => {
  const parts=sourceParts('[מייל](mailto:rechush@fnx.co.il) [טלפון](tel:*5070)');
  assert.equal(parts[0].href,'mailto:rechush@fnx.co.il');
  assert.equal(parts[2].href,'tel:*5070');
});
test('Invisible formatting from a source does not break its mailto target', () => {
  const parts=sourceParts('[מייל](mailto:Tviotgimel@fnx.co.il\u200b)');
  assert.equal(parts[0].href,'mailto:Tviotgimel@fnx.co.il');
});
test('Unsafe protocols and credentials never become links', () => {
  for(const target of ['javascript:alert%281%29','data:text/html,hello','https://user:secret@example.com/']) {
    assert.deepEqual(sourceParts(`[תווית](${target})`),[{text:'תווית'}]);
  }
});
test('Raw source HTML remains plain text', () => {
  assert.deepEqual(sourceParts('<img src=x onerror=alert(1)>'),[{text:'<img src=x onerror=alert(1)>'}]);
});
test('Multiple links and encoded parentheses are retained', () => {
  const p=sourceParts('[א](https://example.com/a%28b%29.pdf)\n[ב](https://example.com/b)');
  assert.equal(p.filter(x=>x.href).length,2);
  assert.equal(p[0].href,'https://example.com/a%28b%29.pdf');
});
