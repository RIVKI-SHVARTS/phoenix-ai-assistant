'use strict';

// Interpret source Markdown links without interpreting any source HTML.
function sourceParts(text) {
  const parts = [];
  const links = /\[([^\]\n]+)\]\(([^)\n]+)\)/g;
  let end = 0;
  for (const match of text.matchAll(links)) {
    if (match.index > end) parts.push({ text: text.slice(end, match.index) });
    let href = null;
    try {
      const target = match[2].startsWith('mailto:') ? match[2].replace(/[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]/g, '') : match[2];
      const url = new URL(target);
      if (['https:', 'http:', 'mailto:', 'tel:'].includes(url.protocol) && !url.username && !url.password) href = url.href;
    } catch (_) { /* Retain the label when an original URL is invalid. */ }
    parts.push(href ? { text: match[1], href } : { text: match[1] });
    end = match.index + match[0].length;
  }
  if (end < text.length) parts.push({ text: text.slice(end) });
  return parts;
}

function answerParts(text) {
  return sourceParts(text).flatMap(part => {
    if (part.href) return [part];
    const pieces = [];
    const urls = /https?:\/\/[^\s<>"'\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069]+/g;
    let end = 0;
    for (const match of part.text.matchAll(urls)) {
      let target = match[0].replace(/[.,;!?:\u05c3]+$/g, '');
      while (target.endsWith(')') && (target.match(/\)/g) || []).length > (target.match(/\(/g) || []).length) target = target.slice(0, -1);
      if (match.index > end) pieces.push({text: part.text.slice(end, match.index)});
      let href = null;
      try {
        const url = new URL(target);
        if (['https:', 'http:'].includes(url.protocol) && !url.username && !url.password) href = url.href;
      } catch (_) { /* Keep invalid URLs as ordinary text. */ }
      pieces.push(href ? {text: target, href} : {text: target});
      end = match.index + target.length;
    }
    if (end < part.text.length) pieces.push({text: part.text.slice(end)});
    return pieces;
  });
}

function appendSourceText(element, text, detectBareUrls = false) {
  for (const part of detectBareUrls ? answerParts(text) : sourceParts(text)) {
    if (!part.href) { element.append(document.createTextNode(part.text)); continue; }
    const anchor = document.createElement('a');
    anchor.href = part.href;
    anchor.textContent = part.text;
    anchor.dir = 'auto';
    if (part.href.startsWith('https:') || part.href.startsWith('http:')) {
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
    }
    element.append(anchor);
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { sourceParts, answerParts, appendSourceText };

if (typeof document !== 'undefined') {
const form = document.getElementById('search-form');
const question = document.getElementById('question');
const button = document.getElementById('submit');
const reset = document.getElementById('new-chat');
const results = document.getElementById('results');
const status = document.getElementById('status');
const coverage = document.getElementById('coverage');
const loadingBrand = document.getElementById('loading-brand');
let history = [];
let ready = false;
let pending = false;
question.addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  if (event.repeat || pending || !ready || !question.value.trim()) return;
  form.requestSubmit(button);
});
fetch('/api/health').then(async response => {
  const data = await response.json();
  if (!response.ok) throw new Error();
  coverage.textContent = "חיפוש במידע מאתר הפניקס, עם מקורות וציטוטים תומכים.";
  ready = data.chat_ready;
  button.disabled = !ready;
  if (!ready) status.textContent = data.semantic_ready ? 'המענה אינו מוכן. יש להגדיר מפתח API בחלון שבו מופעל השרת ולהפעיל אותו מחדש.' : 'נדרש אינדקס סמנטי פעיל כדי להתחיל בשיחה.';
}).catch(() => { coverage.textContent = 'מאגר החיפוש אינו זמין. יש לפעול לפי הוראות ההפעלה.'; });
reset.addEventListener('click', () => {
  if (pending) return;
  history = [];
  results.replaceChildren();
  question.value = '';
  status.textContent = 'נפתחה שיחה חדשה.';
  question.focus();
});
function addSource(parent, source) {
  const url = new URL(source.page_url);
  if (url.protocol !== 'https:' || url.hostname !== 'www.fnx.co.il' || url.username || url.password) return;
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = `[${source.source_id}] ${source.heading || source.page_title}`;
  const body = document.createElement('p');
  body.className = 'source-text';
  appendSourceText(body, source.text, true);
  const link = document.createElement('a');
  link.href = url.href;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.textContent = `פתיחת המקור: ${source.page_title}`;
  const date = document.createElement('small');
  date.textContent = `נאסף בתאריך ${new Date(source.fetched_at).toLocaleDateString('he-IL')}`;
  details.append(summary, body, link, date);
  parent.append(details);
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || pending || !ready) return;
  pending = true;
  form.setAttribute('aria-busy', 'true');
  status.className = 'is-loading';
  if (loadingBrand) loadingBrand.hidden = false;
  button.disabled = true;
  reset.disabled = true;
  status.textContent = 'מחפש במקורות ובודק את התשובה…';
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, history }), signal: AbortSignal.timeout(300000) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'המענה לא הושלם.');
    const card = document.createElement('article');
    const title = document.createElement('h2');
    title.textContent = text;
    card.append(title);
    if (data.status === 'answered') {
      for (const claim of data.claims) {
        const paragraph = document.createElement('p');
        paragraph.className = 'answer-text';
        appendSourceText(paragraph, claim.text, true);
        const refs = [...new Set(claim.evidence.map(e => e.source_id))];
        for (const id of refs) {
          const source = data.sources.find(s => s.source_id === id);
          if (!source) continue;
          const anchor = document.createElement('a');
          anchor.href = source.page_url;
          anchor.target = '_blank';
          anchor.rel = 'noopener noreferrer';
          anchor.textContent = ` [${id}]`;
          anchor.setAttribute('aria-label', `פתיחת מקור ${id}`);
          paragraph.append(anchor);
        }
        card.append(paragraph);
        const proof = document.createElement('details');
        proof.className = 'claim-proof';
        const proofTitle = document.createElement('summary');
        proofTitle.textContent = 'למה זו התשובה? הצגת הציטוטים';
        proof.append(proofTitle);
        for (const evidence of claim.evidence) {
          const quote = document.createElement('blockquote');
          appendSourceText(quote, `[${evidence.source_id}] ${evidence.quote || ''}`, true);
          proof.append(quote);
        }
        card.append(proof);
      }
      for (const source of data.sources) addSource(card, source);
    } else {
      const paragraph = document.createElement('p');
      paragraph.className = 'notice';
      paragraph.textContent = data.answer;
      card.append(paragraph);
    }
    results.append(card);
    // Keep at most two previous turns and the newest turn, within the server limit.
    history.push({ role: 'user', content: text }, { role: 'assistant', content: data.answer });
    history = history.slice(-6);
    while (history.reduce((sum, row) => sum + row.content.length, 0) > 12000) history.splice(0, 2);
    question.value = '';
    status.textContent = data.status === 'answered' ? 'התשובה מוצגת עם מקורות. אפשר לשאול שאלת המשך.' : data.status === 'clarification' ? 'נדרש פירוט נוסף כדי להשיב.' : data.status === 'limited' ? 'הוצגה מגבלת המענה. אפשר לשאול על מידע המבוסס על המקורות.' : 'לא נמצא במקורות מידע מספיק למענה.';
    question.focus();
  } catch (error) {
    status.textContent = error.name === 'TimeoutError' ? 'המענה ארך זמן רב. המתיני לפני ניסיון נוסף; הבקשה עשויה עדיין להתבצע.' : (error.message || 'אירעה תקלה במענה.');
  } finally { pending = false; status.className = ''; if (loadingBrand) loadingBrand.hidden = true; form.setAttribute('aria-busy', 'false'); button.disabled = !ready; reset.disabled = false; }
});
}
