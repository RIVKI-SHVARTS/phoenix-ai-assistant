const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
class Element {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.handlers={}; this.value=''; this.textContent=''; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  addEventListener(event, fn) { this.handlers[event]=fn; }
  setAttribute(key,value) { this[key]=value; }
  focus() {}
}
const tick = () => new Promise(resolve=>setImmediate(resolve));
test('Hebrew browser request, source rendering, followup history and reset', async () => {
  const elements = Object.fromEntries(['search-form','question','submit','new-chat','results','status','coverage'].map(id=>[id,new Element()]));
  const requests=[];
  const responses=[
    {status:'answered',answer:'יש לפנות למשטרה.',claims:[{text:'יש לפנות למשטרה.',evidence:[{source_id:'S1',quote:'ציטוט מדויק'}]}],sources:[{source_id:'S1',page_url:'https://www.fnx.co.il/test/',page_title:'תביעות',heading:'פריצה',text:'<script>unsafe</script> [טופס](https://www.fnx.co.il/test.pdf)',fetched_at:'2026-10-15'}]},
    {status:'insufficient',answer:'לא נמצא מידע מספיק.',claims:[],sources:[]},
    {status:'limited',answer:'אין לי גישה למידע אישי.',claims:[],sources:[]}
  ];
  const context={URL,AbortSignal,Set,Date,document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),createTextNode:text=>({text})},fetch:async(url,options)=>{
    if(url==='/api/health') return {ok:true,json:async()=>({pages:6,chunks:178,chat_ready:true})};
    requests.push(JSON.parse(options.body));
    return {ok:true,json:async()=>responses.shift()};
  }};
  vm.runInNewContext(fs.readFileSync(require.resolve('../static/app.js'),'utf8'),context);
  await tick();
  assert.equal(elements.submit.disabled,false);
  elements.question.value='פרצו לי לבית. מה קודם?';
  await elements['search-form'].handlers.submit({preventDefault(){}});
  assert.equal(requests[0].question,'פרצו לי לבית. מה קודם?');
  assert.equal(elements.results.children.length,1);
  const card=elements.results.children[0];
  const proof=card.children.find(node=>node.className==='claim-proof');
  assert.equal(proof.children[1].textContent,'[S1] ציטוט מדויק');
  const details=card.children.find(node=>node.tag==='details' && node.className!=='claim-proof');
  assert.ok(details);
  const body=details.children.find(node=>node.className==='source-text');
  assert.equal(body.children[0].text,'<script>unsafe</script> ');
  assert.equal(body.children[1].tag,'a');
  elements.question.value='ואיזה אישור?';
  await elements['search-form'].handlers.submit({preventDefault(){}});
  assert.equal(requests[1].history.length,2);
  assert.equal(requests[1].history[0].content,'פרצו לי לבית. מה קודם?');
  assert.equal(elements.results.children.length,2);
  elements.question.value='ומה מצב התביעה שלי?';
  await elements['search-form'].handlers.submit({preventDefault(){}});
  assert.equal(elements.results.children.length,3);
  assert.equal(elements.results.children[2].children[1].textContent,'אין לי גישה למידע אישי.');
  assert.match(elements.status.textContent,/הוצגה מגבלת המענה/);
  elements['new-chat'].handlers.click();
  assert.equal(elements.results.children.length,0);
});
