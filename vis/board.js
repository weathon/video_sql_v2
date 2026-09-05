const data = window.BOARD_DATA;
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
const scoreClass = score => score === null ? '' : score >= 4 ? 'high' : score === 0 ? 'low' : 'mid';
const state = {view:'rollouts', batch:null, selected:null, tab:'answer', page:0, version:0, skill:null, skillMode:'diff', mediaPage:0};
const pending = new Map();
let selectionRequest = 0;
let currentDetail;

function loadData(key) {
  if (Object.hasOwn(window.BOARD_CACHE, key)) return Promise.resolve(window.BOARD_CACHE[key]);
  if (pending.has(key)) return pending.get(key);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'data/' + key + '.js';
    script.onload = () => { script.remove(); resolve(window.BOARD_CACHE[key]); };
    script.onerror = () => { script.remove(); reject(new Error('Cannot load exported file: ' + script.src)); };
    document.head.append(script);
  });
  pending.set(key, promise);
  return promise;
}

function renderOverview() {
  const scored = data.rollouts.filter(r => r.score !== null);
  const mean = scored.length ? (scored.reduce((n,r) => n+r.score,0)/scored.length).toFixed(2) : '—';
  $('metrics').innerHTML = [
    ['Rollouts', data.rollouts.length.toLocaleString(), `${scored.length} scored · ${data.rollouts.length-scored.length} no result`],
    ['Mean score', mean, 'raw reward · scored rollouts only'],
    ['Trace segments', data.trace_count.toLocaleString(), 'rollouts + summaries + updater'],
    ['Skill snapshots', data.versions.length-1, `${Object.keys(data.versions.at(-1).files).length} current skill files`],
  ].map(([label,value,note]) => `<div class="metric"><span class="metric-label">${label}</span><strong class="metric-value">${value}</strong><span class="metric-note">${note}</span></div>`).join('');
  $('landscape').innerHTML = data.batches.map(b => {
    const rows = data.rollouts.filter(r => r.batch === b.id);
    const values = rows.filter(r => r.score !== null);
    const average = values.length ? values.reduce((n,r) => n+r.score,0)/values.length : null;
    return `<button class="batch ${state.batch===b.id?'active':''}" data-batch="${esc(b.id)}" aria-pressed="${state.batch===b.id}" title="${esc(b.id)} · ${values.length} scored / ${rows.length} rollouts"><div class="batch-top"><span>E${Number(b.id.split('/')[0].split('-')[1])} / B${Number(b.id.split('/')[1].split('-')[1])}</span><b>${average===null?'—':average.toFixed(2)}</b></div><div class="bar-area"><div class="bar" style="height:${average===null?0:average/5*100}%"></div></div><small>${values.length}/${rows.length} scored</small></button>`;
  }).join('');
}

function renderList() {
  const query = $('search').value.trim().toLowerCase();
  const score = $('scoreFilter').value;
  const tool = $('toolFilter').value;
  let rows = data.rollouts.filter(r => (!state.batch || r.batch===state.batch) &&
    (score==='all' || (score==='none' ? r.score===null : r.score!==null && r.score===Number(score))) &&
    (tool==='all' || r.tools.includes(tool)) &&
    `${r.id} Q${r.question_id} ${r.question===null?'':r.question} ${r.answer===null?'':r.answer}`.toLowerCase().includes(query));
  if ($('sort').value==='low') rows.sort((a,b) => (a.score===null?Infinity:a.score)-(b.score===null?Infinity:b.score));
  if ($('sort').value==='high') rows.sort((a,b) => (b.score===null?-Infinity:b.score)-(a.score===null?-Infinity:a.score));
  if ($('sort').value==='calls') rows.sort((a,b) => b.calls-a.calls);
  const pages = Math.max(1, Math.ceil(rows.length/30));
  state.page = Math.min(state.page, pages-1);
  $('resultCount').textContent = rows.length;
  $('rolloutList').innerHTML = rows.slice(state.page*30,(state.page+1)*30).map(r => `<button class="rollout-row ${state.selected===r.id?'active':''}" data-rollout="${esc(r.id)}" aria-pressed="${state.selected===r.id}"><div><span class="row-id">${esc(r.batch.replace('epoch-','E').replace('/batch-',' / B'))} · Q${r.question_id} / R${r.rollout}</span><span class="row-question">${esc(r.question===null?'Question unavailable: no answer trace':r.question)}</span><span class="row-meta">${r.calls} calls · ${r.traces} segments${r.errors?` · ${r.errors} errors`:''}${r.score===null?' · no result':''}</span></div><span class="score ${scoreClass(r.score)}">${r.score===null?'—':r.score}</span></button>`).join('') || '<div class="empty">No matching rollouts.<br>Try another query or reset the filters.</div>';
  $('pagination').innerHTML = `<button data-page="-1" ${state.page===0?'disabled':''} aria-label="Previous">←</button><span>${state.page+1} / ${pages} · ${rows.length} results</span><button data-page="1" ${state.page>=pages-1?'disabled':''} aria-label="Next">→</button>`;
}

async function selectRollout(id) {
  const request = ++selectionRequest;
  state.selected = id;
  state.mediaPage = 0;
  history.replaceState(null,'','#rollout='+encodeURIComponent(id));
  renderList();
  $('detail').innerHTML = '<div class="empty">Loading this rollout…</div>';
  const detail = await loadData(id);
  if (request !== selectionRequest) return;
  currentDetail = detail;
  const row = data.rollouts.find(r => r.id===id);
  const siblings = data.rollouts.filter(r => r.batch===row.batch && r.question_id===row.question_id);
  $('detail').innerHTML = `<div class="detail-top"><div class="detail-eyebrow"><span>Q${row.question_id} / R${row.rollout} · ${esc(row.batch)}</span><span class="score ${scoreClass(row.score)}">${row.score===null?'—':row.score}</span></div><h3>${esc(row.question===null?'Question unavailable: no answer trace':row.question)}</h3><div class="detail-meta"><span>${esc(row.status)}</span><span>${row.calls} tool calls</span><span>${detail.media.length} media</span></div><div class="siblings"><span>Same question</span>${siblings.map(r => `<button data-rollout="${esc(r.id)}" class="${r.id===id?'active':''}">R${r.rollout} · ${r.score===null?'—':r.score}</button>`).join('')}</div></div><div class="tabs" role="tablist">${[['answer','Answer & reflection'],['trace','Execution trace'],['media','Media evidence'],['learning','Learning path']].map(([key,label]) => `<button data-tab="${key}" role="tab" aria-selected="${state.tab===key}" class="${state.tab===key?'active':''}">${label}</button>`).join('')}</div><div id="detailBody" class="detail-body"></div>`;
  renderDetail();
}

function renderDetail() {
  const detail = currentDetail;
  const row = data.rollouts.find(r => r.id===state.selected);
  document.querySelectorAll('[data-tab]').forEach(b => { b.classList.toggle('active', b.dataset.tab===state.tab); b.setAttribute('aria-selected', b.dataset.tab===state.tab); });
  const body = $('detailBody');
  if (state.tab==='answer') {
    const r = detail.result;
    body.innerHTML = (r ? `<div class="answer-grid"><article class="answer-card"><span class="label">MODEL ANSWER</span><p>${esc(r.answer)}</p></article><article class="answer-card reference"><span class="label">REFERENCE ANSWER</span><p>${esc(r.reference_answer)}</p></article></div><details open><summary>Reflection</summary><div class="prose">${esc(r.reflection)}</div></details>${r.meta_reflection ? `<details><summary>Meta reflection</summary><div class="prose">${esc(r.meta_reflection)}</div></details>`:''}<p class="trace-links"><a href="${data.source+row.id}/result.json" target="_blank" rel="noopener">Open original result.json ↗</a></p>` : '<div class="empty">No result.json at export time. Status and score are unknown.</div>') + detail.errors.map(e => `<details><summary>Error record · ${esc(e.path.split('/').slice(-2).join('/'))}</summary><pre class="error">${esc(pretty(e.error))}</pre></details>`).join('');
  } else if (state.tab==='trace') {
    const groups = [...new Set(detail.traces.map(t => t.group))];
    body.innerHTML = `<div class="trace-toolbar"><select id="traceGroup" aria-label="Trace stage"><option value="all">All stages</option>${groups.map(g => `<option>${esc(g)}</option>`).join('')}</select><select id="eventType" aria-label="Event type"><option value="all">All events</option><option value="function_call">Tool inputs</option><option value="function_call_output">Tool outputs</option><option value="reasoning">Reasoning</option></select><input id="traceSearch" type="search" placeholder="Search expanded segments…" aria-label="Search expanded events"></div><div id="traceList"></div>`;
    renderTraces(detail.traces, $('traceList'));
    $('traceGroup').onchange = () => document.querySelectorAll('#traceList>.trace').forEach(t => { t.hidden=$('traceGroup').value!=='all' && t.dataset.group!==$('traceGroup').value; });
    $('eventType').onchange = filterEvents;
    $('traceSearch').oninput = filterEvents;
  } else if (state.tab==='media') {
    const start=state.mediaPage*24;
    body.innerHTML = `<p class="media-note">Original media · Ordered by file path · Click to inspect full size</p><div class="media-grid">${detail.media.slice(start,start+24).map(path => `<button class="media-tile" data-image="${esc(data.source+path)}" data-caption="${esc(path)}"><img loading="lazy" src="${esc(data.source+path)}" alt="${esc(path.split('/').at(-1))}"><span>${esc(path.split('/').slice(-3).join('/'))}</span></button>`).join('')}</div>${detail.media.length===0?'<div class="empty">No media files for this rollout.</div>':`<div class="pagination"><button data-media-page="-1" ${state.mediaPage===0?'disabled':''}>← Previous</button><span>${start+1}–${Math.min(start+24,detail.media.length)} / ${detail.media.length}</span><button data-media-page="1" ${start+24>=detail.media.length?'disabled':''}>Next →</button></div>`}`;
  } else {
    const batch = data.batches.find(b => b.id===row.batch);
    const version = data.versions.findIndex(v => v.id===row.batch.replace('/','-'));
    body.innerHTML = `<span class="label">ROLLOUT → QUESTION SUMMARY → SKILL UPDATE</span>${detail.summary?`<details open><summary>Learning summary across this question’s rollouts</summary><div class="prose">${esc(detail.summary.summary)}</div></details>`:'<p class="prose">No saved question summary.</p>'}<div id="summaryTraces"></div>${batch.update?`<details><summary>Batch updater’s change rationale</summary><div class="prose">${esc(batch.update.reason)}</div></details>`:'<p class="prose">No updater result file for this batch.</p>'}${version>=0?`<button class="small-button" data-skill-version="${version}">Explore skills after this batch →</button>`:''}`;
    renderTraces(detail.summary_traces, $('summaryTraces'));
  }
}

function renderTraces(records, target) {
  target.innerHTML = records.map(t => `<details class="trace" data-key="${esc(t.key)}" data-group="${esc(t.group)}"><summary><span class="trace-summary-name">${esc(t.attempt)} ${esc(t.file)}</span><small>${t.events} events · ${t.complete?'complete':'incomplete'}</small></summary><div class="trace-body"><span class="label">Expand to load all events</span></div></details>`).join('') || '<div class="empty">No saved trace segments.</div>';
  target.querySelectorAll('details.trace').forEach(node => {
    node.addEventListener('toggle', async () => {
      if (!node.open || node.dataset.loaded) return;
      node.dataset.loaded='loading';
      const trace = await loadData(node.dataset.key);
      const calls = new Map(trace.history.filter(e => e.type==='function_call').map(e => [e.call_id,e.name]));
      node.querySelector('.trace-body').innerHTML = `<div class="trace-links"><a href="${esc(data.source+node.dataset.key)}" target="_blank" rel="noopener">Original JSON (including raw_responses) ↗</a></div>` + trace.history.map((e,i) => {
        const type=e.type || e.role;
        const isCall=type==='function_call', isOutput=type==='function_call_output';
        const name=isCall?e.name:isOutput?(calls.has(e.call_id)?calls.get(e.call_id):'unmatched call_id'):type;
        let content;
        if (isCall) content=e.arguments;
        else if (isOutput) content=e.output;
        else if (type==='reasoning') content = JSON.stringify({content:e.content,summary:e.summary},null,2);
        else content=pretty(e.content);
        const label=`${String(i+1).padStart(2,'0')} · ${type} · ${name}${e.call_id?' · '+e.call_id:''}`;
        const text=`<pre>${esc(pretty(content))}</pre>`;
        // Raw JSON text stays intact, including error returns and source truncation markers.
        return `<article class="event ${isCall?'call':isOutput?'output':type==='user'?'user':''}" data-type="${esc(type)}"><div class="event-label">${esc(label)}</div>${type==='reasoning'||type==='user'?`<details><summary>${type==='reasoning'?'Expand full reasoning':'Expand input prompt'}</summary>${text}</details>`:text}</article>`;
      }).join('') + (trace.final_output===null?'':`<details><summary>Final output</summary><pre class="prose">${esc(pretty(trace.final_output))}</pre></details>`);
      node.dataset.loaded='true';
      if ($('eventType')) filterEvents();
    });
  });
}

function filterEvents() {
  const type=$('eventType').value, query=$('traceSearch').value.toLowerCase();
  document.querySelectorAll('#traceList .event').forEach(event => {
    const typeMatch=type==='all'||event.dataset.type===type;
    const queryMatch=event.textContent.toLowerCase().includes(query);
    event.hidden=!(typeMatch&&queryMatch);
    if (query && queryMatch && typeMatch) {
      const details=event.querySelector('details');
      if (details) details.open=true;
    }
  });
}

function showView(view) {
  state.view=view;
  $('rolloutView').hidden=view!=='rollouts';
  $('skillView').hidden=view!=='skills';
  $('breadcrumb').textContent=view==='skills'?'Skill evolution':'Rollout explorer';
  $('heading').innerHTML=view==='skills'?'Watch the learning unfold<span>.</span>':'Every step tells a story<span>.</span>';
  $('subtitle').textContent=view==='skills'?'Inspect skill versions, line-by-line changes, and the traces behind each update.':'Explore answers, reasoning, and tool calls. Follow how skills evolve.';
  document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active',b.dataset.view===view));
  if (view==='skills') renderSkills();
  else { renderList(); history.replaceState(null,'',state.selected?'#rollout='+encodeURIComponent(state.selected):'#rollouts'); }
}

function renderSkills() {
  const version=data.versions[state.version];
  history.replaceState(null,'','#skills='+encodeURIComponent(version.id));
  $('versionTimeline').innerHTML=data.versions.map((v,i) => `<button class="version ${i===state.version?'active':''}" data-version="${i}" aria-pressed="${i===state.version}"><strong>${v.id==='skills'?'Current files':v.id.replace('epoch-','E').replace('-batch-',' / B')}</strong><span>${v.baseline?'First saved snapshot':Object.keys(v.changes).length+' files changed'}</span></button>`).join('');
  const batch=data.batches.find(b => b.id.replace('/','-')===version.id);
  $('updateContext').innerHTML=`<article class="update-card"><span class="label">${version.id==='skills'?'CURRENT FILE CONTENTS':'POST-UPDATE SNAPSHOT'}</span><div class="prose">${version.id==='skills'?'Current skills/ contents, compared with the last saved snapshot.':batch&&batch.update?esc(batch.update.reason):'No matching updater result for this snapshot.'}</div>${batch&&batch.update?`<div class="update-actions"><button class="text-button" data-show-batch="${esc(batch.id)}">Explore this batch’s rollouts ↗</button><span>commit ${esc(batch.update.skill_commit)}</span></div><details><summary>Inspect updater execution</summary><div id="updaterTraces"></div></details>`:''}</article>`;
  if ($('updaterTraces')) renderTraces(batch.traces,$('updaterTraces'));
  const names=[...new Set([...Object.keys(version.files),...Object.keys(version.changes)])];
  const query=$('skillSearch').value.trim().toLowerCase();
  const shown=names.filter(name => `${name} ${Object.hasOwn(version.files,name)?version.files[name]:''}`.toLowerCase().includes(query));
  if (!names.includes(state.skill)) state.skill=Object.keys(version.changes)[0] || names[0];
  $('skillList').innerHTML=shown.map(name => `<button class="skill-item ${state.skill===name?'active':''}" data-skill="${esc(name)}"><span>${esc(name.replace('.md',''))}</span>${Object.hasOwn(version.changes,name)?'<span class="change-dot" title="Changed since previous version">●</span>':''}</button>`).join('') || '<div class="empty">No matching skills.</div>';
  renderSkillContent();
}

function renderSkillContent() {
  const version=data.versions[state.version], name=state.skill;
  const changed=Object.hasOwn(version.changes,name);
  const present=Object.hasOwn(version.files,name);
  $('skillHeader').innerHTML=`<strong>${esc(name)}</strong><p>${version.baseline?'First saved snapshot; no earlier snapshot to compare.':changed?'Changed since the previous saved version.':'Identical to the previous saved version.'}</p><div class="skill-controls"><button class="small-button ${state.skillMode==='diff'?'active':''}" data-skill-mode="diff">± Changes</button><button class="small-button ${state.skillMode==='content'?'active':''}" data-skill-mode="content">Full content</button></div>`;
  if (state.skillMode==='content') $('skillBody').innerHTML=present?`<pre class="skill-markdown">${esc(version.files[name])}</pre>`:'<div class="empty">This skill was deleted in this version.</div>';
  else if (version.baseline) $('skillBody').innerHTML='<div class="empty">No earlier snapshot to compare.<br>Read this skill in the Full content view.</div>';
  else if (!changed) $('skillBody').innerHTML='<div class="empty">No changes to this skill since the previous version.</div>';
  else $('skillBody').innerHTML=`<pre class="diff">${version.changes[name].split('\n').map(line => `<span class="${line.startsWith('+')?'add':line.startsWith('-')?'remove':line.startsWith('@@')?'hunk':''}">${esc(line)}</span>`).join('')}</pre>`;
}

document.addEventListener('click', async event => {
  const button=event.target.closest('button');
  if (!button || button.disabled) return;
  if (button.dataset.view) showView(button.dataset.view);
  if (button.dataset.batch) { state.batch=state.batch===button.dataset.batch?null:button.dataset.batch; state.page=0; renderOverview(); renderList(); }
  if (button.dataset.rollout) await selectRollout(button.dataset.rollout);
  if (button.dataset.page) { state.page+=Number(button.dataset.page); renderList(); $('rolloutList').scrollTop=0; }
  if (button.dataset.tab) { state.tab=button.dataset.tab; renderDetail(); }
  if (button.dataset.mediaPage) { state.mediaPage+=Number(button.dataset.mediaPage); renderDetail(); }
  if (button.dataset.image) { $('largeImage').src=button.dataset.image; $('imageCaption').textContent=button.dataset.caption; $('lightbox').showModal(); }
  if (button.dataset.version) { state.version=Number(button.dataset.version); renderSkills(); }
  if (button.dataset.skill) { state.skill=button.dataset.skill; renderSkills(); }
  if (button.dataset.skillMode) { state.skillMode=button.dataset.skillMode; renderSkillContent(); }
  if (button.dataset.skillVersion) { state.version=Number(button.dataset.skillVersion); showView('skills'); window.scrollTo({top:0,behavior:'smooth'}); }
  if (button.dataset.showBatch) { state.batch=button.dataset.showBatch; state.page=0; $('search').value=''; $('scoreFilter').value='all'; $('toolFilter').value='all'; showView('rollouts'); renderOverview(); }
});

$('model').textContent=data.model;
$('runName').textContent=data.run;
$('built').textContent=data.built.replace('T',' ').slice(0,19);
$('navCount').textContent=data.rollouts.length;
$('versionCount').textContent=data.versions.length-1;
for (const score of [...new Set(data.rollouts.filter(r=>r.score!==null).map(r=>r.score))].sort((a,b)=>a-b)) $('scoreFilter').insertAdjacentHTML('beforeend',`<option value="${score}">Score ${score}</option>`);
for (const tool of Object.keys(data.tools).sort()) $('toolFilter').insertAdjacentHTML('beforeend',`<option value="${esc(tool)}">${esc(tool)}</option>`);
for (const id of ['search','scoreFilter','toolFilter','sort']) $(id).addEventListener(id==='search'?'input':'change',()=>{state.page=0;renderList();});
$('skillSearch').oninput=renderSkills;
$('clearFilters').onclick=()=>{state.batch=null;state.page=0;$('search').value='';$('scoreFilter').value='all';$('toolFilter').value='all';$('sort').value='original';renderOverview();renderList();};
$('reload').onclick=()=>location.reload();
document.querySelector('.brand').onclick=event=>{event.preventDefault();showView('rollouts');window.scrollTo({top:0,behavior:'smooth'});};
$('closeImage').onclick=()=>$('lightbox').close();
$('lightbox').onclick=event=>{if(event.target===$('lightbox')) $('lightbox').close();};
document.addEventListener('keydown',event=>{if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();$(state.view==='skills'?'skillSearch':'search').focus();}});
window.addEventListener('unhandledrejection',event=>{const container=state.view==='skills'?$('skillBody'):$('detail');container.innerHTML=`<pre class="error">${esc(event.reason.message)}</pre>`;});
renderOverview();
renderList();
const initialHash=location.hash;
if (initialHash.startsWith('#skills=')) {
  const version=data.versions.findIndex(v=>v.id===decodeURIComponent(initialHash.slice(8)));
  if (version<0) throw new Error('Unknown skill version in URL');
  state.version=version;showView('skills');
} else if (data.rollouts.length) {
  const id=initialHash.startsWith('#rollout=')?decodeURIComponent(initialHash.slice(9)):data.rollouts[0].id;
  if (!data.rollouts.some(r=>r.id===id)) throw new Error('Unknown rollout in URL');
  selectRollout(id);
}
