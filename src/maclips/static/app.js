'use strict';
const $ = selector => document.querySelector(selector);
const notice = (message, error=false) => { $('#notice').textContent = message; $('#notice').classList.toggle('error', error); };
async function request(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json','X-CSRF-Token':document.body.dataset.csrf}, body:JSON.stringify(body)});
  let data; try { data = await response.json(); } catch (_) { throw new Error(`Request failed (${response.status})`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
async function waitJob(data) {
  const id = data.job_id || data.jobid;
  if (!id) { location.reload(); return; }
  notice('Working locally. You can keep this page open.');
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    await refreshProgress();
    const response = await fetch('/jobs');
    if (!response.ok) throw new Error('Could not read job progress. Reload to check the stage status.');
    const payload = await response.json();
    const jobs = payload.jobs || payload;
    const job = Array.isArray(jobs) ? jobs.find(j => j.id === id) : jobs[id];
    if (!job) throw new Error('Job not found. Reload to inspect the saved stage status.');
    const status = typeof job === 'string' ? job : job.status;
    if (['failed','error','gated'].includes(status)) throw new Error(job.reason || job.error || 'The stage stopped at a gate.');
    if (['done','complete','completed','succeeded','finished'].includes(status)) { location.reload(); return; }
    notice(job.reason || job.progress || `Processing · ${status || 'running'}`);
  }
}
function action(element, callback) {
  if (!element) return;
  element.addEventListener(element.tagName === 'FORM' ? 'submit' : 'click', async event => {
    event.preventDefault();
    const controls = element.tagName === 'FORM' ? element.querySelectorAll('button') : [element];
    controls.forEach(x => x.disabled = true);
    try { await callback(event); } catch (error) { notice(error.message, true); }
    finally { controls.forEach(x => x.disabled = false); }
  });
}
function formData(form) { return Object.fromEntries(new FormData(form)); }
action($('#ingest-form'), async () => {
  const data = formData($('#ingest-form'));
  for (const key of ['expected_speakers','candidates','min_duration_s','max_duration_s']) data[key] = data[key] ? Number(data[key]) : null;
  if (data.clip_class === 'campaign' && !data.campaign_id) throw new Error('Choose a confirmed campaign brief first.');
  await waitJob(await request('/ingest', data));
});
action($('#brief-form'), async () => {
  notice('Extracting the brief. No confirmation is being made.');
  const data = await request('/brief/extract', formData($('#brief-form')));
  buildBriefForm(data.config || data);
  $('#confirm-form').hidden = false;
  notice('Review and edit the extracted fields, then confirm only when they match the brief.');
});
action($('#confirm-form'), async () => {
  const config = readBriefForm();
  await request('/brief/confirm', {config}); location.reload();
});
document.querySelectorAll('.rerun-form').forEach(form => action(form, async () => waitJob(await request(`/rerun/${encodeURIComponent(form.dataset.source)}`, formData(form)))));
$('#source-select')?.addEventListener('change', event => { location.href = `/?tab=review&source=${encodeURIComponent(event.target.value)}`; });
const editor = $('.editor');
let trimMode = 'in';
if (editor) {
  const clipId = encodeURIComponent(editor.dataset.clip);
  const video = $('#preview');
  const layout = $('#layout');
  let startWord = Number(editor.dataset.startWord), endWord = Number(editor.dataset.endWord);
  const selectedLayout = () => layout.value;
  const edit = async extra => {
    if (startWord > endWord) throw new Error('In must come before out.');
    await waitJob(await request(`/clips/${clipId}/edit`, {revision:Number(editor.dataset.revision),start_word:startWord, end_word:endWord, hook_text:$('#hook-text').value, end_card:$('#end-card').checked, end_card_text:$('#end-card-text').value, layout:selectedLayout(), ...extra}));
  };
  const changeMode = mode => { trimMode = mode; $('#in-mode').classList.toggle('selected', mode === 'in'); $('#out-mode').classList.toggle('selected', mode === 'out'); };
  $('#in-mode').addEventListener('click', () => changeMode('in'));
  $('#out-mode').addEventListener('click', () => changeMode('out'));
  document.querySelectorAll('.word').forEach(word => action(word, async () => {
    const index = Number(word.dataset.index);
    if (trimMode === 'in') { if (index > endWord) throw new Error('Choose an in word before the out word.'); startWord=index; }
    else { if (index < startWord) throw new Error('Choose an out word after the in word.'); endWord=index; }
    await edit({});
  }));
  const previewLayouts = JSON.parse(editor.dataset.previewLayouts || '[]');
  layout.addEventListener('change', async () => {
    if (previewLayouts.includes(selectedLayout())) { video.src = `/media/${clipId}/${encodeURIComponent(selectedLayout())}`; video.load(); }
    else { try { await edit({}); } catch (error) { notice(error.message, true); } }
  });
  action($('#save-hook'), () => edit({}));
  const endCard = $('#end-card');
  endCard.addEventListener('change', async () => {
    endCard.disabled = true;
    try { await edit({}); } catch (error) { endCard.checked = !endCard.checked; notice(error.message, true); }
    finally { endCard.disabled = false; }
  });
  action($('#save-end-card'), () => edit({}));
  async function decision(value, reason='') {
    const result = await request(`/clips/${clipId}/decision`, {revision:Number(editor.dataset.revision),decision:value, reason, platform:$('#platform').value, account:$('#account').value, account_class:$('#account-class').value, caption:$('#caption').value, layout:selectedLayout(), hook_text:$('#hook-text').value});
    await waitJob(result);
  }
  action($('#approve'), () => decision('approved'));
  $('#reject-toggle').addEventListener('click', () => { $('#reject-reasons').hidden = !$('#reject-reasons').hidden; });
  document.querySelectorAll('[data-reject]').forEach(button => action(button, () => decision('rejected', button.dataset.reject)));
  document.addEventListener('keydown', async event => {
    if (event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (!['j','k','a','r','[',']','l'].includes(key)) return;
    event.preventDefault();
    if (key === 'a') $('#approve').click();
    else if (key === 'r') $('#reject-toggle').click();
    else if (key === 'l') { layout.selectedIndex=(layout.selectedIndex+1)%layout.options.length; layout.dispatchEvent(new Event('change')); }
    else if (key === 'j' || key === 'k') {
      const links = [...document.querySelectorAll('[data-clip-nav]')];
      const current = links.findIndex(link => link.classList.contains('selected'));
      links[current+(key === 'j' ? 1 : -1)]?.click();
    } else {
      const sourceTime = Number(editor.dataset.start)+video.currentTime;
      const words = [...document.querySelectorAll('.word')].filter(w => Number.isFinite(Number(w.dataset.start)));
      const nearest = words.reduce((best,word) => !best || Math.abs(Number(word.dataset.start)-sourceTime)<Math.abs(Number(best.dataset.start)-sourceTime) ? word : best, null);
      if (nearest) { changeMode(key === '[' ? 'in' : 'out'); nearest.click(); }
    }
  });
}
document.querySelectorAll('.post-form').forEach(form => action(form, async () => {
  const data = formData(form);
  data.disclosure_ticked = form.elements.disclosure_ticked.checked;
  data.views = Number(data.views || 0);
  await waitJob(await request(`/posts/${encodeURIComponent(form.dataset.id)}`, data));
}));
document.querySelectorAll('[data-bundle]').forEach(button => action(button, async () => {
  const data = await request(`/bundles/${encodeURIComponent(button.dataset.bundle)}/open`, {});
  const files = data.files;
  if (!files) throw new Error('Bundle file links were not returned.');
  let links = button.parentElement.querySelector('.bundle-links');
  if (!links) { links = document.createElement('div'); links.className='bundle-links'; button.after(links); }
  links.replaceChildren();
  const entries = Array.isArray(files) ? files.map(file => [file.name || file.path || 'Download', file.url || file.download_url]) : Object.entries(files);
  for (const [name, value] of entries) {
    const url = typeof value === 'string' ? value : value.url || value.download_url;
    const link = document.createElement('a'); link.href=url; link.textContent=name; link.target='_blank'; link.rel='noopener'; links.append(link);
  }
}));
function updateTimers() {
  document.querySelectorAll('[data-deadline]').forEach(timer => {
    const value = timer.dataset.deadline;
    const numeric = Number(value);
    const deadline = Number.isFinite(numeric) && value.trim() !== '' ? numeric*(numeric<1e12 ? 1000 : 1) : Date.parse(value);
    if (!Number.isFinite(deadline)) { timer.textContent='Submission deadline unavailable'; return; }
    const remaining = deadline-Date.now();
    timer.classList.toggle('overdue', remaining<0);
    const minutes = Math.ceil(Math.abs(remaining)/60000);
    timer.textContent = remaining<0 ? `Submission window overdue by ${minutes} min` : `Submission window · ${minutes} min remaining`;
  });
}
updateTimers(); setInterval(updateTimers, 30000);

let extractedBrief = null;
const numericFields = new Set(['rate_per_1k','pool_total','pool_used','pool_used_pct_at_join','min_duration_s','max_duration_s','submission_window_min']);
const booleanFields = new Set(['overlays_allowed','commentary_allowed']);
const businessFields = new Set(['rate_per_1k','pool_total','pool_used_pct_at_join','deadline']);
function buildBriefForm(config) {
  extractedBrief = structuredClone(config);
  const container = $('#brief-fields'); container.replaceChildren();
  $('#brief-missing').textContent = config.missing?.length ? `Needs your attention: ${config.missing.join(', ')}` : 'Check every extracted value against the original brief.';
  for (const [name,value] of Object.entries(config)) {
    if (['raw_brief','confirmed','missing','unknown_confirmed'].includes(name)) continue;
    const label = document.createElement('label'); label.textContent=name.replaceAll('_',' ');
    let input;
    if (booleanFields.has(name)) {
      input=document.createElement('select');
      for (const [val,text] of [['','Not specified'],['true','Allowed'],['false','Not allowed']]) { const option=new Option(text,val); input.add(option); }
      input.value=value == null ? '' : String(value);
    } else if (Array.isArray(value) || name==='rate_per_1k_by_platform') {
      input=document.createElement('textarea'); input.rows=3;
      input.value=Array.isArray(value) ? value.join('\n') : Object.entries(value || {}).map(([platform,rate])=>`${platform}: ${rate}`).join('\n');
      input.dataset.kind=Array.isArray(value) ? 'array' : 'rates';
      input.placeholder=name==='rate_per_1k_by_platform' ? 'instagram: 1.5' : 'One requirement per line';
    } else {
      input=document.createElement('input'); input.type=numericFields.has(name) ? 'number' : 'text';
      if (numericFields.has(name)) { input.step='any'; input.min='0'; }
      input.value=value ?? '';
    }
    input.dataset.briefField=name; label.append(input); container.append(label);
    if (businessFields.has(name)) {
      const unknownLabel=document.createElement('label'); unknownLabel.className='checkbox';
      const checkbox=document.createElement('input'); checkbox.type='checkbox'; checkbox.dataset.unknownField=name;
      checkbox.checked=(config.unknown_confirmed || []).includes(name);
      unknownLabel.append(checkbox,document.createTextNode('I confirm this value is unknown')); container.append(unknownLabel);
    }
  }
}
function readBriefForm() {
  if (!extractedBrief) throw new Error('Extract a brief first.');
  const config=structuredClone(extractedBrief); config.confirmed=false; config.unknown_confirmed=[];
  for (const input of document.querySelectorAll('[data-brief-field]')) {
    const name=input.dataset.briefField, value=input.value.trim();
    if (input.dataset.kind==='array') config[name]=value ? value.split('\n').map(x=>x.trim()).filter(Boolean) : [];
    else if (input.dataset.kind==='rates') {
      config[name]={};
      for (const line of value.split('\n').filter(x=>x.trim())) {
        const match=line.match(/^([^:]+):\s*(\d+(?:\.\d+)?)$/);
        if (!match) throw new Error('Platform rates must use platform: number, one per line.');
        config[name][match[1].trim()]=Number(match[2]);
      }
    } else if (numericFields.has(name)) config[name]=value==='' ? null : Number(value);
    else if (booleanFields.has(name)) config[name]=value==='' ? null : value==='true';
    else config[name]=value==='' && ['deadline','category','disclosure_text'].includes(name) ? null : value;
  }
  for (const checkbox of document.querySelectorAll('[data-unknown-field]')) {
    if (!checkbox.checked) continue;
    const name=checkbox.dataset.unknownField;
    if (config[name] != null && config[name] !== '') throw new Error(`Clear ${name.replaceAll('_',' ')} before marking it unknown.`);
    if (name==='rate_per_1k' && Object.keys(config.rate_per_1k_by_platform || {}).length) throw new Error('Clear platform rates before marking the rate unknown.');
    config.unknown_confirmed.push(name);
  }
  return config;
}
async function refreshProgress() {
  const response=await fetch('/progress');
  if (!response.ok) throw new Error('Could not read stage progress.');
  const payload=await response.json();
  for (const source of payload.sources || []) {
    let container=[...document.querySelectorAll('[data-progress-source]')].find(x=>x.dataset.progressSource===String(source.id));
    if (!container) {
      const history=$('.source-history'); if (!history) continue;
      container=document.createElement('div'); container.className='stage-status'; container.dataset.progressSource=source.id;
      const title=document.createElement('h3'); title.textContent=source.path || 'Processing source'; history.append(title,container);
    }
    container.replaceChildren();
    for (const [stage,record] of Object.entries(source.progress || {})) {
      const line=document.createElement('div'); line.className='stage-line';
      const name=document.createElement('strong'); name.textContent=`${stage} · ${record.name || stage}`;
      const status=document.createElement('span'); status.textContent=record.status || 'pending'; line.append(name,status);
      if (record.reason) { const reason=document.createElement('p'); reason.textContent=record.reason; line.append(reason); }
      container.append(line);
    }
  }
}
// New previews appear while the single processing worker is still busy.
let previewCount = Number(document.body.dataset.previewCount || 0);
setInterval(async () => {
  if (document.body.dataset.tab === 'ingest') { await refreshProgress().catch(()=>{}); return; }
  if (document.body.dataset.tab !== 'review' || !$('#source-select')?.value) return;
  try {
    const data=await (await fetch('/progress')).json();
    const source=data.sources.find(s=>String(s.id)===$('#source-select').value);
    if (source && source.preview_count>previewCount && !document.querySelector('input:focus,textarea:focus') && !($('#preview') && !$('#preview').paused)) location.reload();
  } catch (_) { /* A stopped local server is shown by the next user action. */ }
}, 3000);
