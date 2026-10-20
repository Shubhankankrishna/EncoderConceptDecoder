import {finite,formatNumber as fmt,formatBytes,duration,age,localDate,statusInfo,lossChart,sparkline} from './view.js';

const $=id=>document.getElementById(id);
const text=(id,value)=>{$(id).textContent=value??'—';};
const state={snapshot:null,rows:[],paused:false,inFlight:false,selectedEval:null,sampleIndex:0,sampleTotal:0,sampleRequest:0,followLatest:true,serverOffset:0};
const node=(tag,content,className)=>{const n=document.createElement(tag);if(content!=null)n.textContent=content;if(className)n.className=className;return n;};
const selectedArm=new URLSearchParams(location.search).get('arm');
async function request(path) {
  if(selectedArm){const url=new URL(path,location.origin);url.searchParams.set('arm',selectedArm);path=url.pathname+url.search;}
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
  try{const response=await fetch(path,{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);return await response.json();}finally{clearTimeout(timer);}
}
function connection(label,mode='') {
  const target=$('connection');target.className=`connection ${mode}`;target.replaceChildren(node('i'),document.createTextNode(label));
}
function updateClock() {
  if(!state.snapshot)return;
  const run=state.snapshot.run,now=Date.now()/1000+state.serverOffset;
  text('remaining',finite(run.deadline)?duration(Math.max(0,run.deadline-now)):'—');
  const budget=run.deadline-run.start;
  if(finite(run.start)&&budget>0){const fraction=Math.max(0,Math.min(1,(now-run.start)/budget));$('progress').style.width=`${fraction*100}%`;text('progress-label',`${(fraction*100).toFixed(1)}%`);text('elapsed',`${duration(Math.min(Math.max(0,now-run.start),budget))} elapsed`);}
  const t=state.snapshot.latest.timestamp;
  text('metric-freshness',state.paused?'Live updates paused':finite(t)?`Last training row ${age(Math.max(0,now-t))}`:'Waiting for metrics');
}
function renderSignals(snapshot) {
  const target=$('signals');target.replaceChildren();
  const alerts=[...snapshot.alerts].sort((a,b)=>(a.severity==='critical'?-1:0)-(b.severity==='critical'?-1:0));
  if(!alerts.length)alerts.push({severity:'good',title:'No active telemetry alerts',detail:'Continue checking held-out loss and generated text together.'});
  for(const alert of alerts.slice(0,4)) {
    const item=node('div',null,`signal ${alert.severity}`),body=node('div');
    item.append(node('span',alert.severity==='good'?'✓':'!','signal-icon'));body.append(node('strong',alert.title),node('p',alert.detail));item.append(body);target.append(item);
  }
}
function renderEvents(snapshot) {
  const target=$('event-list');target.replaceChildren();
  const entries=[...snapshot.events];
  for(const alert of snapshot.alerts)entries.unshift({id:alert.id,level:alert.severity,title:alert.title,detail:alert.detail,timestamp:null});
  text('event-count',`${snapshot.events.length} saved events`);
  for(const event of entries.slice(0,30)) {
    const row=node('div',null,'event-row'),body=node('div'),time=node('time',finite(event.timestamp)?localDate(event.timestamp,false):'ACTIVE');
    body.append(node('strong',event.title),node('p',event.detail));row.append(node('span',null,`event-icon ${event.level}`),body,time);target.append(row);
  }
  if(!entries.length)target.append(node('p','No activity recorded yet.','muted'));
}
function renderCheckpoints(snapshot) {
  const target=$('checkpoints');target.replaceChildren();
  for(const checkpoint of snapshot.checkpoints) {
    const row=node('tr');row.append(node('td',checkpoint.name),node('td',formatBytes(checkpoint.bytes)),node('td',age(checkpoint.age_seconds)));target.append(row);
  }
  if(!snapshot.checkpoints.length){const row=node('tr'),cell=node('td','No saved checkpoint yet');cell.colSpan=3;row.append(cell);target.append(row);}
}
function renderSetup(snapshot) {
  const model=snapshot.model,data=snapshot.dataset;
  const facts=[['Architecture','GRAM · stochastic slow state'],['Trainable parameters',fmt(model.parameters?.trainable)],['Precision / device',`${model.precision||'—'} / ${(model.device||'—').toUpperCase()}`],['Effective batch',`${fmt(model.batch)} × ${fmt(model.accumulation)} accumulation`],['Dataset',`${fmt(data.unique_blocks)} blocks · ${fmt(data.examples)} windows`],['History',`${fmt(data.history)} preceding blocks`],['Codec',`${fmt(model.config.vocab)} choices · ${model.config.levels?.length||'—'} levels`],['Seed',fmt(model.seed)]];
  const finalTest=snapshot.evaluations?.find(e=>e.key==='final_test.json');
  if(finalTest)facts.push(['Final held-out test NLL',fmt(finalTest.nll_per_code,3)]);
  const target=$('experiment-facts');target.replaceChildren();for(const [key,value] of facts){const row=node('div');row.append(node('dt',key),node('dd',value));target.append(row);}
  text('parameters',finite(model.parameters?.trainable)?`${(model.parameters.trainable/1e6).toFixed(1)}M`:'—');
  const latest=snapshot.validation.latest;
  text('context-probe',fmt(latest?.context_nll_delta,4));
  text('text-speed',finite(latest?.end_to_end_content_tokens_per_second)?`${fmt(latest.end_to_end_content_tokens_per_second,1)} tokens/s`:'—');
  text('trigram-repeat',finite(latest?.repeated_trigram_fraction)?`${fmt(100*latest.repeated_trigram_fraction,1)}%`:'—');
  const boundary=latest?.boundary_continuity;
  text('boundary-repeat',boundary?`${fmt(boundary.same_word_seams)} / ${fmt(boundary.seam_count)}`:'—');
}
function renderChart() {
  $('loss-chart').innerHTML=lossChart(state.rows,state.snapshot?.validation.history||[],$('chart-range').value);
  $('chart-empty').hidden=state.rows.some(row=>finite(row.loss));
  $('spark-steps').innerHTML=sparkline(state.rows.slice(-30).map(r=>r.step));
  $('spark-speed').innerHTML=sparkline(state.rows.slice(-30).map(r=>r.blocks_per_second));
  $('spark-val').innerHTML=sparkline(state.snapshot?.validation.history.map(r=>r.nll_per_code)||[],'#bb985f');
}
function updateEvaluations(snapshot) {
  const choices=[...(snapshot.evaluations||snapshot.validation.history)].reverse(),target=$('evaluation-select');
  const oldOptions=[...target.options].map(o=>o.value).join(',');
  if(oldOptions!==choices.map(e=>e.key).join(',')) {
    target.replaceChildren();
    for(const e of choices){const option=node('option',`${e.split==='test'?'Test':'Valid'} · ${finite(e.step)?'step '+fmt(e.step):'final'} · NLL ${fmt(e.nll_per_code,3)}`);option.value=e.key;target.append(option);}
    if(!choices.length){const option=node('option','Waiting for evaluation');option.value='';target.append(option);}
  }
  const desired=state.followLatest?choices[0]?.key:state.selectedEval;
  if(desired&&desired!==state.selectedEval){state.selectedEval=desired;state.sampleIndex=0;loadSample();}
  if(state.selectedEval)target.value=state.selectedEval;
}
function render(snapshot) {
  state.snapshot=snapshot;state.serverOffset=snapshot.generated_at-Date.now()/1000;
  const latest=snapshot.latest,validation=snapshot.validation,host=snapshot.resources;
  const [label,mode]=statusInfo(snapshot.run.status);$('run-state').className=`status-pill ${mode}`;$('run-state').replaceChildren(node('i'),document.createTextNode(label));
  text('run-name',snapshot.run.name);text('deadline',localDate(snapshot.run.deadline));
  text('phase',snapshot.run.phase||'Raw-text next-concept pretraining');
  if(snapshot.run.status==='encoding'&&finite(snapshot.dataset.preparation?.encoded_blocks)){
    const preparation=snapshot.dataset.preparation;
    text('phase',`Encoding ${fmt(preparation.encoded_blocks)} / ${fmt(preparation.total_blocks)} native32 blocks`);
  }
  const contextHint=$('sample-context-hint');if(contextHint)contextHint.textContent=`${snapshot.dataset.history||'—'} preceding blocks`;
  text('step',fmt(latest.step));text('examples',`${fmt(latest.examples_seen)} next-block examples seen`);
  text('val-loss',fmt(validation.latest?.nll_per_code,3));
  const change=validation.relative_nll_reduction;
  text('val-change',finite(change)?`${change>=0?'↓':'↑'} ${Math.abs(change*100).toFixed(1)}% ${change>=0?'lower':'higher'} than initial validation`:'Initial validation baseline');$('val-change').className=`stat-note ${change>0?'good':''}`;
  if(validation.latest){
    $('val-change').append(document.createTextNode(` · step ${fmt(validation.latest.step)}`));
    if(validation.initial_checkpoint_only){text('val-change','Step 0 · untrained checkpoint');$('val-change').className='stat-note';}
  }
  $('throughput').replaceChildren(document.createTextNode(`${fmt(latest.blocks_per_second,1)} `),node('small','blocks/s'));
  text('update-speed',`${fmt(snapshot.rolling.median_update_seconds,2)}s / update · batch ${fmt((snapshot.model.batch||0)*(snapshot.model.accumulation||0))}`);
  text('kl',finite(latest.kl_sum)?`${latest.kl_sum.toExponential(2)} nats`:'—');text('sigma',fmt(latest.sigma_median,4));text('grad',fmt(latest.grad_norm,3));text('lr',finite(latest.lr)?latest.lr.toExponential(2):'—');
  text('ram',formatBytes(host.ram_available_bytes));text('ram-detail',`${formatBytes(host.ram_total_bytes)} total · available memory shown`);
  $('ram-meter').style.width=finite(host.ram_total_bytes)&&host.ram_total_bytes>0?`${Math.max(0,Math.min(100,host.ram_available_bytes/host.ram_total_bytes*100))}%`:'0%';
  text('mps-memory',formatBytes(latest.mps_driver_bytes));text('cpu',finite(host.cpu_percent)?`${fmt(host.cpu_percent,0)}%`:'—');text('disk',formatBytes(host.disk_free_bytes));text('swap',formatBytes(host.swap_used_bytes));
  text('power',host.ac_power===true?'AC POWER':host.ac_power===false?`BATTERY ${fmt(host.battery_percent)}%`:'POWER UNKNOWN');
  text('collector-note',`Refresh: 5s · ${snapshot.collector.metric_rows} logged updates · collector ${fmt(snapshot.collector.poll_seconds*1000,0)} ms`);
  if(!snapshot.collector.fresh||snapshot.collector.error){$('error-banner').hidden=false;text('error-banner',snapshot.collector.error||'The telemetry collector is delayed. Displayed measurements may be stale.');}else{$('error-banner').hidden=true;}
  renderSignals(snapshot);renderEvents(snapshot);renderCheckpoints(snapshot);renderSetup(snapshot);renderChart();updateEvaluations(snapshot);updateClock();
}
async function setupArmSelector(){
  try{
    const {runs}=await request('/api/runs');if(runs.length<2)return;
    const label=node('label','Model ','select-label'),select=node('select');select.setAttribute('aria-label','Choose model arm');
    for(const run of runs){const option=node('option',run.id.toUpperCase());option.value=run.id;select.append(option);}
    select.value=selectedArm||runs[0].id;
    select.addEventListener('change',()=>{const url=new URL(location.href);url.searchParams.set('arm',select.value);location.href=url.href;});
    label.append(select);document.querySelector('.top-actions').prepend(label);
    const exporter=document.querySelector('.button.export');if(exporter&&selectedArm)exporter.href=`/api/export.csv?arm=${encodeURIComponent(selectedArm)}`;
  }catch{}
}
setupArmSelector();
async function loadSample() {
  if(!state.selectedEval)return;
  const version=++state.sampleRequest;$('samples').setAttribute('aria-busy','true');
  try {
    const result=await request(`/api/samples?evaluation=${encodeURIComponent(state.selectedEval)}&offset=${state.sampleIndex}&limit=1`);
    if(version!==state.sampleRequest)return;
    const record=result.records[0];state.sampleTotal=result.total;
    text('sample-position',`Sample ${result.total?state.sampleIndex+1:0} / ${result.total}`);
    $('sample-prev').disabled=state.sampleIndex<=0;$('sample-next').disabled=state.sampleIndex+1>=result.total;
    text('sample-generated',record?.free_generated_text||'No generated text saved.');
    text('sample-reference',record?.reference_pretokenized_text||record?.reference_codec_reconstruction||'No reference saved.');
    text('sample-context',record?.observed_history||'Context was not included in this evaluation.');
    text('sample-nll',`NLL ${fmt(record?.nll_per_code,3)}`);
    const repetition=record?.repetition;
    const repetitive=repetition?.dominant_word_fraction>.5;
    text('sample-repetition',repetitive?`Repetition: “${repetition.dominant_word}” is ${fmt(repetition.dominant_word_fraction*100,0)}% of words`:'No dominant-word repetition detected');
    $('sample-repetition').className='sample-badge';
    const e=result.evaluation;
    text('sample-summary',`${e.repetitive_samples}/${e.samples} outputs repeat one word for more than half their word tokens. Saved validation samples; viewing them performs no inference.`);
    const seams=record?.seams||[];
    text('sample-seams',seams.length?seams.map((s,i)=>`Boundary ${i+1}\n${s.left_tail}\n→ ${s.right_head}`).join('\n\n'):'No rolling boundary samples in this evaluation.');
    if(record?.generated_chunks)text('sample-summary',`${record.generated_chunks.length} rolling generated chunks; reference shows the first next block. ${e.repetitive_samples}/${e.samples} outputs have dominant-word repetition. Boundary excerpts support human review; these diagnostics do not score semantic coherence.`);
    $('sample-summary').prepend(document.createTextNode(`Saved checkpoint step ${fmt(e.step)} · ${finite(e.timestamp)?age(Math.max(0,Date.now()/1000+state.serverOffset-e.timestamp)):'time unknown'}. `));
  } catch(error) {
    if(version===state.sampleRequest)text('sample-generated',`Unable to load this sample (${error.message}). Choose an evaluation to retry.`);
  } finally {if(version===state.sampleRequest)$('samples').removeAttribute('aria-busy');}
}
async function refresh() {
  if(state.paused||state.inFlight||document.hidden)return;
  state.inFlight=true;
  try {
    const [snapshot,history]=await Promise.all([request('/api/snapshot'),request('/api/history?limit=1200')]);
    state.rows=history.rows;render(snapshot);connection('Live · local');
  } catch(error) {
    connection('Disconnected','error');$('error-banner').hidden=false;
    text('error-banner',`Cannot reach the telemetry service (${error.message}). Showing the last received data. Training runs independently. Retrying every 5 seconds.`);
  } finally {state.inFlight=false;}
}
$('pause').addEventListener('click',()=>{state.paused=!state.paused;$('pause').textContent=state.paused?'Resume updates':'Pause updates';$('pause').setAttribute('aria-pressed',String(state.paused));if(state.paused)connection('Updates paused','paused');else refresh();updateClock();});
$('chart-range').addEventListener('change',renderChart);
$('evaluation-select').addEventListener('change',()=>{state.selectedEval=$('evaluation-select').value;state.followLatest=false;state.sampleIndex=0;loadSample();});
$('sample-prev').addEventListener('click',()=>{if(state.sampleIndex>0){state.sampleIndex--;loadSample();}});
$('sample-next').addEventListener('click',()=>{if(state.sampleIndex+1<state.sampleTotal){state.sampleIndex++;loadSample();}});
for(const link of document.querySelectorAll('.nav-item'))link.addEventListener('click',()=>{for(const other of document.querySelectorAll('.nav-item'))other.classList.toggle('active',other===link);});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
setInterval(refresh,5000);setInterval(updateClock,1000);refresh();
