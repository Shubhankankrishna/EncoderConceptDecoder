'use strict';
const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = x => typeof x === 'number' && Number.isFinite(x);
const num = (x, digits=0) => finite(x) ? x.toLocaleString('en-IN', {minimumFractionDigits:digits, maximumFractionDigits:digits}) : '—';
const pct = (x, digits=1) => finite(x) ? num(x*100,digits)+'%' : '—';
const time = x => x ? new Date(x*1000).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})+' IST' : '—';
const age = seconds => !finite(seconds) ? 'unavailable' : seconds < 60 ? Math.round(seconds)+'s ago' : seconds < 3600 ? Math.floor(seconds/60)+'m ago' : (seconds/3600).toFixed(1)+'h ago';
const ago = t => t ? age(Date.now()/1000-t) : 'not yet saved';
const colors={grid:'#09877d',prog:'#6879cf',state:'#ca963d',plan:'#c47787',text:'#7198ad'};
const names={grid:'Grid',prog:'Program',state:'State',plan:'Planning',text:'Text'};
const titles={overview:['Overview','A clear view of the run, from training to evidence.'],training:['Training','Follow learning, throughput and the data behind each update.'],concept:['Concept health','Understand the future-code predictions and the targets they learn from.'],evaluation:['Evaluation','The registered questions, measured outcomes and supporting evidence.'],system:['System & network','Machine health and access from any device on the same Wi-Fi.'],evidence:['Logs & evidence','Inspect the original records behind every reading.']};
let state=null, family='all', windowSize='all', refreshing=false, logGeneration=0, connectionFailed=false;
let view = titles[location.hash.slice(1)] ? location.hash.slice(1) : 'overview';
const chartStore = new Map();
function set(id, html){$(id).innerHTML=html;}
function pill(label, cls=''){return `<span class="pill ${esc(cls)}">${esc(label)}</span>`;}
function metric(label, value, unit, note){return `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}${unit?`<small>${esc(unit)}</small>`:''}</div><div class="metric-note">${esc(note)}</div></div>`;}
function kv(items){return items.map(([key,value])=>`<div class="key-value"><span>${esc(key)}</span><b>${esc(value)}</b></div>`).join('');}
function empty(title, body){return `<div class="empty"><b>${esc(title)}</b>${esc(body)}</div>`;}
function jsonDetails(label,data){return `<details class="evidence-block"><summary>${esc(label)}</summary><pre class="evidence-json">${esc(JSON.stringify(data,null,2))}</pre></details>`;}
function minmax(values){let lo=Infinity,hi=-Infinity;for(const v of values){lo=Math.min(lo,v);hi=Math.max(hi,v);}return [lo,hi];}

function chart(id, series, opts={}){
  const el=$(id); if(!el) return;
  chartStore.set(id,{series,opts});
  const all=series.flatMap(s=>s.points.filter(p=>finite(p.x)&&finite(p.y)));
  if(!all.length){el.innerHTML='<div class="chart-empty">No observations available yet.</div>';return;}
  const W=Math.max(270,el.clientWidth), H=Math.max(150,el.clientHeight), left=43,right=12,top=13,bottom=29;
  const [xmin,xmax0]=minmax(all.map(p=>p.x));const xmax=xmax0===xmin?xmin+1:xmax0;
  let [ymin,ymax]=minmax(all.map(p=>p.y));
  if(opts.zero) ymin=0;
  if(finite(opts.max)) ymax=opts.max;
  if(ymax===ymin){ymax+=Math.max(.05,Math.abs(ymax)*.1);ymin=Math.max(0,ymin-Math.max(.05,Math.abs(ymin)*.1));}
  if(!opts.max)ymax+=(ymax-ymin)*.1;
  const X=x=>left+(x-xmin)/(xmax-xmin)*(W-left-right);
  const Y=y=>H-bottom-(y-ymin)/(ymax-ymin)*(H-top-bottom);
  const labelY=y=>opts.percent?pct(y,0):Math.abs(y)>=1000?num(y/1000,1)+'k':num(y,Math.abs(ymax)<2?2:1);
  let grid='';for(let i=0;i<4;i++){const val=ymin+(ymax-ymin)*i/3;grid+=`<line x1="${left}" y1="${Y(val)}" x2="${W-right}" y2="${Y(val)}" stroke="#eaf0f3"/><text x="${left-9}" y="${Y(val)+3}" text-anchor="end">${esc(labelY(val))}</text>`;}
  for(let i=0;i<4;i++){const val=xmin+(xmax-xmin)*i/3;const label=opts.clock?new Date(val*1000).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:false}):num(val);grid+=`<text x="${X(val)}" y="${H-5}" text-anchor="${i===0?'start':i===3?'end':'middle'}">${esc(label)}</text>`;}
  let paths='';for(const s of series){const points=s.points.filter(p=>finite(p.x)&&finite(p.y));if(!points.length)continue;
    const line=points.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(1)+','+Y(p.y).toFixed(1)).join(' ');
    paths+=`<path d="${line}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const last=points[points.length-1];paths+=`<circle cx="${X(last.x)}" cy="${Y(last.y)}" r="3" fill="${s.color}"/>`;
  }
  const desc=series.map(s=>{const p=s.points.filter(p=>finite(p.y)).at(-1);return s.label+': latest '+(p?labelY(p.y):'unavailable');}).join('. ');
  el.innerHTML=`<svg role="img" aria-label="${esc(desc)}" viewBox="0 0 ${W} ${H}"><title>${esc(desc)}</title>${grid}${paths}<line class="crosshair" y1="${top}" y2="${H-bottom}" visibility="hidden"/></svg><div class="tooltip hidden"></div>`;
  const tip=el.querySelector('.tooltip'),line=el.querySelector('.crosshair');
  el.onpointermove=e=>{const rect=el.getBoundingClientRect(),px=Math.max(left,Math.min(W-right,e.clientX-rect.left));const target=xmin+(px-left)/(W-left-right)*(xmax-xmin);
    const near=series.map(s=>{let p=null,d=Infinity;for(const item of s.points){if(finite(item.y)&&Math.abs(item.x-target)<d){p=item;d=Math.abs(item.x-target);}}return {s,p};}).filter(r=>r.p);
    if(!near.length)return;
    const step=near[0].p.x;tip.innerHTML=`<b>${esc(opts.clock?time(step):'Step '+num(step))}</b><br>`+near.map(({s,p})=>esc(s.label)+': '+esc(opts.percent?pct(p.y):num(p.y,3))).join('<br>');
    tip.classList.remove('hidden');tip.style.left=Math.min(W-tip.offsetWidth-4,Math.max(0,px+10))+'px';line.setAttribute('x1',px);line.setAttribute('x2',px);line.setAttribute('visibility','visible');};
  el.onpointerleave=()=>{tip.classList.add('hidden');line.setAttribute('visibility','hidden');};
}
function series(rows,key,color='#09877d',label=key){return {label,color,points:rows.map(r=>({x:r.step,y:typeof key==='function'?key(r):r[key]}))};}
function updateFreshness(){if(!state)return;const stale=connectionFailed||state.collector_error||Date.now()/1000-state.generated>Math.max(45,state.interval*3);$('connection-dot').classList.toggle('bad',!!stale);$('connection').textContent=connectionFailed?'Disconnected':stale?'Telemetry stale':'Connected · '+ago(state.generated);$('updated').textContent='Telemetry '+time(state.generated)+' · training log '+ago(state.base.logged_at);}
function overview(){const b=state.base,l=b.last,p=state.pipeline;
  $('phase').textContent=state.phase;
  $('phase-summary').textContent=b.status==='training'?'Learning from five task families. Final evaluation and concept diagnostics follow the base run.':b.status==='complete'?'The base checkpoint is saved. Follow the pipeline below for evaluation and continual-learning results.':b.status==='interrupted'?'Training records exist, but the v14 training process is not currently running.':'Follow the latest recorded state of the experiment below.';
  set('run-status',esc(b.status==='training'?'Training active':b.status));$('run-status').className='pill '+b.status;
  $('seed-label').textContent='Seed '+(b.config.seed??'—');$('param-label').textContent=num(b.config.params/1e6,2)+'M parameters';
  $('progress-label').textContent=pct(b.progress);$('progress-bar').style.width=(b.progress*100)+'%';$('hours-label').textContent=num(l.h,2)+'h logged';
  $('eta-label').textContent=b.eta?'Estimated base finish '+time(b.eta)+' · excludes post-training work':b.done?'Base training complete':'Finish estimate unavailable until training is active';
  document.querySelector('.progress-labels span:last-child').textContent=num(b.config.hours||11)+'h target';
  set('overview-metrics',metric('TRAINING STEP',num(l.step),'','Latest log '+ago(b.logged_at))+metric('TOKENS PROCESSED',num(l.tokens/1e6,2),'M','Cumulative training tokens')+metric('RECENT THROUGHPUT',num(b.throughput),'tok/s','Median of last 20 intervals after warmup')+metric('MODEL MEMORY',num(l.mem_gb,2),'GB','MPS driver allocation · latest logged batch'));
  const maxLoss=Math.max(1,...Object.values(b.families).map(f=>f.median_ntp||0));
  set('family-summary',Object.entries(b.families).map(([f,r])=>`<div class="family-row"><span>${names[f]}</span><div class="bar"><i style="width:${(r.median_ntp||0)/maxLoss*100}%;background:${colors[f]}"></i></div><b>${num(r.median_ntp,3)}</b></div>`).join(''));
  const urgent=state.alerts.filter(a=>a.level!=='info').length;set('alert-count',urgent?urgent+' to review':'No active warnings');
  set('alerts',state.alerts.length?state.alerts.map(a=>`<div class="alert ${esc(a.level)}"><span class="alert-icon">${a.level==='critical'?'!':a.level==='warning'?'!':'i'}</span><div><b>${esc(a.title)}</b><p>${esc(a.detail)}</p></div></div>`).join(''):'<div class="alert"><span class="alert-icon">✓</span><div><b>No active warnings</b><p>The collected health checks have not flagged an issue. This is operational status, not a quality verdict.</p></div></div>');
  $('pipeline-status').textContent=p.finished?'Pipeline ended':p.armed?'Supervisor active':'Supervisor not observed';
  const stages=[{id:'base',name:'Base training',state:b.done?'complete':['training','stale'].includes(b.status)?'running':b.status==='interrupted'?'interrupted':'pending',start:b.started_at},...p.stages];
  set('pipeline',stages.map((s,i)=>`<a class="stage ${s.state}" href="${s.id==='base'?'#training':'#evidence'}" ${s.id!=='base'?`data-log="${esc(s.id)}"`:''}><div class="stage-top"><span>${String(i+1).padStart(2,'0')}</span><span>${s.state==='complete'?'✓':s.state==='running'?'●':s.state==='failed'?'!':'○'}</span></div><b>${esc(s.name)}</b><small>${esc(s.state==='pending'?'Queued':s.state==='running'?'In progress':s.state)}${s.exit_code!==null&&s.exit_code!==undefined&&s.exit_code!==0?' · exit '+s.exit_code:''}</small></a>`).join(''));
  chart('overview-pace',[series(b.series,'tok_s',colors.grid,'Tokens/s')],{zero:true});
}
function training(){const b=state.base,l=b.last;const filter=rows=>rows.filter(r=>windowSize==='all'||r.step>=l.step-Number(windowSize));
  const fams=family==='all'?Object.keys(names):[family];
  chart('ntp-chart',fams.map(f=>series(filter(b.families[f].series),'ntp',colors[f],names[f])));
  chart('ncp-chart',fams.map(f=>series(filter(b.families[f].series),'ncp',colors[f],names[f])));
  set('family-legend',fams.map(f=>`<span style="--color:${colors[f]}">${names[f]}</span>`).join(''));
  set('training-metrics',metric('TIME PER STEP',num(b.step_seconds,2),'s','Recent median, including overhead')+metric('AVERAGE THROUGHPUT',num(b.average_tok_s),'tok/s','Total tokens / logged training time')+metric('DATA WAIT',num(l.wait_s,3),'s/step','Latest interval waiting for batches')+metric('SKIPPED UPDATES',num(l.skipped),'','Non-finite-gradient updates excluded'));
  chart('pace-chart',[series(filter(b.series),'tok_s',colors.grid,'Tokens/s')],{zero:true});
  chart('gradient-chart',[series(filter(b.series),'gn',colors.prog,'Gradient norm')],{zero:true});
  set('optimizer-details',kv([['Learning rate',finite(l.lr)?l.lr.toExponential(2):'—'],['NCP weight λ',num(l.lam,3)],['Gradient cap λ scale',num(l.lam_scale,3)],['Records retained',num(b.records)]]));
  const total=Object.values(b.family_tokens).reduce((a,v)=>a+v,0);
  set('data-mix',total?Object.keys(names).map(f=>`<div class="family-row"><span>${names[f]}</span><div class="bar"><i style="background:${colors[f]};width:${(b.family_tokens[f]||0)/total*100}%"></i></div><b>${pct((b.family_tokens[f]||0)/total)}</b></div>`).join(''):empty('Token counters pending','Saved with the periodic training checkpoint.'));
  $('data-age').textContent='Counters saved '+ago(b.family_tokens_at)+'. These can lag the latest training log by about 30 minutes.';
  set('checkpoints',Array.from({length:10},(_,i)=>{const n=(i+1)*10,found=b.snapshots.find(s=>s.percent===n);return `<div class="checkpoint ${found?'saved':''}" title="${found?esc(time(found.time)):'Not saved yet'}"><b>${n}%</b>${found?'Saved':'Pending'}</div>`;}).join(''));
  $('checkpoint-age').textContent='Resume checkpoint '+ago(b.checkpoint_at)+'. Experiment budget ends '+time(b.deadline)+'.';
}
function concept(){const b=state.base,l=b.last;
  set('concept-metrics',metric('NEXT-CONCEPT LOSS',num(l.ncp,3),'','Latest batch · weighted horizon CE')+metric('IDENTICAL HORIZONS',pct(l.same4),'','Same predicted tuple at all 4 horizons')+metric('CODE RESETS',num(l.resets),'','Cumulative dead-code replacements')+metric('TARGET MOMENTUM',num(l.mu,5),'','EMA update momentum · approaches 1'));
  $('horizon-caption').textContent='Latest batch: '+(names[l.fam]||'—')+' · step '+num(l.step)+' · segment accuracy';
  set('horizons',(l.hor||[]).map((h,i)=>`<tr><td>+${i+1} groups</td><td>${pct(h[1])}</td><td>${pct(h[2])}</td><td>${pct(h[3])}</td><td>${pct(h[4])}</td><td>${num(h[0])}</td><td>${num(h[5])}</td><td>${num(h[6],3)}</td></tr>`).join(''));
  set('codebook',(l.ppl_ema||[]).map((v,i)=>`<div class="code-row"><div><span>Segment ${i+1} <small>· ${i<2?'local a':'global b'}</small></span><b>${num(v,1)} <small>/ 32</small></b></div><div class="bar"><i style="width:${Math.min(v/32*100,100)}%;background:${i<2?colors.grid:colors.prog}"></i></div><small>Current batch: ${num(l.ppl_batch?.[i],1)} effective codes</small></div>`).join('')||empty('Waiting for code metrics','These are emitted with the training log.'));
  const cap=b.cap?.cap,ratio=cap?.ntp?.[0]>0?cap.ncp[0]/cap.ntp[0]:null;
  set('target-details',kv([['Schedule state',!finite(l.frac)?'Pending':l.frac>=(b.config.t_freeze??.75)?'Freeze threshold reached':'Targets adapting'],['Freeze point',pct(b.config.t_freeze??.75,0)+' of training'],['Residue-map fit flags',l.p_ok?l.p_ok.filter(Boolean).length+' / '+l.p_ok.length+' ready':'—'],['NCP / NTP gradient share',num(ratio,3)],['Gradient share measured at','Step '+num(b.cap?.step)],['Source mix β',(l.beta||[]).map(v=>pct(v,1)).join(' / ')||'—']]));
  chart('same-chart',[series(b.series,'same4',colors.prog,'Identical horizons')],{zero:true,max:1,percent:true});
  chart('sig-chart',[series(b.series,'sig_a',colors.grid,'Local a'),series(b.series,'sig_b',colors.prog,'Global b'),series(b.series,'sig_bu',colors.state,'Global b uncentered')],{zero:true});
}
const criteria=[
 ['V1','Does prediction beat the easy baselines?','Global head accuracy exceeds both copy-current and the family marginal by ≥10 percentage points in at least 3 of 4 reasoning families.'],
 ['V2','Does the target need the concept path?','The concept-output probe beats the local-only probe by ≥2 percentage points on the global half in at least 2 families.'],
 ['V3','Does the concept module compute?','The concept-output probe beats its input probe by ≥5 percentage points on the global half. Reported per family; no aggregate pass rule.'],
 ['V4','Did the barrier prevent slow-feature collapse?','Normalized global-target χ is at least 0.5 in every family at the final checkpoint. Interpret together with V1 and action sensitivity.'],
 ['V5','Are the concept codes more stable?','Code reassignment from the 10% to 50% snapshots is lower than V13. The 30% to 70% comparison is reported separately.'],
 ['V6','What does the redesign cost?','Text NLL within 0.02 nats of V13 at matched training time; reasoning reported against V13 and baseline. Target redesign and model shrink are a joint effect.'],
 ['V7','Can it learn without forgetting?','Compare the continual-learning matrix, acquisition and retention using the registered §9.4 table and dataset thresholds.']
];
function stageState(id){return state.pipeline.stages.find(s=>s.id===id)?.state;}
function evaluation(){const a=state.artifacts,ev=a.evaluation,dg=a.diagnostics,cl=a.continual;
  set('evaluation-metrics',metric('REASONING SCORE',num(ev?.reasoning,4),'',ev?'Saved base-evaluation result':'Pending base evaluation')+metric('TEXT NLL',num(ev?.text_nll,4),'nats',ev?'Held-out text · lower is better':'Pending base evaluation')+metric('DIAGNOSTIC CHECKPOINTS',num(dg?Object.keys(dg.per_ckpt||{}).length:null),'',dg?.meta?.finished?'Diagnostic file marked finished':'Awaiting or collecting diagnostics')+metric('TARGET ALIGNMENT',dg?.guard?.passes===true?'Pass':dg?.guard?.passes===false?'Fail':'Pending','','Twin / online agreement ≥0.8 per half'));
  set('verdicts',criteria.map(([id,title,desc])=>{const v=dg?.V?.[id];let label='Pending',cls='';
    if(v?.passes===true){label='Pass';cls='pass';}else if(v?.passes===false){label='Fail';cls='fail';}else if(v){label='Reported';cls='partial';}
    if(id==='V6'&&ev){label='Review needed';cls='partial';}if(id==='V7'&&cl){label='Reported';cls='partial';}
    const details=v||(id==='V6'&&ev?{reasoning:ev.reasoning,text_nll:ev.text_nll,elapsed_h:ev.elapsed_h,note:'No automatic V6 verdict: a matched-time V13 comparison is required.'}:id==='V7'&&cl?Object.fromEntries(Object.entries(cl).filter(([k])=>k!=='paired').map(([k,v])=>[k,v.metrics||'Incomplete'])):null);
    return `<div class="verdict"><div class="verdict-id">${id}</div><div><h3>${title}</h3><p>${desc}</p>${details?`<details><summary>Inspect recorded evidence</summary><pre>${esc(JSON.stringify(details,null,2))}</pre></details>`:''}</div>${pill(label,cls)}</div>`;}).join(''));
  $('base-result-status').textContent=ev?(stageState('final_eval')==='complete'?'Stage complete':'Partial / stage not confirmed complete'):'Pending';
  if(ev){const fs=Object.entries(ev.family_scores||{});set('base-results',`<div class="table-scroll"><table><thead><tr><th>Family</th><th>ID area</th><th>Challenge area</th><th>Combined score</th></tr></thead><tbody>${fs.map(([f,v])=>`<tr><td>${esc(names[f]||f)}</td><td>${num(v.id_area,4)}</td><td>${num(v.challenge_area,4)}</td><td>${num(v.score,4)}</td></tr>`).join('')}</tbody></table></div>`+jsonDetails('Task ladders, challenge panels and interventions',ev));}else set('base-results',empty('Evaluation has not produced results yet','Base training must finish before the official evaluation stage begins.'));
  set('continual-progress',['seq','v14','replay'].map(m=>{const rows=state.continual[m]||[],phases=new Set(rows.map(r=>r.phase));return `<div class="continual-method"><b>${m==='seq'?'Sequential LoRA':m==='v14'?'V14 continual':'Replay'}</b><div class="phase-dots">${Array.from({length:6},(_,i)=>`<span class="${phases.has('S'+(i+1))?'done':''}">S${i+1}</span>`).join('')}</div><small>${phases.size}/6 phase records saved</small></div>`;}).join(''));
  if(cl){set('continual-results',`<div class="table-scroll"><table><thead><tr><th>Method</th><th>Final average</th><th>Backward transfer</th><th>Forgetting ↓</th><th>S0 before return</th></tr></thead><tbody>${Object.entries(cl).filter(([k])=>k!=='paired').map(([m,v])=>`<tr><td>${esc(m)}</td><td>${num(v.metrics?.final_average,4)}</td><td>${num(v.metrics?.backward_transfer,4)}</td><td>${num(v.metrics?.forgetting,4)}</td><td>${num(v.metrics?.retention_before_return,4)}</td></tr>`).join('')}</tbody></table></div><p class="footnote">${stageState('cl_eval')==='complete'?'Continual evaluation stage complete.':'Partial results: the continual evaluation stage is not confirmed complete.'} Missing metrics remain blank until all required phases exist.</p>`+jsonDetails('Full accuracy matrices and paired comparisons',cl));}else set('continual-results',empty('Continual results are pending','The three methods are evaluated after all six training phases.'));
  set('diagnostic-details',dg?jsonDetails('Concept diagnostics · '+(dg.meta?.finished?'finished':'in progress'),dg):empty('No diagnostic artifact yet','V1–V5 will populate from diagnostics_v14.json.'));
  if(a.era)$('diagnostic-details').insertAdjacentHTML('beforeend',jsonDetails('Target alignment / era refit',a.era));
  if(a.choice)$('diagnostic-details').insertAdjacentHTML('beforeend',jsonDetails('Continual method selection',a.choice));
}
function system(){const m=state.machine,b=state.base;
  set('system-metrics',metric('GPU UTILIZATION',num(m.gpu_pct),'%',m.gpu_pct===null?'Host counter unavailable':'Host-wide · latest sample')+metric('FREE MEMORY',num(m.memory_free_pct),'%','macOS memory-pressure free percentage')+metric('SWAP IN USE',num(m.swap_gb,2),'GB','Host-wide virtual-memory swap')+metric('DISK AVAILABLE',num(m.disk_free_gb,1),'GB','Volume holding the experiment'));
  set('machine-details',kv([['Power source',m.power||'Unavailable'],['Battery charge',finite(m.battery_pct)?num(m.battery_pct)+'%':'Unavailable'],['Idle sleep prevented',m.sleep_prevented===true?'Yes':m.sleep_prevented===false?'No':'Unavailable'],['Load average · 1 minute',num(m.load1,2)],['Peak model allocation',num(b.peak_memory_gb,2)+' GB'],['Host sample',ago(m.time)],['Collector cadence',state.interval+' seconds'],['Evidence source','Read-only logs and OS counters']]));
  const link=state.connection.lan;$('lan-url').textContent=link||'Network address unavailable';$('lan-url').href=link||'#';$('copy-link').disabled=!link;$('local-url').textContent=state.connection.local;$('local-url').href=state.connection.local;
  const hist=state.history||[];
  chart('gpu-chart',[{label:'GPU utilization',color:colors.grid,points:hist.map(r=>({x:r.time,y:finite(r.gpu_pct)?r.gpu_pct/100:null}))}],{clock:true,zero:true,max:1,percent:true});
  chart('swap-chart',[{label:'Swap GB',color:colors.prog,points:hist.map(r=>({x:r.time,y:r.swap_gb}))}],{clock:true,zero:true});
  set('processes',state.processes.length?state.processes.map(p=>`<tr><td>${esc(p.role)}</td><td>${p.pid}</td><td>${num(p.cpu,1)}%</td><td>${num(p.memory_mb)} MB</td><td>${esc(p.elapsed)}</td></tr>`).join(''):'<tr><td colspan="5">No matching experiment processes observed.</td></tr>');
}
async function refreshLog(){const name=$('log-select').value,generation=++logGeneration;
  try{const res=await fetch('/api/log?name='+encodeURIComponent(name),{signal:AbortSignal.timeout(6000)});if(!res.ok)throw Error('Log request failed');const log=await res.json();if(generation!==logGeneration)return;
    const box=$('log-output'),atEnd=box.scrollHeight-box.scrollTop-box.clientHeight<60,first=box.dataset.source!==name;
    box.textContent=log.text||'No output yet. This stage may still be queued.';box.dataset.source=name;$('log-meta').textContent=log.path+' · '+ago(log.updated)+' · latest 48 KB';if(first||atEnd)box.scrollTop=box.scrollHeight;
  }catch(e){if(generation===logGeneration)$('log-meta').textContent='Log could not refresh. The last displayed output is retained.';}
}
function evidence(){set('artifacts',state.files.map(f=>`<div class="artifact"><div><b>${esc(f.name)}</b><small>${f.available?'Updated '+esc(time(f.updated)):'Waiting for this artifact'}</small></div>${f.available?`<a class="text-link" href="/api/artifact?id=${encodeURIComponent(f.id)}" download>Download ↓</a>`:'<span class="unavailable">Pending</span>'}</div>`).join(''));refreshLog();}
function render(){if(!state)return;const open=new Set([...document.querySelectorAll('[data-page="'+view+'"] details[open]')].map(d=>d.querySelector('summary')?.textContent));chartStore.clear();({overview,training,concept,evaluation,system,evidence}[view])();document.querySelectorAll('[data-page="'+view+'"] details').forEach(d=>{if(open.has(d.querySelector('summary')?.textContent))d.open=true;});updateFreshness();}
function navigate(){view=titles[location.hash.slice(1)]?location.hash.slice(1):'overview';document.querySelectorAll('[data-page]').forEach(p=>p.classList.toggle('hidden',p.dataset.page!==view));document.querySelectorAll('[data-view]').forEach(a=>{a.classList.toggle('active',a.dataset.view===view);if(a.dataset.view===view)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});$('page-title').textContent=titles[view][0];$('page-description').textContent=titles[view][1];render();}
async function refresh(){if(refreshing)return;refreshing=true;$('refresh').disabled=true;
  try{const res=await fetch('/api/state',{signal:AbortSignal.timeout(8000)});if(!res.ok)throw Error(res.status===503?'Collector is starting; retrying automatically.':'Telemetry request failed.');const next=await res.json();state=next;connectionFailed=false;
    $('loading').classList.add('hidden');$('dashboard').classList.remove('hidden');$('service-notice').classList.toggle('hidden',!state.collector_error);$('service-notice').textContent=state.collector_error?'The collector encountered an error. Last successful readings are retained: '+state.collector_error:'';render();
  }catch(e){connectionFailed=true;$('service-notice').classList.remove('hidden');$('service-notice').textContent=state?'Connection lost. Showing the last successful readings; retrying automatically.':e.message;$('connection').textContent='Disconnected';$('connection-dot').classList.add('bad');}
  finally{refreshing=false;$('refresh').disabled=false;}
}
$('refresh').addEventListener('click',refresh);
$('family-filter').addEventListener('click',e=>{const button=e.target.closest('[data-family]');if(!button)return;family=button.dataset.family;document.querySelectorAll('[data-family]').forEach(b=>b.classList.toggle('selected',b===button));if(state)training();});
$('window').addEventListener('change',e=>{windowSize=e.target.value;if(state)training();});
$('log-select').addEventListener('change',refreshLog);
document.addEventListener('click',e=>{const link=e.target.closest('[data-log]');if(link){$('log-select').value=link.dataset.log;if(view==='evidence')refreshLog();}});
$('copy-link').addEventListener('click',async()=>{const value=state?.connection.lan;if(!value)return;let ok=false;try{if(navigator.clipboard){await navigator.clipboard.writeText(value);ok=true;}else{const input=document.createElement('textarea');input.value=value;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();ok=document.execCommand('copy');input.remove();}}catch(_){}$('copy-link').textContent=ok?'Copied':'Select link';setTimeout(()=>$('copy-link').textContent='Copy',2000);});
window.addEventListener('hashchange',navigate);
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{for(const [id,{series,opts}]of chartStore)chart(id,series,opts);},120);});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
navigate();refresh();setInterval(()=>{if(!document.hidden)refresh();},10000);setInterval(updateFreshness,5000);
