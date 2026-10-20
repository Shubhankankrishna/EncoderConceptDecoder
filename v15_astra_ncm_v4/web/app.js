const $=s=>document.querySelector(s);
const pct=v=>Number.isFinite(v)?`${(100*v).toFixed(1)}%`:'—';
const num=v=>Number.isFinite(v)?v.toFixed(3):'—';
function cell(tag,text){const e=document.createElement(tag);e.textContent=text;return e;}
function card(label,value,detail=''){const e=document.createElement('div');e.className='card';e.append(cell('small',label),cell('strong',value));if(detail)e.append(cell('p',detail));return e;}
function render(data){
 $('#connection').textContent=`Updated ${new Date(data.timestamp*1000).toLocaleTimeString()} · telemetry reads saved results`;
 const state=data.status;const end=data.plan.deadline?new Date(data.plan.deadline*1000).toLocaleString():'Pending';
 const currentState=data.followup?.status||state;
 $('#summary').replaceChildren(card('Current experiment',currentState.status||'Preparing',currentState.active_stage||''),card('Maximum deadline',end,'May finish earlier when the comparison completes'),card('Available memory',`${data.resources.available_gb.toFixed(1)} GB`),card('Original pilot',state.status||'Preparing',data.final_available?'Held-out results below':'Final evaluation pending'));
 const finalBody=$('#final-results');finalBody.replaceChildren();
 for(const result of data.final_summary||[]){const row=document.createElement('tr');for(const value of [result.arm,result.selected_step,pct(result.test),pct(result.ood)])row.append(cell('td',String(value)));finalBody.append(row);}
 const f=data.followup;$('#followup').replaceChildren();$('#followup-samples').replaceChildren();
 if(f){$('#followup-status').textContent=`${f.status.status} ${f.status.active_stage?`· active: ${f.status.active_stage}`:''} · 4,000 additional updates per arm, alternating in 500-update chunks. Same starting checkpoint and original deadline. Actual compute differs.`;
  for(const arm of f.arms){const e=arm.evaluation;$('#followup').append(card(arm.name==='on_policy'?'Predicted-prefix recovery':'Original-training control',`${arm.state.step??0} / 4,000`,`${arm.state.status} · evaluated at +${e?.step??'—'} · free exact ${pct(e?.greedy_path_exact)} · route similarity ${pct(e?.route_similarity)} · ${e?.count??48} validation cases`));
   for(const s of (e?.samples||[]).slice(0,2)){const box=document.createElement('article');box.className='sample';box.append(cell('small',`${arm.name} · +${e.step} updates · ${s.graph_id}`),cell('pre',`Verified: ${s.path.join(' → ')}\nGenerated: ${s.parsed_path.join(' → ')}`),cell('p',s.text),cell('small',`Exact route: ${s.exact?'yes':'no'}`));$('#followup-samples').append(box);}}
  for(const [name,e] of Object.entries(f.final_summary||{}))$('#followup').append(card(`Held-out text · ${name}`,pct(e.greedy_path_exact),`${e.count} test routes · selected +${e.step} updates · similarity ${pct(e.route_similarity)}`));
 }
 const d=data.diagnosis;$('#diagnosis').textContent=d?.count?`Diagnosis on ${d.count} earlier validation cases: ${pct(d.teacher_argmax_path_exact)} exact routes when predicting each level from correct earlier codes, versus ${pct(d.path_exact_by_gold_prefix_levels?.['0'])} in free generation. Recovery training recomputes residual targets after model-generated prefixes. This is a testable correction, not a claimed success.`:'';
 const body=$('#stages');body.replaceChildren();
 for(const s of data.stages){const e=s.evaluation;const r=e?.strategies?.['0'];const tr=document.createElement('tr');if(s.name===state.active_stage)tr.className='active';
  for(const value of [s.name,s.state.status,s.state.step??'—',e?.step??'—',pct(r?.rollouts?.greedy?.success_rate),pct(r?.rollouts?.random?.success_rate),num(r?.discounted_value_mse??e?.code_nll)])tr.append(cell('td',String(value)));body.append(tr);
 }
 const flag=v=>v===true?'Passed':v===false?'Not demonstrated':'Pending';
 $('#gates').replaceChildren(card('Grounded interface',flag(data.gates.grounded_interface_gate),'Both recurrent seeds must beat random and reach the preregistered success threshold.'),card('Recurrence benefit',flag(data.gates.recurrence_benefit_demonstrated),'Separate from the ability to fit training examples.'),card('Language conditioning',flag(data.quality.condition_used),'Correct versus shuffled verified-plan inputs.'),card('Free text path fidelity',flag(data.quality.tiny_free_generation_probe_acceptable),'Small exploratory sample; inspect actual text below.'));
 const renderer=data.stages.find(s=>s.name==='renderer');const ev=renderer?.evaluation;
 if(ev?.samples){$('#samples').replaceChildren();for(const s of ev.samples){const box=document.createElement('article');box.className='sample';box.append(cell('small',`Saved step ${ev.step} · ${new Date(ev.timestamp*1000).toLocaleString()} · temperature ${s.temperature}`),cell('pre',`Verified path: ${s.verified_path.join(' → ')}`),cell('p',s.plain_text),cell('small',`Parsed path matches: ${s.exact_plain_path?'yes':'no'}`));$('#samples').append(box);}}
}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json());}catch(e){$('#connection').textContent='Telemetry unavailable. Showing the last received snapshot, if any.';}}
refresh();setInterval(refresh,5000);
