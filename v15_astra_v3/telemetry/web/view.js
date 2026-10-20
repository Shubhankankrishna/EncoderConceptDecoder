export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const escapeText = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const formatNumber = (value, digits=0) => finite(value) ? value.toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';
export const formatBytes = value => finite(value) ? value >= 1024**3 ? `${(value/1024**3).toFixed(2)} GiB` : `${(value/1024**2).toFixed(1)} MiB` : '—';
export function duration(seconds) {
  if (!finite(seconds)) return '—';
  const total = Math.max(0,Math.floor(seconds));
  if (total < 60) return `${total}s`;
  if (total < 3600) return `${Math.floor(total/60)}m ${String(total%60).padStart(2,'0')}s`;
  return `${Math.floor(total/3600)}h ${String(Math.floor(total%3600/60)).padStart(2,'0')}m`;
}
export const age = seconds => finite(seconds) ? seconds < 8 ? 'just now' : `${duration(seconds)} ago` : '—';
export function localDate(timestamp, withDate=true) {
  if (!finite(timestamp)) return '—';
  return new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',...(withDate?{day:'numeric',month:'short'}:{}),hour:'2-digit',minute:'2-digit',hour12:true}).format(timestamp*1000)+(withDate?' IST':'');
}
export function smooth(rows, window=7) {
  return rows.map((row,index) => {
    const values=rows.slice(Math.max(0,index-window+1),index+1).map(r=>r.loss).filter(finite);
    return {...row,loss:values.length?values.reduce((a,b)=>a+b,0)/values.length:null};
  });
}
export function domain(values, pad=.09) {
  const clean=values.filter(finite);
  if (!clean.length) return [0,1];
  const lo=Math.min(...clean),hi=Math.max(...clean),span=Math.max(hi-lo,Math.abs(hi)*.05,.1);
  return [lo-span*pad,hi+span*pad];
}
export function polyline(rows,key,x,y) {
  return rows.filter(r=>finite(r.step)&&finite(r[key])).map(r=>`${x(r.step).toFixed(2)},${y(r[key]).toFixed(2)}`).join(' ');
}
export function statusInfo(status) {
  const states={training:['Training in progress',''],final_evaluation:['Final evaluation',''],complete:['Experiment complete',''],
    stopped:['Stopped','neutral'],waiting:['Waiting for telemetry','waiting'],failed:['Run failed','critical'],offline:['Trainer offline','critical'],
    stale:['Metrics delayed','warning'],recovering:['Recovery in progress','warning'],paused_resource:['Paused for resources','warning'],
    budget_finished_evaluation_incomplete:['Budget finished · evaluation incomplete','warning'],preparing_data:['Preparing data','waiting'],
    waiting_turn:['Waiting for model turn','waiting'],codec_acceptance:['Checking native32 codec','waiting'],
    encoding:['Preparing code batches','waiting'],preflight:['Testing MPS and recovery','waiting']};
  return states[status]||[String(status||'Waiting').replaceAll('_',' '),'neutral'];
}
export function lossChart(rows,evaluations,range='all') {
  let data=rows.filter(r=>finite(r.step)&&finite(r.loss));
  if (!data.length) return '';
  const latest=Math.max(...data.map(r=>r.step));
  if (range!=='all') data=data.filter(r=>r.step>=latest-Number(range));
  const left=52,right=876,top=15,bottom=238;
  const steps=data.map(r=>r.step),minX=Math.min(...steps),maxX=Math.max(...steps,minX+1);
  const ev=evaluations.filter(e=>finite(e.step)&&e.step>=minX&&e.step<=maxX&&finite(e.nll_per_code));
  const [minY,maxY]=domain([...data.map(r=>r.loss),...ev.map(e=>e.nll_per_code)]);
  const x=s=>left+(s-minX)/(maxX-minX)*(right-left),y=v=>bottom-(v-minY)/(maxY-minY)*(bottom-top);
  let svg='';
  for(let i=0;i<5;i++) {
    const value=minY+(maxY-minY)*i/4,py=y(value);
    svg+=`<line class="chart-grid" x1="${left}" x2="${right}" y1="${py}" y2="${py}"/><text class="chart-axis" x="40" y="${py+3}" text-anchor="end">${value.toFixed(1)}</text>`;
  }
  for(let i=0;i<5;i++) {
    const step=minX+(maxX-minX)*i/4;
    svg+=`<text class="chart-axis" x="${x(step)}" y="261" text-anchor="middle">${Math.round(step).toLocaleString('en-US')}</text>`;
  }
  svg+=`<text class="plot-caption" x="${right}" y="280" text-anchor="end">TRAINING STEP</text>`;
  svg+=`<polyline class="chart-path" stroke="#bdcfb3" stroke-width="1.2" opacity=".75" points="${polyline(data,'loss',x,y)}"/>`;
  svg+=`<polyline class="chart-path" stroke="#4b7651" stroke-width="2.3" points="${polyline(smooth(data),'loss',x,y)}"/>`;
  if(ev.length>1)svg+=`<polyline class="chart-path" stroke="#c09b5f" stroke-width="1.5" stroke-dasharray="4 5" points="${polyline(ev,'nll_per_code',x,y)}"/>`;
  for(const e of ev)svg+=`<circle cx="${x(e.step)}" cy="${y(e.nll_per_code)}" r="4.3" stroke="white" stroke-width="2" fill="#bd9456"><title>Validation at step ${e.step}: ${e.nll_per_code.toFixed(4)}</title></circle>`;
  return svg;
}
export function sparkline(values,color='#668b54') {
  const clean=values.filter(finite);
  if(clean.length<2)return '';
  const [lo,hi]=domain(clean,.05);
  return `<polyline fill="none" stroke="${color}" stroke-width="1.5" points="${clean.map((v,i)=>`${i/(clean.length-1)*150},${30-(v-lo)/(hi-lo)*28}`).join(' ')}"/>`;
}
