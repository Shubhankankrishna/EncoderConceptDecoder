import {finite,number,integer,esc,ema,segments,downsample,windowPoints} from './core.mjs';
const COLORS=['var(--accent)','var(--teal)','var(--orange)','var(--purple)'];
export class Chart{
 constructor(host,{title='',description='',explanation='',unit='',digits=3,reference=null,domain=null,bare=false,badge='Diagnostic'}={}){
  this.host=host;this.config={title,description,explanation,unit,digits,reference,domain,bare};this.series=[];this.hidden=new Set();this.options={range:'all',axis:'step',smooth:false,now:Date.now()/1000};this.visible=false;this.dirty=true;this.focusIndex=0;
  host.innerHTML=bare?'':`<div class="chart-head"><div><div class="chart-title"><h3>${esc(title)}</h3><details class="explain"><summary aria-label="Explain ${esc(title)}">?</summary><p>${esc(explanation)}</p></details></div><span class="chart-badge">${esc(badge)}</span></div><div class="chart-latest">—<small>${esc(unit)}</small></div></div><p class="chart-desc">${esc(description)}</p>`;
  this.plot=document.createElement('div');this.plot.className='chart-plot';this.legend=document.createElement('div');this.legend.className='chart-legend';this.hint=document.createElement('div');this.hint.className='chart-hint';this.hint.textContent='Hover or use ← → for raw values · drag to compare two points';host.append(this.plot,this.legend,this.hint);
  this.observer=new IntersectionObserver(entries=>{this.visible=entries[0].isIntersecting;if(this.visible&&this.dirty)this.draw();},{rootMargin:'100px'});this.observer.observe(host);
  this.resize=new ResizeObserver(()=>{this.dirty=true;if(this.visible)this.draw();});this.resize.observe(host);
 }
 update(series,options={}){this.series=series.map((s,i)=>({...s,color:s.color||COLORS[i%COLORS.length]}));Object.assign(this.options,options);this.dirty=true;if(this.visible)this.draw();}
 dispose(){this.observer.disconnect();this.resize.disconnect();}
 format(v){return this.config.digits===0?integer(v):number(v,this.config.digits);}
 draw(){
  this.dirty=false;const {axis,range,smooth,now}=this.options,c=this.config;
  const width=Math.max(290,this.plot.clientWidth),height=this.plot.clientHeight||205,L=43,R=10,T=12,B=29,W=width-L-R,H=height-T-B;
  const source=this.series.map(s=>({...s,points:windowPoints(s.points,range,now)}));const enabled=source.filter(s=>!this.hidden.has(s.name));
  const all=enabled.flatMap(s=>s.points).filter(p=>finite(p.value)&&finite(p[axis]));const latest=source[0]?.points.filter(p=>finite(p.value)).at(-1)?.value;
  const value=this.host.querySelector('.chart-latest');if(value)value.innerHTML=`${this.format(latest)}<small>${esc(c.unit)}</small>`;
  this.legend.innerHTML=source.map((s,i)=>`<button data-series="${i}" aria-pressed="${!this.hidden.has(s.name)}"><i class="legend-line" style="--color:${s.color}"></i>${esc(s.name)}${smooth?' · raw + EMA':''}</button>`).join('');
  this.legend.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{const name=source[Number(b.dataset.series)].name;if(this.hidden.has(name))this.hidden.delete(name);else this.hidden.add(name);this.draw();}));
  if(!all.length){this.plot.innerHTML='<div class="chart-empty">No measurements in this window.</div>';return;}
  let xmin=Math.min(...all.map(p=>p[axis])),xmax=Math.max(...all.map(p=>p[axis])),ymin=Math.min(...all.map(p=>p.value)),ymax=Math.max(...all.map(p=>p.value));
  if(finite(c.reference)){ymin=Math.min(ymin,c.reference);ymax=Math.max(ymax,c.reference);}if(xmin===xmax){xmin-=axis==='step'?1:.01;xmax+=axis==='step'?1:.01;}
  const spread=ymax-ymin,pad=spread?spread*.13:Math.max(Math.abs(ymax)*.1,.001);ymin-=pad;ymax+=pad;if(ymin<0&&all.every(p=>p.value>=0)&&c.reference!==0)ymin=0;
  if(c.domain){[ymin,ymax]=c.domain;}
  const x=v=>L+(v-xmin)/(xmax-xmin)*W,y=v=>T+H-(v-ymin)/(ymax-ymin)*H;
  const ticks=Array.from({length:5},(_,i)=>{const v=ymin+(ymax-ymin)*i/4,yy=y(v);return `<line class="chart-gridline" x1="${L}" x2="${width-R}" y1="${yy}" y2="${yy}"/><text class="chart-label" x="${L-7}" y="${yy+3}" text-anchor="end">${Math.abs(v)>=1000?integer(v):number(v,Math.abs(ymax-ymin)<.03?4:Math.abs(ymax-ymin)<2?2:1)}</text>`;}).join('');
  const xticks=Array.from({length:4},(_,i)=>{const v=xmin+(xmax-xmin)*i/3;return `<text class="chart-label" x="${x(v)}" y="${height-8}" text-anchor="${i===0?'start':i===3?'end':'middle'}">${axis==='step'?integer(v):number(v,2)+'h'}</text>`;}).join('');
  let paths='';for(const s of enabled){for(const seg of segments(s.points)){const raw=downsample(seg,500);const path=raw.map((p,i)=>`${i?'L':'M'}${x(p[axis]).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ');paths+=`<path d="${path}" stroke="${s.color}" fill="none" stroke-width="${smooth?1:1.8}" opacity="${smooth?.24:1}"/>`;if(seg.length<=20)paths+=raw.map(p=>`<circle cx="${x(p[axis])}" cy="${y(p.value)}" r="2.4" fill="${s.color}"/>`).join('');if(smooth){const trend=downsample(ema(seg),500);paths+=`<path d="${trend.map((p,i)=>`${i?'L':'M'}${x(p[axis]).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')}" stroke="${s.color}" fill="none" stroke-width="1.8"/>`;}}}
  const ref=finite(c.reference)?`<line x1="${L}" x2="${width-R}" y1="${y(c.reference)}" y2="${y(c.reference)}" stroke="var(--warn)" stroke-dasharray="5 4" opacity=".7"/>`:'';
  this.plot.innerHTML=`<svg role="img" tabindex="0" aria-label="${esc(c.title||'Metric series')} chart. ${all.length} raw points. Arrow keys inspect values." viewBox="0 0 ${width} ${height}">${ticks}${xticks}${ref}${paths}<line class="crosshair" x1="${L}" x2="${L}" y1="${T}" y2="${T+H}" stroke="var(--muted)" stroke-dasharray="2 3" visibility="hidden"/></svg><div class="chart-tip" hidden></div>`;
  const svg=this.plot.querySelector('svg'),tip=this.plot.querySelector('.chart-tip'),cross=this.plot.querySelector('.crosshair');let anchor=null;
  // Hover searches full raw data, never the reduced path or EMA.
  const primary=enabled[0].points.filter(p=>finite(p.value)&&finite(p[axis]));
  const show=p=>{if(!p)return;const xx=x(p[axis]);cross.setAttribute('x1',xx);cross.setAttribute('x2',xx);cross.setAttribute('visibility','visible');const values=enabled.map(s=>{let nearest=s.points.filter(q=>finite(q.value)&&finite(q[axis])).reduce((a,q)=>!a||Math.abs(q[axis]-p[axis])<Math.abs(a[axis]-p[axis])?q:a,null);return nearest?`<div><span style="color:${s.color}">${esc(s.name)}</span> <strong>${this.format(nearest.value)}</strong> <small>step ${integer(nearest.step)}</small></div>`:'';}).join('');tip.innerHTML=`<div>Step ${integer(p.step)} · ${number(p.hours,3)}h · event ${integer(p.id)}</div>${values}${anchor?`<div>From step ${integer(anchor.step)}: Δ <strong>${this.format(p.value-anchor.value)}</strong> (raw)</div>`:''}`;tip.hidden=false;tip.style.left=`${Math.max(5,Math.min(width-215,xx+10))}px`;};
  const atPointer=e=>{const rect=svg.getBoundingClientRect(),vx=(e.clientX-rect.left)/rect.width*width,v=xmin+(vx-L)/W*(xmax-xmin);return primary.reduce((a,p)=>!a||Math.abs(p[axis]-v)<Math.abs(a[axis]-v)?p:a,null);};
  svg.addEventListener('pointermove',e=>show(atPointer(e)));svg.addEventListener('pointerdown',e=>{anchor=atPointer(e);svg.setPointerCapture(e.pointerId);show(anchor);});svg.addEventListener('pointerup',e=>{show(atPointer(e));anchor=null;});svg.addEventListener('pointerleave',()=>{if(!anchor){tip.hidden=true;cross.setAttribute('visibility','hidden');}});
  svg.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End','Escape'].includes(e.key))return;e.preventDefault();if(e.key==='Escape'){tip.hidden=true;return;}this.focusIndex=e.key==='Home'?0:e.key==='End'?primary.length-1:Math.max(0,Math.min(primary.length-1,this.focusIndex+(e.key==='ArrowRight'?1:-1)));show(primary[this.focusIndex]);});
 }
}
