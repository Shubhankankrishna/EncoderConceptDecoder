// Pure data rules shared by the dashboard and its tests.
export const finite=v=>typeof v==='number'&&Number.isFinite(v);
export const number=(v,d=2)=>finite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
export const integer=v=>number(v,0);
export const compact=v=>finite(v)?new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:1}).format(v):'—';
export const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const duration=s=>!finite(s)?'—':s<=0?'0m':s>=3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`:`${Math.floor(s/60)}m ${Math.floor(s%60)}s`;
export const ist=s=>finite(s)?new Date(s*1000).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit',hour12:false})+' IST':'—';
export const familyName=k=>({text:'Text',math:'Math',sudoku_easy:'Sudoku · easy',sudoku_hard:'Sudoku · hard'}[k]||k);
export const get=(o,path)=>path.split('.').reduce((a,k)=>a?.[k],o);
// Final trainer status moves the last interval metrics into last_metrics.
export const runStatus=run=>({...run?.status?.last_metrics,...run?.status});
export function wilson(correct,n){
 if(!Number.isInteger(n)||!Number.isInteger(correct)||n<=0||correct<0||correct>n)return null;
 const z=1.959963984540054,p=correct/n,den=1+z*z/n,c=(p+z*z/(2*n))/den,r=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;
 return {value:p,low:Math.max(0,c-r),high:Math.min(1,c+r)};
}
export function mergeEvents(old,added){const m=new Map(old.map(e=>[e.id,e]));for(const e of added)m.set(e.id,e);return [...m.values()].sort((a,b)=>a.id-b.id);}
// Backfilled evaluation files may be ingested after newer checkpoints.
// Plot checkpoint measurements in checkpoint order; training keeps ingestion order.
export const checkpointOrder=events=>[...events].sort((a,b)=>(a.step??-1)-(b.step??-1)||a.id-b.id);
export function points(events,kind,path,start=0){let rows=events.filter(e=>e.kind===kind);if(kind==='evaluation')rows=checkpointOrder(rows);return rows.map(e=>({id:e.id,step:e.step,time:e.wall_time,hours:(e.wall_time-start)/3600,value:get(e.payload,path)}));}
export function ema(data,alpha=.15){let last=null,step=null;return data.map(p=>{if(!finite(p.value)||!finite(p.step)){last=null;step=null;return {...p,value:null};}if(step!==null&&p.step<=step)last=null;last=last===null?p.value:alpha*p.value+(1-alpha)*last;step=p.step;return {...p,value:last};});}
export function segments(data){let out=[],seg=[],last=null;for(const p of data){if(!finite(p.value)||!finite(p.step)){if(seg.length)out.push(seg);seg=[];last=null;continue;}if(last!==null&&p.step<=last){if(seg.length)out.push(seg);seg=[];}seg.push(p);last=p.step;}if(seg.length)out.push(seg);return out;}
// Preserve first/last and each bucket's extrema; reduce only the rendered geometry.
export function downsample(data,max=500){if(data.length<=max)return data;const out=[data[0]],size=(data.length-2)/Math.max(1,Math.floor((max-2)/2));for(let a=1;a<data.length-1;a+=size){const bucket=data.slice(Math.floor(a),Math.min(data.length-1,Math.floor(a+size)));if(!bucket.length)continue;let lo=bucket[0],hi=lo;for(const p of bucket){if(p.value<lo.value)lo=p;if(p.value>hi.value)hi=p;}out.push(...(lo===hi?[lo]:[lo,hi].sort((x,y)=>data.indexOf(x)-data.indexOf(y))));}out.push(data.at(-1));return out;}
export function windowPoints(data,range,now){return range==='all'?data:data.filter(p=>finite(p.time)&&p.time>=now-Number(range)*3600);}
export function codeStats(counts){if(!Array.isArray(counts)||counts.some(v=>!finite(v)||v<0))return null;const total=counts.reduce((a,b)=>a+b,0);if(!total)return null;const entropy=-counts.reduce((a,c)=>a+(c?c/total*Math.log(c/total):0),0);return {total,active:counts.filter(c=>c>0).length,effective:Math.exp(entropy)};}
export async function pages(fetchPage,cursor=0){let out=[],next=cursor;for(let i=0;i<1000;i++){const d=await fetchPage(next);const rows=d.events??d.points;if(!Array.isArray(rows)||!Number.isInteger(d.next_after_id)||d.next_after_id<next)throw Error('Invalid API cursor');out.push(...rows);if(d.has_more&&d.next_after_id<=next)throw Error('API pagination made no progress');next=d.next_after_id;if(!d.has_more)return {rows:out,cursor:next};}throw Error('API pagination limit exceeded');}
export function csv(points){return 'event_id,step,unix_seconds,value\n'+points.map(p=>[p.id??p.event_id,p.step,p.time??p.wall_time,p.value].map(v=>finite(v)?String(v):'').join(',')).join('\n')+'\n';}
