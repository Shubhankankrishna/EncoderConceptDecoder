import {test} from 'node:test';
import assert from 'node:assert/strict';
import {throughputWindow} from './core.mjs';
const event=(step,tokens,seconds,elapsed,source='a')=>({id:step,step,kind:'training',wall_time:elapsed,payload:{input_tokens:tokens,step_seconds:seconds,elapsed_hours:elapsed/3600,source_sha256:source}});
test('throughput weights durations rather than averaging fast/slow rates',()=>{
 const points=throughputWindow([event(1,100,1,1),event(2,100,3,4)]);
 assert.equal(points[1].value,50);assert.notEqual(points[1].value,(100+100/3)/2);
});
test('evaluation gaps count; stopped downtime absent from active elapsed does not',()=>{
 const a=event(1,100,1,1),b=event(2,100,1,12);b.wall_time=120;
 assert.equal(throughputWindow([a,b])[1].value,200/12);
});
test('window eviction and source transitions do not bridge incompatible rates',()=>{
 const es=[event(1,100,1,1),event(2,100,1,2),event(3,200,2,4),event(4,20,1,5,'b')];
 const p=throughputWindow(es,0,2);assert.equal(p[2].value,100);assert.equal(p[3].value,20);
});
test('missing durations are gaps and zero token counts remain real zero',()=>{
 assert.equal(throughputWindow([event(1,0,1,1)])[0].value,0);
 assert.equal(throughputWindow([event(1,10,null,1)])[0].value,null);
});
