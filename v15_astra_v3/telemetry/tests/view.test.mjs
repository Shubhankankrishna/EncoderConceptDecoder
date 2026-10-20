import test from 'node:test';
import assert from 'node:assert/strict';
import {duration,formatBytes,formatNumber,domain,smooth,lossChart,statusInfo,escapeText} from '../web/view.js';

test('formatting handles zero, missing values, finite numbers and time boundaries',()=>{
  assert.equal(duration(0),'0s');assert.equal(duration(-2),'0s');assert.equal(duration(null),'—');
  assert.equal(duration(3661),'1h 01m');assert.equal(duration(61),'1m 01s');
  assert.equal(formatNumber(0),'0');assert.equal(formatNumber(NaN),'—');
  assert.equal(formatBytes(1024**3),'1.00 GiB');assert.equal(formatBytes(null),'—');
});
test('chart domain and smoothing tolerate flat and missing readings',()=>{
  const [lo,hi]=domain([7,7,null,NaN]);assert.ok(lo<7&&hi>7);
  assert.deepEqual(domain([]),[0,1]);
  assert.equal(smooth([{loss:1},{loss:3},{loss:null}],2)[1].loss,2);
});
test('chart is safe with one point and includes validation and all finite endpoints',()=>{
  assert.equal(lossChart([],[]),'');
  const one=lossChart([{step:1,loss:9}],[]);assert.ok(!one.includes('NaN'));assert.ok(!one.includes('Infinity'));
  const svg=lossChart([{step:1,loss:9},{step:5000,loss:7}],[{step:5000,nll_per_code:7.2}]);
  assert.ok(svg.includes('Validation at step 5000'));assert.ok(svg.includes('5,000'));
  assert.ok(!lossChart([{step:1,loss:null}],[]).includes('NaN'));
});
test('terminal states stay explicit and arbitrary strings are escaped',()=>{
  assert.equal(statusInfo('offline')[1],'critical');assert.equal(statusInfo('complete')[0],'Experiment complete');
  assert.equal(escapeText('<img onerror="x">'),'&lt;img onerror=&quot;x&quot;&gt;');
});
