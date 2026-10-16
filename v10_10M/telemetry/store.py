"""Incremental, durable ingestion. This module never imports torch or checkpoints."""
import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time
from . import SCHEMA_VERSION


def strict_json(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    def number(value):
        result=float(value)
        if not math.isfinite(result):raise ValueError('Nonfinite JSON number: '+value)
        return result
    return json.loads(text, parse_constant=invalid, parse_float=number)


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten_numeric(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_numeric(child, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten_numeric(child, f'{prefix}.{index}'))
    elif finite_number(value):
        result[prefix] = float(value)
    return result


def training_metrics(payload):
    if not isinstance(payload.get('loss',{}),dict) or not isinstance(payload.get('family_supervised_tokens',{}),dict):
        raise ValueError('Training loss and family token counters must be objects')
    result = {f'train.loss.{k}': v for k, v in payload.get('loss', {}).items() if finite_number(v)}
    mapping = {
        'dense_grad_norm': 'train.gradient.dense_preclip_norm',
        'tokens_per_second': 'train.tokens_per_second', 'tokens': 'train.tokens',
        'learning_rate': 'train.learning_rate', 'elapsed_hours': 'train.elapsed_hours',
        'remaining_hours': 'train.remaining_hours', 'rss_gb': 'train.memory.rss_gib',
        'system_available_gb': 'train.memory.system_available_gib',
        'mps_allocated_gb': 'train.memory.mps_allocated_gib',
        'mps_driver_gb': 'train.memory.mps_driver_gib',
    }
    for old, new in mapping.items():
        if finite_number(payload.get(old)):
            result[new] = payload[old]
    for family, count in payload.get('family_supervised_tokens', {}).items():
        if finite_number(count):
            result[f'train.supervised_tokens.{family}'] = count
    # Both are explicitly derived from the trainer's documented clipping/LR rule.
    if finite_number(payload.get('dense_grad_norm')):
        result['derived.dense_clip_scale'] = min(1., 1./(payload['dense_grad_norm']+1e-6))
    if finite_number(payload.get('learning_rate')):
        result['derived.engram_learning_rate'] = 5*payload['learning_rate']
    return result


def evaluation_metrics(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('families',{}),dict) or not isinstance(payload.get('generation',{}),dict):
        raise ValueError('Evaluation must have object-valued families and generation')
    result = flatten_numeric(payload.get('families', {}), 'eval')
    for family, entry in payload.get('families', {}).items():
        if not isinstance(entry,dict):raise ValueError('Family statistics must be an object')
        counts = entry.get('code_counts', [])
        if not isinstance(counts,list) or any(not finite_number(c) or c<0 for c in counts):raise ValueError('Invalid code counts')
        total = sum(counts)
        if total > 0:
            entropy = -sum((c/total)*math.log(c/total) for c in counts if c)
            result[f'eval.{family}.code_entropy_nats'] = entropy
            result[f'eval.{family}.effective_codes'] = math.exp(entropy)
            result[f'eval.{family}.active_codes'] = sum(c > 0 for c in counts)
    for family, entry in payload.get('generation', {}).items():
        if not isinstance(entry,dict):raise ValueError('Generation statistics must be an object')
        n, correct = entry.get('n', 0), entry.get('correct', 0)
        if not isinstance(n,int) or not isinstance(correct,int) or not 0<=correct<=n:raise ValueError('Invalid generation trial counts')
        result[f'eval.{family}.generation_count'] = n
        result[f'eval.{family}.generation_correct'] = correct
        if n:
            result[f'eval.{family}.exact_accuracy'] = correct/n
            # Wilson interval: useful reminder that 2 Sudoku examples are not a benchmark.
            z = 1.959963984540054
            denom = 1+z*z/n
            center = (correct/n+z*z/(2*n))/denom
            radius = z*math.sqrt(correct/n*(1-correct/n)/n+z*z/(4*n*n))/denom
            result[f'eval.{family}.accuracy_ci95_low'] = max(0., center-radius)
            result[f'eval.{family}.accuracy_ci95_high'] = min(1., center+radius)
    return result


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, manifest TEXT NOT NULL,
                    status TEXT NOT NULL, health TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (
                    run_id TEXT, path TEXT, identity TEXT, offset INTEGER NOT NULL,
                    PRIMARY KEY(run_id,path));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    source TEXT NOT NULL, record_key TEXT NOT NULL, step INTEGER,
                    wall_time REAL NOT NULL, time_quality TEXT NOT NULL,
                    ingested_at REAL NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(run_id,kind,source,record_key));
                CREATE TABLE IF NOT EXISTS metrics (
                    event_id INTEGER NOT NULL, run_id TEXT NOT NULL, name TEXT NOT NULL,
                    value REAL NOT NULL, step INTEGER, wall_time REAL NOT NULL,
                    PRIMARY KEY(event_id,name));
                CREATE INDEX IF NOT EXISTS metric_series ON metrics(run_id,name,event_id);
                CREATE INDEX IF NOT EXISTS event_stream ON events(run_id,kind,id);
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL,
                    size INTEGER, mtime REAL, digest TEXT, payload TEXT,
                    PRIMARY KEY(run_id,path));
                CREATE TABLE IF NOT EXISTS issues (
                    run_id TEXT NOT NULL, source TEXT NOT NULL, detail TEXT NOT NULL,
                    seen REAL NOT NULL, PRIMARY KEY(run_id,source,detail));
            ''')
            old = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if old and int(old[0]) != SCHEMA_VERSION:
                raise ValueError('Incompatible telemetry database schema')
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('schema_version', str(SCHEMA_VERSION)))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def issue(self, db, run_id, source, detail):
        db.execute('INSERT OR REPLACE INTO issues VALUES (?,?,?,?)', (run_id, source, detail, time.time()))

    def event(self, db, run_id, kind, source, record_key, payload, step, wall_time, quality, metrics):
        cur = db.execute('''INSERT OR IGNORE INTO events
            (run_id,kind,source,record_key,step,wall_time,time_quality,ingested_at,payload)
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (run_id,kind,source,record_key,step,wall_time,quality,time.time(),encode(payload)))
        if not cur.rowcount:
            return None
        event_id = cur.lastrowid
        db.executemany('INSERT INTO metrics VALUES (?,?,?,?,?,?)',
                       [(event_id,run_id,name,float(value),step,wall_time)
                        for name,value in metrics.items() if finite_number(value)])
        return event_id

    def ingest_training(self, db, run_id, run_path, manifest):
        path = run_path/'train.jsonl'
        if not path.exists():
            return
        stat = path.stat()
        with path.open('rb') as f:
            first_line = f.readline()
            # Do not establish an identity from a partial first line.
            if not first_line.endswith(b'\n'):
                return
            identity = f'{stat.st_dev}:{stat.st_ino}:'+hashlib.sha256(first_line).hexdigest()
            source = db.execute('SELECT * FROM sources WHERE run_id=? AND path=?', (run_id,path.name)).fetchone()
            same_source = source and (source['identity']==identity or source['identity'].startswith(identity+':truncated:'))
            offset = source['offset'] if same_source else 0
            if same_source:
                identity = source['identity']
            if stat.st_size < offset:
                # Same inode truncated. Preserve the old records; new generation key.
                self.issue(db,run_id,path.name,'Log truncation observed; old history retained')
                identity = identity.split(':truncated:')[0]+f':truncated:{stat.st_mtime_ns}'
                offset = 0
            f.seek(offset)
            while True:
                start = f.tell()
                line = f.readline()
                if not line or not line.endswith(b'\n'):
                    break
                try:
                    payload = strict_json(line)
                    if not isinstance(payload,dict) or not isinstance(payload.get('step'),int):
                        raise ValueError('Training record must contain integer step')
                    elapsed = payload.get('elapsed_hours')
                    if not finite_number(elapsed):
                        raise ValueError('Missing/nonfinite logged elapsed_hours')
                    wall_time = manifest['start_unix']+elapsed*3600
                    # Include line hash: corrections/rollback at the same step are not lost.
                    key = f'{identity}:{start}:'+hashlib.sha256(line).hexdigest()
                    self.event(db,run_id,'training',path.name,key,payload,payload['step'],wall_time,
                               'derived_from_logged_elapsed',training_metrics(payload))
                except (ValueError, UnicodeDecodeError, KeyError, TypeError) as exc:
                    self.issue(db,run_id,path.name,f'Byte {start}: {type(exc).__name__}: {exc}')
                offset = f.tell()
            db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?)', (run_id,path.name,identity,offset))

    def ingest_evaluations(self, db, run_id, run_path):
        for path in sorted(run_path.glob('evaluation_*.json')):
            stat = path.stat()
            known = db.execute('SELECT size,mtime FROM artifacts WHERE run_id=? AND path=?', (run_id,path.name)).fetchone()
            if known and known['size']==stat.st_size and known['mtime']==stat.st_mtime:
                continue
            try:
                raw = path.read_bytes()
                payload = strict_json(raw)
                checksum = hashlib.sha256(raw).hexdigest()
                step_text = path.stem.removeprefix('evaluation_')
                step = 0 if step_text=='initial' else int(step_text)
                self.event(db,run_id,'evaluation',path.name,checksum,payload,step,stat.st_mtime,
                           'file_completion_mtime',evaluation_metrics(payload))
                db.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?)',
                           (run_id,path.name,'evaluation',stat.st_size,stat.st_mtime,checksum,encode(payload)))
            except (ValueError, KeyError, TypeError) as exc:
                self.issue(db,run_id,path.name,f'Cannot ingest evaluation: {exc}')

    def ingest(self, run_path, health, host_sample=None):
        run_path = Path(run_path).resolve()
        run_id = run_path.name
        manifest = strict_json((run_path/'manifest.json').read_bytes())
        status_path = run_path/'status.json'
        status = strict_json(status_path.read_bytes()) if status_path.exists() else {'status':'unknown'}
        with self.connect() as db:
            existing = db.execute('SELECT path FROM runs WHERE id=?', (run_id,)).fetchone()
            if existing and existing['path']!=str(run_path):
                raise ValueError('Run name already refers to another path')
            db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)',
                       (run_id,str(run_path),encode(manifest),encode(status),encode(health),time.time()))
            self.ingest_training(db,run_id,run_path,manifest)
            self.ingest_evaluations(db,run_id,run_path)
            # Explicit audit artifacts have distinct kinds and metric namespaces.
            # They never masquerade as samples produced by the trainer.
            for filename,kind in [('checkpoint_probe.json','probe'),('findings.json','audit'),('balanced_evaluation.json','probe_evaluation')]:
                path=run_path.parent.parent/'audit'/filename
                if not path.exists():continue
                raw=path.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
                payload=strict_json(raw)
                if payload.get('run_id')!=run_id:continue
                if payload.get('source_sha256') and payload['source_sha256']!=manifest.get('source_sha256'):continue
                metrics={}
                if kind=='probe':
                    for family,stats in payload.get('families',{}).items():
                        for key,value in stats.items():
                            if key in ('gradient_probes','chosen','causality','code_counts'):continue
                            if finite_number(value):metrics[f'probe.{family}.{key}']=value
                        metrics.update(flatten_numeric(stats.get('gradient_probes',{}),f'probe.{family}.gradient'))
                elif kind=='probe_evaluation':
                    metrics={name.replace('eval.','probe_eval.',1):value for name,value in evaluation_metrics(payload).items()}
                self.event(db,run_id,kind,str(path),checksum,payload,payload.get('step'),path.stat().st_mtime,
                           'offline_audit_completion_mtime',metrics)
            checkpoint = run_path/'latest.pt'
            if checkpoint.exists():
                stat = checkpoint.stat()
                # Metadata only: never deserialize a 292MB checkpoint in the polling loop.
                db.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?)',
                           (run_id,checkpoint.name,'checkpoint',stat.st_size,stat.st_mtime,None,None))
            if host_sample:
                now = time.time()
                self.event(db,run_id,'host','collector',str(time.time_ns()),host_sample,status.get('step'),now,
                           'collector_clock',flatten_numeric(host_sample,'host'))
        return run_id

    def runs(self):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM runs ORDER BY id').fetchall()
        return [dict(id=r['id'],path=r['path'],manifest=strict_json(r['manifest']),
                     status=strict_json(r['status']),health=strict_json(r['health']),updated=r['updated']) for r in rows]

    def events(self, run_id, kind=None, after_id=0, limit=1000):
        sql = 'SELECT * FROM events WHERE run_id=? AND id>?'
        params = [run_id,after_id]
        if kind:
            sql += ' AND kind=?';params.append(kind)
        sql += ' ORDER BY id LIMIT ?';params.append(limit)
        with self.connect() as db:
            rows = db.execute(sql,params).fetchall()
        return [dict(id=r['id'],kind=r['kind'],source=r['source'],step=r['step'],wall_time=r['wall_time'],
                     time_quality=r['time_quality'],ingested_at=r['ingested_at'],payload=strict_json(r['payload'])) for r in rows]

    def series(self, run_id, name, after_id=0, limit=1000):
        with self.connect() as db:
            rows = db.execute('SELECT event_id,step,wall_time,value FROM metrics WHERE run_id=? AND name=? AND event_id>? ORDER BY event_id LIMIT ?',
                              (run_id,name,after_id,limit)).fetchall()
        return [dict(r) for r in rows]

    def catalog(self, run_id):
        with self.connect() as db:
            rows = db.execute('''SELECT name,COUNT(*) samples,MIN(step) first_step,MAX(step) last_step
                                 FROM metrics WHERE run_id=? GROUP BY name ORDER BY name''',(run_id,)).fetchall()
        return [dict(r) for r in rows]

    def latest_evaluation(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM events WHERE run_id=? AND kind='evaluation' ORDER BY step DESC,id DESC LIMIT 1", (run_id,)).fetchone()
        return dict(event_id=row['id'],step=row['step'],wall_time=row['wall_time'],payload=strict_json(row['payload'])) if row else None

    def issues(self, run_id):
        with self.connect() as db:
            rows=db.execute('SELECT source,detail,seen FROM issues WHERE run_id=? ORDER BY seen DESC LIMIT 100',(run_id,)).fetchall()
        return [dict(r) for r in rows]

    def artifacts(self, run_id):
        with self.connect() as db:
            rows=db.execute('SELECT path,kind,size,mtime,digest FROM artifacts WHERE run_id=? ORDER BY path',(run_id,)).fetchall()
        return [dict(r) for r in rows]

    def overview(self, run_id):
        run = next((r for r in self.runs() if r['id']==run_id),None)
        if run is None:
            raise KeyError(run_id)
        evaluation = self.latest_evaluation(run_id)
        alerts = []
        if not run['health'].get('trainer_alive') and run['status'].get('status')=='running':
            alerts.append(dict(severity='error',code='trainer_missing',message='Status says running but the matching trainer process is absent.'))
        if (run['health'].get('status_age_seconds') or 0)>300 and run['status'].get('status')=='running':
            alerts.append(dict(severity='warning',code='stale_status',message='No new status for over five minutes; inspect process/checkpoint health.'))
        if evaluation:
            for family, stats in evaluation['payload'].get('families',{}).items():
                delta=stats.get('concept_ablation_delta_nll')
                if delta is not None and abs(delta)<.005:
                    alerts.append(dict(severity='research',code='small_concept_effect',family=family,
                                       message='Concept ablation changes NLL by less than 0.005 on this small sample; benefit is unproven.',value=delta))
                if stats.get('pooled_coordinate_std',1)<.02:
                    alerts.append(dict(severity='research',code='low_target_spread',family=family,
                                       message='Very low pooled coordinate spread; inspect possible representation collapse.'))
            for family, stats in evaluation['payload'].get('generation',{}).items():
                if stats.get('n',0)>0 and stats.get('correct')==0:
                    alerts.append(dict(severity='research',code='zero_sample_success',family=family,
                                       message=f'Zero complete answers in {stats["n"]} sampled problems; not proof of zero population accuracy.'))
        with self.connect() as db:
            audit=db.execute("SELECT payload FROM events WHERE run_id=? AND kind='audit' ORDER BY id DESC LIMIT 1",(run_id,)).fetchone()
        if audit:
            alerts.extend(strict_json(audit['payload']).get('findings',[]))
        return dict(schema_version=SCHEMA_VERSION,generated_at=time.time(),run=run,latest_evaluation=evaluation,
                    alerts=alerts,ingestion_issues=self.issues(run_id),artifacts=self.artifacts(run_id),
                    metric_availability=METRIC_AVAILABILITY)


METRIC_AVAILABILITY = {
    'train.loss.*': {'available':True,'source':'train.jsonl','cadence':'20 optimizer steps (first at step 1)',
                     'meaning':'Mean of microbatch-normalized losses over the logging interval; not token-weighted across families. Exploration/selector include zero on non-auxiliary updates.'},
    'train.gradient.dense_preclip_norm': {'available':True,'source':'train.jsonl',
        'meaning':'Dense parameters only, L2 norm before clipping, measured on the final update of the logging interval. Excludes sparse Engram table; not an interval average.'},
    'derived.dense_clip_scale': {'available':True,'meaning':'Inferred from clip_grad_norm_(max_norm=1); not separately measured.'},
    'train.memory.*': {'available':True,'meaning':'Logged GiB counters; Apple unified-memory counters overlap and must not be summed.'},
    'eval.*': {'available':True,'cadence':'About 30 minutes; small fixed development samples',
              'meaning':'Per-family NLL, concept ablation, coordinate spread, code counts and strict complete-answer generation; zero-trial accuracy is absent, not zero.'},
    'probe.*': {'available':True,'cadence':'Explicit one-off CPU checkpoint audit',
               'meaning':'Fresh gradient and ablation probes, labeled by stored checkpoint step. Never historical optimizer-step gradients.'},
    'probe_eval.*': {'available':True,'cadence':'Explicit balanced development audit',
                    'meaning':'Independent stratified generation evaluation; separate from the trainer small fixed sample.'},
    'host.*': {'available':True,'cadence':'30 seconds while trainer is active','meaning':'External CPU/RSS/system memory samples, independent from train memory counters.'},
    'per_layer_gradient_norms': {'available':False,'reason':'Not emitted by the running trainer; unavailable historically. Can be measured in explicitly labeled checkpoint probes.'},
    'engram_gradient_norm': {'available':False,'reason':'Trainer checks/clips this norm but does not log it.'},
    'gradient_histograms': {'available':False,'reason':'Not captured by this run.'},
    'gpu_utilization_and_temperature': {'available':False,'reason':'No reliable public cross-process MPS utilization/temperature source is configured.'},
    'rbf_activity': {'available':False,'reason':'RBF bank is intentionally empty during pretraining; no continual-learning episode has been run in this training run.'},
    'checkpoint_step': {'available':False,'reason':'Collector records checkpoint filesystem metadata without deserializing it; only an explicit checkpoint audit can establish its exact stored step.'},
}
