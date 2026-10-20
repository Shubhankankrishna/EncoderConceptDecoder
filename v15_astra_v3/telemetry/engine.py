"""Incremental log ingestion and persistent telemetry, without importing training.

The source run is only ever opened for reading. SQLite, cursors and server state
belong to the telemetry directory. Nothing loads checkpoints or touches MPS.
"""
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

SCHEMA_VERSION = 1
MAX_JSON = 16 * 1024 * 1024
MAX_LINE = 1024 * 1024
TERMINAL = {'complete', 'stopped', 'failed', 'budget_finished_evaluation_incomplete', 'paused_resource',
    'waiting_turn', 'codec_acceptance', 'encoding', 'preflight'}


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def read_json(path, default=None):
    try:
        if path.stat().st_size > MAX_JSON:
            return default
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return default


def generation_stats(text):
    # Descriptive repetition diagnostic; deliberately not a semantic quality score.
    words = re.findall(r"\w+(?:['’]\w+)?", str(text).casefold())
    counts = Counter(words)
    top, count = counts.most_common(1)[0] if counts else ('', 0)
    return dict(words=len(words), dominant_word=top,
                dominant_word_fraction=count / len(words) if words else 0,
                distinct_word_fraction=len(counts) / len(words) if words else 0)


def evaluation_summary(key, data, modified):
    records = data.get('records', [])
    repetition = [generation_stats(r.get('free_generated_text', '')) for r in records if isinstance(r, dict)]
    return clean(dict(key=key, step=data.get('step'), split=data.get('split', 'valid'),
        timestamp=data.get('timestamp', modified), nll_per_code=data.get('nll_per_code'),
        nll_kind=data.get('nll_kind'), seconds=data.get('seconds'), samples=len(records),
        context_nll_delta=data.get('context_relevance',{}).get('mean_shuffled_minus_true_nll'),
        repeated_trigram_fraction=data.get('repetition',{}).get('mean_repeated_trigram_fraction'),
        end_to_end_content_tokens_per_second=data.get('end_to_end_content_tokens_per_second'),
        boundary_continuity=data.get('boundary_continuity'),
        repetitive_samples=sum(r['dominant_word_fraction'] > .5 for r in repetition),
        median_dominant_word_fraction=statistics.median(r['dominant_word_fraction'] for r in repetition) if repetition else None,
        repetition_definition='More than half of word tokens are the same word; punctuation ignored. This is not a semantic quality metric.'))


class Engine:
    def __init__(self, run, database, *, clock=time.time, system_provider=None, interval=5):
        self.run = Path(run).resolve()
        self.gram = self.run / 'gram'
        self.database = Path(database).resolve()
        if self.database.is_relative_to(self.run):
            raise ValueError('Telemetry storage must be outside the training run')
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.interval = interval
        self.system_provider = system_provider or self._system
        self.lock = threading.RLock()
        self.started = clock()
        self.last_poll = None
        self.last_error = None
        self.poll_seconds = None
        self.db = sqlite3.connect(str(self.database), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS cursors(path TEXT PRIMARY KEY, inode TEXT, offset INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS metrics(id TEXT PRIMARY KEY, step INTEGER NOT NULL, timestamp REAL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS metrics_step ON metrics(step, timestamp);
          CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, timestamp REAL, kind TEXT, level TEXT, title TEXT, detail TEXT);
          CREATE TABLE IF NOT EXISTS evaluations(key TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, summary TEXT, payload TEXT);
          CREATE TABLE IF NOT EXISTS resources(timestamp REAL PRIMARY KEY, payload TEXT NOT NULL);
        ''')
        owner = self.db.execute("SELECT value FROM metadata WHERE key='run'").fetchone()
        if owner and owner[0] != str(self.run):
            self.db.close()
            raise ValueError('Telemetry database belongs to a different run')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('run', ?)", (str(self.run),))
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def event(self, key, kind, level, title, detail='', timestamp=None):
        self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)',
                        (key, timestamp or self.clock(), kind, level, title, str(detail)[:4000]))

    def _ingest_metrics(self):
        path = self.gram / 'metrics.jsonl'
        try:
            stat = path.stat()
        except OSError:
            return
        cursor = self.db.execute('SELECT * FROM cursors WHERE path=?', ('metrics',)).fetchone()
        inode = str(stat.st_ino)
        offset = cursor['offset'] if cursor and cursor['inode'] == inode and stat.st_size >= cursor['offset'] else 0
        if cursor and offset == 0 and cursor['offset']:
            self.event(f'rotation:{inode}:{stat.st_mtime_ns}', 'ingestion', 'info', 'Metrics file rotated or truncated', 'Rescanning safely; identical records are deduplicated.')
        with path.open('rb') as handle:
            handle.seek(offset)
            # Bound each poll even if a corrupt producer writes an enormous file.
            for _ in range(10000):
                start = handle.tell()
                line = handle.readline(MAX_LINE + 1)
                if not line:
                    break
                if not line.endswith(b'\n'):
                    # A concurrent append may still be writing this line.
                    if len(line) > MAX_LINE:
                        self.event(f'oversized:{inode}:{start}', 'ingestion', 'warning', 'Oversized metrics line', 'This line is not ingested. Fix the log producer before it grows further.')
                    break
                offset = handle.tell()
                digest = hashlib.sha256(line).hexdigest()
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Expected a JSON object')
                    if 'loss' in row:
                        if number(row.get('step')) is None or row['step'] < 0 or row['step'] != int(row['step']):
                            raise ValueError('Invalid step')
                        safe = clean(row)
                        self.db.execute('INSERT OR IGNORE INTO metrics VALUES (?,?,?,?)',
                            (digest, row['step'], number(row.get('timestamp')), json.dumps(safe, allow_nan=False)))
                        if number(row['loss']) is None:
                            self.event('nonfinite:' + digest, 'numerics', 'critical', 'Non-finite training loss', 'The log contains a non-finite value.', row.get('timestamp'))
                    elif row.get('event') and row['event'] != 'evaluation':
                        self.event(digest, str(row['event']), 'info', str(row['event']).replace('_', ' ').capitalize(), json.dumps(clean(row)), row.get('timestamp'))
                except (ValueError, TypeError, UnicodeError) as error:
                    self.event('malformed:' + digest, 'ingestion', 'warning', 'Malformed metrics line skipped', str(error))
        self.db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?)', ('metrics', inode, offset))

    def _ingest_evaluations(self):
        # Evaluation marker lines omit timestamps. The saved evaluation is the
        # authoritative event; do not date historic markers at ingestion time.
        self.db.execute("DELETE FROM events WHERE kind='evaluation'")
        paths = list(self.gram.glob('eval_*.json'))
        paths += [p for p in (self.run/'final_valid.json', self.run/'final_test.json') if p.exists()]
        for path in paths:
            stat = path.stat()
            known = self.db.execute('SELECT mtime_ns,size FROM evaluations WHERE key=?', (path.name,)).fetchone()
            if known and known['mtime_ns'] == stat.st_mtime_ns and known['size'] == stat.st_size:
                continue
            data = read_json(path)
            if not isinstance(data, dict) or not isinstance(data.get('records'), list) or number(data.get('nll_per_code')) is None:
                self.event('bad-eval:' + path.name, 'ingestion', 'warning', 'Evaluation is unreadable', path.name)
                continue
            data = clean(data)
            summary = evaluation_summary(path.name, data, stat.st_mtime)
            self.db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?,?)',
                            (path.name, stat.st_mtime_ns, stat.st_size, json.dumps(summary), json.dumps(data, allow_nan=False)))
            self.event('eval:' + path.name, 'validation' if summary['split'] == 'valid' else 'test', 'info',
                f"{summary['split'].capitalize()} evaluation saved", f"Step {summary['step'] if summary['step'] is not None else 'final'} · code NLL {summary['nll_per_code']:.4f} · {summary['samples']} examples", summary['timestamp'])

    @staticmethod
    def _process(info, module):
        if not psutil or not isinstance(info, dict) or not isinstance(info.get('pid'), int):
            return dict(pid=None if not isinstance(info, dict) else info.get('pid'), alive=None, identity_verified=False)
        try:
            proc = psutil.Process(info['pid'])
            cmd = proc.cmdline()
            expected = info.get('command')
            matches = module in cmd and (not expected or cmd[1:] == expected[1:])
            # Do not report a reused PID as a healthy trainer.
            matches = matches and proc.create_time() <= info.get('timestamp', time.time()) + 5
            return dict(pid=proc.pid, alive=matches and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE,
                        identity_verified=matches, rss_bytes=proc.memory_info().rss if matches else None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return dict(pid=info['pid'], alive=False, identity_verified=False)

    def _system(self):
        if not psutil:
            return dict(available=False, reason='Install psutil for live host counters')
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        battery = psutil.sensors_battery()
        times=psutil.cpu_times()
        busy=sum(getattr(times,key,0.) for key in ('user','nice','system','irq','softirq','steal'))
        total=busy+sum(getattr(times,key,0.) for key in ('idle','iowait'))
        previous=getattr(self,'_cpu_sample',None)
        self._cpu_sample=(busy,total)
        cpu_percent=max(0.,min(100.,100*(busy-previous[0])/(total-previous[1]))) if previous and total>previous[1] else None
        trainer_info = read_json(self.run/'child.json', {})
        command = trainer_info.get('command', [])
        module = next((name for name in ('v15_astra_v3.shortblock.train','v15_astra_v3.shortblock.finalize',
            'v15_astra_v3.evaluate_raw') if name in command), 'v15_astra_v3.train')
        plan = read_json(self.run/'plan.json', {})
        supervisor_module = plan.get('supervisor_module','v15_astra_v3.run24')
        return dict(available=True, cpu_percent=cpu_percent,
            ram_total_bytes=memory.total, ram_available_bytes=memory.available, ram_percent=memory.percent,
            swap_used_bytes=swap.used, disk_free_bytes=shutil.disk_usage(self.run).free,
            battery_percent=battery.percent if battery else None, ac_power=battery.power_plugged if battery else None,
            trainer=self._process(trainer_info, module),
            supervisor=self._process(read_json(self.run/'process.json', {}), supervisor_module))

    def poll(self):
        started = time.monotonic()
        with self.lock:
            try:
                with self.db:
                    self._ingest_metrics()
                    self._ingest_evaluations()
                    now = self.clock()
                    resource = clean(dict(timestamp=now, **self.system_provider()))
                    self.db.execute('INSERT OR REPLACE INTO resources VALUES (?,?)', (now, json.dumps(resource)))
                    self.db.execute('DELETE FROM resources WHERE timestamp < ?', (now - 3*86400,))
                self.last_poll = now
                self.last_error = None
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
            self.poll_seconds = time.monotonic() - started

    def history(self, limit=1200, after_step=-1):
        with self.lock:
            rows = self.db.execute('SELECT payload FROM metrics WHERE step>? ORDER BY step,timestamp', (after_step,)).fetchall()
            values = [json.loads(row[0]) for row in rows]
            # Uniformly retain the whole time range, including first and latest.
            if len(values) > limit:
                values = [values[round(i*(len(values)-1)/(limit-1))] for i in range(limit)]
            return values

    def evaluations(self):
        with self.lock:
            values = [json.loads(row[0]) for row in self.db.execute('SELECT summary FROM evaluations')]
            return sorted(values, key=lambda row: (row.get('timestamp') or 0, row['key']))

    def samples(self, key, offset=0, limit=1):
        with self.lock:
            row = self.db.execute('SELECT payload,summary FROM evaluations WHERE key=?', (key,)).fetchone()
            if not row:
                raise KeyError(key)
            payload = json.loads(row['payload'])
            records = payload['records']
            selected = []
            for record in records[offset:offset+limit]:
                selected.append(dict(record, repetition=generation_stats(record.get('free_generated_text', ''))))
            return dict(evaluation=json.loads(row['summary']), offset=offset, total=len(records), records=selected)

    def events(self, limit=80):
        with self.lock:
            return [dict(row) for row in self.db.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))]

    def resource_history(self, limit=180):
        with self.lock:
            rows = self.db.execute('SELECT payload FROM resources ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
            return [json.loads(row[0]) for row in reversed(rows)]

    def export_csv(self):
        with self.lock:
            keys = ['step','timestamp','loss','nll','kl_sum','kl_per_coordinate','grad_norm','lr','update_seconds','blocks_per_second','available_ram','mps_driver_bytes','sigma_median']
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for row in self.db.execute('SELECT payload FROM metrics ORDER BY step,timestamp'):
                writer.writerow(json.loads(row[0]))
            return out.getvalue()

    def snapshot(self):
        with self.lock:
            now = self.clock()
            plan = read_json(self.run/'plan.json', {})
            supervisor = read_json(self.run/'status.json', {})
            trainer = read_json(self.gram/'status.json', {})
            manifest = read_json(self.gram/'manifest.json', {})
            recent_rows = self.db.execute('SELECT payload FROM metrics ORDER BY step DESC,timestamp DESC LIMIT 20').fetchall()
            recent = [json.loads(row[0]) for row in recent_rows]
            latest = recent[0] if recent else {}
            if number(trainer.get('step')) is not None and trainer['step'] >= latest.get('step', -1):
                latest = dict(latest, **trainer)
            resources = self.resource_history(1)
            host = resources[-1] if resources else {}
            evaluations = self.evaluations()
            validation = [e for e in evaluations if e['split'] == 'valid' and number(e['nll_per_code']) is not None]
            baseline = validation[0] if validation else None
            last_eval = validation[-1] if validation else None
            best_eval = min(validation, key=lambda e: e['nll_per_code']) if validation else None
            evaluation_age=max(0,now-last_eval['timestamp']) if last_eval and number(last_eval.get('timestamp')) is not None else None
            evaluation_step_lag=max(0,latest.get('step',0)-last_eval['step']) if last_eval and number(last_eval.get('step')) is not None else None
            initial_samples_only=bool(last_eval and last_eval.get('step')==0 and latest.get('step',0)>0)
            sample_age = max(0, now - latest['timestamp']) if number(latest.get('timestamp')) is not None else None
            state = supervisor.get('status', 'waiting')
            parent_state=read_json(self.run.parent/'status.json', {})
            if plan.get('supervisor_module')=='v15_astra_v3.shortblock.run' and parent_state.get('status') in ('failed','stopped'):
                state=parent_state['status']
                supervisor=parent_state
            if state not in TERMINAL:
                if host.get('trainer', {}).get('alive') is False:
                    state = 'recovering' if host.get('supervisor', {}).get('alive') else 'offline'
                elif sample_age is not None and sample_age > 180:
                    state = 'stale'
            checkpoints = []
            for path in sorted(self.gram.glob('*.pt')):
                stat = path.stat()
                checkpoints.append(dict(name=path.name, bytes=stat.st_size, modified=stat.st_mtime, age_seconds=max(0, now-stat.st_mtime)))
            alerts = []
            def alert(key, severity, title, detail):
                alerts.append(dict(id=key, severity=severity, title=title, detail=detail))
            if self.last_error:
                alert('collector', 'critical', 'Telemetry collection error', self.last_error)
            if 'loss' in latest and number(latest['loss']) is None:
                alert('numerics','critical','Latest training loss is non-finite','Check the run events and trainer status. Telemetry does not modify or restart the trainer.')
            if state in ('stale', 'offline', 'recovering', 'failed', 'paused_resource'):
                alert('process', 'critical' if state in ('offline','failed') else 'warning', state.replace('_',' ').capitalize(), supervisor.get('reason') or 'Check the run status and recent events. The dashboard never restarts training.')
            if sample_age is not None and sample_age > 180 and state not in TERMINAL:
                alert('stale', 'warning', 'Training metrics are stale', f'The most recent training row is {sample_age:.0f} seconds old. A longer evaluation can also cause this.')
            if last_eval and last_eval['samples'] and last_eval['repetitive_samples'] / last_eval['samples'] > .5:
                alert('repetition', 'warning', 'Generated text is highly repetitive', f"{last_eval['repetitive_samples']}/{last_eval['samples']} examples repeat one word for more than half their word tokens. Inspect the samples alongside code loss.")
            if initial_samples_only:
                alert('evaluation_initial','warning','Samples are from the untrained checkpoint',
                    f"Displayed validation is step 0; training has reached step {latest.get('step',0)}. Fresh checkpoint evaluation is needed to assess progress.")
            elif evaluation_step_lag and evaluation_age is not None and evaluation_age>1800:
                alert('evaluation_lag','warning','Generated samples trail training',
                    f"Validation is {evaluation_step_lag} updates behind the current model. Read the saved evaluation step alongside the training step.")
            kl = [r['kl_sum'] for r in recent if number(r.get('kl_sum')) is not None]
            if len(kl) >= 5 and latest.get('step',0) >= 1000 and statistics.median(kl) < .001:
                alert('kl', 'warning', 'KL remains near zero', 'The posterior and prior are close. The stochastic component has not demonstrated a benefit; this alone does not prove failure.')
            if number(host.get('ram_available_bytes')) is not None and host['ram_available_bytes'] < 1.5*1024**3:
                alert('memory','warning','Available memory is low','The trainer has its own clean-stop resource guard.')
            if number(host.get('disk_free_bytes')) is not None and host['disk_free_bytes'] < 3*1024**3:
                alert('disk','warning','Disk space is low','The trainer stops below its own 2 GiB reserve.')
            if host.get('ac_power') is False:
                alert('power','warning','Running on battery','Connect AC power for the long run.')
            latest_check = next((c for c in checkpoints if c['name']=='latest.pt'), None)
            if latest_check and latest_check['age_seconds'] > 660 and state == 'training':
                alert('checkpoint','warning','Checkpoint is older than expected','The latest resumable checkpoint is more than 11 minutes old.')
            if latest_check is None and state == 'training' and number(plan.get('start')) is not None and now-plan['start']>600:
                alert('checkpoint','warning','No resumable checkpoint found','The run has been active for over ten minutes without a latest.pt checkpoint.')
            if len(validation) >= 2 and baseline['nll_per_code']:
                reduction = (baseline['nll_per_code']-last_eval['nll_per_code'])/baseline['nll_per_code']
            else:
                reduction = None
            last_resource = host.get('timestamp')
            data_path = Path(plan.get('data', ''))
            # Read only the small manifest, and only from this experiment tree.
            experiment_root=next((p.parent for p in self.run.parents if p.name=='runs'),self.run.parent.parent)
            allowed_data=data_path.is_absolute() and data_path.resolve().is_relative_to(experiment_root)
            dataset = read_json(data_path/'manifest.json', {}) if allowed_data else {}
            preparation=read_json(data_path/'preparation.json', {}) if allowed_data else {}
            deadline, start = number(plan.get('deadline')), number(plan.get('start'))
            elapsed = max(0, now-start) if start is not None else None
            budget = deadline-start if deadline is not None and start is not None else None
            completed = min(elapsed, budget) if elapsed is not None and budget is not None else elapsed
            metric_count = self.db.execute('SELECT count(*) FROM metrics').fetchone()[0]
            return clean(dict(schema_version=SCHEMA_VERSION, generated_at=now,
                collector=dict(last_poll=self.last_poll, poll_seconds=self.poll_seconds, error=self.last_error, uptime_seconds=now-self.started,
                    fresh=last_resource is not None and now-last_resource < max(20,self.interval*3), interval_seconds=self.interval,
                    metric_rows=metric_count, read_only_training=True, imports_torch=False),
                run=dict(name=self.run.name, status=state, supervisor_status=supervisor.get('status'), phase=plan.get('phase'),
                    start=start, deadline=deadline, training_deadline=plan.get('training_deadline'),elapsed_seconds=completed,
                    remaining_seconds=max(0,deadline-now) if deadline is not None else None,
                    progress_fraction=min(1,max(0,(now-start)/budget)) if budget and budget>0 else None,
                    started_ist=plan.get('started_ist'),deadline_ist=plan.get('deadline_ist'),disabled=plan.get('disabled',[])),
                model=dict(arm=manifest.get('arm'),device=manifest.get('device'),precision=manifest.get('precision'),
                    parameters=manifest.get('ledger',{}),batch=manifest.get('batch'),accumulation=manifest.get('accumulation'),
                    seed=manifest.get('seed'),objective=manifest.get('objective'),config=manifest.get('config',{})),
                latest=latest, metric_age_seconds=sample_age,
                rolling=dict(window_logged_updates=len(recent),loss_mean=statistics.mean(r['loss'] for r in recent if number(r.get('loss')) is not None) if any(number(r.get('loss')) is not None for r in recent) else None,
                    median_update_seconds=statistics.median(r['update_seconds'] for r in recent if number(r.get('update_seconds')) is not None) if any(number(r.get('update_seconds')) is not None for r in recent) else None),
                evaluations=evaluations,
                validation=dict(baseline=baseline,latest=last_eval,best=best_eval,relative_nll_reduction=reduction,history=validation,
                    evaluation_age_seconds=evaluation_age,training_steps_since_evaluation=evaluation_step_lag,
                    initial_checkpoint_only=initial_samples_only,
                    next_expected_at=(last_eval['timestamp']+manifest.get('eval_every',1800)) if last_eval and state not in TERMINAL else None,
                    label='Prior-only code NLL; lower is better. This is not text perplexity or accuracy.'),
                dataset=dict(unique_blocks=dataset.get('unique_blocks'),examples=dataset.get('examples'),history=dataset.get('history'),
                    target_ids=dataset.get('block_size',plan.get('target_ids')),context_ids=dataset.get('observed_context_ids',plan.get('context_ids')),
                    preparation={k:preparation.get(k) for k in ('encoded_blocks','total_blocks','blocks_per_second','timestamp')},
                    decode_audit=dataset.get('decode_audit'),sha=plan.get('dataset_sha'),codec_id=plan.get('codec_id')),
                resources=host,checkpoints=checkpoints,alerts=alerts,events=self.events(30)))
