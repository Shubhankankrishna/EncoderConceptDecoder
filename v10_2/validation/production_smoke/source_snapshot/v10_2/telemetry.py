"""V10.2 adapter for the tested read-only SQLite collector and dashboard."""
from pathlib import Path
import time
import psutil
from urllib.parse import urlparse
from v10_10M.telemetry import server,store

ROOT=Path(__file__).resolve().parent
BASE_HANDLER=server.make_handler


def make_handler(database,collector):
    base=BASE_HANDLER(database,collector)
    class Handler(base):
        def do_GET(self):
            if urlparse(self.path).path=='/api/v1/schema':
                return self.respond(200,dict(schema_version=1,metric_availability=store.METRIC_AVAILABILITY,
                    cursor='Integer event id, not training step; preserve ingestion order.',
                    timestamps='Unix seconds UTC. Training uses payload.wall_time (trainer clock); evaluation uses atomic-file completion time.',
                    architecture='V10.2 RCDL RF',loss_weights=dict(token=1,future=.1,explore=.1,selector=.05),
                    future_offsets='Numerators and counts indexed 0..7; CE means require division by counts. Zero count means unmeasured.',
                    routes=['health','runs','overview','catalog','series','events','evaluations','artifacts','issues','download'],read_only=True))
            return super().do_GET()
    return Handler


class Store(store.Store):
    def __init__(self,path):
        super().__init__(path)
        # Catalogue aggregation runs on every collector poll. Cover its step
        # columns so SQLite does not fetch millions of metric table rows.
        with self.connect() as db:
            db.execute('CREATE INDEX IF NOT EXISTS metric_catalog_cover ON metrics(run_id,name,step)')

    def event(self,db,run_id,kind,source,record_key,payload,step,wall_time,quality,metrics):
        if kind=='training':
            if store.finite_number(payload.get('wall_time')):wall_time=payload['wall_time'];quality='trainer_clock'
            for key in ('dense_grad_norm_post','dense_clip_factor','engram_grad_norm','engram_grad_norm_post','engram_clip_factor','engram_rows_touched'):
                if store.finite_number(payload.get(key)):metrics['train.gradient.'+key]=payload[key]
            for key in ('component_grad_norms','weighted_losses','future_visits','future_offset_numerators','future_offset_counts','winner_counts','code_counts','family_input_tokens','objective_numerators','objective_denominators'):
                metrics.update(store.flatten_numeric(payload.get(key,{}),'train.'+key))
            for key in ('step_seconds','data_seconds','batch_prepare_seconds','last_checkpoint_seconds','last_evaluation_seconds','input_tokens','supervised_tokens','selector_regret','candidate_documents','future_documents','future_supervised_targets','utility_gap','utility_spread','selector_agreement'):
                if store.finite_number(payload.get(key)):metrics['train.'+key]=payload[key]
        if kind=='evaluation':
            metrics.update(store.flatten_numeric(payload.get('objective_gradient_probe',{}),'eval.objective_gradient_probe'))
        return super().event(db,run_id,kind,source,record_key,payload,step,wall_time,quality,metrics)


class Monitor(server.Monitor):
    def sample(self,include_host=False):
        # Base monitor protects against PID reuse by checking exact module/run.
        # This adapter applies the same rule to V10.2's process names.
        health,host=super().sample(include_host=False)
        pid=health.get('trainer_pid');valid=False
        try:
            proc=psutil.Process(pid) if pid else None
            if proc:
                cmd=proc.cmdline()
                valid=any(x in ('v10_2.supervise','v10_2.train') for x in cmd) and str(self.run) in cmd
                valid=valid and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
                if valid:
                    if self.process is None or self.process.pid!=pid or self.process.create_time()!=proc.create_time():
                        self.process=proc;self.last_cpu_pid=None
                    health['process_started_unix']=proc.create_time()
        except (psutil.NoSuchProcess,psutil.AccessDenied):self.process=None
        health['trainer_alive']=valid
        if include_host:
            vm=psutil.virtual_memory();swap=psutil.swap_memory();disk=psutil.disk_usage(self.run)
            host=dict(system_available_gib=vm.available/2**30,system_memory_percent=vm.percent,
                swap_used_gib=swap.used/2**30,disk_free_gib=disk.free/2**30,
                collector_rss_mib=psutil.Process().memory_info().rss/2**20,trainer_alive=valid)
            if valid:
                try:
                    cpu=self.process.cpu_percent(None)
                    host.update(trainer_rss_gib=self.process.memory_info().rss/2**30,trainer_threads=self.process.num_threads())
                    if self.last_cpu_pid==pid:host['trainer_cpu_percent']=cpu
                    self.last_cpu_pid=pid
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return health,host


def configure():
    server.ROOT=ROOT;server.DEFAULT_DATA=ROOT/'telemetry_data';server.WEB=ROOT/'web'
    server.Store=Store;server.Monitor=Monitor;server.make_handler=make_handler
    store.METRIC_AVAILABILITY.update({
        'train.loss.*':dict(available=True,source='train.jsonl',cadence='Every optimizer step',
            meaning='Effective-batch normalized losses. Total = token + .1 forecast-only future CE + .1 exploration + .05 selector. Two visit future losses weighted 1/3 and 2/3; candidate losses zero on non-candidate updates. No interval averaging.'),
        'train.gradient.dense_preclip_norm':dict(available=True,cadence='Every optimizer step',meaning='Dense gradient before clip=1; Engram excluded'),
        'engram_gradient_norm':dict(available=True,cadence='Every optimizer step',meaning='CPU sparse coalesced table gradient, independently clipped to 1'),
        'train.component_grad_norms.*':dict(available=True,cadence='Every 20 steps and step 1',meaning='Preclip component L2 norm of the accumulated total objective'),
        'per_layer_gradient_norms':dict(available=False,reason='Component norms recorded; layer histograms are not collected'),
        'eval.objective_gradient_probe.*':dict(available=True,cadence='Routine evaluation when 30 seconds remain in its budget',meaning='Fixed text training probe; dense token and weighted auxiliary gradients; no optimizer step'),
        'eval.*.pq_segment_entropy.*':dict(available=True,meaning='Entropy of mean soft PQ probability per segment, distinct from K selector utilization'),
        'eval.*':dict(available=True,cadence='Every 30 minutes',meaning='Fixed dev panels, balanced math generation, separate aligned PQ, cross PQ, cross continuous, whole reader, all-concept and recurrence ablations; 8 dev documents per family; up to 8 generations per math module / Sudoku difficulty under a 300-second cap')})


if __name__=='__main__':
    configure();server.main()
