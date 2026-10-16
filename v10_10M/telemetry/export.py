"""Consistent offline JSON/JSONL export for UI development and analysis."""
import argparse
from pathlib import Path
from .server import DEFAULT_DATA
from .store import Store, strict_json, encode, atomic_json
from . import SCHEMA_VERSION


def export(store, run_id, destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    # One SQLite read transaction captures a consistent event/metric cutoff.
    with store.connect() as db:
        db.execute('BEGIN')
        watermark=db.execute('SELECT COALESCE(MAX(id),0) FROM events WHERE run_id=?',(run_id,)).fetchone()[0]
        for table,filename,id_field in [('events','events.jsonl','id'),('metrics','metrics.jsonl','event_id')]:
            tmp=destination/(filename+'.tmp')
            with tmp.open('w') as f:
                rows=db.execute(f'SELECT * FROM {table} WHERE run_id=? AND {id_field}<=? ORDER BY {id_field}',(run_id,watermark))
                for row in rows:
                    obj=dict(row)
                    if 'payload' in obj:obj['payload']=strict_json(obj['payload'])
                    f.write(encode(obj)+'\n')
            tmp.replace(destination/filename)
    atomic_json(destination/'overview.json',store.overview(run_id))
    atomic_json(destination/'catalog.json',dict(schema_version=SCHEMA_VERSION,metrics=store.catalog(run_id)))
    atomic_json(destination/'export.json',dict(schema_version=SCHEMA_VERSION,run_id=run_id,event_watermark=watermark,
        note='Events and metrics share this cutoff; overview and catalog reflect export completion time.'))
    return destination


def main():
    p=argparse.ArgumentParser();p.add_argument('--database',default=str(DEFAULT_DATA/'telemetry.sqlite3'))
    p.add_argument('--run-id',default='pretrain_24h');p.add_argument('--output',default=str(DEFAULT_DATA/'export'))
    a=p.parse_args();print(export(Store(a.database),a.run_id,a.output))


if __name__=='__main__':main()
