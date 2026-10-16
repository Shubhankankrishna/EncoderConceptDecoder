"""Build a NEW clean corpus without changing the active run or its tokenizer.

Uses the pinned original parquet files with corrected field normalization.
Refuses an existing destination. This does NOT launch or resume training.
"""
import argparse
from pathlib import Path
import shutil
import types
import pyarrow as pa
from .. import prepare_data as base
from .clean_math import clean_math_field


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    destination=Path(a.output).resolve()
    if destination.exists():raise SystemExit('Refusing existing output; choose a new empty corpus path')
    destination.mkdir(parents=True)
    original=base.DATA
    # Preparation writes processed data into the new directory; raw sources are read copies.
    shutil.copytree(original/'raw',destination/'raw')
    read=base.pq.read_table
    def clean_table(path,*args,**kwargs):
        table=read(path,*args,**kwargs)
        for field in ('question','answer'):
            column=pa.array([clean_math_field(x) for x in table[field].to_pylist()],type=pa.string())
            table=table.set_column(table.schema.get_field_index(field),field,column)
        return table
    base.DATA=destination
    base.pq=types.SimpleNamespace(read_table=clean_table)
    base.main()
    # The base builder recomputes normalized identities, split hashes and trains a
    # fresh tokenizer on clean training text. Never pair this tokenizer with old weights.
    import json
    manifest=json.loads((destination/'manifest.json').read_text())
    manifest['normalization']='UTF-8 bytes and serialized byte literals decoded, surrounding whitespace stripped'
    manifest['preparation_version']=2
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
