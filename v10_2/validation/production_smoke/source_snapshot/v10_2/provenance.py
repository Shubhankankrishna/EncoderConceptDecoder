"""Hash every imported project Python dependency, including nested optimizations."""
import hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def source_files():
    paths=[]
    for package in ('v10_2','v10_1','v10_10M'):
        paths.extend((ROOT/package).glob('*.py'))
    for folder in ('v10_1/optimization','v10_10M/data_tools'):
        paths.extend((ROOT/folder).rglob('*.py'))
    return sorted(set(paths))

def provenance():
    return dict(files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()},
        torch_version=torch.__version__)
