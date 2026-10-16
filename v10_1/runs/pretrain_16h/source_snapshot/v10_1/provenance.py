"""Source and immutable corpus identities for isolated validation artifacts."""
import hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def package_hash(directory):
    h=hashlib.sha256()
    for p in sorted((ROOT/directory).glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def provenance():
    return dict(v10_1_source_sha256=package_hash('v10_1'),v10_source_sha256=package_hash('v10_10M'),
        active_data_manifest_sha256=hashlib.sha256((ROOT/'v10_10M/data/manifest.json').read_bytes()).hexdigest(),
        torch_version=torch.__version__)
