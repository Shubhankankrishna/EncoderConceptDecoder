"""Check that every local link in v10_2_review.md resolves and that code fences pair."""
from pathlib import Path
import json, re, sys

ROOT = Path(__file__).resolve().parents[2]
doc = (ROOT / 'v10_2_review.md').read_text()
links = re.findall(r'\]\(([^)]+)\)', doc)
local = [l for l in links if '://' not in l and not l.startswith('#')]
missing = sorted({l for l in local if not (ROOT / l.split('#')[0]).exists()})
result = dict(local_links=len(local), distinct=len(set(local)), missing=missing,
              fences_paired=doc.count('```') % 2 == 0)
print(json.dumps(result, indent=2))
sys.exit(1 if missing or not result['fences_paired'] else 0)
