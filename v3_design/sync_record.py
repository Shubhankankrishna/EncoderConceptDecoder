"""Copy the record documents into run_m2_25M/webapp/data/record/ so the dashboard's Record page can render them.
Read-only for the sources; re-run after the logs change.   python3 sync_record.py
"""
import json, os, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "run_m2_25M", "webapp", "data", "record")
DOCS = [  # (id, title, source, what it is)
    ("decisions_v3", "v3 implementation decisions", "run_v3_19M/DECISIONS.md", "How v3 was built, tested, sped up and launched (V3-001 …)."),
    ("advisor_v3", "Advisor log (v3 implementation, run and webapp)", "run_v3_19M/ADVISOR_LOG.md", "Every advisor call: the advice in one line and what was done."),
    ("draft_v3", "draft_v3.md (r2) — the v3 design", "draft_v3.md", "The specification v3 implements, revised after the external review."),
    ("review_v3", "External deep review of the draft", "draft_v3_review.md", "The review that led to r2 (findings F1–F7, C1–C7, A1–A8)."),
    ("design_review_v3", "Design-review log (R1–R9)", "v3_design/REVIEW_v3.md", "Section-by-section review notes of the draft."),
    ("decisions_v2", "v2 decisions", "run_v2_19M/DECISIONS.md", "v2's implementation and run record, including V-012 (retired)."),
    ("design_webapp", "Dashboard design record", "run_m2_25M/webapp/DESIGN.md", "Why this dashboard is organised the way it is."),
]
os.makedirs(DST, exist_ok=True)
index = []
for did, title, src, what in DOCS:
    p = os.path.join(ROOT, src)
    if not os.path.exists(p):
        continue
    shutil.copyfile(p, os.path.join(DST, did + ".md"))
    st = os.stat(p)
    index.append(dict(id=did, title=title, source=src, what=what, bytes=st.st_size,
                      modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))))
json.dump(dict(docs=index, synced=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(DST, "index.json"), "w"), indent=1)
print("synced", len(index), "documents to", DST)
