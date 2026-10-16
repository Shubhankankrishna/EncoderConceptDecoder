#!/bin/sh
# Re-run the V10.2 document's own review script without overwriting its recorded
# review_checks.json. Only three lines differ from the original: the project root,
# the evidence directory, and the output path (a fresh temporary directory).
set -eu
ROOT=/Users/shubhankankrishna/Desktop/EncoderConceptDecoder
SRC="$ROOT/v10_2_design_evidence/2026-10-12/review_checks.py"
OUT=$(mktemp -d)
sed -e "s#^ROOT=Path(__file__).resolve().parents\[2\]#ROOT=Path('$ROOT')#" \
    -e "s#^DEST=Path(__file__).resolve().parent#DEST=ROOT/'v10_2_design_evidence/2026-10-12'; OUT=Path('$OUT')#" \
    -e "s#(DEST/'review_checks.json').write_text#(OUT/'review_checks.json').write_text#" \
    "$SRC" > "$OUT/review_checks_rerun.py"
diff "$SRC" "$OUT/review_checks_rerun.py" || true
/opt/anaconda3/bin/python -B "$OUT/review_checks_rerun.py" > "$OUT/stdout.json"
# review_script_sha256 hashes the script file itself, so it necessarily differs for the patched copy.
/opt/anaconda3/bin/python -c "import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); d=[k for k in a if a[k]!=b.get(k)]; print('fields differing from recorded results:', d); print('identical apart from script hash:', d==['review_script_sha256'])" \
    "$ROOT/v10_2_design_evidence/2026-10-12/review_checks.json" "$OUT/review_checks.json"
echo "outputs in $OUT"
