# V14 notes: open items carried forward from the V13-tiny run

Started 14 Oct 2026 during the V13-tiny 24 h test (`v13_tiny/`, run `runs/base`). Each item records where it came from, the
evidence, and what must happen before it changes a model. Nothing here changes the run in progress.

---

## 1. NCP output looks collapsed: confirm the cause before fixing it (V13.1 candidate)

### Observation (14 Oct, ~15:50 IST)

Source: one CPU forward of a copy of `runs/base/latest.pt` at **step 6,898** (6.0 h of 11), on a single 16-token
document ("Lily found a red ball in the garden and gave it to her brother.", 4 concept groups). Written by
`v13_viz/extract.py --journey` to `v13_viz/data/journey.json`.

| Group g | NCP predicted codes (argmax per segment) | Target codes of m_{g+1} | p(target) per segment |
|---|---|---|---|
| 0 | [8, 22, 15, 30] | [8, 22, 15, 30] | 0.94, 0.86, 0.69, 0.92 |
| 1 | [8, 22, 15, 30] | [8, 22, 15, 30] | 0.92, 0.83, 0.70, 0.92 |
| 2 | [8, 22, 15, 30] | [9, 26, 20, 12] | 0.0001, 0.0009, 0.0005, 0.0016 |

The measurement shows two things:
- **The predictions are constant.** The head outputs the same tuple for every group.
- **The targets are partly constant too.** m_1 and m_2 quantise to that same tuple; only m_3 differs. Run telemetry also shows
  codebook use falling from about 25 to about 22 effective codes.

Related evidence: the interim u_g-vs-p_g probe test (checkpoint ≈ step 2,400) failed in all four families (−0.6, −0.7,
−0.3, +2.9 points against the ≥ 5-point bar).

**One document is weak evidence.** This is an observation to confirm, not a finding.

### Decision: do not change the current run

The run is pre-registered. Its readings (`v13_tiny/REPORT.md` §5), the continual-learning budget and tomorrow's
equal-compute baseline-transformer comparison all assume this exact model. A mid-run change would void the comparison
and break the protocol. Log the observation and let the final diagnostics read it.

### Two hypotheses, which need different fixes

1. **The head predicts the marginal.** u_g carries too little information, so the head outputs the most common codes
   whatever the input. This fits the failing u_g-vs-p_g probe. Candidate fixes:
   - raise or schedule λ;
   - predict 2–4 groups ahead, so the easy "same as now" answer stops working;
   - strengthen the concept module's inputs, for example by lowering β₁ (the raw-embedding share; learned β ≈ 0.10, 0.08,
     0.82 at step 6,898).
2. **The targets themselves collapse.** m maps to few codes. Candidate fixes:
   - a variance or covariance term on m (VICReg-style);
   - a stricter dead-code reset, or a larger product codebook;
   - an EMA target encoder, so the target does not drift with the online encoder.

### Checks: CPU only, after training ends (never during the 20:15–20:25 probe window)

- Over a few hundred documents per family, measure:
  - the entropy of the **predicted** codes;
  - the entropy of the **target** codes;
  - the conditional entropy of the target given u_g.
- Track the same numbers across `snap_010` … `snap_050` and the final checkpoint, to see whether collapse is getting worse
  or recovering.
- **Deciding test:**
  - prediction entropy much lower than target entropy → hypothesis 1 (head predicts the marginal);
  - both low → hypothesis 2 (target collapse).

### Then: V13.1

V13.1 is a pre-registered follow-up:
- the fix is chosen by the checks above;
- it is logged in `v13_tiny/DEVIATIONS.md`;
- it runs against the same baseline transformer.

**Prior pick if one change had to be made now:** predict several groups ahead, plus a variance term on m. Together they
address both hypotheses cheaply, without an EMA encoder's extra memory on the 16 GB M2 Air.
