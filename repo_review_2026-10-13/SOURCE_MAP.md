# Python source map

Generated from the inspection snapshot of 13 October 2026. All 847 Python paths are represented below, grouped into 475 byte-identical content groups. Line numbers refer to the file as read; the active V12 workspace may change later. Imports and symbols are structural evidence, not a claim that every function was executed or manually reviewed line by line.

## Concept Design/concept_testbench.py

221 lines; SHA-256 `e94428f9dc1f17a75e46ee066ca21e768b0234de10557e78223b8f1058567024`.

Primary copy: [Concept Design/concept_testbench.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py>).

Concept-module test bench (isolated from encoder/decoder). Inputs are concept vectors c_hat (fixed random points on the sphere of radius sqrt(d), standing in for a frozen encoder). Output: NCP over a product-quantised vocabulary (k-means codebook of the concept vectors), CE on the next concept's codes. Block = encoder_v1 FULL layer anatomy (gated attention, NoPE + per-head temperature, QKV conv) in a 1-stream gated residual. Arms (layer applications in brackets): N4 4 untied layers \[4\] NCP-style stack U6 6 untied layers \[6\] equal FLOPs to the loops SUB P + (B1 B2) x2 + C \[6\] loop a subset (v1) SUB_D SUB + DiscoLoop channel between passe

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_testbench.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_testbench.py>)

Imports:

```python
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms
```

Symbols:

- [GRLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:24>) — lines 24–31
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:25>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:29>).
- [kmeans_pq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:34>) — lines 34–48
- [ConceptModule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:51>) — lines 51–86
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:52>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:67>), [phi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:70>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:72>), [concept_logp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:83>).
- [ic_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:91>) — lines 91–107
- [iw_build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:110>) — lines 110–133: Two entity-disjoint graphs sharing relations. ids: entities 0..2NE-1, relations 2NE..2NE+NR-1.
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:136>) — lines 136–180
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_testbench.py:184>) — lines 184–213

## Concept Design/concept_v0.py

188 lines; SHA-256 `46cefbfcf5d674f34bcd5326f00a63e1e6b8a4b0ab781049f4d80aa718e14920`.

Primary copy: [Concept Design/concept_v0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py>).

HRM THINKER concept core v0 (design draft) - reference implementation + checks. Reads encoder v1.0's c_hat (one sphere-normed vector per 4-token chunk), runs a looped two-state core (TRM-style: Z = scratch state that reads the input, Y = plan state that reads only Z and itself), predicts the next concept over a product-quantised vocabulary (NCP), and emits the plan for the decoder. Block anatomy = encoder v1.0's FULL layer (gated residual, QKV conv, NoPE, per-head temperature) at concept rate. Run `python concept_v0.py` for the checks.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_v0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_v0.py>)

Imports:

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from encoder_v1 import Encoder, GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms
```

Symbols:

- [CoreLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:13>) — lines 13–31: One layer of the shared step function f. Attention + FFN weights are shared by both modes; each mode has its own read/write gates. z-mode: read {X, Z, Y} -> write Z. y-mode: read {Z, Y} -> write Y.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:16>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:23>).
- [ConceptVocab](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:34>) — lines 34–69: Product-quantised concept vocabulary: S segments x N codes x ds dims (S*ds = dc). Trained ONLY by L_VQ = ||sg(c^s) - e^s_n||^2 (NCP-ArchPreview form); everything else reads sg(e). Health (F3): k-means++ init from the first batch of encoder
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:39>), [codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:43>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:44>), [init_from](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:48>), [_restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:57>), [vq_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:63>).
- [NCPHead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:72>) — lines 72–82: pi^s_n = softmax_n( -||q^s - sg(e^s_n)||^2 / tau_s ), q = W_q u. c_pred^s = sum_n pi^s_n sg(e^s_n).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:74>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:77>).
- [ConceptCore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:85>) — lines 85–111
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:86>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:90>), [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:101>).
- [token_plan](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:114>) — lines 114–123: Plan visible to decoder position t (0-indexed, predicting x_{t+1}): concept j*(t) = floor((t+1)/k) - 1. Null plan (zeros here; a learned vector in the decoder) when j* < 0 OR chunk j* belongs to another document: the first k-1 tokens of EVE
- [_randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:127>) — lines 127–130
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v0.py:132>) — lines 132–184

## Concept Design/concept_v2_testbench.py

128 lines; SHA-256 `fac312416372ab6944f12abfc270854f515338739d10ed261978b111c9f6119c`.

Primary copy: [Concept Design/concept_v2_testbench.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py>).

Concept module v2 test: loop-only core with input injection and sampled depth, vs NCP's plain stack. Task (compositional chains, in-context): facts \[a R b\] for an 8-cycle f in random order, SEP, then 2 queries \[Q x R R ... R PAD..\] with h relation markers. At the k-th R the target is f^k(x) (dense, like NCP at every position). Train chains h in {1,2,3}; test h up to 6, so positions k = 4..6 are unseen depths (hop-count extrapolation). Arms: N4 : 4 untied GR layers on one stream (NCP stack, option A) U6 : 6 untied layers L1S : ONE shared layer looped; state S (starts at 0), input X re-read every pass (GR reads {X,S}, writes S); passes R ~ 

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_v2_testbench.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/concept_v2_testbench.py>)

Imports:

```python
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms
from exp_concept import GRLayer, kmeans_pq
```

Symbols:

- [chain_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:25>) — lines 25–45
- [LoopLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:48>) — lines 48–56: GR layer that reads {X, S} and writes S only (X = input stream, never written).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:50>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:54>).
- [Core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:59>) — lines 59–86
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:60>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:72>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:75>), [concept_logp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:85>).
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:90>) — lines 90–97
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/concept_v2_testbench.py:100>) — lines 100–122

## Concept Design/e2e.py

226 lines; SHA-256 `8fabde1e2f33ee25a951816ee821127c37647c4d23c387abd46ff8ea5cdaf727`.

Primary copy: [Concept Design/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py>).

End-to-end test bench for the concept module: tokens -> encoder_v1 (frozen spec) -> concept core -> contract decoder. Encoder: encoder_v1 exactly (S S S F S S S F, two streams, Engram, CRC pooling, sphere norm) at test width d=64 (4 heads x 16, partial RoPE 4/16, Engram tables 1021 rows). Core arms: loop = v2 : ONE shared GR layer looped; streams X (input, re-read every pass), S (state, S0 = 0); R ~ U{1..4} loopz = v2.1: loop + read-only latent stream Z = z_k (K = 4 learned), one hypothesis per document, Forward XM with relaxed winner-take-all on the NCP loss; decoder reads the selected hypothesis (hard-EM); likelihood is always evaluated as 

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e.py>)

Imports:

```python
import sys, os, math, json, time, torch, torch.nn as nn, torch.nn.functional as F
import encoder_v1 as E
from encoder_v1 import GRRead, GRWrite, SwiGLU, ZCRMSNorm, rms
```

Symbols:

- [make_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:29>) — lines 29–54: Facts \[a R b\] (8-cycle; if branch, 2 nodes get a 2nd successor) at 0..23, filler to T0, then nq query units: \[Q x R^h PAD..\](8) \[= ans PAD..\](4). path=True -> answer is the whole path b1..bh (h <= 3).
- [ZLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:58>) — lines 58–64
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:59>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:62>).
- [GRL](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:67>) — lines 67–73
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:68>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:71>).
- [Core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:76>) — lines 76–122
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:77>), [core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:85>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:95>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:99>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:102>), [vq_init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:105>), [vq_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:113>).
- [Decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:125>) — lines 125–139
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:126>), [plan_index](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:129>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:132>).
- [Model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:142>) — lines 142–150
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:143>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:146>).
- [ncp_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:153>) — lines 153–157
- [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:161>) — lines 161–198
- [answer_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:202>) — lines 202–203
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e.py:207>) — lines 207–219

## Concept Design/e2e_analyze.py

100 lines; SHA-256 `fcbfb9420cfc225f4df66b46f3b71e33142db4a7df3bd40614d93b1f96feaac3`.

Primary copy: [Concept Design/e2e_analyze.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_analyze.py>).

Post-training analysis of end-to-end checkpoints (chain task), and the XM decision metrics (branch task).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e_analyze.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e_analyze.py>)

Imports:

```python
import sys, math, json, torch, torch.nn.functional as F
from e2e import *
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_analyze.py:6>) — lines 6–7
- [chain_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_analyze.py:11>) — lines 11–30
- [pass_dynamics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_analyze.py:33>) — lines 33–47: State norm, step size and gradient reaching each pass (answer NLL at the query positions).
- [branch_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_analyze.py:51>) — lines 51–83

## Concept Design/e2e_tests.py

46 lines; SHA-256 `e051952ffd3fb4993cca54dbf25d81ff1b6858e6c8a3608c3e1897e434f21651`.

Primary copy: [Concept Design/e2e_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_tests.py>).

Bit-exact contract tests for the end-to-end pipeline (float64, randomised weights so no path is hidden by zero-init).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/e2e_tests.py>)

Imports:

```python
import torch, math, torch.nn.functional as F
from e2e import *
```

Symbols:

- [randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/e2e_tests.py:6>) — lines 6–9

## Concept Design/exp_gradcuit.py

80 lines; SHA-256 `a8d8dfffc5a5cbc8a6b872de209278e25758fb2bf5903154abdddd28503b4721`.

Primary copy: [Concept Design/exp_gradcuit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_gradcuit.py>).

H1 test: GradCuit-style test-time latent steering in the concept stream of a trained v2 core (frozen weights). Latents = offsets dz on the concept-stream inputs X at the 2 positions just before the query (SEP, Q), i.e. between the 'prompt' (facts) and the 'continuation' (the chain answers). Update exactly as GradCuit: loss = -R * sum_t log p(pred_t) over the model's own greedy continuation, R = -1 if the verifier rejects the final answer, Adam, <= 10 updates, stop when correct. Baselines at the same verifier budget (11 calls): random-direction steering, sampling (BoN), rank enumeration; and extra passes.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_gradcuit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_gradcuit.py>)

Imports:

```python
import sys, json, math, torch, torch.nn.functional as F
from exp_v2 import Core, chain_batch, V
from exp_concept import kmeans_pq
```

Symbols:

- [fwd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_gradcuit.py:22>) — lines 22–28
- [regime](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_gradcuit.py:31>) — lines 31–74

## Concept Design/exp_s0.py

41 lines; SHA-256 `f1a4460493aaf0adecb6bf0abd8dffd2d2fff600c00ceb62aa22febc0f861649`.

Primary copy: [Concept Design/exp_s0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_s0.py>).

Review A/B: does a learned initial state S0 = s0 (position-independent, carries no input information) fix the pass-1 read-scale mismatch (S0 = 0 => rms(S) = 0 => pass-1 read is half-scale), and does it change hop extrapolation? Same task, sizes and schedule as exp_v2.py. Arms: L1S (baseline, S0 = 0) and L1S0 (S0 = learned s0, init 0.02*randn, no WD).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_s0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_s0.py>)

Imports:

```python
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
import exp_v2
from exp_v2 import Core, chain_batch, evaluate
from exp_concept import kmeans_pq
```

Symbols:

- [CoreS0](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_s0.py:10>) — lines 10–17
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_s0.py:11>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_s0.py:13>).
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_s0.py:19>) — lines 19–38

## Concept Design/exp_xm.py

135 lines; SHA-256 `a97a2a66d8c6b93d3745d00f19d4322765017f4dcfda2bb4bcf72c395270412d`.

Primary copy: [Concept Design/exp_xm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py>).

H2 test: Explorative-Modeling latent z in the v2 concept core (GRAM-style hypotheses without a VAE). Task: branching in-context graph. An 8-cycle f plus 2 nodes with a second successor g, so "x R" has 1 or 2 valid answers and "x R R" up to 4 valid chains. Each training query carries ONE sampled valid path (data realise one mode). The two R positions are predicted without seeing each other's answers -> joint prediction of a multi-part future, the same situation as a concept plan describing 4 future tokens at once. Arms: K1 v2 core, no latent (deterministic) XM4 v2 core + read-only latent stream Z = z_k (k = 1..4), Forward XM: per query, train 

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_xm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/exp_xm.py>)

Imports:

```python
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm
from exp_concept import kmeans_pq
```

Symbols:

- [branch_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:20>) — lines 20–38
- [Layer3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:41>) — lines 41–48: reads {X, S, Z} (Z optional), writes S.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:43>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:46>).
- [CoreZ](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:51>) — lines 51–68
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:52>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:59>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:62>), [concept_logp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:67>).
- [seg_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:71>) — lines 71–74
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:78>) — lines 78–103
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/exp_xm.py:106>) — lines 106–130

## Concept Design/probe_s0.py

37 lines; SHA-256 `0b7504e283d9b0861767236be42d4957c84b5a00f8b4843f4f2984ff342925a2`.

Primary copy: [Concept Design/probe_s0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/probe_s0.py>).

Per-pass dynamics for a trained arm: read RMS, |S|, step, turn angle, gradient reaching each pass; plus accuracy at R = 4, 8, 12, 16.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/probe_s0.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/probe_s0.py>)

Imports:

```python
import sys, json, torch, torch.nn.functional as F
from exp_v2 import Core, chain_batch
from exp_s0 import CoreS0
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/probe_s0.py:6>) — lines 6–9
- [dyn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/probe_s0.py:10>) — lines 10–23
- [acc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/probe_s0.py:25>) — lines 25–29

## Concept Design/scale.py

21 lines; SHA-256 `2ce4a96fe830a7bf240101c967d6c7272df213ce5b43cf145938ce2fcacde68a`.

Primary copy: [Concept Design/scale.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/scale.py>).

Analytic params / FLOPs / KV for the concept module (per TOKEN; the core runs once per k=4 tokens).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/scale.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/scale.py>)

Symbols:

- [layer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/scale.py:2>) — lines 2–5
- [core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/scale.py:6>) — lines 6–11
- [row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/scale.py:12>) — lines 12–12

## Concept Design/vq_check2.py

34 lines; SHA-256 `7b881bfb8ea1a38cceb6070cb1818c45aa837f4702097d0ca398e0f0ee33c46c`.

Primary copy: [Concept Design/vq_check2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check2.py>).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/vq_check2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/vq_check2.py>)

Imports:

```python
import torch
```

Symbols:

- [data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check2.py:4>) — lines 4–6
- [kmeanspp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check2.py:7>) — lines 7–11
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check2.py:12>) — lines 12–31

## Concept Design/vq_check3.py

41 lines; SHA-256 `09d5fc0183efa0c079e5a6304a52de89ed5aa35249ff9f4cbc313eb7cc68bcf0`.

Primary copy: [Concept Design/vq_check3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check3.py>).

Drifting targets (the encoder keeps training): cluster centres rotate + translate over 3000 steps. Adam-trained codes vs EMA k-means codes (decay 0.99), both with k-means++ init and dead-code restart every 100 steps.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Concept Design/vq_check3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Concept Design/vq_check3.py>)

Imports:

```python
import torch, math
```

Symbols:

- [centres](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check3.py:7>) — lines 7–9
- [data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check3.py:10>) — lines 10–10
- [kmeanspp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check3.py:11>) — lines 11–15
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Concept Design/vq_check3.py:16>) — lines 16–18

## v4_tiny/code/concept_core_v2.py

105 lines; SHA-256 `5692df2c57c42c5d9fa4bd51f7fd9b4bca632631608de80ec9784e474a43bdf7`.

Primary copy: [v4_tiny/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py>).

Stand-in for the concept module (v2 as tested in the concept test bench, K = 1, i.e. without the XM latent). ONE shared layer looped R times: reads {X, S}, writes S; X = c_hat (re-read every pass, gradient flows), S0 = 0; R ~ U{1..4} in training. Layer = the encoder's FULL layer at concept rate (gated attention, NoPE + per-head temperature, q/k/v convs) wrapped in the gated-residual grammar. Output u = zcRMSNorm(S). NCP: product-quantised vocabulary (k-means++ init, dead-code restarts, trained by L_VQ only), distance head, cross-entropy on the next concept's code indices. c_pred = sum_n pi_n sg(e_n). Only its interface matters for the decoder

Identical copies:

- [v41_tiny/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/concept_core_v2.py>)
- [run_m2_25M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/concept_core_v2.py>)
- [run_v2_19M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_v2.py>)
- [run_v3_19M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/concept_core_v2.py>)
- [Decoder Design/decoder_v1_bundle/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/concept_core_v2.py>)
- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/concept_core_v2.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_v2.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/concept_core_v2.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_v2.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/concept_core_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/concept_core_v2.py>)

Imports:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import GRRead, GRWrite, GatedAttention, SwiGLU, ZCRMSNorm, build_masks
```

Symbols:

- [LoopLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:17>) — lines 17–29
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:18>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:25>).
- [ConceptVocab](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:32>) — lines 32–72
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:33>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:40>), [init_from](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:45>), [vq_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:54>), [restart_dead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:63>).
- [ConceptCoreV2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:75>) — lines 75–105
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:76>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:85>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:92>), [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/concept_core_v2.py:101>).

## run_m2_25M/code/e2e.py

222 lines; SHA-256 `18b7c19c15ccfb1467d5ea8c2826bc5835186dbd9cd24b38a8bdec0a96c5a90e`.

Primary copy: [run_m2_25M/code/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py>).

End-to-end test of Decoder v1: frozen encoder v1.0 -> concept core (v2 stand-in) -> decoder. python e2e.py --test pipeline unit tests (float64, exact) python e2e.py --arm confined --steps N train one arm on the long-range recall task (harness.py task) Arms: confined (proposed, D0 = tok + P) | stack (NCP-style control) | confined_read (rejected arm); --nextlat adds the NextLat auxiliary. --task single | multi (--n_facts 1 = the 3-token carry task). Task: 4 facts \[F k v S\] at the start, ~204 tokens of Markov filler, 4 queries \[Q k P P\]\[P P P v\]; every answer is >= 200 tokens after its fact, far beyond the token-path reach (70 confined / 8

Identical copies:

- [Decoder Design/decoder_v1_bundle/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/e2e.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/e2e.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/e2e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/e2e.py>)

Imports:

```python
import argparse, json, math, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import EncoderConfig, Encoder, engram_indices
from concept_core_v2 import ConceptCoreV2
from hrm_decoder import DecoderConfig, Decoder, NextLat, param_groups, plan_index
from harness import MarkovFiller, make_batch, FILL0, V, T
```

Symbols:

- [make_batch_multi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:19>) — lines 19–38: Multi-token copy: each value is a random L-token string that must be written out from ONE plan. facts \[k v1 v2 v3\] x4 | Markov filler | queries \[Q k P P\]\[P v1 v2 v3\] x4. The query chunk's plan is the only source of v1, v2, v3 (offsets 1, 2,
- [HRMThinker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:41>) — lines 41–52
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:42>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:47>).
- [configs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:55>) — lines 55–61
- [pipeline_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:65>) — lines 65–117
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:124>) — lines 124–144
- [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py:147>) — lines 147–196

CLI arguments:

```python
ap.add_argument("--test", action="store_true")
ap.add_argument("--arm", default="confined")
ap.add_argument("--nextlat", action="store_true")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_dec", type=int, default=2)
ap.add_argument("--window", type=int, default=8)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--lam_h", type=float, default=1.0)
ap.add_argument("--lam_kl", type=float, default=1.0)
ap.add_argument("--value_weight", type=float, default=2.0)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--task", default="single")
ap.add_argument("--n_facts", type=int, default=4)
ap.add_argument("--out", default="run.json")
```

## Encoder Design/harness.py

204 lines; SHA-256 `430050ea568df5b2088c9b05edefd369b67bd581fab9f4440eb62b9cd97f5454`.

Primary copy: [Encoder Design/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py>).

Verification harness for Encoder v1.0: encoder + stand-in concept core + stand-in decoder. Task (synthetic, V=512, T=256, one document per row): \[BOS P P P\] \[F k v S\]x4 facts | ~204 tokens of local Markov filler | 4 queries: \[Q k P P\]\[P P P v\] The value v is >200 tokens after its fact, far beyond the token-path reach (encoder 63 + decoder 20 = 83), so it can only be predicted through the concept core (two-stream arm) or through G (bypass arm). Arms: two_stream : decoder reads L (the frozen design) bypass : decoder reads G instead (NCP-style token path; the F6 failure mode) Eval ablation: plan zeroed ("core off").

Identical copies:

- [run_m2_25M/code/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/harness.py>)
- [Decoder Design/decoder_v1_bundle/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/harness.py>)
- [v3_design_review_2026-09-29/source_snapshot/Encoder Design/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Encoder Design/harness.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/harness.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/harness.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/harness.py>)

Imports:

```python
import math, time, json, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, engram_indices, param_groups, rms, ZCRMSNorm,
                         GatedAttention, SwiGLU, ShortConv, build_masks, rope_cache)
```

Symbols:

- [MarkovFiller](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:25>) — lines 25–38
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:26>), [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:32>).
- [make_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:41>) — lines 41–57
- [DecLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:60>) — lines 60–69: Single-stream PreNorm SWA block reusing the encoder's gated attention (stand-in decoder).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:62>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:67>).
- [CoreLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:72>) — lines 72–81: Stand-in concept core block: full causal attention over concepts (NoPE, gated), SwiGLU.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:74>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:79>).
- [Harness](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:84>) — lines 84–120
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:85>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:95>).
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:123>) — lines 123–141
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/harness.py:144>) — lines 144–200

CLI arguments:

```python
ap.add_argument("--arm", default="two_stream")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="result.json")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--value_weight", type=float, default=2.0)
```

## v4_tiny/code/hrm_decoder.py

384 lines; SHA-256 `6ac28af1d8f181a38102c222da2ffdf89742ee6ed0554512ced34aa95abb1085`.

Primary copy: [v4_tiny/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py>).

HRM THINKER - Decoder v1 (design candidate). Reference implementation + exact unit tests. What the decoder receives (and nothing else): tok (B, T, d) encoder token stream = zcRMSNorm(L), reach 63 u (B, M, dc) concept-core output state per chunk (post-loop only) c_pred (B, M, dc) NCP prediction of the NEXT chunk, made at each chunk doc (B, T) document ids E (V, d) the encoder's embedding matrix (tied output head) Plan (target-aligned concept shift): position t predicts x_{t+1}; it reads chunk j*(t) = floor((t+1)/4) - 1, the newest chunk whose last token 4j*+3 <= t. P_t = W_u u_{j*} + W_c c_pred_{j*} + o_{r(t)}; r(t) = (t+1) mod 4 is the index 

Identical copies:

- [v41_tiny/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/hrm_decoder.py>)
- [Decoder Design/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/hrm_decoder.py>)
- [run_m2_25M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/hrm_decoder.py>)
- [run_v2_19M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/hrm_decoder.py>)
- [run_v3_19M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/hrm_decoder.py>)
- [Decoder Design/decoder_v1_bundle/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/hrm_decoder.py>)
- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_decoder.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_decoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/hrm_decoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_decoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/hrm_decoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/hrm_decoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/hrm_decoder.py>)

Imports:

```python
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import (EncoderConfig, rms, ZCRMSNorm, GRRead, GRWrite, GatedAttention, SwiGLU,
                         apply_partial_rope, rope_cache, build_masks)
```

Symbols:

- [DecoderConfig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:35>) — lines 35–70
  Methods: [dc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:57>), [reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:60>), [attn_cfg](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:66>).
- [plan_index](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:74>) — lines 74–81: j*(t) = floor((t+1)/k) - 1 and validity (j* >= 0 and chunk j* in the same document as t).
- [PlanBuilder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:84>) — lines 84–111: P_t = W_u u_{j*} + W_c c_pred_{j*} + o_{r(t)} (null plan instead of W_u u + W_c c when invalid).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:86>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:95>).
- [PlanRead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:115>) — lines 115–138: Gated-residual read of {D, P}: x = 1/2 \[ g_D * (1 + gamma_D) * rms(D) + g_P * rms(P) \] g_D = sigmoid(U_D V_D rms(D) + b), g_P = f + (1 - f) * sigmoid(U_P V_P rms(P) + b), f = plan_floor. P has no learnable gain (a gain could learn
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:120>), [gates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:130>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:136>).
- [ConfinedLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:141>) — lines 141–176
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:142>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:161>).
- [StackLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:180>) — lines 180–193: One-stream gated residual, encoder S-layer anatomy (SWA + q/k/v convs + partial RoPE + gated output).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:182>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:189>).
- [Decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:197>) — lines 197–222
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:198>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:207>).
- [NextLat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:225>) — lines 225–250: Training-only latent dynamics (Teoh et al. 2025), horizon d = 1: h_hat_{t+1} = h_t + f_psi(LN\[h_t ; rms(E x_{t+1})\]) L_h = SmoothL1(sg\[h_{t+1}\], h_hat_{t+1}) L_KL = KL( p(. | sg h_{t+1}) || p(. | h_hat_{t+1}) ), head frozen (E detach
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:231>), [loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:239>).
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:253>) — lines 253–262: No decay: embeddings, Engram table, log_tau, conv kernels, biases, codebook, null plan, chunk offsets.
- [_randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:266>) — lines 266–269
- [_small](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:272>) — lines 272–274
- [run_unit_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_decoder.py:277>) — lines 277–380

## run_m2_25M/code/probe_multi.py

34 lines; SHA-256 `e0a8a90d41fa28fba692747a1e5fb34c2192858ee56240904f56b59543cdcb7b`.

Primary copy: [run_m2_25M/code/probe_multi.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/probe_multi.py>).

Where does the multi-token value get lost? Linear probes on a trained checkpoint.

Identical copies:

- [Decoder Design/decoder_v1_bundle/probe_multi.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/probe_multi.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/probe_multi.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/probe_multi.py>)
- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/probe_multi.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/probe_multi.py>)

Imports:

```python
import sys, torch, torch.nn.functional as F, harness
from e2e import HRMThinker, configs, make_batch_multi, MarkovFiller
from hrm_encoder import engram_indices
```

Symbols:

- [probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/probe_multi.py:23>) — lines 23–30

## Decoder Design/decoder_v1_bundle/summarize.py

16 lines; SHA-256 `6e69f941fb635dfa711caae9c0180826b9d4b9aac01d60c09b44ac75b52cf9ca`.

Primary copy: [Decoder Design/decoder_v1_bundle/summarize.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Decoder Design/decoder_v1_bundle/summarize.py>).

Collect every run in runs/ into one table (RESULTS_DECODER.txt is written from this output).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/summarize.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Decoder Design/decoder_v1_bundle/summarize.py>)

Imports:

```python
import json, glob, os
```

## Encoder Design/enc_check.py

206 lines; SHA-256 `61417cf9a6bfed313d4cc0119e5c47e09d3efa6259d5d26dfe59c12080a942f2`.

Primary copy: [Encoder Design/enc_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py>).

Prototype of the frozen-candidate encoder, written exactly from the spec, to test its claims. Tests: reach of L and G, causality, Engram init gradients, gate values at init, param count, document-boundary leakage, QK-norm logit ceiling.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Encoder Design/enc_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Encoder Design/enc_check.py>)

Imports:

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
```

Symbols:

- [ZCRMSNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:7>) — lines 7–12
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:8>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:10>).
- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:14>) — lines 14–14
- [ShortConv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:16>) — lines 16–27
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:17>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:19>).
- [GRRead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:29>) — lines 29–38
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:30>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:36>).
- [GRWrite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:40>) — lines 40–43
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:41>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:43>).
- [Attn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:45>) — lines 45–72
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:46>), [rope](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:52>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:60>).
- [SwiGLU](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:74>) — lines 74–76
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:75>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:76>).
- [Layer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:78>) — lines 78–90
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:79>), [_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:84>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:88>).
- [Engram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:92>) — lines 92–123
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:94>), [keys](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:104>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:117>).
- [Encoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:125>) — lines 125–145
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:126>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:135>).
- [randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/enc_check.py:147>) — lines 147–151

## Encoder Design/encoder_v1.py

237 lines; SHA-256 `39d8b1b9aef92f617074b8d7c9c3499bbdc1c8b00b562f9702858ff48ef2c0b4`.

Primary copy: [Encoder Design/encoder_v1.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py>).

HRM THINKER encoder v1.0 (frozen spec) - reference implementation + unit tests. Two role streams: L (local, bounded reach 63) feeds the decoder; G (global) feeds the concept core via CRC. Layer pattern S S S F S S S F; Engram after S1. Run `python encoder_v1.py` to execute the tests.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/Encoder Design/encoder_v1.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Encoder Design/encoder_v1.py>)

Imports:

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
```

Symbols:

- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:9>) — lines 9–10
- [ZCRMSNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:13>) — lines 13–19: Zero-centred RMSNorm: (1 + gamma) * x / rms(x); gamma starts at 0 and IS weight-decayed (Qwen3.8).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:15>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:17>).
- [ShortConv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:22>) — lines 22–34: Causal depthwise conv, kernel 4, dilation 1, residual form x + sum_i w_i * x_{t-i}. w starts at 0 (exact identity at init). Document-aware: never mixes across segment boundaries.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:25>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:28>).
- [GRRead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:37>) — lines 37–50: x = mean_i g_i * zcRMS(R_i), g_i = sigmoid(U_i V_i rms(R_i) + b_i); rank 32, b_i = +2 (gate ~0.88).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:39>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:45>).
- [GRWrite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:53>) — lines 53–58: s_i = 2 sigmoid(u_i^T x + a_i); u and a start at 0, so s_i = 1 exactly at init.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:55>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:57>).
- [Attn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:61>) — lines 61–92: Gated attention. SWA: window 8, partial RoPE (16/64). FULL: NoPE + learnable per-head temperature (no WD).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:63>), [rope](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:70>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:78>).
- [SwiGLU](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:95>) — lines 95–98
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:96>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:98>).
- [Layer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:101>) — lines 101–116: 'S': both sublayers read {L}, write {L,G}. 'F': both sublayers (attention AND FFN) read {L,G}, write {G} only.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:103>), [_sub](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:108>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:114>).
- [Engram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:119>) — lines 119–147: Orders {2,3}, 4 hash heads each, prime-row tables x 48 dims; context gate; conv k=4 dilation 1. Keys use canonical token ids (case + leading-space merged, built offline) and reset at document starts.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:122>), [keys](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:130>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:142>).
- [Encoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:150>) — lines 150–176
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:151>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:164>).
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:179>) — lines 179–184: No weight decay: Engram tables, FULL-layer temperatures, token embedding, conv kernels. Decay elsewhere (incl. zc gammas).
- [_randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:188>) — lines 188–191
- [run_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/encoder_v1.py:193>) — lines 193–233

## v4_tiny/code/hrm_encoder.py

455 lines; SHA-256 `145fc983015d269215e407421a96cce80f3deb4fe3e4a390880987ca42be57cd`.

Primary copy: [v4_tiny/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py>).

HRM THINKER - Encoder v1.0 (frozen). Reference implementation. Two role streams: L (local) : bounded reach (63 tokens at window 8). Only stream the decoder may read. G (global) : includes FULL-attention writes. Read only by the CRC -> concept core. Layer pattern "SSSFSSSF", Engram after layer index 0 (S1). Inputs (all int64, shape (B, T), T % chunk == 0): ids token ids doc document id per position (packing); positions of different docs never interact eng_idx (B, T, n_tables) Engram row indices, precomputed with `engram_indices` (CPU is fine) Outputs: tok (B, T, d) normalised L stream -> decoder token path c_hat (B, M, d_c) concepts on the sph

Identical copies:

- [v41_tiny/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/hrm_encoder.py>)
- [Encoder Design/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Encoder Design/hrm_encoder.py>)
- [run_m2_25M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/hrm_encoder.py>)
- [run_v2_19M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/hrm_encoder.py>)
- [run_v3_19M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/hrm_encoder.py>)
- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_encoder.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_encoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/Encoder Design/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/Encoder Design/hrm_encoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/hrm_encoder.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/hrm_encoder.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/hrm_encoder.py>)

Imports:

```python
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Symbols:

- [EncoderConfig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:25>) — lines 25–64
  Methods: [dc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:50>), [n_tables](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:54>), [reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:57>).
- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:68>) — lines 68–69
- [ZCRMSNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:72>) — lines 72–80: y = (1 + gamma) * x / rms(x); gamma initialised to 0 and weight-decayed.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:74>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:79>).
- [same_doc_shift_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:83>) — lines 83–87: (B, T, 1) bool: position t and t-i are in the same document (False for t < i).
- [ShortConv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:90>) — lines 90–106: Causal depthwise conv, kernel K, dilation 1, residual form: y_t = x_t + sum_{i=0}^{K-1} w_i * x_{t-i} (w initialised to 0 -> identity) Document-aware: x_{t-i} from another document contributes 0.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:94>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:100>).
- [GRRead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:109>) — lines 109–126: Gated-residual read over the allowed streams: n_i = rms(R_i), g_i = sigmoid(U_i V_i n_i + b_i), x = mean_i g_i * (1+gamma_i) * n_i
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:112>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:121>).
- [GRWrite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:129>) — lines 129–138: Per-stream write scale s_i = 2 * sigmoid(u_i . x + a_i); u, a initialised to 0 -> s_i = 1.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:131>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:137>).
- [rope_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:141>) — lines 141–145
- [apply_partial_rope](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:148>) — lines 148–151
- [GatedAttention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:154>) — lines 154–192: kind 'S': sliding window, partial RoPE. kind 'F': full causal, NoPE, per-head temperature.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:156>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:173>).
- [SwiGLU](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:195>) — lines 195–203
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:196>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:202>).
- [EncoderLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:206>) — lines 206–230: 'S': reads {L}, writes {L, G}. 'F': reads {L, G}, writes {G} (attention AND FFN).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:208>), [_apply](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:220>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:225>).
- [engram_indices](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:237>) — lines 237–256: (B, T) canonical ids + doc ids -> (B, T, n_tables) int64 row indices (already offset per table). Order-n key at t = (x_{t-n+1}, ..., x_t); a position in another document is replaced by boundary id 0, real ids are shifted by +1. Hash: multip
- [Engram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:259>) — lines 259–282
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:260>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:274>).
- [build_masks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:286>) — lines 286–294: Boolean (B, 1, T, T) masks, True = may attend. SWA: causal & same doc & t-j < window. FULL: causal & same doc.
- [Encoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:298>) — lines 298–344
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:299>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:325>).
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:347>) — lines 347–357: No weight decay: Engram table, FULL temperatures, token embedding, conv kernels, biases. Weight decay: all matrices and the zero-centred norm gammas (pulls effective scale to 1).
- [_randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:361>) — lines 361–365
- [run_unit_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:368>) — lines 368–423
- [mps_smoke_test](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/hrm_encoder.py:426>) — lines 426–448: Run on the MacBook: CPU vs MPS forward agreement, one backward pass, memory report.

## v10_1/model.py

314 lines; SHA-256 `093c2617c0d417861d67220d867590590f9767da02bc3e01093b5a549ad578f1`.

Primary copy: [v10_1/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py>).

Executable V10.1 specification. Reuses unchanged V10 encoder primitives. Default configuration is D+J+H+S: both CRC paths, normalized visit supervision, sampled chunk readout and utility-based code exploration. No EMA or memory writes occur in forward(). Source dependency hashes are recorded by validation tools.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_1/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_1/model.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/model.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/model.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/model.py>)

Imports:

```python
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.config import ModelConfig
from v10_10M.model import V10, RMSNorm, Block, Decoder, geometry
```

Symbols:

- [Architecture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:17>) — lines 17–29
  Methods: [arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:25>).
- [EncoderConceptCRC](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:32>) — lines 32–47
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:33>), [sources](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:42>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:45>).
- [Decoder101](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:50>) — lines 50–87: Reuse exact baseline parameters; expand only the CRC routers.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:54>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:66>).
- [ChunkReadout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:90>) — lines 90–104
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:91>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:98>).
- [V101](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:107>) — lines 107–212
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:108>), [candidates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:118>), [bind_memory_era](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:121>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:126>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:147>), [candidate_forecasts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:192>), [chosen_forecasts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:205>).
- [token_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:215>) — lines 215–217
- [latent_errors](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:220>) — lines 220–224
- [chunk_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:227>) — lines 227–256: One sampled boundary per eligible row; returns only eligible rows. targets\[:,i\] is x\[i+1\]. Current groups must be complete; future chunks may be partial and may end in EOS. Prompt targets remain context but not loss.
- [per_document_chunk_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:259>) — lines 259–262
- [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/model.py:265>) — lines 265–314

## v10_10M/config.py

39 lines; SHA-256 `2ad369624439af84700bee67eef1eb938dc5a30d0a82f6390aa05a151ec1d560`.

Primary copy: [v10_10M/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/config.py>).

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_10M/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_10M/config.py>)
- [v10_10M/runs/pretrain_24h/source_snapshot/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/config.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/config.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/config.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/config.py>)

Imports:

```python
from dataclasses import dataclass, asdict
```

Symbols:

- [ModelConfig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/config.py:5>) — lines 5–39
  Methods: [validate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/config.py:29>), [to_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/config.py:38>).

## v10_10M/memory.py

113 lines; SHA-256 `7cda2f9be8c8d5c4308819aae16283c6db7936cd0ecd324b06955f328be17d61`.

Primary copy: [v10_10M/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py>).

Exact, code-partitioned RBF correction; no writes during pretraining/evaluation.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_10M/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_10M/memory.py>)
- [v10_10M/runs/pretrain_24h/source_snapshot/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/memory.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/memory.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/memory.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/memory.py>)

Imports:

```python
from dataclasses import dataclass
import math
import threading
import torch
from .model import losses
```

Symbols:

- [Snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:10>) — lines 10–30
  Methods: [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:18>).
- [RBFMemory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:33>) — lines 33–86
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:34>), [snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:44>), [commit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:49>), [state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:76>), [from_state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:80>).
- [learn_episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/memory.py:89>) — lines 89–113: Prequential teacher-forced scoring, then frozen trajectory replay and commit. This is observed next-token adaptation, not a claim of free-generation success. No VQ loss, selector updates, imagined feedback or backbone optimizer involved.

## v10_10M/model.py

413 lines; SHA-256 `46997b31a7f28fa7039f5ab84b96794914f9af7b18105cce419f46a39474b238`.

Primary copy: [v10_10M/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py>).

Implementation of proposed_v10.md steps 1--32 (open choices in ARCHITECTURE.md). Each batch row is one document/window. Right padding only; document packing is deliberately unsupported. BOS is an observed token and counts toward groups. All candidate branches preserve earlier selected code history.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_10M/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_10M/model.py>)
- [v10_10M/runs/pretrain_24h/source_snapshot/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/model.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/model.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/model.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/model.py>)

Imports:

```python
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
```

Symbols:

- [RMSNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:15>) — lines 15–21
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:16>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:20>).
- [Attention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:24>) — lines 24–40
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:25>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:31>).
- [FFN](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:43>) — lines 43–49
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:44>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:48>).
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:52>) — lines 52–60
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:53>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:58>).
- [Reader](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:63>) — lines 63–84
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:64>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:79>).
- [TwoStreamBlock](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:87>) — lines 87–99
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:88>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:93>).
- [Engram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:102>) — lines 102–150: Orders 2/3, two hashes each, context gate and dilation-3 causal conv. CPU sparse lookup avoids dense 20M-entry gradient/optimizer traffic on MPS. Only gathered rows cross the device boundary, with autograd intact.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:108>), [indices](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:120>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:138>).
- [Pooler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:153>) — lines 153–177
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:154>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:164>).
- [geometry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:180>) — lines 180–191
- [ConceptCore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:194>) — lines 194–229
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:195>), [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:208>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:211>).
- [PQHead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:232>) — lines 232–252
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:233>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:239>), [fitting_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:246>).
- [Decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:255>) — lines 255–285
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:256>), [align](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:269>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:275>).
- [V10](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:288>) — lines 288–382
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:289>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:303>), [counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:309>), [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:317>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:336>), [candidates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:365>).
- [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/model.py:385>) — lines 385–413

## v10_2/memory.py

103 lines; SHA-256 `d2291282a28f9fbdc53b4df56a7591a57869875f1fdd1503479d6834670591d3`.

Primary copy: [v10_2/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py>).

Frozen-era exact RBF updates for both V10.2 downstream routes.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_2/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_2/memory.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_2/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/memory.py>)

Imports:

```python
import hashlib
import json
import math
import threading
from pathlib import Path
from dataclasses import asdict
import torch
from v10_10M.memory import Snapshot
from .model import token_loss
```

Symbols:

- [model_era](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:13>) — lines 13–24: Fingerprint actual weights, architecture and query/read semantics once per era.
- [RBFMemory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:27>) — lines 27–82
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:28>), [snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:35>), [commit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:38>), [state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:64>), [from_state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:70>).
- [write_objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:85>) — lines 85–86
- [learn_episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/memory.py:89>) — lines 89–103

## v10_2/model.py

198 lines; SHA-256 `cde5171748fbdd6e29674c65ca0d3f31a7be029f38b8f55f79cde3e01e2148d8`.

Primary copy: [v10_2/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py>).

V10.2 RF specification: completed concept history plus forecast-only utility.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_2/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_2/model.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_2/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/model.py>)

Imports:

```python
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.model import V10, Decoder, RMSNorm, FFN
from v10_1.model import V101, EncoderConceptCRC, token_loss, chunk_batch
```

Symbols:

- [Architecture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:11>) — lines 11–17
  Methods: .
- [ForecastReadout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:19>) — lines 19–28
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:20>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:26>).
- [history_geometry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:30>) — lines 30–40
- [ConceptReader](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:42>) — lines 42–60
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:43>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:49>).
- [Decoder102](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:62>) — lines 62–92
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:63>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:71>).
- [V102](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:94>) — lines 94–152
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:95>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:103>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:108>).
- [future_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:155>) — lines 155–160
- [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/model.py:162>) — lines 162–198

## v10_3_tiny/config.py

47 lines; SHA-256 `e792efbf033b06ce9c58daa0208005849697e65fb84420e907bfe06da7be5641`.

Primary copy: [v10_3_tiny/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/config.py>).

Tiny test-bed configuration for V10.3 RCDL. The V10 module classes read their sizes from a config object and call its validate(); v10_10M.config.ModelConfig hard-asserts width 128, so the tiny version supplies its own config with the same attribute names. The budget convention matches V10: the dense-parameter limit (here 600k) excludes the sparse Engram table, which is counted and reported separately.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/config.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/config.py>)

Imports:

```python
from dataclasses import dataclass, asdict
```

Symbols:

- [TinyConfig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/config.py:13>) — lines 13–47
  Methods: [validate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/config.py:38>), [to_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/config.py:46>).

## v10_3_tiny/evaluate.py

355 lines; SHA-256 `7e070c8bac781a3f80179281ef366d41f7912e536b55ae92583bdc32f072e568`.

Primary copy: [v10_3_tiny/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py>).

Deep evaluation of one trained tiny arm on fixed dev panels. python -B -m v10_3_tiny.evaluate --run-dir v10_3_tiny/runs/X8_s1 Panels are deterministic (seeded index choice, seeded text crops) and identical across arms, so per-document results can be paired. Writes evaluation.json in the run directory and refuses to overwrite it.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/evaluate.py>)

Imports:

```python
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_1.evaluate import score_math
from v10_1.model import chunk_batch, per_document_chunk_ce
from v10_2.model import future_ce
from .model import build, C0
from .config import TinyConfig
```

Symbols:

- [load_family](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:29>) — lines 29–32
- [example](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:35>) — lines 35–44
- [panels](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:47>) — lines 47–63: Scored and calibration panels; disjoint document indices within each family.
- [to](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:66>) — lines 66–67
- [doc_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:70>) — lines 70–73
- [stack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:79>) — lines 79–85
- [batch_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:88>) — lines 88–91
- [MeanSubstitution](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:94>) — lines 94–103: Replace the decoder's concept inputs (forecast F and state U) by calibration means.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:96>), [__enter__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:98>), [__exit__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:102>).
- [interventions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:106>) — lines 106–112
- [nll_and_interventions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:116>) — lines 116–150
- [grid_shift](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:154>) — lines 154–174: Sudoku has a constant layout (data_audit.json): re-score each board with `shift` extra header-newline tokens so every cell moves half a group. All arms see the same out-of-distribution edit; compare their degradation.
- [ridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:177>) — lines 177–180
- [fit_ridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:183>) — lines 183–186
- [fvu](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:190>) — lines 190–224
- [exploration](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:228>) — lines 228–281
- [greedy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:285>) — lines 285–292
- [generation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:295>) — lines 295–319
- [scales](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:322>) — lines 322–327
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/evaluate.py:330>) — lines 330–349

CLI arguments:

```python
p.add_argument('--run-dir', required=True)
p.add_argument('--checkpoint', default='final.pt')
p.add_argument('--device', default='mps')
p.add_argument('--output', default='evaluation.json')
p.add_argument('--threads', type=int, default=2)
```

## v10_3_tiny/losses.py

76 lines; SHA-256 `60bad110573d30cb04410a0083b46d2881938e19c8919eddd3d86efe847cdc9c`.

Primary copy: [v10_3_tiny/losses.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/losses.py>).

Objectives for the tiny arms. C0 token CE only. A v10_1.model.losses unchanged (latent MSE, codeword fitting, chunk head, hard-argmin exploration with selector CE). X token CE + 0.1 * forecast-only future CE on the committed codes (V10.2) + 0.1 * mixture likelihood on candidate updates (V10.3): ll_k = -sum_j mask_j * CE_{k,j} visit-2 forecast of code k log_pi = log_softmax(selector\[rows, at\]) prefix-only prior L_mix = -mean_doc( logsumexp_k(log_pi_k + ll_k) / n_doc ) Every one of the K branches carries gradient, weighted by its posterior responsibility; the selector is trained as the mixture prior rather than by cross-entropy on a hard winn

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/losses.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/losses.py>)

Imports:

```python
import torch
from torch.nn import functional as F
from v10_1.model import token_loss, chunk_batch
from v10_1.model import losses as losses_a
from v10_2.model import future_ce
```

Symbols:

- [mixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/losses.py:25>) — lines 25–47
- [losses_x](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/losses.py:50>) — lines 50–62
- [losses_c0](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/losses.py:65>) — lines 65–67
- [compute](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/losses.py:70>) — lines 70–76

## v10_3_tiny/model.py

119 lines; SHA-256 `4532f4ca55ddae53e0c0330248746547439bad7d74198674fa5127aac42e0602`.

Primary copy: [v10_3_tiny/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py>).

Tiny V10.3 RCDL and its comparison arms. Arms (see ARCHITECTURE.md): C0 encoder + Engram + decoder with encoder-depth CRC; no concept branch A the unmodified V10.1 model class (v10_1.model.V101) under TinyConfig X8, X4 V10.3: the V10.2 RF model with (i) a full learned map in place of the per-coordinate forecast gain and (ii) mixture-likelihood exploration (losses.py); the PQ codebook is excluded from weight decay (train.py) The V10.1/V10.2 classes are reused unchanged; only the forecast map and the tiny-config-safe constructor are new here.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/model.py>)

Imports:

```python
import torch
from torch import nn
from v10_10M.model import V10, Decoder
from v10_1.model import V101, Architecture as ArchitectureA
from v10_2.model import V102, Decoder102, ForecastReadout, history_geometry
from v10_1.model import EncoderConceptCRC
from .config import TinyConfig
```

Symbols:

- [C0](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:22>) — lines 22–44: Concept-free control: same encoder, Engram and decoder blocks with E1/E3 CRC.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:25>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:30>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:43>).
- [Decoder103](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:47>) — lines 47–84: V10.2 decoder with the aligned forecast entering through a full learned map. V10.1/V10.2 add `fusion * Align(F)` with a per-coordinate gain. §1.2 of the V10.3 design measured that gain shrinking while F itself carried 3% of its target's va
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:56>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:63>).
- [V103Tiny](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:87>) — lines 87–98: V10.2 RF forward with Decoder103; V102.__init__ is bypassed because it rejects any architecture other than its own default.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:91>).
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:101>) — lines 101–110
- [counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/model.py:113>) — lines 113–119

## v10_3_tiny/sleep.py

464 lines; SHA-256 `5addc8a2c566f1b24fb1657204fe2b870712605bce840b46b69058193e0ccbbb`.

Primary copy: [v10_3_tiny/sleep.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py>).

Sleep-layer tests for tiny V10.3 on the held-out algebra module (newmath). python -B -m v10_3_tiny.sleep --run-dir v10_3_tiny/runs/X8_s1 --out-dir v10_3_tiny/runs/sleep_X8_s1 Every condition starts from the same frozen pretrained checkpoint; the ones that train see the identical newmath stream and identical adapter init. base no change plain rank-4 LoRA on every attention (qkv, out) and FFN (up, down) matrix of E2, the concept core and the decoder gated the same adapters; each sub-layer's adapter output is multiplied per row by the Local-Support-Learning gate open(x) = log Phi_pos(x) > log Phi_neg(x) where x is the frozen model's own input to

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/sleep.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/sleep.py>)

Imports:

```python
import os
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_1.evaluate import score_math
from v10_2.memory import RBFMemory, learn_episode, model_era
from v10_10M.train import atomic_json
from .config import TinyConfig
from .model import build
from .losses import compute
from .compare import paired
from . import evaluate as E
```

Symbols:

- [LoRA](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:63>) — lines 63–75
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:64>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:71>).
- [sites](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:78>) — lines 78–83
- [ema](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:86>) — lines 86–90
- [density](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:93>) — lines 93–101
- [Controller](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:104>) — lines 104–133: Per-forward bookkeeping of sub-layer calls: collect inputs, record gates, apply gates.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:106>), [begin](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:111>), [hook](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:114>), [decide](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:129>).
- [set_enabled](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:136>) — lines 136–138
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:141>) — lines 141–149
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:152>) — lines 152–157
- [newmath_corpus](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:160>) — lines 160–164
- [cap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:167>) — lines 167–168
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:172>) — lines 172–181
- [fit_gates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:184>) — lines 184–201
- [install](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:204>) — lines 204–205
- [attach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:208>) — lines 208–213
- [schedule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:216>) — lines 216–217
- [adapt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:220>) — lines 220–242
- [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:246>) — lines 246–275
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:279>) — lines 279–301
- [bind](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:304>) — lines 304–307
- [write_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:310>) — lines 310–324
- [mean_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:328>) — lines 328–334
- [choose_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:337>) — lines 337–356: Predeclared calibration on TRAIN-split documents disjoint from the write stream (dev untouched): minimise held-out newmath loss subject to at most 0.01 nats/token harm on held-out pretraining-family documents.
- [contrasts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:359>) — lines 359–372
- [g4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:375>) — lines 375–390
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/sleep.py:393>) — lines 393–452

CLI arguments:

```python
p.add_argument('--run-dir', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--device', default='mps')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--seed', type=int, default=5)
p.add_argument('--steps', type=int, default=600)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--episodes', type=int, default=512)
p.add_argument('--calibration-episodes', type=int, default=96)
p.add_argument('--eta-grid', type=float, nargs='+', default=[.1, 10., 100., 1000.])
p.add_argument('--sigma-grid', type=float, nargs='+', default=[1., .3, .1])
p.add_argument('--consolidate', action='store_true')
```

## v10_3_tiny/tests.py

213 lines; SHA-256 `2259be24572165149b5d7dd24badf7591ac53689e3e5d1d1663d77916c443e73`.

Primary copy: [v10_3_tiny/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py>).

Correctness tests for the tiny V10.3 arms. Run: python -B -m v10_3_tiny.tests Each test prints PASS/FAIL with its measured quantity; the process exits non-zero if any test fails. Results are also written to tests_result.json.

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/tests.py>)

Imports:

```python
import json, math, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from v10_10M.data import Corpus
from v10_2.model import history_geometry
from .config import TinyConfig
from .model import build, counts, C0, V103Tiny
from .losses import compute, mixture
from .train import optimizers
from v10_1.model import chunk_batch
```

Symbols:

- [record](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:24>) — lines 24–26
- [fresh](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:29>) — lines 29–30
- [Float64Norm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:33>) — lines 33–42: Test-only: v10_10M RMSNorm computes its scale in float32 even for float64 inputs, which limits finite differences to float32 resolution. Inside this context the norm runs in the input dtype; production code is unchanged.
  Methods: [__enter__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:37>), [__exit__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:41>).
- [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:45>) — lines 45–46
- [test_counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:49>) — lines 49–51
- [test_roundtrip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:54>) — lines 54–59
- [test_optimizer_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:62>) — lines 62–71
- [test_causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:74>) — lines 74–87
- [test_release_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:90>) — lines 90–102
- [test_candidate_consistency](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:105>) — lines 105–116
- [test_mixture_gradient](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:119>) — lines 119–159
- [test_rbf_derivative](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:162>) — lines 162–178
- [test_mps_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:181>) — lines 181–197
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/tests.py:200>) — lines 200–209

## v10_3_tiny/train.py

154 lines; SHA-256 `a849bb0ab768367e3f7642a474c2aafc1d63d02eff8370ac85e34831e66db5f3`.

Primary copy: [v10_3_tiny/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py>).

Wall-time-bounded pretraining for one tiny arm. python -B -m v10_3_tiny.train --arm X8 --minutes 25 --run-dir v10_3_tiny/runs/X8_s1 --seed 1 Same schedule for every arm: AdamW (0.9, 0.95), warmup 200 updates, cosine to 10% over the wall budget, dense clip 1.0, CPU SparseAdam Engram at 5x LR with its own clip. Candidate (exploration) updates every --candidate-every updates for A and X arms. Optimizer grouping differs by arm on purpose: A and C0 use V10.1's rule (decay every dense parameter with ndim >= 2, which includes the PQ codebook); X arms exclude head.codebook from decay (V10.3 condition N3).

Identical copies:

- [V11_Paper/V10_3_Evidence/source/v10_3_tiny/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V11_Paper/V10_3_Evidence/source/v10_3_tiny/train.py>)

Imports:

```python
import argparse, hashlib, json, math, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids, atomic_json
from .config import TinyConfig
from .model import build, counts
from .losses import compute
```

Symbols:

- [optimizers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py:29>) — lines 29–37
- [source_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py:40>) — lines 40–44
- [quick_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py:48>) — lines 48–60
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py:63>) — lines 63–139
- [parser](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/train.py:142>) — lines 142–150

CLI arguments:

```python
p.add_argument('--arm', choices=['C0', 'A', 'X8', 'X4'], required=True)
p.add_argument('--run-dir', required=True)
p.add_argument('--minutes', type=float, default=120)
p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--context', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--candidate-every', type=int, default=4)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--eval-seconds', type=float, default=300)
p.add_argument('--max-steps', type=int, required=True)
```

## audit_v2_2026-09-28/audit.py

184 lines; SHA-256 `3668cf2e69739677e0ca897aaa681d5ce150974e6fc6605ed804da86e38c68e5`.

Primary copy: [audit_v2_2026-09-28/audit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/audit.py>).

Read-only inspection of run_m2_25M; writes findings only in this audit directory.

Imports:

```python
from pathlib import Path
from collections import Counter, defaultdict
import ast, datetime, hashlib, json, math, os, sys
import numpy as np
import torch
import v2_budget
```

Symbols:

- [dump](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/audit.py:15>) — lines 15–16

## audit_v2_2026-09-28/build_appendices.py

213 lines; SHA-256 `6e8c2536409ac6c75657c3084d69fe393f37105df2104cb2461d77df3d80222a`.

Primary copy: [audit_v2_2026-09-28/build_appendices.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/build_appendices.py>).

Render audit snapshots into human-readable tables; does not read live run state.

Imports:

```python
from pathlib import Path
from collections import Counter
import json
```

Symbols:

- [fmt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/build_appendices.py:13>) — lines 13–16
- [table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/build_appendices.py:18>) — lines 18–21
- [role](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/build_appendices.py:157>) — lines 157–185

## audit_v2_2026-09-28/diagnostics.py

67 lines; SHA-256 `e77d2e02033cb5f2545023c64f4aa46a50efedb5cc3ec6013d20563904252806`.

Primary copy: [audit_v2_2026-09-28/diagnostics.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/diagnostics.py>).

Small CPU-only demonstrations of review findings; never changes training.

Imports:

```python
from pathlib import Path
import json, sys, hashlib
import numpy as np
import torch
from model import build_model
from hrm_encoder import engram_indices
from tokenizers import Tokenizer
```

## audit_v2_2026-09-28/plot_evidence.py

48 lines; SHA-256 `b28d5919a267cba93851e5d97edf58519f38202e7ca3a931ff92a4077b15f9d1`.

Primary copy: [audit_v2_2026-09-28/plot_evidence.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/plot_evidence.py>).

Plot the frozen main evaluation; no model execution.

Imports:

```python
from pathlib import Path
import os,json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## draft_v3_review_2026-09-29/build_coverage.py

76 lines; SHA-256 `73d6e881fa14148a30e1bcde2892f42444ddf28ee3c8dcc7e88dc50a74b4b8b4`.

Primary copy: [draft_v3_review_2026-09-29/build_coverage.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/build_coverage.py>).

Imports:

```python
from pathlib import Path
import json, collections
```

Symbols:

- [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/build_coverage.py:29>) — lines 29–43

## draft_v3_review_2026-09-29/extract_papers.py

20 lines; SHA-256 `888dbb4ead1a47a80449e48deef04c151a75a654dda91cf2430905c94a2172fd`.

Primary copy: [draft_v3_review_2026-09-29/extract_papers.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/extract_papers.py>).

Imports:

```python
from pathlib import Path
import json,hashlib
from pypdf import PdfReader
```

## draft_v3_review_2026-09-29/inventory.py

26 lines; SHA-256 `c3750b674902c46d9632388c03785c91d4dffa8dedcdc9970e2fca3a6e1468cc`.

Primary copy: [draft_v3_review_2026-09-29/inventory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/inventory.py>).

Imports:

```python
from pathlib import Path
import os,json,hashlib,collections,datetime
```

## draft_v3_review_2026-09-29/probe_claims.py

46 lines; SHA-256 `e9ad8202d952c56e3eb07dfeeb913de5176f2759535ef0823a6e01bba6220071`.

Primary copy: [draft_v3_review_2026-09-29/probe_claims.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/probe_claims.py>).

Imports:

```python
import os,sys,json,math,copy
from pathlib import Path
import torch
import torch.nn.functional as F
from v3_ref import HRMThinkerV3, param_table, SCHED
from hrm_encoder import engram_indices,rms
from check_engram2 import reach
```

Symbols:

- [model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/probe_claims.py:11>) — lines 11–11
- [hook](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/probe_claims.py:15>) — lines 15–15

## draft_v3_review_2026-09-29/scan_repository.py

40 lines; SHA-256 `e733e65940f82a67b3941f4bbd7af16110fe91c9bf723ef3f9a44b7355b6e3e6`.

Primary copy: [draft_v3_review_2026-09-29/scan_repository.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/scan_repository.py>).

Imports:

```python
from pathlib import Path
import json,ast,hashlib,collections,zipfile,re,datetime
```

## draft_v3_review_2026-09-29/snapshot_evidence.py

59 lines; SHA-256 `c62bb55619abac74cc4efa34ff3ae08a76e7f9834154d83d4c7b08c123b8efb2`.

Primary copy: [draft_v3_review_2026-09-29/snapshot_evidence.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/snapshot_evidence.py>).

Read-only, low-memory evidence snapshot for the draft review.

Imports:

```python
from pathlib import Path
import json, hashlib, datetime, zipfile, collections
import numpy as np
```

## run_m2_25M/code/build_data.py

416 lines; SHA-256 `6c8319508ecac22d06b2771e043d5ef26b94454ba175a5e9c4e5b95d4be91d1d`.

Primary copy: [run_m2_25M/code/build_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py>).

Build tokenizer, canonical-id table, packed T = 1024 shards, evaluation sets and MANIFEST.json (PROMPT section 4-5). python build_data.py --out ../data \[--train_tokens 2.6e8\] Reuses the D-MIX design and generators (tasks.py, copied unchanged from the earlier programme) and the D-MIX raw parquet files (the same HF datasets the PROMPT names: roneneldan/TinyStories, Salesforce/wikitext 103-raw-v1). Mixture (by training tokens): language 45%, symbolic 22%, code 10%, retrieval 13%, long-range probes 10%. Deterministic: every generator draws from random.Random(f"{family}|{split}|{seed}").

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/build_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/build_data.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/build_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/build_data.py>)

Imports:

```python
import argparse, hashlib, json, os, random, re, sys, time, unicodedata
import numpy as np
import pandas as pd
import torch
import tasks as TK
import probes as PR
```

Symbols:

- [wiki_detok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:30>) — lines 30–34
- [wiki_articles](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:37>) — lines 37–52
- [tinystories](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:55>) — lines 55–56
- [sentences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:59>) — lines 59–65
- [train_tokenizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:69>) — lines 69–79
- [build_canonical_map](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:82>) — lines 82–91: Encoder spec 5.8: decode, strip one leading space marker, NFKC, lowercase; equal strings share an id. Special tokens are decoded with their names (<pad>, <eos>) so they keep distinct canonical ids (DECISIONS D-013).
- [token_bytes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:94>) — lines 94–95
- [pack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:99>) — lines 99–148: Greedy k-aligned packing into SEQ rows (D-MIX algorithm). Each doc (EOS included) is padded with PAD to a multiple of 4; padding keeps the doc id of the document it pads. Splittable (language) docs may be cut into 4-aligned fragments >= 64
- [same_doc_context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:151>) — lines 151–158: For each row: number of same-document tokens before position SEQ-16 (context of the scored last 16 positions).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py:162>) — lines 162–412

CLI arguments:

```python
ap.add_argument("--out", default=os.path.join(HERE, "..", "data"))
ap.add_argument("--train_tokens", type=float, default=2.6e8)
ap.add_argument("--vocab", type=int, default=8192)
ap.add_argument("--eval_only", action="store_true")
```

## run_m2_25M/code/data.py

108 lines; SHA-256 `28f7ffcbe7aaa3a445fbd6e981d477d84495526b98eebdde76144a153fac2d96`.

Primary copy: [run_m2_25M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py>).

Data loading: packed T = 1024 shards (uint16 tokens, int32 doc ids), Engram indices on CPU, NTP target mask. NTP mask (PROMPT 4.5): the target of position t is ids\[t+1\]; it counts only if doc\[t+1\] == doc\[t\] (same document) and ids\[t+1\] != PAD (padding is never a target; EOS is a real target).

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/data.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/data.py>)

Imports:

```python
import os, threading, queue
import numpy as np
import torch
from hrm_encoder import engram_indices
```

Symbols:

- [Canon](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:14>) — lines 14–19
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:15>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:18>).
- [make_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:22>) — lines 22–26
- [target_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:29>) — lines 29–31: (B, T-1) bool: which next-token targets count.
- [ShardStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:34>) — lines 34–81: Iterates the training shard in a fixed random row order (seeded); the cursor (rows consumed) is resumable. A background thread prepares the next batch (ids, doc, Engram indices) on CPU while the GPU works.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:38>), [_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:53>), [_work](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:59>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:67>), [epochs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:72>), [close](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:75>).
- [load_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:84>) — lines 84–86
- [pack_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py:89>) — lines 89–108: Pack evaluation items (each ends with EOS) into T-token rows, each item padded to a multiple of 4 with its own doc id. Returns tok (R, T), doc (R, T), and per item (row, offset).

## run_m2_25M/code/evaluate.py

247 lines; SHA-256 `1319e645ca8b4ae72a233810e333255b30eb79f74102343327d3f037fd45ad6f`.

Primary copy: [run_m2_25M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py>).

Evaluation benchmark (PROMPT section 5): language NTP (nats, bits/byte), item exact match (teacher-forced greedy), interventions (core off, plan swap, u off, c off, loop depth R, context truncation), codebook health, telemetry. Every function takes the model in eval mode (so ConceptVocab.vq_loss does not touch the restart counters).

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/evaluate.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/evaluate.py>)

Imports:

```python
import math, os, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from model import forward
from data import Canon, make_batch, target_mask, pack_items, PAD
```

Symbols:

- [Evaluator](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:17>) — lines 17–247
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:18>), [_to](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:26>), [_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:29>), [items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:38>), [lang_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:43>), [language](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:52>), [truncation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:71>), [item_accuracy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:98>), [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:131>), [codebook_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:145>), [telemetry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:176>), [loop_dynamics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:191>), [quick](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:223>), [_maint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:241>).

## run_m2_25M/code/model.py

106 lines; SHA-256 `5c6be4bbcc546e7caaf0c9e29e930fef8937928571678325850857f68f4746f2`.

Primary copy: [run_m2_25M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py>).

HRM THINKER-25M: encoder v1.0 (frozen) -> concept module v2 (looped core + NCP) -> decoder v1 (confined). The wiring is e2e.HRMThinker with configs() replaced by PROMPT.md section 2.1 (d = 384). The decoder receives only tok, u, c_pred, doc and E. This module adds nothing to the architecture; it only exposes the evaluation interventions of section 5.2 (core off, plan swap, u off, c off, loop depth R) on the same wiring.

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/model.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/model.py>)

Imports:

```python
import math
import torch
import torch.nn as nn
import compat
from hrm_encoder import EncoderConfig, Encoder, engram_indices
from concept_core_v2 import ConceptCoreV2
from hrm_decoder import DecoderConfig, Decoder, param_groups
from e2e import HRMThinker
```

Symbols:

- [configs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py:20>) — lines 20–26: PROMPT.md 2.1: EncoderConfig defaults except width/heads/ffn/vocab; decoder confined, 4 layers, w = 8, add.
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py:29>) — lines 29–32
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py:35>) — lines 35–46: Same wiring as HRMThinker.forward, plus the plan-swap intervention (u and c_pred of another row, roll(1, dim=0)). Returns logits (B,T,V) and aux (c_hat, cdoc, NCP logits, u, c_pred, h, P, encoder aux).
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py:49>) — lines 49–63: Section 2.2 table, read off the live model.
- [macs_per_token](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py:66>) — lines 66–85: Multiply-accumulates per token (weights + attention scores); training FLOPs = 6 x MACs.

## run_m2_25M/code/tasks.py

555 lines; SHA-256 `54e665389397b2d1ab3c6542f0f881f32491177400d61f213711d8682454b0ab`.

Primary copy: [run_m2_25M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py>).

Deterministic task generators for the HRM-CC vs Nanbeige-loop experiment (generator version v1). Every generator: gen(rng, split) -> Example(segments=\[(text, scored, label), ...\], family, meta). Documents are the concatenation of segment texts; scored segments are the evaluated answer spans (teacher-forced greedy exact match == all scored tokens argmax-correct). Training uses dense NTP on every token of every document (both models identically). Splits: 'train', 'val', 'test' share the in-distribution (ID) structural ranges and differ only by seed. 'ood' uses held-out structure: longer chains / deeper hops, larger graphs, unseen identifier a

Identical copies:

- [run_v2_19M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/tasks.py>)
- [run_v3_19M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/tasks.py>)
- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/tasks.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/tasks.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/tasks.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/tasks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/tasks.py>)

Imports:

```python
from __future__ import annotations
import json
import random
import string
from dataclasses import dataclass, field
```

Symbols:

- [Example](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:35>) — lines 35–42
  Methods: [text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:41>).
- [names_for](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:45>) — lines 45–46
- [gen_pointer_chase](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:50>) — lines 50–68: Follow a chain of pointers h hops. ID: 8-16 pointers, hops 1-4. OOD: 20-28 pointers, hops 5-8.
- [gen_graph_walk](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:71>) — lines 71–89: Labeled directed graph; follow a sequence of edge labels. ID nodes<=8, walk<=4; OOD nodes 10-16, walk 5-7.
- [gen_state_machine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:92>) — lines 92–109: Finite-state program over input symbols. ID states<=5, input len 2-6; OOD states 7-9, input 8-12.
- [gen_permutation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:112>) — lines 112–125: Apply a permutation r times to a list. ID size 3-5, r 1-3; OOD size 6-7 (unseen family), r 1-3.
- [gen_variable_binding](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:128>) — lines 128–150: Chains of assignments with copies and +/- constants. ID chain 1-4; OOD chain 5-8 or unseen names.
- [gen_variable_binding_newnames](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:153>) — lines 153–172: OOD-only: same as variable_binding (ID depth range) but with the unseen identifier alphabet B.
- [gen_kv_updates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:175>) — lines 175–189: set/get over a key-value store with overwrites. ID ops 4-12; OOD ops 20-30.
- [gen_relational](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:192>) — lines 192–205: Family-tree composition: 'X is the parent of Y'. Query the h-th descendant. ID h 1-3; OOD h 4-6.
- [gen_arith_transform](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:211>) — lines 211–236: start value then a sequence of ops mod 100. ID: 1-4 ops, 'mul' never directly after 'sub'. OOD: 3-6 ops, always containing the held-out pair 'sub' then 'mul' (unseen operator combination). Eval buckets OOD by meta.depth: 3-4 ops isolates th
- [gen_stack_ops](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:239>) — lines 239–255: push/pop sequence; answer = popped values in order then final stack. ID ops 3-10; OOD ops 14-20.
- [gen_nested_scopes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:258>) — lines 258–284: Nested blocks with shadowing; each print shows the innermost binding. ID nesting 1-3; OOD 4-6.
- [gen_brackets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:287>) — lines 287–303: Complete an unbalanced bracket prefix. ID open depth 1-6; OOD depth 8-12.
- [gen_seq_transform](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:306>) — lines 306–324: reverse / sort / dedupe a list of numbers. ID length 3-8; OOD length 12-16.
- [rand_ident](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:328>) — lines 328–338
- [gen_kv_retrieval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:341>) — lines 341–353: Many key: value pairs, then query one key. Distance (in pairs) between the queried pair and the query is recorded. ID pairs 4-24; OOD (longer context) pairs 40-60.
- [gen_copy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:356>) — lines 356–365: Copy a random string after a filler gap. The gap (filler sentences) controls distance.
- [gen_needle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:368>) — lines 368–378: A code inserted into filler prose; asked at the end. Needle depth (sentences after needle) varies.
- [gen_json_lookup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:381>) — lines 381–401: Nested JSON object; query a key path. ID depth 1-2; OOD depth 3-4.
- [gen_py_exec](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:405>) — lines 405–425: Tiny Python function + call; answer = printed value. ID 1-3 statements; OOD 4-6 statements.
- [gen_json_build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:428>) — lines 428–437: Emit a JSON object from a key=value description (syntactic exactness).
- [gen_yaml_like](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:440>) — lines 440–453: YAML-like config; query a nested field (identifier copy through structure).
- [gen_sort_trace](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:456>) — lines 456–468: Algorithm trace: bubble-sort passes over a small list (each pass is an answer step).
- [gen_expr_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:471>) — lines 471–483: Evaluate a nested arithmetic expression (mod 100). ID nesting 1-2; OOD 3-4.
- [gen_tool_traj](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py:487>) — lines 487–544: Deterministic simulated tools. GOAL -> CALL -> OBS -> ... -> ANSWER. Every CALL and the ANSWER are scored segments (next-action accuracy, final-state accuracy). With p=0.2 an injected error observation appears after the first call; the corr

## run_m2_25M/code/train.py

398 lines; SHA-256 `98449f1a52f70c1852ba88e8f0c0c3c64802522a184d4989505293979a1ada6b`.

Primary copy: [run_m2_25M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py>).

Train HRM THINKER-25M (PROMPT section 6). One process, one device; resumable. python train.py --run_dir ../runs/main --hours 24 --micro_bs 4 --accum 8 --total_steps N --warmup 300 python train.py --run_dir ../runs/main --resume ../runs/main/ckpt_last.pt Loss = NTP + 0.1 NCP + VQ (plain NTP on every valid position; no answer up-weighting). R ~ U{1..4} per optimizer step (same R for all micro-batches of the step), R = 4 at evaluation. Codebook: k-means++ init from the first real optimizer batch before the first training forward; dead-code restart every 100 optimizer steps after optimizer.step() (own counter; DECISIONS D-007). Guards (PROMPT 6.3

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_m2_25M/code/train.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/train.py>)

Imports:

```python
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward, param_groups, param_table
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator
```

Symbols:

- [now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:26>) — lines 26–27
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:30>) — lines 30–35
- [restart_dead_now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:39>) — lines 39–53: Copy of ConceptVocab.restart_dead without its `step % every` check (called every 100 optimizer steps).
- [rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:56>) — lines 56–60
- [set_rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:63>) — lines 63–66
- [save_ckpt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:69>) — lines 69–83
- [swap_used_mb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:86>) — lines 86–91
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py:94>) — lines 94–394

CLI arguments:

```python
ap.add_argument("--run_dir", required=True)
ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ap.add_argument("--device", default="mps")
ap.add_argument("--layout", default="confined")
ap.add_argument("--fixed_R", type=int, default=0)
ap.add_argument("--eval_R", type=int, default=4)
ap.add_argument("--micro_bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--lr_scale", type=float, default=1.0)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--total_steps", type=int, required=True)
ap.add_argument("--hours", type=float, default=24.0)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--eval_every_min", type=float, default=60.0)
ap.add_argument("--eval_every_steps", type=int, default=0)
ap.add_argument("--extra_eval_steps", default="")
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--V", type=int, default=8192)
ap.add_argument("--heartbeat", type=int, default=1)
ap.add_argument("--small", action="store_true")
ap.add_argument("--final_eval", action="store_true")
ap.add_argument("--no_final_eval", action="store_true")
ap.add_argument("--resume", default="")
ap.add_argument("--init_from", default="")
ap.add_argument("--lr_min_frac", type=float, default=0.1)
ap.add_argument("--warmup_start_frac", type=float, default=0.0)
ap.add_argument("--step_offset", type=int, default=0)
ap.add_argument("--mem_frac_guard", type=float, default=0.90)
ap.add_argument("--keep", type=int, default=3)
ap.add_argument("--div_min_wall_min", type=float, default=30.0)
ap.add_argument("--debug_force_divergence_step", type=int, default=0)
```

## run_v2_19M/code/concept_core_d6.py

219 lines; SHA-256 `91f76af24441beba40183cdfdc25196897425723fe26ce5f207b3596813b783c`.

Primary copy: [run_v2_19M/code/concept_core_d6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py>).

Concept module v2 = D6 LOOPFORMER-MoE-DS (model_v2/MODEL_V2_19M.md §6, concept_v2/DESIGN_D6.md). Four DISTINCT layers per pass, d 256, full causal attention over the chunk positions (doc-masked, NoPE, per-head temperature, sigmoid output gate, no q/k/v convs), each sublayer reads \[X~, S\] through the gated-residual read and writes the carry S. The FFN of every layer is top-2 of ONE shared pool of E SwiGLU experts (per-layer routers). Identity at init: exactly one zero-init per path (attention W_o, experts' W_2). The TRAINER owns the pass/window loop (per-window backward, detached carry, per-pass NCP, final-window consistency); this module ex

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_d6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_d6.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_d6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_d6.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_d6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/concept_core_d6.py>)

Imports:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import GRRead, GRWrite, ZCRMSNorm, rms, build_masks
from concept_core_v2 import ConceptVocab
```

Symbols:

- [CoreAttention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:24>) — lines 24–46: Gated full causal attention at concept rate: q/k zcRMSNorm, per-head temperature, NoPE, sigmoid output gate.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:26>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:38>).
- [ExpertPool](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:49>) — lines 49–59: E SwiGLU experts d -> de -> d stored as batched weights; W2 zero-init.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:51>), [expert_fn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:58>).
- [moe_dispatch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:62>) — lines 62–106: Top-k routing of the rows of x (N, d) over the shared pool with Switch-style capacity. Returns y (N, d), aux dict (lb, z, dropped fraction, load (E,), top-k indices (N, k)). dense=True evaluates every expert on every row and masks (referenc
- [gr_read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:109>) — lines 109–119: GRRead over \[X~, S\] with the X~ branch cached: X~ is constant within a window, so its gated term sigma(U V rms(X~) + b) * rms(X~) * (1 + gamma) is computed once per window per read module (same arithmetic).
- [CoreLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:122>) — lines 122–143
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:123>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:135>).
- [ConceptCoreD6](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:146>) — lines 146–219
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:147>), [init_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:164>), [one_pass](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:167>), [readout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:177>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:180>), [mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:192>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:196>), [ncp_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:210>), [losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/concept_core_d6.py:216>).

## v4_tiny/code/data.py

116 lines; SHA-256 `e05d9022bc2b43dcbe5ded15a96a2a57be55b0b6f10647572538fdaf277535cd`.

Primary copy: [v4_tiny/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py>).

Data loading: packed T = 1024 shards (uint16 tokens, int32 doc ids), Engram indices on CPU, NTP target mask. NTP mask (PROMPT 4.5): the target of position t is ids\[t+1\]; it counts only if doc\[t+1\] == doc\[t\] (same document) and ids\[t+1\] != PAD (padding is never a target; EOS is a real target).

Identical copies:

- [v41_tiny/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data.py>)
- [run_v2_19M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/data.py>)
- [run_v3_19M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/data.py>)
- [draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/data.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/data.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/data.py>)

Imports:

```python
import os, threading, queue
import numpy as np
import torch
from hrm_encoder import engram_indices
```

Symbols:

- [Canon](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:14>) — lines 14–19
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:15>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:18>).
- [make_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:22>) — lines 22–26
- [target_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:29>) — lines 29–31: (B, T-1) bool: which next-token targets count.
- [ShardStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:34>) — lines 34–89: Iterates the training shard in a random row order reshuffled every epoch (seed + epoch; v1 replayed epoch 1's order); the cursor (rows consumed, global) is resumable. A background thread prepares the next batch (ids, doc, Engram indices) on
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:39>), [order](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:55>), [_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:60>), [_work](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:67>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:75>), [epochs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:80>), [close](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:83>).
- [load_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:92>) — lines 92–94
- [pack_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data.py:97>) — lines 97–116: Pack evaluation items (each ends with EOS) into T-token rows, each item padded to a multiple of 4 with its own doc id. Returns tok (R, T), doc (R, T), and per item (row, offset).

## run_v2_19M/code/evaluate.py

257 lines; SHA-256 `b10e449941d614b2755ea86f11d901b020ade352c7276b7c0b0047d7a746ad2a`.

Primary copy: [run_v2_19M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py>).

Evaluation benchmark (PROMPT section 5): language NTP (nats, bits/byte), item exact match (teacher-forced greedy), interventions (core off, plan swap, u off, c off, loop depth R, context truncation), codebook health, telemetry. Every function takes the model in eval mode (so ConceptVocab.vq_loss does not touch the restart counters).

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/evaluate.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/evaluate.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/evaluate.py>)

Imports:

```python
import math, os, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from model import forward
from data import Canon, make_batch, target_mask, pack_items, PAD
```

Symbols:

- [Evaluator](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:17>) — lines 17–257
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:18>), [_to](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:26>), [_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:29>), [items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:38>), [lang_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:43>), [language](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:52>), [truncation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:71>), [item_accuracy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:98>), [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:131>), [codebook_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:145>), [telemetry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:176>), [loop_dynamics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:191>), [quick](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:233>), [_maint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/evaluate.py:251>).

## run_v2_19M/code/model.py

115 lines; SHA-256 `81e23c4d723679d2969cd392cb7d21ef3449be3a89a313f54ba24eb18af5c6ce`.

Primary copy: [run_v2_19M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py>).

HRM THINKER v2 (19M): encoder (v1 design, d 256, Engram 80M) -> concept module D6 (16 experts) -> confined decoder, plus the training-only NextLat head. Spec: run_m2_25M/model_v2/MODEL_V2_19M.md. forward(m, ids, doc, eidx, R, core_off, plan_swap, use_u, use_c) keeps v1's signature, so the v1 evaluation code ports.

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/model.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/model.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/model.py>)

Imports:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import compat
import speed
from hrm_encoder import EncoderConfig, Encoder, rms
from concept_core_d6 import ConceptCoreD6
from hrm_decoder import DecoderConfig, Decoder, NextLat
```

Symbols:

- [HRMThinkerV2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:23>) — lines 23–39
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:24>).
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:42>) — lines 42–45
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:48>) — lines 48–62: Evaluation / generation path (R passes, no windows). Returns logits (B,T,V) and aux.
- [nextlat_losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:65>) — lines 65–80: NextLat (Teoh et al.), horizon 1, on the decoder's final hidden state h = out_norm(D): h_hat_{t+1} = h_t + f(LN\[h_t ; rms(sg E x_{t+1})\]); L_h = SmoothL1(sg h_{t+1}, h_hat); L_KL = KL(softmax(sg logits_{t+1}) || softmax(h_hat E^T)) with E
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:83>) — lines 83–91: v1 convention (PROMPT 2.2): compute = encoder non-embedding + core without codebook + decoder.
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/model.py:94>) — lines 94–105: No decay: embeddings, Engram table, temperatures, conv kernels, biases, codebook, null plan, chunk offsets, norm gains (zcRMSNorm gammas are decayed in v1's encoder grouping; kept as v1 here) and LayerNorm params.

## run_v2_19M/code/train.py

529 lines; SHA-256 `9500e83e4ca79bfe57632f234c66f3a0c218792ace1ce47193d660ebf8dbafc2`.

Primary copy: [run_v2_19M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py>).

Train HRM THINKER v2 (19M). One process, one device; resumable; pause-aware; 48 h in ONE schedule (v1 lesson D-041). python train.py --run_dir ../runs/main --hours 48 --milestone_hours 24 python train.py --run_dir ../runs/main --resume ../runs/main/ckpt_last.pt Per optimizer step: K ~ U{8..12} core passes (same K for every micro-batch of the step), windows of 4 (remainder merged into the last), carry detached between windows, per-window backward (c_hat and tok are leaves whose gradients accumulate; one backward into the encoder at the end of the micro-batch). Loss = NTP + 0.1 * mean_r NCP(u_r) + 0.1 * consistency(final window) + VQ + 0.01 LB 

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/run_v2_19M/code/train.py>)

Imports:

```python
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward, nextlat_losses, param_groups, param_table
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator
```

Symbols:

- [now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:27>) — lines 27–28
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:31>) — lines 31–34
- [windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:37>) — lines 37–44: Pass ranges (1-based, inclusive) of the detached windows: windows of w, remainder merged into the last.
- [restart_dead_now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:48>) — lines 48–60
- [rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:63>) — lines 63–67
- [set_rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:70>) — lines 70–73
- [save_ckpt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:76>) — lines 76–89
- [phys_footprint_gib](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:92>) — lines 92–103: This process's physical memory footprint (macOS proc_pid_rusage ri_phys_footprint, includes compressed pages), the number Activity Monitor shows. -1 if unavailable.
- [swap_used_mb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:106>) — lines 106–111
- [micro_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:114>) — lines 114–180: One micro-batch of D6 training with per-window backward. Returns detached statistics (device tensors / floats).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/train.py:183>) — lines 183–525

CLI arguments:

```python
ap.add_argument("--run_dir", required=True)
ap.add_argument("--data_dir", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")))
ap.add_argument("--device", default="mps")
ap.add_argument("--micro_bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--lr_scale", type=float, default=1.0)
ap.add_argument("--lr_min_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--hours", type=float, default=48.0)
ap.add_argument("--milestone_hours", type=float, default=24.0)
ap.add_argument("--est_total_steps", type=int, default=8000)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--K_min", type=int, default=8)
ap.add_argument("--K_max", type=int, default=12)
ap.add_argument("--eval_R", type=int, default=6)
ap.add_argument("--lam_cons", type=float, default=0.1)
ap.add_argument("--lam_lb", type=float, default=0.01)
ap.add_argument("--lam_z", type=float, default=0.001)
ap.add_argument("--lam_h", type=float, default=1.0)
ap.add_argument("--lam_kl", type=float, default=1.0)
ap.add_argument("--amp", default="none")
ap.add_argument("--eval_every_min", type=float, default=60.0)
ap.add_argument("--eval_every_steps", type=int, default=0)
ap.add_argument("--log_every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--heartbeat", type=int, default=1)
ap.add_argument("--small", action="store_true")
ap.add_argument("--no_final_eval", action="store_true")
ap.add_argument("--quick_eval_n", type=int, default=128)
ap.add_argument("--resume", default="")
ap.add_argument("--mem_frac_guard", type=float, default=0.90)
ap.add_argument("--keep", type=int, default=2)
ap.add_argument("--div_min_wall_min", type=float, default=30.0)
ap.add_argument("--mem_restart_gib", type=float, default=24.0)
ap.add_argument("--mem_restart_growth_gib", type=float, default=12.0)
ap.add_argument("--debug_restart_step", type=int, default=0)
ap.add_argument("--dash_file", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "webapp", "data", "v2_live.json")))
```

## v3_design/check_engram2.py

91 lines; SHA-256 `b0153b451cab9dd287456a22bf95b69a9e71eaf9770a01b93951ea35cfcba1a7`.

Primary copy: [v3_design/check_engram2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py>).

Exact CPU check (float64, small config): v1 encoder + a second Engram read that shares the table. Read 2 sits after layer index 4 (first S of the second SSSF block), gathers the SAME table rows as read 1 (the n-gram key is depth-invariant) and has its own W_k, W_v (zero-init), gated read and write. Variant A: read 2 has NO short conv. Variant B: read 2 keeps a conv (kernel 4). Measures the L-stream reach by editing one token and finding the last changed position, and checks W_v2 = 0 => output identical to the one-read encoder at init.

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/v3_design/check_engram2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/v3_design/check_engram2.py>)

Imports:

```python
import copy, math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, Engram, GRRead, GRWrite, ShortConv, rms, build_masks, rope_cache,
                         engram_indices)
```

Symbols:

- [EngramRead2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:15>) — lines 15–33
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:16>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:27>).
- [Encoder2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:36>) — lines 36–57
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:37>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:42>).
- [reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/check_engram2.py:60>) — lines 60–66

## v3_design/v3_budget.py

148 lines; SHA-256 `4704f913f50d801e6fe1fafe1a86d67cea3fc50a9290e2ca252bd75ab3ccf60a`.

Primary copy: [v3_design/v3_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py>).

HRM Thinker v3 (d 256): params, MACs and a throughput range for every design option. CPU only; never touches the GPU or any run directory. NOTE: rows here use core ff 768 (pre-trim); the final design uses 736 -- exact count in v3_ref.py (18,986,871). The encoder and decoder are built from the real v1 classes (run_m2_25M/code) with a tiny Engram table (the table is added analytically). The v3 concept core is counted analytically (it does not exist yet). Parameter convention = run_m2_25M/code/model.py:param_table: compute = encoder non-embedding + concept module without codebook + decoder (embedding, Engram table, codebook excluded). Throughput

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/v3_design/v3_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/v3_design/v3_budget.py>)

Imports:

```python
import json, os, sys
import torch
from hrm_encoder import EncoderConfig, Encoder
from hrm_decoder import DecoderConfig, Decoder
```

Symbols:

- [fit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py:27>) — lines 27–30
- [encoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py:33>) — lines 33–48
- [decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py:51>) — lines 51–58
- [core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py:61>) — lines 61–99: v3 concept core: n_layers distinct FULL-attention layers (gated attention, NoPE, tau, q/k/v convs as v1's F layer) wrapped in GR read {C, S} / GR write {S}, each with its own SwiGLU d_ff; optional shared SwiGLU (hidden shared_ff) used by ev
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_budget.py:102>) — lines 102–117

## v3_design/v3_ref.py

228 lines; SHA-256 `7536a2dfc2c98c6152013fb545f2f72995f2b691c9a7b80473955bb020632ba3`.

Primary copy: [v3_design/v3_ref.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py>).

HRM Thinker v3 — reference implementation of the draft_v3.md architecture (CPU, for exact counts and tests). Encoder: v1 hrm_encoder at d 256, SSSFSSSF, F layers with 6 heads (n_heads_full), Engram read 1 (after layer 0, with conv) and read 2 (after layer 4, shared table, no conv, W_v zero-init). Core: 4 distinct F-anatomy layers over {C, S} with per-stream (t, dt)-table read gains and write gains, own SwiGLU + shared SwiGLU (parallel, w2 zero-init), S0 = 0, trajectories on the grid of 1/3 (main M=3, shortcuts 2a, 2b, 1). NCP head / PQ 32x128x8 = v1's ConceptCoreV2 pieces. Decoder: v1 confined decoder at d 256, ff 1024. Not a trainer; no GPU.

Identical copies:

- [draft_v3_review_2026-09-29/source_snapshot/v3_design/v3_ref.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v3_review_2026-09-29/source_snapshot/v3_design/v3_ref.py>)

Imports:

```python
import copy, math, os, sys
from dataclasses import replace
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms, build_masks,
                         engram_indices)
from hrm_decoder import DecoderConfig, Decoder
from concept_core_v2 import ConceptVocab
from check_engram2 import Encoder2
```

Symbols:

- [encoder_v3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:26>) — lines 26–32
- [gr_read_mod](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:35>) — lines 35–41: v1 GRRead with an extra per-stream multiplicative gain (1 + mod_i); mod_i = 0 -> exactly v1.
- [CoreLayerV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:44>) — lines 44–59
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:45>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:52>).
- [ConceptCoreV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:62>) — lines 62–103
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:63>), [mods](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:79>), [one_pass](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:84>), [trajectory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:89>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:98>).
- [HRMThinkerV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:106>) — lines 106–135
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:107>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:124>).
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:138>) — lines 138–150
- [tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/v3_ref.py:154>) — lines 154–213

## run_m2_25M/code/calibrate.py

84 lines; SHA-256 `e73db5d7c1205626a5ed97e35e090a43d1ac7cf69df400f41418c8b1adf31b24`.

Primary copy: [run_m2_25M/code/calibrate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/calibrate.py>).

Calibration on MPS (PROMPT 6.1): micro-batch B in {4, 2, 1} at T = 1024; 10 warm-up + 30 timed micro-steps each, with real optimizer steps every `accum` micro-steps (accum = 32 / B so that a step sees 32,768 tokens). Keeps the largest B whose peak allocated memory is <= 75% of torch.mps.recommended_max_memory() and causes no swap growth. Writes runs/calibration.json. Each B runs in a fresh subprocess (one MPS job at a time, clean allocator).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/calibrate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/calibrate.py>)

Imports:

```python
import json, os, subprocess, sys, time
```

Symbols:

- [swap_mb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/calibrate.py:11>) — lines 11–13
- [one](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/calibrate.py:16>) — lines 16–60

## run_m2_25M/code/check_data.py

66 lines; SHA-256 `4325208d5b15d2268830521a4b57c8bb691d6423811e5a02ff821da39bf49a20`.

Primary copy: [run_m2_25M/code/check_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/check_data.py>).

Data checks before training (PROMPT 4.6): every document (and padding run) spans a multiple of 4 positions and starts at a multiple of 4; every probe's evidence-to-answer distance is > 90 tokens (>= the 128 floor); 3 decoded samples per family (test ID and OOD) written to runs/data_samples.md.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/check_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/check_data.py>)

Imports:

```python
import json, os, random, sys
import numpy as np
import torch
from tokenizers import Tokenizer
```

## v4_tiny/code/compat.py

22 lines; SHA-256 `d59fb50a6c17d7dfadfa5eadb0033960c235d88a5f0036e99418544c70506384`.

Primary copy: [v4_tiny/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/compat.py>).

Run-folder compatibility shim (the reference files stay byte-identical; see DECISIONS.md D-001). hrm_encoder.EncoderLayer defines a helper named `_apply(L, G, y, s)`, which shadows torch.nn.Module._apply(fn). Module.to(device) / .float() / .double() call child._apply(fn), so moving any model that contains an EncoderLayer raises TypeError. This shim renames the helper to `_write_streams` and rebinds forward to it. The arithmetic is unchanged token for token (verified bit-exact against the original forward in float64, see runs/tests/).

Identical copies:

- [v41_tiny/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compat.py>)
- [run_m2_25M/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/compat.py>)
- [run_v2_19M/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/compat.py>)
- [run_v3_19M/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/compat.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/compat.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/compat.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/compat.py>)

Imports:

```python
import hrm_encoder as _E
```

## run_m2_25M/code/dmix_build_reference.py

334 lines; SHA-256 `d27a039d91ea2f7525d971511cc3bb4e37e3726f5642ce0cb63ffe1e473e8378`.

Primary copy: [run_m2_25M/code/dmix_build_reference.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py>).

Build tokenizer, packed train/val/test rows and evaluation sets (deterministic). python dataset/generators/build.py --stage all Outputs under dataset/: tokenizer.json, train/rows_{tok,doc}.u16/u8, validation/, test/, eval/*.pt, DATASET_MANIFEST.json. Everything is seeded; see mixture.json for proportions.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/dmix_build_reference.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/dmix_build_reference.py>)

Imports:

```python
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import tasks as T
```

Symbols:

- [load_mixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:35>) — lines 35–36
- [wiki_detok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:40>) — lines 40–45
- [wiki_articles](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:48>) — lines 48–63
- [tinystories](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:66>) — lines 66–67
- [sentences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:70>) — lines 70–76
- [train_tokenizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:80>) — lines 80–90
- [encode_example](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:93>) — lines 93–101
- [make_generator](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:105>) — lines 105–111
- [gen_examples](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:114>) — lines 114–120
- [pack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:124>) — lines 124–169: docs: list of np.uint16 arrays (EOS included). Greedy chunk-aligned packing into SEQ rows. Splittable (language) docs may be cut into k-aligned fragments (>=64 tokens); others never cross a row.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py:173>) — lines 173–330

CLI arguments:

```python
ap.add_argument("--train_tokens", type=float, default=90e6)
ap.add_argument("--vocab", type=int, default=8192)
ap.add_argument("--n_eval", type=int, default=400)
```

## run_m2_25M/code/extend_ckpt.py

10 lines; SHA-256 `c0bf5aaa0963197799b719054e18ee6181200ea5efc796a7dde447dd90bb9295`.

Primary copy: [run_m2_25M/code/extend_ckpt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/extend_ckpt.py>).

Change the cosine length stored in runs/main/ckpt_last.pt (D-033). os.replace breaks the hard link, so the ckpt_stepNNNNNN.pt file stays untouched as the fallback.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/extend_ckpt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/extend_ckpt.py>)

Imports:

```python
import os, sys, torch
```

## run_m2_25M/code/final_eval.py

305 lines; SHA-256 `265d6018da826b51f1e21ddf901bc215bca9b6c8b28f145664944a480e8f1924`.

Primary copy: [run_m2_25M/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py>).

Detailed testing after training (PROMPT section 7) on one checkpoint. python final_eval.py --ckpt ../runs/main/ckpt_last.pt --out ../runs/main/eval_final.json \[--device mps\] \[--parts ...\] Parts (all by default): lang test language (TinyStories-test, WikiText-test; 256 rows each): nats, bits/byte; interventions core_off, plan_swap, u_off, c_off; loop depth R in {1,2,3,4,6,8} trunc context truncation r in {72,128,256,512,1024} on the last 16 positions + plan-swap cost (G-D2) items synthetic families (200/family) and probes (512/family), test ID and OOD; interventions on every suite probes linear probes: concept -> its 4 tokens (G-E1); u at 

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/final_eval.py>)

Imports:

```python
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward
from evaluate import Evaluator
from data import make_batch, pack_items, target_mask, Canon, PAD, EOS
from probes import PROBES
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:32>) — lines 32–42
- [part_lang](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:45>) — lines 45–58
- [part_trunc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:61>) — lines 61–67
- [part_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:70>) — lines 70–95
- [fit_probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:100>) — lines 100–121: Multinomial logistic regression, 80/20 split, standardised features; returns held-out top-1 accuracy.
- [part_probes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:125>) — lines 125–203
- [part_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:206>) — lines 206–215
- [part_gen](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:219>) — lines 219–257
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py:260>) — lines 260–301

CLI arguments:

```python
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--parts", default="lang,trunc,items,probes,health,gen")
ap.add_argument("--probe_only_interv", action="store_true")
ap.add_argument("--R", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
```

## run_m2_25M/code/main_steps.py

15 lines; SHA-256 `3bf6cd54858ff13353fc03b28952e5135f6f9a510aab5dd35d208aa5d3096bdc`.

Primary copy: [run_m2_25M/code/main_steps.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/main_steps.py>).

Main-run cosine length from screen S1's measured speed (PROMPT 6.1 step 3; DECISIONS D-021): total_steps = floor((24 h - 24 x eval_sec - 24 x ckpt_sec) / sec_per_step), sec_per_step from S1's steady-state tok/s.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/main_steps.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/main_steps.py>)

Imports:

```python
import json, sys, statistics
```

## run_m2_25M/code/plots.py

58 lines; SHA-256 `89a1007c9521765339123207e455c23c31a3364eac39d21f350eefd6e47f53af`.

Primary copy: [run_m2_25M/code/plots.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/plots.py>).

Learning curves (PROMPT 7.4) from runs/<run>/train_log.jsonl and log.jsonl -> runs/<run>/figs/*.png. python plots.py ../runs/main \[../runs/screens/S1 ...\] (screens are overlaid on the language panel)

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/plots.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/plots.py>)

Imports:

```python
import json, os, sys
import matplotlib
import matplotlib.pyplot as plt
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/plots.py:11>) — lines 11–12
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/plots.py:15>) — lines 15–54

## v4_tiny/code/probes.py

180 lines; SHA-256 `07f18d204a23cc80b07a5e89a7c9ea129204c00ef8a81145d4625fce5c4f5e8a`.

Primary copy: [v4_tiny/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py>).

Long-range probe families (PROMPT 4.4). Every item puts its evidence at least FLOOR tokens before the answer (FLOOR = 128 > 90 required; chosen above the S2 stack decoder's reach of 103, DECISIONS D-011). Documents are assembled from pre-encoded token segments, so a name or value has exactly the same tokens in the fact and in the answer (each segment is encoded once, with its leading space, and reused). Distances are measured in tokens on the assembled document and asserted. Item = dict(ids, spans=\[(start, end, label)\], family, meta, dist) with ids ending in EOS; spans index the answer tokens (the prediction for token i is logits\[i-1\]).

Identical copies:

- [v41_tiny/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/probes.py>)
- [run_m2_25M/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/probes.py>)
- [run_v2_19M/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/probes.py>)
- [run_v3_19M/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/probes.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/probes.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/probes.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/probes.py>)

Imports:

```python
import random
```

Symbols:

- [ProbeFactory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:31>) — lines 31–159: Holds the tokenizer-dependent pieces (encoded filler sentences, names, values) for one split.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:34>), [e](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:44>), [filler_sents](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:50>), [filler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:58>), [gap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:61>), [lr_recall1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:66>), [lr_carry3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:69>), [lr_multi4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:72>), [_facts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:75>), [lr_hop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:102>), [lr_bind](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:127>).
- [probe_text_for_tokenizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/probes.py:165>) — lines 165–180: Plain-text version of a probe (for tokenizer training only, before the tokenizer exists).

## run_m2_25M/code/report_tables.py

115 lines; SHA-256 `41a9a1e369559be08b26edbeb8d4c6d09917fa243f7f10278a2eaba283c7a73b`.

Primary copy: [run_m2_25M/code/report_tables.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py>).

Generate the numeric tables for REPORT.md directly from run files (so every number traces to a file). python report_tables.py > ../runs/report_tables.md

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/report_tables.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/report_tables.py>)

Imports:

```python
import json, os, glob
import numpy as np
```

Symbols:

- [rd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:10>) — lines 10–11
- [jl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:14>) — lines 14–15
- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:18>) — lines 18–19
- [screens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:22>) — lines 22–42
- [main_curve](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:45>) — lines 45–57
- [final](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py:60>) — lines 60–110

## run_m2_25M/code/test_compat.py

14 lines; SHA-256 `fb1a332b9d9d31c11449bff5c2380c85978721a0d625daf5420dd3f3150ed1cb`.

Primary copy: [run_m2_25M/code/test_compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/test_compat.py>).

Verify the compat shim: (1) forward bit-identical to the original in float64; (2) run hrm_encoder.mps_smoke_test.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/test_compat.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/test_compat.py>)

Imports:

```python
import torch, hrm_encoder as E
import compat
```

## run_m2_25M/code/test_p5.py

71 lines; SHA-256 `01e69286cc5b364e10ab2c2055a2b114302c7ba4db5cecff0fd1820ccd9ce8b7`.

Primary copy: [run_m2_25M/code/test_p5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/test_p5.py>).

P5 (d = 384 config): rerun pipeline tests P1-P3 with the PROMPT 2.1 configuration at small V and T, float64, randomised weights (so zero-initialised paths cannot hide a leak), exact equality. Also usable on trained weights (section 7.1): `python test_p5.py --ckpt path.pt` loads a checkpoint (V = 8192, T = 1024 row from the test split is replaced by random ids) and reruns P1-P3 without randomising.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/test_p5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/code/test_p5.py>)

Imports:

```python
import argparse, sys
import torch
from model import build_model, forward, engram_indices
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/test_p5.py:12>) — lines 12–41

CLI arguments:

```python
ap.add_argument("--ckpt", default=None)
ap.add_argument("--V", type=int, default=96)
ap.add_argument("--T", type=int, default=512)
```

## run_m2_25M/concept_v2/core_budget.py

59 lines; SHA-256 `5ece45ea4b8ca50393d775a2abef7ec071cfdcce9b27b27222513e84165ad2f7`.

Primary copy: [run_m2_25M/concept_v2/core_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py>).

Analytic params / MACs for the five concept-core designs (CPU only, no torch). Conventions match code/model.py:macs_per_token: MACs per TOKEN; the core runs once per 4-token chunk (M = T/4 = 256 chunk positions), so per-chunk costs are divided by 4. NON_CORE = encoder + decoder + LM head of the current model. C7 share = core / (NON_CORE + core), scored at the design's stated INFERENCE R (advisor #22).

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/concept_v2/core_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/concept_v2/core_budget.py>)

Symbols:

- [layer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py:14>) — lines 14–30: One core layer at chunk rate. Returns (params, macs_per_chunk_position).
- [core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py:33>) — lines 33–41: Per-token MACs and params of a core that runs `passes` cycles of `n_layers` layers.
- [share](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py:44>) — lines 44–45
- [report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py:48>) — lines 48–50

## run_m2_25M/model_v2/make_arch_v2.py

114 lines; SHA-256 `7ec441e6bdc364b5b8328eed1c272ee80892403905461a07258c1ac66c7d80b5`.

Primary copy: [run_m2_25M/model_v2/make_arch_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/make_arch_v2.py>).

Emit model_v2/arch_v2.json: the v2 architecture in the webapp's data/arch.json shape (params / table / macs / ecfg / dcfg / reach / S / N) plus the blocks arch.json has no slot for (core, nextlat, losses, nodes, edges, layout3d). Numbers come from v2_budget.build() for the recommended row C. CPU only. Run: python3 model_v2/make_arch_v2.py

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/model_v2/make_arch_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/model_v2/make_arch_v2.py>)

Imports:

```python
import json, os, sys
from v2_budget import build, D, H, KV, DH, DC, PQ_S, PQ_N, PQ_DS, ENG_ROWS, ENG_DIM, ENG_TABLES, V, T, M
```

## run_m2_25M/model_v2/v2_budget.py

155 lines; SHA-256 `4b65f0f702c56ca69d46869e686208984f71658c909ea419245892dc39d68db7`.

Primary copy: [run_m2_25M/model_v2/v2_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py>).

HRM Thinker v2 (d 256, Engram 80M, concept module D6, decoder + NextLat): params and MACs per token. CPU only, never touches runs/ or the GPU. Encoder and decoder are instantiated from the real code at d 256 with a tiny Engram table (the table is added analytically), the D6 core is counted analytically (it is not implemented yet), and the non-core MACs reuse code/model.py:macs_per_token. Parameter convention = code/model.py:param_table ("25M" of today): compute = encoder non-embedding + concept module without codebook + decoder; embedding, Engram table, codebook excluded. Writes model_v2/v2_budget.json. Run: python3 model_v2/v2_budget.py

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/model_v2/v2_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/model_v2/v2_budget.py>)

Imports:

```python
import json, math, os, sys
import torch
from hrm_encoder import EncoderConfig, Encoder
from hrm_decoder import DecoderConfig, Decoder, NextLat
from model import macs_per_token
```

Symbols:

- [encoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py:27>) — lines 27–38
- [decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py:41>) — lines 41–49
- [d6_core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py:52>) — lines 52–77: D6 LOOPFORMER-MoE-DS at d 256: 4 distinct layers, full causal attention 4 x 64 (GQA kv 2), sigmoid output gate, GR read over \[X~, S\] (rank 32), shared pool of E SwiGLU experts x d_expert, top-k, per-layer routers, zero-init block outputs; S
- [nextlat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py:80>) — lines 80–87
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py:90>) — lines 90–120

## run_m2_25M/webapp/make_arch_v3.py

63 lines; SHA-256 `c0a36a0c337762002909d77afa17db63d3cf0bd808963e24b7c6c0d5f8ae4c86`.

Primary copy: [run_m2_25M/webapp/make_arch_v3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/make_arch_v3.py>).

Build the static proposed-v3 page data from checked architecture and review artifacts.

Imports:

```python
from pathlib import Path
import json, shutil
```

## run_m2_25M/webapp/phone_server.py

103 lines; SHA-256 `db85038ab2f5c5b6d134958c7438e361738a026a05c0fa06d5e6a706a99f2433`.

Primary copy: [run_m2_25M/webapp/phone_server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py>).

Read-only phone gateway for the dashboard (LAN). Serves this webapp folder on 0.0.0.0:8780 so a phone on the same Wi-Fi can open it, and forwards the live endpoints of the two existing localhost-only servers: /api/* -> 127.0.0.1:8765/api/* and /v3api/live -> 127.0.0.1:8766/api/live. GET / HEAD only. Access needs the key in .phone_key: open http://<mac-ip>:8780/?k=<key> once; a cookie keeps it. Never touches a training run. Stop with: kill $(cat webapp/.phone_server.pid)

Imports:

```python
import http.server, json, os, secrets, socketserver, sys, urllib.request
```

Symbols:

- [Handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:20>) — lines 20–91
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:21>), [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:24>), [authorised](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:27>), [deny](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:33>), [proxy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:39>), [end_headers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:60>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:68>), [do_HEAD](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:82>), [do_POST](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:89>).
- [Server](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/phone_server.py:94>) — lines 94–96
  Methods: .

## run_m2_25M/webapp/server.py

650 lines; SHA-256 `4c6d0cfa698f3093a70d9ec1abcaf6b1c570c95c9d65c42b804c1de2ab2410dc`.

Primary copy: [run_m2_25M/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py>).

HRM THINKER-25M dashboard server (read-only, stdlib only; D-036). nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 & open http://127.0.0.1:8765 Never writes under runs/, never imports torch, never touches the GPU. Log files are read incrementally (byte offset per file) and the assembled JSON is rebuilt only when a watched file changed.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_m2_25M/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_m2_25M/webapp/server.py>)

Imports:

```python
import json, os, re, subprocess, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
```

Symbols:

- [Tail](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:20>) — lines 20–57: Incremental JSONL / text reader: keeps a byte offset and the unterminated trailing fragment.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:22>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:26>).
- [cont_active](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:77>) — lines 77–78
- [active_dir](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:81>) — lines 81–82
- [cont_sched](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:85>) — lines 85–92: Schedule of the continuation from its config.json (step_offset = global step of its step 0).
- [update_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:95>) — lines 95–98
- [shifted](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:101>) — lines 101–111: Continuation rows on the global step axis (train rows carry gstep; restart and eval rows are shifted here).
- [shift_line_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:114>) — lines 114–115
- [sh](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:126>) — lines 126–130
- [system_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:133>) — lines 133–161: Cheap OS probes, at most once every 20 s.
- [_load_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:168>) — lines 168–175
- [_sig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:178>) — lines 178–182
- [sample_live](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:185>) — lines 185–227: Called every 5 s by the sampler thread: GPU, trainer process, swap, battery (30 s), signatures, heartbeat.
- [sampler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:230>) — lines 230–236
- [live_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:239>) — lines 239–260
- [slim_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:263>) — lines 263–275
- [run_status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:278>) — lines 278–314
- [events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:317>) — lines 317–330: Resume / pause / guard / schedule events from the trainer output, for chart markers.
- [screens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:333>) — lines 333–344
- [plan_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:350>) — lines 350–378: Two-stage plan for the dashboard: stage 1 = main run (+ its final evaluation), stage 2 = continuation (D-041).
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:381>) — lines 381–420
- [final_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:423>) — lines 423–442: Final evaluations (D-044): full eval_final.json of the main run and, once it exists, of the continuation.
- [_f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:451>) — lines 451–452
- [build_digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:455>) — lines 455–529: Plain-text digest of every page's data (architecture, screens, main run, final eval, gates, continuation).
- [run_analysis](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:551>) — lines 551–568
- [analysis_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:571>) — lines 571–585
- [H](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:588>) — lines 588–638
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:589>), [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:592>), [end_headers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:595>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py:600>).

## run_v2_19M/code/final_eval.py

295 lines; SHA-256 `aa79e31100c53ae4e03bccfb7f6e4fe353da023bec260f70cef5d502d37f11f0`.

Primary copy: [run_v2_19M/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py>).

Detailed testing (v2: at the 24 h milestone and at 48 h) on one checkpoint; v1's suite with R sweep to 16 and D6 loop dynamics. python final_eval.py --ckpt ../runs/main/ckpt_last.pt --out ../runs/main/eval_final.json \[--device mps\] \[--parts ...\] Parts (all by default): lang test language (TinyStories-test, WikiText-test; 256 rows each): nats, bits/byte; interventions core_off, plan_swap, u_off, c_off; loop depth R in {1,2,3,4,6,8} trunc context truncation r in {72,128,256,512,1024} on the last 16 positions + plan-swap cost (G-D2) items synthetic families (200/family) and probes (512/family), test ID and OOD; interventions on every suite p

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/final_eval.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/final_eval.py>)

Imports:

```python
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward
from evaluate import Evaluator
from data import make_batch, pack_items, target_mask, Canon, PAD, EOS
from probes import PROBES
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:32>) — lines 32–37
- [part_lang](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:40>) — lines 40–53
- [part_trunc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:56>) — lines 56–62
- [part_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:65>) — lines 65–90
- [fit_probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:95>) — lines 95–116: Multinomial logistic regression, 80/20 split, standardised features; returns held-out top-1 accuracy.
- [part_probes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:120>) — lines 120–193
- [part_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:196>) — lines 196–205
- [part_gen](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:209>) — lines 209–247
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/final_eval.py:250>) — lines 250–291

CLI arguments:

```python
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--parts", default="lang,trunc,items,probes,health,gen")
ap.add_argument("--probe_only_interv", action="store_true")
ap.add_argument("--R", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
```

## run_v2_19M/code/prof_attn.py

43 lines; SHA-256 `17c132d38c02aba7a14e216b9b81524750df02337e7b949a9836b85ddd00e290`.

Primary copy: [run_v2_19M/code/prof_attn.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_attn.py>).

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/prof_attn.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_attn.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_attn.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_attn.py>)

Imports:

```python
import time, torch, torch.nn.functional as F
from hrm_encoder import build_masks, same_doc_shift_mask
```

Symbols:

- [t](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_attn.py:13>) — lines 13–19
- [local_attn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_attn.py:22>) — lines 22–31: Exact sliding-window attention over keys t-W+1..t (same doc), without the T x T matrix.

## run_v2_19M/code/prof_enc.py

32 lines; SHA-256 `1e8bab3b077d82fd2dc194133f6a5bd3e19ff3ddd7bf2679066737bc601eb4f4`.

Primary copy: [run_v2_19M/code/prof_enc.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_enc.py>).

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/prof_enc.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_enc.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_enc.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_enc.py>)

Imports:

```python
import time, torch
from model import build_model
from hrm_encoder import build_masks, rope_cache
```

Symbols:

- [t](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_enc.py:13>) — lines 13–19
- [fb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_enc.py:22>) — lines 22–23

## run_v2_19M/code/prof_layer.py

29 lines; SHA-256 `e69d10ce17ac35d0e7e5bcb086d218c9d0bb514caca09396be0ec602682b23e8`.

Primary copy: [run_v2_19M/code/prof_layer.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_layer.py>).

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/prof_layer.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_layer.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_layer.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_layer.py>)

Imports:

```python
import time, torch, torch.nn.functional as F
from model import build_model
from concept_core_d6 import moe_dispatch
```

Symbols:

- [t](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_layer.py:9>) — lines 9–14
- [fb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_layer.py:25>) — lines 25–26

## run_v2_19M/code/prof_moe.py

34 lines; SHA-256 `43a1f9273d4dd5e33dfd37793267d90a5495ace9d85d248af15a46c41f4aed89`.

Primary copy: [run_v2_19M/code/prof_moe.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_moe.py>).

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_moe.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_moe.py>)

Imports:

```python
import time, math, torch, torch.nn.functional as F
```

Symbols:

- [t](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_moe.py:5>) — lines 5–9
- [top2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_moe.py:27>) — lines 27–31

## run_v2_19M/code/prof_moe2.py

29 lines; SHA-256 `0a12c7c467ec7a5275c6398ee9c451c8f0dc1f2bc42eee19c46a327e73428489`.

Primary copy: [run_v2_19M/code/prof_moe2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_moe2.py>).

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe2.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_moe2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/prof_moe2.py>)

Imports:

```python
import time, math, torch, torch.nn.functional as F
from concept_core_d6 import ExpertPool
```

Symbols:

- [s](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/prof_moe2.py:7>) — lines 7–8

## run_v2_19M/code/profile_step.py

80 lines; SHA-256 `a62ab0d2a61a2f82b57125e9ace8aba1d39f7def8a9b6ab1313b46bf184af216`.

Primary copy: [run_v2_19M/code/profile_step.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/profile_step.py>).

Time the pieces of one v2 training micro-batch on MPS (synchronised wall clock). python profile_step.py \[--B 2\] \[--K 10\]

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/profile_step.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/profile_step.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/profile_step.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/profile_step.py>)

Imports:

```python
import argparse, contextlib, time, os
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, nextlat_losses
from data import ShardStream, target_mask
import train as T
import speed, hrm_encoder
```

Symbols:

- [tic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/profile_step.py:34>) — lines 34–35

CLI arguments:

```python
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--amp", default="none")
ap.add_argument("--cf", type=float, default=1.5)
ap.add_argument("--cache", type=int, default=1)
ap.add_argument("--memo", type=int, default=1)
```

## v4_tiny/code/speed.py

39 lines; SHA-256 `008717a075d8082e0213cff84a054c0e765e4548c3934959bcff834c318e926f`.

Primary copy: [v4_tiny/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed.py>).

Throughput shims with identical arithmetic (V-006). The encoder's ShortConv recomputes `same_doc_shift_mask(doc, i)` (pad + compare) for every conv and every shift: 3 shifts x 3 convs x 8 layers + Engram = 75 boolean masks per forward, all from the same `doc`. Inside one Encoder.forward the masks depend only on (doc, i), so they are memoised for the duration of that call. The masks carry no gradient, so the backward pass is unaffected. The memo is THREAD-LOCAL: the data-prefetch thread also calls same_doc_shift_mask (engram_indices) concurrently and always takes the original path.

Identical copies:

- [v41_tiny/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/speed.py>)
- [run_v2_19M/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/speed.py>)
- [run_v3_19M/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/speed.py>)
- [v2_underperformance_review_2026-09-28/source_snapshot/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/speed.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/speed.py>)

Imports:

```python
import threading
import hrm_encoder as _E
```

Symbols:

- [_memo_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed.py:16>) — lines 16–24

## run_v2_19M/code/test_v2.py

185 lines; SHA-256 `b7a6f2f219772de7a2807f29af993f271085fc503cf1fcbe6de5f185e2e628b7`.

Primary copy: [run_v2_19M/code/test_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py>).

Exact CPU tests for HRM Thinker v2 (float64, small config). python test_v2.py T1 capacity MoE dispatch == masked dense MoE when nothing is dropped T2 per-window backward (c_hat / tok leaves) == one-graph backward with the carry detached between windows T3 identity at init: pass 1 leaves S_0 unchanged (W_o, W_2 zero), readout finite T4 causality + document isolation through encoder -> D6 core -> decoder (eval mode) T5 parameter table of the full model == budget (18,833,529 = 18,833,513 + 16 per-head temperatures) T6 NextLat never updates the head E; gradients reach psi, decoder, core (final window) and encoder T7 windows(K) layout for K = 8..1

Identical copies:

- [v2_underperformance_review_2026-09-28/source_snapshot/code/test_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/test_v2.py>)
- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/test_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/test_v2.py>)

Imports:

```python
import copy, math, sys
import torch
import torch.nn.functional as F
from model import build_model, forward, nextlat_losses, param_table
from concept_core_d6 import moe_dispatch, ExpertPool
from data import target_mask
import train as T
```

Symbols:

- [cfg_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:19>) — lines 19–25
- [randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:28>) — lines 28–32
- [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:35>) — lines 35–42
- [t1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:45>) — lines 45–56
- [reference_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:59>) — lines 59–89: One graph: same arithmetic as train.micro_step, carry detached between windows, single backward.
- [t2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:92>) — lines 92–114
- [t3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:117>) — lines 117–126
- [t4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:129>) — lines 129–148
- [t5](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:151>) — lines 151–155
- [t6](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:158>) — lines 158–172
- [t7](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/code/test_v2.py:175>) — lines 175–179

## run_v2_19M/webapp/server.py

661 lines; SHA-256 `19190a993731899da72c41c502bb50231cd212996f16271371f8dbb5dde3ab83`.

Primary copy: [run_v2_19M/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py>).

HRM THINKER-25M dashboard server (read-only, stdlib only; D-036). nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 & open http://127.0.0.1:8765 Never writes under runs/, never imports torch, never touches the GPU. Log files are read incrementally (byte offset per file) and the assembled JSON is rebuilt only when a watched file changed.

Identical copies:

- [run_v3_19M/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/webapp/server.py>)

Imports:

```python
import json, os, re, subprocess, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
```

Symbols:

- [budget_h](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:20>) — lines 20–25: v2: the training-time budget from the run's config.json (48 h), else 24 h.
- [Tail](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:28>) — lines 28–65: Incremental JSONL / text reader: keeps a byte offset and the unterminated trailing fragment.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:30>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:34>).
- [cont_active](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:85>) — lines 85–86
- [active_dir](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:89>) — lines 89–90
- [cont_sched](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:93>) — lines 93–100: Schedule of the continuation from its config.json (step_offset = global step of its step 0).
- [update_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:103>) — lines 103–106
- [shifted](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:109>) — lines 109–119: Continuation rows on the global step axis (train rows carry gstep; restart and eval rows are shifted here).
- [shift_line_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:122>) — lines 122–123
- [sh](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:134>) — lines 134–138
- [system_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:141>) — lines 141–172: Cheap OS probes, at most once every 20 s.
- [_load_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:179>) — lines 179–186
- [_sig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:189>) — lines 189–193
- [sample_live](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:196>) — lines 196–238: Called every 5 s by the sampler thread: GPU, trainer process, swap, battery (30 s), signatures, heartbeat.
- [sampler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:241>) — lines 241–247
- [live_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:250>) — lines 250–271
- [slim_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:274>) — lines 274–286
- [run_status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:289>) — lines 289–329
- [events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:332>) — lines 332–345: Resume / pause / guard / schedule events from the trainer output, for chart markers.
- [screens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:348>) — lines 348–359
- [plan_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:365>) — lines 365–386: v2 four stages from runs/chain.log: train 0 -> 24 h · 24 h evaluation · train 24 -> 48 h · final evaluation.
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:389>) — lines 389–430
- [final_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:433>) — lines 433–452: Final evaluations (D-044): full eval_final.json of the main run and, once it exists, of the continuation.
- [_f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:461>) — lines 461–462
- [build_digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:465>) — lines 465–539: Plain-text digest of every page's data (architecture, screens, main run, final eval, gates, continuation).
- [run_analysis](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:561>) — lines 561–578
- [analysis_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:581>) — lines 581–595
- [H](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:598>) — lines 598–649
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:599>), [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:602>), [end_headers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:605>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/webapp/server.py:611>).

## run_v3_19M/code/ab_interleave.py

95 lines; SHA-256 `280aedb294ee3342e34e8d57445a984196cfd8919cbe0b8f44c0dba57aed0a7d`.

Primary copy: [run_v3_19M/code/ab_interleave.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/ab_interleave.py>).

Interleaved throughput A/B on MPS (thermal drift on the fanless M2 Air makes sequential runs unusable: the same base config measured 2,059 then 1,478 tok/s three minutes apart). One process, one model, real optimizer steps (32,768 tokens each); configurations alternate every optimizer step (A B A B ...), so drift affects both equally. Reports the median seconds per step and tok/s per configuration and the paired ratio (median of B_i / A_i over adjacent pairs). Configs (comma list): base | block64 | block128 | memo0 | b4 | b4block64 | compcore (torch.compile core layers, train only) python ab_interleave.py --configs base,block64 --steps 16

Imports:

```python
import argparse, os, statistics, time
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups
import train as TR
from data import ShardStream
import subprocess
```

Symbols:

- [Args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/ab_interleave.py:41>) — lines 41–42
  Methods: .
- [one_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/ab_interleave.py:45>) — lines 45–68

CLI arguments:

```python
ap.add_argument("--configs", default="base,block64")
ap.add_argument("--steps", type=int, default=16)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--no_memo", action="store_true")
```

## run_v3_19M/code/compile_check.py

63 lines; SHA-256 `d5ee58c3d7d4a864dfd4c0b88997f77117e3a54dd9bf82d405f11b82f563a491`.

Primary copy: [run_v3_19M/code/compile_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/compile_check.py>).

torch.compile parity on MPS (V3-006): one training micro-step eager vs with the core layers (and optionally the encoder / decoder layers) compiled, same weights, same batch (B 2, T 1024, block-local attention 64, memo off). Pass: max |Δ logits| < 1e-3, per-parameter gradient relative error < 1e-2, and the core.table / core.shared.w2 gradients nonzero in both and within tolerance. Exit 1 on failure. python compile_check.py \[--all\]

Imports:

```python
import copy, sys, time
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, rms
from hrm_encoder import engram_indices
from data import target_mask
```

Symbols:

- [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/compile_check.py:29>) — lines 29–43

## run_v3_19M/code/evaluate.py

264 lines; SHA-256 `a934d0d5c5b9d9a529bcabd5f2ffe5c9bead5f9bb913bdd02b3e4adb081dd654`.

Primary copy: [run_v3_19M/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py>).

v3 evaluation (ported from v2's evaluate.py; language / items / truncation code unchanged so numbers stay comparable). v3 changes: R is a schedule name ("1" | "2a" | "2b" | "3"); codebook health uses the r2 target masks (ncp_ce = masked pairs: nonempty current, complete target, same doc) and also reports ncp at M = 1; loop_dynamics measures the v3 main trajectory (norms, updates, angles, rho, relative updates, gradient of NTP at each pass state). Evaluation benchmark (PROMPT section 5): language NTP (nats, bits/byte), item exact match (teacher-forced greedy), interventions (core off, plan swap, u off, c off, loop depth R, context truncation),

Imports:

```python
import math, os, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from model import forward, chunk_masks, R_MAP
from data import Canon, make_batch, target_mask, pack_items, PAD
```

Symbols:

- [Evaluator](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:22>) — lines 22–264
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:23>), [_to](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:31>), [_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:34>), [items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:43>), [lang_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:48>), [language](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:57>), [truncation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:76>), [item_accuracy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:103>), [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:136>), [codebook_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:150>), [telemetry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:186>), [loop_dynamics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:204>), [quick](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:236>), [_maint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/evaluate.py:258>).

## run_v3_19M/code/final_eval.py

395 lines; SHA-256 `848b467416f4eb7881428e925bd565128a8a261d3b93acde4dc124f01ec8e907`.

Primary copy: [run_v3_19M/code/final_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py>).

Detailed testing of v3 (draft_v3.md r2 §7.4): at the 24 h milestone on VAL (the test split stays sealed) and at 48 h on TEST + OOD (the number of record). v1's suite ported from v2; M sweep {1, 2a, 2b, 3}; the full intervention set (core, u, c_pred, plan swap, shared FFN, table, Engram, Engram read 2, F layers); v3 loop dynamics; the per-term gradient audit. python final_eval.py --ckpt ../runs/v3_48h/ckpt_last.pt --out ../runs/v3_48h/eval_24h.json --split val python final_eval.py --ckpt ../runs/v3_48h/ckpt_last.pt --out ../runs/v3_48h/eval_final.json --split test Parts (all by default): lang test language (TinyStories-test, WikiText-test; 256

Imports:

```python
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward, chunk_masks, rms, param_groups
from evaluate import Evaluator
from data import make_batch, pack_items, target_mask, Canon, PAD, EOS
from probes import PROBES
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:41>) — lines 41–53
- [part_lang](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:56>) — lines 56–72
- [part_trunc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:75>) — lines 75–81
- [part_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:84>) — lines 84–110
- [fit_probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:115>) — lines 115–137: Multinomial logistic regression, 80/20 split, standardised features; returns held-out top-1 accuracy.
- [part_probes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:141>) — lines 141–214
- [part_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:217>) — lines 217–226
- [part_gen](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:230>) — lines 230–268
- [audit_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:279>) — lines 279–292: Fixed diagnostic batches: TinyStories, WikiText (language rows) and the probe / synthetic item families.
- [part_audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:295>) — lines 295–344: g_main = grad(NTP + 0.1 NCP(u3)), g_short = grad(0.05 NCP(u_s) + 0.1 L_scons) per group; report |g_main|, |g_short|, |g_main + g_short| (norm of the SUM), cos(g_main, g_short) and the projection of g_short on g_main.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/final_eval.py:347>) — lines 347–391

CLI arguments:

```python
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--parts", default="lang,trunc,items,probes,health,audit,gen")
ap.add_argument("--split", default="test", choices=("val", "test"))
ap.add_argument("--probe_only_interv", action="store_true")
ap.add_argument("--R", default="")
ap.add_argument("--limit", type=int, default=0)
```

## run_v3_19M/code/model.py

374 lines; SHA-256 `3825b31d3c931694fca53d7b620bbfe43f370cf18a8f2c74c1cf322514efd6d7`.

Primary copy: [run_v3_19M/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py>).

HRM THINKER v3 (19M compute, d 256) — the draft_v3.md (r2) architecture, trainer-grade. Encoder: v1 hrm_encoder at d 256, SSSFSSSF, F layers with 6 heads; ONE Engram gather feeding read 1 (after layer 0, with conv, v1's module) and read 2 (after layer 4, no conv, W_v zero-init, gate reads L only). Core: 4 distinct F-anatomy layers over {C, S}; per-stream (t, dt)-table read gains and write gains; own SwiGLU 736 + shared SwiGLU 1024 in parallel (shared w2 zero-init); S0 = 0; main trajectory M = 3 on the grid of 1/3 plus a training-only shortcut (1 | 2a | 2b), batched with the main passes at concept rate (rows \[main; shortcut\]). Decoder: v1 co

Identical copies:

- [v4_tiny/code/model_v3ref.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v3ref.py>)
- [v41_tiny/code/model_v3ref.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v3ref.py>)

Imports:

```python
import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
import compat
import speed as _speed
from hrm_encoder import (EncoderConfig, Encoder, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms, build_masks,
                         rope_cache)
from hrm_decoder import DecoderConfig, Decoder
from concept_core_v2 import ConceptVocab
```

Symbols:

- [EngramRead2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:44>) — lines 44–60: Second read of the shared Engram rows: own W_k, W_v (zero-init), gated read of L only, write to {L, G}, no conv.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:46>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:55>).
- [engram_read1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:63>) — lines 63–70: v1 Engram.forward with the gathered rows passed in (identical arithmetic).
- [EncoderV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:73>) — lines 73–118
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:74>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:84>), [_fwd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:94>).
- [CoreLayerV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:122>) — lines 122–155
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:123>), [c_half](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:131>), [read_mod](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:137>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:146>).
- [ConceptCoreV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:158>) — lines 158–268
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:159>), [_gains](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:176>), [one_pass](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:186>), [c_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:192>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:195>), [trajectory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:226>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:230>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:239>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:246>), [vq_sum](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:249>), [ncp_sum](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:265>).
- [chunk_masks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:271>) — lines 271–279: nonempty (B, M): chunk has >= 1 real token; complete (B, M): 4 real tokens; pair_ok (B, M-1): current chunk nonempty, target chunk complete, same document (documents start on chunk boundaries, verified in the data).
- [HRMThinkerV3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:283>) — lines 283–298
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:284>).
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:301>) — lines 301–304
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:307>) — lines 307–324: Evaluation / generation path (one trajectory, no shortcut). Returns logits (B, T, V) and aux.
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:328>) — lines 328–347: Repository convention: compute = encoder non-embedding + core without codebook + decoder. Parts reconcile.
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/model.py:353>) — lines 353–363: No decay: biases, embeddings, Engram table, temperatures, conv kernels (.w), codebook, null plan, chunk offsets, and the (t, dt) table (draft r2 §4.3). Everything else (matrices, zc gains) is decayed, as v1.

## run_v3_19M/code/mps_check.py

62 lines; SHA-256 `16fb37aaebb8653044af6549c5bf75acf3fc7111cda3cf855c3b789287cb4034`.

Primary copy: [run_v3_19M/code/mps_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/mps_check.py>).

CPU <-> MPS parity (draft r2 §8.5 step 6): fp32 MPS forward and gradients against the CPU reference, at the FULL config (B 1, T 1024, tiny Engram table so CPU memory stays small), with and without the speed paths. Tolerance: max |d logits| < 1e-3 (the v1 mps_smoke_test bar), gradient relative error < 1e-2. Exit 1 on failure. python mps_check.py

Imports:

```python
import copy, sys, time
import torch, torch.nn.functional as F
import model as M
import speed_v3 as SV
from model import build_model, rms
from hrm_encoder import engram_indices
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/mps_check.py:15>) — lines 15–27
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/mps_check.py:30>) — lines 30–58

## run_v3_19M/code/profile_step.py

71 lines; SHA-256 `bcade01efa4d4ec93e534b9d2e078b3b33800c1b288dcbc8915a2e6b927faa5b`.

Primary copy: [run_v3_19M/code/profile_step.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/profile_step.py>).

Where does one v3 training micro-step spend its time on MPS? (full model, real 80M table, real data, B2) Sections are separated by torch.mps.synchronize(); medians over N repeats after warm-up. python profile_step.py \[--block_attn 64\] \[--micro_bs 2\]

Imports:

```python
import argparse, os, time, statistics
import numpy as np
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups, rms
from data import ShardStream, target_mask
```

Symbols:

- [tick](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/profile_step.py:31>) — lines 31–32

CLI arguments:

```python
ap.add_argument("--block_attn", type=int, default=0)
ap.add_argument("--micro_bs", type=int, default=2)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--eval", action="store_true")
```

## v4_tiny/code/speed_v3.py

191 lines; SHA-256 `c051b4b9759466acfe50789f06a78f27e8bde2087ec4025043e501ed6b558df2`.

Primary copy: [v4_tiny/code/speed_v3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py>).

v3 throughput paths (each is opt-in and must pass its parity test before use; see DECISIONS.md V3-xx). 1. Block-local sliding-window attention for the window-8 layers (encoder S layers, confined decoder layers). v1 builds a (B, 1, T, T) mask and runs SDPA over all T keys although each query may see only the last `window` positions. Here queries are split into blocks of `block` positions; block b attends to keys \[b*block - (window-1), (b+1)*block) with the SAME mask rule (causal, t - j < window, same document). Masked entries are exactly zero after the softmax in both forms, so the result equals the full form up to float reduction order. Scor

Identical copies:

- [v41_tiny/code/speed_v3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/speed_v3.py>)
- [run_v3_19M/code/speed_v3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/speed_v3.py>)

Imports:

```python
import contextlib
import torch
import torch.nn.functional as F
import hrm_encoder as _E
import hrm_decoder as _D
import speed as _speed
from hrm_encoder import apply_partial_rope
```

Symbols:

- [_memo](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:25>) — lines 25–26
- [_key](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:29>) — lines 29–30
- [memo_shift_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:36>) — lines 36–45: speed.py's memo with strides in the key (a view such as doc\[:, ::4\] shares doc's data_ptr).
- [block_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:51>) — lines 51–71: (B * nb, 1, block, block + window - 1) bool: query i (global) may attend key j (global) iff 0 <= i - j < window, j >= 0 and doc\[i\] == doc\[j\]. Cached only inside the current memo scope (never across steps).
- [local_sdpa](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:74>) — lines 74–84: q: (B, H, T, dh); k, v: (B, H, T, dh) (already GQA-repeated). Returns (B, H, T, dh).
- [_ga_forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:92>) — lines 92–112
- [_cl_forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:115>) — lines 115–134
- [_dec_forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:141>) — lines 141–146
- [enable_block_attention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:149>) — lines 149–159: Route window-8 attention through local_sdpa (block must divide T). block=0 restores v1's full-mask SDPA.
- [step_memo](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:163>) — lines 163–173: Doc-shift mask memo for a whole micro-step (encoder + core ShortConvs). Nested encoder forwards reuse it.
- [compile_layer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/speed_v3.py:181>) — lines 181–191: Compile layer.forward (dynamic=False); the compiled path is used only when active(layer) is true (training).

## run_v3_19M/code/test_speed.py

86 lines; SHA-256 `b3835e57a384d952809f4f51012ef92399631165b94f0abb3c9551e7bba89a02`.

Primary copy: [run_v3_19M/code/test_speed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_speed.py>).

Parity tests for the v3 speed paths (CPU float64; the MPS fp32 check is in mps_check.py). Exit 1 on any failure. nice -n 19 python test_speed.py

Imports:

```python
import copy, sys
import torch, torch.nn.functional as F
import model as M
import speed_v3 as SV
from model import build_model, forward, rms
from hrm_encoder import engram_indices
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_speed.py:15>) — lines 15–17
- [loss_and_grads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_speed.py:20>) — lines 20–29
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_speed.py:32>) — lines 32–82

## run_v3_19M/code/test_v3.py

383 lines; SHA-256 `4d958bfa5803f7bc95d242aab99f0d14ea9b3d512df26d508992af6a70211623`.

Primary copy: [run_v3_19M/code/test_v3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py>).

v3 acceptance harness (CPU, float64, small config) — draft_v3.md r2 §3.6, §4.10, §5.5, §7.1 (C1–C7 repaired). Every test prints PASS/FAIL; the process exits 1 if any test fails (review C6). Deliberate-defect mutants must make their test FAIL (a test that cannot fail proves nothing). nice -n 19 python test_v3.py # all tests

Imports:

```python
import copy, math, os, sys
import torch, torch.nn.functional as F
import model as M
from model import build_model, forward, chunk_masks, rms, PAIRS, SCHED
from hrm_encoder import engram_indices, Encoder
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:20>) — lines 20–22
- [perturb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:25>) — lines 25–29
- [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:32>) — lines 32–38
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:41>) — lines 41–353
- [ntp_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:356>) — lines 356–370
- [shortcut_losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/test_v3.py:373>) — lines 373–379

## run_v3_19M/code/train.py

590 lines; SHA-256 `1112c549ce18424c053e0257448ba314292cc84a67b0c48bbf0a45a560ea8334`.

Primary copy: [run_v3_19M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py>).

Train HRM THINKER v3 (19M, draft_v3.md r2). One process, one device; resumable; pause-aware; 48 h in ONE schedule. python train.py --run_dir ../runs/v3_48h --hours 48 --milestone_hours 24 python train.py --run_dir ../runs/v3_48h --resume ../runs/v3_48h/ckpt_last.pt Per micro-batch: main trajectory M = 3 (1/3, 1/3, 1/3) + a shortcut S ~ {1: 0.5, 2a: 0.25, 2b: 0.25} (own Generator, checkpointed), batched with the main passes at concept rate; full backprop; one backward per micro-batch. Loss (full-batch denominators; all micro-batches of an optimizer step are fetched first): L = NTP(u3) + 0.1 NCP(u3) + 0.05 NCP(u_s) + 0.1 L_scons(rms S_s, sg rms

Imports:

```python
import argparse, contextlib, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import model as M
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups, param_table, rms
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator
```

Symbols:

- [now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:33>) — lines 33–34
- [nz](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:37>) — lines 37–38
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:41>) — lines 41–44
- [sample_short](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:47>) — lines 47–54
- [restart_dead_now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:58>) — lines 58–75: Re-seed codes unused since the last restart from COMPLETE real chunks; reset their Adam moments.
- [rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:78>) — lines 78–82
- [set_rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:85>) — lines 85–88
- [save_ckpt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:91>) — lines 91–104
- [phys_footprint_gib](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:107>) — lines 107–116
- [swap_used_mb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:119>) — lines 119–124
- [batch_counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:127>) — lines 127–134: Full-batch denominators from ids/doc only (before any forward).
- [micro_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:137>) — lines 137–165: One micro-batch: forward (main + batched shortcut), weighted sum over full-batch denominators, one backward. Returns detached SUMS (divide by N for means).
- [loop_geometry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:169>) — lines 169–176: Per-pass norms, update norms, angle(dS2, dS3), rho3 = |S3| / sum |dS_i|, relative updates (draft §4.7).
- [table_norms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:180>) — lines 180–189
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/code/train.py:192>) — lines 192–586

CLI arguments:

```python
ap.add_argument("--run_dir", required=True)
ap.add_argument("--data_dir", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")))
ap.add_argument("--device", default="mps")
ap.add_argument("--size", default="full")
ap.add_argument("--micro_bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--lr_scale", type=float, default=1.0)
ap.add_argument("--lr_min_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--hours", type=float, default=48.0)
ap.add_argument("--milestone_hours", type=float, default=24.0)
ap.add_argument("--est_total_steps", type=int, default=7600)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--init_only", action="store_true")
ap.add_argument("--lam_ncp", type=float, default=0.1)
ap.add_argument("--lam_s", type=float, default=0.05)
ap.add_argument("--lam_c", type=float, default=0.1)
ap.add_argument("--no_shortcut", action="store_true")
ap.add_argument("--no_table", action="store_true")
ap.add_argument("--shared_ff", type=int, default=-1)
ap.add_argument("--core_ff", type=int, default=-1)
ap.add_argument("--main_sched", default="3")
ap.add_argument("--eval_R", default="3")
ap.add_argument("--eval_every_min", type=float, default=60.0)
ap.add_argument("--eval_at_v1_steps", type=int, default=1)
ap.add_argument("--log_every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--traj_seed", type=int, default=1234)
ap.add_argument("--heartbeat", type=int, default=1)
ap.add_argument("--no_final_eval", action="store_true")
ap.add_argument("--quick_eval_n", type=int, default=128)
ap.add_argument("--resume", default="")
ap.add_argument("--mem_frac_guard", type=float, default=0.90)
ap.add_argument("--keep", type=int, default=2)
ap.add_argument("--div_min_wall_min", type=float, default=30.0)
ap.add_argument("--mem_restart_gib", type=float, default=24.0)
ap.add_argument("--mem_restart_growth_gib", type=float, default=12.0)
ap.add_argument("--debug_restart_step", type=int, default=0)
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--block_attn", type=int, default=0)
ap.add_argument("--step_memo", type=int, default=1)
ap.add_argument("--compile", default="none")
ap.add_argument("--dash_file", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "webapp", "data", "v3_live.json")))
```

## v10_1/__init__.py

1 lines; SHA-256 `47a6aec9ba1ff0ea8b45881683425f49714de20007fae19d1227869433c5afdc`.

Primary copy: [v10_1/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/__init__.py>).

V10.1 reference implementation and isolated validation, not a promoted training run.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/__init__.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/__init__.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/__init__.py>)

## v10_1/accumulate.py

50 lines; SHA-256 `2595fbfe85e2f39df30a8bb896c31b26d6ea6a8f2f16994a959f750aed21e41f`.

Primary copy: [v10_1/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/accumulate.py>).

Memory-bounded effective batches with each objective's correct denominator.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/accumulate.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/accumulate.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/accumulate.py>)

Imports:

```python
import torch
from .model import losses
```

Symbols:

- [backward_effective_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/accumulate.py:6>) — lines 6–50

## v10_1/evaluate.py

101 lines; SHA-256 `9500357208d221a568e326066758031062c617fc620b0b531f8d44d2a5f5d904`.

Primary copy: [v10_1/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/evaluate.py>).

Fixed, balanced development panels; explicit route ablations and strict solves.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/evaluate.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/evaluate.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/evaluate.py>)

Imports:

```python
import json
import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus,verify_sudoku
from v10_10M.evaluate import generate
from .model import token_loss,latent_errors
from .prepare_data import DATA
```

Symbols:

- [parse_answer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/evaluate.py:18>) — lines 18–28: Exact numeric grammar, including ordered lists; no answer repair.
- [score_math](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/evaluate.py:31>) — lines 31–34
- [generation_indices](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/evaluate.py:37>) — lines 37–41
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/evaluate.py:45>) — lines 45–101

## v10_1/launch.py

37 lines; SHA-256 `3631bd572f0f3eee5e8589afd04b2a9c5642ce6d544a1dbf97876710b88e4e6d`.

Primary copy: [v10_1/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/launch.py>).

Detach the locked trainer, prevent idle sleep, and verify startup.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/launch.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/launch.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/launch.py>)

Imports:

```python
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .train import parser,ROOT
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/launch.py:11>) — lines 11–34

## v10_1/memory.py

103 lines; SHA-256 `0d5ce95e1c538a3c0d8dba0d4076c8114a24cc23b547b103bf386bec93fe3f5b`.

Primary copy: [v10_1/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py>).

Frozen-era exact RBF updates for both V10.1 downstream routes.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/memory.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/memory.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/memory.py>)

Imports:

```python
import hashlib
import json
import math
import threading
from pathlib import Path
from dataclasses import asdict
import torch
from v10_10M.memory import Snapshot
from .model import token_loss, latent_errors
```

Symbols:

- [model_era](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:13>) — lines 13–24: Fingerprint actual weights, architecture and query/read semantics once per era.
- [RBFMemory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:27>) — lines 27–82
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:28>), [snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:35>), [commit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:38>), [state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:64>), [from_state_dict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:70>).
- [write_objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:85>) — lines 85–86
- [learn_episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/memory.py:89>) — lines 89–103

## v10_1/optimization/__init__.py

1 lines; SHA-256 `e468878b0a85d96ad756466cdaae8d70f5837426e802aeb7b2397cdd035d2457`.

Primary copy: [v10_1/optimization/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/__init__.py>).

Measured, objective-preserving training-loop optimizations.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/__init__.py>)

## v10_1/optimization/benchmark.py

55 lines; SHA-256 `ae3adeef7387919b1a651cf214945e339efc0213975d01314d6452532d3d39bc`.

Primary copy: [v10_1/optimization/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/benchmark.py>).

Alternating synchronized full-checkpoint update timings; no live-run writes.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/benchmark.py>)

Imports:

```python
import gc,json,time,statistics
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.train import build_optimizers
from v10_1.model import V101
from v10_1.accumulate import backward_effective_batch
from v10_1.train import gradient_metrics as reference_metrics
from .fast import backward_full_batch,gradient_metrics
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/benchmark.py:15>) — lines 15–53

## v10_1/optimization/fast.py

64 lines; SHA-256 `6c9bfabcf35065fd2c26f1f26897bf78e9cebc89672fdded7ad8fedde621170a`.

Primary copy: [v10_1/optimization/fast.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/fast.py>).

Keep batch bookkeeping on CPU and combine device-to-host scalar transfers. The model, losses, sampled boundary RNG, optimizer groups and update frequency are unchanged. This path is intentionally restricted to full-batch arm S.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/fast.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/fast.py>)

Imports:

```python
import math
import torch
from v10_1.model import losses,chunk_batch
```

Symbols:

- [backward_full_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/fast.py:11>) — lines 11–42
- [gradient_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/fast.py:45>) — lines 45–64

## v10_1/optimization/validate.py

47 lines; SHA-256 `17c209bcc3c8a35fb7de67be64a9484f9ff136d7986b60b278db371662bd6ecb`.

Primary copy: [v10_1/optimization/validate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/validate.py>).

Compare real forward/backward losses and every gradient against the reference.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/validate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/optimization/validate.py>)

Imports:

```python
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.train import build_optimizers
from v10_1.model import V101
from v10_1.accumulate import backward_effective_batch
from v10_1.train import gradient_metrics as original_metrics
from .fast import backward_full_batch,gradient_metrics
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/optimization/validate.py:15>) — lines 15–43

CLI arguments:

```python
p.add_argument('--device',default='cpu')
```

## v10_1/parity.py

54 lines; SHA-256 `9f4e342ad4fb8936ef30952b3aa975536a7682b33025e05cc0cc88735d6d864e`.

Primary copy: [v10_1/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/parity.py>).

Bounded CPU/Metal parity and full-size, context-384 integration checks.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/parity.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/parity.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/parity.py>)

Imports:

```python
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.train import canonical_ids,build_optimizers
from .model import V101,Architecture,losses
from .validate_training import load_examples,pack,clip_and_log
from .provenance import provenance
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/parity.py:15>) — lines 15–51

## v10_1/pipeline_tests.py

60 lines; SHA-256 `7da7211dfe88fe78da1a63ccbdb426d0fe1ff4ae78eab452fa9fabf9a837fc10`.

Primary copy: [v10_1/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/pipeline_tests.py>).

Clean-data identity, scoring, telemetry persistence and production resume checks.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/pipeline_tests.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/pipeline_tests.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/pipeline_tests.py>)

Imports:

```python
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .evaluate import parse_answer,score_math,generation_indices
from .prepare_data import DATA,ORIGINAL,FAMILIES
from .telemetry import Store
from .train import verify_data
from .provenance import provenance
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/pipeline_tests.py:14>) — lines 14–57

## v10_1/prepare_data.py

116 lines; SHA-256 `54d5661ac245fa67c07e1191e8cfaa186ff1e786c71812698c0e0aa1bbf66edd`.

Primary copy: [v10_1/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/prepare_data.py>).

Clean train/dev in a new corpus, preserving original identities and partitions. No sealed-test records are opened. Test stays reserved in the original corpus; it must be re-encoded with this tokenizer only when final evaluation is authorized.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/prepare_data.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/prepare_data.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/prepare_data.py>)

Imports:

```python
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
from v10_10M.prepare_data import digest, key, SPECIAL
from v10_10M.data_tools.clean_math import clean_math_field
from v10_10M.data import verify_sudoku
```

Symbols:

- [records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/prepare_data.py:25>) — lines 25–62
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/prepare_data.py:65>) — lines 65–112

CLI arguments:

```python
p.add_argument('--output',default=str(DATA))
```

## v10_1/provenance.py

19 lines; SHA-256 `47310dc406c8ea0fb5eaf093ad5a44a9698300244b0fe5697526268a6a28df50`.

Primary copy: [v10_1/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/provenance.py>).

Source and immutable corpus identities for isolated validation artifacts.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/provenance.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/provenance.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/provenance.py>)

Imports:

```python
import hashlib
from pathlib import Path
import torch
```

Symbols:

- [package_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/provenance.py:9>) — lines 9–13
- [provenance](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/provenance.py:16>) — lines 16–19

## v10_1/supervise.py

27 lines; SHA-256 `b734d812203ed2721d6de5f153b3690ed9d61f8131fef474e1bb5ac78cd7de6e`.

Primary copy: [v10_1/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/supervise.py>).

Exclusive run lock and reproducible source snapshot; no automatic restart.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/supervise.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/supervise.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/supervise.py>)

Imports:

```python
import fcntl
from pathlib import Path
import shutil
from .train import main,parser,ROOT
```

Symbols:

- [supervise](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/supervise.py:8>) — lines 8–24

## v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py

85 lines; SHA-256 `bc71f1870c7233695434dee801e117d2e47eff2bb324f311a6afc39875214d77`.

Primary copy: [v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py>).

V10.1 adapter for the tested read-only SQLite collector and dashboard.

Imports:

```python
from pathlib import Path
import time
import psutil
from urllib.parse import urlparse
from v10_10M.telemetry import server,store
```

Symbols:

- [make_handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:12>) — lines 12–22
- [Store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:25>) — lines 25–35
  Methods: [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:26>).
- [Monitor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:38>) — lines 38–68
  Methods: [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:39>).
- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/telemetry.py:71>) — lines 71–81

## v10_1/tests.py

249 lines; SHA-256 `9801eb7b29287134cf60a94abb60bcc0226d56a4ab93e263fc30ab71778b7148`.

Primary copy: [v10_1/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/tests.py>).

Behavioral end-to-end contracts; small vocabulary/table, full width/depth.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/tests.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/tests.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/tests.py>)

Imports:

```python
import argparse
import copy
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.model import V10, losses as original_losses
from v10_10M.train import build_optimizers,clip
from .model import V101,Architecture,losses,token_loss,chunk_batch
from .memory import RBFMemory,learn_episode,write_objective,model_era
from .provenance import provenance
from .accumulate import backward_effective_batch
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/tests.py:20>) — lines 20–241

CLI arguments:

```python
p.add_argument('--device',choices=['cpu','mps'],default='cpu')
p.add_argument('--output',required=True)
```

## v10_1/train.py

197 lines; SHA-256 `1630357c5995e35bb30d020537337d0ec3256d7a0846c9a6d40e0d35c989df9b`.

Primary copy: [v10_1/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py>).

Deadline-bounded V10.1 pretraining with reproducible checkpoints and telemetry.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/train.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/train.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/train.py>)

Imports:

```python
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.prepare_data import digest
from v10_10M.train import atomic_json,canonical_ids,build_optimizers,memory_stats
from .model import V101,Architecture
from .accumulate import backward_effective_batch
from .evaluate import evaluate
from .prepare_data import DATA
from .provenance import provenance
```

Symbols:

- [identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py:30>) — lines 30–35
- [verify_data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py:38>) — lines 38–43
- [gradient_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py:46>) — lines 46–62
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py:65>) — lines 65–181
- [parser](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/train.py:184>) — lines 184–194

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_16h'))
p.add_argument('--data-dir',default=str(DATA))
p.add_argument('--hours',type=float,default=16)
p.add_argument('--device',choices=['cpu','mps'],default='mps')
p.add_argument('--seed',type=int,default=1012)
p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--microbatch-size',type=int,default=1)
p.add_argument('--context',type=int,default=384)
p.add_argument('--lr',type=float,default=3e-4)
p.add_argument('--candidate-every',type=int,default=4)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--memory-fraction',type=float,default=.55)
p.add_argument('--log-every',type=int,default=20)
p.add_argument('--checkpoint-seconds',type=float,default=600)
p.add_argument('--eval-seconds',type=float,default=1800)
p.add_argument('--max-steps',type=int,default=0)
p.add_argument('--resume',action='store_true')
p.add_argument('--no-eval',action='store_true')
```

## v10_1/validate_default_batch.py

51 lines; SHA-256 `7f62b912f9c70ac456115b842ce107a1241a763d9271c8acb1a98877a680cd37`.

Primary copy: [v10_1/validate_default_batch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_default_batch.py>).

One bounded default-size Metal optimizer update under a temporary memory cap.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/validate_default_batch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/validate_default_batch.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/validate_default_batch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/validate_default_batch.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/validate_default_batch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/validate_default_batch.py>)

Imports:

```python
import json
import time
from pathlib import Path
import psutil
import torch
from .model import V101,losses
from .validate_training import load_examples,pack,clip_and_log
from .provenance import provenance
from .accumulate import backward_effective_batch
from v10_10M.train import canonical_ids,build_optimizers
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_default_batch.py:14>) — lines 14–46

CLI arguments:

```python
p.add_argument('--microbatch-size',type=int)
```

## v10_1/validate_training.py

139 lines; SHA-256 `b798b6c6179259feb4603b7da12995804412f2c6437a2241cec30ffdfc3fa0b1`.

Primary copy: [v10_1/validate_training.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py>).

Bounded real-data integration runs; explicitly not a quality comparison. Uses cleaned train/dev examples with the existing tokenizer for compatibility tests only. No active arrays or sealed-test contents are read/write-modified. Records per-step telemetry and matched CPU timing. Production clean-tokenizer pretraining and sustained exclusive-device benchmarks remain separate work.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_1/validate_training.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_1/validate_training.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_1/validate_training.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/validate_training.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_1/validate_training.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/validate_training.py>)

Imports:

```python
import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.train import canonical_ids,build_optimizers
from v10_10M.data_tools.clean_math import clean_math_field
from .model import V101,Architecture,losses,token_loss
from .provenance import provenance
```

Symbols:

- [load_examples](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:29>) — lines 29–57
- [pack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:60>) — lines 60–65
- [objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:68>) — lines 68–71
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:75>) — lines 75–80
- [clip_and_log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:83>) — lines 83–90
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validate_training.py:93>) — lines 93–134

CLI arguments:

```python
p.add_argument('--steps',type=int,default=16)
p.add_argument('--arms',nargs='+',default=['A','B','C','D','S'])
```

## v10_10M/__init__.py

1 lines; SHA-256 `8400e3844cce4f76ff8480026651e6d1d67793de039ec529ce5d41652a20daba`.

Primary copy: [v10_10M/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/__init__.py>).

V10: causal encoder, recurrent concept prediction, CRC decoder.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/__init__.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/__init__.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/__init__.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/__init__.py>)

## v10_10M/adapt.py

47 lines; SHA-256 `3ac581f216031c6cd408747d7ee0b8c4de981938d11b47b7ded617ac3ce09997`.

Primary copy: [v10_10M/adapt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/adapt.py>).

Explicit post-pretraining RBF adaptation on permitted training feedback only.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/adapt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/adapt.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/adapt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/adapt.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/adapt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/adapt.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/adapt.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/adapt.py>)

Imports:

```python
import argparse
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .model import V10, losses
from .memory import RBFMemory, learn_episode
from .data import Corpus
from .prepare_data import DATA, digest
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/adapt.py:13>) — lines 13–44

CLI arguments:

```python
p.add_argument('checkpoint')
p.add_argument('--output',required=True)
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--family',choices=['text','math','sudoku_easy','sudoku_hard'],default='math')
p.add_argument('--device',default='mps')
p.add_argument('--eta',type=float,default=.1)
p.add_argument('--sigma',type=float,default=1.)
```

## v10_10M/audit_data.py

45 lines; SHA-256 `91823d9b394a36777879de57bfa425bada5145cd025172488bd95498376c7265`.

Primary copy: [v10_10M/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit_data.py>).

Validate actual stored records and disjoint identities before launch.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/audit_data.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/audit_data.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/audit_data.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/audit_data.py>)

Imports:

```python
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from .prepare_data import DATA, digest
from .data import Corpus, verify_sudoku
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit_data.py:10>) — lines 10–42

## v10_10M/benchmark.py

41 lines; SHA-256 `cabf315f0a1e7f26f3eeb532a46d15b9dc0eb1efb29aa5ae2d5c0772a3a23bff`.

Primary copy: [v10_10M/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/benchmark.py>).

Measure actual end-to-end updates on each corpus family and batch size.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/benchmark.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/benchmark.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/benchmark.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/benchmark.py>)

Imports:

```python
import argparse
import json
import time
from pathlib import Path
import torch
from .model import V10, losses
from .data import Corpus
from .train import build_optimizers, clip, memory_stats
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/benchmark.py:12>) — lines 12–38

CLI arguments:

```python
p.add_argument('--device',default='mps')
```

## v10_10M/data.py

70 lines; SHA-256 `0d63b2c4a591d62c4000fefa5f7b08334b438faa8743b89fd68e54a65faccd65`.

Primary copy: [v10_10M/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data.py>).

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/data.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/data.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/data.py>)

Imports:

```python
import json
from pathlib import Path
import numpy as np
import torch
from .prepare_data import DATA
```

Symbols:

- [Corpus](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data.py:8>) — lines 8–51
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data.py:9>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data.py:19>).
- [verify_sudoku](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data.py:54>) — lines 54–70

## v10_10M/data_tools/__init__.py

1 lines; SHA-256 `c66cf383e6628efb2713bbd7d8f8b3e07199a9be74c8182fe97cae5a42d57ce0`.

Primary copy: [v10_10M/data_tools/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/__init__.py>).

Preparation fixes isolated from the frozen active training source.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/__init__.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/__init__.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/__init__.py>)

## v10_10M/data_tools/clean_math.py

18 lines; SHA-256 `3c2b7d879ad02a850fa04d677d009c1e6acc0fbbaea8cc9d04a10aa109f4d3a9`.

Primary copy: [v10_10M/data_tools/clean_math.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/clean_math.py>).

Normalize the upstream parquet's serialized Python byte literals safely.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/clean_math.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/clean_math.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/clean_math.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/clean_math.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/clean_math.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/clean_math.py>)

Imports:

```python
import ast
```

Symbols:

- [clean_math_field](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/clean_math.py:5>) — lines 5–18

## v10_10M/data_tools/prepare_v2.py

42 lines; SHA-256 `dfd3b7ac65e7b0ad0be75e0a2b7eb51ca17d27fc7846bac4de900fe99b84d134`.

Primary copy: [v10_10M/data_tools/prepare_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/prepare_v2.py>).

Build a NEW clean corpus without changing the active run or its tokenizer. Uses the pinned original parquet files with corrected field normalization. Refuses an existing destination. This does NOT launch or resume training.

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/prepare_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/prepare_v2.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/prepare_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/prepare_v2.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/prepare_v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/prepare_v2.py>)

Imports:

```python
import argparse
from pathlib import Path
import shutil
import types
import pyarrow as pa
from .. import prepare_data as base
from .clean_math import clean_math_field
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/prepare_v2.py:15>) — lines 15–39

CLI arguments:

```python
p.add_argument('--output',required=True)
```

## v10_10M/data_tools/tests.py

17 lines; SHA-256 `9eafd15b82191edcbe4360a291537165432ca0e99b32aa1682aebd1bf9074657`.

Primary copy: [v10_10M/data_tools/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/tests.py>).

Identical copies:

- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/data_tools/tests.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/data_tools/tests.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/data_tools/tests.py>)

Imports:

```python
import unittest
from .clean_math import clean_math_field
```

Symbols:

- [MathNormalizationTests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/tests.py:5>) — lines 5–14
  Methods: [test_native_and_serialized_fields](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/tests.py:6>), [test_invalid_input](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/data_tools/tests.py:12>).

## v10_10M/evaluate.py

95 lines; SHA-256 `1029b8924ba8f2b16718422e8453ff8558a028f9b2bb25a263d85ea58c8cc742`.

Primary copy: [v10_10M/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/evaluate.py>).

Development NTP and unconstrained greedy symbolic evaluation. No memory writes.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/evaluate.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/evaluate.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/evaluate.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/evaluate.py>)

Imports:

```python
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from .model import losses
from .data import Corpus, verify_sudoku
from .prepare_data import DATA
```

Symbols:

- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/evaluate.py:15>) — lines 15–26
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/evaluate.py:30>) — lines 30–83

CLI arguments:

```python
p.add_argument('checkpoint')
p.add_argument('--device',default='mps')
p.add_argument('--split',choices=['dev','test'],default='dev')
p.add_argument('--output',required=True)
```

## v10_10M/launch.py

40 lines; SHA-256 `da316e4eadca11742c880a78fe316f4d3dc5ff6d4b4ed28b3f8bdfc3815faa2a`.

Primary copy: [v10_10M/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/launch.py>).

Launch once, hold an OS process lock, prevent idle sleep and retain logs.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/launch.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/launch.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/launch.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/launch.py>)

Imports:

```python
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .prepare_data import ROOT
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/launch.py:13>) — lines 13–37

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
p.add_argument('--resume',action='store_true')
p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--accumulate',type=int,default=1)
```

## v10_10M/prepare_data.py

189 lines; SHA-256 `f4a76ad19b8c1ef2f29954c26c7f7d5db6eb5a594967657b09e2f0b9c0d9d35c`.

Primary copy: [v10_10M/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py>).

Prepare independently sourced text/math/Sudoku, with provenance and sealed tests. Only data bytes are fetched; no remote dataset Python is executed.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/prepare_data.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/prepare_data.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/prepare_data.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/prepare_data.py>)

Imports:

```python
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:26>) — lines 26–31
- [key](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:34>) — lines 34–35
- [download](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:38>) — lines 38–48
- [sudoku_records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:51>) — lines 51–74
- [text_records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:77>) — lines 77–95
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/prepare_data.py:98>) — lines 98–185

## v10_10M/supervise.py

21 lines; SHA-256 `0ac500be65a6439d15c19bd39ea244790457386354e9e2f7d013da2560a83dac`.

Primary copy: [v10_10M/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/supervise.py>).

Run lock and immutable source snapshot; trainer owns deadline/checkpoints.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/supervise.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/supervise.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/supervise.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/supervise.py>)

Imports:

```python
import argparse
import fcntl
from pathlib import Path
import shutil
from .prepare_data import ROOT
from .train import main, parser
```

## v10_10M/tests.py

141 lines; SHA-256 `cba0348c24a5927e62c6db264a6ea47876ab14e701d036ad530656f3972c49d1`.

Primary copy: [v10_10M/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/tests.py>).

Behavioral contracts for causality, conditioning, recurrence and memory learning.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/tests.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/tests.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/tests.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/tests.py>)

Imports:

```python
import json
import math
import time
from dataclasses import replace
from pathlib import Path
import torch
from .config import ModelConfig
from .model import V10, losses, Decoder
from .memory import RBFMemory, learn_episode
from .data import verify_sudoku
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/tests.py:14>) — lines 14–132

CLI arguments:

```python
parser.add_argument('--device',default='cpu')
```

## v10_10M/train.py

223 lines; SHA-256 `8586cfea18af112ab26f78b45de09247ba4307fd9585bd860419ee27ea6b6999`.

Primary copy: [v10_10M/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py>).

A deadline-bounded, resumable 24-hour M2/MPS pretraining run.

Identical copies:

- [v10_10M/runs/pretrain_24h/source_snapshot/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/runs/pretrain_24h/source_snapshot/train.py>)
- [v10_1/runs/pretrain_16h/source_snapshot/v10_10M/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h/source_snapshot/v10_10M/train.py>)
- [v10_1/validation/launch_mps/source_snapshot/v10_10M/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_10M/train.py>)
- [v10_2/validation/production_smoke/source_snapshot/v10_10M/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_10M/train.py>)

Imports:

```python
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import psutil
import torch
from tokenizers import Tokenizer
import unicodedata
from .config import ModelConfig
from .model import V10, losses
from .data import Corpus
from .evaluate import evaluate
from .prepare_data import DATA, ROOT, digest
```

Symbols:

- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:25>) — lines 25–27
- [source_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:30>) — lines 30–34
- [canonical_ids](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:37>) — lines 37–48
- [memory_stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:51>) — lines 51–57
- [build_optimizers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:60>) — lines 60–70
- [clip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:73>) — lines 73–83
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:86>) — lines 86–199
- [parser](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/train.py:202>) — lines 202–220

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
p.add_argument('--device',choices=['cpu','mps'],default='mps')
p.add_argument('--hours',type=float,default=24)
p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--accumulate',type=int,default=1)
p.add_argument('--context',type=int,default=384)
p.add_argument('--lr',type=float,default=3e-4)
p.add_argument('--seed',type=int,default=1011)
p.add_argument('--candidate-every',type=int,default=4)
p.add_argument('--log-every',type=int,default=20)
p.add_argument('--checkpoint-seconds',type=float,default=600)
p.add_argument('--eval-seconds',type=float,default=1800)
p.add_argument('--max-steps',type=int,default=0)
p.add_argument('--resume',action='store_true')
p.add_argument('--no-eval',action='store_true')
p.add_argument('--skip-initial-eval',action='store_true')
```

## v10_1/telemetry.py

92 lines; SHA-256 `4183ecdd0e12f5a7d9eaa00f882b47e6e028ff01cbf550ac08b24b3e057c943d`.

Primary copy: [v10_1/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py>).

V10.1 adapter for the tested read-only SQLite collector and dashboard.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_1/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_1/telemetry.py>)

Imports:

```python
from pathlib import Path
import time
import psutil
from urllib.parse import urlparse
from v10_10M.telemetry import server,store
```

Symbols:

- [make_handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:12>) — lines 12–22
- [Store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:25>) — lines 25–42
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:26>), [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:33>).
- [Monitor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:45>) — lines 45–75
  Methods: [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:46>).
- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry.py:78>) — lines 78–88

## v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py

70 lines; SHA-256 `e7bbb812bfdba7eb31092dfe47106701633d44e618ee2a687d3af52bd4cc5463`.

Primary copy: [v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py>).

V10.1 adapter for the tested read-only SQLite collector and dashboard.

Imports:

```python
from pathlib import Path
import time
import psutil
from v10_10M.telemetry import server,store
```

Symbols:

- [Store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py:10>) — lines 10–20
  Methods: [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py:11>).
- [Monitor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py:23>) — lines 23–53
  Methods: [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py:24>).
- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/validation/launch_mps/source_snapshot/v10_1/telemetry.py:56>) — lines 56–66

## v10_10M/audit/__init__.py

1 lines; SHA-256 `4870c73b9823cda5b6cf920263bb63d9f82259a7ee056952ee849e4a6cc663e0`.

Primary copy: [v10_10M/audit/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/__init__.py>).

Explicit, offline checkpoint diagnostics; separate from training telemetry.

## v10_10M/audit/balanced_evaluation.py

50 lines; SHA-256 `317545831c1ceb3a21843dcb336ae3adfe0abd2f66a402988bef9b521d93fd16`.

Primary copy: [v10_10M/audit/balanced_evaluation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/balanced_evaluation.py>).

Stratified development generation, without teacher answers or output repair.

Imports:

```python
import json
from pathlib import Path
import time
import torch
import numpy as np
from tokenizers import Tokenizer
from ..prepare_data import ROOT,DATA
from ..config import ModelConfig
from ..model import V10
from ..data import Corpus,verify_sudoku
from ..evaluate import generate
from ..telemetry.store import atomic_json
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/balanced_evaluation.py:16>) — lines 16–47

## v10_10M/audit/checkpoint.py

109 lines; SHA-256 `39dc5fff8cd9f3663aae393df82931a74f53d07d733400c8723252f65c0e517a`.

Primary copy: [v10_10M/audit/checkpoint.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/checkpoint.py>).

One-off CPU checkpoint probes; these are NOT historical training gradients.

Imports:

```python
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from ..config import ModelConfig
from ..model import V10, losses
from ..data import Corpus
from ..evaluate import generate
from ..prepare_data import ROOT, DATA
from ..train import source_hash
from ..telemetry.store import atomic_json
```

Symbols:

- [grad_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/checkpoint.py:20>) — lines 20–32
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/checkpoint.py:35>) — lines 35–106

CLI arguments:

```python
p.add_argument('--checkpoint',default=str(ROOT/'runs/pretrain_24h/latest.pt'))
p.add_argument('--output',default=str(ROOT/'audit/checkpoint_probe.json'))
```

## v10_10M/audit/review_telemetry.py

130 lines; SHA-256 `0794735fcf2be1fda69acbbcd8876b37b7632cbecce40e945def04c5c612fe06`.

Primary copy: [v10_10M/audit/review_telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py>).

Read-only progress review plus bounded CPU development checks. Does not update telemetry's canonical audit files, train, or use the sealed test.

Imports:

```python
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo
import numpy as np
import torch
from tokenizers import Tokenizer
from ..config import ModelConfig
from ..data import Corpus,verify_sudoku
from ..evaluate import generate
from ..model import V10,losses
from ..prepare_data import ROOT,DATA
from ..telemetry.store import atomic_json
```

Symbols:

- [api](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:26>) — lines 26–27
- [events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:30>) — lines 30–34
- [sudoku_detail](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:37>) — lines 37–44
- [telemetry_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:47>) — lines 47–67
- [text_check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:71>) — lines 71–92
- [balanced_check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:96>) — lines 96–117
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/audit/review_telemetry.py:120>) — lines 120–128

CLI arguments:

```python
p.add_argument('--output-dir',required=True)
```

## v10_10M/telemetry/__init__.py

3 lines; SHA-256 `b0c31ba3611b5b764f79399e10dc1891e52d425b4060d635fedc68014aef12e9`.

Primary copy: [v10_10M/telemetry/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/__init__.py>).

Read-only training telemetry, independent of the running trainer.

## v10_10M/telemetry/export.py

37 lines; SHA-256 `88178b65f666ac42d0ad2027375960f0cb9b5bee48fc3e3529ea202595825949`.

Primary copy: [v10_10M/telemetry/export.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/export.py>).

Consistent offline JSON/JSONL export for UI development and analysis.

Imports:

```python
import argparse
from pathlib import Path
from .server import DEFAULT_DATA
from .store import Store, strict_json, encode, atomic_json
from . import SCHEMA_VERSION
```

Symbols:

- [export](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/export.py:9>) — lines 9–28
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/export.py:31>) — lines 31–34

CLI arguments:

```python
p.add_argument('--database',default=str(DEFAULT_DATA/'telemetry.sqlite3'))
p.add_argument('--run-id',default='pretrain_24h')
p.add_argument('--output',default=str(DEFAULT_DATA/'export'))
```

## v10_10M/telemetry/launch.py

30 lines; SHA-256 `327b22be0b1253c23ed38298e7aafd441f5b1b6306edd1e515316e80754110b8`.

Primary copy: [v10_10M/telemetry/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/launch.py>).

Start the independent telemetry collector without touching the trainer.

Imports:

```python
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from .server import ROOT, DEFAULT_DATA
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/launch.py:12>) — lines 12–27

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
p.add_argument('--data-dir',default=str(DEFAULT_DATA))
p.add_argument('--port',type=int,default=8766)
```

## v10_10M/telemetry/server.py

226 lines; SHA-256 `f8ec71fefe5a2e0726d01219815b681bc563531d264b0d46053ea6011d5a2578`.

Primary copy: [v10_10M/telemetry/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py>).

Read-only local API and independent incremental training-log collector.

Imports:

```python
import argparse
import csv
import fcntl
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import parse_qs, urlparse
import psutil
from . import SCHEMA_VERSION
from .store import Store, atomic_json, encode, strict_json, METRIC_AVAILABILITY
```

Symbols:

- [Monitor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:26>) — lines 26–70
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:27>), [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:32>).
- [Collector](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:73>) — lines 73–100
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:74>), [once](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:80>), [loop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:89>), [health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:97>).
- [make_handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:103>) — lines 103–197
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/server.py:200>) — lines 200–223

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
p.add_argument('--data-dir',default=str(DEFAULT_DATA))
p.add_argument('--port',type=int,default=8766)
p.add_argument('--poll',type=float,default=5)
p.add_argument('--host-interval',type=float,default=30)
```

## v10_10M/telemetry/store.py

379 lines; SHA-256 `4d66a0862930ab491d2e5946886ea8e1217902866f36c33411e4e797a6ad0d07`.

Primary copy: [v10_10M/telemetry/store.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py>).

Incremental, durable ingestion. This module never imports torch or checkpoints.

Imports:

```python
import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time
from . import SCHEMA_VERSION
```

Symbols:

- [strict_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:12>) — lines 12–19
- [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:22>) — lines 22–23
- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:26>) — lines 26–30
- [finite_number](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:33>) — lines 33–34
- [flatten_numeric](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:37>) — lines 37–47
- [training_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:50>) — lines 50–74
- [evaluation_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:77>) — lines 77–106
- [Store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:109>) — lines 109–356
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:110>), [connect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:149>), [issue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:158>), [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:161>), [ingest_training](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:174>), [ingest_evaluations](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:218>), [ingest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:237>), [runs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:283>), [events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:289>), [series](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:300>), [catalog](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:306>), [latest_evaluation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:312>), [issues](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:317>), [artifacts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:322>), [overview](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/store.py:327>).

## v10_10M/telemetry/tests.py

133 lines; SHA-256 `c71954171fad662b15360663633fff13db2cf118c456ccff50bfee8cf43def36`.

Primary copy: [v10_10M/telemetry/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py>).

Failure-mode and API contract tests; no model or active-run mutation.

Imports:

```python
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from .store import Store, strict_json, evaluation_metrics
from .server import Collector, make_handler
```

Symbols:

- [TelemetryTests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:14>) — lines 14–130
  Methods: [setUp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:15>), [tearDown](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:22>), [record](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:24>), [ingest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:28>), [test_partial_tail_restart_duplicate_step_and_nonfinite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:30>), [test_truncation_and_rotation_idempotence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:46>), [test_evaluation_rewrite_zero_trials_and_research_alerts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:55>), [test_http_pagination_errors_cors_and_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:69>), [test_offline_probe_namespace_and_export](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:108>), [test_malformed_records_are_quarantined](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/tests.py:124>).

## v10_10M/telemetry/web/qa_server.py

28 lines; SHA-256 `5d27ee2579c8cf512181538faf87ee82c81569ddcc212b2b3f21721769414750`.

Primary copy: [v10_10M/telemetry/web/qa_server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/web/qa_server.py>).

Isolated UI failure-mode fixture. Never loads torch or writes to a training run. Run from the project root: python -m v10_10M.telemetry.web.qa_server Change telemetry/data/qa_scenario.txt to zero, empty, offline, or stale.

Imports:

```python
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,time
from ..server import WEB,WEB_ASSETS,DEFAULT_DATA
from ..store import METRIC_AVAILABILITY
```

Symbols:

- [Handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/web/qa_server.py:11>) — lines 11–27
  Methods: [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/web/qa_server.py:12>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_10M/telemetry/web/qa_server.py:13>).

## v10_1_design_evidence/2026-10-11/collect_data_evidence.py

80 lines; SHA-256 `7e7775d78b2126e3499caeb8d4cd48b6d79adef2a5960cfdfa09a6bc9a4cc33f`.

Primary copy: [v10_1_design_evidence/2026-10-11/collect_data_evidence.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/collect_data_evidence.py>).

Read only active train/dev data; never reads sealed test contents or modifies a run.

Imports:

```python
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_10M.data_tools.clean_math import clean_math_field
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/collect_data_evidence.py:17>) — lines 17–18
- [key](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/collect_data_evidence.py:20>) — lines 20–21

## v10_1_design_evidence/2026-10-11/probe_interfaces.py

65 lines; SHA-256 `012897db5b03426144dbcb4955727e01f93bb1f8c4a3859ad11909a37918854c`.

Primary copy: [v10_1_design_evidence/2026-10-11/probe_interfaces.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/probe_interfaces.py>).

Small CPU checkpoint interventions; not retrained architecture comparisons.

Imports:

```python
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.model import V10, losses
```

## v10_1_design_evidence/2026-10-11/review_checks.py

139 lines; SHA-256 `3f2a18aeed63d2a6fe2cfe9c0a208338e09d2bb528a7fb0110c4163ed004aa09`.

Primary copy: [v10_1_design_evidence/2026-10-11/review_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/review_checks.py>).

Specification checks only. Does not implement/train V10.1 or touch active state.

Imports:

```python
import hashlib
import json
import re
from pathlib import Path
import torch
from torch import nn
from v10_10M.config import ModelConfig
from v10_10M.model import V10, Block, RMSNorm, Decoder, geometry
```

Symbols:

- [auxiliary_hidden](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/review_checks.py:69>) — lines 69–72
- [loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1_design_evidence/2026-10-11/review_checks.py:95>) — lines 95–99

## v10_2/__init__.py

1 lines; SHA-256 `2e3ac6608c0278c3150bd033a3ec0225a70a8fd5eb0382779b216ac2531ee019`.

Primary copy: [v10_2/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/__init__.py>).

V10.2 RCDL: causal concept history and forecast-only supervision.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/__init__.py>)

## v10_2/accumulate.py

57 lines; SHA-256 `119dfd5d5e52faa67727050992a9ff1dc336698e45f802e38f1c59c68ebb6cb4`.

Primary copy: [v10_2/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/accumulate.py>).

CPU boundary/hash preparation; objective-correct effective-batch accumulation.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/accumulate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/accumulate.py>)

Imports:

```python
import time
import torch
from .model import losses,chunk_batch
from v10_1.optimization.fast import gradient_metrics
```

Symbols:

- [backward_effective_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/accumulate.py:8>) — lines 8–57

## v10_2/benchmark.py

46 lines; SHA-256 `f9ca1cbbb372ce60536757ed3c2538f19b453b94530d00582f0e2b8cadf14308`.

Primary copy: [v10_2/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/benchmark.py>).

Real-corpus, full-parameter Metal benchmark. Does not touch training runs.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/benchmark.py>)

Imports:

```python
import argparse,json,time,gc,statistics
from pathlib import Path
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.train import build_optimizers,canonical_ids,memory_stats
from .model import V102
from .accumulate import backward_effective_batch,gradient_metrics
from .prepare_data import DATA
from .provenance import provenance
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/benchmark.py:14>) — lines 14–41

CLI arguments:

```python
p.add_argument('--steps',type=int,default=24)
p.add_argument('--microbatch',type=int,default=8)
p.add_argument('--foreach',action='store_true')
p.add_argument('--output',required=True)
```

## v10_2/evaluate.py

168 lines; SHA-256 `5edca09a94b58ad748defbb6115311be97cad452f50c6360c7bdbdd13136aee0`.

Primary copy: [v10_2/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py>).

Fixed, balanced development panels; explicit route ablations and strict solves.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/evaluate.py>)

Imports:

```python
import json
import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus,verify_sudoku
from v10_10M.evaluate import generate
from .model import token_loss,losses,chunk_batch,future_ce
from .prepare_data import DATA
```

Symbols:

- [parse_answer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py:18>) — lines 18–28: Exact numeric grammar, including ordered lists; no answer repair.
- [score_math](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py:31>) — lines 31–34
- [generation_indices](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py:37>) — lines 37–41
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py:45>) — lines 45–128
- [gradient_probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/evaluate.py:131>) — lines 131–168: Fixed text probe; no optimizer step or .grad mutation. Dense gradients only.

## v10_2/integration_check.py

22 lines; SHA-256 `4b2cd0fe4fd6800c4ed8aa72e4462731ad4047cb3e3016a7d15e51f162149855`.

Primary copy: [v10_2/integration_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/integration_check.py>).

Exercise the full-size evaluator, generation, and gradient probes on a smoke checkpoint.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/integration_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/integration_check.py>)

Imports:

```python
import argparse,json,time
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from .model import V102,Architecture
from .evaluate import evaluate
from .prepare_data import DATA
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/integration_check.py:10>) — lines 10–21

CLI arguments:

```python
p.add_argument('--checkpoint',required=True)
```

## v10_2/launch.py

37 lines; SHA-256 `76e0a5be552277a9bdfb0ea79da1c7f7c261058cf025efbfe2e928437194d367`.

Primary copy: [v10_2/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/launch.py>).

Detach the locked trainer, prevent idle sleep, and verify startup.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/launch.py>)

Imports:

```python
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .train import parser,ROOT
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/launch.py:11>) — lines 11–34

## v10_2/optimizer_parity.py

28 lines; SHA-256 `d72698536b443b2da6c3fe3178899c2a2292f22550cd5e793c217be1a03c5532`.

Primary copy: [v10_2/optimizer_parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/optimizer_parity.py>).

Verify the chosen foreach optimizer matches the scalar optimizer on Metal.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/optimizer_parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/optimizer_parity.py>)

Imports:

```python
from dataclasses import replace
from pathlib import Path
import json,torch
from .model import V102
from .accumulate import backward_effective_batch,gradient_metrics
from v10_10M.config import ModelConfig
from v10_10M.train import build_optimizers
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/optimizer_parity.py:10>) — lines 10–27

## v10_2/parity.py

23 lines; SHA-256 `8d1e3ebea24b6a32c72f1f331246e610e0b380f43e6eed5d572060fd12bb7c80`.

Primary copy: [v10_2/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/parity.py>).

Independent CPU versus Metal loss/all-gradient comparison, including sparse table.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/parity.py>)

Imports:

```python
from dataclasses import replace
import json,torch
from pathlib import Path
from v10_10M.config import ModelConfig
from .model import V102
from .accumulate import backward_effective_batch
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/parity.py:9>) — lines 9–22

## v10_2/pipeline_tests.py

60 lines; SHA-256 `54119c84b0edf9174805df040ad92aff80031241564be110625bc49a52af9ea6`.

Primary copy: [v10_2/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/pipeline_tests.py>).

Clean-data identity, scoring, telemetry persistence and production resume checks.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/pipeline_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/pipeline_tests.py>)

Imports:

```python
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .evaluate import parse_answer,score_math,generation_indices
from v10_1.prepare_data import DATA,ORIGINAL,FAMILIES
from .telemetry import Store
from .train import verify_data
from .provenance import provenance
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/pipeline_tests.py:14>) — lines 14–57

## v10_2/prepare_data.py

3 lines; SHA-256 `f6e66eae88c0adf2850e3fce779306f1b3aeac94acab8af1c69378c9bb37ddfc`.

Primary copy: [v10_2/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/prepare_data.py>).

Reuse the immutable, cleaned V10.1 corpus and tokenizer; never rebuild in place.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/prepare_data.py>)

Imports:

```python
from pathlib import Path
```

## v10_2/provenance.py

17 lines; SHA-256 `00dd80e3f6eb2c91869a0f20513d0f558862ed0176ebf4c118fc272e2e1f6913`.

Primary copy: [v10_2/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/provenance.py>).

Hash every imported project Python dependency, including nested optimizations.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/provenance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/provenance.py>)

Imports:

```python
import hashlib
from pathlib import Path
import torch
```

Symbols:

- [source_files](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/provenance.py:7>) — lines 7–13
- [provenance](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/provenance.py:15>) — lines 15–17

## v10_2/supervise.py

27 lines; SHA-256 `32b96bc81e844b372fc40800a1211ff87224c74594a38dc451a5af1e0c56e0aa`.

Primary copy: [v10_2/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/supervise.py>).

Exclusive run lock and reproducible source snapshot; no automatic restart.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/supervise.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/supervise.py>)

Imports:

```python
import fcntl
from pathlib import Path
import shutil
from .train import main,parser,ROOT
```

Symbols:

- [supervise](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/supervise.py:8>) — lines 8–24

## v10_2/telemetry.py

98 lines; SHA-256 `f960124392591eb73727fb66fe0969d392c1e1a5d030e918d7dc57a9939085c9`.

Primary copy: [v10_2/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py>).

V10.2 adapter for the tested read-only SQLite collector and dashboard.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/telemetry.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/telemetry.py>)

Imports:

```python
from pathlib import Path
import time
import psutil
from urllib.parse import urlparse
from v10_10M.telemetry import server,store
```

Symbols:

- [make_handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:12>) — lines 12–24
- [Store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:27>) — lines 27–46
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:28>), [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:35>).
- [Monitor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:49>) — lines 49–79
  Methods: [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:50>).
- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry.py:82>) — lines 82–94

## v10_2/telemetry_tests.py

135 lines; SHA-256 `d1cf724ac8866d537cf8dd79353212e1ab282c4200efd2d99d4c947b0584d1b6`.

Primary copy: [v10_2/telemetry_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py>).

Failure-mode and API contract tests; no model or active-run mutation.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/telemetry_tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/telemetry_tests.py>)

Imports:

```python
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from v10_10M.telemetry.store import strict_json, evaluation_metrics
from .telemetry import Store,configure
from v10_10M.telemetry.server import Collector, make_handler
```

Symbols:

- [TelemetryTests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:16>) — lines 16–132
  Methods: [setUp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:17>), [tearDown](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:24>), [record](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:26>), [ingest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:30>), [test_partial_tail_restart_duplicate_step_and_nonfinite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:32>), [test_truncation_and_rotation_idempotence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:48>), [test_evaluation_rewrite_zero_trials_and_research_alerts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:57>), [test_http_pagination_errors_cors_and_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:71>), [test_offline_probe_namespace_and_export](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:110>), [test_malformed_records_are_quarantined](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_tests.py:126>).

## v10_2/tests.py

249 lines; SHA-256 `0a6242890e794d34158c9dd598599332aaaa0c240b5fdbc226df02fbea985d05`.

Primary copy: [v10_2/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/tests.py>).

Behavioral end-to-end contracts; small vocabulary/table, full width/depth.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/tests.py>)

Imports:

```python
import argparse
import copy
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.model import V10, losses as original_losses
from v10_10M.train import build_optimizers,clip
from .model import V102,Architecture,losses,token_loss,chunk_batch,history_geometry,future_ce
from .memory import RBFMemory,learn_episode,write_objective,model_era
from .provenance import provenance
from .accumulate import backward_effective_batch
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/tests.py:20>) — lines 20–241

CLI arguments:

```python
p.add_argument('--device',choices=['cpu','mps'],default='cpu')
p.add_argument('--output',required=True)
```

## v10_2/train.py

188 lines; SHA-256 `9fb5b81157c1c94b77da67f43535c6031b04e4447101eca685aabeed9ae983d9`.

Primary copy: [v10_2/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/train.py>).

Deadline-bounded V10.2 pretraining with reproducible checkpoints and telemetry.

Identical copies:

- [v10_2/validation/production_smoke/source_snapshot/v10_2/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/validation/production_smoke/source_snapshot/v10_2/train.py>)

Imports:

```python
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.prepare_data import digest
from v10_10M.train import atomic_json,canonical_ids,build_optimizers,memory_stats
from .model import V102,Architecture
from .accumulate import backward_effective_batch,gradient_metrics
from .evaluate import evaluate
from .prepare_data import DATA
from .provenance import provenance
```

Symbols:

- [identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/train.py:30>) — lines 30–35
- [verify_data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/train.py:38>) — lines 38–43
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/train.py:47>) — lines 47–172
- [parser](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/train.py:175>) — lines 175–185

CLI arguments:

```python
p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_16h'))
p.add_argument('--data-dir',default=str(DATA))
p.add_argument('--hours',type=float,default=16)
p.add_argument('--device',choices=['cpu','mps'],default='mps')
p.add_argument('--seed',type=int,default=1020)
p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--microbatch-size',type=int,default=8)
p.add_argument('--context',type=int,default=384)
p.add_argument('--lr',type=float,default=3e-4)
p.add_argument('--candidate-every',type=int,default=4)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--memory-fraction',type=float,default=.55)
p.add_argument('--log-every',type=int,default=20)
p.add_argument('--checkpoint-seconds',type=float,default=600)
p.add_argument('--eval-seconds',type=float,default=1800)
p.add_argument('--foreach',action='store_true')
p.add_argument('--max-steps',type=int,default=0)
p.add_argument('--resume',action='store_true')
p.add_argument('--no-eval',action='store_true')
```

## v10_2_design_evidence/2026-10-12/collect_evidence.py

52 lines; SHA-256 `655c9c6fc342c231ad136f3843ff066eea63fb6cd4e1a0c7d75e1183115a1bd0`.

Primary copy: [v10_2_design_evidence/2026-10-12/collect_evidence.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/collect_evidence.py>).

Read-only capture of a running experiment; never loads weights or GPU tensors.

Imports:

```python
from pathlib import Path
from collections import deque
from datetime import datetime,timezone
import hashlib,json,math,statistics,urllib.request
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/collect_evidence.py:10>) — lines 10–10
- [quantile](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/collect_evidence.py:24>) — lines 24–24
- [panel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/collect_evidence.py:25>) — lines 25–34

## v10_2_design_evidence/2026-10-12/review_checks.py

138 lines; SHA-256 `31d01d46392b1f0f0166c836ebb22d5c0639fc646abaeed24c8e15d3af78e249`.

Primary copy: [v10_2_design_evidence/2026-10-12/review_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py>).

CPU-only checks of the V10.2 specification, NOT a V10.2 model implementation.

Imports:

```python
from pathlib import Path
import hashlib,json,math,re,sys
import torch
from torch import nn
from torch.nn import functional as F
from v10_1.model import V101
from v10_10M.model import FFN,RMSNorm
from v10_10M.config import ModelConfig
```

Symbols:

- [ForecastReadout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:14>) — lines 14–20
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:15>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:18>).
- [ReferenceDoubleNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:22>) — lines 22–26: Float64 mathematical norm for finite differences, without FP32 casts.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:24>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:26>).
- [ConceptReader](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:28>) — lines 28–43
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:29>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:32>).
- [release_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:45>) — lines 45–48
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_design_evidence/2026-10-12/review_checks.py:50>) — lines 50–136

## v10_2_review_evidence/2026-10-12/check_review_links.py

13 lines; SHA-256 `b5d913e5a8bc0921fe939761d2d0d287364b7bf478b6d6e355896424b31ee8dc`.

Primary copy: [v10_2_review_evidence/2026-10-12/check_review_links.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/check_review_links.py>).

Check that every local link in v10_2_review.md resolves and that code fences pair.

Imports:

```python
from pathlib import Path
import json, re, sys
```

## v10_2_review_evidence/2026-10-12/probe_episode_kernel.py

67 lines; SHA-256 `654d42c5ab9738df33c7ec7c8dc2226afc7cb34d9a1f6121d9c02ecd186988d1`.

Primary copy: [v10_2_review_evidence/2026-10-12/probe_episode_kernel.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_episode_kernel.py>).

Read-only companion probe: kernel spectrum of single adaptation episodes. The V10.2 protocol (§6.3) commits three support examples per episode. A write changes that episode's replayed corrections by -eta*K_episode*G, so the finite-step amplification is lambda_max of the episode's own code-partitioned Gram matrix. This samples 3-document episodes from the same frozen V10.1 checkpoint and dev panel as the companion probes. Output refuses to overwrite.

Imports:

```python
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_episode_kernel.py:29>) — lines 29–63

## v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.py

66 lines; SHA-256 `04e9ffd9f3e444d4b841844c8664b75f1d44de41cf3b792abcd09dbfae278b33`.

Primary copy: [v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.py>).

Read-only companion probe: spectrum of the code-partitioned RBF kernel. Uses the same frozen V10.1 checkpoint and the same 64/64/32/32 dev panel as probe_v10_1_interfaces.py (seed 20261012). For each family and sigma it builds the kernel Gram matrix of all panel queries within each code partition and reports its largest eigenvalue. A gradient write with step eta changes the replayed corrections by -eta*K*G, so finite-step stability scales with eta*lambda_max(K). Output refuses to overwrite.

Imports:

```python
from pathlib import Path
import hashlib, json, sys, time
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.py:28>) — lines 28–62

## v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py

173 lines; SHA-256 `a7ac676fdd5b4d0f87fe267a9003309018298d018edae2a6597f1d6720a9effd`.

Primary copy: [v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py>).

Read-only CPU probe of the stopped V10.1 checkpoint for the V10.2 review. Measures (1) decoder concept-route router weights and contribution magnitudes, (2) mask-and-renormalize versus value-zero ablations, (3) RBF query geometry and kernel neighbourhoods, and (4) a calibrated mean-substitution control. It never trains, never writes memory, and never modifies an existing file. Output: probe_v10_1_interfaces.json beside this script (refuses to overwrite).

Imports:

```python
from pathlib import Path
import hashlib, json, math, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture, token_loss
from v10_1.prepare_data import DATA
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:30>) — lines 30–34
- [draws](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:37>) — lines 37–39
- [MeanSubstitution](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:42>) — lines 42–53: Replace decoder concept inputs (forecast and continuous u) by fixed means.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:44>), [__enter__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:47>), [__exit__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:52>).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py:57>) — lines 57–169

## v10_3_design_evidence/2026-10-12/probe_candidate_utility.py

135 lines; SHA-256 `8a1b887f6f56c6eac390c902691a17f190315d84fabf01ecd133130086656814`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_candidate_utility.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py>).

Read-only probe: do V10.1's K=8 candidate codes differ, and is the hindsight winner predictable from the prefix? Uses the trainer's own path (chunk_batch -> candidate_forecasts -> per_document_chunk_ce + 0.1 * next-pool MSE) so utilities match what selected winners during training. For several eligible boundaries per document it records all eight utilities and reports: spread max - min utility over codes (nats of chunk CE, plus 0.1 MSE) gap second best - best regret utility of the prefix-selected code - best code effect share of utility variance (after removing each boundary's mean) explained by a per-code constant: high means a code helps or

Imports:

```python
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture, chunk_batch, per_document_chunk_ce
from v10_1.prepare_data import DATA
```

Symbols:

- [draws](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py:43>) — lines 43–45
- [boundaries_for](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py:49>) — lines 49–73
- [_eligible](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py:76>) — lines 76–80
- [summarize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py:83>) — lines 83–100
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_candidate_utility.py:103>) — lines 103–131

## v10_3_design_evidence/2026-10-12/probe_forecast_spread.py

88 lines; SHA-256 `e2d0b06b4510d0e91ce8bd67fc71a41402aa72a8128a0358f97a60972b1ad723`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_forecast_spread.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_forecast_spread.py>).

Read-only companion to probe_latent_baselines.py: why is the raw PQ forecast worse than a linear map of itself? Two mechanisms predict different things. If the soft-PQ convex combination shrinks the forecast toward the codebook centroid, var(F2)/var(target) is small and a per-coordinate affine fit (scale and shift only) recovers most of the ridge gain. If F lives in a rotated subspace, the affine fit does not help and only a full linear map does. Same checkpoint, panels and seeds as the baseline probe; step 61,384 only. Output refuses to overwrite.

Imports:

```python
from pathlib import Path
import hashlib, json, math, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [draws](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_forecast_spread.py:30>) — lines 30–32
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_forecast_spread.py:36>) — lines 36–47
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_forecast_spread.py:50>) — lines 50–84

## v10_3_design_evidence/2026-10-12/probe_gmm_gates.py

150 lines; SHA-256 `d0d4c5fdbafaa442ce4ddbb9b43f1e02f0777a9a41e0a3df5e899fae424433fb`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_gmm_gates.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py>).

Read-only probe: can Local-Support-Learning GMM gates separate data distributions inside the frozen V10.1 model? LSL (Ben-Kish et al., 2026) gates a low-rank adapter per weight matrix with open(x) = log Phi_pos(x) > log Phi_neg(x) where Phi_pos is a diagonal GMM fitted on the new phase's inputs to that matrix and Phi_neg on a small generic reference sample. The paper reports results on 1.5B-7B models and notes that retention depends on representation quality. This probe measures the gate at width 128. Sites are the inputs of attention (qkv) and FFN (up) matrices: e2.<i>.attn / e2.<i>.ffn Reader output of two-stream block i, per visit core.<i>

Imports:

```python
import os
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py:50>) — lines 50–77: acts\[site\]\[family\] -> list of per-document arrays \[rows, 128\] in sequence order.
- [split_docs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py:80>) — lines 80–82
- [fit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py:85>) — lines 85–88
- [rates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py:91>) — lines 91–99
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_gates.py:102>) — lines 102–146

## v10_3_design_evidence/2026-10-12/probe_gmm_modules.py

91 lines; SHA-256 `cdcf8bd46c4766c191a03fc5a65d5978ede8bc6be88bb5b7134e9db5f565a3ed`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_gmm_modules.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_modules.py>).

Read-only probe: do GMM gates separate phases within one data family? probe_gmm_gates.py treated whole families (text, math, Sudoku) as phases. A realistic continual-learning phase is narrower. Here each of the four math modules is a phase. For each held-out module, Phi_pos is fitted on half of its documents and Phi_neg on a reference covering everything else the model trained on (the other three modules, text and both Sudoku families). Reported: hit rate on the module's held-out documents and open rate on the other modules' held-out documents, raw per row (no smoothing; see the companion probe). Sites: E2 and concept-core attention/FFN input

Imports:

```python
import os
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
from probe_gmm_gates import fit, split_docs
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_modules.py:34>) — lines 34–53
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_gmm_modules.py:56>) — lines 56–87

## v10_3_design_evidence/2026-10-12/probe_latent_baselines.py

131 lines; SHA-256 `32c70cb47966d4c6e1ce4873289e882b55cff3e80d79eb59ba0c6e11480a40db`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_latent_baselines.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py>).

Read-only probe: does V10.1's concept forecast beat trivial baselines? The training "latent" loss is the MSE between the PQ forecast of group g and the stopped, online pooled E3 state of group g+1. Its absolute value can rise while prediction improves, because the target's spread changes during training. This probe scores the forecast against baselines on the same targets: mean per-coordinate mean of targets from a disjoint calibration draw persistence predict P3\[g+1\] by P3\[g\] ridge(P3) linear map P3\[g\] -> P3\[g+1\], fitted on the calibration draw ridge(U2) linear map from the final concept state U\[g\] -> P3\[g+1\] ridge(F2) linear map

Imports:

```python
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:40>) — lines 40–44
- [draws](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:47>) — lines 47–49
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:53>) — lines 53–63: Per document: arrays of (P3\[g\], U2\[g\], F1\[g\], F2\[g\], P3\[g+1\]) over valid pairs.
- [ridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:66>) — lines 66–70
- [per_doc_mse](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:73>) — lines 73–75
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_latent_baselines.py:78>) — lines 78–127

## v10_3_design_evidence/2026-10-12/probe_parameter_scales.py

38 lines; SHA-256 `68df59c2f9963c3e738f9245d213ca8aad98e61bc323fae1528d05beda17e2a5`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_parameter_scales.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_parameter_scales.py>).

Read-only: RMS of scale-bearing V10.1 parameters at two checkpoints, and whether each sits in the weight-decayed optimizer group (v10_10M/train.py build_optimizers decays every dense parameter with ndim >= 2). Also records the entropy of the hindsight-winner distribution from probe_candidate_utility.json, the best constant prior for the selector. Output refuses to overwrite.

Imports:

```python
from pathlib import Path
import hashlib, json, math
import torch
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_parameter_scales.py:19>) — lines 19–34

## v10_3_design_evidence/2026-10-12/probe_visit_delta.py

91 lines; SHA-256 `a6c943f0866d4fd150227001a62ff1f5d0cdb26440048e6fec8e3eea29f2ba53`.

Primary copy: [v10_3_design_evidence/2026-10-12/probe_visit_delta.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py>).

Read-only probe: what does the second concept visit add, and would the decoder gain anything from the visit-1 state or the change between visits? For the frozen step-61,384 checkpoint, per family: relative change RMS(U2 - U1) / RMS(U2) over valid groups cosine mean cosine similarity of U1 and U2 per slot predictive value FVU of ridge readouts of the next pooled state P3\[g+1\] from U1, U2, the delta U2 - U1, and \[U1, U2\] concatenated, fitted on the calibration draw and scored on the panel Same panels and seeds as probe_latent_baselines.py. Dev split, no training; output refuses to overwrite.

Imports:

```python
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
```

Symbols:

- [draws](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py:33>) — lines 33–35
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py:39>) — lines 39–49
- [ridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py:52>) — lines 52–55
- [fit_score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py:58>) — lines 58–61
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_design_evidence/2026-10-12/probe_visit_delta.py:64>) — lines 64–87

## v11_tiny/__init__.py

0 lines; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Primary copy: [v11_tiny/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/__init__.py>).

Identical copies:

- [v10_3_tiny/__init__.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/__init__.py>)

## v10_3_tiny/audit_data.py

87 lines; SHA-256 `6b23096b764bf048d11a9e42258694c1b1774c70bf9e385b3761bd989c9b1a6a`.

Primary copy: [v10_3_tiny/audit_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/audit_data.py>).

Does the tiny corpus carry structure aligned to the architecture's 8-token groups? python -B -m v10_3_tiny.audit_data -> v10_3_tiny/data_audit.json V9's GraphNav responses had a designed four-token step structure that matched V9's four-token concept chunks (review_v9.md:113), so its hierarchy result could not separate the architecture from the data. The V10 concept path pools fixed 8-token groups (release at 8g+7) and its training head forecasts the next 8 tokens. The audit asks, per family and split: 1. Where does the answer start relative to the group grid? (answer index mod 8; uniform means no alignment; a single residue means a fixed alig

Imports:

```python
import json
from collections import Counter
from pathlib import Path
import numpy as np
```

Symbols:

- [chi2_uniform](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/audit_data.py:35>) — lines 35–37
- [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/audit_data.py:40>) — lines 40–72
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/audit_data.py:75>) — lines 75–83

## v10_3_tiny/compare.py

112 lines; SHA-256 `d8c66db112d6216b087e64b8b6ad21573a31b38af7ea9d0910a44bb5f92d7019`.

Primary copy: [v10_3_tiny/compare.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py>).

Paired comparisons between evaluated tiny arms, and the predeclared verdicts. python -B -m v10_3_tiny.compare --runs v10_3_tiny/runs/C0_s1 v10_3_tiny/runs/A_s1 ... --output v10_3_tiny/comparison.json Every arm was scored on identical documents (evaluate.panels), so per-document NLL differences are paired. Intervals are 95% document bootstraps of the token-weighted difference (resampling documents, 4,000 draws). G1/G2/G3 follow ARCHITECTURE.md exactly; nothing here is tuned after reading results.

Imports:

```python
import argparse, json
from pathlib import Path
import numpy as np
```

Symbols:

- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py:17>) — lines 17–26: Token-weighted NLL difference b - a over shared documents, with a bootstrap interval.
- [intervention_ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py:29>) — lines 29–32
- [paired_correct](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py:35>) — lines 35–42: Accuracy difference b - a on identical problems, with a 95% problem bootstrap.
- [math_accuracy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py:45>) — lines 45–47
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/compare.py:50>) — lines 50–107

CLI arguments:

```python
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--output', required=True)
```

## v10_3_tiny/grid_shift8.py

26 lines; SHA-256 `139480a443e4757d4365071c60c6062ab50090e2fd0b08707145dfd85caef450`.

Primary copy: [v10_3_tiny/grid_shift8.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/grid_shift8.py>).

Shift-8 control for the Sudoku grid-shift test (see ARCHITECTURE.md data audit). A shift of 4 moves every cell half a group; a shift of 8 moves it exactly one group, keeping the cell-to-group alignment while applying the same kind of out-of-distribution header edit. If an arm's extra shift-4 degradation is alignment, its shift-8 degradation should fall back toward C0's. python -B -m v10_3_tiny.grid_shift8 --runs C0_s1 A_s1 X8_s1 X4_s1

Imports:

```python
import argparse, json
from pathlib import Path
import torch
from . import evaluate as E
from .sleep import load
```

CLI arguments:

```python
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--device', default='mps')
```

## v10_3_tiny/paired_training.py

44 lines; SHA-256 `0f17530a196505dccf156260199a4ba3afc0f261cae93123c45fa181d8a185d8`.

Primary copy: [v10_3_tiny/paired_training.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/paired_training.py>).

Paired training-stream comparison: every seed-1 arm saw the identical batch at each step (step-based schedule, same seed), so per-step token-loss differences are paired. Reports mean (arm - reference) token loss by family in step bins. python -B -m v10_3_tiny.paired_training --runs C0_s1 A_s1 X8_s1 X4_s1 --reference C0_s1

Imports:

```python
import argparse, json
from pathlib import Path
import numpy as np
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/paired_training.py:15>) — lines 15–16
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/paired_training.py:19>) — lines 19–38

CLI arguments:

```python
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--output', default=str(RUNS.parent / 'paired_training.json'))
```

## v10_3_tiny/prepare_data.py

98 lines; SHA-256 `ce320f994291e2ef2e877c5e58e0272455f95bd57b7aea2e712edb87d2e4f6c0`.

Primary copy: [v10_3_tiny/prepare_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/prepare_data.py>).

Build the tiny corpus by re-tokenizing the clean V10.1 corpus. Source: v10_1/data (read only). Its builder verified that the 8,192-token tokenizer round-trips every record, so decoding the stored arrays recovers the exact prompt and target text. A 1,024-token BPE with the same pre-tokenizer (digits isolated, byte level) is fitted on the decoded train split only. The math module HELD_OUT is removed from the pretraining 'math' family and written as a separate 'newmath' family: it is the new phase for the sleep and continual-learning tests. Identities, train/dev assignments and answer boundaries are preserved. No sealed-test data exist in the so

Imports:

```python
from array import array
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
from v10_10M.prepare_data import SPECIAL
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/prepare_data.py:29>) — lines 29–33
- [decoded_records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/prepare_data.py:36>) — lines 36–51
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/prepare_data.py:54>) — lines 54–94

## v10_3_tiny/report.py

221 lines; SHA-256 `579698817f8033c6737760b026acc55d5968c5e6687f4485d8a0b35c8bba7c9f`.

Primary copy: [v10_3_tiny/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py>).

Render every numeric table of the tiny V10.3 report from the saved JSON. python -B -m v10_3_tiny.report --runs C0_s1 A_s1 X8_s1 X4_s1 \[C0_s2 X8_s2\] --sleep sleep_X8_s1 --out v10_3_tiny/REPORT_TABLES.md No number in the report is typed by hand; prose is written around this file.

Imports:

```python
import argparse, json, statistics as st
from pathlib import Path
```

Symbols:

- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:15>) — lines 15–16
- [ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:19>) — lines 19–20
- [table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:23>) — lines 23–25
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:28>) — lines 28–33
- [training](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:36>) — lines 36–86
- [results](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:89>) — lines 89–103
- [mechanisms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:106>) — lines 106–135
- [verdicts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:138>) — lines 138–151
- [sleep](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:154>) — lines 154–193
- [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:196>) — lines 196–204
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_3_tiny/report.py:207>) — lines 207–215

CLI arguments:

```python
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--comparison', required=True)
p.add_argument('--sleep')
p.add_argument('--out', required=True)
p.add_argument('--root', default=str(RUNS))
```

## v11_design_evidence/2026-10-12/alignment_table.py

63 lines; SHA-256 `5cdda3d0e6da1c0b1801439d5dcfe9d98d952101b36a22b8122609ad7d768fe2`.

Primary copy: [v11_design_evidence/2026-10-12/alignment_table.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/alignment_table.py>).

Decoder alignment and release table for V11 rows 0-17, derived from the inherited code. python -B v11_design_evidence/2026-10-12/alignment_table.py (from the repo root) Uses v10_10M.model.Decoder.align (aligned forecast route) and v10_2.model.history_geometry (concept-history reader mask) unchanged. Each forecast slot carries a unique tag 100*(g+1)+s so its landing rows are visible. Reads code only; writes alignment_table.json next to this script.

Imports:

```python
import json, sys
from pathlib import Path
import torch
from v10_10M.model import Decoder
from v10_2.model import history_geometry
```

Symbols:

- [table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/alignment_table.py:21>) — lines 21–39
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/alignment_table.py:42>) — lines 42–59

## v11_design_evidence/2026-10-12/allocation_arms.py

64 lines; SHA-256 `3df48fc3f619c7b9c94347c132d8ffb659146d90e632050cb2b3acceb8f074fb`.

Primary copy: [v11_design_evidence/2026-10-12/allocation_arms.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/allocation_arms.py>).

Allocation arms and the 120/60/20 stress case, by instantiation plus analytic MACs. python -B v11_design_evidence/2026-10-12/allocation_arms.py (from the repo root) Uses summarize() and macs_per_token() from param_budget.py. Writes allocation_arms.json.

Imports:

```python
import contextlib, io, json, sys
from dataclasses import replace
from pathlib import Path
from param_budget import summarize, macs_per_token
from v10_3_tiny.config import TinyConfig
```

Symbols:

- [stress_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/allocation_arms.py:29>) — lines 29–40: Params and executed token-rate block-equivalents for 120/60/20 with E2 = b blocks (each run twice). Concept blocks run 2 visits on T/4 slot rows -> 2 * 60 / 4 = 30 token-block equivalents.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/allocation_arms.py:43>) — lines 43–60

## v11_design_evidence/2026-10-12/param_budget.py

158 lines; SHA-256 `d1a30f2fd697c20dec7a35870e4b2057cdc9adb9d553edd3f24388b3112affaf`.

Primary copy: [v11_design_evidence/2026-10-12/param_budget.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py>).

Verifiable parameter arithmetic for the V11 configurations. python -B v11_design_evidence/2026-10-12/param_budget.py (from the repo root) Inherited modules are counted by instantiating the existing classes (V10 / V10.2 / V10.3-tiny) with a config object; nothing is trained. Modules that are new in V11 (bridge, visit-1 head is shared so free, mixture log-variance, LoRA) are built here as plain nn.Modules with the shapes given in v11.md, purely to count them. Writes param_budget.json next to this script.

Imports:

```python
import json, sys
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from v10_10M.config import ModelConfig
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import V103Tiny, C0
from v10_10M.model import RMSNorm, Block
```

Symbols:

- [Bridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:26>) — lines 26–33: Training-only concept-anchored latent bridge (v11.md §3.3, "Bridge"): Init, glu_cross, residual MLP.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:28>).
- [n](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:36>) — lines 36–37
- [lora_count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:40>) — lines 40–44
- [summarize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:47>) — lines 47–81
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:84>) — lines 84–95
- [macs_per_token](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:108>) — lines 108–141
- [gate_params](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/param_budget.py:144>) — lines 144–149

## v11_design_evidence/2026-10-12/scaling_path.py

88 lines; SHA-256 `fe2c399b06c0f943d08297211960af146954004d8a0c8d8de83f148beec0b255`.

Primary copy: [v11_design_evidence/2026-10-12/scaling_path.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/scaling_path.py>).

Assumption-based scaling arithmetic for V11 (no hardware measured; nothing trained). python -B v11_design_evidence/2026-10-12/scaling_path.py (from the repo root) Formulas (per block, width d): dense block params 12 d^2 (attention 4 d^2, 4x FFN 8 d^2) MoE block params 4 d^2 + n_e * 3 d h + d n_e (SwiGLU experts of hidden h, router) MoE active params 4 d^2 + k * 3 d h + d n_e (top-k) executed token-rate block-equivalents per token X = (E - b) + 2 b + D + 2 C / 4 = E + b + D + C/2 (E unique encoder blocks of which b are recurrent E2, run twice; C concept blocks, 2 visits on T/4 slot rows; D decoder blocks) active execut

Imports:

```python
import json
from pathlib import Path
```

Symbols:

- [rung](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/scaling_path.py:28>) — lines 28–61
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_design_evidence/2026-10-12/scaling_path.py:64>) — lines 64–84

## v11_tiny/bench_10m.py

110 lines; SHA-256 `c524d9bf5fd053c1f654c25f3b1ce46b129bfe05cbca1074505981a677bd4a11`.

Primary copy: [v11_tiny/bench_10m.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/bench_10m.py>).

Solo Metal throughput benchmark of the 10M reference shapes (v11.md §6.2). Nothing is kept. python -B -m v11_tiny.bench_10m (writes v11_tiny/bench_10m.json; run with the GPU otherwise idle) Reference V11 (K1 objective, K = 1) and its C0 at width 128, vocab 8,192, E1/E2/E3 4/4/2, core 4, decoder 4, PQ 8x32, Engram 4 x ~156k x 32 on CPU; context 384, batch 8, S = 3; data v10_1/data (read only, train split). Per arm: 200 timed updates after 20 warm-up updates (unsynchronized, the production loop), then 40 updates with torch.mps.synchronize() between phases for a time split.

Imports:

```python
import json, math, statistics as st, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import C0
from v10_3_tiny.losses import compute as compute_v103
from .model import V11Tiny, counts
from .losses import compute as compute_v11
from .train import optimizers
```

Symbols:

- [sync_dev](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/bench_10m.py:33>) — lines 33–34
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/bench_10m.py:37>) — lines 37–39
- [bench](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/bench_10m.py:42>) — lines 42–98

## v11_tiny/compare.py

197 lines; SHA-256 `c0811a45c5ee9de847a64316609c2b87e1befe8fa072238d65403bf1ccdfaad8`.

Primary copy: [v11_tiny/compare.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py>).

Paired comparisons and the v11.md §7.2 Tier-1 gates, computed mechanically. python -B -m v11_tiny.compare (writes v11_tiny/comparison.json and v11_tiny/COMPARISON.md) Estimator: family NLL = sum(ce) / sum(scored tokens) over the fixed dev panel documents (the inherited definition). Uncertainty: paired document bootstrap (documents matched by (source, index), resampled with replacement, both arms recomputed on the same resample), 2,000 draws, per seed; 95% percentile interval. A gate passes only if its condition holds in EVERY seed with the interval on the stated side of the margin (v11.md §7.2). Seed variation is reported separately (per-seed

Imports:

```python
import json
from pathlib import Path
import numpy as np
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:24>) — lines 24–26
- [docs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:29>) — lines 29–32
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:35>) — lines 35–45: Delta = NLL(a) - NLL(b) with paired bootstrap CI. a, b: dict key -> row; must share keys and n.
- [within](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:48>) — lines 48–50
- [fmt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:53>) — lines 53–53
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/compare.py:56>) — lines 56–193

## v11_tiny/evaluate.py

327 lines; SHA-256 `1f0beb8fa439bcda96126fd0b943422e443ace1381de4b8de3722c97d07ab87c`.

Primary copy: [v11_tiny/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py>).

Evaluation of one trained arm of the v11.md §6.6 objective test (dev split only). python -B -m v11_tiny.evaluate --run-dir v11_tiny/runs/K1_s1 Panels are v10_3_tiny.evaluate.panels() (deterministic, identical across arms), so per-document rows can be paired. C0 and X8 use the inherited V10.3 evaluation functions unchanged (comparable to V10_3_tiny_REPORT.md); the V11 arms use the functions here; ORD gets base NLL, alignment, grid shift and generation. Every arm additionally gets: - alignment sensitivity: math/newmath re-scored with r = 0..7 dev-text tokens inserted after BOS - batched greedy generation, limit = answer tokens + 64, truncation 

Imports:

```python
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_1.evaluate import score_math
from v10_3_tiny import evaluate as E
from v10_3_tiny.config import TinyConfig
from .model import build, V11Tiny, Ordinary
```

Symbols:

- [to](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:30>) — lines 30–30
- [rows_of](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:33>) — lines 33–40
- [summarize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:43>) — lines 43–54
- [calibration_means](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:58>) — lines 58–63
- [nll_v11](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:72>) — lines 72–94
- [nll_plain](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:98>) — lines 98–106
- [latent_v11](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:110>) — lines 110–149: Next-concept forecast quality in the model's normalized target space (gate G3) and in raw pooled space (comparable to V10.3), plus visit-2 gain.
- [pooled_participation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:153>) — lines 153–164: Participation ratio (effective rank) of raw pooled slot states on the full scored panel, per family.
- [calibration_stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:168>) — lines 168–173
- [participation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:176>) — lines 176–178
- [filler_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:181>) — lines 181–184
- [alignment](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:188>) — lines 188–212: Re-score math/newmath with r = 0..7 filler tokens inserted after BOS; scored tokens unchanged.
- [generate_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:216>) — lines 216–231: Greedy, full recompute per step (no cache), right-padded batch with per-row lengths.
- [generation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:234>) — lines 234–262
- [scales](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:265>) — lines 265–276
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/evaluate.py:279>) — lines 279–320

CLI arguments:

```python
p.add_argument('--run-dir', required=True)
p.add_argument('--checkpoint', default='final.pt')
p.add_argument('--device', default='mps')
p.add_argument('--output', default='evaluation.json')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--quick', action='store_true')
```

## v11_tiny/grad_share.py

76 lines; SHA-256 `73753b882496a162c2a1befa91ca5987347854cea21efeb75fddbe3087e4671d`.

Primary copy: [v11_tiny/grad_share.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/grad_share.py>).

v11.md §4.4 control 3: where does each loss's gradient land? Final V11 checkpoints, frozen copies. python -B -m v11_tiny.grad_share (writes v11_tiny/grad_share.json) For 20 training-split batches (a fixed Corpus seed not used by any run: 777), per-loss gradient L2 norms on module groups {embedding+Engram+E1/E2/E3 ('encoder'), pooler, E->C CRC + core ('core'), PQ head, decoder, bridge}, with all losses at their final weights (ramp = 1). Also a frozen-model proxy for the constant-anchor control: the bridge KL with u1 replaced by its batch mean over valid groups.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import torch
from v10_10M.data import Corpus
from v10_3_tiny.config import TinyConfig
from .model import build
from .losses import compute, bridge_losses, LAMBDA
```

Symbols:

- [group_norms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/grad_share.py:24>) — lines 24–32
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/grad_share.py:35>) — lines 35–72

## v11_tiny/launch.py

34 lines; SHA-256 `8ac8ece56dfab171b9bf69a97274289d83995905de06e65b065639052ceb8d47`.

Primary copy: [v11_tiny/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/launch.py>).

Run the 12 objective-test trainings on Metal, 3 at a time, longest first (dynamic queue). nohup python3 -B -m v11_tiny.launch > v11_tiny/runs/launch.log 2>&1 &

Imports:

```python
import json, subprocess, sys, threading, time
from pathlib import Path
```

Symbols:

- [worker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/launch.py:14>) — lines 14–27

## v11_tiny/losses.py

125 lines; SHA-256 `a4442ee2cccf92681e059b7249202a52250d277d02cdef747f3b755762825f08`.

Primary copy: [v11_tiny/losses.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py>).

V11 objectives (v11.md §4) for the K = 1 arms, plus NTP for ORD. L = L_NTP + ramp * ( lambda_c (L_v1/3 + 2 L_v2/3) + lambda_vq L_vq + lambda_h L_h + lambda_KL L_KL ) + L_probe L_v1 Gaussian NLL per dimension of the visit-1 forecast vs tau_{g+1}, all eligible groups L_v2 same for visit 2 (K = 1 special case of the mixture) on S sampled boundaries per document, from stopped committed inputs (v11_tiny.model.V11Tiny.visit2_from_stopped) L_vq NCP Eq. 21 codebook fitting to the normalized target (codebook only) L_h SmoothL1 between RMS-normalised bridge state and sg\[pre-outn decoder state\] L_KL KL(sg teacher logits || student logits through a det

Imports:

```python
import math
import torch
from torch.nn import functional as F
from v10_1.model import token_loss, chunk_batch
from v10_2.model import future_ce
```

Symbols:

- [gaussian_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:25>) — lines 25–31: Per-dimension isotropic Gaussian NLL, mean over valid groups. pred/target \[..., 2, d\]; valid \[...\].
- [sample_boundaries](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:34>) — lines 34–42: Up to S eligible groups per document, uniformly without replacement; returns (rows, groups).
- [_Detached](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:45>) — lines 45–46
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:46>).
- [rmsn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:49>) — lines 49–49
- [bridge_losses](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:52>) — lines 52–76: Vectorised over groups; loop over the 8 within-group steps (v11.md §5.3).
- [probe_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:79>) — lines 79–86
- [compute](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/losses.py:89>) — lines 89–125

## v11_tiny/model.py

230 lines; SHA-256 `5004b4ffef71ba55e163a44f7ae3975138e9cad71d7c7ebf8f699606b6e34d88`.

Primary copy: [v11_tiny/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py>).

V11 tiny test bed: the arms of v11.md §6.6. Arms C0 v10_3_tiny.model.C0 (concept-free control, same backbone) -- inherited, unchanged X8 v10_3_tiny.model.V103Tiny with K = 8 (V10.3 recipe) -- inherited, unchanged ORD ordinary pre-norm transformer, 11 full-causal blocks, tied head, no Engram (parameter-matched) NOAUX V11 topology (code-free two-visit core), NTP only (+ detached probe) K1NB V11 topology + latent next-concept objective (L_v1, L_v2 at K = 1, L_vq), no bridge K1 K1NB + concept-anchored NextLat bridge (training-only) The V11 topology reuses every inherited module (V10 encoder/Engram/E2/E3, Pooler, EncoderConceptCRC, ConceptCore blo

Imports:

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.model import V10, Block, RMSNorm, Engram, geometry
from v10_1.model import EncoderConceptCRC
from v10_2.model import ForecastReadout, history_geometry
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import C0, Decoder103, build as build_v103
```

Symbols:

- [Decoder11](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:32>) — lines 32–60: Decoder103 unchanged, but forward also returns the pre-`outn` residual d_t (the bridge target).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:35>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:42>).
- [Bridge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:63>) — lines 63–75: beta_{g,0} = Init(flatten u1_g); beta_j = beta_{j-1} + MLP(RMSNorm((W_h beta) * sigmoid(W_e sg\[E x\]))) (v11.md §3.3).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:66>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:73>).
- [V11Tiny](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:78>) — lines 78–171
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:79>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:98>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:102>), [visit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:115>), [normalized_target](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:121>), [update_target_buffers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:125>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:137>), [visit2_from_stopped](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:160>).
- [Ordinary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:174>) — lines 174–193: Plain pre-norm transformer: tied embedding, `layers` full-causal Blocks with RoPE, final RMSNorm.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:177>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:185>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:187>).
- [OrdinaryEngram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:196>) — lines 196–214: Ordinary + the inherited Engram after block 1 (same position as in E1): the fair plain-model control, since every ECD arm also carries the 1.05M-row n-gram table outside its dense count.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:200>), [place](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:203>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:206>).
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:217>) — lines 217–221
- [counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/model.py:224>) — lines 224–230

## v11_tiny/offsets_extra.py

61 lines; SHA-256 `e231b5fb864f08c8c1c4657aeb8bb7e084a709cb0793e5d2f6061523a67719cc`.

Primary copy: [v11_tiny/offsets_extra.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/offsets_extra.py>).

Distraction vs alignment: re-score math/newmath with r in {0, 8, 16} filler tokens after BOS (r = 8 and 16 keep the group alignment of r = 0) and Sudoku with header shifts 4 and 8 (v10_3_tiny/grid_shift8.py logic), for every final checkpoint. A residue (alignment) effect is periodic in r; distraction grows with filler. python -B -m v11_tiny.offsets_extra (writes v11_tiny/offsets_extra.json; run with the GPU idle)

Imports:

```python
import json
from pathlib import Path
import torch
from v10_3_tiny import evaluate as E
from v10_3_tiny.config import TinyConfig
from .model import build
from . import evaluate as V
```

Symbols:

- [filler16](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/offsets_extra.py:18>) — lines 18–20
- [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/offsets_extra.py:24>) — lines 24–42
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/offsets_extra.py:45>) — lines 45–57

## v11_tiny/post.py

76 lines; SHA-256 `a29c08ac15feaf2a0374fb0e7560954e00158721ad8d96dc1fa8b1c237619432`.

Primary copy: [v11_tiny/post.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post.py>).

After-training pipeline (never more than 3 GPU processes together with the trainer launcher): 1. evaluate each finished run in a free slot; 2. when the launcher's 12 jobs are done, train K1SC_s1 (diagnostic: K1 with separate clip groups) and C0_s3 (third C0 seed), then evaluate them; 3. idle 5 min, solo throughput smokes for all six arms (thermal state recorded before/after); 4. idle 5 min, the 10M benchmark (solo); 5. grad_share, training_summary, compare. nohup python3 -B -m v11_tiny.post > v11_tiny/runs/post.log 2>&1 &

Imports:

```python
import json, subprocess, sys, time
from pathlib import Path
```

Symbols:

- [therm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post.py:19>) — lines 19–21
- [active_trainers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post.py:24>) — lines 24–29
- [py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post.py:32>) — lines 32–33
- [train_cmd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post.py:36>) — lines 36–40

## v11_tiny/post2.py

65 lines; SHA-256 `ec4d6fe0f5c51ed88b429156b9c740076a9ff88175346e8f2927da94728ee212`.

Primary copy: [v11_tiny/post2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post2.py>).

Replacement after-training pipeline. Slot use is counted from live processes (pgrep), so a restart can never oversubscribe the GPU (max 3). Queue: trainers NOAUXSC_s1, K1NBSC_s1 (NOAUX / K1NB with --separate-clip; seed 1); evaluations for every run with final.pt and no evaluation.json; then idle 3 min -> solo throughput set -> idle 3 min -> 10M bench -> offsets_extra -> grad_share -> training_summary -> compare. nohup python3 -B -m v11_tiny.post2 > v11_tiny/runs/post2.log 2>&1 &

Imports:

```python
import json, subprocess, sys, time
from pathlib import Path
```

Symbols:

- [live](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post2.py:17>) — lines 17–19
- [therm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post2.py:22>) — lines 22–23
- [py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post2.py:26>) — lines 26–27
- [runs_with_final](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/post2.py:31>) — lines 31–32

## v11_tiny/tests.py

223 lines; SHA-256 `9fc5d0003a4402e5ea6ea27d7b5412c1b7ecd7cc43b61d8848661233ac5d5e97`.

Primary copy: [v11_tiny/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py>).

Correctness tests for v11_tiny (v11.md §7.6 subset). CPU fp32 unless stated. python -B -m v11_tiny.tests (writes v11_tiny/tests_result.json)

Imports:

```python
import copy, json, math, time
from pathlib import Path
import torch
from torch.nn import functional as F
from v10_10M.data import Corpus
from .model import build, counts, V11Tiny, ARMS
from .losses import compute, gaussian_nll, bridge_losses, probe_loss, sample_boundaries
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:18>) — lines 18–20
- [batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:23>) — lines 23–28
- [fresh](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:31>) — lines 31–35
- [grads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:38>) — lines 38–46
- [parts_for](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:49>) — lines 49–51
- [test_counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:54>) — lines 54–60
- [test_gradient_contracts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:63>) — lines 63–101
- [test_noaux_k1nb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:104>) — lines 104–111
- [test_gaussian](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:114>) — lines 114–125
- [test_boundaries](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:128>) — lines 128–135
- [test_forward_contracts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:139>) — lines 139–165
- [test_causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:169>) — lines 169–186
- [test_bridge_edge](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:189>) — lines 189–194
- [test_mps_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/tests.py:197>) — lines 197–209

## v11_tiny/train.py

237 lines; SHA-256 `2f89fa32a3482742d7c4f78419001493e83ea3327b3ed4e221b72403767821c1`.

Primary copy: [v11_tiny/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py>).

Step-based pretraining for one arm of the v11.md §6.6 objective test. python -B -m v11_tiny.train --arm K1 --seed 1 --run-dir v11_tiny/runs/K1_s1 Every arm uses the inherited V10.3-tiny schedule and data stream: batch 16, context 256, AdamW (0.9, 0.95) lr 1e-3, warmup 200 updates then cosine to 10% over --max-steps, dense clip 1.0, CPU SparseAdam Engram at 5x LR with its own clip, one Corpus('train', seed) batch per update. C0 and X8 run the inherited objective code (v10_3_tiny.losses.compute, candidate updates every 4th step, the same generator formula); `tests_trainer.py` checks that this trainer reproduces v10_3_tiny.train step for step. T

Imports:

```python
import argparse, hashlib, json, math, queue, random, threading, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids, atomic_json
from v10_3_tiny.train import quick_eval
from v10_3_tiny.losses import compute as compute_v103
from .model import build, counts, V11Tiny, ARMS
from .losses import compute as compute_v11, LAMBDA
```

Symbols:

- [optimizers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:42>) — lines 42–50
- [source_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:53>) — lines 53–58
- [Prefetch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:61>) — lines 61–76: Producer thread: the Corpus is only ever called from this thread, in step order.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:64>), [get](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:76>).
- [stream_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:79>) — lines 79–80
- [scalar](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:83>) — lines 83–86
- [buffer_stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:90>) — lines 90–98
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:101>) — lines 101–220
- [parser](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/train.py:223>) — lines 223–233

CLI arguments:

```python
p.add_argument('--arm', choices=ARMS, required=True)
p.add_argument('--run-dir', required=True)
p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--context', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--candidate-every', type=int, default=4)
p.add_argument('--threads', type=int, default=2)
p.add_argument('--eval-every', type=int, default=1000)
p.add_argument('--max-steps', type=int, default=6000)
p.add_argument('--S', type=int, default=2)
p.add_argument('--foreach', action='store_true')
p.add_argument('--separate-clip', action='store_true', help='diagnostic: clip trunk, bridge and probe separately at 1.0')
```

## v11_tiny/training_summary.py

73 lines; SHA-256 `cf39482d49222265c0602daa2bade470150dcac4d437d841ee316ba68a7c0f42`.

Primary copy: [v11_tiny/training_summary.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/training_summary.py>).

Summaries from train.jsonl for every run (no model is loaded). python -B -m v11_tiny.training_summary (writes v11_tiny/training_summary.json) - Stream identity: per seed, the (family, document) hash sequence must be identical across all six arms. - Paired training-stream gaps: every arm saw the identical batch at each step, and the stream is one pass over fresh documents, so the per-step token-loss difference on the same batch is a low-variance, effectively held-out comparison (as in v10_3_tiny/paired_training.py). Reported over the last 1,000 updates per family, with a block-bootstrap CI over 50-step blocks. - Thermal: median step time in th

Imports:

```python
import json
from pathlib import Path
import numpy as np
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/training_summary.py:22>) — lines 22–24
- [block_ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/training_summary.py:27>) — lines 27–32
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v11_tiny/training_summary.py:35>) — lines 35–69

## v12_tiny/data.py

144 lines; SHA-256 `dfb8689446ce628ce7f68658fc84682417a11adf59b763850f91f9bcfcbcab71`.

Primary copy: [v12_tiny/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py>).

Episode batches from the packed pinpad splits (right-padded, per-target weights, ground truth).

Imports:

```python
from pathlib import Path
import numpy as np
import torch
import pinpad as P
```

Symbols:

- [Split](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:10>) — lines 10–42
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:11>), [episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:20>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:24>).
- [Sampler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:45>) — lines 45–66: Fresh-without-replacement episode order per epoch, deterministic in the seed. Length bucketing: a pool of `pool` batches is drawn, sorted by length, cut into batches, and the batches are shuffled (cuts padding).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:48>), [_refill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:54>), [next_idx](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:64>).
- [BPEFamily](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:84>) — lines 84–111
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:85>), [window](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:91>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:98>).
- [Mixed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:114>) — lines 114–138: Family-per-batch stream, deterministic in the seed (identical across arms with the same seed).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:116>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:125>).
- [_N](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:141>) — lines 141–144: Minimal stand-in so Sampler can permute a BPE family (only .n and .off are used).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/data.py:143>).

## v12_tiny/devnll.py

24 lines; SHA-256 `308e8c5ef4cb77900ed60202f5bd47e798614922a7d2d66cd5c868b9cc32165f`.

Primary copy: [v12_tiny/devnll.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/devnll.py>).

Per-family dev NLL for any run/checkpoint (forward-only): pinpad act/pos/end, text, math, newmath. python -B devnll.py RUN CKPT \[CKPT ...\] -> runs/RUN/devnll_CKPT.json

Imports:

```python
import json, sys
from pathlib import Path
import torch
from data import Split
from evaluate import load_run, dev_nll, bpe_dev_nll
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/devnll.py:15>) — lines 15–20

## v12_tiny/drift.py

36 lines; SHA-256 `f05e6b7b3673b31640c6e6841b94a944b06c09efd0efc19fc4be5c087126fbbb`.

Primary copy: [v12_tiny/drift.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/drift.py>).

Codebook drift (advisor #13): the NCP head learns against a moving EMA k-means codebook. For the same dev targets (summaries from the later checkpoint's model), the fraction whose code changes between the earlier and the later checkpoint's codebook (normaliser + codewords), per family and per codebook segment. python -B drift.py RUN CKPT_A CKPT_B -> runs/RUN/drift_A_B.json

Imports:

```python
import json, sys
from pathlib import Path
import torch
from evaluate import load_run
from losses import group_targets
from info_test import family_batches
from losses import group_valid
import train
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/drift.py:20>) — lines 20–32

## v12_tiny/evaluate.py

128 lines; SHA-256 `4654ab6644a50d0fae3c9a35960817c8cf0f4adbce1422a571aa0a0a47aa92e4`.

Primary copy: [v12_tiny/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py>).

Phase-1 evaluation of a trained base (dev splits only). python -B evaluate.py RUN -> runs/RUN/eval.json - dev NLL: action NLL (nats/action), position NLL, END NLL. - probes (probes.py): every token-rate depth; for ECD also the concept stream (token view of u and of the state at the concept control point). - competence: closed-loop rollouts judged 'any' (completes some pretraining task) on fresh layouts. - post task raw success (no controller): the base's own chance of solving the held-out composition. - NCP: dev code accuracy and CE, and for target='dec' the shortcut monitor (accuracy with the decoder's concept read zeroed when computing the 

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pinpad as P
from data import Split
from model import token_view
from probes import probe_all
from rollout import rollout
```

Symbols:

- [load_run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py:27>) — lines 27–35
- [dev_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py:39>) — lines 39–50
- [ncp_dev](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py:54>) — lines 54–80
- [bpe_dev_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py:84>) — lines 84–96: Token NLL on the fixed dev panel of a BPE family: text = first 256 tokens of each of 400 dev docs (all tokens scored); math/newmath = every dev problem (answer tokens + EOS).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/evaluate.py:99>) — lines 99–124

## v12_tiny/grad_share.py

51 lines; SHA-256 `72d81224aa2728de396d3b93f71fecf102ea79b57546986c00218604a6d21caa`.

Primary copy: [v12_tiny/grad_share.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/grad_share.py>).

Pre-registered gradient-share check (Decision 3.5): per family, ||d(lam*L_aux)/d theta|| / ||dL_NTP/d theta|| on the encoder (emb excluded: shared with the head) and on the concept module. Acceptance: aux share on E <= 1. python -B grad_share.py RUN CKPT -> runs/RUN/grad_share_CKPT.json

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Mixed
from evaluate import load_run
from losses import ntp, ncp_ce, nextlat_loss
import train
```

Symbols:

- [norm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/grad_share.py:18>) — lines 18–20
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/grad_share.py:23>) — lines 23–47

## v12_tiny/gtctrl.py

122 lines; SHA-256 `6b5413e6a2667e91472b3ca7c5602902e7b4f2c111466d88c0375f28f8f1f691`.

Primary copy: [v12_tiny/gtctrl.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py>).

Ground-truth subgoal controllers on a frozen base (paper §"internal controllers", C.3, Fig. 4). One rank-16 linear residual controller per abstract action (4 pairs), e <- e + A_g B_g e, inserted at the base's control point (TF: residual before block L//2; ECD: the concept stream or the decoder input). Trained with the base frozen on the clean `meta` split, labels = the ground-truth pair of the *next* token (row t gets abs_act\[t+1\]; a group-rate controller reads the label of its release row k g + k - 1), action NLL only. Evaluated closed-loop on the post-training task (0-1-2-3-4-5-6-7-0-1-2-3: new order, longer) by activating the controllers

Imports:

```python
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from rollout import rollout
from evaluate import load_run
```

Symbols:

- [LowRankBank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:27>) — lines 27–36
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:28>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:33>).
- [group_labels](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:39>) — lines 39–41
- [make_ctrl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:44>) — lines 44–49: sub: (B, T) per-row labels. Returns the callable passed as model(ids, ctrl=...).
- [train_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:52>) — lines 52–69
- [base_action_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:73>) — lines 73–81
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/gtctrl.py:84>) — lines 84–117

## v12_tiny/info_test.py

107 lines; SHA-256 `3d671195474997d1e6b68291e3acad40d67b48be24535fcaf2ffc2841374ca09`.

Primary copy: [v12_tiny/info_test.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/info_test.py>).

Information test (Decision 3.4): does the concept state u_g carry information about the next segment's codes beyond what C's own input p_g already carries? (V10.3's forecast was a worse predictor than its own input.) Per family (pinpad, math, text) and per codebook segment s, probes predict idx_s(tau_{g+1}) from (a) u_g, (b) p_g (the pooled input to C); fit on train-split rows, scored on dev rows (as the NCP head). Two probe classes: linear softmax, and an MLP with the NCP head's shape fit from scratch (the fair comparison with the head). Baselines: majority code, and copy (idx_s(tau_g) as the prediction). Also reported: the trained NCP head'

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Split, BPEFamily
from evaluate import load_run
from losses import group_targets, group_valid
from probes import fit_probe
import train
```

Symbols:

- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/info_test.py:25>) — lines 25–41: Features from `model`; targets (summaries + codebook) from `src` on the same rows (src = model normally).
- [family_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/info_test.py:44>) — lines 44–55: Probes are fit on train-split rows and scored on dev rows (the NCP head was also trained on train data).
- [fit_mlp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/info_test.py:58>) — lines 58–68: Probe with the NCP head's architecture (RMSNorm -> 2d GELU -> classes), fit from scratch.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/info_test.py:71>) — lines 71–101

## v12_tiny/internal_rl.py

217 lines; SHA-256 `1e569a726df555e1f2b50fc4fa752f132acb1e27b5175333c571d8fc9d071280`.

Primary copy: [v12_tiny/internal_rl.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py>).

Internal RL (ETA paper §"Internal reinforcement learning", C.5, Algorithms 1–3) and the raw-action RL baseline. Internal RL: the frozen base, the frozen metacontroller switch unit and controller decoder are the environment. A causal policy pi(z | e_{1:t}) (1-layer GRU, Gaussian with learned log-std) replaces the acausal encoder. At each decision unit the frozen switch emits beta (thresholded at 0.5; the first unit always switches); only at switches is a new z sampled, otherwise z is held, so the policy acts on the abstract timescale. Sparse reward: 1 if the whole task is completed. Advantage = (R - mean) / (std + 1e-3) over the batch (critic-

Imports:

```python
import json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from evaluate import load_run
from meta import MetaController
```

Symbols:

- [Policy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:28>) — lines 28–33
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:29>).
- [Env](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:36>) — lines 36–56: Batched pinpad episodes driven by a (controlled) frozen model; full recompute each step.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:38>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:49>).
- [ZController](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:59>) — lines 59–117: ctrl callable for internal RL: decision units = token decision rows (header + 2j) or concept groups. Holds z per unit (B, J, nz); at the newest unit it queries the switch + policy (if `act`).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:62>), [units](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:67>), [new_unit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:74>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:93>).
- [run_internal_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:121>) — lines 121–133
- [ppo_update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:136>) — lines 136–152: recs: list over units of (e_j (B,d), switch (B,), z (B,nz)). Recompute log pi under the current policy.
- [internal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:155>) — lines 155–177
- [raw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/internal_rl.py:180>) — lines 180–211

## v12_tiny/losses.py

171 lines; SHA-256 `0ae8ef6e12825fde6fbc165e01806591d0046e6784d63597c409d3c362935a05`.

Primary copy: [v12_tiny/losses.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py>).

V12 objectives. NTP weighted next-token CE (actions 1, positions lam_obs, END 1; header unscored). NCP-CE next-concept prediction as cross-entropy over *discrete segment codes* (FIRST_PRINCIPLES P2). The target tau_{g+1} is a stopped, EMA-normalised group summary, quantised by a product codebook (S segments x N words, EMA k-means with dead-code reset; no gradient). A head on u_g predicts, per segment, logits over the N words; the loss is sum_s CE(logits_s, idx_s(tau_{g+1})). The joint over segments is the product of per-segment categoricals (factorised), which is mode-preserving per segment, unlike MSE regression to the conditional mean. targ

Imports:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm
```

Symbols:

- [ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:25>) — lines 25–29
- [PQCodebook](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:32>) — lines 32–85: Product quantiser over an EMA-normalised target; updated by EMA k-means, never by gradient.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:34>), [normalise](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:44>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:48>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:54>), [usage_perplexity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:83>).
- [NCPHead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:88>) — lines 88–95
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:89>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:94>).
- [group_targets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:98>) — lines 98–104: Raw (unnormalised) per-group target summaries, shape (B, G, d).
- [group_valid](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:107>) — lines 107–112: Group g+1 is a valid target if it lies entirely inside the trajectory of the episode (after p_0).
- [ncp_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:115>) — lines 115–131
- [NextLat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:134>) — lines 134–145
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:135>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:143>).
- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:148>) — lines 148–149
- [nextlat_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/losses.py:152>) — lines 152–171: d = 1. Rows t whose next token x_{t+1} and target state h_{t+1} lie in the scored trajectory.

## v12_tiny/make_data.py

35 lines; SHA-256 `504a4d9a34e2a21e82b6312cf86d67cf2d2968a673049068e7723d59dffed0f5`.

Primary copy: [v12_tiny/make_data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/make_data.py>).

Generate the pinpad splits once (seeds disjoint per split; the test split is sealed and never read here). python -B make_data.py -> data/{train,meta,dev,dev_meta,post_dev,test}.npz train : pretraining tasks (Table A1), expert eps = 0.05 (paper Fig. A2: mild suboptimality helps the base) meta : pretraining tasks, eps = 0 (paper C.4.4: clean demos for metacontroller discovery) dev : pretraining tasks, eps = 0.05, for NLL / probes dev_meta : pretraining tasks, eps = 0, for metacontroller evaluation (switch F1, rate–distortion) post_dev : the post-training task, eps = 0 (diagnostic only; never trained on) test : sealed (pretraining tasks + post t

Imports:

```python
import sys, time
from multiprocessing import Pool
from pathlib import Path
import pinpad as P
```

Symbols:

- [job](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/make_data.py:22>) — lines 22–24

## v12_tiny/meta.py

226 lines; SHA-256 `454b28fbd98d2379d2cc94e2410cf59ec7ff790f517d38b8769241da20498b9f`.

Primary copy: [v12_tiny/meta.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py>).

Paper-faithful metacontroller on a frozen base (ETA paper §"Unsupervised metacontroller", D.2, E.1, C.4). Timestep = one environment step. For token-rate control the decision row of step j is the position row p_{j-1} (index header + 2j, which predicts action a_j); its controller U_j is applied to rows header+2j and header+2j+1. For group-rate control (ECD 'concept' point) the timestep is a concept group. e_j = residual at the control point (pre-control, frozen base) s = f_emb(e_{1:J}) non-causal sequence embedding (bi-GRU, mean over valid steps), n_s = 32 h_j = GRU(e_j, h_{j-1}) n_h = 32 (history, causal) mu, logvar = f_enc(e_j, h_{j-1}, s) h

Imports:

```python
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from evaluate import load_run
```

Symbols:

- [MetaController](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:32>) — lines 32–78
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:33>), [history](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:45>), [switch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:50>), [sequence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:54>), [controllers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:75>).
- [decision_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:81>) — lines 81–89: Token-rate decision rows header + 2j for j < n_steps (valid where the row predicts an action).
- [Controlled](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:92>) — lines 92–128: Builds the ctrl callable for one batch: reads e at decision rows, runs the metacontroller, writes U.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:94>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:98>).
- [kl_std](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:131>) — lines 131–132
- [step_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:135>) — lines 135–142
- [switch_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:145>) — lines 145–174: Switch precision/recall/F1 against ground-truth changes of `key` ('abs_act' = pair, 'tgt_col' = colour). Units are mapped to environment-step times (token rate: unit j = step j; concept rate: a group's release row r -> step (r - header) //
- [evaluate_mc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:178>) — lines 178–187
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta.py:190>) — lines 190–221

## v12_tiny/meta_xm.py

124 lines; SHA-256 `333936bf830bdb8f56c1f79fa6f07f0150886831c2ecc01cc1178d72f021490e`.

Primary copy: [v12_tiny/meta_xm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py>).

Ablation (DESIGN_STEPS Step 6.2): XM codes on the VAE's switches. A trained paper-faithful metacontroller's switch unit is frozen and its posterior-mean pass fixes the segments (β > 0.5, first step always a switch). K = 8 learned controller codes c_1..c_K (in z-space) replace the Gaussian posterior. Training is forward explorative modelling in the memory-saving mode (XM App. C): 1. K no-grad forward passes, pass k assigning code k to every segment; per-segment action NLL ℓ_{s,k} (an approximation: other segments' codes in pass k are also k); 2. per segment, the soft assignment r_{s,k} = softmax_k(−ℓ_{s,k} / τ) (τ → hard best-of-K as τ → 0); 3

Imports:

```python
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Split, Sampler
from evaluate import load_run
from meta import MetaController, decision_rows
```

Symbols:

- [XMCodes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:36>) — lines 36–47
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:37>), [segments](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:43>).
- [controlled_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:50>) — lines 50–63: Apply U(z_units) at token-rate decision rows (and the following row), return logits.
- [unit_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:66>) — lines 66–68
- [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:71>) — lines 71–91
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/meta_xm.py:94>) — lines 94–119

## v12_tiny/model.py

205 lines; SHA-256 `96fc0582ee91fdbaa69ccd74227deda1e4fde43677601036a9f5991b802965ac`.

Primary copy: [v12_tiny/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py>).

V12 tiny models. Every model exposes the same interface: out = model(ids, ctrl=None) -> dict(logits, resid=\[e_0 .. e_L\] (token-rate residual stream per depth), ...) `ctrl` is an optional callable applied to the token-rate residual at the model's control depth (`model.ctrl_layer`, between blocks): e <- ctrl(e). The metacontroller (meta.py) uses it on a frozen model.

Imports:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Symbols:

- [RMSNorm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:14>) — lines 14–19
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:15>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:18>).
- [rope](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:22>) — lines 22–30
- [Attention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:33>) — lines 33–54
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:34>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:38>).
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:57>) — lines 57–66
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:58>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:64>).
- [init_weights](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:69>) — lines 69–72
- [Transformer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:75>) — lines 75–97: Plain causal transformer (the paper's base model family): the size-matched control.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:79>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:88>).
- [n_params](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:100>) — lines 100–101
- [CrossBlock](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:104>) — lines 104–116: Decoder block: causal self-attention, cross-attention to released concept states, MLP.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:106>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:112>).
- [ECD](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:119>) — lines 119–194: Encoder -> fixed-group pooling (k tokens) -> group-rate causal concept module (optionally a loop of 2, shared weights) -> decoder reading the released concept history by cross-attention. Group g covers tokens \[k g, k g + k - 1\] and is rele
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:130>), [groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:145>), [pooled](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:148>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:152>), [release_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:169>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:173>).
- [token_view](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/model.py:197>) — lines 197–205: Map a group-rate state (B, G, D) to token rows: row t sees the latest released group g with k g + k - 1 <= t (zeros before the first release).

## v12_tiny/phase2.py

151 lines; SHA-256 `6bf6d7fcff443f1ed32892f84ee245ce9a73391923f7805dad01b6ff29c589b8`.

Primary copy: [v12_tiny/phase2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py>).

Phase-2 driver: for each base run, GT controllers, then the metacontroller α sweep, then evaluation. python -B phase2.py RUN \[RUN ...\] Order (revised 12:15, advisor #13): stage 1 for every run first (GT controllers, then evaluate), then the α jobs of all runs interleaved by priority 0.1, 0.3, 0, 1, 0.05, 0.17, 0.5, then one 6000-step sensitivity run per base (α 0.3). Parallelism: as given while any train.py runs, else 2. Cut (pre-registered): a base with < 4 α points started by 17:30 keeps only {0, 0.1, 0.3, 1}. If the 6000-step run moves pair F1 by > 0.1 vs its 2000-step twin, the whole sweep is labelled budget-limited (paper: 64 000 steps

Imports:

```python
import json, subprocess, sys, time
from pathlib import Path
```

Symbols:

- [is_ecd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:34>) — lines 34–35
- [stage1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:38>) — lines 38–39
- [tf_point](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:45>) — lines 45–57: TF point (pre-registered 11:58; noise floor added 14:05 before L5 was seen): whichever of mid (e_3) / L5 (e_5) has 3-pair GT success above base by >= 0.03, the larger margin if both; ties and neither -> mid.
- [stage2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:60>) — lines 60–79: α jobs from all runs interleaved by priority; ECD dec_in points (0.3 then 0.1) after the concept sweep's first two; one 6000-step sensitivity run per base last (TF: chosen point; ECD: concept), at α = 0.3.
- [parallel_now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:82>) — lines 82–85: 2 only when no training process is running (never three GPU processes; advisor #14).
- [output_of](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:88>) — lines 88–97: Result file a job writes (jobs whose result exists are skipped on restart).
- [running](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:103>) — lines 103–106: Command lines of Phase-2 jobs currently running (from any driver instance): restart-safe accounting.
- [run_jobs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:109>) — lines 109–127
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/phase2.py:130>) — lines 130–144

## v12_tiny/pilot.py

70 lines; SHA-256 `332b39578334d85c3e544a790033910b4b106cbe45bc32132eab6eecc45db09a`.

Primary copy: [v12_tiny/pilot.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pilot.py>).

Pilot (advisor #1): MPS throughput of the plain base at several widths, then a short training run with linear probes of the ground-truth subgoal by depth. Decides the world size and the phase budget. python -B pilot.py bench -> pilot_bench.json python -B pilot.py train D MIN -> runs/pilot_TF_dD/ (train.jsonl, final.pt, probes.json)

Imports:

```python
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from model import Transformer, n_params
from probes import probe_all
```

Symbols:

- [loss_fn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pilot.py:21>) — lines 21–25
- [bench](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pilot.py:28>) — lines 28–41
- [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pilot.py:44>) — lines 44–65

## v12_tiny/pinpad.py

179 lines; SHA-256 `b8af0e26a766f002d73242e93ed4a30fc8fd345df9991241ced00ba0b0486b19`.

Primary copy: [v12_tiny/pinpad.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py>).

Token-form gridworld-pinpad (ETA metacontroller paper, Appendix A.1), as an autoregressive token stream. World: G x G grid, O uniquely coloured cells, W walls, agent; 4 cardinal actions; bumping a wall or the border is a no-op. A colour is *visited* when the agent moves onto it from a different cell. An episode ends on success (all task colours visited in order), on failure (a colour visited out of order), or after T steps. Token stream of one episode (no task token: the subgoal is only inferable from behaviour): BOS C_0 c_0 .. C_{O-1} c_{O-1} WALL w_0 .. WALL w_{W-1} SEP p_0 a_1 p_1 a_2 p_2 ... a_n p_n END c_i / w_j / p_t are position tokens

Imports:

```python
import random
from collections import deque
import numpy as np
```

Symbols:

- [parse](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:32>) — lines 32–33
- [World](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:36>) — lines 36–84
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:37>), [pos_tok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:43>), [tok_cell](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:46>), [layout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:50>), [connected](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:58>), [step_cell](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:67>), [dist_map](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:73>).
- [header_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:87>) — lines 87–93: BOS (C_i pos_i)_i (WALL pos_j)_j SEP p_0 — each cell is tagged with its object identity, as in the paper's observation (one-hot object per cell), so binding colour to position is a content lookup.
- [expert_episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:96>) — lines 96–134: Roll out the (stochastic, shortest-path) expert. Returns tokens and per-token ground truth, or None.
- [env_reset](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:137>) — lines 137–141
- [env_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:144>) — lines 144–156: Apply action a (0..3). Returns the new position token; sets done/success.
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:159>) — lines 159–167
- [pack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/pinpad.py:170>) — lines 170–179: Store a list of episodes as one flat int16 array + offsets (+ ground truth).

## v12_tiny/probe_diag.py

63 lines; SHA-256 `8a7d276395d9f45bf8fff88593f9878a2b9a00053745900895c11372dffc8310`.

Primary copy: [v12_tiny/probe_diag.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probe_diag.py>).

Probe diagnostics: which latent does the base represent linearly? (pilot follow-up) Targets per trajectory row: abstract action (4), target colour (8), target cell (G^2 location of the target colour), last visited colour (9 incl. none), plus a label-copy sanity feature. python -B probe_diag.py RUN

Imports:

```python
import json, sys
import numpy as np
import torch
import pinpad as P
from data import Split
from evaluate import load_run
from probes import fit_probe
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probe_diag.py:15>) — lines 15–57

## v12_tiny/probes.py

57 lines; SHA-256 `00e83a04db12030ece62bcceb0bdbc5600c657dd4e7b69cb1bbc70714aa55ca4`.

Primary copy: [v12_tiny/probes.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probes.py>).

Linear probes of ground-truth latents from frozen residual streams (paper Fig. 3 / A1). For every depth, a multinomial logistic regression is fitted on episodes \[0, n/2) of the dev split and scored on \[n/2, n). Rows: trajectory tokens (after p_0) whose ground truth is defined. Targets: the abstract action (pair index, 4 classes) and the current target colour (8 classes). Chance = held-out majority-class rate.

Imports:

```python
import numpy as np
import torch
import torch.nn.functional as F
```

Symbols:

- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probes.py:13>) — lines 13–23
- [fit_probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probes.py:26>) — lines 26–35
- [probe_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probes.py:38>) — lines 38–50
- [collect_half_index](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/probes.py:53>) — lines 53–57

## v12_tiny/report.py

119 lines; SHA-256 `2b0402dd1e888749a0eedcef02b92ece6effe3ac9e69904af410713c6659d6c1`.

Primary copy: [v12_tiny/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/report.py>).

Render every saved result into RESULTS.md (tables only; prose lives in v12_Architecture.md). python -B report.py

Imports:

```python
import json
from pathlib import Path
```

Symbols:

- [j](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/report.py:12>) — lines 12–14
- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/report.py:17>) — lines 17–18
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/report.py:21>) — lines 21–115

## v12_tiny/rollout.py

85 lines; SHA-256 `a7399c3e57cf0c9c50d18140f6960abaf24d15abfb11887d530df7b0e0f0d414`.

Primary copy: [v12_tiny/rollout.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/rollout.py>).

Batched closed-loop rollouts of a (possibly controlled) frozen model in the pinpad world. The model only chooses actions; positions come from the environment. Each row records `sub\[row\]`, the abstract action (pair index) the environment currently requires *when that row was appended* — i.e. the subgoal of the next action — which ground-truth controllers use as their label (paper C.3), and which a group-rate controller reads at the group's release row. judge='task' : success iff the given task is completed (post-training evaluation; sparse reward). judge='any' : success iff the visited colour sequence completes *some* pretraining task, failu

Imports:

```python
import random
import numpy as np
import torch
import pinpad as P
```

Symbols:

- [_prefix_ok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/rollout.py:18>) — lines 18–19
- [rollout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/rollout.py:23>) — lines 23–73: Returns dict(success (n,), steps (n,), ids, sub, colours_visited).
- [_judge_any_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/rollout.py:76>) — lines 76–85

## v12_tiny/switch_chance.py

51 lines; SHA-256 `d51852b9c669b5041827d04f8b359f0b4a0403dd787c38336e4d63a5bdb08e57`.

Primary copy: [v12_tiny/switch_chance.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/switch_chance.py>).

Chance baseline for switch F1 (advisor #20). For a trained metacontroller, keep each episode's number of predicted switches but place them uniformly at random over that episode's valid units (20 draws); score pair and colour F1 with the same tolerance and clock as `meta.switch_metrics`. CPU only. python -B switch_chance.py RUN TAG -> adds 'chance' to runs/RUN/meta/TAG.json (original eval untouched)

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Split
from evaluate import load_run
from meta import MetaController, step_loss, switch_metrics
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/switch_chance.py:19>) — lines 19–47

## v12_tiny/switch_emit_check.py

49 lines; SHA-256 `f9beea5c17479bad483062e16f20d1f292b8f347a2a023a8f6809a3f2de87b50`.

Primary copy: [v12_tiny/switch_emit_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/switch_emit_check.py>).

Scaling check (§7 #5): with binarised β (inference / internal RL), z is piecewise constant, so the hypernetwork needs to run only at switch events and U can be cached per segment. Verifies exact equivalence to the per-row path on a real batch and reports the materialised-controller memory of both. python -B switch_emit_check.py RUN

Imports:

```python
import sys
import numpy as np
import torch
from data import Split
from evaluate import load_run
from meta import MetaController, decision_rows
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/switch_emit_check.py:16>) — lines 16–45

## v12_tiny/tests.py

124 lines; SHA-256 `75f423f78ef3a8cae156bdde71e00a5335b07e6268b603a79749219bb621ae5a`.

Primary copy: [v12_tiny/tests.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/tests.py>).

Correctness tests (CPU). python -B tests.py -> tests_result.json

Imports:

```python
import json, random
from pathlib import Path
import numpy as np
import torch
import pinpad as P
import train
from data import Split, Sampler
from model import ECD, Transformer, token_view
from losses import PQCodebook, ncp_ce, nextlat_loss, NCPHead, NextLat
from meta import MetaController, Controlled, decision_rows, switch_metrics, step_loss
from gtctrl import LowRankBank, make_ctrl, group_labels
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/tests.py:18>) — lines 18–19
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/tests.py:22>) — lines 22–120

## v12_tiny/train.py

110 lines; SHA-256 `0e1d8701be98a42cf827257005e5dee61d30f15faa159b3fd75db3f6c843b653`.

Primary copy: [v12_tiny/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/train.py>).

Phase 1: pretrain a base on the pinpad train split. python -B train.py ARM SEED STEPS -> runs/ARM_sSEED/{train.jsonl, final.pt, manifest.json} Every arm with the same seed sees the identical episode stream (same Sampler seed and call order; a per-step hash of the batch indices is logged). Step-based schedule: AdamW (0.9, 0.95), lr 1e-3, wd 0.03 on matrices, warmup 200, cosine to 10%; clip 1.0 on the trunk only (auxiliary heads clipped separately, V11 lesson).

Imports:

```python
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import pinpad as P
from data import Split, Sampler, Mixed, VOCAB
from model import Transformer, ECD, n_params
from losses import ntp, PQCodebook, NCPHead, NextLat, ncp_ce, nextlat_loss
```

Symbols:

- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/train.py:33>) — lines 33–44
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/train.py:47>) — lines 47–53
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/train.py:56>) — lines 56–59
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/train.py:62>) — lines 62–106

## v12_tiny/usage.py

80 lines; SHA-256 `dec2e8e875cb1435807de3224e0368e85ec1b20ab90ea2c9b7416220a45603f9`.

Primary copy: [v12_tiny/usage.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py>).

Declared concept-usage metric (Decision 2.1): dev NLL with and without the concept route, per family. C counts as used on a family if zeroing it costs >= 0.02 nats/token (pinpad: per action) — the declared metric. Also reported (advisors #13/#14): shuffle-ablation (u from another sequence in the batch: in-distribution removal) and mean-ablation, u replaced by its per-family dev mean. Zeroing is a distribution shift for D (u is RMS-normalised, never zero in training); mean-ablation removes the information while keeping D's input in-distribution, so it bounds usage from below. Forward-only; model.py untouched (the final concept norm `cn` is swa

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data import Split, BPEFamily
from evaluate import load_run, dev_nll, bpe_dev_nll
from losses import group_valid
```

Symbols:

- [MeanU](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:23>) — lines 23–25
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:24>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:25>).
- [ShuffleU](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:28>) — lines 28–33: In-distribution removal (advisor #14): u of another sequence in the batch (a fixed derangement).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:30>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:31>).
- [mean_u](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:37>) — lines 37–48
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/usage.py:52>) — lines 52–76

## v12_tiny/z_check.py

47 lines; SHA-256 `a94fde1bb80144287036916d4df5cf15030699d935a6c0140c66a740d8885ae5`.

Primary copy: [v12_tiny/z_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/z_check.py>).

Is the metacontroller's held z a task-level code? (advisor #21) For a trained point: (a) cosine between z at unit 0 (forced) and the z held after the observed switch; (b) linear probes from the held z (unit 3) to the episode's first pair and second pair (pair after the first true switch), trained on half of dev_meta, scored on the other half; chance = majority class. CPU only. python -B z_check.py RUN TAG

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from data import Split
from evaluate import load_run
from meta import MetaController, step_loss
from probes import fit_probe
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v12_tiny/z_check.py:19>) — lines 19–43

## v2_underperformance_review_2026-09-28/build_report.py

499 lines; SHA-256 `73679c8c50be894441189f61eb3d9932a11ba638196c0ad53b9fd53c41e6cbc1`.

Primary copy: [v2_underperformance_review_2026-09-28/build_report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/build_report.py>).

Build a reproducible written review from the frozen logs and CPU diagnostics.

Imports:

```python
from pathlib import Path
import json, math, statistics
```

Symbols:

- [render](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/build_report.py:14>) — lines 14–15
- [role](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/build_report.py:457>) — lines 457–472

## v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py

243 lines; SHA-256 `946b53adc314c5e2e2aedb3ed0682d328c06609b2417a85f7f353560c32460c6`.

Primary copy: [v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py>).

Read-only CPU checkpoint diagnostics. No MPS, optimizer, or writes in either run. Uses one CPU thread and memory-mapped checkpoint tensors assigned into a meta model. Diagnostics are interventions on fixed weights, not controlled retraining ablations.

Imports:

```python
from pathlib import Path
import argparse,datetime,gc,json,math,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model,nextlat_losses
from data import Canon,make_batch,target_mask
import concept_core_d6 as D6
```

Symbols:

- [save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:36>) — lines 36–36
- [note](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:37>) — lines 37–37
- [masked_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:41>) — lines 41–44
- [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:46>) — lines 46–49
- [trace](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:52>) — lines 52–87
- [inspect_dispatch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:125>) — lines 125–141
- [batch_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:151>) — lines 151–158
- [intervention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_diagnostics.py:171>) — lines 171–188

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--rows',type=int,default=8)
ap.add_argument('--skip-grad',action='store_true')
```

## v2_underperformance_review_2026-09-28/gradient_replication.py

88 lines; SHA-256 `1e0b9ba5f39ab29d73a648f261c65e4e1b61b0b0d5bfb836de85aeddb6ff7a6f`.

Primary copy: [v2_underperformance_review_2026-09-28/gradient_replication.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/gradient_replication.py>).

Replicate the auxiliary-gradient diagnostic across four independent B2 training batches, CPU only.

Imports:

```python
from pathlib import Path
import datetime,json,math,sys,gc
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model,nextlat_losses
from data import Canon,make_batch,target_mask
```

Symbols:

- [one](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/gradient_replication.py:25>) — lines 25–68

## v2_underperformance_review_2026-09-28/plot_findings.py

54 lines; SHA-256 `9ea2b4c1cf0ebb8004883cdeac3a76cf188d852fdb8e80cdba084f3313ba0b56`.

Primary copy: [v2_underperformance_review_2026-09-28/plot_findings.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/plot_findings.py>).

Render the frozen audit measurements; no model loads or training operations.

Imports:

```python
from pathlib import Path
import os, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

## v2_underperformance_review_2026-09-28/routing_replication.py

47 lines; SHA-256 `59346fa1c4abf0983d366a9c931ca80e34209c00053f66b28f36e868b9061c89`.

Primary copy: [v2_underperformance_review_2026-09-28/routing_replication.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/routing_replication.py>).

Separate real and padding chunk overflow on four CPU batches; fixed checkpoint, no training.

Imports:

```python
from pathlib import Path
import sys,json,math
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model
from data import Canon,make_batch
import concept_core_d6 as D6
```

Symbols:

- [wrap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/routing_replication.py:24>) — lines 24–36

## v2_underperformance_review_2026-09-28/snapshot.py

69 lines; SHA-256 `b83da0598d188eea06e2aff167cf598b00497e86142a59b782e731096205afb3`.

Primary copy: [v2_underperformance_review_2026-09-28/snapshot.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/snapshot.py>).

Read-only snapshots and comparisons. Writes only inside this review directory.

Imports:

```python
from pathlib import Path
import ast,datetime,hashlib,json,math,sys
from collections import Counter
import numpy as np
```

Symbols:

- [dump](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/snapshot.py:12>) — lines 12–12
- [jl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/snapshot.py:13>) — lines 13–13

## v2_underperformance_review_2026-09-28/source_snapshot/code/train.py

485 lines; SHA-256 `a0a589d5f9c0f0cf6ea5e7004b5a85c07728f56769ba3ea0abff71b4d1ae887d`.

Primary copy: [v2_underperformance_review_2026-09-28/source_snapshot/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py>).

Train HRM THINKER v2 (19M). One process, one device; resumable; pause-aware; 48 h in ONE schedule (v1 lesson D-041). python train.py --run_dir ../runs/main --hours 48 --milestone_hours 24 python train.py --run_dir ../runs/main --resume ../runs/main/ckpt_last.pt Per optimizer step: K ~ U{8..12} core passes (same K for every micro-batch of the step), windows of 4 (remainder merged into the last), carry detached between windows, per-window backward (c_hat and tok are leaves whose gradients accumulate; one backward into the encoder at the end of the micro-batch). Loss = NTP + 0.1 * mean_r NCP(u_r) + 0.1 * consistency(final window) + VQ + 0.01 LB 

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/code/train.py>)

Imports:

```python
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, forward, nextlat_losses, param_groups, param_table
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator
```

Symbols:

- [now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:27>) — lines 27–28
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:31>) — lines 31–34
- [windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:37>) — lines 37–44: Pass ranges (1-based, inclusive) of the detached windows: windows of w, remainder merged into the last.
- [restart_dead_now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:48>) — lines 48–60
- [rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:63>) — lines 63–67
- [set_rng_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:70>) — lines 70–73
- [save_ckpt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:76>) — lines 76–89
- [swap_used_mb](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:92>) — lines 92–97
- [micro_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:100>) — lines 100–166: One micro-batch of D6 training with per-window backward. Returns detached statistics (device tensors / floats).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py:169>) — lines 169–481

CLI arguments:

```python
ap.add_argument("--run_dir", required=True)
ap.add_argument("--data_dir", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")))
ap.add_argument("--device", default="mps")
ap.add_argument("--micro_bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--lr_scale", type=float, default=1.0)
ap.add_argument("--lr_min_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--hours", type=float, default=48.0)
ap.add_argument("--milestone_hours", type=float, default=24.0)
ap.add_argument("--est_total_steps", type=int, default=8000)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--K_min", type=int, default=8)
ap.add_argument("--K_max", type=int, default=12)
ap.add_argument("--eval_R", type=int, default=6)
ap.add_argument("--lam_cons", type=float, default=0.1)
ap.add_argument("--lam_lb", type=float, default=0.01)
ap.add_argument("--lam_z", type=float, default=0.001)
ap.add_argument("--lam_h", type=float, default=1.0)
ap.add_argument("--lam_kl", type=float, default=1.0)
ap.add_argument("--amp", default="none")
ap.add_argument("--eval_every_min", type=float, default=60.0)
ap.add_argument("--eval_every_steps", type=int, default=0)
ap.add_argument("--log_every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--heartbeat", type=int, default=1)
ap.add_argument("--small", action="store_true")
ap.add_argument("--no_final_eval", action="store_true")
ap.add_argument("--quick_eval_n", type=int, default=128)
ap.add_argument("--resume", default="")
ap.add_argument("--mem_frac_guard", type=float, default=0.90)
ap.add_argument("--keep", type=int, default=2)
ap.add_argument("--div_min_wall_min", type=float, default=30.0)
ap.add_argument("--dash_file", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "webapp", "data", "v2_live.json")))
```

## v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py

661 lines; SHA-256 `a3e111d3d1bf371fe97f4f0cf2a0bc0563d2fc712240f77f27356bfa48a6c01e`.

Primary copy: [v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py>).

HRM THINKER-25M dashboard server (read-only, stdlib only; D-036). nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 & open http://127.0.0.1:8765 Never writes under runs/, never imports torch, never touches the GPU. Log files are read incrementally (byte offset per file) and the assembled JSON is rebuilt only when a watched file changed.

Identical copies:

- [v3_design_review_2026-09-29/source_snapshot/run_v2_19M/webapp/server.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/source_snapshot/run_v2_19M/webapp/server.py>)

Imports:

```python
import json, os, re, subprocess, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
```

Symbols:

- [budget_h](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:20>) — lines 20–25: v2: the training-time budget from the run's config.json (48 h), else 24 h.
- [Tail](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:28>) — lines 28–65: Incremental JSONL / text reader: keeps a byte offset and the unterminated trailing fragment.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:30>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:34>).
- [cont_active](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:85>) — lines 85–86
- [active_dir](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:89>) — lines 89–90
- [cont_sched](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:93>) — lines 93–100: Schedule of the continuation from its config.json (step_offset = global step of its step 0).
- [update_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:103>) — lines 103–106
- [shifted](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:109>) — lines 109–119: Continuation rows on the global step axis (train rows carry gstep; restart and eval rows are shifted here).
- [shift_line_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:122>) — lines 122–123
- [sh](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:134>) — lines 134–138
- [system_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:141>) — lines 141–172: Cheap OS probes, at most once every 20 s.
- [_load_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:179>) — lines 179–186
- [_sig](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:189>) — lines 189–193
- [sample_live](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:196>) — lines 196–238: Called every 5 s by the sampler thread: GPU, trainer process, swap, battery (30 s), signatures, heartbeat.
- [sampler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:241>) — lines 241–247
- [live_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:250>) — lines 250–271
- [slim_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:274>) — lines 274–286
- [run_status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:289>) — lines 289–329
- [events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:332>) — lines 332–345: Resume / pause / guard / schedule events from the trainer output, for chart markers.
- [screens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:348>) — lines 348–359
- [plan_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:365>) — lines 365–386: v2 four stages from runs/chain.log: train 0 -> 24 h · 24 h evaluation · train 24 -> 48 h · final evaluation.
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:389>) — lines 389–430
- [final_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:433>) — lines 433–452: Final evaluations (D-044): full eval_final.json of the main run and, once it exists, of the continuation.
- [_f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:461>) — lines 461–462
- [build_digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:465>) — lines 465–539: Plain-text digest of every page's data (architecture, screens, main run, final eval, gates, continuation).
- [run_analysis](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:561>) — lines 561–578
- [analysis_body](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:581>) — lines 581–595
- [H](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:598>) — lines 598–649
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:599>), [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:602>), [end_headers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:605>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py:611>).

## v2_underperformance_review_2026-09-28/validate_report.py

83 lines; SHA-256 `2b63d44aa71799d9c44408e69bfccef1fe5bb486ee9c724a5bd2baace164812d`.

Primary copy: [v2_underperformance_review_2026-09-28/validate_report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/validate_report.py>).

Check the written review's central numbers against saved evidence, without running a model.

Imports:

```python
from pathlib import Path
import datetime,hashlib,json,math,re,statistics
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/validate_report.py:10>) — lines 10–11
- [near](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/validate_report.py:12>) — lines 12–12

## v2_underperformance_review_2026-09-28/verify_gradient_bridge.py

29 lines; SHA-256 `29e0d7da3fc94979db6cdafbcbefc4a62cd9a8d31b5d16e7312572ad21c3c2a8`.

Primary copy: [v2_underperformance_review_2026-09-28/verify_gradient_bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/verify_gradient_bridge.py>).

Additional targeted check: K=12, three windows, real capacity drops, accumulation scaling.

Imports:

```python
from pathlib import Path
import sys,json,copy,contextlib
import torch
import test_v2 as test
import train
from model import build_model
```

## v3_design/make_arch_v3draft.py

115 lines; SHA-256 `9606985fd1e1eeca6ec249243f031b45546b7053602a49e2bba956b0ea5f8775`.

Primary copy: [v3_design/make_arch_v3draft.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/make_arch_v3draft.py>).

Build webapp/data/arch_v3d.json (the draft_v3.md architecture pages: overview, 2D figures, 3D model). Every parameter count is instantiated from v3_ref.HRMThinkerV3 (the acceptance harness); MACs come from v3_budget's analytical counters at the final widths (encoder ff 1024 + 2 Engram reads, core own ff 736 + shared 1024, decoder ff 1024). Throughput rows and gates are quoted from draft_v3.md §7.2 / §9. CPU only; writes only into run_m2_25M/webapp/data. Run: nice -n 19 python3 make_arch_v3draft.py

Imports:

```python
import json, os, shutil
import torch
import v3_ref as R
import v3_budget as B
```

## v3_design/make_ref_v1v2.py

51 lines; SHA-256 `62e1944a95e285897190b434cb15318a2a3dc5ee3e7c174c1d2be88b672cdce0`.

Primary copy: [v3_design/make_ref_v1v2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/make_ref_v1v2.py>).

Reference curves for the v3 live page: v1 (run_m2_25M/runs/main) and v2 (run_v2_19M/runs/v2_48h) evaluations and training losses, read-only, written to run_m2_25M/webapp/data/ref_v1v2.json. Same data order in all runs (seed 0), so step k is the same tokens: these are matched-data references. Run: python3 make_ref_v1v2.py

Imports:

```python
import json, os
```

Symbols:

- [rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/make_ref_v1v2.py:12>) — lines 12–21
- [ev_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/make_ref_v1v2.py:24>) — lines 24–33

## v3_design/sync_record.py

28 lines; SHA-256 `345fc715a0063ffdc91acba5b63a6ea3ad0fa7990b6102cc3c126363076181ca`.

Primary copy: [v3_design/sync_record.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design/sync_record.py>).

Copy the record documents into run_m2_25M/webapp/data/record/ so the dashboard's Record page can render them. Read-only for the sources; re-run after the logs change. python3 sync_record.py

Imports:

```python
import json, os, shutil, time
```

## v3_design_review_2026-09-29/build_proposal.py

618 lines; SHA-256 `95e4448fed243cb2bfacacf5ca29e6fbf9d89a0dae6578006a7770bb0d40023c`.

Primary copy: [v3_design_review_2026-09-29/build_proposal.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/build_proposal.py>).

Write the v3 proposal and evidence ledgers. Reads only saved audit evidence.

Imports:

```python
from pathlib import Path
import json,statistics,math,re
```

Symbols:

- [render](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/build_proposal.py:8>) — lines 8–10
- [coverage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/build_proposal.py:583>) — lines 583–601

## v3_design_review_2026-09-29/check_design.py

97 lines; SHA-256 `a29d03df8a47868635875481ae3d7df1656f440a94d54163c519a934c2bdb63d`.

Primary copy: [v3_design_review_2026-09-29/check_design.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/check_design.py>).

CPU-only mechanical checks for the proposed v1-compatible v3 architecture. No training, optimizer updates, or edits to model/run directories.

Imports:

```python
from pathlib import Path
import sys,json,math,datetime
import torch
import torch.nn.functional as F
from model import build_model,param_table,macs_per_token
from hrm_encoder import engram_indices,build_masks
from hrm_decoder import plan_index
```

Symbols:

- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/check_design.py:43>) — lines 43–45
- [ans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/check_design.py:80>) — lines 80–80

## v3_design_review_2026-09-29/latest_depth_check.py

112 lines; SHA-256 `c0f8237c0d19d84c360e00588418781087491b307201ca7e11f39e897217421e`.

Primary copy: [v3_design_review_2026-09-29/latest_depth_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py>).

Read-only CPU checkpoint diagnostics. No MPS, optimizer, or writes in either run. Uses one CPU thread and memory-mapped checkpoint tensors assigned into a meta model. Diagnostics are interventions on fixed weights, not controlled retraining ablations.

Imports:

```python
from pathlib import Path
import argparse,datetime,gc,json,math,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model,nextlat_losses
from data import Canon,make_batch,target_mask
import concept_core_d6 as D6
```

Symbols:

- [save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py:36>) — lines 36–36
- [note](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py:37>) — lines 37–37
- [masked_ce](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py:41>) — lines 41–44
- [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py:46>) — lines 46–49
- [trace](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/latest_depth_check.py:52>) — lines 52–87

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--rows',type=int,default=8)
ap.add_argument('--skip-grad',action='store_true')
```

## v3_design_review_2026-09-29/plot_evidence.py

27 lines; SHA-256 `820d89a4c34c9294e39f0a044f6449bf4e720fb4cfdc00d65843700579f87143`.

Primary copy: [v3_design_review_2026-09-29/plot_evidence.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/plot_evidence.py>).

Imports:

```python
from pathlib import Path
import os,json,numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

## v3_design_review_2026-09-29/review_proposal.py

125 lines; SHA-256 `df53ea99c096091b6db0b0b46b6c581c2dcebd6b16832b9079602281abad5608`.

Primary copy: [v3_design_review_2026-09-29/review_proposal.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/review_proposal.py>).

Self-review key proposal claims against frozen evidence; does not touch runs.

Imports:

```python
from pathlib import Path
import datetime, hashlib, json, re, statistics
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/review_proposal.py:15>) — lines 15–16
- [displayed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/review_proposal.py:18>) — lines 18–20

## v3_design_review_2026-09-29/snapshot.py

45 lines; SHA-256 `bc7ea9df689dfed0fac62b182ecce8e36d9fb52226b5c1b55a4f07538d0f1ae0`.

Primary copy: [v3_design_review_2026-09-29/snapshot.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/snapshot.py>).

Freeze evidence for the v3 proposal. Does not modify either training run.

Imports:

```python
from pathlib import Path
import os,json,hashlib,datetime,ast,statistics
```

Symbols:

- [jl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v3_design_review_2026-09-29/snapshot.py:24>) — lines 24–24

## v41_tiny/code/additional_analysis_queue.py

13 lines; SHA-256 `8276bcd799582100b6e1370813158e3f3bdd6d46d2829b733aa4ed35d15f2fb8`.

Primary copy: [v41_tiny/code/additional_analysis_queue.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/additional_analysis_queue.py>).

Finish the matched-MPS control's analysis after its training completes.

Imports:

```python
import json,os,pathlib,subprocess,sys,time
```

## v41_tiny/code/analysis_queue.py

23 lines; SHA-256 `a64e35d5694c3b4c81052e7744020c941894482b5f7fe2fce37ddab7367daa3e`.

Primary copy: [v41_tiny/code/analysis_queue.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analysis_queue.py>).

Analyse completed registered runs while the training queue continues.

Imports:

```python
import json, os, pathlib, subprocess, sys, time
from experiment_queue import JOBS
```

## v41_tiny/code/analyze_vs.py

410 lines; SHA-256 `4f21034dd8a305fece6bd9e472a08d8751ac21158a14396196cf4f6d86dc9409`.

Primary copy: [v41_tiny/code/analyze_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py>).

Analysis of the v4-vs-LLM arms (CPU, reads checkpoints; never touches v3). Implements VS_LLM_REPORT.md §3 as pre-registered: paired per-row / per-item / per-episode / per-layout readings, closed-loop rollouts (P2), linear probes (P3) and v4's coarse-code NMI. python analyze_vs.py arm <run name> -> runs/<name>/analysis.json (+ per_unit.npz) python analyze_vs.py compare -> runs/vs_compare.json, runs/vs_compare.md

Imports:

```python
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import data_v4 as D
import pinpad as PP
import run_vs as R
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:27>) — lines 27–36
- [text_units](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:41>) — lines 41–54
- [item_hits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:58>) — lines 58–74
- [pin_units](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:79>) — lines 79–138: Per-episode: optimal-action rate (all actions; flag-2 actions), action NTP, position accuracy.
- [roll_layouts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:145>) — lines 145–151
- [rollouts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:155>) — lines 155–190: All 256 layouts in one batch; finished rows are dropped from the forward (they are never read again).
- [features](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:195>) — lines 195–223: Causal feature sites at positions whose next token is an action. Returns {site: (N, d)}, sub (N,), tgt (N,).
- [probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:226>) — lines 226–236
- [probes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:239>) — lines 239–253
- [code_nmi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:257>) — lines 257–288: v4 only, non-causal: NMI between the coarse code of each complete chunk and the chunk's majority subgoal.
- [nmi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:291>) — lines 291–297
- [arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:301>) — lines 301–348
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:352>) — lines 352–355
- [compare](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/analyze_vs.py:358>) — lines 358–403: Pairs v4 and LLM runs with the same (data, budget, seed); one table per (data, budget).

## v41_tiny/code/audit_analysis_v41.py

87 lines; SHA-256 `d189fbfff2fbb32efd7738089425803e7455eb9bbcebf892ee600e5ee66e5730`.

Primary copy: [v41_tiny/code/audit_analysis_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/audit_analysis_v41.py>).

Check saved inference units, final checkpoints and diagnostic consistency. This audits artifacts only; it never reruns training or changes its metrics.

Imports:

```python
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiment_queue import JOBS
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/audit_analysis_v41.py:12>) — lines 12–12
- [same](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/audit_analysis_v41.py:13>) — lines 13–14
- [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/audit_analysis_v41.py:16>) — lines 16–73

## v41_tiny/code/bench_v41.py

69 lines; SHA-256 `e7949215a4a0fb150359ea7671365cd3fd9189f0211c0546c2df10563658a333`.

Primary copy: [v41_tiny/code/bench_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/bench_v41.py>).

Synchronised CPU/MPS benchmarks and full forward/backward parity.

Imports:

```python
import copy, json, time, torch
import run_vs as R
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import speed_v3
```

Symbols:

- [sync](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/bench_v41.py:12>) — lines 12–14
- [parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/bench_v41.py:16>) — lines 16–44
- [benchmark](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/bench_v41.py:46>) — lines 46–61

## v41_tiny/code/build_report_v41.py

117 lines; SHA-256 `a2e1a999603409391b5b0ab1f71a7910185a7ccd89713353299099cb95bd2710`.

Primary copy: [v41_tiny/code/build_report_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py>).

Assemble a readable report while keeping every number traceable to raw files.

Imports:

```python
import argparse,datetime,json
from pathlib import Path
import numpy as np
from report_tables_v41 import main as tables_main,read,fmt
from decisions_v41 import main as decisions_main
from validate_artifacts_v41 import main as validate_main
```

Symbols:

- [link](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py:10>) — lines 10–10
- [js](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py:11>) — lines 11–11
- [mean](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py:12>) — lines 12–13
- [effect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py:14>) — lines 14–16
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/build_report_v41.py:18>) — lines 18–115

CLI arguments:

```python
ap.add_argument('--score',type=float)
```

## v41_tiny/code/compare_v41.py

69 lines; SHA-256 `03eedc2b9f322764645e8770ad09067e089859386dffa37d420ba96a1093ce82`.

Primary copy: [v41_tiny/code/compare_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compare_v41.py>).

Explicit paired comparisons; never collapse distinct trained configurations.

Imports:

```python
import json, math
from pathlib import Path
import numpy as np
```

Symbols:

- [record](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compare_v41.py:8>) — lines 8–11
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compare_v41.py:13>) — lines 13–16
- [comparison](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compare_v41.py:18>) — lines 18–41
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compare_v41.py:43>) — lines 43–67

## v4_tiny/code/compile_try.py

32 lines; SHA-256 `62343c02bc43c575bd28a9efb5474f968ec180736009c42d1399a4443363330b`.

Primary copy: [v4_tiny/code/compile_try.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/compile_try.py>).

<= 20 min experiment: does torch.compile (inductor, CPU) speed up a tiny-v4 training step? fp32 parity vs eager.

Identical copies:

- [v41_tiny/code/compile_try.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/compile_try.py>)

Imports:

```python
import os, sys, time, torch
import speed_v3
import model_v4 as M4, data_v4 as D
from tokenizers import Tokenizer
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/compile_try.py:13>) — lines 13–28

## v41_tiny/code/control_v41.py

110 lines; SHA-256 `bf3a894aab3f484113f2de7c5837a4c75bee1f367222653581cdefd075f477a5`.

Primary copy: [v41_tiny/code/control_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py>).

Exploratory oracle controllability, not RL or deployable latent reasoning. Frozen weights, real legal prefixes, zero offsets, 10 steps, three registered learning rates. Compare C input, intermediate S1 and output u. Optimise the probability of the expert optimal-action set; compare random projected offsets with the oracle run's per-example step norms. All prompts run the full budget, so both corrections and damage to initially correct predictions are measured.

Imports:

```python
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import analyze_vs as A
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import run_vs as R
from counterfactual_v41 import fixtures
```

Symbols:

- [project](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:21>) — lines 21–22
- [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:24>) — lines 24–37
- [predict](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:39>) — lines 39–49
- [values](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:51>) — lines 51–53
- [run_one](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:55>) — lines 55–80
- [metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:82>) — lines 82–87
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:89>) — lines 89–108

## v41_tiny/code/counterfactual_v41.py

93 lines; SHA-256 `5e35f5dd214fb650714c779e09c08a4c6f84d954e6fb9f62d1a05eb1dca9e6cf`.

Primary copy: [v41_tiny/code/counterfactual_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/counterfactual_v41.py>).

Change a relevant layout fact while keeping the entire action history valid. At the first pair-second decision, swap the target odd colour's location with another unvisited colour. Require disjoint optimal next-action sets. This tests whether the prediction follows the changed fact, rather than merely the prefix. No task/subgoal label is fed to the model.

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
import data_v4 as D
import pinpad as PP
import run_vs as R
import analyze_vs as A
```

Symbols:

- [fixtures](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/counterfactual_v41.py:17>) — lines 17–47
- [predictions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/counterfactual_v41.py:50>) — lines 50–61
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/counterfactual_v41.py:63>) — lines 63–87

## v41_tiny/code/data_v4.py

223 lines; SHA-256 `b3f6124bd31ff65d78d64bf5fef632d3baed9d1d8a6093d2115b831041ea294d`.

Primary copy: [v41_tiny/code/data_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py>).

Data for the v4 tiny experiment (CPU). Text: documents re-extracted from v3's packed shards (run_m2_25M/data/{train,val}, read-only), stored once as a flat token array + offsets (v4_tiny/data/). Rows are re-packed per k: every document padded to a multiple of k, rows of T tokens, the SAME document order for every k (so arms see the same documents; padding differs, as it would at full size). Probes: v3's ProbeFactory families recall1 / carry3 / multi4 with gaps of 80-200 tokens (beyond the decoder's reach of 70, so only the concept route can answer), fillers = real sentences from the training documents. Items are packed with k-token padding, o

Imports:

```python
import os
import random
import numpy as np
import torch
from tokenizers import Tokenizer
from hrm_encoder import engram_indices
from probes import ProbeFactory, VALUE_CANDIDATES
```

Symbols:

- [extract_docs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:27>) — lines 27–40: Real documents (pads stripped) from a random subset of v3's packed rows.
- [build_store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:43>) — lines 43–50
- [TextStore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:53>) — lines 53–60
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:54>), [doc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:59>).
- [token_freq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:63>) — lines 63–64
- [pack_sequences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:68>) — lines 68–91: Pack token sequences (each its own document) into rows of T: each padded to a multiple of k, a new row when it does not fit; the row tail is PAD with a fresh doc id. Sequences longer than T are cut into T-sized pieces. Returns tok (R, T), d
- [TinyProbes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:95>) — lines 95–110: v3's probe families with fillers taken from real sentences and gaps of 80-200 tokens ('far': 200-300).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:97>), [gap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:106>), [item](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:109>).
- [filler_sentences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:113>) — lines 113–124: Sentences (5-40 tokens) cut at '.' from real training documents.
- [Canon](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:128>) — lines 128–133
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:129>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:132>).
- [to_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:136>) — lines 136–138
- [MixedStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:141>) — lines 141–185: Training rows: with probability p_syn a row of packed probe items, else a row of packed text documents. The row-type sequence, the document order and the item sequence depend only on the seed, not on k.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:144>), [_text_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:154>), [_syn_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:163>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:180>).
- [val_text_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:188>) — lines 188–196
- [probe_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:199>) — lines 199–201
- [item_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/data_v4.py:204>) — lines 204–206

## v41_tiny/code/decisions_v41.py

53 lines; SHA-256 `b833df122bb52b626377056f67d5bb52a016e9b358d51f5434b9a0e89cf677d6`.

Primary copy: [v41_tiny/code/decisions_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/decisions_v41.py>).

Apply the registered rules without substituting secondary controls.

Imports:

```python
import json
from pathlib import Path
import numpy as np
from compare_v41 import paired
from report_tables_v41 import agg
```

Symbols:

- [metric](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/decisions_v41.py:9>) — lines 9–9
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/decisions_v41.py:10>) — lines 10–52

## v41_tiny/code/experiment_queue.py

46 lines; SHA-256 `cdebc433d75610fc60a88680d877d307ff0b09e58a3cce4f21b5e4919908ff97`.

Primary copy: [v41_tiny/code/experiment_queue.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/experiment_queue.py>).

Run the registered arms without touching previous experiments.

Imports:

```python
import datetime, hashlib, json, os, pathlib, subprocess, sys, time
```

Symbols:

- [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/experiment_queue.py:21>) — lines 21–24

## v41_tiny/code/gradient_audit.py

60 lines; SHA-256 `079969a3c6e3affadaebc356111dfde7557f3e4122872513331a9b0103d0fb79`.

Primary copy: [v41_tiny/code/gradient_audit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/gradient_audit.py>).

Separate action NTP, ordinary NTP and weighted concept-loss gradients.

Imports:

```python
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import analyze_vs as A
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import run_vs as R
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/gradient_audit.py:13>) — lines 13–58

## v41_tiny/code/long_audit_v41.py

62 lines; SHA-256 `c6741710998373bf15ff4e7a37e3d59af3b32cad481cc842f9ca39f2d4069d42`.

Primary copy: [v41_tiny/code/long_audit_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/long_audit_v41.py>).

Decompose long-episode action NLL into action choice and vocabulary mass.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A,data_v4 as D,pinpad as PP,run_vs as R
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/long_audit_v41.py:10>) — lines 10–44

## v4_tiny/code/model_llm.py

158 lines; SHA-256 `dfd7a7b1430b71a5d1f9f75855abf3d1fc6c3add71ca6b65788a84e104598f3f`.

Primary copy: [v4_tiny/code/model_llm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py>).

LLM-stack baseline for the v4-vs-LLM tiny experiment: a plain decoder-only transformer, pre-norm residual stream, 3 : 1 sliding-window : full attention (pattern SSSF repeated), recurrence placed and sized to mirror v4. Mirror of v4 (tiny, recommended defaults: pooler attention only, decoder Engram reads off): v4 encoder 8 layers SSSFSSSF, F layers applied twice (enc_loop 2) -> core 4 layers x M = 2 -> decoder 4 layers LLM prelude 8 layers SSSFSSSF, F layers applied twice (enc_loop 2) -> middle 4 layers SSSF x M = 2 -> coda 4 SSSF 16 unique layers each, 22 sequential layer applications each. The LLM has no hierarchy: one residual stream over a

Identical copies:

- [v41_tiny/code/model_llm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_llm.py>)

Imports:

```python
import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
import compat
import speed as _speed
from hrm_encoder import EncoderConfig, GatedAttention, SwiGLU, ZCRMSNorm, rms, build_masks, rope_cache
```

Symbols:

- [LLMLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:31>) — lines 31–41
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:32>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:39>).
- [LLMStack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:44>) — lines 44–101
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:45>), [make_schedule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:60>), [_init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:73>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:80>).
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:104>) — lines 104–106
- [target_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:109>) — lines 109–110
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:113>) — lines 113–117: Same call shape as model_v4.forward (eidx unused: no Engram).
- [loss_terms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:120>) — lines 120–126
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:129>) — lines 129–135
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_llm.py:141>) — lines 141–148

## v4_tiny/code/model_v4.py

565 lines; SHA-256 `78b1c56cf22b06aac9b81a72f5c937b28c12a272c076b52b583b334eac8ba819`.

Primary copy: [v4_tiny/code/model_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py>).

HRM Thinker v4 (draft_v4.md, revision 2), size-parametric: FULL = 19,898,011 compute parameters at k = 6; TINY for the CPU experiment in v4_tiny/. Encoder (§3): v3's 8 layers SSSFSSSF (F layers 6 heads), L/G gated residual, ONE Engram gather feeding read 1 (after layer 0, with conv) and read 2 (after layer 4); NEW: F layers 3 and 7 applied twice in place (enc_loop = 2); NEW: shared SwiGLU beside every layer's own FFN, y = FFN_l(x) + g_l * FFN_shared(x) (g_l = 1, shared W2 = 0); NEW: pooler over k tokens (§3.5.4): 4-head attention (W_V = W_O = I, q = b = p = 0) then a SwiGLU on the pooled vector (W2 = 0, g = 1); pads masked; an empty chunk fal

Identical copies:

- [v41_tiny/code/model_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py>)

Imports:

```python
import math
import sys
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
import compat
import speed as _speed
from hrm_encoder import (EncoderConfig, EncoderLayer, Engram, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms,
                         build_masks, rope_cache)
from hrm_decoder import DecoderConfig, PlanBuilder, ConfinedLayer
```

Symbols:

- [EngramRead2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:41>) — lines 41–60: v3's second Engram read (also the anatomy of the decoder reads): own W_k, W_v (zero-init), gated read of one stream, write to n_write streams, no conv.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:44>), [gate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:52>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:57>).
- [engram_read1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:63>) — lines 63–68: v1/v3 Engram.forward with the gathered rows passed in (identical arithmetic).
- [layer_forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:71>) — lines 71–79: compat's EncoderLayer forward, plus the shared FFN as a parallel branch on the same read and write gate.
- [PoolLinear](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:82>) — lines 82–93: v3's pool: one linear map over the concatenated chunk, initialised to the exact mean (pads included).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:84>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:91>).
- [PoolAttnFFN](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:96>) — lines 96–131: §3.5.4: H-head attention pooling over the chunk's k tokens (shared W_K, W_V across slots, position vectors p_r, position biases b_{h,r}), then an optional SwiGLU on the pooled vector. Init = masked mean of the real tokens.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:99>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:115>).
- [PoolGConv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:134>) — lines 134–171: Owner's convolution hypothesis (2026-10-01), content-gated so it is not v3's pool in disguise. A plain Conv1d(d, d, kernel k, stride k) over the chunk IS v3's linear pool (its weight reshapes to W_p), so the 'linear' arm already covers it.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:143>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:158>).
- [EncoderV4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:174>) — lines 174–239
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:175>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:204>), [_fwd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:214>).
- [CoreLayerV4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:243>) — lines 243–259
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:244>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:252>).
- [ConceptCoreV4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:262>) — lines 262–382
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:263>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:284>), [fine_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:296>), [head](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:302>), [assign](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:314>), [vq_terms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:323>), [ncp_terms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:338>), [init_codebooks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:348>), [restart_dead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:358>).
- [_zero_moments](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:385>) — lines 385–391
- [kpp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:394>) — lines 394–403: k-means++ seeding of n centres from rows of x (with replacement when x has fewer distinct rows).
- [DecoderV4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:407>) — lines 407–439
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:408>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:418>).
- [HRMThinkerV4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:443>) — lines 443–458
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:444>).
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:461>) — lines 461–463
- [chunk_masks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:466>) — lines 466–473: nonempty: >= 1 real token; complete: k real tokens; pair_ok (j -> j+1): j nonempty, j+1 complete, same doc.
- [target_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:476>) — lines 476–477
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:480>) — lines 480–498: Evaluation path. Returns logits (B, T, V) and aux.
- [loss_terms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:504>) — lines 504–519: One batch, means over the batch's own counts. Returns (loss, dict of detached terms, aux).
- [param_table](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:523>) — lines 523–540: Repository convention: compute = encoder (no embedding, no Engram table) + core (no codebooks) + decoder.
- [param_groups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/model_v4.py:547>) — lines 547–554

## v41_tiny/code/model_v41.py

154 lines; SHA-256 `3e259a20837670c4dd44bbbb80c6d421149296e82a44ca161ea5615ee3c459fb`.

Primary copy: [v41_tiny/code/model_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py>).

v4.1: token-memory retrieval at decoder layer 2, with private plans. The v4 backbone and initial weights are preserved. Only R's four zero-initialised log temperatures are added. The parameter-identical stack control reads its private decoder state for K/V in all four layers; it has no short convolutions.

Imports:

```python
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_v4 as M4
from hrm_encoder import apply_partial_rope, build_masks, rope_cache
from hrm_decoder import ConfinedLayer
```

Symbols:

- [MemoryLayer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:16>) — lines 16–45
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:17>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:24>).
- [DecoderV41](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:48>) — lines 48–95
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:49>), [masks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:68>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:72>).
- [build_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:98>) — lines 98–105
- [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:108>) — lines 108–129
- [loss_terms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:132>) — lines 132–150

## v4_tiny/code/mutants_v4.py

94 lines; SHA-256 `4fa1ba7ff1c9a91f3c6f729faada728331ffdfe01a41281eda70d51feaa585f4`.

Primary copy: [v4_tiny/code/mutants_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_v4.py>).

Deliberate-defect mutants for test_v4 (draft §8.5 step 3: every acceptance test needs a mutant that fails it). Each mutant rewrites ONE line of a function's source (string replace, exec in the defining module) and runs the tests that should catch it. A mutant that SURVIVES (all its tests pass) means the harness is blind to that defect.

Identical copies:

- [v41_tiny/code/mutants_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v4.py>)

Imports:

```python
import inspect
import textwrap
import torch
import model_v4 as M4
import hrm_decoder as HD
import test_v4 as TV
```

Symbols:

- [patch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_v4.py:17>) — lines 17–28: old/new: one string each, or lists of strings applied in order.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_v4.py:72>) — lines 72–90

## v41_tiny/code/mutants_v41.py

53 lines; SHA-256 `9036c623031fc0b6ba1a426d98cef33aa3bd3611b8237e6c14ecbb1dc9486b58`.

Primary copy: [v41_tiny/code/mutants_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py>).

The new tests must reject plausible implementation defects by assertion.

Imports:

```python
import contextlib, json, sys, os
import torch
import model_v41 as M41
import hrm_decoder as HD
import test_v41 as T
import speed_v3
```

Symbols:

- [patch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:11>) — lines 11–14
- [noncausal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:16>) — lines 16–18
- [nodoc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:20>) — lines 20–23
- [onlylocal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:25>) — lines 25–26
- [leakinit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:29>) — lines 29–30
- [wrongindex](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_v41.py:32>) — lines 32–35

## v4_tiny/code/mutants_vs.py

152 lines; SHA-256 `0ae26f9faca8ae82b3185e099eb0c3b5700a328c8550f999a7176574bf50ddd7`.

Primary copy: [v4_tiny/code/mutants_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py>).

Deliberate defects for the v4-vs-LLM tests: each mutant must make at least one named test fail (by assertion). python mutants_vs.py

Identical copies:

- [v41_tiny/code/mutants_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/mutants_vs.py>)

Imports:

```python
import contextlib
import sys
import numpy as np
import torch
import model_llm as ML
import pinpad as PP
import data_v4 as D
import run_vs as R
import test_vs as TV
import speed_v3
```

Symbols:

- [patch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:21>) — lines 21–27
- [m_swa_full](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:30>) — lines 30–33
- [m_no_causal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:36>) — lines 36–40
- [m_no_doc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:43>) — lines 43–48
- [_two](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:52>) — lines 52–54
- [m_schedule_ignores_M](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:57>) — lines 57–59
- [m_expert_ignores_colours](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:62>) — lines 62–68
- [m_position_lag](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:71>) — lines 71–76
- [m_flags_swapped](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:79>) — lines 79–81
- [m_metric_misaligned](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:84>) — lines 84–89
- [m_llm_skips_init_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:92>) — lines 92–97
- [_main_no_skip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:100>) — lines 100–107
- [m_val_fillers_from_train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:110>) — lines 110–116
- [m_ffn_256](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/mutants_vs.py:119>) — lines 119–121

## v41_tiny/code/parallel_queue.py

54 lines; SHA-256 `ce43a58d3f0bdcb97e7fc9d8895c0de9ef460fb082fc697c43bd79ea2a23605b`.

Primary copy: [v41_tiny/code/parallel_queue.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/parallel_queue.py>).

Two CPU workers for the remaining registered arms; same budgets and rows.

Imports:

```python
import datetime,json,os,pathlib,signal,subprocess,sys,time
from experiment_queue import JOBS
```

Symbols:

- [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/parallel_queue.py:7>) — lines 7–10

## v41_tiny/code/phase_audit.py

25 lines; SHA-256 `70c9ad9cdec343d6c73c0477c5f89cf7d81509a480259dba7328fc13be06a32f`.

Primary copy: [v41_tiny/code/phase_audit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/phase_audit.py>).

Observational chunk-phase comparison on the same held-out episodes. Action queries are even positions; with k=6 their concept plan lags 1,3 or 5 tokens. Phase effects are observational because path length and layout differ.

Imports:

```python
import json, sys
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A
import run_vs as R
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/phase_audit.py:13>) — lines 13–23

## v4_tiny/code/pinpad.py

230 lines; SHA-256 `fef776c277f8960610fa953ce56e7abd5f993b3f640a5dc89157113ff522d8d8`.

Primary copy: [v4_tiny/code/pinpad.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py>).

Gridworld-pinpad (Kobayashi et al. 2025, arXiv 2512.20605, Appendix A.1), tokenised for the v4-vs-LLM tiny experiment. Environment (paper A.1.1-A.1.2): G x G grid (G = 7), O = 8 uniquely coloured cells, W = 4 walls, episode limit 100 steps. 4 actions (N, E, S, W); moving into a wall or off the grid is a no-op. A coloured cell is visited when the agent moves onto it from a different cell. Visiting the wrong colour ends the episode (failure); visiting all task colours in order is success. Colours, walls and the start cell are sampled without overlap at the start of every episode. Tasks: the 16 pretraining tasks of Table A1 (compositions of the 

Identical copies:

- [v41_tiny/code/pinpad.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/pinpad.py>)

Imports:

```python
import random
from collections import deque
import numpy as np
```

Symbols:

- [pos_tok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:44>) — lines 44–45
- [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:48>) — lines 48–55
- [bfs_dist](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:58>) — lines 58–73: Distance to target from every cell over cells not in `blocked` (target itself is allowed). -1 = unreachable.
- [optimal_actions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:76>) — lines 76–87: Set of actions on a shortest path from cell to target avoiding walls and every other coloured cell.
- [sample_layout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:90>) — lines 90–92
- [expert_episode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:95>) — lines 95–116: Returns dict(tokens, sub, tgt, opt, layout) for one successful expert episode on a fresh layout.
- [header_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:119>) — lines 119–126
- [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:129>) — lines 129–137
- [boundary_flags](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:140>) — lines 140–154: For every ACT token: 1 if it is the first action after a visit (or the episode's first action) whose target is the FIRST colour of a pair (the next pair is latent), 2 if the target is the second colour of a pair (determined by the subgoal s
- [PinpadStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:157>) — lines 157–176: Training rows of packed expert episodes on the 16 pretraining tasks, fresh layouts every episode. Depends only on the seed (identical rows for every architecture).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:160>), [_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:167>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:174>).
- [eval_episodes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:179>) — lines 179–182
- [Ident](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:185>) — lines 185–188: Engram canonicalisation for pinpad: identity (the text Canon maps 8,192 BPE ids).
  Methods: [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:187>).
- [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:191>) — lines 191–202
- [Env](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:206>) — lines 206–219: Task-free environment for rollouts of a generative model: records the order of coloured-cell visits. The rollout is judged afterwards (does the visit order form a pretraining task? pair integrity?).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:209>), [act](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/pinpad.py:214>).

## v41_tiny/code/plan_diagnostics.py

70 lines; SHA-256 `35be4135378cc597526e0a951a0ffcf6ad1dbfa89928ecec24e80059cb749d5b`.

Primary copy: [v41_tiny/code/plan_diagnostics.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plan_diagnostics.py>).

Calibrated plan controls estimated on independent training examples.

Imports:

```python
import contextlib, json, sys
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import run_vs as R
```

Symbols:

- [constant_contents](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plan_diagnostics.py:13>) — lines 13–23
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plan_diagnostics.py:26>) — lines 26–68

## v41_tiny/code/plot_controls_v41.py

82 lines; SHA-256 `667a11cb5e7c8d71bfabf7842407a51029e94f749e52454010a9ed0dbf6006ed`.

Primary copy: [v41_tiny/code/plot_controls_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_controls_v41.py>).

Static research figure: distinguish capability, reliance and control authority.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

Symbols:

- [get](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_controls_v41.py:10>) — lines 10–12
- [bars](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_controls_v41.py:13>) — lines 13–21
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_controls_v41.py:22>) — lines 22–81

## v41_tiny/code/plot_counterfactual_v41.py

58 lines; SHA-256 `4c0cd0fda6938cf84feb642c2436f755b9b5cac47d349423ab4ffd1dd638db47`.

Primary copy: [v41_tiny/code/plot_counterfactual_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_counterfactual_v41.py>).

Illustrate a measured failure, with selection rule and aggregate rates shown.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pinpad as PP
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_counterfactual_v41.py:13>) — lines 13–57

## v41_tiny/code/plot_v41.py

69 lines; SHA-256 `1a530fa4b36ee2722e209911b9b60b82a0fb233db450581b253c10c28fcc82e9`.

Primary copy: [v41_tiny/code/plot_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_v41.py>).

Standalone scientific plots from saved evaluations, including seed spread.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

Symbols:

- [curve](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_v41.py:13>) — lines 13–25
- [draw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_v41.py:27>) — lines 27–36
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/plot_v41.py:38>) — lines 38–67

## v4_tiny/code/prof_v4.py

29 lines; SHA-256 `ae171f4ef04533a098849a93f183461c3dda6435f717f7208aecb012d0bc4219`.

Primary copy: [v4_tiny/code/prof_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/prof_v4.py>).

Profile a few training steps of the tiny model on CPU (memo on), B = 8 vs 16.

Identical copies:

- [v41_tiny/code/prof_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/prof_v4.py>)

Imports:

```python
import sys, time, torch
import model_v4 as M4, data_v4 as D
from tokenizers import Tokenizer
import os
import speed_v3
```

## v4_tiny/code/reach_vs.py

40 lines; SHA-256 `2b4aa26532e75dbe5f56c8a2ed1a3b3c6c8be9562d7cd77f427e6fa509a8fbf1`.

Primary copy: [v4_tiny/code/reach_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/reach_vs.py>).

Reach diagnostic (VS_LLM_REPORT §6, pre-specified 00:55): per-bucket action readings for a v4 / LLM pair. python reach_vs.py vs_pin_v4_s0_6M vs_pin_llm_s0_6M -> runs/<v4 name>/reach.json and a printed table

Identical copies:

- [v41_tiny/code/reach_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/reach_vs.py>)

Imports:

```python
import json
import os
import sys
import numpy as np
import torch
import analyze_vs as A
import run_vs as R
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/reach_vs.py:15>) — lines 15–36

## v4_tiny/code/reads_check.py

18 lines; SHA-256 `3740bfae58f8d8a787521294e1f5fdbe5c488384e4d8236bf119d3e8f4f45cd8`.

Primary copy: [v4_tiny/code/reads_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/reads_check.py>).

Decoder Engram reads off: paired NTP delta on probe rows built from the TRAINING filler pool vs the validation pool.

Identical copies:

- [v41_tiny/code/reads_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/reads_check.py>)

Imports:

```python
import os, sys, json, math, torch
import data_v4 as D, decomp_v4tiny as DC
from tokenizers import Tokenizer
```

## v4_tiny/code/recall_check.py

19 lines; SHA-256 `3db0a4d8e847a8a7f5b4eacebd3c454b26fc5a80a321b56bb335f62d4c8bc39e`.

Primary copy: [v4_tiny/code/recall_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/recall_check.py>).

Recall with items built from the TRAINING filler pool (fresh rng seed, new names/values) vs the validation pool.

Identical copies:

- [v41_tiny/code/recall_check.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/recall_check.py>)

Imports:

```python
import os, sys, json, torch
import data_v4 as D, decomp_v4tiny as DC, run_v4tiny as RV
from tokenizers import Tokenizer
```

## v41_tiny/code/report_tables_v41.py

196 lines; SHA-256 `086d3b863f1ec9811ece832c67e9508b4b9e074f7d36301af6b364017b970505`.

Primary copy: [v41_tiny/code/report_tables_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py>).

Regenerate auditable result tables from final checkpoints and saved units.

Imports:

```python
import datetime,json,math
from pathlib import Path
import numpy as np
from experiment_queue import JOBS
from compare_v41 import paired,main as compare_main
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py:10>) — lines 10–12
- [avg](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py:13>) — lines 13–13
- [fmt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py:14>) — lines 14–14
- [agg](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py:15>) — lines 15–19
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/report_tables_v41.py:21>) — lines 21–194

## v41_tiny/code/run_vs.py

416 lines; SHA-256 `8c9ce4159bee7c90118059f4493008e995c8dc56fef34000b280cfd81b3daec0`.

Primary copy: [v41_tiny/code/run_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py>).

v4 vs LLM-stack tiny experiment (CPU): train and evaluate one arm. python run_vs.py --name vs_text_v4_s0 --arch v4 --data text --seed 0 --real_tokens 5e6 python run_vs.py --name vs_pin_llm_s1 --arch llm --data pinpad --seed 1 --real_tokens 3e6 Writes v4_tiny/runs/<name>/{log.jsonl, final.json, ckpt.pt}. Held equal across arch: training rows (same stream, same seed, rows padded per v4's k = 6 for both), tokenizer, tied embedding, NTP target mask, optimizer (AdamW 0.9/0.95, wd 0.1 by v4's decay rule), lr schedule (warm-up 400, cosine to 0.1 of peak over the token budget), B, T, clip, token budget, threads. v4 adds its own concept terms (0.05 CE

Imports:

```python
import argparse
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import model_v41 as M41
import model_llm as ML
import data_v4 as D
import pinpad as PP
```

Symbols:

- [get_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:36>) — lines 36–60
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:64>) — lines 64–73
- [is_v4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:76>) — lines 76–77
- [fwd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:80>) — lines 80–86
- [loss_fn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:89>) — lines 89–92
- [ptable](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:95>) — lines 95–97
- [interventions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:100>) — lines 100–109
- [flops_per_token](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:112>) — lines 112–139: Forward multiply-adds per token (x2 = FLOPs), matmuls + attention scores/values; v4's core at T / k positions.
- [val_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:144>) — lines 144–155
- [item_acc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:159>) — lines 159–183: Teacher-forced greedy exact match of every answer span (prediction for token i is logits\[i-1\]).
- [item_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:187>) — lines 187–189
- [pin_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:193>) — lines 193–207: Pack episodes; per-row label arrays aligned with token positions.
- [pin_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:211>) — lines 211–247: NTP split by token kind; action optimality (argmax over the 4 actions in the expert's optimal set), split by boundary flag; position exact accuracy.
- [pin_eval_sets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:250>) — lines 250–252
- [text_setup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:256>) — lines 256–268
- [pin_setup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:271>) — lines 271–272
- [periodic_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:275>) — lines 275–285
- [final_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:288>) — lines 288–304
- [save_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:308>) — lines 308–312
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py:315>) — lines 315–412

CLI arguments:

```python
ap.add_argument("--name", required=True)
ap.add_argument("--arch", choices=["v4", "v41", "v41stack", "llm"], required=True)
ap.add_argument("--data", choices=["text", "pinpad"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--T", type=int, default=480)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--p_syn", type=float, default=0.5)
ap.add_argument("--real_tokens", type=float, default=5e6)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--eval_every_tokens", type=float, default=1e6)
ap.add_argument("--n_probe", type=int, default=128)
ap.add_argument("--n_probe_final", type=int, default=192)
ap.add_argument("--restart_every", type=int, default=200)
ap.add_argument("--block", type=int, default=60)
ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
ap.add_argument("--checkpoint_every", type=int, default=500)
ap.add_argument("--over", default="", help="model overrides, e.g. ff=296")
```

## v41_tiny/code/secondary_queue.py

37 lines; SHA-256 `ea23bcdf79f0f82ddfc29e75f41ed5469db90816c1255adf7348b12dcdf936f3`.

Primary copy: [v41_tiny/code/secondary_queue.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/secondary_queue.py>).

Read-only checkpoint diagnostics, explicitly secondary to the main matrix.

Imports:

```python
import json, os, pathlib, subprocess, sys, time
```

## v4_tiny/code/test_v4.py

493 lines; SHA-256 `25d31edfff789873192eee356886ec159f500f9335824ddd44ea2cc662757884`.

Primary copy: [v4_tiny/code/test_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py>).

Acceptance tests for model_v4 (draft_v4 §3.8, §4.9, §5.5, §6.2), float64 on CPU, at k = 4, 6 and 8. Every structural test runs on a RANDOMIZED model: at init the pooler is uniform, the FFN branches and decoder Engram reads output exactly zero, so an un-randomized test would be vacuous for every new branch. Run: python test_v4.py (prints one PASS line per test; any failure raises)

Identical copies:

- [v41_tiny/code/test_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v4.py>)

Imports:

```python
import math
import sys
import torch
import torch.nn.functional as F
import model_v4 as M4
from hrm_encoder import engram_indices
from hrm_decoder import plan_index
```

Symbols:

- [ok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:21>) — lines 21–23
- [randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:26>) — lines 26–31
- [mk](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:34>) — lines 34–39
- [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:42>) — lines 42–48
- [logits_of](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:51>) — lines 51–54
- [changed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:57>) — lines 57–59: positions (row 0) whose last-dim values differ at all
- [test_causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:63>) — lines 63–74
- [test_doc_isolation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:78>) — lines 78–86
- [test_reach_L](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:90>) — lines 90–103
- [test_pool_init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:107>) — lines 107–128
- [test_alive](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:132>) — lines 132–168
- [test_codebook_isolation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:172>) — lines 172–192
- [test_frames](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:196>) — lines 196–224
- [test_decoder](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:228>) — lines 228–278
- [test_empty_and_pads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:282>) — lines 282–316
- [test_pool_selects](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:320>) — lines 320–367: §3.8 test 9: train only the pooler (toy objective: copy the marked token's vector) -> one head puts > 0.9 of its weight on the marked token at EVERY slot. v3's linear pool (per-slot weights) cannot.
- [test_restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:371>) — lines 371–394
- [test_v3_identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:398>) — lines 398–417
- [test_switch_offs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:421>) — lines 421–433
- [test_block_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:437>) — lines 437–448
- [test_decoder_bypass](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_v4.py:452>) — lines 452–475

## v41_tiny/code/test_v41.py

244 lines; SHA-256 `b4006c99c1836dba9c7998f338795fa40d49dc540264e677aef944963907d58a`.

Primary copy: [v41_tiny/code/test_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py>).

v4.1 invariants tested at input, concept, plan, memory and gradient sites.

Imports:

```python
import argparse, copy, gc, json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import model_v41 as M41
import data_v4 as D
import pinpad as PP
import run_vs as R
import hrm_decoder as HD
from hrm_encoder import engram_indices, build_masks, rope_cache
import speed_v3
```

Symbols:

- [model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:18>) — lines 18–25
- [inputs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:27>) — lines 27–32
- [decoder_inputs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:34>) — lines 34–40
- [test_parameter_ledger](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:42>) — lines 42–53
- [test_backbone_identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:55>) — lines 55–64
- [test_retrieval_oracle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:66>) — lines 66–83
- [test_R_reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:85>) — lines 85–97
- [test_R_document_isolation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:99>) — lines 99–106
- [test_private_plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:108>) — lines 108–118
- [test_stack_positive_control](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:120>) — lines 120–127
- [test_null_alignment](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:129>) — lines 129–137
- [test_no_decoder_engram](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:139>) — lines 139–145
- [test_full_causality_documents](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:147>) — lines 147–157
- [test_block_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:159>) — lines 159–165
- [test_core_input_reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:167>) — lines 167–176
- [test_gradient_routes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:178>) — lines 178–197
- [test_padding_finite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:199>) — lines 199–208
- [test_runner_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py:210>) — lines 210–227

CLI arguments:

```python
ap.add_argument("--block",type=int,default=60)
ap.add_argument("--only",default="")
```

## v4_tiny/code/test_vs.py

318 lines; SHA-256 `f978c40861b4d980d3341cc647437f2240c983f143d3c0a05676476f18ea7676`.

Primary copy: [v4_tiny/code/test_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py>).

Acceptance tests for the v4-vs-LLM experiment (CPU, float64 where exactness matters). python test_vs.py \[--block 60\]

Identical copies:

- [v41_tiny/code/test_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_vs.py>)

Imports:

```python
import argparse
import random
import sys
import numpy as np
import torch
import model_v4 as M4
import model_llm as ML
import data_v4 as D
import pinpad as PP
import run_vs as R
```

Symbols:

- [rand_llm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:20>) — lines 20–27
- [rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:30>) — lines 30–34
- [test_param_match](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:38>) — lines 38–52
- [test_causal_and_docs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:56>) — lines 56–70
- [test_window](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:73>) — lines 73–96: An S attention module sees exactly the last 60 positions; an F module sees everything causal.
- [test_loops](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:99>) — lines 99–110
- [test_block_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:113>) — lines 113–125
- [test_same_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:129>) — lines 129–144
- [test_runner_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:147>) — lines 147–171: The real training loop (R.main, 4 steps) feeds byte-identical batches to v4 and the LLM.
- [test_pinpad_expert](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:175>) — lines 175–216
- [test_boundary_flags](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:219>) — lines 219–228
- [test_metrics_oracle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:231>) — lines 231–251: pin_metrics on an oracle (logits = one-hot of the true next token, with a non-optimal action shift for a mutant check) gives action optimality 1 and position accuracy 1; a shifted oracle does not.
- [test_filler_fix](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:255>) — lines 255–264
- [test_train_steps](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/test_vs.py:267>) — lines 267–295

CLI arguments:

```python
ap.add_argument("--block", type=int, default=60)
ap.add_argument("--only", default="")
```

## v41_tiny/code/validate_artifacts_v41.py

70 lines; SHA-256 `95ff07d39760da26124b244ae58c25c0ed8cac1c9c6b3e009cb5375b32968c85`.

Primary copy: [v41_tiny/code/validate_artifacts_v41.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/validate_artifacts_v41.py>).

Audit completed artifacts, budgets, provenance and finite training logs.

Imports:

```python
import hashlib,json,math
from pathlib import Path
from experiment_queue import JOBS
from secondary_queue import jobs as diagnostic_jobs
```

Symbols:

- [sha](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/validate_artifacts_v41.py:9>) — lines 9–9
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/validate_artifacts_v41.py:10>) — lines 10–69

## v4_tiny/code/analyze_v4tiny.py

106 lines; SHA-256 `cf04172be7f5bbc2a7a37c51b9092320bc7a1338199cc289963445397c5752a4`.

Primary copy: [v4_tiny/code/analyze_v4tiny.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_v4tiny.py>).

Summarise every arm in v4_tiny/runs: eval curves (log.jsonl) and final evaluations (final.json). python analyze_v4tiny.py \[--md out.md\] (stdlib only; safe to run while arms train)

Imports:

```python
import glob
import json
import math
import os
import sys
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_v4tiny.py:13>) — lines 13–29
- [se](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_v4tiny.py:32>) — lines 32–33
- [fmt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_v4tiny.py:36>) — lines 36–37
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_v4tiny.py:40>) — lines 40–102

## v4_tiny/code/analyze_vs.py

376 lines; SHA-256 `12c156304511c009c31cddc4992ab092023811decb56a4069f83f0275dfc737f`.

Primary copy: [v4_tiny/code/analyze_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py>).

Analysis of the v4-vs-LLM arms (CPU, reads checkpoints; never touches v3). Implements VS_LLM_REPORT.md §3 as pre-registered: paired per-row / per-item / per-episode / per-layout readings, closed-loop rollouts (P2), linear probes (P3) and v4's coarse-code NMI. python analyze_vs.py arm <run name> -> runs/<name>/analysis.json (+ per_unit.npz) python analyze_vs.py compare -> runs/vs_compare.json, runs/vs_compare.md

Imports:

```python
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import data_v4 as D
import pinpad as PP
import run_vs as R
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:27>) — lines 27–36
- [text_units](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:41>) — lines 41–54
- [item_hits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:58>) — lines 58–74
- [pin_units](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:79>) — lines 79–127: Per-episode: optimal-action rate (all actions; flag-2 actions), action NTP, position accuracy.
- [roll_layouts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:134>) — lines 134–140
- [rollouts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:144>) — lines 144–178: All 256 layouts in one batch; finished rows are dropped from the forward (they are never read again).
- [features](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:183>) — lines 183–206: Causal feature sites at positions whose next token is an action. Returns {site: (N, d)}, sub (N,), tgt (N,).
- [probe](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:209>) — lines 209–219
- [probes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:222>) — lines 222–236
- [code_nmi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:240>) — lines 240–259: v4 only, non-causal: NMI between the coarse code of each complete chunk and the chunk's majority subgoal.
- [nmi](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:262>) — lines 262–268
- [arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:272>) — lines 272–314
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:318>) — lines 318–321
- [compare](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/analyze_vs.py:324>) — lines 324–369: Pairs v4 and LLM runs with the same (data, budget, seed); one table per (data, budget).

## v4_tiny/code/data_v4.py

223 lines; SHA-256 `98d45a7b4c04e169210539d7f00560fd3d2746c47e8e8ae079b4c7d6bd8ee364`.

Primary copy: [v4_tiny/code/data_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py>).

Data for the v4 tiny experiment (CPU). Text: documents re-extracted from v3's packed shards (run_m2_25M/data/{train,val}, read-only), stored once as a flat token array + offsets (v4_tiny/data/). Rows are re-packed per k: every document padded to a multiple of k, rows of T tokens, the SAME document order for every k (so arms see the same documents; padding differs, as it would at full size). Probes: v3's ProbeFactory families recall1 / carry3 / multi4 with gaps of 80-200 tokens (beyond the decoder's reach of 70, so only the concept route can answer), fillers = real sentences from the training documents. Items are packed with k-token padding, o

Imports:

```python
import os
import random
import numpy as np
import torch
from tokenizers import Tokenizer
from hrm_encoder import engram_indices
from probes import ProbeFactory, VALUE_CANDIDATES
```

Symbols:

- [extract_docs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:27>) — lines 27–40: Real documents (pads stripped) from a random subset of v3's packed rows.
- [build_store](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:43>) — lines 43–50
- [TextStore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:53>) — lines 53–60
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:54>), [doc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:59>).
- [token_freq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:63>) — lines 63–64
- [pack_sequences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:68>) — lines 68–91: Pack token sequences (each its own document) into rows of T: each padded to a multiple of k, a new row when it does not fit; the row tail is PAD with a fresh doc id. Sequences longer than T are cut into T-sized pieces. Returns tok (R, T), d
- [TinyProbes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:95>) — lines 95–110: v3's probe families with fillers taken from real sentences and gaps of 80-200 tokens ('far': 200-300).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:97>), [gap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:106>), [item](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:109>).
- [filler_sentences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:113>) — lines 113–124: Sentences (5-40 tokens) cut at '.' from real training documents.
- [Canon](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:128>) — lines 128–133
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:129>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:132>).
- [to_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:136>) — lines 136–138
- [MixedStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:141>) — lines 141–185: Training rows: with probability p_syn a row of packed probe items, else a row of packed text documents. The row-type sequence, the document order and the item sequence depend only on the seed, not on k.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:144>), [_text_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:154>), [_syn_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:163>), [next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:180>).
- [val_text_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:188>) — lines 188–196
- [probe_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:199>) — lines 199–201
- [item_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/data_v4.py:204>) — lines 204–206

## v4_tiny/code/decomp_v4tiny.py

144 lines; SHA-256 `4558ccce4f1f64ed54557f76aad04704b4ef70ec86495780f238f86f0bd389f5`.

Primary copy: [v4_tiny/code/decomp_v4tiny.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py>).

Post-hoc analysis of a finished arm (consult #14): (1) pooler weights split by slot, so position and content effects separate; (2) paired standard errors for the switch-off deltas (per validation row). python decomp_v4tiny.py <arm> \[<arm> ...\] -> runs/<arm>/decomp.json

Imports:

```python
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import model_v4 as M4
import data_v4 as D
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py:20>) — lines 20–23
- [per_row_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py:27>) — lines 27–36
- [alphas](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py:40>) — lines 40–46
- [slot_split](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py:49>) — lines 49–67: per slot: mean alpha (averaged over heads) for tokens in class a vs class b, on FULL chunks only.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/decomp_v4tiny.py:70>) — lines 70–140

## v4_tiny/code/make_arch_v4.py

108 lines; SHA-256 `dccdf4b2fe8365290adde481a13ba391981333bff5e36c91c72048f511c3a0c3`.

Primary copy: [v4_tiny/code/make_arch_v4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_arch_v4.py>).

Write run_m2_25M/webapp/data/arch_v4.json: the draft_v4 architecture with exact counts from model_v4 (full size). Same key shape as data/arch_v3d.json so the dashboard's v4 pages follow the v3 pages. Run on CPU: python make_arch_v4.py

Imports:

```python
import json
import os
import torch
import model_v4 as M4
```

Symbols:

- [counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_arch_v4.py:16>) — lines 16–41
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_arch_v4.py:44>) — lines 44–104

## v4_tiny/code/make_v4tiny_json.py

83 lines; SHA-256 `7caeb1a16c9c4f46d6b785bdd58758e2b54357a68cfa86efb42eaba3d7796ce0`.

Primary copy: [v4_tiny/code/make_v4tiny_json.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py>).

Compile every v4_tiny result into run_m2_25M/webapp/data/v4_tiny.json for the dashboard's "Tiny experiment" tab. Stdlib only; reads v4_tiny/runs/* (logs, final.json, decomp*.json, recall_check.json, reads_check.json, test logs).

Imports:

```python
import glob
import json
import os
import re
```

Symbols:

- [jl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py:17>) — lines 17–24
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py:27>) — lines 27–28
- [arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py:31>) — lines 31–55
- [passes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py:58>) — lines 58–61
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/make_v4tiny_json.py:64>) — lines 64–79

## v4_tiny/code/run_v4tiny.py

303 lines; SHA-256 `3edd2f66791023242701b427497370e5f33d32058d9ab723258829ba6b858bf2`.

Primary copy: [v4_tiny/code/run_v4tiny.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py>).

Train and evaluate one tiny v4 arm on CPU (v4_tiny experiment). Never touches the v3 run or the GPU. python run_v4tiny.py --name k6_attnffn_s0 --k 6 --pool attnffn --seed 0 --real_tokens 12e6 Writes v4_tiny/runs/<name>/{log.jsonl, final.json, ckpt.pt}.

Imports:

```python
import argparse
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import model_v4 as M4
import data_v4 as D
```

Symbols:

- [get_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:24>) — lines 24–48
- [val_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:53>) — lines 53–64
- [item_acc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:68>) — lines 68–94: Teacher-forced greedy exact match: every answer token of every span is the argmax (prediction for token i is logits\[i-1\]). plan_swap rolls plans across rows within a forward chunk, so chunks hold >= 2 rows.
- [codebook_health](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:98>) — lines 98–115
- [pool_telemetry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:119>) — lines 119–175: Pooler weights vs the uniform baseline (1 / real tokens in the chunk): text: mean (alpha / uniform) on the 100 most frequent tokens vs the rest, per head; per-slot profile. probes: mean (alpha / uniform) on fact content (names, values) vs f
- [full_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:178>) — lines 178–192
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_v4tiny.py:196>) — lines 196–299

CLI arguments:

```python
ap.add_argument("--name", required=True)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--pool", default="attnffn", choices=["attnffn", "attn", "linear", "gconv"])
ap.add_argument("--enc_loop", type=int, default=2)
ap.add_argument("--M", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--T", type=int, default=480)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--p_syn", type=float, default=0.5)
ap.add_argument("--real_tokens", type=float, default=12e6, help="stop after this many real (non-pad) input tokens")
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--eval_every_tokens", type=float, default=1.5e6)
ap.add_argument("--n_probe", type=int, default=128, help="items per family in periodic evals")
ap.add_argument("--n_probe_final", type=int, default=192)
ap.add_argument("--restart_every", type=int, default=200)
ap.add_argument("--block", type=int, default=60, help="block-local window attention (0 = full-mask SDPA)")
ap.add_argument("--over", default="", help="extra model config overrides, e.g. d=96,enc_ff=384")
```

## v4_tiny/code/run_vs.py

384 lines; SHA-256 `8a1e847548e741982ad62dcc1c63948ee538bd18f81a23c39bfc788d697dfa67`.

Primary copy: [v4_tiny/code/run_vs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py>).

v4 vs LLM-stack tiny experiment (CPU): train and evaluate one arm. python run_vs.py --name vs_text_v4_s0 --arch v4 --data text --seed 0 --real_tokens 5e6 python run_vs.py --name vs_pin_llm_s1 --arch llm --data pinpad --seed 1 --real_tokens 3e6 Writes v4_tiny/runs/<name>/{log.jsonl, final.json, ckpt.pt}. Held equal across arch: training rows (same stream, same seed, rows padded per v4's k = 6 for both), tokenizer, tied embedding, NTP target mask, optimizer (AdamW 0.9/0.95, wd 0.1 by v4's decay rule), lr schedule (warm-up 400, cosine to 0.1 of peak over the token budget), B, T, clip, token budget, threads. v4 adds its own concept terms (0.05 CE

Imports:

```python
import argparse
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import model_llm as ML
import data_v4 as D
import pinpad as PP
```

Symbols:

- [get_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:35>) — lines 35–57
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:61>) — lines 61–67
- [is_v4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:70>) — lines 70–71
- [fwd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:74>) — lines 74–78
- [loss_fn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:81>) — lines 81–82
- [ptable](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:85>) — lines 85–87
- [interventions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:90>) — lines 90–93
- [flops_per_token](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:96>) — lines 96–121: Forward multiply-adds per token (x2 = FLOPs), matmuls + attention scores/values; v4's core at T / k positions.
- [val_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:126>) — lines 126–137
- [item_acc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:141>) — lines 141–165: Teacher-forced greedy exact match of every answer span (prediction for token i is logits\[i-1\]).
- [item_ntp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:169>) — lines 169–171
- [pin_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:175>) — lines 175–189: Pack episodes; per-row label arrays aligned with token positions.
- [pin_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:193>) — lines 193–229: NTP split by token kind; action optimality (argmax over the 4 actions in the expert's optimal set), split by boundary flag; position exact accuracy.
- [pin_eval_sets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:232>) — lines 232–234
- [text_setup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:238>) — lines 238–250
- [pin_setup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:253>) — lines 253–254
- [periodic_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:257>) — lines 257–267
- [final_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:270>) — lines 270–286
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code/run_vs.py:290>) — lines 290–380

CLI arguments:

```python
ap.add_argument("--name", required=True)
ap.add_argument("--arch", choices=["v4", "llm"], required=True)
ap.add_argument("--data", choices=["text", "pinpad"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--T", type=int, default=480)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--p_syn", type=float, default=0.5)
ap.add_argument("--real_tokens", type=float, default=5e6)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--eval_every_tokens", type=float, default=1e6)
ap.add_argument("--n_probe", type=int, default=128)
ap.add_argument("--n_probe_final", type=int, default=192)
ap.add_argument("--restart_every", type=int, default=200)
ap.add_argument("--block", type=int, default=60)
ap.add_argument("--over", default="", help="model overrides, e.g. ff=296")
```

## v5_tiny/code/bench2_v5.py

27 lines; SHA-256 `56db6886d98b8b26844bc109f747712b960da84394f7033f3674b3cf9c04fd2c`.

Primary copy: [v5_tiny/code/bench2_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench2_v5.py>).

Imports:

```python
import time, torch
from model_v5 import *
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench2_v5.py:3>) — lines 3–23

## v5_tiny/code/bench_cpu_v5.py

13 lines; SHA-256 `3437acbd75a83e015f39135f6dc5185051355ff0576b9d199bbd0c1441f92c54`.

Primary copy: [v5_tiny/code/bench_cpu_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench_cpu_v5.py>).

Imports:

```python
import time, torch, sys
from model_v5 import *
```

Symbols:

- [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench_cpu_v5.py:7>) — lines 7–8

## v5_tiny/code/bench_v5.py

27 lines; SHA-256 `976692415fdfc2386e11a146770d136434f939a67feb5f11033bd5b7d3f91282`.

Primary copy: [v5_tiny/code/bench_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench_v5.py>).

Imports:

```python
import time, torch, sys
from model_v5 import *
```

Symbols:

- [bench](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/bench_v5.py:3>) — lines 3–22

## v5_tiny/code/build_tables.py

143 lines; SHA-256 `fa1c81fa6d5f8ed96b3825da0b8c38b66b01313e8b7ebffb8857a32dbe92784e`.

Primary copy: [v5_tiny/code/build_tables.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py>).

Collect every result JSON into RESULT_TABLES.md (numbers only; interpretation lives in REPORT.md).

Imports:

```python
import json
import os
import glob
```

Symbols:

- [J](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py:11>) — lines 11–15
- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py:18>) — lines 18–19
- [pm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py:22>) — lines 22–25
- [fix_oracle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py:31>) — lines 31–40: Evaluations run before 15:33 omitted token positions 0..2 from oracle/mixture window sums (text only); add the exact correction (teacher NLL of those positions per row / scored tokens; identical for every mode).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/build_tables.py:43>) — lines 43–139

## v5_tiny/code/common_v5.py

156 lines; SHA-256 `5ac3989ae64bca198d5e10f2478b79ea60a36b500a5097c8165dd4580dc2f109`.

Primary copy: [v5_tiny/code/common_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py>).

Shared harness pieces: mixed batches, Phase A evaluation, GraphNav free generation.

Imports:

```python
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import (NCPModel, TransformerLM, token_nll, concept_mask, ncp_mse, apply_plan_mode, shift_to_tokens,
                      K, D)
```

Symbols:

- [maybe_compile](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:19>) — lines 19–21: torch.compile unless V5_NOCOMPILE is set (compiled CPU training fails here with an OpenMP error).
- [Mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:24>) — lines 24–35
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:25>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:29>).
- [domain_mean](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:38>) — lines 38–44: Pre-registered reduction: per-token mean within each domain (rows 0..B_TEXT-1 text, rest graph), 0.5/0.5.
- [domain_mean_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:47>) — lines 47–51: Same 0.5/0.5 domain reduction for per-concept quantities. per_row_vals,w \[B,M'\]
- [cosine_lr](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:54>) — lines 54–58
- [make_opt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:61>) — lines 61–66
- [eval_sets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:73>) — lines 73–81
- [text_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:84>) — lines 84–86
- [eval_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:90>) — lines 90–111: fwd(x) -> logits. Returns text NLL/token, graph walk NLL/token, colour-token NLL, node-token NLL.
- [generate_walks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:116>) — lines 116–141: Free generation from the GraphNav header. step_logits(x \[B,T\], pos) -> logits at position pos \[B,V\]. Fixed T=192 inputs (pads after the cursor are invisible under the causal mask). Returns success flags etc.
- [save_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:144>) — lines 144–147
- [Logger](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:150>) — lines 150–156
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:151>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/common_v5.py:154>).

## v5_tiny/code/data_v5.py

266 lines; SHA-256 `3ba29d10d10af24a0122532696f7ba3bc6fb23749fbcc4bd15a5c243562dd949`.

Primary copy: [v5_tiny/code/data_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py>).

Data for v5_tiny: TinyStories text + GraphNav symbolic task, one 2048-entry tokenizer. Text: TinyStories (local HF cache), ByteLevel BPE with 2048 entries, of which the first NSPEC are reserved specials (pad, eos, GraphNav markers, node and colour tokens). Stories are concatenated with <eos> into one flat uint16 stream; training rows are random T-token windows (standard packed LM training). GraphNav (the symbolic task). One instance = one row of T=192 tokens, every decision exactly k=4 tokens: chunk 0 \[<gnav>, goal, start, <sep>\] chunks 1..30 \[v, colour, f_colour(v), <sep>\] one per (node, colour), shuffled; N=10 nodes, 3 colours hops \[co

Imports:

```python
import os
import glob
import numpy as np
```

Symbols:

- [build_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:49>) — lines 49–83
- [TextStream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:86>) — lines 86–100
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:87>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:90>), [fixed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:96>).
- [gen_graph](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:104>) — lines 104–111
- [gen_rules](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:114>) — lines 114–122
- [gen_instance](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:125>) — lines 125–161: Returns tokens \[T\], loss mask \[T\], meta dict. Node indices are instance-local 0..N-1.
- [gen_hardgoal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:164>) — lines 164–192: Hard-goal OOD instance (pre-registered 11:20): same graph/header format, goal = uniformly random node at shortest-path distance >= min_dist from start (any colours), so reaching it is not guaranteed by rule-structured walks. Only the heade
- [hardgoal_set](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:195>) — lines 195–198
- [graph_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:201>) — lines 201–203
- [parse_header](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:206>) — lines 206–214: Recover (label->local index, f, start, goal) from the header tokens alone (what a deployed verifier sees).
- [verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:217>) — lines 217–247: Exact verifier. x_header: the first WALK_START tokens. hop_tokens: generated tokens after the header. Success iff every hop chunk is \[colour, <to>, f_colour(cur), <sep>\], a <done> chunk comes, the walk is at the goal then, and at most MAX_H
- [fixed_graph_set](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/data_v5.py:250>) — lines 250–253

## v5_tiny/code/dbg_compile.py

27 lines; SHA-256 `795b995f0fe12b1a8cd31e17a768008dbc28b920a55ac9a8d0e0d753e20d3d24`.

Primary copy: [v5_tiny/code/dbg_compile.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/dbg_compile.py>).

Imports:

```python
import torch, numpy as np
import data_v5 as dv
from model_v5 import *
from common_v5 import Mixer, domain_mean
```

Symbols:

- [cmp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/dbg_compile.py:9>) — lines 9–15

## v5_tiny/code/diag_lookup.py

34 lines; SHA-256 `2fc87f7844c3fefc8dbc724f2480b51b630ec33da8983b1323681ebfc16a799d`.

Primary copy: [v5_tiny/code/diag_lookup.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/diag_lookup.py>).

Node-lookup diagnostic (advisor consult 8): graph-only 4-layer TransformerLM; node NLL every 250 steps.

Imports:

```python
import os, sys, json, time, numpy as np, torch
import data_v5 as dv
from model_v5 import TransformerLM, token_nll
```

## v5_tiny/code/drive_v5.py

117 lines; SHA-256 `fe36f2ed998f08a15ade94d662aa95606323fab318b4920ed411813b608c3722`.

Primary copy: [v5_tiny/code/drive_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/drive_v5.py>).

Unattended driver for Phases B-E. Each stage is idempotent: it skips work whose output file already exists. usage: python drive_v5.py --stage b --steps_b 3500 python drive_v5.py --stage b_eval | c | c_eval | d | e

Imports:

```python
import argparse
import os
import subprocess
import sys
import time
```

Symbols:

- [run_parallel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/drive_v5.py:22>) — lines 22–45: jobs: list of (name, argv, done_file). Runs those not done, at most `limit` at a time (MPS contention is superlinear: 2 concurrent jobs ran at 0.6 s/step each vs 0.17 solo), waits for all.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/drive_v5.py:48>) — lines 48–113

CLI arguments:

```python
ap.add_argument("--stage", required=True)
ap.add_argument("--steps_b", type=int, default=3500)
ap.add_argument("--steps_c", type=int, default=1500)
ap.add_argument("--teacher", default="A_ncp_s0")
```

## v5_tiny/code/eval_a.py

207 lines; SHA-256 `f2d5d39a9d7c273902171f9488f0da84ab7826702f32bdbf8bb555553e31a0a7`.

Primary copy: [v5_tiny/code/eval_a.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py>).

Phase A evaluation (pre-registered measurements). usage: python eval_a.py --run A_ncp_s0 \[--split val\]

Imports:

```python
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import NCPModel, TransformerLM, token_nll, concept_mask, rms, K, D, S_PQ, V_PQ
from common_v5 import maybe_compile, DEV, RUNS, Mixer, B_TEXT, eval_sets, text_mask, generate_walks, save_json
```

Symbols:

- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:13>) — lines 13–19
- [plan_stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:23>) — lines 23–36: Training-data means of chat and A per domain (pre-registered: 50 train batches, Mixer seed 777). Graph means use only positions whose target chunk is scored (walk/done).
- [derangement](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:39>) — lines 39–43
- [switch_masks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:47>) — lines 47–55: \[N,T-1\] masks over input positions predicting colour tokens at switch hops / non-switch hops (h >= 1).
- [per_item](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:58>) — lines 58–72: Per-row (NLL sum, token count) arrays, plus colour/node sums for graph rows.
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:75>) — lines 75–80: Mean per-token delta (b - a) with SE over items (per-item token-weighted means).
- [predicted_history_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:84>) — lines 84–95: NCP inference protocol: the concept module consumes previously predicted concepts (position 0 = actual c_0).
- [forecast_baselines](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:99>) — lines 99–126: MSE of chat_j vs c_{j+1} on val (scored targets) against copy-last, train-mean and a ridge linear predictor.
- [codebook_usage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:130>) — lines 130–140
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_a.py:143>) — lines 143–203

CLI arguments:

```python
ap.add_argument("--run", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--n_gen", type=int, default=512)
```

## v5_tiny/code/eval_b.py

351 lines; SHA-256 `45452b814340f73a3b253ee38b1c5c8cd8a0f67e5b3f2145c46eca171ab35bb0`.

Primary copy: [v5_tiny/code/eval_b.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py>).

Phase B/C evaluation of a replacement core. usage: python eval_b.py --run B_v6_s0 \[--lam 1.0\] \[--tag x\] NLL modes (z/gram variants): posterior (diagnostic), prior-sample x4 (deployable gate), prior-mean, oracle best-of-8 per window (diagnostic). Control tests on GraphNav (advisor consult 5 design): hold vs resample, Z-swap, eps-only, with pre-drawn noise and greedy tokens.

Imports:

```python
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import GramCore, token_nll, concept_mask, random_windows, K, D
from common_v5 import maybe_compile, DEV, RUNS, eval_sets, text_mask, save_json
from system_v5 import System, load_ncp, window_tensors, chunk_sums
from eval_a import per_item, paired, switch_masks
from common_v5 import Mixer, B_TEXT
```

Symbols:

- [load_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:25>) — lines 25–33
- [starts_hold](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:36>) — lines 36–38
- [starts_hold4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:41>) — lines 41–45: In-distribution schedule: a new window every 4 chunks from the first walk boundary (B trains lengths 1-4).
- [starts_every](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:48>) — lines 48–49
- [make_logits_fn](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:52>) — lines 52–66: f(x, starts, eta_z, eta_eps, z_mode, eps_mode) -> logits \[B,T,V\]. z_mode: 'sample' | 'mean'.
- [noise](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:69>) — lines 69–71
- [nll_modes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:75>) — lines 75–134: Per-item results for each mode. Windows: fixed seeded segmentation (lengths 1-4).
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:139>) — lines 139–158: Free generation from the header with a fixed noise draw per instance. Returns tokens \[N,T\] (numpy).
- [hop_colours](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:161>) — lines 161–168
- [walk_stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:171>) — lines 171–183
- [first_colour](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:186>) — lines 186–187
- [control_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:190>) — lines 190–228: P-control (pre-registered) + Z-swap / eps-only effects. Greedy tokens, pre-drawn noise.
- [route_split](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:232>) — lines 232–276: Replacement-route authority (deterministic path: prior-mean Z, eps mean): replace chat (pq_mean) or A (crc_mean) or both (mean) by the replacement's own training-data means per domain (20 train batches, Mixer seed 777).
- [hardgoal_success](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:280>) — lines 280–291: Hard-goal success with hold4 prior samples and with the prior mean (per-instance flags saved for pairing).
- [raw_norms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:295>) — lines 295–307
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/eval_b.py:310>) — lines 310–347

CLI arguments:

```python
ap.add_argument("--run", required=True)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--split", default="val")
ap.add_argument("--tag", default="")
ap.add_argument("--n_gen", type=int, default=512)
ap.add_argument("--no_control", action="store_true")
```

## v5_tiny/code/model_v5.py

393 lines; SHA-256 `22487921ecfe602e11a437dfe888ba9e31bcd7f802fc08b07951e9af9d683026`.

Primary copy: [v5_tiny/code/model_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py>).

v5_tiny models (d=128, ~2-3M parameters, built for throughput). Block: pre-RMSNorm, 4-head causal attention with RoPE (fused scaled_dot_product_attention), SwiGLU FFN (384 hidden). NCPModel (Phase A, reduced NCP \[R1\]): encoder (3 blocks over T tokens) -> H pool: c_j = RMS(mean of H over chunk j) (no-affine RMS normalisation of the pooled vector; declared) concept module (2 blocks over M=T/4 concepts, causal) -> u_j (u_j has seen c_0..c_j and predicts c_{j+1}) PQ heads: S=8 segments x V=32 codes x 16 dims; pi = softmax(W u); chat_j = sum_v pi_v Q_v (soft expectation) one CRC port: A_j = W_A RMS(u_j) (IRC omitted; CRC reduced to one concept->

Imports:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Symbols:

- [rope_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:34>) — lines 34–38
- [rope_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:44>) — lines 44–51
- [apply_rope](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:54>) — lines 54–59
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:62>) — lines 62–80
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:63>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:73>).
- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:83>) — lines 83–84
- [shift_to_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:87>) — lines 87–90: v \[B,M,d\] (index j = prediction made at boundary j for chunk j+1) -> \[B,T,d\] with b_t = v_{(t-3)//4}, t>=3.
- [TransformerLM](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:93>) — lines 93–107
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:94>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:102>).
- [_init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:110>) — lines 110–116
- [NCPModel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:119>) — lines 119–220
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:120>), [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:138>), [pool](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:146>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:150>), [pq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:157>), [port](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:164>), [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:167>), [vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:176>), [count_usage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:187>), [vq_init_or_restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:193>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:212>).
- [apply_plan_mode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:223>) — lines 223–237: Route-authority interventions (eval only). 'mean' uses training-mean vectors per domain (not a null).
- [token_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:241>) — lines 241–245: Per-position NLL \[B,T-1\] for predicting x\[:,1:\] from positions 0..T-2, masked by m\[:, :T-1\].
- [concept_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:248>) — lines 248–253: cm\[j\]=1 if the target chunk j+1 has a scored token: inputs 4j+3..4j+6 (j = 0..M-2).
- [ncp_mse](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:256>) — lines 256–259: Per-feature MSE between chat_j and detached c_{j+1}, averaged over scored targets. Returns (loss, per-row).
- [MLP2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:263>) — lines 263–272
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:264>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:270>).
- [GramCore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:275>) — lines 275–370: variant: 'det' (V2: no Z, no eps), 'adapter' (V2a: one non-recursive block, no Z/eps), 'z' (V3: Z + posterior/prior, no eps), 'gram' (V4/V6: Z + eps).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:279>), [z_prior](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:316>), [z_post](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:321>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:333>).
- [gauss_kl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:373>) — lines 373–375: KL(N(mq, e^{2lsq}) || N(mp, e^{2lsp})) elementwise.
- [random_windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:378>) — lines 378–389: Window id per position, lengths uniform in {lo..hi}. Returns win \[B,M\] long, starts \[B,M\] bool.
- [count_params](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/model_v5.py:392>) — lines 392–393

## v5_tiny/code/phase_a.py

109 lines; SHA-256 `0fb8ccf93e0ac1aa848c76c9dd5d0e3e3f2972970bef409eaea9128b57529713`.

Primary copy: [v5_tiny/code/phase_a.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_a.py>).

Phase A: train the reduced NCP foundation (or the V0 transformer) on the TinyStories + GraphNav mix. usage: python phase_a.py --arch ncp|v0 --seed 0 --steps 18000 --out <name>

Imports:

```python
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import NCPModel, TransformerLM, token_nll, concept_mask, ncp_mse, count_params
from common_v5 import (DEV, RUNS, Mixer, domain_mean, domain_mean_rows, cosine_lr, make_opt, eval_nll, Logger,
                       save_json)
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_a.py:20>) — lines 20–105

CLI arguments:

```python
ap.add_argument("--arch", default="ncp")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=18000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--out", required=True)
ap.add_argument("--eval_every", type=int, default=2000)
ap.add_argument("--no_compile", action="store_true")
ap.add_argument("--alpha", type=float, default=ALPHA, help="NCP-MSE weight (slack ablation: 0)")
```

## v5_tiny/code/phase_b.py

171 lines; SHA-256 `e0ee45dc4eff9115b37f2ae93729e168289791f5f65cc56ddefb0271d281c245`.

Primary copy: [v5_tiny/code/phase_b.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_b.py>).

Phase B: train a replacement concept core against the frozen NCP encoder / PQ heads / codebook / decoder. variants: det (V2), adapter (V2a), z (V3), gram (V4 = --xm 0, V6 = --xm 1), xmonly (V5: Z+eps prior only, XM loss). usage: python phase_b.py --variant gram --xm 1 --teacher A_ncp_s0 --steps 6000 --out B_v6_s0

Imports:

```python
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import GramCore, token_nll, concept_mask, random_windows, gauss_kl, count_params, K, D
from common_v5 import DEV, RUNS, Mixer, domain_mean, domain_mean_rows, cosine_lr, make_opt, Logger, save_json, B_TEXT
from system_v5 import System, load_ncp, window_tensors, chunk_sums
```

Symbols:

- [kl_per_token](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_b.py:24>) — lines 24–27: kl_rows \[B\] summed KL per row, ntok_rows \[B\] scored tokens -> domain-balanced per-token KL.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_b.py:30>) — lines 30–167

CLI arguments:

```python
ap.add_argument("--variant", default="gram")
ap.add_argument("--xm", type=int, default=1)
ap.add_argument("--K", type=int, default=2)
ap.add_argument("--teacher", default="A_ncp_s0")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--no_compile", action="store_true")
ap.add_argument("--bench", type=int, default=0)
ap.add_argument("--init", default="", help="Phase C: start from this run's core.pt")
ap.add_argument("--distill", type=int, default=1, help="0 disables the distillation term (Phase C)")
ap.add_argument("--lam_var", type=float, default=1.0, help="weight of the variational/reconstruction path")
ap.add_argument("--beta_z", type=float, default=BETA_Z, help="KL_Z weight (slack arm: 1.0 = exact-ELBO weighting)")
```

## v5_tiny/code/phase_d.py

359 lines; SHA-256 `f7e30ff909d18da4f02c54664df230af66a07654ecb3fbf8d03628a07d623bff`.

Primary copy: [v5_tiny/code/phase_d.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py>).

Phase D: discover persistent latent actions over the frozen transplanted system, then internal RL over Z. Stage 1 (discovery, ETA-style metacontroller \[R5\] adapted to our Z pathway): switch beta_m = sigmoid(MLP(\[e_m ; Z_{m-1}\])) causal proposal Zt_m ~ q_D(. | demonstration) (bidirectional GRU over \[e_m ; c_{m+1}\]) training only Z_m = (1 - beta_m) Z_{m-1} + beta_m Zt_m ; forced beta=1 at the first walk boundary (chunk 30) loss = walk NLL through the frozen core (prior guidance) + decoder + lam_D * sum_m KL(q_D(Zt_m) || p_ref(Zt_m | e_m)) (p_ref = Phase B prior; ungated, per review §7.3) Stage 2 (RL): Gaussian policy pi(Z | e_m, Z_prev) =

Imports:

```python
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import token_nll, concept_mask, gauss_kl, MLP2, K, D
from common_v5 import DEV, RUNS, Logger, save_json, eval_sets, cosine_lr, maybe_compile
from system_v5 import window_tensors
from eval_b import load_system, noise, hop_colours, walk_stats, starts_hold, starts_every
```

Symbols:

- [Meta](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:35>) — lines 35–49
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:36>), [beta](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:43>), [proposals](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:46>).
- [Policy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:52>) — lines 52–61
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:53>), [dist](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:59>).
- [z_sequence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:64>) — lines 64–84: Builds Z_pos \[B,M,dz\] for the walk boundaries by the soft/hard integration recursion. Positions < W0 use z_header (convention shared with eval_b/phase_e: one prior draw at position 0).
- [walk_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:87>) — lines 87–89
- [discovery](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:92>) — lines 92–129
- [discovery_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:133>) — lines 133–171: Soft vs hardened NLL (same proposals, proposal means) and switch alignment with true colour switches.
- [step_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:178>) — lines 178–195: Compiled pieces for rollouts: context (encode+pool+in_proj+prior) and core+decoder logits.
- [rollout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:199>) — lines 199–254: Generate walks; at each walk boundary m the controller decides Z_m. controller in {'policy','prior','prior_hold','always','fixed3'}; switch for 'policy'/'prior' = hardened beta. Returns tokens, and (if record) per-decision (e_m, z_prev, z,
- [rewards](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:257>) — lines 257–258
- [n_decisions_used](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:261>) — lines 261–263: Number of boundaries actually used per episode (hops + done).
- [rl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:266>) — lines 266–296
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_d.py:299>) — lines 299–355

CLI arguments:

```python
ap.add_argument("--core_run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps_d", type=int, default=1500)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr_d", type=float, default=1e-3)
ap.add_argument("--iters_rl", type=int, default=200)
ap.add_argument("--bs_rl", type=int, default=128)
ap.add_argument("--lr_rl", type=float, default=3e-3)
ap.add_argument("--n_eval", type=int, default=512)
ap.add_argument("--gate_kl", action="store_true",
                    help="post-hoc arm (16:02): KL_m weighted by the (non-detached) switch beta_m (review §7.3)")
```

## v5_tiny/code/phase_e.py

238 lines; SHA-256 `66774331f532831aefea4014576543d3fa66728719081be7611d2177be5d0698`.

Primary copy: [v5_tiny/code/phase_e.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py>).

Phase E: GradCuit-style test-time optimisation of the control offset dZ (review_v5 §8.3 corrected loop). One control window per episode (Z held over the walk from the first walk boundary), Z = Z0 + dZ, dZ = 0 at start. Weights frozen; tokens and reward detached; reward 0 (accepted) / -1 (rejected) from the header-only verifier. Loss on rejection: -R * sum_t log p(y_t | ...) = sum log p -> lowers the rejected walk's probability. Generate + verify after EVERY update (U updates -> at most U+1 generations). Baselines at the same number of generations: Z resampled from the prior; token sampling T=1; norm-matched random dZ. usage: python phase_e.py

Imports:

```python
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import token_nll, K, D
from common_v5 import maybe_compile, DEV, RUNS, eval_sets, save_json
from eval_b import load_system, noise, starts_hold4
```

Symbols:

- [make_fz](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:27>) — lines 27–40: fz(x, eta_z, delta, ee): Z = per-window prior (hold4 schedule, fixed draws eta_z) + one shared offset delta.
- [zpos_hold](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:43>) — lines 43–45: z \[B,dz\] -> single window; z \[B,M,dz\] (per-position, hold4 schedule) passes through.
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:49>) — lines 49–60
- [verify_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:63>) — lines 63–65
- [walk_loss_mask](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:68>) — lines 68–80: Score the generated walk: inputs WALK_START-1 .. through the end of the <done> chunk (or 16 hops).
- [base_controls](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:83>) — lines 83–88: Z0 = per-window prior samples on the hold4 schedule (windows of 4 chunks from the first walk boundary; in distribution for B's 1-4 chunk windows). Uses the header-only context at each window start: windows after the first read the context
- [gradcuit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:91>) — lines 91–132: Z0 here is the fixed eta_z draw \[B,M,dz\] defining the per-window prior controls. Returns accepted flags, attempts used, final |dZ| (per instance).
- [resample_baseline](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:135>) — lines 135–146: U+1 attempts, each with a fresh prior Z draw, same eps noise.
- [token_sampling_baseline](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:149>) — lines 149–157
- [summ](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:160>) — lines 160–162
- [pair](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:165>) — lines 165–167
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/phase_e.py:170>) — lines 170–234

CLI arguments:

```python
ap.add_argument("--core_run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--U", type=int, default=10)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--lrs", default="0.01,0.1,0.3")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--set", default="hardgoal", help="hardgoal (primary, pre-registered 11:12) | standard")
```

## v5_tiny/code/stats_graphnav.py

56 lines; SHA-256 `567e46c9c0be7bf3cf130ad78bad1201ac1b251d45b1280f3243a00e533e46b2`.

Primary copy: [v5_tiny/code/stats_graphnav.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/stats_graphnav.py>).

Measures the three designed properties of GraphNav (advisor consult 2).

Imports:

```python
import itertools, numpy as np, json, sys
from data_v5 import *
```

Symbols:

- [all_rule_seqs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/stats_graphnav.py:5>) — lines 5–12
- [first_arrival_ok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/stats_graphnav.py:15>) — lines 15–21
- [count_walks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/stats_graphnav.py:23>) — lines 23–34

## v5_tiny/code/system_v5.py

75 lines; SHA-256 `4ca9d3976093178074c0572d404dcad49a00aca58b2ebc98d730d62f2f36db65`.

Primary copy: [v5_tiny/code/system_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py>).

Frozen NCP skeleton + replacement core: forward modes shared by Phases B-E.

Imports:

```python
import torch
import torch.nn.functional as F
import data_v5 as dv
from model_v5 import NCPModel, GramCore, token_nll, K, D, apply_plan_mode
from common_v5 import DEV
```

Symbols:

- [load_ncp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:10>) — lines 10–17
- [window_tensors](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:20>) — lines 20–24
- [chunk_sums](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:27>) — lines 27–31: nll \[B,T-1\] over input positions -> \[B,M\]: chunk j collects positions 4j+3..4j+6 (those reading chat_j).
- [System](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:34>) — lines 34–75: teacher: frozen NCPModel. core: GramCore or None (None = original concept module).
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:37>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:42>), [z_prior_schedule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:47>), [plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:57>), [logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/system_v5.py:74>).

## v5_tiny/code/teacher_swapref.py

34 lines; SHA-256 `7bb67d5011d2a224d0162226c1bee9ad58cc4c1282517d13fe93a05605e91af6`.

Primary copy: [v5_tiny/code/teacher_swapref.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/teacher_swapref.py>).

Teacher-calibrated swap reference (pre-registered 15:00): NCP teacher at T=1, two seeds, 512 standard instances.

Imports:

```python
import json, os, numpy as np, torch
import data_v5 as dv
from common_v5 import RUNS, eval_sets, generate_walks, save_json, maybe_compile, DEV
from eval_a import load
from eval_b import hop_colours
```

Symbols:

- [gen](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/teacher_swapref.py:14>) — lines 14–24

## v5_tiny/code/test_v5.py

189 lines; SHA-256 `a7c10919a21c5999b09f0ce9f70521a47d5ba88a8d43029784ba24431c2e7b5a`.

Primary copy: [v5_tiny/code/test_v5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py>).

Pre-launch tests (advisor consult 3) + review_v5 §12 numeric checks. Run: python test_v5.py

Imports:

```python
import math
import numpy as np
import torch
import data_v5 as dv
from model_v5 import (NCPModel, GramCore, token_nll, concept_mask, ncp_mse, random_windows, shift_to_tokens,
                      apply_plan_mode, K)
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:14>) — lines 14–17
- [t_review_numeric](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:20>) — lines 20–43
- [t_graph_mask_and_verifier](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:46>) — lines 46–61
- [t_shift_leak](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:64>) — lines 64–89
- [t_gram_causal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:92>) — lines 92–111
- [t_windows_and_swap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:114>) — lines 114–122
- [t_compile_equiv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:125>) — lines 125–144
- [t_core_compile_equiv](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:147>) — lines 147–158
- [t_phase_d_zseq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/test_v5.py:161>) — lines 161–177

## v5_tiny/code/v2_bestof.py

21 lines; SHA-256 `3b03cea8cb6b0738d5c769429ca2b1777cc7b0c7f88e2235c25b32c037f24941`.

Primary copy: [v5_tiny/code/v2_bestof.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/v2_bestof.py>).

Equal-budget token-search baseline (pre-registered 14:50): the no-Z V2 core with T=1 token sampling, best-of-(U+1)=11 by the header-only verifier, on the hard-goal and standard sets.

Imports:

```python
import json, os, numpy as np, torch
import data_v5 as dv
from common_v5 import RUNS, eval_sets, save_json
from eval_b import load_system, make_logits_fn, generate, starts_every
```

## v5_tiny/code/v2_hardgoal_flags.py

9 lines; SHA-256 `bbda19943f7f068f2d81bdf9a4104efda194b970e49c06395ad79fa8ba210a2c`.

Primary copy: [v5_tiny/code/v2_hardgoal_flags.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/code/v2_hardgoal_flags.py>).

Per-instance hard-goal flags for the no-Z V2 core (for paired comparisons; pre-registered 15:10).

Imports:

```python
import os, json, numpy as np
import data_v5 as dv
from common_v5 import RUNS, save_json
from eval_b import load_system, make_logits_fn, hardgoal_success
```

## v6_tiny/code/anchor_gate.py

27 lines; SHA-256 `e7c12583aed1f8f29c5f8f2f1d7410ae76bcd68f2ae2ee6cea96a2b1a802df30`.

Primary copy: [v6_tiny/code/anchor_gate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/anchor_gate.py>).

Development competence screen before adapter training; confirmation remains separate.

Imports:

```python
import argparse
import numpy as np
import torch
from model import freeze
from runtime import load_system,atomic_json
from evaluate import likelihood
from generation import generate
from data import DATA
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/anchor_gate.py:12>) — lines 12–24

CLI arguments:

```python
ap.add_argument('--foundation',required=True)
ap.add_argument('--anchor',required=True)
ap.add_argument('--out',required=True)
```

## v6_tiny/code/audit.py

67 lines; SHA-256 `5396b21418a3520fbecde70e5ec3b99daadb99e909f20c1e479879a8e94279b4`.

Primary copy: [v6_tiny/code/audit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/audit.py>).

Post-training checks of frozen weights, latent authority and failure credit.

Imports:

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import freeze,chunk_info,DZ,safe_rms,ball
from data import fixed_graph,fixed_text,PAD,T,WALK_START,COL0,N_COL,NODE0,N_NODE_TOK,DONE,SEP,TO
from runtime import DEVICE,load_system,atomic_json
from generation import generate,expand_bank,replay_batch
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/audit.py:13>) — lines 13–64

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--anchor',required=True)
ap.add_argument('--out',required=True)
```

## v6_tiny/code/benchmark.py

72 lines; SHA-256 `a052a4ff1f332367e0e11a4f21946e562e57584587c9cecdf2cd44c05dfbe364`.

Primary copy: [v6_tiny/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/benchmark.py>).

Measure throughput and reject optimizations that change loss/gradients.

Imports:

```python
import argparse
import copy
import gc
import time
from pathlib import Path
import numpy as np
import torch
from model import Foundation,Core,Adapters,System,freeze,chunk_info,active_count
from data import Mixer,gen_instance
from train import tensors_forward,do_update
from runtime import DEVICE,ROOT,atomic_json,atomic_save,optimizer,sync,seed_all
from generation import generate
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/benchmark.py:16>) — lines 16–69

CLI arguments:

```python
ap.add_argument('--steps',type=int,default=20)
```

## v6_tiny/code/checks.py

229 lines; SHA-256 `aeeeeb7e683d87443d66b534deb11e6cb45570a524df9ecb51ef3775b3307ed2`.

Primary copy: [v6_tiny/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py>).

Executable V6 draft contracts. Skipped execution modes are labeled explicitly.

Imports:

```python
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import torch
from model import *
from data import Mixer,gen_instance,verify,oracle,WALK_START,T,PAD,DONE,SEP,fixed_text,DATA
from runtime import (candidate_backward,candidate_rows,selection_loss,controls,optimizer,save_checkpoint,load_system,
                     nll,balanced_sums,ncp_weights,atomic_json,DEVICE,sync)
from generation import Incremental,FastIncremental,generate,MaskedAdam,active_windows,expand_bank
from evaluate import repair_gradient,search_batch
from train import train_loss,tensors_forward,do_update
```

Symbols:

- [close](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:20>) — lines 20–21
- [make_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:24>) — lines 24–25
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:28>) — lines 28–28
- [check_counts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:31>) — lines 31–33
- [check_data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:36>) — lines 36–43
- [check_verifier](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:46>) — lines 46–55
- [check_causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:58>) — lines 58–63
- [check_isolation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:66>) — lines 66–70
- [check_anchor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:73>) — lines 73–79
- [check_control_gradients](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:82>) — lines 82–90
- [check_finite_difference](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:93>) — lines 93–101
- [check_selection](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:104>) — lines 104–119
- [check_moments](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:122>) — lines 122–126
- [check_masked_adam](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:129>) — lines 129–138
- [check_accumulation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:141>) — lines 141–150
- [check_resume](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:153>) — lines 153–163
- [check_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:166>) — lines 166–173
- [check_generation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:176>) — lines 176–191
- [check_compile](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:194>) — lines 194–206
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/checks.py:209>) — lines 209–226

CLI arguments:

```python
ap.add_argument('--optimized',action='store_true')
```

## v6_tiny/code/data.py

168 lines; SHA-256 `e992890adf37b33a99704e698bf8b230e9cb2279cf7ef5f18e081642ed5269ec`.

Primary copy: [v6_tiny/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py>).

Document-aware TinyStories packing and independent symbolic streams.

Imports:

```python
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import K, VOCAB
from graphnav import (PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_NODE_TOK,
                      T, WALK_START, MAX_HOPS, gen_instance, gen_hardgoal, parse_header)
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:16>) — lines 16–21
- [prepare](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:24>) — lines 24–62
- [Text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:65>) — lines 65–85
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:66>), [row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:72>).
- [Mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:88>) — lines 88–112
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:89>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:95>), [state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:108>), [restore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:111>).
- [fixed_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:115>) — lines 115–118
- [canonical](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:121>) — lines 121–124
- [fixed_graph](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:127>) — lines 127–137
- [verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:140>) — lines 140–157: Strict full-chunk terminator check. Never uses hidden trajectory labels.
- [oracle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/data.py:160>) — lines 160–168

## v6_tiny/code/drive.py

156 lines; SHA-256 `5beb151a476ee53ec539a97cfa79c49ca02dd7239086ee84ba549a06da0ea4f7`.

Primary copy: [v6_tiny/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/drive.py>).

Serial six-hour experiment with seven-hour ceiling and an evaluation reserve.

Imports:

```python
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from runtime import ROOT,atomic_json,Log,implementation_hash
```

Symbols:

- [now](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/drive.py:18>) — lines 18–18
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/drive.py:21>) — lines 21–153

CLI arguments:

```python
ap.add_argument('--hours',type=float,default=6)
ap.add_argument('--max-hours',type=float,default=7)
```

## v6_tiny/code/evaluate.py

271 lines; SHA-256 `05ed8a442405b75b53b5c057c9ef106bab0f22ef2abffbdf18cc45b36d558fea`.

Primary copy: [v6_tiny/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py>).

Per-item generation/search evidence, strict verifier and measured search costs.

Imports:

```python
import argparse
from collections import Counter
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import DZ,ball,safe_rms,freeze,chunk_info,active_count
from data import T,WALK_START,fixed_text,fixed_graph,oracle,canonical,PAD,verify
from runtime import DEVICE,load_system,atomic_json,Log,nll,sync,controls
from generation import generate,expand_bank,replay_batch,MaskedAdam,active_windows
```

Symbols:

- [repair_gradient](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:17>) — lines 17–38
- [search_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:41>) — lines 41–100
- [likelihood](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:104>) — lines 104–126
- [bootstrap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:129>) — lines 129–132
- [reachability](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:135>) — lines 135–152
- [evaluate_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:155>) — lines 155–237
- [text_samples](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:241>) — lines 241–260
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/evaluate.py:263>) — lines 263–268

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--split',default='dev')
ap.add_argument('--standard',type=int,default=128)
ap.add_argument('--hard',type=int,default=256)
ap.add_argument('--bs',type=int,default=16)
ap.add_argument('--quick',action='store_true')
ap.add_argument('--budget',type=float)
ap.add_argument('--deadline',type=float)
ap.add_argument('--methods')
```

## v6_tiny/code/generation.py

194 lines; SHA-256 `04393aa10ecdf53f20ad9498237bb066270d099293ce550417fc06e7a1427112`.

Primary copy: [v6_tiny/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py>).

Correct incremental caches; every recursive application owns a separate slot.

Imports:

```python
import time
import numpy as np
import torch
from model import D,K,DZ,chunk_info,shift,ball
from data import T,WALK_START,PAD,DONE,verify
from runtime import sync
```

Symbols:

- [expand_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:10>) — lines 10–13
- [Incremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:16>) — lines 16–63
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:17>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:19>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:45>).
- [FastIncremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:66>) — lines 66–109
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:67>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:93>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:101>).
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:113>) — lines 113–151
- [replay_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:154>) — lines 154–159
- [MaskedAdam](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:162>) — lines 162–183: Per-problem clock, frozen inactive moments, and bounded window writes.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:164>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:169>).
- [active_windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/generation.py:186>) — lines 186–194

## v6_tiny/code/graphnav.py

148 lines; SHA-256 `31afc5e2d07f676c03c6a8bd035d46aa853165e86d1023ccebf572716c72a9f9`.

Primary copy: [v6_tiny/code/graphnav.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py>).

GraphNav generator derived from v5_tiny/code/data_v5.py. The text pipeline and strict V6 verifier live in data.py.

Imports:

```python
import os
import glob
import numpy as np
```

Symbols:

- [gen_graph](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:30>) — lines 30–37
- [gen_rules](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:40>) — lines 40–48
- [gen_instance](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:51>) — lines 51–87: Returns tokens \[T\], loss mask \[T\], meta dict. Node indices are instance-local 0..N-1.
- [gen_hardgoal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:90>) — lines 90–118: Hard-goal OOD instance (pre-registered 11:20): same graph/header format, goal = uniformly random node at shortest-path distance >= min_dist from start (any colours), so reaching it is not guaranteed by rule-structured walks. Only the heade
- [hardgoal_set](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:121>) — lines 121–124
- [graph_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:127>) — lines 127–129
- [parse_header](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:132>) — lines 132–140: Recover (label->local index, f, start, goal) from the header tokens alone (what a deployed verifier sees).
- [fixed_graph_set](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/graphnav.py:144>) — lines 144–147

## v6_tiny/code/model.py

308 lines; SHA-256 `6edeb00be700f3c67de824d3b4a366bafcecf894587fa08c69621eaab016164f`.

Primary copy: [v6_tiny/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py>).

V6 tiny: document-isolated NCP, deterministic recurrence, zero-anchored Z. No imports from mutable V5 code. FP32 is the reference execution mode.

Imports:

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
```

Symbols:

- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:13>) — lines 13–14
- [safe_rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:17>) — lines 17–18
- [ball](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:21>) — lines 21–22
- [init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:25>) — lines 25–29
- [positions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:32>) — lines 32–38
- [rotate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:41>) — lines 41–47
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:50>) — lines 50–88
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:51>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:57>), [static_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:78>).
- [chunk_info](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:91>) — lines 91–94
- [shift](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:97>) — lines 97–103
- [Foundation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:106>) — lines 106–192
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:107>), [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:124>), [pool](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:131>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:134>), [pq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:139>), [port](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:143>), [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:146>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:152>), [vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:160>), [count_usage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:170>), [restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:176>).
- [Adapters](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:195>) — lines 195–231
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:196>), [raw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:212>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:217>), [calibrate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:225>).
- [Core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:234>) — lines 234–263
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:235>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:245>).
- [System](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:266>) — lines 266–290
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:267>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:271>), [plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:275>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:287>).
- [count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:293>) — lines 293–294
- [active_count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:297>) — lines 297–301
- [freeze](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/model.py:304>) — lines 304–308

## v6_tiny/code/report.py

65 lines; SHA-256 `4c6c288c61fd6e31416402ad88a096900d334403c4e0be829b2d30c1718ba64e`.

Primary copy: [v6_tiny/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/report.py>).

Build a readable progress/final report strictly from saved evidence.

Imports:

```python
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from runtime import ROOT,atomic_json
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/report.py:9>) — lines 9–10
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/report.py:13>) — lines 13–62

## v6_tiny/code/runtime.py

190 lines; SHA-256 `0fc4a94e52ee56af1bff635b3e9f79f1df610ca5027c056ba0121fb12f619424`.

Primary copy: [v6_tiny/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py>).

Atomic recovery, exact loss units and coherent candidate replay.

Imports:

```python
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import (Foundation, Core, Adapters, System, freeze, chunk_info, ball, K, DZ)
from data import WALK_START, T
```

Symbols:

- [sync](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:19>) — lines 19–20
- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:23>) — lines 23–27
- [atomic_save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:30>) — lines 30–34
- [Log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:37>) — lines 37–41
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:38>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:39>).
- [seed_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:44>) — lines 44–45
- [cpu_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:48>) — lines 48–53
- [save_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:56>) — lines 56–71
- [load_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:74>) — lines 74–81
- [optimizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:84>) — lines 84–87
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:90>) — lines 90–94
- [nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:97>) — lines 97–99
- [balanced_sums](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:102>) — lines 102–106
- [ncp_weights](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:109>) — lines 109–114
- [controls](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:117>) — lines 117–136
- [candidate_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:139>) — lines 139–162
- [selection_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:165>) — lines 165–168
- [candidate_backward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:171>) — lines 171–184
- [implementation_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/runtime.py:187>) — lines 187–190

## v6_tiny/code/train.py

169 lines; SHA-256 `e97dfa0ae9787e464fbcf78bffb6518840d5f2823c21627cbd8f539b9a5df250`.

Primary copy: [v6_tiny/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py>).

Independent stage trainer. Every completed or time-limited stage is recoverable.

Imports:

```python
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import Foundation,Core,Adapters,System,count,active_count,freeze,chunk_info
from data import Mixer,DATA
from runtime import (DEVICE,ROOT,Log,atomic_json,seed_all,load_system,save_checkpoint,optimizer,lr_at,
                     nll,balanced_sums,ncp_weights,controls,candidate_backward,sync,implementation_hash)
```

Symbols:

- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py:14>) — lines 14–26
- [tensors_forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py:29>) — lines 29–42
- [train_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py:45>) — lines 45–65
- [do_update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py:68>) — lines 68–90
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/code/train.py:93>) — lines 93–166

CLI arguments:

```python
ap.add_argument('--stage',choices=['A','B0','B1','det_extra'],required=True)
ap.add_argument('--source')
ap.add_argument('--out',required=True)
ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--steps',type=int,required=True)
ap.add_argument('--seconds',type=float,default=20000)
ap.add_argument('--kind',default='gated')
ap.add_argument('--beta',type=float,default=.25)
ap.add_argument('--alpha',type=float,default=1)
ap.add_argument('--batch',type=int,default=64)
ap.add_argument('--eager',action='store_true')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--stop-after',type=int,help='Predeclared prefix control; preserves the original LR/restart horizon')
```

## v7_tiny/code/audit.py

62 lines; SHA-256 `dcc016511befee60aef68981e1c1f98fe0db2e62e783b9be29730f7c3aa49748`.

Primary copy: [v7_tiny/code/audit.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/audit.py>).

Trained endpoint, forecast, zero-anchor, routing and deployment audits.

Imports:

```python
import argparse,copy,time
import numpy as np
import torch
from model import System,freeze,shift,count
from runtime import *
from data import fixed_text,T,DATA
from generation import generate,Incremental
from evaluate import text_metrics
```

Symbols:

- [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/audit.py:11>) — lines 11–59

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--out',required=True)
```

## v7_tiny/code/benchmark.py

67 lines; SHA-256 `3f27177ab27faf9b68687b9ef0b15ac34b73333c1c743c86f5fc4f5ba02bb3f7`.

Primary copy: [v7_tiny/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/benchmark.py>).

Correctness-checked warm throughput calibration on the actual device.

Imports:

```python
import argparse,gc,json,time
import numpy as np
import torch
from model import Config,System,freeze
from data import Mixer,ROOT,T,gen_symbolic
from runtime import DEVICE,optimizer,sync,atomic_json,seed_all
from runtime import controls,candidate_backward
from train import configure,forward_function,train_loss,update
from generation import generate
```

Symbols:

- [bench](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/benchmark.py:12>) — lines 12–41
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/benchmark.py:43>) — lines 43–65

CLI arguments:

```python
ap.add_argument('--out',default=str(ROOT/'runs/calibration/benchmark.json'))
```

## v7_tiny/code/checks.py

177 lines; SHA-256 `b091d2d6df09074391594eb0e5a4bcea6095428cae8056c6b1977c57f13a2d66`.

Primary copy: [v7_tiny/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py>).

First-principles contracts before and after training; smoke quality is not claimed.

Imports:

```python
import argparse,copy,json,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from train import train_loss,forward_function,configure,update
from generation import *
from search import Node,attach,select,search_one
```

Symbols:

- [close](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:13>) — lines 13–13
- [tiny](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:14>) — lines 14–14
- [system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:15>) — lines 15–18
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:19>) — lines 19–20
- [parameters](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:22>) — lines 22–25
- [data_contract](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:27>) — lines 27–33
- [verifiers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:35>) — lines 35–42
- [causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:44>) — lines 44–51
- [boundaries_and_padding](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:53>) — lines 53–59
- [anchors](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:61>) — lines 61–70
- [gradients](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:72>) — lines 72–78
- [objective_ownership](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:80>) — lines 80–87
- [codebook_moments](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:89>) — lines 89–93
- [selection_and_ownership](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:95>) — lines 95–104
- [masked_optimizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:106>) — lines 106–110
- [accumulation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:112>) — lines 112–119
- [recovery](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:121>) — lines 121–128
- [caches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:130>) — lines 130–137
- [tree_backups](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:139>) — lines 139–143
- [generation_and_repair](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:145>) — lines 145–154
- [optimized](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:156>) — lines 156–161
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/checks.py:163>) — lines 163–175

CLI arguments:

```python
ap.add_argument('--optimized',action='store_true')
```

## v7_tiny/code/data.py

207 lines; SHA-256 `a97496c59147bee8904c4d51a784b7373916d335777c26305039ab8f915a7de1`.

Primary copy: [v7_tiny/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py>).

Boundary-correct text packing and harder, independently generated symbolic tasks.

Imports:

```python
import gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:11>) — lines 11–15
- [prepare_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:17>) — lines 17–56
- [parse_prompt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:58>) — lines 58–68
- [verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:70>) — lines 70–96
- [canonical](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:98>) — lines 98–100
- [shortest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:102>) — lines 102–108
- [gen_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:110>) — lines 110–146
- [prepare_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:148>) — lines 148–161
- [Text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:163>) — lines 163–178
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:164>), [row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:167>).
- [Mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:180>) — lines 180–200
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:181>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:184>), [state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:198>), [restore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:199>).
- [fixed_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/data.py:202>) — lines 202–204

## v7_tiny/code/drive.py

233 lines; SHA-256 `b6d04aed839274a0e5437fd74f45067040761b66b34e1ebe03a4e232c45256ac`.

Primary copy: [v7_tiny/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py>).

One serial, budget-limited experiment queue; monitors never own training jobs.

Imports:

```python
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from data import ROOT,DATA,T,K,digest
from runtime import atomic_json,Log,implementation_hash,DEVICE
from evaluate import evaluate,load_eval,search_eval,paired,weighted_nll_pair,symbolic_metrics
from generation import generate
from audit import audit
```

Symbols:

- [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:12>) — lines 12–12
- [release](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:13>) — lines 13–15
- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:17>) — lines 17–93
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:18>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:45>), [check_source](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:48>), [skip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:54>), [checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:55>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:56>), [eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:81>), [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:92>).
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:95>) — lines 95–220
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py:222>) — lines 222–231

CLI arguments:

```python
ap.add_argument('--smoke',action='store_true')
```

## v7_tiny/code/evaluate.py

124 lines; SHA-256 `f6460402517836feb0917604dfb8a8a7da6f34838bd4e57979f0064e7196634b`.

Primary copy: [v7_tiny/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py>).

Fixed development/test likelihood, strict generation, and charged search experiments.

Imports:

```python
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from model import freeze,ball
from data import ROOT,DATA,T,K,Text,canonical
from runtime import DEVICE,load_system,atomic_json,nll,sync,tensor_hash
from generation import generate
from search import search_one
```

Symbols:

- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:14>) — lines 14–20
- [weighted_nll_pair](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:22>) — lines 22–28
- [load_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:30>) — lines 30–32
- [text_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:35>) — lines 35–46
- [symbolic_metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:49>) — lines 49–75
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:77>) — lines 77–82
- [search_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:84>) — lines 84–118
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/evaluate.py:120>) — lines 120–122

CLI arguments:

```python
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--split',choices=['dev','test'],default='dev')
ap.add_argument('--n',type=int,default=128)
ap.add_argument('--text-n',type=int,default=256)
ap.add_argument('--batch',type=int,default=16)
ap.add_argument('--fast',action='store_true')
ap.add_argument('--suites',default=','.join(SUITES))
ap.add_argument('--latent-k',type=int,default=0)
```

## v7_tiny/code/fetch_text.py

32 lines; SHA-256 `991db45dc5b63b1c449865972ec83e3de3b294bf4e0b9e4d91117a644bccda54`.

Primary copy: [v7_tiny/code/fetch_text.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/fetch_text.py>).

Fetch a bounded fresh educational-text sample; no training is performed here.

Imports:

```python
import gzip,json,hashlib,time
from pathlib import Path
from datasets import load_dataset
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/fetch_text.py:8>) — lines 8–30

## v7_tiny/code/generation.py

161 lines; SHA-256 `369a49b57ce7dc9a4ad36779e3e5d8b5089169af2ca688e0b99e6c7ad7910fe6`.

Primary copy: [v7_tiny/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py>).

Real-token refresh, independent branch caches, and full differentiable replay.

Imports:

```python
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import chunks,shift,ball,norm
from data import T,K,PAD,DONE,EOS,verify
from runtime import sync,nll
```

Symbols:

- [expand_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:10>) — lines 10–12
- [Incremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:14>) — lines 14–60
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:15>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:16>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:39>).
- [FastIncremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:62>) — lines 62–92
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:63>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:79>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:85>).
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:95>) — lines 95–128
- [replay_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:130>) — lines 130–133
- [active_windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:135>) — lines 135–138
- [MaskedAdam](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:140>) — lines 140–151
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:141>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:143>).
- [repair_gradient](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/generation.py:153>) — lines 153–161

## v7_tiny/code/model.py

252 lines; SHA-256 `253661715e255fac66428496e886bd6da22dadc61ed54923cce2322b867fcf99`.

Primary copy: [v7_tiny/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py>).

V7 reference: causal distinct stacks, bounded IRC/CRC, soft PQ, exact-zero Z.

Imports:

```python
from dataclasses import dataclass,asdict
import math,copy
import torch
from torch import nn
from torch.nn import functional as F
```

Symbols:

- [training_attention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:9>) — lines 9–11
- [Config](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:14>) — lines 14–35
  Methods: [__post_init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:33>).
- [norm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:37>) — lines 37–37
- [scale](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:38>) — lines 38–38
- [ball](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:39>) — lines 39–39
- [freeze](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:40>) — lines 40–42
- [count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:43>) — lines 43–43
- [positions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:44>) — lines 44–48
- [chunks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:49>) — lines 49–51
- [shift](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:52>) — lines 52–57
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:59>) — lines 59–97
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:60>), [rotate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:69>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:74>), [static_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:89>).
- [IRC](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:99>) — lines 99–108
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:100>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:103>).
- [CRC](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:110>) — lines 110–116
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:111>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:114>).
- [Stack](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:118>) — lines 118–137
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:119>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:125>).
- [Adapters](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:139>) — lines 139–150
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:140>), [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:147>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:150>).
- [Backbone](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:152>) — lines 152–217
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:153>), [pool](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:166>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:167>), [pq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:171>), [plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:174>), [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:178>), [vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:194>), [record_usage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:202>), [restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:206>).
- [Flat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:219>) — lines 219–227
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:220>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:224>).
- [System](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:229>) — lines 229–252
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:230>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:233>), [from_context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:234>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:241>), [make_refit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:245>), [make_adapters](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:248>), [active_count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/model.py:250>).

## v7_tiny/code/reachability.py

23 lines; SHA-256 `cbd91980f0524be4232dd0a654333638f9a2265647d68839355236e725e85177`.

Primary copy: [v7_tiny/code/reachability.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/reachability.py>).

Privileged valid-target optimization: development diagnosis, never search credit.

Imports:

```python
import time
import numpy as np
import torch
from data import DATA,T,K
from runtime import DEVICE,nll,atomic_json,sync
from generation import MaskedAdam,expand_bank,active_windows,generate
```

Symbols:

- [privileged_reach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/reachability.py:9>) — lines 9–23

## v7_tiny/code/report.py

85 lines; SHA-256 `e8da4404ee5541d9c04853bee5cc212f6d8fcbab6f14469f143c43ebc22c864f`.

Primary copy: [v7_tiny/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/report.py>).

Evidence-only reporting; incomplete endpoints and quality gates remain explicit.

Imports:

```python
import json,time
from pathlib import Path
import numpy as np
from runtime import atomic_json
from evaluate import paired,weighted_nll_pair
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/report.py:8>) — lines 8–8
- [f](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/report.py:9>) — lines 9–9
- [build_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/report.py:10>) — lines 10–81

CLI arguments:

```python
ap.add_argument('--root',required=True)
ap.add_argument('--out',required=True)
```

## v7_tiny/code/runtime.py

127 lines; SHA-256 `58df604c28d7c3ab8bdd529b1cfc3619ba8ee9c3f161230bb461f7bfa60c96e7`.

Primary copy: [v7_tiny/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py>).

Explicit reductions, coherent schedules, ownership, and atomic recovery.

Imports:

```python
import hashlib,json,math,os,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from model import Config,System,Stack,Adapters,chunks,ball,freeze
from data import ROOT,DATA,T,K
```

Symbols:

- [sync](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:20>) — lines 20–21
- [seed_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:22>) — lines 22–22
- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:23>) — lines 23–26
- [atomic_save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:27>) — lines 27–30
- [Log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:31>) — lines 31–35
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:32>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:33>).
- [cpu_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:36>) — lines 36–40
- [finite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:41>) — lines 41–45
- [implementation_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:46>) — lines 46–49
- [tensor_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:50>) — lines 50–53
- [save_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:54>) — lines 54–61
- [load_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:62>) — lines 62–67
- [optimizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:68>) — lines 68–74
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:75>) — lines 75–77
- [alpha_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:78>) — lines 78–78
- [nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:79>) — lines 79–81
- [balanced](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:82>) — lines 82–85
- [ncp_weights](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:86>) — lines 86–89
- [controls](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:90>) — lines 90–101
- [repeated](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:102>) — lines 102–106
- [candidate_scores](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:107>) — lines 107–111
- [selection](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:112>) — lines 112–115
- [candidate_backward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/runtime.py:116>) — lines 116–127

## v7_tiny/code/search.py

111 lines; SHA-256 `b09b961389b8c6c5538dfd802be3060763118f88d25ef27cc9ab6b64a37e1b17`.

Primary copy: [v7_tiny/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py>).

AB-MCTS-A over realized answer/control refinements, with charged wall time.

Imports:

```python
import hashlib,time
from dataclasses import dataclass,field
import numpy as np
import torch
from model import ball
from data import T,K
from generation import generate,repair_gradient,active_windows,MaskedAdam
from runtime import sync
```

Symbols:

- [Stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:12>) — lines 12–17
  Methods: [add](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:14>), [draw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:17>).
- [Node](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:19>) — lines 19–28
  Methods: .
- [attach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:30>) — lines 30–33
- [eligible](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:35>) — lines 35–35
- [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:36>) — lines 36–43
- [control_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:45>) — lines 45–45
- [search_one](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/search.py:47>) — lines 47–111

## v7_tiny/code/train.py

143 lines; SHA-256 `402791ab1835ccbc0913f1aac9b33f4eebb35edab1deaeafd85d0570f5b9db35`.

Primary copy: [v7_tiny/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py>).

Target-token-budget trainer; all stages keep finite atomic recovery checkpoints.

Imports:

```python
import argparse,copy,json,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from model import Config,System,freeze,count,scale
from data import Mixer,DATA,ROOT
from runtime import *
```

Symbols:

- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py:11>) — lines 11–22
- [forward_function](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py:24>) — lines 24–28
- [train_loss](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py:30>) — lines 30–45
- [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py:47>) — lines 47–67
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/train.py:69>) — lines 69–134

CLI arguments:

```python
ap.add_argument('--stage',choices=['A','B0','B1','det_extra','full_extra'],required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--source')
ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--tokens',type=int,required=True)
ap.add_argument('--batch',type=int,default=32)
ap.add_argument('--seconds',type=float,default=86400)
ap.add_argument('--alpha',type=float,default=.1)
ap.add_argument('--beta',type=float,default=.25)
ap.add_argument('--lr',type=float,default=.001)
ap.add_argument('--flat',action='store_true')
ap.add_argument('--plain',action='store_true')
ap.add_argument('--constant-alpha',action='store_true')
ap.add_argument('--eager',action='store_true')
ap.add_argument('--replay',action='store_true')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--max-steps',type=int)
ap.add_argument('--mini',action='store_true')
```

## v8_tiny/code/benchmark.py

24 lines; SHA-256 `89869ee5b9e34b0bf63253e9493f253ff79525a6e5fa2ab08480a972827a87c2`.

Primary copy: [v8_tiny/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/benchmark.py>).

Calibration selects execution only after matching eager values AND parameter gradients.

Identical copies:

- [v8_tiny_12h/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/benchmark.py>)
- [v9_tiny/frozen/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/benchmark.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/benchmark.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/benchmark.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/benchmark.py>)

Imports:

```python
import copy,json,time
import torch
from model import *
from data import Mixer
from runtime import *
from train import objective,forward_function
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/benchmark.py:9>) — lines 9–23

## v8_tiny/code/bridge.py

23 lines; SHA-256 `eb76048d10960e31d802d2d9acde5fc7ed980757e6dd5de7f9a99bf781002949`.

Primary copy: [v8_tiny/code/bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/bridge.py>).

Historical checkpoint replay, not an independent replication or V8 training seed.

Identical copies:

- [v8_tiny_12h/code/bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/bridge.py>)
- [v9_tiny/frozen/code/bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/bridge.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/bridge.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/bridge.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/bridge.py>)

Imports:

```python
import json
import numpy as np
import torch
from runtime import *
from model import freeze
from generation import generate
from evaluate import wilson
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/bridge.py:10>) — lines 10–22

## v8_tiny/code/checks.py

57 lines; SHA-256 `182019234a8ccdbfcffc15d933a50792c28b4bf625581e402104dd38e9d4cc31`.

Primary copy: [v8_tiny/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/checks.py>).

Executable leakage, trained decoder, control, continual and gradient contracts.

Identical copies:

- [v8_tiny_12h/spec/recovery_2026-10-06/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/checks.py>)

Imports:

```python
import argparse,copy,json,traceback
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import Incremental,FastIncremental,generate,expand_bank,MaskedAdam,repair_gradient
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/checks.py:11>) — lines 11–53

CLI arguments:

```python
ap.add_argument('--source')
ap.add_argument('--out',required=True)
ap.add_argument('--fast',action='store_true')
```

## v8_tiny/code/continual.py

82 lines; SHA-256 `33e40bc91b1e2d534c1534aef444de4f9f6ae3717acec8dc4f436a946c046107`.

Primary copy: [v8_tiny/code/continual.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py>).

Frozen-reference local-support fitting, positive support, routing diagnostics.

Imports:

```python
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from model import *
from data import Mixer,DATA
from runtime import *
```

Symbols:

- [Reservoir](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:12>) — lines 12–15
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:13>), [add](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:14>).
- [fit_gmm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:18>) — lines 18–33: Diagonal EM with random-sample initial means; avoids conflicting macOS OpenMP runtimes. Three independent starts, full-reservoir likelihood, fixed50-iteration/tolerance budget.
- [fit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:35>) — lines 35–67
- [gate_rates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/continual.py:69>) — lines 69–80

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--positive',action='store_true')
ap.add_argument('--mode',default='foundation')
ap.add_argument('--tokens',type=int,default=1000000)
```

## v8_tiny/code/data.py

216 lines; SHA-256 `9ef8c94ed2d3cb65b78f03e1a08a0165bcb851e4f6da4b1a7a5d2302809e336e`.

Primary copy: [v8_tiny/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py>).

Boundary-correct text packing and harder, independently generated symbolic tasks.

Imports:

```python
import gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:11>) — lines 11–15
- [prepare_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:17>) — lines 17–43
- [parse_prompt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:45>) — lines 45–57
- [verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:59>) — lines 59–86
- [canonical](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:88>) — lines 88–90
- [shortest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:92>) — lines 92–98
- [gen_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:100>) — lines 100–146
- [prepare_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:148>) — lines 148–161
- [Text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:163>) — lines 163–177
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:164>), [row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:168>).
- [Mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:179>) — lines 179–209
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:180>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:183>), [state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:206>), [restore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:207>).
- [fixed_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/data.py:211>) — lines 211–213

## v8_tiny/code/drive.py

212 lines; SHA-256 `ce766601f42601e29234a355888ebb2d99ba6a599623552e583636c5f869e2dd`.

Primary copy: [v8_tiny/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py>).

Detached serial, competence-first V8 pilot; preserves exposure before arm breadth.

Identical copies:

- [v8_tiny_12h/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive.py>)
- [v9_tiny/frozen/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/drive.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive.py>)

Imports:

```python
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from runtime import *
from model import *
from data import DATA,Mixer,digest
from evaluate import evaluate,retention,symbolic_eval,text_eval,paired,cluster_ci
from continual import gate_rates
```

Symbols:

- [BudgetEnd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:12>) — lines 12–12
  Methods: .
- [GateEnd](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:13>) — lines 13–13
  Methods: .
- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:14>) — lines 14–202
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:15>), [stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:19>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:22>), [remaining](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:24>), [skip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:25>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:26>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:46>), [eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:72>), [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:74>), [inline](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:75>), [continual](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:76>), [finish](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:125>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:131>), [final_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/drive.py:193>).

CLI arguments:

```python
ap.add_argument('--hours',type=float,default=12)
```

## v8_tiny/code/evaluate.py

88 lines; SHA-256 `502fcd4964b10a231165068ea87e749ed1f2d36e06a590a4bb60ebe20c3c5362`.

Primary copy: [v8_tiny/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py>).

Public-header generation scoring, document-cluster uncertainty and absolute gates.

Imports:

```python
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
from model import *
from data import *
from runtime import *
from generation import generate
```

Symbols:

- [wilson](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:13>) — lines 13–17
- [exact](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:18>) — lines 18–18
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:19>) — lines 19–23
- [cluster_ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:24>) — lines 24–28
- [text_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:30>) — lines 30–35
- [symbolic_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:37>) — lines 37–56
- [interface_audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:59>) — lines 59–66
- [competence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:68>) — lines 68–69
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:71>) — lines 71–77
- [retention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/evaluate.py:79>) — lines 79–86

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--split',default='dev')
ap.add_argument('--names',default='standard,lookup,compose,hard10')
ap.add_argument('--n',type=int,default=512)
ap.add_argument('--fast',action='store_true')
```

## v8_tiny/code/generation.py

143 lines; SHA-256 `a6f019728bba1d9fcd48fd3002fe0ff86fae580419011a1567ee92b46b052dbc`.

Primary copy: [v8_tiny/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py>).

Actual-token concept refresh, arbitrary prefix lengths, version-owned caches.

Identical copies:

- [v8_tiny_12h/spec/recovery_2026-10-06/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/generation.py>)

Imports:

```python
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import *
from data import T,PAD,DONE,EOS,parse_prompt,verify
from runtime import sync,nll
```

Symbols:

- [expand_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:10>) — lines 10–12
- [Incremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:13>) — lines 13–43
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:14>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:15>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:31>).
- [FastIncremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:45>) — lines 45–93
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:46>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:74>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:85>).
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:96>) — lines 96–126
- [active_windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:128>) — lines 128–129
- [MaskedAdam](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:130>) — lines 130–136
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:131>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:133>).
- [repair_gradient](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/generation.py:137>) — lines 137–143

## v8_tiny/code/model.py

395 lines; SHA-256 `f8330079fa14cde1191592600c7cf2acf5b081e405dd5dce3518833fbf5e35c4`.

Primary copy: [v8_tiny/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py>).

V6 tiny: document-isolated NCP, deterministic recurrence, zero-anchored Z. No imports from mutable V5 code. FP32 is the reference execution mode.

Identical copies:

- [v8_tiny_12h/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/model.py>)
- [v9_tiny/frozen/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/model.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/model.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/model.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/model.py>)

Imports:

```python
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
```

Symbols:

- [rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:14>) — lines 14–15
- [safe_rms](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:18>) — lines 18–19
- [ball](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:22>) — lines 22–23
- [init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:26>) — lines 26–30
- [positions](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:33>) — lines 33–39
- [rotate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:42>) — lines 42–48
- [Block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:51>) — lines 51–96
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:52>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:60>), [static_step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:84>).
- [chunk_info](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:99>) — lines 99–102
- [shift](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:105>) — lines 105–112
- [Foundation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:115>) — lines 115–201
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:116>), [encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:133>), [pool](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:140>), [concept](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:143>), [pq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:148>), [port](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:152>), [decode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:155>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:161>), [vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:169>), [count_usage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:179>), [restart](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:185>).
- [Adapters](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:204>) — lines 204–241
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:205>), [raw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:222>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:227>), [calibrate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:235>).
- [Core](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:244>) — lines 244–273
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:245>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:255>).
- [DenseCore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:276>) — lines 276–287
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:277>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:279>).
- [Flat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:289>) — lines 289–295
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:290>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:292>).
- [System](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:297>) — lines 297–333
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:298>), [is_flat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:301>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:302>), [plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:304>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:314>), [active_blocks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:319>), [add_phase](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:324>), [metadata](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:332>).
- [DenseView](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:335>) — lines 335–345
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:336>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:337>).
- [count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:347>) — lines 347–348
- [active_count](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:351>) — lines 351–355
- [freeze](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:358>) — lines 358–362
- [log_density](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:365>) — lines 365–366
- [LoRABank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:368>) — lines 368–372
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:369>), [delta](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:372>).
- [project](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/model.py:374>) — lines 374–393

## v8_tiny/code/report.py

20 lines; SHA-256 `68a83b405768a5cf197366aeb18cbc1cf1979def273b2b4d0ed409a3ac6363d4`.

Primary copy: [v8_tiny/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/report.py>).

A report may state negative/partial evidence, never convert execution to scientific success.

Imports:

```python
import json
from pathlib import Path
from runtime import ROOT
```

Symbols:

- [write_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/report.py:6>) — lines 6–19

## v8_tiny/code/runtime.py

96 lines; SHA-256 `12d3eeb90331d0a6b04f8a72859a8d061288f7e099a963315164629d0b0312c7`.

Primary copy: [v8_tiny/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py>).

Atomic versioned checkpoints and loss denominators; V6-compatible math.

Imports:

```python
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import *
```

Symbols:

- [sync](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:10>) — lines 10–11
- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:12>) — lines 12–15
- [cpu_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:16>) — lines 16–20
- [finite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:21>) — lines 21–25
- [atomic_save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:26>) — lines 26–30
- [Log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:31>) — lines 31–35
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:32>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:33>).
- [seed_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:36>) — lines 36–36
- [implementation_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:37>) — lines 37–40
- [tensor_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:41>) — lines 41–44
- [save_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:45>) — lines 45–46
- [load_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:47>) — lines 47–61
- [optimizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:63>) — lines 63–70
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:71>) — lines 71–74
- [nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:75>) — lines 75–75
- [balanced](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:76>) — lines 76–80
- [ncp_weights](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:81>) — lines 81–84
- [controls](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/runtime.py:86>) — lines 86–96

## v8_tiny/code/search.py

143 lines; SHA-256 `8506bacdd9e8f9d7dd2b5338bc73d1af2e7fd3b95cadf426a4e8e1fd8e4731cb`.

Primary copy: [v8_tiny/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py>).

AB-MCTS-A over realized answer/control refinements, with charged wall time.

Identical copies:

- [v8_tiny_12h/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/search.py>)
- [v9_tiny/frozen/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/search.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/search.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/search.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/search.py>)

Imports:

```python
import hashlib,time
from dataclasses import dataclass,field
import numpy as np
import torch
from model import ball
from data import T,K,COL0,N_COL,NODE0,TO,SEP,DONE,parse_prompt
from generation import generate,repair_gradient,active_windows,MaskedAdam
from runtime import sync
```

Symbols:

- [Stats](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:12>) — lines 12–17
  Methods: [add](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:14>), [draw](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:17>).
- [Node](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:19>) — lines 19–28
  Methods: .
- [attach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:30>) — lines 30–33
- [eligible](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:35>) — lines 35–35
- [select](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:36>) — lines 36–43
- [control_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:45>) — lines 45–45
- [search_one](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:47>) — lines 47–125
- [public_grade](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search.py:128>) — lines 128–143

## v8_tiny/code/search_experiment.py

50 lines; SHA-256 `325803d2f849fcc0e9e86c14a0086a7ec3f9c52c4cf16772059a307a65a133d8`.

Primary copy: [v8_tiny/code/search_experiment.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search_experiment.py>).

Bounded development tuning and locked equal-time test comparisons.

Imports:

```python
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import generate
from search import search_one
from evaluate import paired,wilson
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/search_experiment.py:13>) — lines 13–48

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--flat-source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--seconds',type=float,default=3600)
```

## v8_tiny/code/stage_checks.py

58 lines; SHA-256 `33cf455c45505b3175f8b29db462e9b560c1a4385b8f208a310c27fbfc572e42`.

Primary copy: [v8_tiny/code/stage_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/stage_checks.py>).

Production-shaped optimizer updates and eager/compiled gradients for every train path.

Imports:

```python
import copy,json,time
import numpy as np
import torch
from model import *
from data import Mixer,gen_symbolic,parse_prompt,verify
from runtime import *
from train import configure,forward_function,objective,candidate_backward
from generation import Incremental,MaskedAdam,generate
from search import Node,attach,select,public_grade,search_one
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/stage_checks.py:12>) — lines 12–57

## v8_tiny/code/train.py

141 lines; SHA-256 `e5d377fd83c883d8d86db6a96387f99939ba2f13d83431c27e53b11e0910bafe`.

Primary copy: [v8_tiny/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py>).

Update/exposure-budget training. Dependent stages require explicit accepted anchors.

Imports:

```python
import argparse,copy,json,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from model import *
from data import Mixer,DATA,digest
from runtime import *
```

Symbols:

- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:10>) — lines 10–31
- [forward_function](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:33>) — lines 33–39
- [objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:41>) — lines 41–58
- [candidate_backward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:60>) — lines 60–68
- [frozen_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:70>) — lines 70–70
- [tensor_hash_tensors](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:71>) — lines 71–74
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/code/train.py:76>) — lines 76–140

CLI arguments:

```python
ap.add_argument('--out',required=True)
ap.add_argument('--stage',default='A',choices=['A','SIMPLE','MEM','HARD','REFIT','EXTRA','Z','CL'])
ap.add_argument('--source')
ap.add_argument('--flat',action='store_true')
ap.add_argument('--core',default='dense')
ap.add_argument('--kind',default='gated')
ap.add_argument('--mode',default='foundation')
ap.add_argument('--steps',type=int,default=24000)
ap.add_argument('--batch',type=int,default=64)
ap.add_argument('--context',type=int,default=192)
ap.add_argument('--seed',type=int,default=811)
ap.add_argument('--seconds',type=float,default=43200)
ap.add_argument('--alpha',type=float,default=.1)
ap.add_argument('--lr',type=float)
ap.add_argument('--beta',type=float,default=.25)
ap.add_argument('--constant-alpha',action='store_true')
ap.add_argument('--compiled',action='store_true')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--no-distill',action='store_true')
ap.add_argument('--always-open',action='store_true')
ap.add_argument('--fixed',action='store_true')
ap.add_argument('--learning-check',action='store_true')
```

## v8_tiny/tests/verify_gmm.py

10 lines; SHA-256 `c253faf150e859f7095aa2301d0bece578c4c86a4593d2719ac1be3e2117d058`.

Primary copy: [v8_tiny/tests/verify_gmm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/tests/verify_gmm.py>).

Imports:

```python
import sys,numpy as np,torch
from continual import fit_gmm
from model import log_density
from runtime import atomic_json,ROOT
from scipy.special import logsumexp
```

## v8_tiny_12h/code/checks.py

61 lines; SHA-256 `dd0a3514945f5bdd278804d5cdb33904b9da6258b8ea483e6772ba0b9c5ea1d9`.

Primary copy: [v8_tiny_12h/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/checks.py>).

Executable leakage, trained decoder, control, continual and gradient contracts.

Identical copies:

- [v9_tiny/frozen/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/checks.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/checks.py>)

Imports:

```python
import argparse,copy,json,traceback
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import Incremental,FastIncremental,generate,expand_bank,MaskedAdam,repair_gradient
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/checks.py:11>) — lines 11–57

CLI arguments:

```python
ap.add_argument('--source')
ap.add_argument('--out',required=True)
ap.add_argument('--fast',action='store_true')
```

## v8_tiny_12h/code/continual.py

88 lines; SHA-256 `586550c2004f4cfaed13481c8e6d66c4c870d68bd5615ba173e1128f13b589c4`.

Primary copy: [v8_tiny_12h/code/continual.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py>).

Frozen-reference local-support fitting, positive support, routing diagnostics.

Identical copies:

- [v9_tiny/frozen/code/continual.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/continual.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/continual.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/continual.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/continual.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/continual.py>)

Imports:

```python
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from model import *
from data import Mixer,DATA
from runtime import *
```

Symbols:

- [Reservoir](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:12>) — lines 12–15
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:13>), [add](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:14>).
- [fit_gmm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:18>) — lines 18–34: Diagonal EM with random-sample initial means; avoids conflicting macOS OpenMP runtimes. Three independent starts, full-reservoir likelihood, fixed50-iteration/tolerance budget.
- [fit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:36>) — lines 36–70
- [gate_rates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/continual.py:72>) — lines 72–83

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--positive',action='store_true')
ap.add_argument('--mode',default='foundation')
ap.add_argument('--tokens',type=int,default=1000000)
```

## v8_tiny_12h/code/data.py

228 lines; SHA-256 `620179347c9b69bd41c9ee9fc81c757368120f0b75efd566479445209e19eaea`.

Primary copy: [v8_tiny_12h/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py>).

Boundary-correct text packing and harder, independently generated symbolic tasks.

Identical copies:

- [v9_tiny/frozen/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/data.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/data.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/data.py>)

Imports:

```python
import gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:11>) — lines 11–15
- [split_tiny_stories](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:17>) — lines 17–29: Partition unique token-content groups, retaining one representative per group.
- [prepare_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:31>) — lines 31–55
- [parse_prompt](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:57>) — lines 57–69
- [verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:71>) — lines 71–98
- [canonical](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:100>) — lines 100–102
- [shortest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:104>) — lines 104–110
- [gen_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:112>) — lines 112–158
- [prepare_symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:160>) — lines 160–173
- [Text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:175>) — lines 175–189
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:176>), [row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:180>).
- [Mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:191>) — lines 191–221
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:192>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:195>), [state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:218>), [restore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:219>).
- [fixed_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py:223>) — lines 223–225

## v8_tiny_12h/code/drive_12h.py

249 lines; SHA-256 `fb968ccb185e752c80726a6a736dd18dcd8500de41b6291a2b58a58781571642`.

Primary copy: [v8_tiny_12h/code/drive_12h.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py>).

User-authorized full-exposure pilot: score thresholds annotate, never abort it.

Identical copies:

- [v9_tiny/frozen/code/drive_12h.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/drive_12h.py>)

Imports:

```python
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from drive import Queue as OriginalQueue, BudgetEnd
from runtime import *
from model import *
from data import Mixer,digest,DATA
from evaluate import retention,text_eval,symbolic_eval,paired,cluster_ci
from continual import gate_rates
```

Symbols:

- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:14>) — lines 14–237
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:15>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:26>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:29>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:68>), [eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:88>), [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:103>), [endpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:106>), [decision](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:108>), [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:110>), [continual](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:112>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:156>), [final_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:222>), [finish](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/drive_12h.py:232>).

CLI arguments:

```python
ap.add_argument('--hours',type=float,default=12)
ap.add_argument('--continue-saved',action='store_true')
```

## v8_tiny_12h/code/evaluate.py

94 lines; SHA-256 `cdd185a26a4105c98ad911466bfe75e912d4c3fd474dc666c2a3f25a9cd0a8ff`.

Primary copy: [v8_tiny_12h/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py>).

Public-header generation scoring, document-cluster uncertainty and absolute gates.

Identical copies:

- [v9_tiny/frozen/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/evaluate.py>)

Imports:

```python
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
from model import *
from data import *
from runtime import *
from generation import generate
```

Symbols:

- [wilson](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:13>) — lines 13–17
- [exact](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:18>) — lines 18–18
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:19>) — lines 19–23
- [cluster_ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:24>) — lines 24–28
- [text_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:30>) — lines 30–36
- [symbolic_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:38>) — lines 38–59
- [interface_audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:62>) — lines 62–70
- [competence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:72>) — lines 72–73
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:75>) — lines 75–83
- [retention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py:85>) — lines 85–92

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--split',default='dev')
ap.add_argument('--names',default='standard,lookup,compose,hard10')
ap.add_argument('--n',type=int,default=512)
ap.add_argument('--fast',action='store_true')
```

## v8_tiny_12h/code/generation.py

148 lines; SHA-256 `eea1601b715b6e8d3ddfc1ddb7cf5da503520511790e0b15fc85b479cdce0f1c`.

Primary copy: [v8_tiny_12h/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py>).

Actual-token concept refresh, arbitrary prefix lengths, version-owned caches.

Identical copies:

- [v9_tiny/frozen/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/generation.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/generation.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/generation.py>)

Imports:

```python
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import *
from data import T,PAD,DONE,EOS,parse_prompt,verify
from runtime import sync,nll
```

Symbols:

- [expand_bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:10>) — lines 10–12
- [Incremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:13>) — lines 13–43
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:14>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:15>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:31>).
- [FastIncremental](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:45>) — lines 45–98: Fixed-size caches with eager steps; training compilation is independent. Metal compiled decoding drifted beyond the trained-weight parity contract. Eager static and dynamic execution retain the original numerical tolerance.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:51>), [prefill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:79>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:90>).
- [generate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:101>) — lines 101–131
- [active_windows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:133>) — lines 133–134
- [MaskedAdam](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:135>) — lines 135–141
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:136>), [update](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:138>).
- [repair_gradient](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/generation.py:142>) — lines 142–148

## v8_tiny_12h/code/report.py

30 lines; SHA-256 `72507f0d067bd6404256e0d8d6b0be297d15e134413fd99542bd8ed95720026e`.

Primary copy: [v8_tiny_12h/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/report.py>).

A report may state negative/partial evidence, never convert execution to scientific success.

Identical copies:

- [v9_tiny/frozen/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/report.py>)

Imports:

```python
import json
from pathlib import Path
from runtime import ROOT
```

Symbols:

- [write_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/report.py:6>) — lines 6–29

## v8_tiny_12h/code/revision_checks.py

55 lines; SHA-256 `b8f82d7091080925eb32439583d3c0fb7302461ed68ef81b785ca6c86cde9a2e`.

Primary copy: [v8_tiny_12h/code/revision_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/revision_checks.py>).

Current-revision content isolation and production cache checks, not quality scores.

Identical copies:

- [v9_tiny/frozen/code/revision_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/revision_checks.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/revision_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/revision_checks.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/revision_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/revision_checks.py>)

Imports:

```python
import argparse,copy,json
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import Incremental,FastIncremental
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/revision_checks.py:10>) — lines 10–50

CLI arguments:

```python
a.add_argument('--source')
a.add_argument('--out',required=True)
```

## v8_tiny_12h/code/runtime.py

99 lines; SHA-256 `1427ebbe53032fa618db886eeae5b1f9c3d8815ca100dd35a6053b39ca993fb6`.

Primary copy: [v8_tiny_12h/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py>).

Atomic versioned checkpoints and loss denominators; V6-compatible math.

Identical copies:

- [v9_tiny/frozen/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/runtime.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/runtime.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/runtime.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/runtime.py>)

Imports:

```python
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import *
```

Symbols:

- [sync](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:10>) — lines 10–11
- [check_deadline](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:12>) — lines 12–14
- [atomic_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:15>) — lines 15–18
- [cpu_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:19>) — lines 19–23
- [finite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:24>) — lines 24–28
- [atomic_save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:29>) — lines 29–33
- [Log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:34>) — lines 34–38
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:35>), [__call__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:36>).
- [seed_all](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:39>) — lines 39–39
- [implementation_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:40>) — lines 40–43
- [tensor_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:44>) — lines 44–47
- [save_checkpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:48>) — lines 48–49
- [load_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:50>) — lines 50–64
- [optimizer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:66>) — lines 66–73
- [lr_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:74>) — lines 74–77
- [nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:78>) — lines 78–78
- [balanced](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:79>) — lines 79–83
- [ncp_weights](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:84>) — lines 84–87
- [controls](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py:89>) — lines 89–99

## v8_tiny_12h/code/search_experiment.py

65 lines; SHA-256 `824635bf50175ef949c108367de35ccb3085506e20e514abf5abfa052d478dbe`.

Primary copy: [v8_tiny_12h/code/search_experiment.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/search_experiment.py>).

Bounded development tuning and locked equal-time test comparisons.

Identical copies:

- [v9_tiny/frozen/code/search_experiment.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/search_experiment.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/search_experiment.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/search_experiment.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/search_experiment.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/search_experiment.py>)

Imports:

```python
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import generate
from search import search_one
from evaluate import paired,wilson
```

Symbols:

- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/search_experiment.py:13>) — lines 13–63

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--flat-source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--seconds',type=float,default=3600)
```

## v8_tiny_12h/code/stage_checks.py

63 lines; SHA-256 `68e1aa14d1eb8d3da54b626588d1eb67a3b2692e1d271549ac61b0c3b8494763`.

Primary copy: [v8_tiny_12h/code/stage_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/stage_checks.py>).

Production-shaped optimizer updates and eager/compiled gradients for every train path.

Identical copies:

- [v9_tiny/frozen/code/stage_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/stage_checks.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/stage_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/stage_checks.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/stage_checks.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/stage_checks.py>)

Imports:

```python
import copy,json,time
import numpy as np
import torch
from model import *
from data import Mixer,gen_symbolic,parse_prompt,verify
from runtime import *
from train import configure,forward_function,objective,candidate_backward
from generation import Incremental,MaskedAdam,generate
from search import Node,attach,select,public_grade,search_one
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/stage_checks.py:12>) — lines 12–62

## v8_tiny_12h/code/train.py

164 lines; SHA-256 `56ed6457ad305d37c8916d19ba71b88b9a581d0373f7e337d9f29f418907badf`.

Primary copy: [v8_tiny_12h/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py>).

Update/exposure-budget training. Dependent stages require explicit accepted anchors.

Identical copies:

- [v9_tiny/frozen/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/frozen/code/train.py>)
- [v8_tiny_12h/spec/recovery_2026-10-06/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/train.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/train.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/train.py>)

Imports:

```python
import argparse,copy,json,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from model import *
from data import Mixer,DATA,digest,NODE0,COL0,N_COL
from runtime import *
```

Symbols:

- [configure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:10>) — lines 10–31
- [forward_function](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:33>) — lines 33–39
- [objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:41>) — lines 41–69
- [candidate_backward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:71>) — lines 71–79
- [frozen_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:81>) — lines 81–81
- [tensor_hash_tensors](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:82>) — lines 82–85
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:87>) — lines 87–163

CLI arguments:

```python
ap.add_argument('--out',required=True)
ap.add_argument('--stage',default='A',choices=['A','SIMPLE','MEM','HARD','REFIT','EXTRA','Z','CL'])
ap.add_argument('--source')
ap.add_argument('--flat',action='store_true')
ap.add_argument('--core',default='dense')
ap.add_argument('--kind',default='gated')
ap.add_argument('--mode',default='foundation')
ap.add_argument('--steps',type=int,default=24000)
ap.add_argument('--batch',type=int,default=64)
ap.add_argument('--context',type=int,default=192)
ap.add_argument('--seed',type=int,default=811)
ap.add_argument('--seconds',type=float,default=43200)
ap.add_argument('--alpha',type=float,default=.1)
ap.add_argument('--lr',type=float)
ap.add_argument('--beta',type=float,default=.25)
ap.add_argument('--constant-alpha',action='store_true')
ap.add_argument('--compiled',action='store_true')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--no-distill',action='store_true')
ap.add_argument('--always-open',action='store_true')
ap.add_argument('--fixed',action='store_true')
ap.add_argument('--learning-check',action='store_true')
ap.add_argument('--development-curves',action='store_true')
```

## v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py

215 lines; SHA-256 `57a2dd7cd0d9164e6692a7e446cd9a470e6fabf376041bbce9ea9a9981fb4b47`.

Primary copy: [v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py>).

User-authorized full-exposure pilot: score thresholds annotate, never abort it.

Imports:

```python
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from drive import Queue as OriginalQueue, BudgetEnd
from runtime import *
from model import *
from data import Mixer,digest,DATA
from evaluate import retention,text_eval,symbolic_eval,paired,cluster_ci
from continual import gate_rates
```

Symbols:

- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:14>) — lines 14–203
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:15>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:18>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:57>), [eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:63>), [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:70>), [endpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:72>), [decision](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:74>), [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:76>), [continual](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:78>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:122>), [final_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:188>), [finish](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/drive_12h.py:198>).

CLI arguments:

```python
ap.add_argument('--hours',type=float,default=12)
```

## v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py

94 lines; SHA-256 `678346cf96c98f3d2ced30952a481fab9423381818af3852b20a7712899a5d95`.

Primary copy: [v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py>).

Public-header generation scoring, document-cluster uncertainty and absolute gates.

Identical copies:

- [v8_tiny_12h/spec/recovery_json_2026-10-06/code/evaluate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/evaluate.py>)

Imports:

```python
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
from model import *
from data import *
from runtime import *
from generation import generate
```

Symbols:

- [wilson](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:13>) — lines 13–17
- [exact](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:18>) — lines 18–18
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:19>) — lines 19–23
- [cluster_ci](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:24>) — lines 24–28
- [text_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:30>) — lines 30–36
- [symbolic_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:38>) — lines 38–59
- [interface_audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:62>) — lines 62–70
- [competence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:72>) — lines 72–73
- [evaluate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:75>) — lines 75–83
- [retention](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/evaluate.py:85>) — lines 85–92

CLI arguments:

```python
ap.add_argument('--source',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--split',default='dev')
ap.add_argument('--names',default='standard,lookup,compose,hard10')
ap.add_argument('--n',type=int,default=512)
ap.add_argument('--fast',action='store_true')
```

## v8_tiny_12h/spec/recovery_2026-10-06/code/report.py

27 lines; SHA-256 `d3e6daf12efa1b607c9511f29bdce6c77097904327d9bf1e5521385bb937f196`.

Primary copy: [v8_tiny_12h/spec/recovery_2026-10-06/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/report.py>).

A report may state negative/partial evidence, never convert execution to scientific success.

Imports:

```python
import json
from pathlib import Path
from runtime import ROOT
```

Symbols:

- [write_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/code/report.py:6>) — lines 6–26

## v8_tiny_12h/tests/verify_gmm.py

11 lines; SHA-256 `364d68d2be215fb34c7ee471a4a7dfec8fac0d8d0f26ac9cb0bf0028d952376b`.

Primary copy: [v8_tiny_12h/tests/verify_gmm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_gmm.py>).

Identical copies:

- [v8_tiny_12h/spec/recovery_2026-10-06/tests/verify_gmm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_2026-10-06/tests/verify_gmm.py>)
- [v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_gmm.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_gmm.py>)

Imports:

```python
import sys,numpy as np,torch
from pathlib import Path
from continual import fit_gmm
from model import log_density
from runtime import atomic_json,ROOT
from scipy.special import logsumexp
```

## v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py

238 lines; SHA-256 `ec2c7f014d04f1693041de99cb67f9a3bdee61f85667bbb4279c82c74624d666`.

Primary copy: [v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py>).

User-authorized full-exposure pilot: score thresholds annotate, never abort it.

Imports:

```python
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from drive import Queue as OriginalQueue, BudgetEnd
from runtime import *
from model import *
from data import Mixer,digest,DATA
from evaluate import retention,text_eval,symbolic_eval,paired,cluster_ci
from continual import gate_rates
```

Symbols:

- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:14>) — lines 14–226
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:15>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:26>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:29>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:68>), [eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:85>), [audit](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:92>), [endpoint](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:95>), [decision](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:97>), [score](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:99>), [continual](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:101>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:145>), [final_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:211>), [finish](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/drive_12h.py:221>).

CLI arguments:

```python
ap.add_argument('--hours',type=float,default=12)
ap.add_argument('--continue-saved',action='store_true')
```

## v8_tiny_12h/spec/recovery_json_2026-10-06/code/report.py

29 lines; SHA-256 `ecbfdf83ea79d6389a4e84215e6a7a2cb65144a7761e03560d5fb1c7aa77b57a`.

Primary copy: [v8_tiny_12h/spec/recovery_json_2026-10-06/code/report.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/report.py>).

A report may state negative/partial evidence, never convert execution to scientific success.

Imports:

```python
import json
from pathlib import Path
from runtime import ROOT
```

Symbols:

- [write_report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/code/report.py:6>) — lines 6–28

## v8_tiny_12h/tests/recovery_diagnose.py

32 lines; SHA-256 `0907dd26a5747bc474729b7dd884e0457c5d4a088d0f3f2a6e74d04f4dd809bd`.

Primary copy: [v8_tiny_12h/tests/recovery_diagnose.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/recovery_diagnose.py>).

Read trained weights and compare causal, dynamic, and static execution.

Identical copies:

- [v8_tiny_12h/spec/recovery_json_2026-10-06/tests/recovery_diagnose.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/tests/recovery_diagnose.py>)

Imports:

```python
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from model import DZ,freeze
from data import gen_symbolic
from runtime import load_system,atomic_json
from generation import Incremental,FastIncremental
```

CLI arguments:

```python
ap.add_argument('--device',required=True)
ap.add_argument('--out',required=True)
```

## v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_recovery_resume.py

26 lines; SHA-256 `44bd2852672430a5acf8a8e380b0510f25b1d10dc55763741697ff80a667fd3d`.

Primary copy: [v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_recovery_resume.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_recovery_resume.py>).

Resume must reuse completed work, retain its provenance, and keep the clock.

Imports:

```python
import ast,json,sys,time
from pathlib import Path
from unittest.mock import patch
from drive_12h import Queue,OriginalQueue
from runtime import implementation_hash,atomic_json
```

Symbols:

- [main_ast](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/spec/recovery_json_2026-10-06/tests/verify_recovery_resume.py:8>) — lines 8–10

## v8_tiny_12h/tests/verify_json_recovery.py

51 lines; SHA-256 `64f0c613533842264de84f3c5da415c3ffb7fa9b44ff201f0628d07997a4bb83`.

Primary copy: [v8_tiny_12h/tests/verify_json_recovery.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_json_recovery.py>).

Exercise the actual failing evaluation flag and both prior-coverage outcomes.

Imports:

```python
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from evaluate import competence,symbolic_eval,wilson
from runtime import atomic_json,implementation_hash
from data import DATA
from model import VOCAB
```

Symbols:

- [Fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_json_recovery.py:31>) — lines 31–33
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_json_recovery.py:32>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_json_recovery.py:33>).

## v8_tiny_12h/tests/verify_recovery_resume.py

38 lines; SHA-256 `99c77ec812ad6eaaa40407e3a4e452f6447619d4bc695cd99c7aa504805b64e1`.

Primary copy: [v8_tiny_12h/tests/verify_recovery_resume.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_recovery_resume.py>).

Resume must reuse completed work, retain its provenance, and keep the clock.

Imports:

```python
import ast,json,sys,time,torch
from pathlib import Path
from unittest.mock import patch
from drive_12h import Queue,OriginalQueue
from runtime import implementation_hash,atomic_json
```

Symbols:

- [main_ast](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/tests/verify_recovery_resume.py:8>) — lines 8–10

## v9_tiny/code/baselines.py

203 lines; SHA-256 `dae6628d35084eaae2ff99170629edb8d833997fd5465c7f41152a669d94eefd`.

Primary copy: [v9_tiny/code/baselines.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py>).

S4 baselines (draft §8.2), from the same promoted checkpoint in code-free mode, same problems per round and the same training verifier calls as expert iteration (L3): rft token rejection-sampling fine-tuning: sample complete answers at T = 1, keep verified successes, fine-tune a rank-16 LoRA on q and v of the three decoder blocks (24,576 parameters); grpo group-relative policy gradient on the same adapter (group of 8 samples per problem); reinforce REINFORCE on the selector with the verifier reward (codes sampled from pi, greedy decoding).

Imports:

```python
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from v9paths import digest, read_json, write_json
from model import D, chunk_info
from runtime import DEVICE, atomic_save, nll
from data import DONE, PAD, verify
import hncp, hgen, train_s4
from hgen import CH, classify
```

Symbols:

- [QVLoRA](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:25>) — lines 25–35: Wraps a frozen Block.qkv: adds rank-16 LoRA deltas to the q and v outputs.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:27>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:32>).
- [add_lora](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:38>) — lines 38–45
- [lora_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:48>) — lines 48–49
- [load_lora](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:52>) — lines 52–56
- [sample_answers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:60>) — lines 60–96: Complete answers, token-sampled at temperature temp (code-free plans), or code-sampled from pi with greedy tokens (code_policy='sample'). Full recomputation. Returns per item (tokens, codes, success).
- [seq_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:99>) — lines 99–104
- [token_logprob](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:107>) — lines 107–110: Per-row sum of response log-probs under code-free plans (adapter active).
- [train_round](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:113>) — lines 113–154: rft: maximise log p of accepted answers. grpo: sum_i A_i log p(answer_i), A normalised within each problem's group. reinforce: sum_i (o_i - mean) log pi(codes_i) on the selector.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/baselines.py:157>) — lines 157–199

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--method', required=True, choices=['rft', 'grpo', 'reinforce'])
ap.add_argument('--ei', required=True, help='expert-iteration run folder: per-round problem seeds, sizes and verifier calls are matched')
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--group', type=int, default=8)
```

## v9_tiny/code/ceiling.py

89 lines; SHA-256 `7b065e3baea749d762689942148ddfac5837b6fd50b20fdbe5078a96afa28fc7`.

Primary copy: [v9_tiny/code/ceiling.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py>).

Exact demonstration next-action distribution for standard GraphNav prefixes (draft §6.6 G2.2(b)). The standard generator (frozen data.py:120-129) draws a walk program: 2-4 colour runs (uniform), the first colour uniform over 3, each later colour uniform over the 2 others, each run 2-4 hops (uniform): 2,322 programs per graph and start. It keeps a program only if the final node is the goal and no earlier node (start included) equals it. Conditional on the prompt (graph, start, goal), the demonstration is a program drawn with probability proportional to its prior weight among the accepted ones ending at the goal.

Imports:

```python
import itertools
from functools import lru_cache
import numpy as np
import v9paths
from data import parse_prompt, N_COL
```

Symbols:

- [programs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py:19>) — lines 19–32
- [accepted](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py:35>) — lines 35–49: Accepted programs for this prompt as (colour sequence, weight), with weights normalised.
- [next_action_distribution](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py:52>) — lines 52–62: P(next action | prompt, demonstration prefix of colours); DONE when a consistent program has exactly that length.
- [top2_mass](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py:65>) — lines 65–66
- [self_test](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/ceiling.py:69>) — lines 69–85

## v9_tiny/code/csearch.py

469 lines; SHA-256 `239d9e5c8eee3f18fb76796e2cfa74fe9c9faf018fe748d689f096f31b9f8ae4`.

Primary copy: [v9_tiny/code/csearch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py>).

Concept search (draft §9.3-§9.4, I15): RH(H) receding horizon, and one-tree search G(R) / IMAG, over K codes + STOP, lockstep over B problems, full recomputation through hgen.Engine primitives (no caches). Children are imagined (c~_{j+1} = chat_j^(k), one core pass from the nearest grounded ancestor's concepts) or grounded (decode the pending chunks under their stored plan records, re-encode). A plan record is immutable: once a chunk has been decoded from a record, every later decode of that prefix applies the same record (node.over). Leaf values: 'rollout' = verified greedy grounded completion with selector-argmax codes; 'value' = V_psi(u0).

Imports:

```python
import gc
import time
from functools import wraps
import numpy as np
import torch
from torch import nn
import v9paths
from model import D
from data import DONE, SEP, verify
import hgen
from hgen import CH, Item, classify, stop_legal
```

Symbols:

- [_release_search_memory](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:28>) — lines 28–43: Reclaim completed MPS trees after their stack frame has released local refs. Nodes link parents and children, so refcounting alone retains their tensors. Variable prefix/batch shapes also leave unused buffers in the MPS allocator. Run only
- [Heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:46>) — lines 46–56: V_psi and tau_psi: 128 -> 1 each on the code-free u^(0) (§8.2), versioned per S4 iteration.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:48>), [value](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:52>), [term](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:55>).
- [Node](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:59>) — lines 59–68
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:63>).
- [Problem](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:71>) — lines 71–87
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:72>), [tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:79>), [code_array](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:82>).
- [Search](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:90>) — lines 90–465: cfg: H (depth cap; one-tree uses the hop limit), children 'imagined'|'grounded', ground_every R (one-tree G(R); None = IMAG), leaf 'rollout'|'value', budget (expansions per search), noise (root Dirichlet), heads, one_tree.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:94>), [stop_prob](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:100>), [leaf_value](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:105>), [set_heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:108>), [ground_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:114>), [imagine_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:124>), [code_matrix](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:141>), [decode_over](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:148>), [f_encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:168>), [finish_child](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:171>), [decode_children](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:182>), [replay](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:197>), [verify_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:235>), [rollout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:240>), [puct](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:274>), [grounding_rule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:281>), [simulate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:288>), [make_roots](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:361>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:377>), [one_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:421>), [solve](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:456>).
- [problems_from](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/csearch.py:468>) — lines 468–469

## v9_tiny/code/drive_q1.py

206 lines; SHA-256 `d71a4f3b5e1105096f21f1292b4db3a1a1e3e186771a38bffe5b90c248e23b71`.

Primary copy: [v9_tiny/code/drive_q1.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py>).

Q1 queue (draft §5.2, §14): S1 foundations, two new seeds of V8's exact recipe, dev evaluation, G1, sealed S1 tests. Detached and resumable: completed runs are verified and skipped; an interrupted run resumes from its checkpoint with identical arguments (frozen train.py enforces this). Only frozen V8 code trains or evaluates.

Imports:

```python
import argparse, json, os, signal, subprocess, sys, time, traceback
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json
```

Symbols:

- [train_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:20>) — lines 20–29
- [Stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:32>) — lines 32–33
  Methods: .
- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:36>) — lines 36–182
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:37>), [stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:43>), [log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:48>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:53>), [preflight](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:67>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:84>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:106>), [once](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:126>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:131>), [g1](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:163>).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q1.py:185>) — lines 185–202

CLI arguments:

```python
ap.add_argument('--dry-run', action='store_true')
```

## v9_tiny/code/drive_q2.py

76 lines; SHA-256 `6874acda34456e355a98f340b86bfc547005264111bc6c541bfaa48fa065ea8b`.

Primary copy: [v9_tiny/code/drive_q2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q2.py>).

Q2 queue (draft §10, §14): S6-base on the S1 anchors: LSL-scaled, always-open and VQ-only, 3 phases, 2 seeds. G6.0 before each phase fixes the phase label; P2 is colour-count or shifted (whichever passes G6.0; if both, the lower mean old-task hit rate, colour-count on a tie; if neither, colour-count, labelled non-separable).

Imports:

```python
import json, time
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, CODE, digest, read_json, write_json
from queue_base import Queue
```

Symbols:

- [Q2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q2.py:16>) — lines 16–72
  Methods: [measure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q2.py:19>), [g60](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q2.py:28>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q2.py:32>).

## v9_tiny/code/drive_q3.py

116 lines; SHA-256 `9a1769bec86b5ee5a96674362d6b552e70f697f2f718c135d3ab7f449d60b0b8`.

Primary copy: [v9_tiny/code/drive_q3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py>).

Q3 queue (draft §6.7, §14): S2 Tier A (dense) and Tier B (recurrent), two seeds, the two V8-schedule replays, all S2 measurements, the register decisions and promotion. Waits for Q1 (the S2 teachers are the new A_delayed seeds).

Imports:

```python
import json, sys, time
import numpy as np
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, V8, digest, read_json, write_json
from queue_base import Queue, Stop
```

Symbols:

- [run_name](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:18>) — lines 18–19
- [Q3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:22>) — lines 22–112
  Methods: [arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:25>), [replay](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:36>), [tier](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:52>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:66>), [promote](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q3.py:99>).

## v9_tiny/code/drive_q4.py

53 lines; SHA-256 `4da0260b9ea2a3764a440230c047ab1092f7e39b0df1829ac15104abb6e104e4`.

Primary copy: [v9_tiny/code/drive_q4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q4.py>).

Q4 queue (draft §7, §14): S3 on the promoted S2 dense model, per promoted seed: the lambda transplant curve, S3a-dec and S3b (6,000 updates, same data order), S2 measurements recomputed, G3, and the model that goes forward. If K8 was not promoted (G2.2(d)), S3a-dec only, on V9-K1, without G3's authority conditions.

Imports:

```python
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, digest, read_json, write_json
from queue_base import Queue
```

Symbols:

- [Q4](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q4.py:14>) — lines 14–49
  Methods: [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q4.py:17>).

## v9_tiny/code/drive_q5.py

58 lines; SHA-256 `4d714f120bbe7dae2420007f58f21f5d019264b2859584c922442eace98c528c`.

Primary copy: [v9_tiny/code/drive_q5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q5.py>).

Q5 queue (draft §8, §14): S4 per promoted seed: the nodes/s and iteration-time pilot (I15 sizing rule), expert iteration over codes (iteration 0 + 4), the three baselines matched in problems and verifier calls, development greedy evaluation of every arm, and the GradCuit lr / norm-bound pilot. Concept branches run only if G2.1 did not fail and K8 was promoted (§6.8, G2.2(d)); otherwise S4 runs the token baselines only.

Imports:

```python
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, read_json, write_json
from queue_base import Queue
```

Symbols:

- [Q5](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q5.py:14>) — lines 14–54
  Methods: [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q5.py:17>).

## v9_tiny/code/drive_q6.py

113 lines; SHA-256 `fd5f5264aea2a75af7d345e8987a75149665ad970c3a508d02352472bf55aa00`.

Primary copy: [v9_tiny/code/drive_q6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q6.py>).

Q6 queue (draft §9.5-§9.6, §14, I17): S5 per promoted seed. 1. Development choices (dev hard10, 256 items, 4x): token decoder (strongest S4 token arm by dev greedy hard10, else the promoted decoder), token tree leaf rule (rollout vs value head), and the best inner method for the answer loop. 2. S5_SEALED_PROCEDURE.json is written before any sealed item is read. 3. Primary matrix on sealed hard10 (512) and hard14 (256): RH(1), RH(H*) if H* > 1, G(1), token tree, best-of-N, at budgets {1, 4, 16} x t_g. Methods the register disallows are skipped and recorded. 4. Mechanism (H16): grounded RH(1) on sealed hard10 at 16x, then decoder-free RH(1) at 

Imports:

```python
import time
import numpy as np
import v9paths
from v9paths import RUNS, SMOKE, CODE, SPEC, read_json, write_json, digest, DATA
from queue_base import Queue
```

Symbols:

- [Q6](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q6.py:23>) — lines 23–109
  Methods: [cell](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q6.py:26>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_q6.py:30>).

## v9_tiny/code/drive_qf.py

35 lines; SHA-256 `8ae28cbcc6770a63642455583281f401f3b763d81fe9e190aba5ea6a68e034d1`.

Primary copy: [v9_tiny/code/drive_qf.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_qf.py>).

QF (final): sealed greedy evaluation of the S4 arms under a procedure fixed before any sealed read (L3), then the final analysis: stage-wise Holm families, per-seed and pooled paired results, FINAL_REPORT.md.

Imports:

```python
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, SPEC, read_json, write_json, digest, DATA
from queue_base import Queue
```

Symbols:

- [QF](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_qf.py:11>) — lines 11–31
  Methods: [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/drive_qf.py:14>).

## v9_tiny/code/export_k1.py

11 lines; SHA-256 `c4db875abcc55ed54450319b243cd1160ab4bd45011609f78ee2a5a51ac49e54`.

Primary copy: [v9_tiny/code/export_k1.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/export_k1.py>).

Export a K = 1 H-NCP checkpoint as the exact frozen-V8 System checkpoint, for the frozen evaluator (replays).

Imports:

```python
import argparse
import v9paths
from runtime import atomic_save
import hncp
```

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
```

## v9_tiny/code/final_analysis.py

118 lines; SHA-256 `dc06a3c9ab63152a16146b6fd18712dcf9bd50ff8b5b38883f778cbfdc5a32fb`.

Primary copy: [v9_tiny/code/final_analysis.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/final_analysis.py>).

Final analysis (draft §13.2, I18): the Holm-corrected families within each stage, per-seed and pooled paired tests, margin claims by one-sided sign-flip tests against the margin, and the summary report. Missing runs are reported as missing, never as passing or failing.

Imports:

```python
import json
from pathlib import Path
import numpy as np
import v9paths
from v9paths import RUNS, SPEC, read_json, write_json
from stats import mcnemar_exact, paired_bootstrap, paired_signflip, holm
import register_s2 as R2
```

Symbols:

- [onesided](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/final_analysis.py:15>) — lines 15–19: Sign-flip p-value for H1: mean(d) > 0 (direction +1) or < 0 (direction -1); d may be (seeds, items).
- [compare](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/final_analysis.py:22>) — lines 22–40: a_list/b_list: per-seed per-item vectors (same items within a seed). kind: diff | noninf | bound | equiv | drop.
- [s5_vec](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/final_analysis.py:43>) — lines 43–45
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/final_analysis.py:48>) — lines 48–114

## v9_tiny/code/hgen.py

230 lines; SHA-256 `c33c62912c8c3b2f900b5fe0a9a8abe80f86d511dd80deea4baf7fd5a83b2db7`.

Primary copy: [v9_tiny/code/hgen.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py>).

H-NCP inference engine (draft §4.2, §6.6, §9.3): grounded states, chunk decoding under plan records, selector-argmax rollouts, decoder-free (imagined) transitions. Full recomputation, no caches: every result is the full causal forward of the prefix, so cached-vs-full parity holds by construction. Plan records: codes\[r, j\] in {-1 (code-free), 0..K-1}; the plan used to decode chunk j+1 is (chat_j^(k), A_j^(k)).

Imports:

```python
from collections import deque
import numpy as np
import torch
import v9paths
from model import D, chunk_info, shift
from data import PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, T, parse_prompt, verify
```

Symbols:

- [bfs_to_goal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:18>) — lines 18–29: Hop distance from every label to the goal along coloured edges.
- [Item](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:32>) — lines 32–55: Parsed symbolic item: prompt, demonstration, graph facts.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:34>), [walk](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:40>), [progress_options](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:52>).
- [classify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:58>) — lines 58–69: Action class of one decoded chunk at node cur: ('hop', colour, legal) / ('done', legal) / ('malformed',).
- [Engine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:72>) — lines 72–224
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:73>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:77>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:87>), [hidden](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:91>), [logits_at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:97>), [at](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:101>), [state_heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:106>), [decode_chunk](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:119>), [chunk_logprob](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:145>), [pooled_next](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:158>), [imagine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:165>), [coded](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:180>), [greedy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:185>).
- [stop_legal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hgen.py:227>) — lines 227–230: The verifier's STOP rule (data.py:85): DONE is legal iff the current node is the goal after >= 1 hop. Also the tau_psi label rule (draft §8.2); it reads only the grounded prefix, never a reference continuation.

## v9_tiny/code/hncp.py

181 lines; SHA-256 `cb240d85da3c74dcdf744016944ddba3649415767b269ab6eda34f4b7a2770d3`.

Primary copy: [v9_tiny/code/hncp.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py>).

V9 H-NCP (draft §6): K committed next-concept hypotheses at the core's final block application B_H. Ownership is V8 REFIT's (§6.1): the foundation (encoder, embedding/tied head, decoder, Q, PQ heads, port) is frozen; a new core (DenseCore copy of C1-C4, or V8's recurrent Core) is trained. V9 adds only the code embeddings e_k, the write W_z and the selector W_pi, and only when K >= 2 (at K = 1 nothing is instantiated).

Imports:

```python
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from model import D, HEADS, S, CODES, Block, DenseCore, Core, System, Foundation, rms, positions, rotate, chunk_info, shift, freeze, count
from runtime import balanced, nll, ncp_weights, load_system
```

Symbols:

- [Codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:18>) — lines 18–32: e_k in R^16, W_z in R^(128x16), selector W_pi in R^(Kx128) + b_pi. 3,208 parameters at K = 8.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:20>), [deltas](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:31>).
- [block_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:35>) — lines 35–63: B_H with the code entering only the query and the residual (§6.2). x: (b, t, D) block input; delta: (K, D) or (b, t, K, D) code writes. Keys and values come from the unperturbed stream, position j's own included. Returns (b, t, K, D). LoRA
- [core_stream](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:66>) — lines 66–88: Run the core up to B_H. Returns (x, block, finish): B_H's input stream, the block, and the readout.
- [pq_any](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:91>) — lines 91–94: Teacher-frozen soft-PQ expectation for any leading shape (NCP paper eq. 9).
- [HNCP](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:97>) — lines 97–150
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:98>), [is_flat](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:106>), [context](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:109>), [heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:114>), [plan](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:125>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:134>), [as_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:142>), [metadata](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:148>).
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:153>) — lines 153–166: Student from a frozen S1 teacher checkpoint (§6.1). The foundation is a copy; the teacher stays separate.
- [save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:169>) — lines 169–171
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/hncp.py:174>) — lines 174–181

## v9_tiny/code/kernels.py

109 lines; SHA-256 `9b31e6ef19c1c842d12d6e5e47c86521f69514b3bded8b411311be794183b737`.

Primary copy: [v9_tiny/code/kernels.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py>).

Refine kernels (draft §8.3): GradCuit on the code embedding at one chunk (e = e_k + delta through W_z) and the random-direction control with the same step norm. Reward R_grad = o - 1 (0 accepted -> stop, -1 rejected), loss -R_grad * sum_t log p(x_t | plans(e_k + delta)) on the current failed answer, teacher-forced; Adam, <= 10 updates; after each update the answer is regenerated greedily from that chunk on and verified. Bound: RMS(delta) <= .5 RMS(e_k).

Imports:

```python
import math
import numpy as np
import torch
import v9paths
from data import DONE, verify
from hgen import CH, classify
```

Symbols:

- [plans_with](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py:16>) — lines 16–29: Plans under the plan records `codes`, with chunk j's plan recomputed from the edited code table e_prime.
- [answer_logprob](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py:32>) — lines 32–37: Teacher-forced sum of log p over response tokens P..L-1 (one row).
- [regenerate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py:41>) — lines 41–65: Keep tokens up to the end of chunk j; regenerate greedily from chunk j+1 with chunk j's plan from e_prime and selector-argmax codes after it. Returns (tokens, codes).
- [project](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py:68>) — lines 68–72
- [refine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/kernels.py:75>) — lines 75–109: One CONT refinement of a failed answer. Returns dict(success, updates, verifier_calls, tokens, codes, j, trace). reward_from_o=True is the contract-6 mutant (R = o): it must be caught as a no-op.

## v9_tiny/code/launch.py

20 lines; SHA-256 `a5d217e8dcb2eb5f140150b33c0f15953f0dd6d228af7489fe08e6a0a4702262`.

Primary copy: [v9_tiny/code/launch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/launch.py>).

Start a V9 block driver detached from this shell (new session), with macOS idle-sleep prevention while it runs.

Imports:

```python
import json, subprocess, sys, time
from pathlib import Path
import v9paths
from v9paths import V9, RUNS, CODE, read_json, write_json
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/launch.py:7>) — lines 7–17

## v9_tiny/code/launch_block.py

43 lines; SHA-256 `1e715ec0242448492a8c4ceaf868a033472e8cc1e38d54787cd32b3b2d59f2dd`.

Primary copy: [v9_tiny/code/launch_block.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/launch_block.py>).

Start a Q2+ block driver detached (new session) with idle-sleep prevention. Refuses to start unless the block's contracts and smoke passed under the current lock, and refuses a duplicate of the same block.

Imports:

```python
import json, subprocess, sys, time
import v9paths
from v9paths import V9, RUNS, CODE, TESTS, SPEC, read_json, write_json
from queue_base import lock_identity
import importlib
```

Symbols:

- [check_q5_search_processes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/launch_block.py:10>) — lines 10–16: A detached search worker may briefly outlive a crashed training parent.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/launch_block.py:19>) — lines 19–39

## v9_tiny/code/lsl.py

263 lines; SHA-256 `42f5d64c22345e04666df097d1c483cf409c13eb66901bb5a3c22642cd568cfa`.

Primary copy: [v9_tiny/code/lsl.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py>).

S6 Local Support Learning, scaled configuration (draft §10.1, §10.4, §10.6). LSL-scaled: rank-16 LoRA on q, k, v, o, w12, w3 of every gated block; one gate for q/k/v (block input n1(x)), one each for o, w12 and w3 on that matrix's own input; JL projection 64; Phi_pos 16 / Phi_neg 32 diagonal GMMs; temporal smoothing s_t = rho*g_t + (1-rho)*s_{t-1}, applied iff s_t > .5, reset at document boundaries. Always-open: the same adapters, applied everywhere. The newest bank is held open while it trains (paper Alg. 4). Execution is full recomputation (no caches), so smoothing state is exact per sequence.

Imports:

```python
import copy, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from model import D, HEADS, FF, Block, System, Foundation, positions, rotate, freeze, count, chunk_info
from runtime import load_system, atomic_save
```

Symbols:

- [component_logp](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:24>) — lines 24–28: log w_k + log N(h; mu_k, diag var_k) via the quadratic expansion (no (n, K, d) intermediate).
- [log_density](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:31>) — lines 31–32
- [smooth](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:35>) — lines 35–45: s_t = rho*hit_t + (1-rho)*s_{t-1}, reset at document boundaries (and at -1 positions). hit: (b, t) float.
- [Bank](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:48>) — lines 48–60: One phase's rank-16 adapters for one block, plus its Phi_pos per gate group.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:50>), [delta](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:59>).
- [LSLBlock](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:63>) — lines 63–115: Block with per-matrix phase banks and gates. Created by converting an existing Block in place.
  Methods: [setup](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:65>), [gate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:74>), [adapted](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:89>), [forward](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:99>).
- [gated_blocks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:118>) — lines 118–120
- [convert](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:123>) — lines 123–127: S6-base: every E3, C4 and D3 block becomes an LSL block (10 gated blocks).
- [add_phase](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:130>) — lines 130–133
- [phases](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:136>) — lines 136–137
- [set_force](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:140>) — lines 140–142
- [fit_gmm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:146>) — lines 146–165: Diagonal EM (V8 fit_gmm's protocol: 3 random-example starts, <= 50 iterations), in torch on the device.
- [capture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:169>) — lines 169–187: Reservoir of JL-projected inputs per (block, group) over valid positions of the given batches.
- [fit_negative](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:190>) — lines 190–197
- [fit_positive](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:200>) — lines 200–210: Phi_pos for bank `phase` from activations on phase data (paper Alg. 1: after training, on M + adapter). target: write into these banks instead (G6.0 proxy with the adapter at zero).
- [hit_rates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:214>) — lines 214–244: Smoothed gate rates of the given (per-block) banks on batches: per (block, group) means, joint all-open share.
- [save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:248>) — lines 248–249
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/lsl.py:252>) — lines 252–263

## v9_tiny/code/measure_s2.py

555 lines; SHA-256 `fdc8caff0766a80a287b4b22249500a98b63685dc9812a8b01d02e4cb1ac2502`.

Primary copy: [v9_tiny/code/measure_s2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py>).

S2 measurements (draft §6.6): everything the register rows G2.1-G2.4 read, per model, with per-item vectors. Development data only. One grounded state per item per distribution, so bootstrap over items is bootstrap over states. Thresholds are not here; they are in register_s2.py (draft §12.3).

Imports:

```python
import argparse, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, RUNS, write_json, read_json, digest
from model import D, chunk_info, shift, Core, freeze
from runtime import DEVICE, nll, ncp_weights, balanced, load_system
from data import PAD, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, fixed_text
import hncp, hgen, ceiling
from hgen import CH, Engine, Item, classify
```

Symbols:

- [load_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:22>) — lines 22–24
- [tf_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:29>) — lines 29–53: Per-row (sum NLL, targets) under a plan mode: codefree | oracle | argmax | sample. Also k*, pi at scored chunks.
- [mixture_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:57>) — lines 57–80: Per-chunk mixture log sum_k pi_k p(chunk | k), other chunks' codes at selector-argmax (exact per chunk).
- [nll_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:83>) — lines 83–117
- [collect_calibration](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:120>) — lines 120–126
- [ece](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:129>) — lines 129–134
- [calibration_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:137>) — lines 137–147
- [eval_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:151>) — lines 151–153: I20: MPS driver memory grows with batch x K x distinct lengths; K > 1 decodes score K code rows per sequence, so batch 16 (plateau 2 GB on hard14) where dense K = 1 keeps 64 (I19); recurrent K = 1 also uses 16 (I22).
- [greedy_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:156>) — lines 156–163
- [self_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:166>) — lines 166–181: D_self: selector-argmax rollouts up to the demonstration's response length or DONE; one legal-history state per item.
- [gold_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:184>) — lines 184–193
- [wrong_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:196>) — lines 196–208: D_wrong: a D_self state after one forced illegal or non-progress hop (claimed node becomes current).
- [fill_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:211>) — lines 211–226: Plan records for prefixes not produced by a rollout: selector-argmax at every coded chunk (causal, one pass).
- [eng_P](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:229>) — lines 229–230
- [decode_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:235>) — lines 235–255: For each state and code k: the decoded chunk, its action class, and (unless port0) the pooled real concept.
- [candidate_concepts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:259>) — lines 259–269: 73 candidate chunks (3 colours x 24 nodes + DONE), each encoded in the state's own context and pooled.
- [g24](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:272>) — lines 272–307: Per-state G2.4 quantities. heads: per-state chatk (K, D) and pi (K,) at the state's last chunk.
- [state_head_values](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:311>) — lines 311–321
- [g23](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:326>) — lines 326–378: Code sequence of length n_max fixed in advance (argmax along the imagined path, or uniform); imagined vs executed.
- [g23_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:381>) — lines 381–388
- [open_loop_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:393>) — lines 393–404: E5 form: every chunk after c0 replaced by the model's own committed prediction, decoder reads true tokens. Codes are used only at chunks with a scored next chunk (§6.2); elsewhere the code-free prediction is fed.
- [drift_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:407>) — lines 407–447: E5-form open-loop NLL increase for this model, and code-free drift of u0/A0 against the frozen teacher.
- [engaged_depth](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:452>) — lines 452–471
- [pblur2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:475>) — lines 475–490
- [measure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/measure_s2.py:494>) — lines 494–549

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--quick', action='store_true')
```

## v9_tiny/code/q1_contracts.py

209 lines; SHA-256 `c4052cb2e41597ebd4482a90a282faefa0054c7aefe5710fd980f34cd4eb40c1`.

Primary copy: [v9_tiny/code/q1_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py>).

Q1 preflight contracts (draft §13.1 items 11 and 13, plus testbed integrity and V8's own contract suite). Runs from a temporary working directory, writes smoke outputs only there, and fails if any file under any project runs/ directory changes during the session (item 13). On success it writes spec/LOCK_Q1.json and tests/q1_contracts.json; on failure it writes only tests/q1_contracts.json.

Imports:

```python
import copy, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import v9paths
from v9paths import V9, PROJECT, V8, RUNS, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json
import torch
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:18>) — lines 18–21
- [runs_snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:24>) — lines 24–34
- [identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:37>) — lines 37–44
- [frozen_integrity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:47>) — lines 47–61
- [fresh_sealed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:64>) — lines 64–101: Contract 11 (S1 part): chunk alignment, reference answers verify, disjointness, training exclusion.
- [seed811](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:104>) — lines 104–107
- [compile_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:110>) — lines 110–126: V8 stage_checks' eager-vs-compiled gradient parity for stage A (hierarchy and flat): decides --compiled as V8 did.
- [driver_args](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:129>) — lines 129–139
- [smoke](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:142>) — lines 142–160: Production-path smoke on the frozen train.py: compiled hierarchy and flat updates, then interrupted-and-resumed training.
- [eval_smoke](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:163>) — lines 163–169
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_contracts.py:172>) — lines 172–205

## v9_tiny/code/q1_smoke.py

56 lines; SHA-256 `8c26ef0a7f2de11ef420aa09275e544adfdda6d80f1d24b61fd1f5242afda803`.

Primary copy: [v9_tiny/code/q1_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_smoke.py>).

End-to-end smoke of the Q1 driver (20-update runs, 64-item evaluations, a temporary runs directory).

Imports:

```python
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import V9, CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q1_smoke.py:9>) — lines 9–52

## v9_tiny/code/q2_contracts.py

187 lines; SHA-256 `4732aaebc09c422ecd746c8d8725ab8a831c46bdd0831c3d181a4cb8cfcc1cc7`.

Primary copy: [v9_tiny/code/q2_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py>).

Q2 preflight contracts (draft §13.1 items 11 (colour-count part), 13, 14 for S6-base), then LOCK_Q2. CPU.

Imports:

```python
import re
import copy, json, os, sys, tempfile, time, traceback
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import CODE, SPEC, TESTS, DATA, read_json, write_json, digest
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q2 import Q2_FILES
from runtime import load_system
from model import freeze
from generation import generate
from data import Mixer, verify, gen_symbolic, parse_prompt
import lsl, s6, v9data
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:23>) — lines 23–26
- [base](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:29>) — lines 29–30
- [randomize](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:33>) — lines 33–37
- [colourcount](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:40>) — lines 40–83: Item 11 (colour-count): unique header, dispatch and verifier reject every V8 format and accept every item; splits disjoint.
- [gates](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:86>) — lines 86–145: Item 14: forced open = plain LoRA; closed = pre-phase model; each gate reads its own unit's input; smoothing resets at documents.
- [engine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:148>) — lines 148–160: S6 engine (full recompute) equals frozen generate() on the anchor, every V8 dev suite S6 measures.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_contracts.py:163>) — lines 163–183

## v9_tiny/code/q2_smoke.py

46 lines; SHA-256 `b11075cfc6a5a7ef37d7c272cc145aebde707de57dcb1c72c0a6a56e0ae1d761`.

Primary copy: [v9_tiny/code/q2_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_smoke.py>).

End-to-end smoke of the Q2 driver (10-update phases, capped suites, 4-batch GMM fits) in a temporary runs directory, plus a restart that verifies all completed work without retraining.

Imports:

```python
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q2_smoke.py:10>) — lines 10–42

## v9_tiny/code/q3_contracts.py

264 lines; SHA-256 `bf5b06e421a77f96cc230e571b05079b9acdb9fa25eaf67a559d34743b28f2a0`.

Primary copy: [v9_tiny/code/q3_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py>).

Q3 preflight contracts (draft §13.1 items 1-5, 8, 9 and 13 for S2), then LOCK_Q3. Items 1, 3, 4 run deterministically on CPU in a subprocess (s2_contracts.py). Item 5 here covers the S2 inference paths (greedy selector decoding, decoder-free transitions); its value/terminal-head mutants belong to Q5.

Imports:

```python
import re
import copy, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import V9, CODE, SPEC, TESTS, DATA, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q3 import Q3_FILES
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:22>) — lines 22–25
- [cpu_contracts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:28>) — lines 28–36
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:39>) — lines 39–47
- [engine_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:50>) — lines 50–62: The H-NCP engine at K = 1 reproduces frozen generate() token for token (standard, lookup, hard14).
- [stop_semantics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:65>) — lines 65–93: Item 8: a forced DONE chunk verifies iff the current node is the goal after >= 1 hop (data.py:85); classify() agrees.
- [closed_loop_reference](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:96>) — lines 96–123: §6.5 rollout: subset-row implementation equals a slow per-row sequential reference on the full batch.
- [restart_rule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:126>) — lines 126–141
- [hindsight_mutants](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:144>) — lines 144–199: Item 5 (S2 paths): inference is invariant to hidden demonstrations; mutants that read them are caught.
- [compile_parity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:202>) — lines 202–234: Compiled heads/decode equal eager: outputs, and gradients of a smooth probe loss through both functions (V8 tolerance). The full S2 objective contains discrete argmin choices that may flip on float-level ties, so its loss is compared at a l
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_contracts.py:237>) — lines 237–260

## v9_tiny/code/q3_smoke.py

57 lines; SHA-256 `05e9f878a769c26e445dca532b026df522253842f1e068d725bf12b582f46fdd`.

Primary copy: [v9_tiny/code/q3_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_smoke.py>).

End-to-end smoke of the Q3 driver: 40-update S2 arms, 20-update replays, quick measurements, register, promotion, an interrupted-and-resumed arm, and a restart that verifies everything without retraining. Temporary runs directory.

Imports:

```python
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q3_smoke.py:10>) — lines 10–53

## v9_tiny/code/q4_contracts.py

146 lines; SHA-256 `84095c5ddb09213c070ce6473d8be50205f95c8eaf12b702abc8861b8d262b1e`.

Primary copy: [v9_tiny/code/q4_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py>).

Q4 preflight contracts (draft §13.1 item 2 and the S3 ownership / distillation / data-order rules of §7.2, item 13), then LOCK_Q4. CPU, on a K = 8 S2 fixture built here from the seed-811 teacher.

Imports:

```python
import re
import copy, json, os, subprocess, sys, tempfile, time, traceback
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q4 import Q4_FILES
from runtime import load_system
from model import freeze
from data import Mixer
import hncp, s3_eval, train_s3, register_s3
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:23>) — lines 23–26
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:29>) — lines 29–36
- [transplant](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:39>) — lines 39–46
- [ownership](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:49>) — lines 49–69
- [distill_targets](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:72>) — lines 72–95: S3b targets come from the ORIGINAL teacher on its own encoder: moving the student encoder must not move them.
- [chunk](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:98>) — lines 98–100
- [g3_register](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:103>) — lines 103–114
- [Path_scratch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:117>) — lines 117–119
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_contracts.py:122>) — lines 122–142

## v9_tiny/code/q4_smoke.py

66 lines; SHA-256 `cdcdf749c20ced4b8c51f5958202aacaa52f34c333fac337f59a559ecdbb08f3`.

Primary copy: [v9_tiny/code/q4_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_smoke.py>).

End-to-end smoke of the Q4 driver on stand-in S2 outputs (a K = 8 fixture promoted on two seeds, fixture K1 measurements), 20-update S3 arms, quick measurements, G3 and the forward decision; then a restart that verifies all completed work without retraining. Temporary runs directory.

Imports:

```python
import json, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot
```

Symbols:

- [stand_in](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_smoke.py:12>) — lines 12–28
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q4_smoke.py:31>) — lines 31–62

## v9_tiny/code/q56_smoke.py

70 lines; SHA-256 `f8c6fd35dabf0b138ba12c245ddc1442e8035c39db6b2395342c9841251fe08d`.

Primary copy: [v9_tiny/code/q56_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q56_smoke.py>).

End-to-end smoke of the Q5 and Q6 drivers on stand-in upstream results (one promoted K = 8 fixture seed), tiny sizes, in one temporary runs directory; then a restart of both that verifies every completed stage without recomputation. Writes tests/q5_smoke.json and tests/q6_smoke.json (each under its own lock). Smoke runs never read sealed items.

Imports:

```python
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot
```

Symbols:

- [stand_in](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q56_smoke.py:12>) — lines 12–21
- [run_driver](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q56_smoke.py:24>) — lines 24–28
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q56_smoke.py:31>) — lines 31–66

## v9_tiny/code/q5_contracts.py

162 lines; SHA-256 `fff92281f82a37bf12510306f62f1d32da643a831d1daa8b11723430175c14ee`.

Primary copy: [v9_tiny/code/q5_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py>).

Q5 preflight contracts (draft §13.1 items 6, 12, 13; S4 label rules §8.2; I15/I16), then LOCK_Q5. CPU, K = 8 fixture.

Imports:

```python
import copy, json, os, sys, tempfile, time, traceback
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, DATA, read_json, write_json, digest
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q5 import Q5_FILES
from data import canonical, verify
import hncp, hgen, csearch, kernels, train_s4, baselines
import measure_s2 as M
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:19>) — lines 19–22
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:25>) — lines 25–31
- [failed_answer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:34>) — lines 34–39
- [item6](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:42>) — lines 42–63: GradCuit reward: R_grad = o - 1 gives a nonzero gradient on a failed answer and one step lowers its log-probability; the mutant R = o is a no-op.
- [item12](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:66>) — lines 66–80: Budget accounting: every verifier call is counted (verify is wrapped and compared with the counters).
- [labels](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:83>) — lines 83–97: tau labels come only from the verifier rule on grounded prefixes; held-out split disjoint; fresh problems exclude evaluation hashes.
- [metrics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:100>) — lines 100–113
- [baselines_check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:116>) — lines 116–123
- [batching](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:126>) — lines 126–137
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q5_contracts.py:140>) — lines 140–158

## v9_tiny/code/q6_contracts.py

156 lines; SHA-256 `846ebf0fd6bbba91f342f66e7ee99d50337434e3f938210b5b9078328418ee3a`.

Primary copy: [v9_tiny/code/q6_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py>).

Q6 preflight contracts (draft §13.1 items 7, 12, 13, 15 for S5; I17), then LOCK_Q6. CPU, K = 8 fixture.

Imports:

```python
import copy, json, os, sys, tempfile, time, traceback
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q6 import Q6_FILES
from data import COL0, DONE
import hncp, hgen, csearch, s5
import measure_s2 as M
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:19>) — lines 19–22
- [fixture](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:25>) — lines 25–34
- [item7](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:37>) — lines 37–87
- [item15](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:90>) — lines 90–106
- [token_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:109>) — lines 109–115
- [accounting](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:118>) — lines 118–131
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/q6_contracts.py:134>) — lines 134–152

## v9_tiny/code/qf_contracts.py

51 lines; SHA-256 `76c6d4ca1e39549dd04ba2312f325f10ff0189cd4763135fcf5c975deba038f1`.

Primary copy: [v9_tiny/code/qf_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_contracts.py>).

QF preflight contracts: the final-analysis statistics (one-sided margin tests, TOST, Holm within a stage, missing runs), then LOCK_QF. CPU only, synthetic data.

Imports:

```python
import json, os, sys, tempfile, time, traceback
import numpy as np
import v9paths
from v9paths import SPEC, TESTS, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_qf import QF_FILES
import final_analysis as F
from stats import holm
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_contracts.py:16>) — lines 16–19
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_contracts.py:22>) — lines 22–47

## v9_tiny/code/qf_smoke.py

49 lines; SHA-256 `9dc6f3c3d7a15ec5b01e6b6b032b7c34684c73171811ca4c40eb2f993427ef5e`.

Primary copy: [v9_tiny/code/qf_smoke.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_smoke.py>).

End-to-end smoke of the QF driver on stand-in results (a K = 8 fixture as every S4 arm, a zero LoRA adapter, synthetic S5 cells, synthetic S6 measures); sealed reads are replaced by dev reads in smoke mode.

Imports:

```python
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot
```

Symbols:

- [stand_in](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_smoke.py:12>) — lines 12–24
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/qf_smoke.py:27>) — lines 27–45

## v9_tiny/code/queue_base.py

132 lines; SHA-256 `dab17ead8c851ff248ba0f2da064965cc23466cf49d9208d0f4c7506b6966015`.

Primary copy: [v9_tiny/code/queue_base.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py>).

Shared detached-queue machinery for V9 blocks after Q1 (Q1's own driver is locked and unchanged).

Imports:

```python
import hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json
```

Symbols:

- [Stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:10>) — lines 10–11
  Methods: .
- [lock_identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:14>) — lines 14–19
- [Queue](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:22>) — lines 22–132
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:26>), [stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:32>), [log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:37>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:42>), [preflight](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:56>), [wait_for](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:66>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:76>), [train](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:98>), [once](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:117>), [execute](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/queue_base.py:122>).

## v9_tiny/code/register_s2.py

202 lines; SHA-256 `fd6aef6ef449d53052cbd4f687254b649d7d23a52c44bbb38c55ecf7f851a127`.

Primary copy: [v9_tiny/code/register_s2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py>).

S2 gate register (draft §12.3): zones, the asymmetric inconclusive rule, the seed-split rule, consequences. Inputs are measure_s2.py outputs for V9-K8 and V9-K1 of one core at one seed. Pooling over suites is declared in spec/PRE_REGISTRATION.md I11 (written before any S2 result).

Imports:

```python
import json
import numpy as np
import v9paths
from v9paths import read_json, write_json
import measure_s2 as M
```

Symbols:

- [boot_mean](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:15>) — lines 15–20
- [boot_ratio](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:23>) — lines 23–26
- [boot_text_diff](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:29>) — lines 29–33: Paired document bootstrap of NLL difference over both text families pooled.
- [zone](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:36>) — lines 36–47: pass: whole interval in pass zone; fail: whole interval in fail zone; else partial/inconclusive. pass_lo: pass zone is >= pass_lo; pass_hi: pass zone is <= pass_hi; fail_lo: fail zone is > fail_lo; fail_hi: fail zone < fail_hi.
- [worst](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:50>) — lines 50–52
- [pooled_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:55>) — lines 55–56
- [rows_for_seed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:59>) — lines 59–139
- [semantics](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:142>) — lines 142–152: NMI between code index and decoded colour (pooled), and per-code colour consistency across contexts.
- [combine](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:155>) — lines 155–171: Rule 4: pass needs both seeds to pass, fail needs both to fail; otherwise inconclusive (seed-dependent).
- [consequences](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:174>) — lines 174–194: Register consequences (§12.3, §6.8) from combined rows.
- [decide](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s2.py:197>) — lines 197–202: pairs: list of (k8_measure_path, k1_measure_path) per seed.

## v9_tiny/code/register_s3.py

52 lines; SHA-256 `68c878466697d3c317818dee7bff9c5407d9b2d451f28b30d7db530fb0ba3bad`.

Primary copy: [v9_tiny/code/register_s3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s3.py>).

G3 (register §12.3, draft §7.3), per S3 arm and seed, dev only, against the promoted S2 model of that seed. G2 rows of the S3 model are computed with register_s2.rows_for_seed against the same V9-K1 reference as in S2.

Imports:

```python
import numpy as np
import register_s2 as R
from register_s2 import boot_mean, boot_text_diff, zone
```

Symbols:

- [deploy_text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s3.py:10>) — lines 10–11
- [g3](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s3.py:14>) — lines 14–40: s3, s2: measure_s2 outputs (S3 arm, promoted S2 model); k1: that seed's V9-K1 measure (G2 reference).
- [decide](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s3.py:43>) — lines 43–52: per_arm: {arm: \[g3 per seed\]}. Seed rule (§12.3 rule 4): an arm passes G3 only if both seeds pass; a split takes the status quo (S2 forward). If several arms pass, S3a-dec is preferred (the predicted safe arm), then S3b.

## v9_tiny/code/register_s6.py

55 lines; SHA-256 `b4758742ee1a2b7681a0260580f423021a1556e81d92636bf84734d388a0a424`.

Primary copy: [v9_tiny/code/register_s6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s6.py>).

S6 register rows G6.0-G6.5, LSL credit and the P3 overlap prediction (draft §10.7, §12.3; definitions in I11).

Imports:

```python
import numpy as np
from register_s2 import boot_mean, boot_ratio
```

Symbols:

- [sym](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s6.py:8>) — lines 8–9
- [text_diff](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s6.py:12>) — lines 12–13
- [phase_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s6.py:16>) — lines 16–48: pre/post: measure_model outputs for one arm; others: post-phase outputs of 'open' and 'vq' arms (same phase, same seed).
- [h12](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/register_s6.py:51>) — lines 51–55

## v9_tiny/code/s1_eval.py

101 lines; SHA-256 `aa5e601b0cd69a7a31589ad725681fccf1b4b555630ba01e1fe704f9b801a2d7`.

Primary copy: [v9_tiny/code/s1_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s1_eval.py>).

S1 sealed evaluation and the S1 hypothesis tests (draft §5.3; procedure fixed in spec/S1_SEALED_PROCEDURE.json). Symbolic suites come from the fresh V9 sealed manifests (split 'v9test'). No fresh text documents exist, so sealed text uses V8's test split, which V8 evaluated; V9 models never trained on it.

Imports:

```python
import argparse, json, time
import numpy as np
import v9paths
from v9paths import RUNS, SPEC, read_json, write_json
import torch
from model import Core, freeze
from runtime import load_system, implementation_hash
from evaluate import symbolic_eval, text_eval
import stats
```

Symbols:

- [sealed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s1_eval.py:20>) — lines 20–33
- [load](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s1_eval.py:36>) — lines 36–37
- [analyze](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s1_eval.py:40>) — lines 40–91: H3, H4, H5 as pre-declared in spec/S1_SEALED_PROCEDURE.json; seed 811 reported per seed, never in the tests.

CLI arguments:

```python
ap.add_argument('--source')
ap.add_argument('--out')
ap.add_argument('--analyze', action='store_true')
```

## v9_tiny/code/s2_contracts.py

172 lines; SHA-256 `c4057f24c83619dd6b72cf24857449ae28f2a9fbf3e3700101404fdec403727a`.

Primary copy: [v9_tiny/code/s2_contracts.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py>).

S2 contracts (draft §13.1 items 1, 3, 4): K = 1 equivalence with V8 REFIT, the B_H code path, causality. Run on CPU (V8_DEVICE=cpu) so the comparison is deterministic; returns a list of check records.

Imports:

```python
import copy, json, math
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import read_json, SPEC
from model import D, HEADS, S, CODES, DenseCore, Core, chunk_info, positions, rotate, count
from runtime import DEVICE, seed_all, optimizer, lr_at, balanced, nll, ncp_weights
from data import Mixer
import train as v8train
import hncp, train_s2
```

Symbols:

- [teacher_path](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:18>) — lines 18–19
- [rel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:22>) — lines 22–23
- [k1_equivalence](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:26>) — lines 26–65: Contract 1: loss, every trainable gradient and one AdamW update equal frozen train.py REFIT to 1e-6 relative.
- [reference_block_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:68>) — lines 68–83: Slow per-(position, code) reference: query/residual from x_j + delta_k, keys/values from the unperturbed stream.
- [code_path](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:86>) — lines 86–124: Contract 3: B_H code path; zero code = plain block; batched = one at a time = slow reference; recurrent visits 1-5 untouched.
- [causality](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:127>) — lines 127–154: Contract 4: the code at chunk j changes logits only at positions >= 4j+3; later tokens cannot reach earlier plans.
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s2_contracts.py:157>) — lines 157–167

## v9_tiny/code/s3_eval.py

40 lines; SHA-256 `c0474d144e17b3b096e4b41f4832f170c935af4e1c687701d66914fb7bf7dfba`.

Primary copy: [v9_tiny/code/s3_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s3_eval.py>).

S3 transplant check (draft §7.2 item 1, contract 2): u = (1-lam) u_T + lam u_S, A = (1-lam) port_T(u_T) + lam A_S through System.plans, with the student's code-free path; full-recompute engine (cached generation bypasses System.plans).

Imports:

```python
import argparse, copy
import numpy as np
import torch
import v9paths
from v9paths import SMOKE, write_json
from model import System, freeze
from runtime import DEVICE, load_system
import hncp, s6
```

Symbols:

- [LamSystem](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s3_eval.py:15>) — lines 15–19
  Methods: [plans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s3_eval.py:18>).
- [transplant](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s3_eval.py:22>) — lines 22–23
- [curve](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s3_eval.py:26>) — lines 26–35

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
```

## v9_tiny/code/s4_eval.py

106 lines; SHA-256 `6e6159f843b77fb1fb41beb7dc12ade85f2eda563bbb6d4de3950afebf0df195`.

Primary copy: [v9_tiny/code/s4_eval.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_eval.py>).

S4/S5 support: dev greedy evaluation per arm, the nodes/s and iteration-time pilot (I15 sizing rule), and the GradCuit lr / norm-bound pilot on failed development answers (§8.3).

Imports:

```python
import argparse, json, time
import numpy as np
import torch
import v9paths
from v9paths import DATA, SMOKE, read_json, write_json
from runtime import DEVICE, sync
import hncp, hgen, csearch, train_s4, baselines, kernels
import measure_s2 as M
```

Symbols:

- [load_arm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_eval.py:13>) — lines 13–25
- [greedy_eval](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_eval.py:28>) — lines 28–40: mode 'codes' = selector-argmax plans (EI, REINFORCE, promoted reference); 'free' = code-free (token arms).
- [pilot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_eval.py:43>) — lines 43–75: Measured wall-clock for mini iterations (iteration-0 mode and decoder-free mode), nodes/s at frontier {1, 16, 64}, and greedy t_g; then the I15 sizing rule.
- [gradcuit_pilot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_eval.py:78>) — lines 78–92: lr in {1e-3, 1e-2, 1e-1} per norm-bound setting (bounded .5, unbounded) on failed dev answers (selector-argmax).

CLI arguments:

```python
g.add_argument('--source', required=True)
g.add_argument('--mode', required=True)
g.add_argument('--adapter')
g.add_argument('--out', required=True)
p.add_argument('--source', required=True)
p.add_argument('--H', type=int, default=1)
p.add_argument('--out', required=True)
k.add_argument('--source', required=True)
k.add_argument('--out', required=True)
```

## v9_tiny/code/s4_sealed.py

23 lines; SHA-256 `d1bc2a6bee28eed14761c9bd4ad92b4a48267c0ba08691b8516f9136d05f6510`.

Primary copy: [v9_tiny/code/s4_sealed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s4_sealed.py>).

Sealed greedy evaluation of one S4 arm (L3 procedure, I18): selector-argmax plans (EI, REINFORCE, reference) or code-free plans (RFT, GRPO), on the fresh sealed hard10 (512) and hard14 (256) manifests.

Imports:

```python
import argparse
import numpy as np
import torch
import v9paths
from v9paths import SMOKE, write_json
import s4_eval, s5_run, baselines
```

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--mode', required=True)
ap.add_argument('--adapter')
ap.add_argument('--out', required=True)
```

## v9_tiny/code/s5.py

178 lines; SHA-256 `fb6b4373264ff012b4cb2c62b985a836e19046742ff914f826ae3aefd394e075`.

Primary copy: [v9_tiny/code/s5.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py>).

S5 methods and budgets (draft §9.5-§9.6, I17). Budgets are multiples b of one greedy answer's wall-clock t_g, measured on the same batch (lockstep over a batch of items, so every method batches across items like greedy decoding). Search methods size their expansions per search adaptively from the measured time per expansion so that the expected wall-clock meets the budget; a hard stop at 2x the budget marks unfinished items as failures ('wall_clock'); overruns are reported. Methods: rh (RH(H), decoder-free), rh_grounded (mechanism comparison), g (one-tree G(R), G(1) grounded; IMAG = R None), token_tree (3 colours + STOP, code-free prior, grou

Imports:

```python
import math, time
import numpy as np
import torch
import v9paths
from data import DONE, COL0, N_COL, verify
from runtime import sync
import hgen, csearch, baselines, kernels
from hgen import CH, classify
```

Symbols:

- [TokenSearch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:22>) — lines 22–90: Token tree (§9.6): actions = the 3 colour tokens at a hop's first position plus STOP (the DONE chunk); prior = the decoder's softmax over those four tokens under code-free plans; each child forces its first token and completes the hop greed
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:27>), [code_matrix](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:30>), [ground_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:34>), [decode_free](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:45>), [decode_children](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:58>), [rollout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:69>), [grounding_rule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:89>).
- [greedy_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:94>) — lines 94–104: Reference greedy answers and t_g (wall per item on this batch).
- [run_search](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:107>) — lines 107–127: RH-type methods: adaptive expansions per step to fit budget_s (whole batch). Returns per-item records.
- [run_one_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:130>) — lines 130–137: G(R)/IMAG: a short probe tree measures time per expansion, then one tree per item with the expansions that fit.
- [run_best_of_n](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:140>) — lines 140–148
- [answer_loop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5.py:151>) — lines 151–178: AB-MCTS-A (§9.3): per item, Thompson sampling between GEN (a fresh inner RH episode with fresh root noise) and CONT (refine a failed answer with the kernel at a uniformly chosen committed chunk). Binary feedback; shared budget.

## v9_tiny/code/s5_fit_token_value.py

10 lines; SHA-256 `27f02dc91caddab7c0db2658c945666c1fdfee6f58b421e37337deb9d444f6ff`.

Primary copy: [v9_tiny/code/s5_fit_token_value.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_fit_token_value.py>).

Fit the token tree's value head on S4's executed trajectories with the chosen token decoder (§9.6).

Imports:

```python
import argparse
import torch
import v9paths
import s5_run
```

CLI arguments:

```python
ap.add_argument('--model', required=True)
ap.add_argument('--s4', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--token-adapter')
```

## v9_tiny/code/s5_run.py

136 lines; SHA-256 `7b26e4ba6509ccad93a81402719ca7c14ff884725b8c550bd8eb2d97e2b09e8d`.

Primary copy: [v9_tiny/code/s5_run.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py>).

One S5 cell: (model, method, suite, budget multiple) -> per-item records (draft §9.6, I17). Concept methods use the S4 model (trained selector, versioned heads, register policy). Token methods use the chosen token decoder (strongest S4 token arm, or the promoted decoder) in code-free mode. Budgets are b x t_g, with t_g the promoted model's batched greedy wall-clock per item on the same batch (measured once per suite and seed, cached).

Imports:

```python
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import DATA, SMOKE, read_json, write_json
from runtime import DEVICE, atomic_save
import hncp, hgen, csearch, s5, baselines, train_s4
from hgen import Item, CH
```

Symbols:

- [suite_items](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:19>) — lines 19–21
- [concept_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:24>) — lines 24–30
- [token_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:33>) — lines 33–39
- [fit_token_value](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:43>) — lines 43–62: Token tree value head: 128 -> 1 on the decoder's final-layer state at the last grounded position, fitted with L_V on S4's executed trajectories and outcomes (every executed prefix labelled with its trajectory's outcome).
- [tg_for](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:65>) — lines 65–72
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s5_run.py:75>) — lines 75–132

CLI arguments:

```python
ap.add_argument('--model', required=True, help='promoted forward checkpoint (S3 or S2)')
ap.add_argument('--s4', help='S4 expert-iteration folder')
ap.add_argument('--token-adapter')
ap.add_argument('--method', required=True)
ap.add_argument('--split', default='dev')
ap.add_argument('--suite', required=True)
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--mult', type=float, required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--H', type=int, default=1)
ap.add_argument('--R', default='1')
ap.add_argument('--tt-leaf', default='rollout')
ap.add_argument('--tt-value')
ap.add_argument('--node-match', type=float, help='fixed expansions per step (mechanism comparison, node-matched rerun)')
ap.add_argument('--kernel', default='gradcuit')
ap.add_argument('--lr', type=float, default=1e-2)
ap.add_argument('--tg-cache', required=True)
```

## v9_tiny/code/s6.py

252 lines; SHA-256 `b227cbc3da0641eaa678e510f9d3997b591569eda3025ac0d737c1b6d7a22f60`.

Primary copy: [v9_tiny/code/s6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py>).

S6-base (draft §10.3-§10.7): phase training for LSL-scaled / always-open / VQ-only, G6.0, per-phase measurements. Evaluation uses full recomputation for every model (pre- and post-phase alike), so paired comparisons share one engine; at zero banks it reproduces frozen generate() (contract 14 suite).

Imports:

```python
import argparse, json, math, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, RUNS, SMOKE, digest, read_json, write_json, files_hash, CODE
from model import chunk_info, shift, freeze, count, Core
from runtime import DEVICE, sync, seed_all, atomic_json, Log, lr_at, nll, balanced, load_system, implementation_hash as frozen_hash
from data import Mixer, fixed_text, PAD, DONE, T, parse_prompt
from evaluate import text_eval
import lsl, v9data
from v9data import dispatch_verify, is_colourcount, phase_mixer
```

Symbols:

- [last_logits](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:30>) — lines 30–43
- [greedy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:47>) — lines 47–62
- [suite_arrays](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:65>) — lines 65–68
- [symbolic](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:71>) — lines 71–78: Greedy success per item, in item order; items are batched by prompt length (colour-count prompts vary).
- [text](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:81>) — lines 81–83
- [calibration_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:86>) — lines 86–96: Old-task inputs from the dedicated calibration split (never development or sealed).
- [function_kl](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:100>) — lines 100–107: Token-level KL(pre || post) on old-task inputs (paper App. H), mean over valid next-token positions.
- [phase_batches](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:111>) — lines 111–112
- [train_phase](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:115>) — lines 115–166
- [save_vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:169>) — lines 169–171
- [load_vq](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:174>) — lines 174–176
- [load_any](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:179>) — lines 179–183
- [g60](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:187>) — lines 187–200: Fit Phi_pos on phase data with the new adapter at zero; smoothed (rho = .3) hit rates on calibration inputs of every earlier phase and every foundation suite. Pass: mean <= 5% and no suite > 15%; 5-10% weakly separable; else not.
- [measure_model](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:204>) — lines 204–216: Every dev measurement a phase decision needs, as per-item vectors (paired across models).
- [telemetry](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6.py:219>) — lines 219–225: Gate telemetry of the newest bank (LSL): old- and new-task calibration inputs, per matrix with tails, joint all-open.

CLI arguments:

```python
t.add_argument('--source', required=True)
t.add_argument('--out', required=True)
t.add_argument('--arm', choices=['lsl', 'open', 'vq'], required=True)
t.add_argument('--mode', required=True)
t.add_argument('--phase', type=int, required=True)
t.add_argument('--steps', type=int, default=2000)
t.add_argument('--seed', type=int, required=True)
t.add_argument('--resume', action='store_true', help='a phase is short and fully seeded: an interrupted phase restarts from its source checkpoint')
g.add_argument('--source', required=True)
g.add_argument('--mode', required=True)
g.add_argument('--earlier', default='')
g.add_argument('--seed', type=int, required=True)
g.add_argument('--out', required=True)
m.add_argument('--source', required=True)
m.add_argument('--modes', default='')
m.add_argument('--out', required=True)
m.add_argument('--pre', default=None)
m.add_argument('--telemetry-mode', default=None)
m.add_argument('--earlier', default='')
```

## v9_tiny/code/s6_pilot.py

45 lines; SHA-256 `ed4ea9e7d57eb3569dc2760b3938eefbf05b75452f6140152e223e768f2a570b`.

Primary copy: [v9_tiny/code/s6_pilot.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6_pilot.py>).

S6 one-phase timing pilot (draft §10.7 budget): Phi_neg fit, training per update, Phi_pos fit and each old-task evaluation, timed separately on the run device, in a temporary directory. Writes tests/s6_pilot.json.

Imports:

```python
import json, sys, tempfile, time, argparse
import numpy as np
import torch
import v9paths
from v9paths import SPEC, TESTS, SMOKE, read_json, write_json
from runtime import DEVICE, sync, load_system
from model import freeze
from data import Mixer
import lsl, s6
```

Symbols:

- [timed](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6_pilot.py:16>) — lines 16–17
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/s6_pilot.py:20>) — lines 20–41

## v9_tiny/code/setup_testbed.py

112 lines; SHA-256 `e6814dca8a5cb1ad33dc9b53f670123accb1f6b206e93c4c07dbcece800048ed`.

Primary copy: [v9_tiny/code/setup_testbed.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/setup_testbed.py>).

Q0: freeze the V8 12-hour testbed by SHA-256 and draw V9's fresh sealed manifests. Run once. Refuses to overwrite an existing testbed. Writes only under v9_tiny/frozen and v9_tiny/spec.

Imports:

```python
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/setup_testbed.py:26>) — lines 26–31
- [copy_verified](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/setup_testbed.py:34>) — lines 34–39
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/setup_testbed.py:42>) — lines 42–108

## v9_tiny/code/stats.py

83 lines; SHA-256 `30b1a76875facfc0624088276cdf97d1d2343e9f909e93c8a117d3fa5626db04`.

Primary copy: [v9_tiny/code/stats.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py>).

V9 statistics (draft §5.3, §13.2): McNemar exact, paired item bootstrap, paired document bootstrap, Holm.

Imports:

```python
import numpy as np
from scipy.stats import binom
```

Symbols:

- [mcnemar_exact](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:8>) — lines 8–15: Two-sided exact McNemar test on paired binary outcomes a, b.
- [paired_bootstrap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:18>) — lines 18–29: Percentile interval of mean(a - b) over items; a and b may be (items,) or (seeds, items) for the pooled interval.
- [document_bootstrap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:32>) — lines 32–44: Paired document bootstrap of the NLL difference (a - b) on identical documents and targets.
- [paired_signflip](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:47>) — lines 47–55: Two-sided sign-flip randomization p-value for mean(a - b) over items; (seeds, items) arrays are averaged over seeds first.
- [holm](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:58>) — lines 58–63: Holm step-down adjusted p-values for a dict name -> p.
- [self_test](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/stats.py:66>) — lines 66–79

## v9_tiny/code/train_s2.py

269 lines; SHA-256 `bd8ddaa880d0e946155f446d521fb6ecf28e695430c4574729163cb8c2f7b7e5`.

Primary copy: [v9_tiny/code/train_s2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py>).

S2 training (draft §6.3-§6.5): H-NCP against a frozen S1 teacher, V8 REFIT ownership and optimizer. L_S2 = L_token(plan) + alpha * L_WTA + delta(u) * L_dist + lambda_sel * L_sel. At K = 1 with the closed loop off this is V8's REFIT objective term for term (contract 1).

Imports:

```python
import argparse, json, math, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import digest, files_hash, CODE, DATA, write_json
from model import chunk_info, freeze, count, S, CODES, D
from runtime import DEVICE, sync, seed_all, atomic_json, atomic_save, Log, lr_at, optimizer, nll, balanced, ncp_weights, implementation_hash as frozen_hash
from data import Mixer
import hncp
```

Symbols:

- [distill_weight](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:21>) — lines 21–26: V8 REFIT schedule (train.py:128): 1 for the first 10%, linear to 0 by 30%.
- [snap](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:29>) — lines 29–34: Replace each PQ segment with its nearest codebook entry (closed-loop 'snap' arm).
- [kstar_prediction](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:37>) — lines 37–43: Committed prediction at source positions: chat^(k*) with k* = argmin distance to the true next concept.
- [choose_spans](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:46>) — lines 46–64: Per sequence with probability p_cl: a scored source chunk j0 uniformly, N ~ U{1..4}, truncated at the document end.
- [closed_loop_inputs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:67>) — lines 67–91: §6.5: replace C_j for j0 < j <= j0+N with the committed prediction from position j-1, sequentially. The core is causal and document-isolated per row, so each rollout step runs the core only on the rows that have a span, truncated to the lo
- [objective](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:94>) — lines 94–167: cpu: the same batch on the CPU, used only to draw closed-loop spans and sampling noise without a device sync.
- [flush_wins](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:170>) — lines 170–172
- [restart_dead](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:175>) — lines 175–191: §6.4: dead if win share over the last 500 updates < 1/(10K); split the most-winning code; zero optimizer state.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s2.py:194>) — lines 194–265

CLI arguments:

```python
ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--core', default='dense', choices=['dense', 'recurrent'])
ap.add_argument('--k', type=int, default=1)
ap.add_argument('--m', type=int, default=2)
ap.add_argument('--p-cl', type=float, default=.25)
ap.add_argument('--variant', default='detached', choices=['detached', 'undetached', 'snap'])
ap.add_argument('--steps', type=int, default=12000)
ap.add_argument('--seed', type=int, default=911)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--context', type=int, default=192)
ap.add_argument('--lr', type=float, default=.001)
ap.add_argument('--alpha', type=float, default=.1)
ap.add_argument('--eps', type=float, default=.05)
ap.add_argument('--lam-sel', type=float, default=.1)
ap.add_argument('--seconds', type=float, default=43200)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--no-distill', action='store_true')
ap.add_argument('--compiled', action='store_true')
```

## v9_tiny/code/train_s3.py

122 lines; SHA-256 `48ff4afad324f52a7265c64163a580715703b7898246ead10cdc0fd77fb87556`.

Primary copy: [v9_tiny/code/train_s3.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s3.py>).

S3 continued pretraining (draft §7.2) from the promoted S2 model. Arms: dec (S3a-dec: decoder blocks, nf, gamma), enc (S3a-enc, optional: + encoder, enc_norm, tied embedding), joint (S3b: everything; WTA and selector stay on; distillation at a constant delta = .1 to the ORIGINAL teacher's code-free outputs computed from the teacher's own encoder on the same tokens). The S2 objective is reused unchanged (train_s2.objective, distillation off); S3b adds its own distillation term.

Imports:

```python
import argparse, json, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import digest, files_hash, CODE, DATA, write_json
from model import chunk_info
from runtime import DEVICE, sync, seed_all, atomic_json, atomic_save, Log, lr_at, optimizer, balanced, load_system, implementation_hash as frozen_hash
from model import freeze
from data import Mixer
import hncp, train_s2
```

Symbols:

- [ownership](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s3.py:25>) — lines 25–33
- [teacher_distill](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s3.py:36>) — lines 36–45: S3b: student code-free (u0, A0, chat0) on its own clean concepts vs the original teacher on ITS encoder.
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s3.py:48>) — lines 48–118

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--arm', required=True, choices=['dec', 'enc', 'joint'])
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--p-cl', type=float, default=.25)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--seconds', type=float, default=43200)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--compiled', action='store_true')
```

## v9_tiny/code/train_s4.py

293 lines; SHA-256 `20429ec9e57b93e05cf1ff3b5537fd6d9e507597c4304049b6a8b359779c2268`.

Primary copy: [v9_tiny/code/train_s4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py>).

S4 expert iteration over codes (draft §8.2): receding-horizon search -> root visit distributions and verified outcomes -> selector pi, value V_psi, terminal tau_psi. Everything else frozen. Heads versioned per iteration and validated against the register's G4 rows before use (fallbacks per I15/I16).

Imports:

```python
import argparse, copy, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, SMOKE, digest, read_json, write_json
from runtime import DEVICE, atomic_save
from data import gen_symbolic
import hncp, hgen, csearch
from hgen import Item, CH, stop_legal
```

Symbols:

- [fresh_problems](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:20>) — lines 20–29: Fresh graph problems from the frozen generator, half standard, half hard10; evaluation hashes excluded.
- [auroc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:33>) — lines 33–40
- [calib](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:43>) — lines 43–48
- [v_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:51>) — lines 51–61
- [records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:65>) — lines 65–72
- [complete_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:76>) — lines 76–81: u0 at the complete decoded answer of each problem (G4-V complete).
- [imagined_pairs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:85>) — lines 85–102: Depth-d imagined states along each executed code sequence from each executed root t, paired with the executed state t+d (G2.3 pairing): returns (u0_imagined, o, stop_label_of_executed_counterpart).
- [fit_linear](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:106>) — lines 106–110
- [fit_iteration](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:113>) — lines 113–129: L_pi on root visit distributions (temperature 1), L_V on (root u0, o), L_tau on grounded states.
- [validate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:133>) — lines 133–153: G4 rows on held-out problems (heads on CPU; states recomputed on the engine device where needed).
- [leaf_policy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:156>) — lines 156–161: Register consequences: V used (value leaves, decoder-free children) only if complete passes (or flagged), intermediate passes and every imagined depth <= H passes; otherwise the iteration falls back to verified grounded completion at iterat
- [search_worker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:164>) — lines 164–188: Run complete RH steps, then hand CPU records back before MPS graphs grow. Recycling a process clears Metal's process-local shape/graph caches, which torch.mps.empty_cache() in Torch 2.10 cannot fully release. Problem RNG state, visits, lab
- [solve_in_workers](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:191>) — lines 191–232: Keep the same frontier and search while bounding process-local GPU caches.
- [solve_batched](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:235>) — lines 235–242
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/train_s4.py:245>) — lines 245–286

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--H', type=int, default=1)
ap.add_argument('--budget', type=int, default=32)
ap.add_argument('--iterations', type=int, default=4)
ap.add_argument('--n0', type=int, default=512)
ap.add_argument('--n', type=int, default=4096)
ap.add_argument('--frontier', type=int, default=64)
ap.add_argument('--rh-allowed', type=int, default=1)
ap.add_argument('--resume', action='store_true', help='resume is automatic from state.json (completed iterations are reloaded)')
```

## v9_tiny/code/v9data.py

137 lines; SHA-256 `6fa6d06da01f4fbd32d6dcb7f0618962d77f8d194ef0b0c5975b3fe7d166bbc7`.

Primary copy: [v9_tiny/code/v9data.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py>).

V9 data additions (draft §10.4-§10.5): the colour-count task and the S6 calibration split. The frozen V8 generator, verifier and splits are untouched; colour-count has its own parser dispatch, verifier and manifest. Colour-count: header \[GNAV, COL c*, DONE, SEP\]; body of L colour tokens (L in 8, 12, ..., 40), uniform over 3 colours; response \[NODE0 + count(c*), DONE, SEP, SEP\] with count <= 23 (resampled otherwise); response-only loss.

Imports:

```python
import hashlib, json
import numpy as np
import torch
import v9paths
from v9paths import DATA, SPEC, digest, read_json, write_json
from data import PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, gen_symbolic, parse_prompt, verify, Mixer, Text, fixed_text
```

Symbols:

- [is_colourcount](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:23>) — lines 23–25
- [parse_cc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:28>) — lines 28–34: Colour-count parser; raises on any other format (including every V8 task).
- [verify_cc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:37>) — lines 37–40
- [dispatch_verify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:43>) — lines 43–45: Task dispatch: colour-count by its unique header, every other prompt through the frozen V8 verifier.
- [canonical_cc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:48>) — lines 48–50
- [gen_cc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:53>) — lines 53–63
- [build](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:66>) — lines 66–93: Colour-count dev/calibration/test manifests (content-hash disjoint) and the calibration split of V8 tasks.
- [save](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:96>) — lines 96–97
- [load_npz](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:100>) — lines 100–101
- [CCMixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:104>) — lines 104–129: Phase-data mixer for colour-count with the frozen Mixer's interface; excludes dev/calibration/test items.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:106>), [batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:111>), [state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:123>), [restore](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:126>).
- [phase_mixer](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9data.py:132>) — lines 132–133

## v9_tiny/code/v9paths.py

46 lines; SHA-256 `285ebaac7e32d43a2438a51bac7fd4aa2239d1fd5b1e37f711c481614d098178`.

Primary copy: [v9_tiny/code/v9paths.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9paths.py>).

Paths shared by V9 code. The frozen V8 testbed is imported from v9_tiny/frozen/code.

Imports:

```python
import hashlib, json, os, sys
from pathlib import Path
```

Symbols:

- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9paths.py:21>) — lines 21–26
- [files_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9paths.py:29>) — lines 29–34: SHA-256 over (name, bytes) of an explicit file list; used by per-block locks.
- [read_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9paths.py:37>) — lines 37–38
- [write_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/code/v9paths.py:41>) — lines 41–46

## v9_tiny/monitor/audit_q5_memory_inputs.py

71 lines; SHA-256 `8f8524be571a8e10400298c519fe95077cfe09cb148b7644b891d911ca2168bd`.

Primary copy: [v9_tiny/monitor/audit_q5_memory_inputs.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/audit_q5_memory_inputs.py>).

Byte-hash data, completed work, and pending lock inputs without loading sealed data.

Imports:

```python
import datetime
import hashlib
import json
from pathlib import Path
import sys
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/audit_q5_memory_inputs.py:12>) — lines 12–13
- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/audit_q5_memory_inputs.py:16>) — lines 16–21
- [files_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/audit_q5_memory_inputs.py:24>) — lines 24–28

## v9_tiny/monitor/collect_health.py

134 lines; SHA-256 `51766a6c4a2e75b8bdddfc9b99dee4e11806e9894c2c3fb6123bd38cb6000797`.

Primary copy: [v9_tiny/monitor/collect_health.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/collect_health.py>).

Read queue/process/lock health and append a durable record. Never restart jobs.

Imports:

```python
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/collect_health.py:16>) — lines 16–20
- [files_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/collect_health.py:23>) — lines 23–28
- [command](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/collect_health.py:31>) — lines 31–32
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/collect_health.py:35>) — lines 35–119

CLI arguments:

```python
ap.add_argument('--action', default='Health check; no recovery action performed.')
ap.add_argument('--notification', default='No new actionable change; stay quiet.')
```

## v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py

555 lines; SHA-256 `a79d096f28ecc5f110509f4d85d4c2f969697426203804232362abbdfbc9fd26`.

Primary copy: [v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py>).

S2 measurements (draft §6.6): everything the register rows G2.1-G2.4 read, per model, with per-item vectors. Development data only. One grounded state per item per distribution, so bootstrap over items is bootstrap over states. Thresholds are not here; they are in register_s2.py (draft §12.3).

Imports:

```python
import argparse, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, RUNS, write_json, read_json, digest
from model import D, chunk_info, shift, Core, freeze
from runtime import DEVICE, nll, ncp_weights, balanced, load_system
from data import PAD, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, fixed_text
import hncp, hgen, ceiling
from hgen import CH, Engine, Item, classify
```

Symbols:

- [load_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:22>) — lines 22–24
- [tf_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:29>) — lines 29–53: Per-row (sum NLL, targets) under a plan mode: codefree | oracle | argmax | sample. Also k*, pi at scored chunks.
- [mixture_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:57>) — lines 57–80: Per-chunk mixture log sum_k pi_k p(chunk | k), other chunks' codes at selector-argmax (exact per chunk).
- [nll_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:83>) — lines 83–117
- [collect_calibration](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:120>) — lines 120–126
- [ece](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:129>) — lines 129–134
- [calibration_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:137>) — lines 137–147
- [eval_batch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:151>) — lines 151–153: I20: MPS driver memory grows with batch x K x distinct lengths; K > 1 decodes score K code rows per sequence, so batch 16 (plateau 2 GB on hard14) where K = 1 keeps 64 (I19).
- [greedy_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:156>) — lines 156–163
- [self_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:166>) — lines 166–181: D_self: selector-argmax rollouts up to the demonstration's response length or DONE; one legal-history state per item.
- [gold_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:184>) — lines 184–193
- [wrong_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:196>) — lines 196–208: D_wrong: a D_self state after one forced illegal or non-progress hop (claimed node becomes current).
- [fill_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:211>) — lines 211–226: Plan records for prefixes not produced by a rollout: selector-argmax at every coded chunk (causal, one pass).
- [eng_P](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:229>) — lines 229–230
- [decode_codes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:235>) — lines 235–255: For each state and code k: the decoded chunk, its action class, and (unless port0) the pooled real concept.
- [candidate_concepts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:259>) — lines 259–269: 73 candidate chunks (3 colours x 24 nodes + DONE), each encoded in the state's own context and pooled.
- [g24](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:272>) — lines 272–307: Per-state G2.4 quantities. heads: per-state chatk (K, D) and pi (K,) at the state's last chunk.
- [state_head_values](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:311>) — lines 311–321
- [g23](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:326>) — lines 326–378: Code sequence of length n_max fixed in advance (argmax along the imagined path, or uniform); imagined vs executed.
- [g23_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:381>) — lines 381–388
- [open_loop_nll](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:393>) — lines 393–404: E5 form: every chunk after c0 replaced by the model's own committed prediction, decoder reads true tokens. Codes are used only at chunks with a scored next chunk (§6.2); elsewhere the code-free prediction is fed.
- [drift_suite](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:407>) — lines 407–447: E5-form open-loop NLL increase for this model, and code-free drift of u0/A0 against the frozen teacher.
- [engaged_depth](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:452>) — lines 452–471
- [pblur2](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:475>) — lines 475–490
- [measure](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/measure_s2.py:494>) — lines 494–549

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--quick', action='store_true')
```

## v9_tiny/monitor/probe_i22.py

87 lines; SHA-256 `bea36ae16eea970d9579d7c6e3d4f57712fb01d1873333ac85df99b008ff92fe`.

Primary copy: [v9_tiny/monitor/probe_i22.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/probe_i22.py>).

Scratch-only full evaluation to diagnose recurrent MPS evaluation batching. Does not edit locked code or write production results. A singleton guard prevents multiple probes from competing for MPS. Progress and MPS memory are durable.

Identical copies:

- [v9_tiny/monitor/i22_validation_20261009T072329Z/probe_i22.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/probe_i22.py>)

Imports:

```python
import argparse
import datetime
import fcntl
import functools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/probe_i22.py:23>) — lines 23–83

CLI arguments:

```python
ap.add_argument('--batch', type=int, default=16)
```

## v9_tiny/monitor/revalidate_i22.py

255 lines; SHA-256 `bc4379b4ca320afc3eceff6b9088ae88d72144020ffe0e0389deacdc69a6f905`.

Primary copy: [v9_tiny/monitor/revalidate_i22.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py>).

Validate the recurrent evaluation memory repair, then resume the existing queue. Waits for the isolated full-size MPS probe before editing evaluation batching. Each affected pending block resumes only after its existing contracts and smoke pass under its current hashes. Completed Q2's original lock/results are archived and retained; no completed training or measurement is recalculated.

Identical copies:

- [v9_tiny/monitor/i22_validation_20261009T072329Z/revalidate_i22.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/i22_validation_20261009T072329Z/revalidate_i22.py>)

Imports:

```python
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
```

Symbols:

- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:28>) — lines 28–29
- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:32>) — lines 32–33
- [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:36>) — lines 36–40
- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:43>) — lines 43–48
- [processes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:51>) — lines 51–58
- [check_cancel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:61>) — lines 61–63
- [stop_waiter](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:66>) — lines 66–77
- [completed_snapshot](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:80>) — lines 80–86
- [verify_inputs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:89>) — lines 89–106
- [run_stage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:109>) — lines 109–124
- [resume](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:127>) — lines 127–169
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_i22.py:172>) — lines 172–251

## v9_tiny/monitor/prelaunch_q2.py

57 lines; SHA-256 `96cabb5bcda9f2cd00f584ad9ee35c35b800ccaad489f85f1fb0cb32c72ead57`.

Primary copy: [v9_tiny/monitor/prelaunch_q2.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/prelaunch_q2.py>).

One-off chain that finishes the I21 restart of Q2, unattended: 1. wait for the scratch S6 pilot (production shape, edited s6.py); it must finish and write its estimate; 2. check that the Q2 contract tests passed under the current lock (they were rerun after the edit); 3. run the Q2 smoke test under that lock (the launcher refuses without it); 4. launch the Q2 block with code/launch_block.py. It stops at the first failure and writes an alert; nothing after a failed step runs. After step 4 the watchdog (watch.py) keeps Q2 going.

Imports:

```python
import json, os, subprocess, sys, time
from pathlib import Path
```

Symbols:

- [log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/prelaunch_q2.py:14>) — lines 14–16
- [alive](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/prelaunch_q2.py:19>) — lines 19–23
- [fail](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/prelaunch_q2.py:26>) — lines 26–29

## v9_tiny/monitor/python_startup.py

59 lines; SHA-256 `7e030f4bdb98f916f8074a80d166a8afa21da9fe7d26a76ab23942a3c868a09c`.

Primary copy: [v9_tiny/monitor/python_startup.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/python_startup.py>).

Launch V9 without importing unrelated editable projects during site startup. Global Anaconda .pth files remain untouched. The normal site hooks and installed libraries are retained, except for editable source paths belonging to other projects. Child Python processes inherit this launcher through sys.executable.

Imports:

```python
import os
import site
import sys
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/python_startup.py:12>) — lines 12–55

## v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py

1106 lines; SHA-256 `bb61f512471acf90db3a2c87579401e08a82b294781c4f16351fbd044f36617b`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py:865>) — lines 865–1065
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py:866>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py:869>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py:875>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/2o/c2ocoyg4ahk63daptschz2jlbispq4obce5nwdjbmnlofij4qn5a.py:1072>) — lines 1072–1101

## v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py

1123 lines; SHA-256 `20b52df801226bbff206d8e1207748ae520643009435de8aa93014758b570555`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py:878>) — lines 878–1082
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py:879>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py:882>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py:888>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/3a/c3aogcjyugenw5o6ze3amcprofia7rwrnqcotavwlzddaszhivbc.py:1089>) — lines 1089–1118

## v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py

1048 lines; SHA-256 `a9d97bd6feb8d0cea943eb76c9a89d587e60b3ca8707841b9597561188662a88`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py:683>) — lines 683–970
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py:684>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py:687>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py:693>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4f/c4fkd2nyslkfijwjqow4irmawhyf445le46osiasfedqaf6z3au7.py:977>) — lines 977–1043

## v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py

109 lines; SHA-256 `6baf1582b30ccf005e5ee1e56d31322117cdda1e4b9ef6cae4ddd59d85a203be`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py:72>) — lines 72–91
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py:73>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py:76>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py:82>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/4y/c4y3opxjbsy2ljgug2hkortf4rgm4ad3pukg24oiunwhw32dxq4a.py:98>) — lines 98–104

## v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py

4393 lines; SHA-256 `25e169ca68f647e9defc9611044495f9d6cce1190bf26b0e9248be2e36bf07bb`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py:3315>) — lines 3315–4223
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py:3316>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py:3319>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py:3325>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/5i/c5i2n3jajctv6dza2seqeawfs26ehh4sryfdfmzyghpycodyghsb.py:4230>) — lines 4230–4388

## v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py

1342 lines; SHA-256 `267a07f2057f64427d09d472d0be69a6e3864a1b93feece9e0b4a8f80b0b2603`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py:916>) — lines 916–1258
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py:917>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py:920>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py:926>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/77/c7776w7yhibeqtskdohll2rsur7mnv6pivqvh4dcmrxux4fshd33.py:1265>) — lines 1265–1337

## v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py

2283 lines; SHA-256 `bdef7046d9393b21324eb819e1ed5dada55a28e9fc63bddd98778939dca32d8a`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py:1870>) — lines 1870–2242
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py:1871>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py:1874>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py:1880>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ak/cak34d2yevi4vz7brrwb7autghkdvhkwepi35ixhfu62tibvufe7.py:2249>) — lines 2249–2278

## v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py

2995 lines; SHA-256 `d65b92c49480c06bc5a859586105505aae0bef62a7d809413c0a00eb472803ec`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py:2150>) — lines 2150–2856
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py:2151>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py:2154>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py:2160>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/bl/cblscqj75awhio6wvuzayaupfnp7aupk4mjudulz3wvwjkw3jcsk.py:2863>) — lines 2863–2990

## v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py

2268 lines; SHA-256 `0e709f47f09a761c298a71c443eaaa06a87497d1dc587a1379c204b934608a39`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py:1871>) — lines 1871–2217
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py:1872>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py:1875>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py:1881>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/dn/cdnna5ckakkz43jllsk5653o5fwpecskjvcl6poa6ercrwntn5ba.py:2224>) — lines 2224–2263

## v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py

145 lines; SHA-256 `421347ceb44f221304d2a5872b9523d256464ad81681c774d9114db745343cfa`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py:112>) — lines 112–128
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py:113>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py:116>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py:122>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fc/cfc46eqqknzgxkaqz7thl2c72jnhs6smeouvoh7mmsocd545agme.py:135>) — lines 135–140

## v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py

1129 lines; SHA-256 `fcec4198efe82031ac1ea4d3cd6ed95a85732bcdad2fd90dc4805231ea81fa1a`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py:880>) — lines 880–1086
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py:881>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py:884>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py:890>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/fi/cfiagy6uxh6corno2evdjkvm3iidm6owxbhyjnnd754xckwkl2a6.py:1093>) — lines 1093–1124

## v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py

3169 lines; SHA-256 `0e9e8b8114d648b70c1c50f6a2523165d1d7a436f1db7f8993fcd9d1158268cd`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py:2371>) — lines 2371–3029
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py:2372>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py:2375>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py:2381>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/hu/chueuyd7ux5pjji76ekag7s53x3eeoiwc67avvaalklck45xjcaj.py:3036>) — lines 3036–3164

## v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py

2267 lines; SHA-256 `0d8fa1b20c176aa7417615344465da9e19a96369f6f457a658902f8c66d7d635`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py:1871>) — lines 1871–2216
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py:1872>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py:1875>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py:1881>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ps/cpslh64xbbpioctrb2k5im4s6bniey4bup2hptxhhq56e3itc5za.py:2223>) — lines 2223–2262

## v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py

1445 lines; SHA-256 `89bd573a9f9c1be7a1aac2b8bd40230276cfdf8990dba7270ccea6ae652424f7`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py:1133>) — lines 1133–1398
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py:1134>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py:1137>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py:1143>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/rv/crvs6apvwohokaeurwyohmeq6qibgxbqz2o7m5bjt3kzyrbj5dk4.py:1405>) — lines 1405–1440

## v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py

1042 lines; SHA-256 `bc28591263e65bbf6b20a2d9a26247fa6af406023c2a80d5118b96c14ae875a9`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py:681>) — lines 681–966
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py:682>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py:685>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py:691>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/t2/ct23wyetindohjs7mruvgiyl7uldfoo6m3mgizndy4da5x6o5rsb.py:973>) — lines 973–1037

## v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py

1505 lines; SHA-256 `839d84cedafb1bd14fc3228c6f1032e78c01968e43230852f4bc9e7d157d8fc4`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py:927>) — lines 927–1396
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py:928>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py:931>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py:937>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/tt/ctt7fgwfer2i7aymnzxnnro7l5gtmku4mbb4elev3va6ebj5hmiw.py:1403>) — lines 1403–1500

## v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py

3252 lines; SHA-256 `20c2d8ad52cd961677cb0d4abb59603e694688dc8980b2f9a3fac851ae43e103`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py:2446>) — lines 2446–3115
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py:2447>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py:2450>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py:2456>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/u2/cu2psc6hwlqhkmfd5ckmb5othufxpg2xpwbzujcp6mfpoeu2fuuz.py:3122>) — lines 3122–3247

## v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py

1127 lines; SHA-256 `dd932d8d401241246feb3eadec719ba9af5cb4a0fec81a24ccb3b98d3f729c14`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py:878>) — lines 878–1086
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py:879>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py:882>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py:888>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/vl/cvllx6r6ouqeojn2xtemvklcwkgoolmmjbxd4r7iw4e7ryd4nuxd.py:1093>) — lines 1093–1122

## v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py

3113 lines; SHA-256 `f35b318a1c23969f972b98da867de8b0db28d8c91da721374c018ea7f84b0046`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py:2614>) — lines 2614–3068
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py:2615>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py:2618>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py:2624>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/y7/cy7msqzzojaoxleigb7a4ookdmrm5ucl3qwvjtqxrb7piwwftqgd.py:3075>) — lines 3075–3108

## v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py

1397 lines; SHA-256 `554a84e8c6210051e8c10ebd63c544408f893e70e4311aae4bc6698f919dbd3b`.

Primary copy: [v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py>).

Imports:

```python
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
```

Symbols:

- [Runner](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py:961>) — lines 961–1311
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py:962>), [recursively_apply_fns](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py:965>), [call](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py:971>).
- [benchmark_compiled_module](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q4_compiler_cache/ys/cysxob2hdhbt7hil7ifrd6mzmy4gddmdeccv7xd3c6j65ukzupnn.py:1318>) — lines 1318–1392

## v9_tiny/monitor/q5_12h.py

591 lines; SHA-256 `f08e253676e1e01c521a53b234de17d6e915bc6da0a49a77b96350e0c9888489`.

Primary copy: [v9_tiny/monitor/q5_12h.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py>).

Isolated, deadline-bounded Q5 amendment. No original scientific output is modified. The stdlib supervisor waits for the running EI round, holds the original queue, then launches only validated, serial GPU workers. ML imports occur in workers.

Imports:

```python
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
```

Symbols:

- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:28>) — lines 28–29
- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:32>) — lines 32–33
- [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:36>) — lines 36–41
- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:44>) — lines 44–49
- [event](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:52>) — lines 52–55
- [procs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:58>) — lines 58–60
- [protocol](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:63>) — lines 63–64
- [identity](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:67>) — lines 67–113
- [preflight](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:116>) — lines 116–123
- [init](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:126>) — lines 126–159
- [snapshot_ei](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:162>) — lines 162–168
- [hold_original](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:171>) — lines 171–194
- [checkpoint_state](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:197>) — lines 197–204
- [clear_cache](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:207>) — lines 207–212
- [RecycleWorker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:215>) — lines 215–217: Saved operational handoff, not a failed scientific update.
  Methods: .
- [fit_data](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:220>) — lines 220–231
- [fit_resumable](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:234>) — lines 234–271
- [train_worker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:274>) — lines 274–342
- [worker](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:345>) — lines 345–392
- [paired](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:395>) — lines 395–398
- [report](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:401>) — lines 401–428
- [Supervisor](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:431>) — lines 431–543
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:432>), [stop](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:437>), [status](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:442>), [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:447>), [_run_worker_once](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:462>), [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:489>), [finish](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:540>).
- [detach](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:546>) — lines 546–559
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h.py:562>) — lines 562–587

CLI arguments:

```python
ap.add_argument('--init', action='store_true')
ap.add_argument('--detach', action='store_true')
ap.add_argument('--supervise', action='store_true')
ap.add_argument('--worker', choices=['calibrate', 'train', 'freeze', 'eval'])
ap.add_argument('--round', type=int, default=0)
ap.add_argument('--arm', choices=['EI', 'grpo'])
```

## v9_tiny/monitor/q5_12h_assess.py

96 lines; SHA-256 `7cd43b8bb237d9374b885303e1f320d8fb14b58a5958352fc9e113e47c80183c`.

Primary copy: [v9_tiny/monitor/q5_12h_assess.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_assess.py>).

Read-only scientific audit of the completed reduced Q5; writes monitor assessment only.

Imports:

```python
import collections
import math
import os
import sys
from pathlib import Path
import q5_12h as q
import v9paths
import numpy as np
import torch
import baselines, hncp, train_s4, hgen
from data import verify
from runtime import finite
```

Symbols:

- [pair](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_assess.py:18>) — lines 18–27
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_assess.py:30>) — lines 30–92

## v9_tiny/monitor/q5_12h_validate.py

192 lines; SHA-256 `bcb485e662c410cffabbef3e2cdf69b65434929740321b5ad04adab10415506c`.

Primary copy: [v9_tiny/monitor/q5_12h_validate.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py>).

CPU-only contracts and isolated end-to-end smoke for the reduced Q5 queue.

Imports:

```python
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import q5_12h as q
import numpy as np
import torch
import baselines
import hncp
import hgen
import train_s4
import q5_contracts as old
from runtime import atomic_save
from s4_eval import greedy_eval
```

Symbols:

- [check](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:29>) — lines 29–32
- [model_diff](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:35>) — lines 35–36
- [contracts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:39>) — lines 39–92
- [smoke](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:95>) — lines 95–138
- [queue_contracts](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:141>) — lines 141–162
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_12h_validate.py:165>) — lines 165–188

## v9_tiny/monitor/q5_memory_parity.py

66 lines; SHA-256 `3d69821eb5154f75af7bfefc3bd3932418de6490adb7e971b68ae299ff1b66be`.

Primary copy: [v9_tiny/monitor/q5_memory_parity.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_parity.py>).

Compare the memory repair against the archived search, with exact CPU records.

Imports:

```python
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
import csearch, hgen, hncp, train_s4
```

Symbols:

- [equal](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_parity.py:28>) — lines 28–37

## v9_tiny/monitor/q5_memory_probe.py

63 lines; SHA-256 `15a92f7fa8e273134fef2d4e78f6f0d659d8e6a2ebc60188f0172478e2bc702f`.

Primary copy: [v9_tiny/monitor/q5_memory_probe.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_probe.py>).

Isolated allocation/lifetime probe; never writes production run outputs.

Imports:

```python
import gc
import json
import os
from pathlib import Path
import sys
import time
import torch
import csearch
import hgen
import hncp
import train_s4
```

Symbols:

- [sample](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_probe.py:28>) — lines 28–34

## v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py

447 lines; SHA-256 `e61601be77ccff987f3eabd2d434d01a050a8bdde04c7f785b2b153b16d9a8aa`.

Primary copy: [v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py>).

Concept search (draft §9.3-§9.4, I15): RH(H) receding horizon, and one-tree search G(R) / IMAG, over K codes + STOP, lockstep over B problems, full recomputation through hgen.Engine primitives (no caches). Children are imagined (c~_{j+1} = chat_j^(k), one core pass from the nearest grounded ancestor's concepts) or grounded (decode the pending chunks under their stored plan records, re-encode). A plan record is immutable: once a chunk has been decoded from a record, every later decode of that prefix applies the same record (node.over). Leaf values: 'rollout' = verified greedy grounded completion with selector-argmax codes; 'value' = V_psi(u0).

Identical copies:

- [v9_tiny/monitor/q5_memory_repair_20261010T102447Z/csearch.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T102447Z/csearch.before.py>)
- [v9_tiny/monitor/q5_memory_repair_20261010T102847Z/csearch.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T102847Z/csearch.before.py>)

Imports:

```python
import time
import numpy as np
import torch
from torch import nn
import v9paths
from model import D
from data import DONE, SEP, verify
import hgen
from hgen import CH, Item, classify, stop_legal
```

Symbols:

- [Heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:26>) — lines 26–36: V_psi and tau_psi: 128 -> 1 each on the code-free u^(0) (§8.2), versioned per S4 iteration.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:28>), [value](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:32>), [term](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:35>).
- [Node](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:39>) — lines 39–48
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:43>).
- [Problem](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:51>) — lines 51–67
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:52>), [tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:59>), [code_array](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:62>).
- [Search](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:70>) — lines 70–443: cfg: H (depth cap; one-tree uses the hop limit), children 'imagined'|'grounded', ground_every R (one-tree G(R); None = IMAG), leaf 'rollout'|'value', budget (expansions per search), noise (root Dirichlet), heads, one_tree.
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:74>), [stop_prob](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:80>), [leaf_value](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:85>), [set_heads](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:88>), [ground_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:94>), [imagine_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:104>), [code_matrix](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:121>), [decode_over](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:128>), [f_encode](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:148>), [finish_child](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:151>), [decode_children](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:162>), [replay](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:177>), [verify_tokens](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:215>), [rollout](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:220>), [puct](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:254>), [grounding_rule](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:261>), [simulate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:268>), [make_roots](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:341>), [step](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:356>), [one_tree](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:399>), [solve](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:434>).
- [problems_from](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/csearch.before.py:446>) — lines 446–447

## v9_tiny/monitor/q5_memory_repair_20261010T101532Z/launch_block.before.py

32 lines; SHA-256 `b150ffcf0d410b91a98113d175b1d9cfb41812ec81e9d1c387af90541c6ca086`.

Primary copy: [v9_tiny/monitor/q5_memory_repair_20261010T101532Z/launch_block.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/launch_block.before.py>).

Start a Q2+ block driver detached (new session) with idle-sleep prevention. Refuses to start unless the block's contracts and smoke passed under the current lock, and refuses a duplicate of the same block.

Imports:

```python
import json, subprocess, sys, time
import v9paths
from v9paths import V9, RUNS, CODE, TESTS, SPEC, read_json, write_json
from queue_base import lock_identity
import importlib
```

Symbols:

- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/launch_block.before.py:10>) — lines 10–28

## v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py

216 lines; SHA-256 `25cbeccd7dc858f02141b970c1de4daf245ed1e330108840f8ba3fd60a30dd35`.

Primary copy: [v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py>).

S4 expert iteration over codes (draft §8.2): receding-horizon search -> root visit distributions and verified outcomes -> selector pi, value V_psi, terminal tau_psi. Everything else frozen. Heads versioned per iteration and validated against the register's G4 rows before use (fallbacks per I15/I16).

Identical copies:

- [v9_tiny/monitor/q5_memory_repair_20261010T102447Z/train_s4.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T102447Z/train_s4.before.py>)
- [v9_tiny/monitor/q5_memory_repair_20261010T102847Z/train_s4.before.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T102847Z/train_s4.before.py>)

Imports:

```python
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, SMOKE, digest, read_json, write_json
from runtime import DEVICE, atomic_save
from data import gen_symbolic
import hncp, hgen, csearch
from hgen import Item, CH, stop_legal
```

Symbols:

- [fresh_problems](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:20>) — lines 20–29: Fresh graph problems from the frozen generator, half standard, half hard10; evaluation hashes excluded.
- [auroc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:33>) — lines 33–40
- [calib](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:43>) — lines 43–48
- [v_row](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:51>) — lines 51–61
- [records](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:65>) — lines 65–72
- [complete_states](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:76>) — lines 76–81: u0 at the complete decoded answer of each problem (G4-V complete).
- [imagined_pairs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:85>) — lines 85–102: Depth-d imagined states along each executed code sequence from each executed root t, paired with the executed state t+d (G2.3 pairing): returns (u0_imagined, o, stop_label_of_executed_counterpart).
- [fit_linear](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:106>) — lines 106–110
- [fit_iteration](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:113>) — lines 113–129: L_pi on root visit distributions (temperature 1), L_V on (root u0, o), L_tau on grounded states.
- [validate](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:133>) — lines 133–153: G4 rows on held-out problems (heads on CPU; states recomputed on the engine device where needed).
- [leaf_policy](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:156>) — lines 156–161: Register consequences: V used (value leaves, decoder-free children) only if complete passes (or flagged), intermediate passes and every imagined depth <= H passes; otherwise the iteration falls back to verified grounded completion at iterat
- [solve_batched](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:164>) — lines 164–168
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/q5_memory_repair_20261010T101532Z/train_s4.before.py:171>) — lines 171–212

CLI arguments:

```python
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--H', type=int, default=1)
ap.add_argument('--budget', type=int, default=32)
ap.add_argument('--iterations', type=int, default=4)
ap.add_argument('--n0', type=int, default=512)
ap.add_argument('--n', type=int, default=4096)
ap.add_argument('--frontier', type=int, default=64)
ap.add_argument('--rh-allowed', type=int, default=1)
ap.add_argument('--resume', action='store_true', help='resume is automatic from state.json (completed iterations are reloaded)')
```

## v9_tiny/monitor/recover_python_startup.py

230 lines; SHA-256 `784e7e5a9cb130bdf4446b29c6a1bb06a134862f0b50aec5b6424e8785bf2ab4`.

Primary copy: [v9_tiny/monitor/recover_python_startup.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py>).

Validate a scoped Python startup repair, then resume the existing V9 queue. No scientific source, lock, official test result, or completed run is edited. The existing Q3 contracts and full smoke run on CPU with isolated outputs.

Imports:

```python
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback
```

Symbols:

- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:24>) — lines 24–25
- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:28>) — lines 28–29
- [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:32>) — lines 32–36
- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:39>) — lines 39–44
- [processes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:47>) — lines 47–54
- [check_cancel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:57>) — lines 57–59
- [run](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:62>) — lines 62–68
- [validation](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:71>) — lines 71–104
- [stop_waiter](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:107>) — lines 107–120
- [resume](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:123>) — lines 123–148
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_python_startup.py:151>) — lines 151–226

## v9_tiny/monitor/recover_q5_memory.py

265 lines; SHA-256 `e074686efa6dccbca572f2ebb9b3954303316f9a805cfbd398f686012eb9c0fe`.

Primary copy: [v9_tiny/monitor/recover_q5_memory.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py>).

Validate the allocation repair in isolation, then publish locks and resume once. The original scientific exposures, pilot sizing, and completed outputs are retained. This helper is protected by a file lock and keeps a durable stage/child heartbeat.

Imports:

```python
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
```

Symbols:

- [read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:23>) — lines 23–24
- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:27>) — lines 27–28
- [digest](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:31>) — lines 31–36
- [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:39>) — lines 39–43
- [processes](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:46>) — lines 46–52
- [check_cancel](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:55>) — lines 55–57
- [run_stage](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:60>) — lines 60–84
- [resume](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:87>) — lines 87–112
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/recover_q5_memory.py:115>) — lines 115–261

## v9_tiny/monitor/revalidate_q4.py

110 lines; SHA-256 `93c006d6a6122e02d178737a4a0708f00d79a2442c7435a540c3cef894682770`.

Primary copy: [v9_tiny/monitor/revalidate_q4.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_q4.py>).

Revalidate Q4's stale lock with its existing contracts and smoke test on CPU.

Imports:

```python
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
```

Symbols:

- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_q4.py:19>) — lines 19–20
- [write](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_q4.py:23>) — lines 23–27
- [files_hash](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_q4.py:30>) — lines 30–35
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/revalidate_q4.py:38>) — lines 38–106

## v9_tiny/monitor/watch.py

288 lines; SHA-256 `e1ab1e1506522f19a7b7321ce88b4412bdc08856835078304c38434dd1185112`.

Primary copy: [v9_tiny/monitor/watch.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py>).

V9 queue watchdog. Every 15 minutes it checks the block drivers (Q2, Q3, Q4, Q5, Q6, QF) and keeps the queue moving: * a driver that died (crash, SIGKILL, reboot) while its block is running or waiting is relaunched with code/launch_block.py, which refuses unless the block's lock and its contract and smoke tests still match; a block whose training child is still alive is never relaunched (that would start a second copy of the same run); * a block whose status is 'failed' is resumed the same way, at most MAX_ATTEMPTS times per stage with RETRY_GAP between tries, then it is left alone and an alert is raised (a deterministic failure would otherwi

Imports:

```python
import argparse, calendar, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
```

Symbols:

- [utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:42>) — lines 42–43
- [local](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:46>) — lines 46–47
- [epoch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:50>) — lines 50–54
- [read_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:57>) — lines 57–61
- [write_json](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:64>) — lines 64–65
- [log](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:68>) — lines 68–73
- [alive](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:76>) — lines 76–82
- [command_of](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:85>) — lines 85–87
- [project_procs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:90>) — lines 90–98: pid -> command line of every running process that executes a script from v9_tiny/code (drivers, trainers, measurers).
- [swap_info](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:101>) — lines 101–107
- [notify](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:110>) — lines 110–114
- [Watch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:117>) — lines 117–253
  Methods: [__init__](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:118>), [raise_level](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:123>), [note](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:128>), [relaunch](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:140>), [failure_reason](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:166>), [check_block](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:180>), [cycle](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:225>).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_tiny/monitor/watch.py:256>) — lines 256–284

CLI arguments:

```python
ap.add_argument('--once', action='store_true')
ap.add_argument('--act', action='store_true')
ap.add_argument('--detach', action='store_true')
ap.add_argument('--loop', action='store_true', help=argparse.SUPPRESS)
```

## v9_webapp/build.py

39 lines; SHA-256 `7c9cc495046d365b6bfc6a78fce3ecd27a77ea3736aee0e3847a6f4a9796e542`.

Primary copy: [v9_webapp/build.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/build.py>).

Inline src/ into one self-contained page: index.html (opens from disk, or served by serve.py) and v9_atlas.html (artifact body, no skeleton). python build.py \[--no-snapshot\] The page reads live telemetry from serve.py when it is reachable. Without a server it falls back to a labelled snapshot of the experiment's files taken at build time (window.V9_SNAPSHOT), so a page opened from disk or published elsewhere still shows real numbers, stamped with when they were taken.

Imports:

```python
import json, sys
from pathlib import Path
```

## v9_webapp/guide/build.py

96 lines; SHA-256 `1693c06739df183723440c1606da1615f6429a2363838bb9afdf73ed4d71c4ee`.

Primary copy: [v9_webapp/guide/build.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/guide/build.py>).

Build the V9 Guide. python3 build.py Inlines src/ into one self-contained page. guide_body.html artifact body (no <!doctype>/<head>; the Artifact skeleton adds those) ../v9_guide.html the same body in a full page, for opening from disk or serving locally Chapter files are src/ch_*.html (sorted), widgets src/viz_*.js (sorted), the 3D module src/three_guide.js. variants.js is read from ../src (the Atlas descriptor table) so an arm is defined once for both pages.

Imports:

```python
import json, sys, time
from pathlib import Path
```

Symbols:

- [_rows](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/guide/build.py:18>) — lines 18–30
- [_blocks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/guide/build.py:33>) — lines 33–39: mean of consecutive logged rows, n rows per point (rows are logged every 100 updates)
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/guide/build.py:42>) — lines 42–76: Real numbers from the experiment's own files (read-only). Embedded as window.GUIDE_DATA.

## v9_webapp/serve.py

272 lines; SHA-256 `dc9493814d8bc002b669bb8145bd5a630a0540735c359ce1149927a15f62a8e7`.

Primary copy: [v9_webapp/serve.py](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py>).

V9 Atlas telemetry server (read-only, loopback only). python serve.py \[--port 8765\] Serves the built page at / and one JSON endpoint, /api/telemetry, aggregated from the experiment's own files under v9_tiny/. Nothing is written, no path comes from the request, and every source is a fixed pattern: runs/status*.json runs/launch_*.json runs/driver*.json runs/G1.json runs/queue.jsonl runs/<run>/log.jsonl runs/<run>/config.json runs/<run>/development_step_*.json runs/<run>/train_summary.json runs/<run>/measure.json (greedy rates only) runs/<run>_dev/summary.json runs/<run>_trained_contracts/summary.json tests/q*_contracts.json tests/q*_smoke.jso

Imports:

```python
import argparse, json, os, re, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
```

Symbols:

- [_round](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:31>) — lines 31–32
- [_read](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:35>) — lines 35–39
- [_cached](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:42>) — lines 42–54: Parse a file once per (mtime, size). A log that is being appended is re-parsed when it grows.
- [_utc](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:57>) — lines 57–61
- [_series](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:64>) — lines 64–81
- [_dev_checks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:84>) — lines 84–91
- [_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:94>) — lines 94–101
- [_measure_summary](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:104>) — lines 104–107
- [_alive](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:110>) — lines 110–115
- [_runs](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:118>) — lines 118–138
- [_queue_events](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:141>) — lines 141–152
- [_blocks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:155>) — lines 155–169
- [_tests](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:172>) — lines 172–181
- [_locks](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:184>) — lines 184–185
- [_system](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:188>) — lines 188–206
- [collect](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:209>) — lines 209–213
- [Handler](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:216>) — lines 216–256
  Methods: [log_message](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:219>), [_host_ok](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:222>), [_send](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:226>), [do_GET](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:240>).
- [main](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v9_webapp/serve.py:259>) — lines 259–268

CLI arguments:

```python
ap.add_argument("--port", type=int, default=8765)
```
