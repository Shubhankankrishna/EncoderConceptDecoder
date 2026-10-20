# Native32 GRAM: paired 30M and 10M experiment

The predecessor run was stopped at step 17,085 at the user's request. Its final
resumable and inference checkpoints remain in `runs/20261018_24h_raw_gram/gram`.
No old checkpoints are used to initialize this experiment.

The target is 32 BERT subword IDs, conditioned on eight preceding native chunks
(256 subword IDs). These are tokenizer IDs, not a fixed number of words.

## Codec compatibility

This is an explicit adaptation of the pinned official LM1B NCM codec, not an
official pretrained 32-position checkpoint. Its hierarchy is
`[1,2,4,6,8,10,13,16,19,23,27,32]`, totaling 161 code IDs rather than 641. Each
code still has 8,192 choices. Root convolution kernels and decoder positions are
resized by a documented deterministic recipe; the decoder alone is then adapted
on 4,096 training chunks. Encoder, codebooks and quantizer buffers remain frozen.

Validation selects the decoder checkpoint. The saved, CPU-reloaded codec must
exceed 97% non-PAD reconstruction accuracy and 94% first/last-token accuracy on
512 validation chunks before data preparation or GRAM training can proceed.
This gate does not certify exact symbolic fidelity or constitute a sealed test.
No test chunks are used for codec fitting or checkpoint selection.
The initial 64-chunk shape probe preceded split filtering and included four
chunks from one test family. That entire family is excluded from this new
experiment's final test; `probe_scope.json` records the exposure and exclusion.

The CPU root sampler uses algebraically equivalent matrix products to avoid an
expensive generic convolution path. Tests cover the equivalence and the resize
recipe. No original NCM assets or source are modified.

## Matched training

Both arms use the same corpus, codec, splits, 32-ID target, 256-ID context,
hybrid context interface, optimizer, effective batch eight, and recurrence
(four fast plus two slow applications per trajectory). Hybrid context retains
the finest code positions plus level summaries: 352 observed vectors. This is
still the custom Astra conditioning interface, not the paper's complete
position-resolved cumulative-reconstruction generator.

Production-shape MPS preflight checks both sizes for finite gradients, fixed-batch
fitting, CPU/MPS objective agreement, stochastic optimizer/RNG recovery,
inference-export equivalence, and actual rolling generation/codec feedback.
Preflight weights never initialize the production arms.

The 30M arm starts first. Jobs alternate in exclusive 30-minute slices, with the
same total arm-time budget. Equal-step inference checkpoints additionally permit
an equal-examples comparison. Both arms use the same learning-rate schedule at
the same update index, calibrated to the slower arm's budget. One shared 24-hour deadline includes preparation,
switching overhead and a final-evaluation reserve. No two heavy jobs run together.
Resource guards stop training cleanly; a failed gate blocks later stages.

## Measurements and limits

- Prior-only Monte Carlo code NLL, compared only between the same-codec arms.
- True-context versus shuffled-context NLL with paired random seeds. This is a
  context-sensitivity diagnostic, not a semantic relevance rating.
- Four rolling chunks, categorical temperature 0.8, plus a greedy first chunk.
  Decoded text is re-encoded before it becomes subsequent observed context.
- Dominant-word and repeated-trigram fractions, duplicate chunks, adjacent
  boundary text and repeated words at seams. Saved text needs human review for
  grammatical and semantic continuity.
- End-to-end content-token throughput includes generator prediction, CPU decode,
  and intermediate canonical feedback encoding.
- Validation selects checkpoints; a sealed test is scored only at the end and
  cannot select a winner checkpoint.

This remains one seed on the small reused FineWeb-Edu corpus. It cannot establish
general language competence or attribute changes against the old run solely to
model size: the codec, target length and context interface also changed.

Artifacts: `reports/shortblock_20261019`, `artifacts/raw_native32_20261019`, and
`runs/20261019_native32_10m_30m`. The supervisor records an explicit failure if
codec acceptance, data integrity or MPS preflight fails.
