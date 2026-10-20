# Residual-consistent renderer recovery

This is a separate continuation of the completed `20261019_lm1b_grounded` pilot.
The original sources, data, checkpoints, final report, and validation scores remain
unchanged. The original maximum deadline, **20 October 2026, 13:40 IST**, still applies.

## What the completed experiment established

The two R2 graph planners reached 100% greedy success on 200 test graphs each and
96.5% / 98.5% on 200 larger graphs. The nonrecurrent controls reached 99.5% / 100%
and 94.5% / 94.5%, respectively. Parameter-matched untied controls reached
96.5% / 99.5% on larger graphs. These results support learning this synthetic task;
they do **not** establish an advantage from weight sharing. Search reached 100%
on the larger graphs for both R2 seeds, with extra model calls and wall time.
There is no equal-compute search superiority claim.

All structured models' selected checkpoints were at step500: the selection rule
kept the first maximum validation success. The displayed step4000 validation and
the final selected-checkpoint results must not be conflated.

The original renderer's code NLL decreased from about10.11 to0.279, but it rendered
the wrong node sequence. All eight joint planner-to-text probes found a valid plan;
none rendered that plan exactly. This is a failed language-fidelity gate.

## Diagnosis before more training

On the same24 **validation** plans, using the completed renderer:

| Intervention | Exact rendered route |
| --- | ---: |
| Free prediction of all levels | 0 / 24 |
| Supply gold first level, then generate | 0 / 24 |
| Supply gold first3 levels, then generate | 0 / 24 |
| Supply gold first6 levels, then generate | 14 / 24 |
| Supply gold first9 levels, then generate | 20 / 24 |
| Supply all gold codes to frozen decoder | 21 / 24 |
| Predict each level using gold earlier inputs, then decode combined predictions | 19 / 24 |

The last row is an oracle-assisted diagnostic, not deployable generation.
Teacher-conditioned code accuracy ranged from66.7% at the first level to roughly
86–96% at later levels. Code loss hides a serious free-generation mismatch.
Inspect `reports/renderer_diagnosis.json` for actual outputs, including boundary
normalization. The error is incorrect node identities, not merely a parser or
trailing-padding issue. The codec itself is imperfect even with gold codes.

## First-principles correction

NCM codes are residual corrections. Let `z*` be the frozen codec's encoded target,
and `z_hat` be the full reconstruction from the **actual** preceding predictions.
The next training label is the nearest code to the level-downsampled residual
`z* - z_hat`, using the frozen upstream quantizer.

Changing earlier codes while keeping the original cached later labels does not
preserve that relationship. The earlier pilot's uniform-prefix-corruption path
made this approximation. It remains unchanged in the comparison control. The new
arm uses generated prefixes **and recomputes the residual label**. This changes
the training distribution and label consistency together; a positive result would
not separately identify their causal contributions.

Gold tokens are used only to construct supervised training targets. The roll-in
generation function accepts the verified route condition and target level, never
gold tokens or target codes. No gradient is propagated through discrete roll-in.
The frozen codec, original generator weights, native128 window, all12 code levels,
rank4 LoRA, FiLM, and route conditioner architecture remain unchanged.

## Registered comparison

- Both arms start from the same12,000-update checkpoint, including AdamW moments.
- Each receives at most4,000 additional updates, effective batch4, seed119.
- Control: existing20% random-code corruption, otherwise gold-prefix training.
- Recovery:50% actual greedy generated prefixes with corrected residual labels,
  otherwise gold-prefix training.
- Same balanced level/position CE and0.1 frozen-base retention KL; same learning rates.
- Runs alternate in500-update chunks under the existing exclusive heavy-job lock.
- Saved elapsed times distinguish matched update budgets from unequal computation.
- Original maximum deadline and memory/disk/battery guards remain active.
- Validation uses48 free generations from validation rows24:72. Checkpoint ranking
  is free exact route, then normalized route edit similarity, then code NLL;
  retention KL must remain below0.15. Shuffled-route fidelity is also reported.
- The remaining renderer test rows24:96 are reserved for one final comparison of
  the parent and both selected checkpoints. The previous finalizer used rows0:24.
- This is a one-seed exploratory correction. No broad language-quality claim is made.

## Verification and operation

The MPS preflight checks gold-prefix residual-label equivalence at all12 levels,
exact generation/input parity, changed labels after actual prefix errors, frozen
weights, finite updates, and bit-exact optimizer/RNG checkpoint replay. An actual
CLI process stops at a chunk boundary and resumes in a new process before launch.

Launch with the existing venv (do not resolve its Python symlink):

```sh
v15_astra_m2/.venv/bin/python -m v15_astra_ncm_v4.followup.run
```

Telemetry: `http://127.0.0.1:8767/`. The dashboard separates completed pilot results,
the new arms, their evaluated checkpoint steps, and their actual generated routes.
Create `runs/20261019_renderer_prefix_followup/STOP` to request a clean stop.

Artifacts: `reports/renderer_diagnosis.json`, `reports/followup_preflight.json`,
`reports/followup_cli_smoke/`, and `runs/20261019_renderer_prefix_followup/`.
