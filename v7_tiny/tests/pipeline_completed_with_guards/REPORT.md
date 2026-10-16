# V7 tiny experiment report

Generated UTC: 2026-10-04 20:21:23. Observed queue status: completed_with_limits.

This is a bounded tiny-model study. Numerical/caching smoke checks establish implementation contracts; they do not establish model quality or scaling.

V7 foundation: **5,365,938** active parameters. Ordinary Transformer: **5,366,208** (270 more, 0.0050%). Exploration adds **41,472** parameters; deployment has **5,407,410**. A resident frozen teacher is excluded from active deployment counts.

Common foundation budget: **4,000** scored targets per arm and seed.

| Run | Completed scored targets | Train status | Fresh text NLL | Standard strict | Hard strict |
|---|---:|---|---:|---:|---:|
| A_alpha0_s171 (test) | 1,654 | pilot_prefix_completed | 7.635 | 0.0% | 0.0% |
| A_flat_s171 (test) | 1,654 | pilot_prefix_completed | 7.630 | 0.0% | 0.0% |
| A_ncp_s171 (test) | 1,654 | pilot_prefix_completed | 7.635 | 0.0% | 0.0% |
| B0_ncp_s171 (test) | 1,654 | pilot_prefix_completed | 7.635 | 0.0% | 0.0% |
| B1_ncp_s171 (test) | 1,483 | pilot_prefix_completed | 7.635 | 0.0% | 0.0% |
| mean_Z (test) | 1,483 | pilot_prefix_completed | 7.635 | 0.0% | 0.0% |
| pilot_flat_lr0.001 (dev) | 1,584 | pilot_prefix_completed | 7.639 | 0.0% | pending% |
| pilot_ncp_lr0.001 (dev) | 1,584 | pilot_prefix_completed | 7.639 | 0.0% | pending% |

Matched foundation differences use whole held-out document resampling for token-weighted NLL, and paired complete problems for task success. Positive task differences favor the first arm; negative NLL differences favor it.

`ncp-flat_s171`: text NLL difference 0.0045, 95% CI [-0.00039382416647968293, 0.009343956531494777]; standard difference 0.00 percentage points, CI [0.0, 0.0]. Complete matched endpoints: False.

`ncp-alpha0_s171`: text NLL difference 0.0000, 95% CI [-1.5936072127937336e-06, 2.6931961896214098e-05]; standard difference 0.00 percentage points, CI [0.0, 0.0]. Complete matched endpoints: False.

`alpha0-flat_s171`: text NLL difference 0.0045, 95% CI [-0.0004160310829899637, 0.009342522285003263]; standard difference 0.00 percentage points, CI [0.0, 0.0]. Complete matched endpoints: False.

## Refit, exploration and deployment

Seed 171: refit accepted = True; zero-control anchor = `B0_ncp_s171`. Acceptance evidence: {"accepted": true, "text": {"difference": -6.399492136304063e-07, "ci95": [-2.239822247706422e-06, 6.424753289473684e-07], "units": 4, "method": "paired document bootstrap, token-weighted ratio"}, "symbolic_nll_difference": -0.0001735687255859375, "standard_success": {"difference": 0.0, "ci95": [0.0, 0.0], "units": 4, "method": "paired unit bootstrap", "draws": 4000}, "action": "advance refit"}.
Anchor errors {'default_zero': 0.0, 'lambda0_original': 0.0, 'lambda1_refit': 0.0}; control gradient norm 0.00022120235371403396; trained cache error 2.384185791015625e-07; deployment reload error 0.0.

## GradCuit and AB-MCTS

Development lock: {"parameters": {"temperature": 0.8, "top_k": 128, "candidate_limit": 16, "length": 1, "lr": 0.03, "kl": 0.0}, "grad_method": "gradcuit", "methods": ["retry", "gradcuit", "ab_hybrid"], "budgets_seconds": [30.0], "primary_budget_s": 30.0, "selection_dataset": "development only", "competence_gate": false, "interpretation": "diagnostic search only: standard foundation competence below 80%", "test_methods": ["retry", "gradcuit", "ab_hybrid", "hybrid_random", "hybrid_chain"], "uncertainty": "paired problem intervals per trained seed; three separately trained seeds shown individually; incomplete runs cannot confirm repeatability"}

### test_search_s171

Status: completed; warm-up 0.074 seconds; timed arms charge prefill, token generation, verification, gradients, and bookkeeping. Completed attempts after deadlines receive no success credit.

| Method and budget | Problems | Success | Charged elapsed seconds |
|---|---:|---:|---:|
| retry@30 | 4 | 0.00% | 15.8 |
| gradcuit@30 | 4 | 0.00% | 18.1 |
| ab_hybrid@30 | 4 | 0.00% | 16.8 |
| hybrid_random@30 | 4 | 0.00% | 16.7 |
| hybrid_chain@30 | 4 | 0.00% | 16.9 |

gradcuit-retry@30: 0.00 percentage points; 95% CI [0.0, 0.0]. Inconclusive at this sample and budget.

ab_hybrid-retry@30: 0.00 percentage points; 95% CI [0.0, 0.0]. Inconclusive at this sample and budget.

hybrid_random-retry@30: 0.00 percentage points; 95% CI [0.0, 0.0]. Inconclusive at this sample and budget.

hybrid_chain-retry@30: 0.00 percentage points; 95% CI [0.0, 0.0]. Inconclusive at this sample and budget.

AB-hybrid minus hybrid_random: 0.00 percentage points; CI [0.0, 0.0].

AB-hybrid minus hybrid_chain: 0.00 percentage points; CI [0.0, 0.0].

## Limits and interpretation

NCP uses feature-mean forecast/VQ losses and bounded routing adaptations. GradCuit uses bounded concept controls and response-mean rejected log-probability descent, rather than the paper’s raw hidden prefix/sequence-sum objective. AB-MCTS-A searches realized complete answers; latent-only dynamics and prefix-tree MCTS are outside this run.

Three separately trained seeds are listed individually; problem-level bootstrap intervals are conditional on each trained model. Larger widths, long-context behavior, BF16, distributed training and LLM-scale gains are not established. Semantic deduplication and graph-isomorphism independence are not claimed.

Token budgets are matched; parameter counts are almost exact. Additional compute for the V7 recipe, tuning and inference is recorded rather than treated as free. A partial endpoint cannot support a matched-completion comparison. Inference port interventions are distribution-shift diagnostics, not retrained ablations.

- larger-scale/distributed/BF16/context>384 training: Requires a separate experiment; not validated by this tiny run

- full V5/V6 causal bridge and retrained port-only ablations: Repository evidence reviewed; targeted 24-hour programme tests V7 matched foundations and staged controls, not the entire research programme

- matched token-control site, control-dimension sweep, binding-language tasks and fractional-feedback search: Not included in the initial 24-hour scope; two-hop composition is tested, but general variable binding is not

**This report is a smoke pipeline test only. No quality conclusion is permitted.**
