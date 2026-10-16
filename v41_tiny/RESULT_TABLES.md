# Saved experiment tables

Generated from completed artifacts at 2026-10-02T11:23:59.764590+00:00.
Partial tables remain labelled by completed seed count. Rates below are per-episode means unless otherwise stated.

## Execution inventory

| Run | Device / threads | Real tokens | Final | Analysis |
|---|---|---:|---|---|
| v41_pin_s0_6M | cpu / 4 | 6,000,082 | complete | complete |
| fresh_pin_llm_s0_6M | cpu / 4 | 6,000,082 | complete | complete |
| v41_text_s0_5M | cpu / 4 | 5,000,545 | complete | complete |
| v41_pin_s1_6M | cpu / 4 | 6,000,430 | complete | complete |
| v41_stack_pin_s0_6M | cpu / 2 | 6,000,082 | complete | complete |
| v41_text_s1_5M | cpu / 2 | 5,000,881 | complete | complete |
| v41_pin_s2_6M | cpu / 2 | 6,000,094 | complete | complete |
| v41_loopoff_pin_s0_6M | cpu / 2 | 6,000,082 | complete | complete |
| v41_stack_pin_s1_6M | cpu / 2 | 6,000,430 | complete | complete |
| v41_loopoff_pin_s1_6M | cpu / 2 | 6,000,430 | complete | complete |
| v41_stack_pin_s2_6M | cpu / 2 | 6,000,094 | complete | complete |
| v41_loopoff_pin_s2_6M | cpu / 2 | 6,000,094 | complete | complete |
| exploratory_noplan_pin_s0_6M | mps / 2 | 6,000,082 | complete | complete |
| exploratory_mps_plan_pin_s0_6M | mps / 2 | 6,000,082 | complete | complete |

## P1: held-out compositions

| Model | Seeds | Action accuracy | Pair-second b2 | Action NTP | Long b2 | Long action NTP |
|---|---:|---:|---:|---:|---:|---:|
| v4 / no R | 3 | 0.6936 | 0.5200 | 0.9110 | 0.5017 | 1.0578 |
| v4.1 | 3 | 0.7491 | 0.6867 | 0.7505 | 0.6562 | 0.9913 |
| Transformer LLM | 3 | 0.8268 | 0.8094 | 0.6100 | 0.7474 | 2.1664 |
| v4.1 stack | 3 | 0.7165 | 0.6000 | 0.8336 | 0.5825 | 1.0424 |
| v4.1 trained F loop off | 3 | 0.7540 | 0.6939 | 0.7532 | 0.6736 | 0.9978 |

## P1: per-seed final numbers

| Model | Seed | Action accuracy | Pair-second b2 | Action NTP |
|---|---:|---:|---:|---:|
| v4 / no R | 0 | 0.6929 | 0.5383 | 0.9152 |
| v4 / no R | 1 | 0.6975 | 0.5233 | 0.9058 |
| v4 / no R | 2 | 0.6904 | 0.4983 | 0.9120 |
| v4.1 | 0 | 0.7557 | 0.6817 | 0.7379 |
| v4.1 | 1 | 0.7501 | 0.6983 | 0.7500 |
| v4.1 | 2 | 0.7416 | 0.6800 | 0.7636 |
| Transformer LLM | 0 | 0.8200 | 0.7883 | 0.6224 |
| Transformer LLM | 1 | 0.8161 | 0.7900 | 0.6184 |
| Transformer LLM | 2 | 0.8442 | 0.8500 | 0.5891 |
| v4.1 stack | 0 | 0.7479 | 0.6867 | 0.7606 |
| v4.1 stack | 1 | 0.6991 | 0.5500 | 0.8895 |
| v4.1 stack | 2 | 0.7024 | 0.5633 | 0.8507 |
| v4.1 trained F loop off | 0 | 0.7692 | 0.7183 | 0.7357 |
| v4.1 trained F loop off | 1 | 0.7407 | 0.6700 | 0.7728 |
| v4.1 trained F loop off | 2 | 0.7520 | 0.6933 | 0.7513 |

## Natural plan-lag strata

These are observational strata of the same held-out expert episodes. Different lags occupy different layouts/path positions, so the table does not identify a causal effect of stale plans. Averages use available per-episode values, then seeds.

| Model | Seeds | Lag | Action accuracy | b2 | Action NTP | Mean actions per episode | Episodes with b2 per seed / 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v4 / no R | 3 | 1 | 0.7021 | 0.5408 | 0.9153 | 10.2000 | 143 |
| v4 / no R | 3 | 3 | 0.6880 | 0.5507 | 0.9057 | 10.8700 | 150 |
| v4 / no R | 3 | 5 | 0.6916 | 0.4444 | 0.9116 | 10.5350 | 137 |
| v4.1 | 3 | 1 | 0.7518 | 0.6997 | 0.7550 | 10.2000 | 143 |
| v4.1 | 3 | 3 | 0.7478 | 0.7163 | 0.7598 | 10.8700 | 150 |
| v4.1 | 3 | 5 | 0.7481 | 0.6436 | 0.7359 | 10.5350 | 137 |
| Transformer LLM | 3 | 1 | 0.8244 | 0.8124 | 0.6103 | 10.2000 | 143 |
| Transformer LLM | 3 | 3 | 0.8220 | 0.8263 | 0.6343 | 10.8700 | 150 |
| Transformer LLM | 3 | 5 | 0.8344 | 0.7839 | 0.5844 | 10.5350 | 137 |
| v4.1 stack | 3 | 1 | 0.7249 | 0.6340 | 0.8367 | 10.2000 | 143 |
| v4.1 stack | 3 | 3 | 0.7114 | 0.6193 | 0.8328 | 10.8700 | 150 |
| v4.1 stack | 3 | 5 | 0.7141 | 0.5369 | 0.8308 | 10.5350 | 137 |
| v4.1 trained F loop off | 3 | 1 | 0.7598 | 0.7323 | 0.7590 | 10.2000 | 143 |
| v4.1 trained F loop off | 3 | 3 | 0.7526 | 0.7074 | 0.7603 | 10.8700 | 150 |
| v4.1 trained F loop off | 3 | 5 | 0.7500 | 0.6221 | 0.7396 | 10.5350 | 137 |

## P2: closed-loop outcomes, 256 shared layouts per seed

| Model | Seeds | Sampled pair integrity | Greedy pair integrity | Sampled valid composition | Greedy valid composition | Sampled completion | Greedy completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| v4 / no R | 3 | 0.1449 | 0.1410 | 0.0000 | 0.0000 | 0.9987 | 0.5677 |
| v4.1 | 3 | 0.1748 | 0.2496 | 0.0000 | 0.0000 | 0.9922 | 0.7383 |
| Transformer LLM | 3 | 0.3260 | 0.4387 | 0.0039 | 0.0221 | 0.9922 | 0.9531 |
| v4.1 stack | 3 | 0.1558 | 0.1818 | 0.0000 | 0.0000 | 0.9922 | 0.6107 |
| v4.1 trained F loop off | 3 | 0.1823 | 0.2091 | 0.0000 | 0.0000 | 0.9961 | 0.7070 |

Pair integrity averages only layouts with an eligible transition from an even colour. Layouts without one have a saved NaN, and paired SEs use the intersection of eligible layouts. The valid-composition and completion rates above use all 256 layouts. Coverage is reported below; the zero-for-no-pair column is a secondary all-layout scoring convention, not a replacement for the registered metric.

| Model | Seed | Policy | Eligible layouts / 256 | Conditional pair integrity | All-layout score, zero for no pair |
|---|---:|---|---:|---:|---:|
| v4 / no R | 0 | temp1 | 249 / 256 | 0.1493 | 0.1452 |
| v4 / no R | 0 | greedy | 175 / 256 | 0.1286 | 0.0879 |
| v4 / no R | 1 | temp1 | 249 / 256 | 0.1725 | 0.1678 |
| v4 / no R | 1 | greedy | 180 / 256 | 0.1463 | 0.1029 |
| v4 / no R | 2 | temp1 | 245 / 256 | 0.1131 | 0.1082 |
| v4 / no R | 2 | greedy | 183 / 256 | 0.1480 | 0.1058 |
| v4.1 | 0 | temp1 | 254 / 256 | 0.1889 | 0.1874 |
| v4.1 | 0 | greedy | 212 / 256 | 0.3023 | 0.2503 |
| v4.1 | 1 | temp1 | 247 / 256 | 0.1768 | 0.1706 |
| v4.1 | 1 | greedy | 196 / 256 | 0.2136 | 0.1635 |
| v4.1 | 2 | temp1 | 252 / 256 | 0.1587 | 0.1562 |
| v4.1 | 2 | greedy | 200 / 256 | 0.2329 | 0.1820 |
| Transformer LLM | 0 | temp1 | 251 / 256 | 0.3149 | 0.3087 |
| Transformer LLM | 0 | greedy | 245 / 256 | 0.4510 | 0.4316 |
| Transformer LLM | 1 | temp1 | 253 / 256 | 0.2352 | 0.2324 |
| Transformer LLM | 1 | greedy | 254 / 256 | 0.3161 | 0.3136 |
| Transformer LLM | 2 | temp1 | 254 / 256 | 0.4278 | 0.4245 |
| Transformer LLM | 2 | greedy | 247 / 256 | 0.5491 | 0.5298 |
| v4.1 stack | 0 | temp1 | 253 / 256 | 0.1930 | 0.1908 |
| v4.1 stack | 0 | greedy | 207 / 256 | 0.2552 | 0.2064 |
| v4.1 stack | 1 | temp1 | 246 / 256 | 0.1541 | 0.1481 |
| v4.1 stack | 1 | greedy | 182 / 256 | 0.1449 | 0.1030 |
| v4.1 stack | 2 | temp1 | 240 / 256 | 0.1201 | 0.1126 |
| v4.1 stack | 2 | greedy | 170 / 256 | 0.1453 | 0.0965 |
| v4.1 trained F loop off | 0 | temp1 | 252 / 256 | 0.1748 | 0.1721 |
| v4.1 trained F loop off | 0 | greedy | 215 / 256 | 0.2168 | 0.1821 |
| v4.1 trained F loop off | 1 | temp1 | 254 / 256 | 0.1785 | 0.1771 |
| v4.1 trained F loop off | 1 | greedy | 178 / 256 | 0.2184 | 0.1518 |
| v4.1 trained F loop off | 2 | temp1 | 252 / 256 | 0.1936 | 0.1906 |
| v4.1 trained F loop off | 2 | greedy | 204 / 256 | 0.1922 | 0.1531 |

## T1/T2: text, 5M tokens per seed

| Model | Seeds | Mean per-row NTP | Token-weighted NTP | recall1 exact | carry3 exact | multi4 exact |
|---|---:|---:|---:|---:|---:|---:|
| v4 / no R | 2 | 2.5411 | 2.5983 | 0.0130 | 0.0000 | 0.0000 |
| v4.1 | 2 | 2.5304 | 2.5878 | 0.0156 | 0.0000 | 0.0000 |
| Transformer LLM | 2 | 2.5481 | 2.6055 | 0.0130 | 0.0000 | 0.0000 |

## Text synthetic-row pool audit

NTP covers entire synthetic rows, not only filler tokens. Training-pool and validation-pool items differ; their loss difference is descriptive and is not a paired memorisation test.

| Model | Seed | Training-pool row NTP | Validation-pool row NTP | Validation minus training pool |
|---|---:|---:|---:|---:|
| v4 / no R | 0 | 3.4638 | 3.2853 | -0.1784 |
| v4 / no R | 1 | 3.4919 | 3.2820 | -0.2099 |
| v4.1 | 0 | 3.4695 | 3.2890 | -0.1804 |
| v4.1 | 1 | 3.4904 | 3.2871 | -0.2033 |
| Transformer LLM | 0 | 3.5367 | 3.3418 | -0.1950 |
| Transformer LLM | 1 | 3.5411 | 3.3344 | -0.2068 |

## Fixed-checkpoint text interventions

Values below use the runner's token-weighted language loss. Retrieval exact-match values use 192 items per family. These interventions measure checkpoint reliance.

| Model | Seed | Mode | Language NTP | Difference from intact | recall1 | carry3 | multi4 |
|---|---:|---|---:|---:|---:|---:|---:|
| v4 / no R | 0 | intact | 2.6004 | 0.0000 | 0.0052 | 0.0000 | 0.0000 |
| v4 / no R | 0 | core_off | 2.7205 | 0.1201 | 0.0104 | 0.0000 | 0.0000 |
| v4 / no R | 0 | plan_swap | 2.7307 | 0.1304 | 0.0156 | 0.0000 | 0.0000 |
| v4 / no R | 0 | M1 | 2.6029 | 0.0025 | 0.0156 | 0.0000 | 0.0000 |
| v4 / no R | 0 | loop_off | 2.6009 | 0.0005 | 0.0052 | 0.0000 | 0.0000 |
| v4 / no R | 1 | intact | 2.5962 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| v4 / no R | 1 | core_off | 2.7232 | 0.1271 | 0.0260 | 0.0000 | 0.0000 |
| v4 / no R | 1 | plan_swap | 2.7555 | 0.1593 | 0.0156 | 0.0000 | 0.0000 |
| v4 / no R | 1 | M1 | 2.5976 | 0.0015 | 0.0208 | 0.0000 | 0.0000 |
| v4 / no R | 1 | loop_off | 2.5975 | 0.0013 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 0 | intact | 2.5927 | 0.0000 | 0.0104 | 0.0000 | 0.0000 |
| v4.1 | 0 | core_off | 2.6834 | 0.0907 | 0.0104 | 0.0000 | 0.0000 |
| v4.1 | 0 | plan_swap | 2.6665 | 0.0738 | 0.0156 | 0.0000 | 0.0000 |
| v4.1 | 0 | M1 | 2.5933 | 0.0005 | 0.0156 | 0.0000 | 0.0000 |
| v4.1 | 0 | loop_off | 2.5934 | 0.0007 | 0.0156 | 0.0000 | 0.0000 |
| v4.1 | 0 | u_off | 2.6251 | 0.0324 | 0.0104 | 0.0000 | 0.0000 |
| v4.1 | 0 | c_off | 2.6009 | 0.0082 | 0.0156 | 0.0000 | 0.0000 |
| v4.1 | 0 | core_shared_off | 2.5941 | 0.0013 | 0.0052 | 0.0000 | 0.0000 |
| v4.1 | 0 | enc_shared_off | 2.6020 | 0.0093 | 0.0156 | 0.0000 | 0.0000 |
| v4.1 | 0 | R_local | 2.7096 | 0.1169 | 0.0104 | 0.0000 | 0.0000 |
| v4.1 | 0 | core_off_R_local | 2.8061 | 0.2134 | 0.0104 | 0.0000 | 0.0000 |
| v4.1 | 1 | intact | 2.5828 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | core_off | 2.6917 | 0.1089 | 0.0260 | 0.0000 | 0.0000 |
| v4.1 | 1 | plan_swap | 2.6722 | 0.0893 | 0.0260 | 0.0000 | 0.0000 |
| v4.1 | 1 | M1 | 2.5833 | 0.0005 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | loop_off | 2.5831 | 0.0002 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | u_off | 2.6253 | 0.0425 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | c_off | 2.5936 | 0.0108 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | core_shared_off | 2.5839 | 0.0011 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | enc_shared_off | 2.5878 | 0.0050 | 0.0208 | 0.0000 | 0.0000 |
| v4.1 | 1 | R_local | 2.6748 | 0.0920 | 0.0260 | 0.0000 | 0.0000 |
| v4.1 | 1 | core_off_R_local | 2.8081 | 0.2252 | 0.0156 | 0.0000 | 0.0000 |
| Transformer LLM | 0 | intact | 2.6090 | 0.0000 | 0.0052 | 0.0000 | 0.0000 |
| Transformer LLM | 0 | M1 | 2.6533 | 0.0443 | 0.0000 | 0.0000 | 0.0000 |
| Transformer LLM | 0 | loop_off | 2.6318 | 0.0228 | 0.0104 | 0.0000 | 0.0000 |
| Transformer LLM | 1 | intact | 2.6020 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| Transformer LLM | 1 | M1 | 2.6572 | 0.0552 | 0.0156 | 0.0000 | 0.0000 |
| Transformer LLM | 1 | loop_off | 2.6237 | 0.0217 | 0.0156 | 0.0000 | 0.0000 |

## Registered paired differences

Left minus right. The item SE follows the registered formula; seed SE is separate.

| Comparison | Metric | Difference | Paired item SE | Training-seed SE | Seed range | Seeds |
|---|---|---:|---:|---:|---|---:|
| pin_v41_minus_noR | heldout/act_opt | 0.0555 | 0.0033 | 0.0037 | 0.0512 … 0.0628 | 3 |
| pin_v41_minus_noR | heldout/act_opt_b2 | 0.1667 | 0.0153 | 0.0118 | 0.1433 … 0.1817 | 3 |
| pin_v41_minus_noR | heldout/ntp_action | -0.1605 | 0.0039 | 0.0087 | -0.1773 … -0.1484 | 3 |
| pin_v41_minus_noR | roll_temp1/pair_ok | 0.0306 | 0.0128 | 0.0125 | 0.0057 … 0.0443 | 3 |
| pin_v41_minus_noR | roll_greedy/pair_ok | 0.1170 | 0.0184 | 0.0239 | 0.0813 … 0.1624 | 3 |
| pin_v41_minus_LLM | heldout/act_opt | -0.0776 | 0.0032 | 0.0125 | -0.1025 … -0.0643 | 3 |
| pin_v41_minus_LLM | heldout/act_opt_b2 | -0.1228 | 0.0115 | 0.0240 | -0.1700 … -0.0917 | 3 |
| pin_v41_minus_LLM | heldout/ntp_action | 0.1406 | 0.0034 | 0.0176 | 0.1156 … 0.1745 | 3 |
| pin_v41_minus_LLM | roll_temp1/pair_ok | -0.1483 | 0.0144 | 0.0607 | -0.2640 … -0.0587 | 3 |
| pin_v41_minus_LLM | roll_greedy/pair_ok | -0.1836 | 0.0183 | 0.0659 | -0.3133 … -0.0987 | 3 |
| pin_stack_minus_v41 | heldout/act_opt | -0.0327 | 0.0029 | 0.0129 | -0.0510 … -0.0078 | 3 |
| pin_stack_minus_v41 | heldout/act_opt_b2 | -0.0867 | 0.0129 | 0.0467 | -0.1483 … 0.0050 | 3 |
| pin_stack_minus_v41 | heldout/ntp_action | 0.0831 | 0.0031 | 0.0338 | 0.0226 … 0.1395 | 3 |
| pin_stack_minus_v41 | roll_temp1/pair_ok | -0.0222 | 0.0132 | 0.0134 | -0.0417 … 0.0034 | 3 |
| pin_stack_minus_v41 | roll_greedy/pair_ok | -0.0659 | 0.0189 | 0.0140 | -0.0897 … -0.0412 | 3 |
| pin_loopoff_minus_v41 | heldout/act_opt | 0.0048 | 0.0027 | 0.0072 | -0.0094 … 0.0135 | 3 |
| pin_loopoff_minus_v41 | heldout/act_opt_b2 | 0.0072 | 0.0087 | 0.0190 | -0.0283 … 0.0367 | 3 |
| pin_loopoff_minus_v41 | heldout/ntp_action | 0.0027 | 0.0024 | 0.0104 | -0.0123 … 0.0228 | 3 |
| pin_loopoff_minus_v41 | roll_temp1/pair_ok | 0.0056 | 0.0133 | 0.0143 | -0.0164 … 0.0324 | 3 |
| pin_loopoff_minus_v41 | roll_greedy/pair_ok | -0.0449 | 0.0179 | 0.0143 | -0.0734 … -0.0279 | 3 |
| text_v41_minus_noR | row_ntp | -0.0107 | 0.0018 | 0.0034 | -0.0141 … -0.0073 | 2 |
| text_v41_minus_LLM | row_ntp | -0.0177 | 0.0040 | 0.0011 | -0.0188 … -0.0166 | 2 |

## Fixed-checkpoint interventions on pinpad

These measure checkpoint reliance, not training-time necessity.

| Model | Seeds | Intervention | Action accuracy | b2 | Action NTP |
|---|---:|---|---:|---:|---:|
| v4.1 | 3 | intact | 0.7491 | 0.6867 | 0.7505 |
| v4.1 | 3 | core_off | 0.6508 | 0.5283 | 3.1778 |
| v4.1 | 3 | plan_swap | 0.7226 | 0.6550 | 0.7995 |
| v4.1 | 3 | M1 | 0.7490 | 0.6844 | 0.7508 |
| v4.1 | 3 | loop_off | 0.7500 | 0.6894 | 0.7507 |
| v4.1 | 3 | u_off | 0.7382 | 0.6711 | 0.7730 |
| v4.1 | 3 | c_off | 0.7462 | 0.6783 | 0.7562 |
| v4.1 | 3 | core_shared_off | 0.7473 | 0.6856 | 0.7538 |
| v4.1 | 3 | enc_shared_off | 0.7472 | 0.6750 | 0.7542 |
| v4.1 | 3 | R_local | 0.5932 | 0.4522 | 3.2406 |
| v4.1 | 3 | core_off_R_local | 0.5681 | 0.4400 | 4.4993 |
| v4.1 stack | 3 | intact | 0.7165 | 0.6000 | 0.8336 |
| v4.1 stack | 3 | core_off | 0.5678 | 0.3994 | 6.3954 |
| v4.1 stack | 3 | plan_swap | 0.6335 | 0.4467 | 1.1993 |
| v4.1 stack | 3 | M1 | 0.7155 | 0.5956 | 0.8343 |
| v4.1 stack | 3 | loop_off | 0.7127 | 0.5889 | 0.8348 |
| v4.1 stack | 3 | u_off | 0.6614 | 0.5106 | 0.9954 |
| v4.1 stack | 3 | c_off | 0.7111 | 0.5950 | 0.8570 |
| v4.1 stack | 3 | core_shared_off | 0.7112 | 0.5972 | 0.8500 |
| v4.1 stack | 3 | enc_shared_off | 0.7150 | 0.5933 | 0.8329 |
| v4.1 stack | 3 | R_local | 0.6543 | 0.4533 | 2.0630 |
| v4.1 stack | 3 | core_off_R_local | 0.5480 | 0.4011 | 7.3693 |

## Calibrated plans: paired effects relative to intact

Constants are estimated from an independent training bank; nulls and offsets stay legal.

| Model / data | Mode | Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---|---|---:|---:|---:|---:|
| v4.1 / pinpad | constant | act_opt | -0.0111 | 0.0020 | 0.0061 | 3 |
| v4.1 / pinpad | constant | act_opt_b2 | -0.0161 | 0.0079 | 0.0055 | 3 |
| v4.1 / pinpad | constant | ntp_action | 0.0207 | 0.0016 | 0.0048 | 3 |
| v4.1 / pinpad | zero_content | act_opt | -0.0140 | 0.0021 | 0.0074 | 3 |
| v4.1 / pinpad | zero_content | act_opt_b2 | -0.0194 | 0.0083 | 0.0049 | 3 |
| v4.1 / pinpad | zero_content | ntp_action | 0.0286 | 0.0018 | 0.0068 | 3 |
| v4.1 / pinpad | null | act_opt | -0.0983 | 0.0036 | 0.0289 | 3 |
| v4.1 / pinpad | null | act_opt_b2 | -0.1583 | 0.0146 | 0.0527 | 3 |
| v4.1 / pinpad | null | ntp_action | 2.4273 | 0.0216 | 1.6716 | 3 |
| v4 / pinpad | constant | act_opt | -0.0571 | 0.0031 | 0.0009 | 3 |
| v4 / pinpad | constant | act_opt_b2 | -0.1128 | 0.0123 | 0.0101 | 3 |
| v4 / pinpad | constant | ntp_action | 0.1446 | 0.0051 | 0.0018 | 3 |
| v4 / pinpad | zero_content | act_opt | -0.0575 | 0.0031 | 0.0007 | 3 |
| v4 / pinpad | zero_content | act_opt_b2 | -0.1189 | 0.0122 | 0.0120 | 3 |
| v4 / pinpad | zero_content | ntp_action | 0.1981 | 0.0058 | 0.0044 | 3 |
| v4 / pinpad | null | act_opt | -0.0730 | 0.0036 | 0.0067 | 3 |
| v4 / pinpad | null | act_opt_b2 | -0.1339 | 0.0125 | 0.0156 | 3 |
| v4 / pinpad | null | ntp_action | 0.5475 | 0.0096 | 0.1452 | 3 |
| stack / pinpad | constant | act_opt | -0.0611 | 0.0033 | 0.0054 | 3 |
| stack / pinpad | constant | act_opt_b2 | -0.1278 | 0.0137 | 0.0223 | 3 |
| stack / pinpad | constant | ntp_action | 0.2344 | 0.0068 | 0.0137 | 3 |
| stack / pinpad | zero_content | act_opt | -0.0640 | 0.0036 | 0.0086 | 3 |
| stack / pinpad | zero_content | act_opt_b2 | -0.1033 | 0.0134 | 0.0169 | 3 |
| stack / pinpad | zero_content | ntp_action | 0.2031 | 0.0064 | 0.0500 | 3 |
| stack / pinpad | null | act_opt | -0.1486 | 0.0042 | 0.0611 | 3 |
| stack / pinpad | null | act_opt_b2 | -0.2006 | 0.0145 | 0.0674 | 3 |
| stack / pinpad | null | ntp_action | 5.5618 | 0.0269 | 3.6979 | 3 |
| loop off / pinpad | constant | act_opt | -0.0234 | 0.0025 | 0.0041 | 3 |
| loop off / pinpad | constant | act_opt_b2 | -0.0244 | 0.0094 | 0.0028 | 3 |
| loop off / pinpad | constant | ntp_action | 0.0474 | 0.0026 | 0.0168 | 3 |
| loop off / pinpad | zero_content | act_opt | -0.0337 | 0.0029 | 0.0055 | 3 |
| loop off / pinpad | zero_content | act_opt_b2 | -0.0367 | 0.0097 | 0.0054 | 3 |
| loop off / pinpad | zero_content | ntp_action | 0.0617 | 0.0029 | 0.0170 | 3 |
| loop off / pinpad | null | act_opt | -0.1056 | 0.0035 | 0.0323 | 3 |
| loop off / pinpad | null | act_opt_b2 | -0.1306 | 0.0135 | 0.0578 | 3 |
| loop off / pinpad | null | ntp_action | 1.3321 | 0.0202 | 0.6500 | 3 |
| v4.1 / text | constant | row_ntp | 0.0398 | 0.0019 | 0.0063 | 2 |
| v4.1 / text | zero_content | row_ntp | 0.0384 | 0.0013 | 0.0079 | 2 |
| v4.1 / text | null | row_ntp | 0.0997 | 0.0030 | 0.0092 | 2 |

## Counterfactual facts, fixed 128 legal prefix pairs

| Model | Seed | Original accuracy | Changed-fact accuracy | Prediction changes | Changed correct given original correct |
|---|---:|---:|---:|---:|---:|
| v4 / no R | 0 | 0.4141 | 0.3672 | 0.0938 | 0.0566 |
| v4 / no R | 1 | 0.4141 | 0.3594 | 0.0781 | 0.0189 |
| v4 / no R | 2 | 0.3984 | 0.3828 | 0.0625 | 0.0588 |
| v4.1 | 0 | 0.5938 | 0.3125 | 0.1016 | 0.0526 |
| v4.1 | 1 | 0.5781 | 0.2812 | 0.0859 | 0.0811 |
| v4.1 | 2 | 0.5547 | 0.3281 | 0.1328 | 0.0845 |
| Transformer LLM | 0 | 0.6953 | 0.6094 | 0.5078 | 0.5393 |
| Transformer LLM | 1 | 0.7266 | 0.4219 | 0.3203 | 0.3226 |
| Transformer LLM | 2 | 0.8125 | 0.7266 | 0.7266 | 0.6923 |
| v4.1 stack | 0 | 0.5938 | 0.3906 | 0.2031 | 0.2105 |
| v4.1 stack | 1 | 0.4297 | 0.3281 | 0.1172 | 0.0727 |
| v4.1 stack | 2 | 0.4688 | 0.3672 | 0.0703 | 0.0833 |
| v4.1 trained F loop off | 0 | 0.6328 | 0.2891 | 0.0547 | 0.0247 |
| v4.1 trained F loop off | 1 | 0.5703 | 0.2578 | 0.1016 | 0.0685 |
| v4.1 trained F loop off | 2 | 0.5547 | 0.3516 | 0.1719 | 0.0986 |
| MPS no plan | 0 | 0.6953 | 0.6328 | 0.5312 | 0.5393 |
| MPS normal plan | 0 | 0.5547 | 0.3047 | 0.0391 | 0.0141 |

## Coarse codes and causally legal probes

The NMI label is the majority subgoal within a chunk; it is non-causal. The permutation null preserves each episode's temporal structure.

| Model | Seed | Codes used | NMI | Null mean | Null tail p | Core pass 1 subgoal probe | Core pass 2 subgoal probe | Majority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v4.1 | 0 | 39 | 0.0147 | 0.0152 | 0.6109 | 0.2878 | 0.2948 | 0.2672 |
| v4.1 | 1 | 46 | 0.0154 | 0.0161 | 0.6109 | 0.2752 | 0.2672 | 0.2672 |
| v4.1 | 2 | 51 | 0.0155 | 0.0183 | 0.8794 | 0.2913 | 0.2998 | 0.2672 |
| v4.1 stack | 0 | 41 | 0.0156 | 0.0148 | 0.3502 | 0.2967 | 0.3136 | 0.2672 |
| v4.1 stack | 1 | 45 | 0.0147 | 0.0159 | 0.7004 | 0.2684 | 0.2648 | 0.2672 |
| v4.1 stack | 2 | 48 | 0.0140 | 0.0172 | 0.9611 | 0.2610 | 0.2693 | 0.2672 |
| MPS no plan | 0 | 2 | 0.0022 | 0.0033 | 0.5564 | 0.2464 | 0.2429 | 0.2672 |
| MPS normal plan | 0 | 48 | 0.0151 | 0.0181 | 0.9183 | 0.2687 | 0.2696 | 0.2672 |

## Frozen oracle control, all learning rates

128 prefixes: 52 initially wrong, 76 initially right. Ten steps, weights frozen.

| Site / learning rate | Wrong fixed | Right broken | Optimal-set probability gain | Random wrong fixed | Oracle step norm | Random achieved step norm |
|---|---:|---:|---:|---:|---:|---:|
| C@0.001 | 0.0577 | 0.0000 | 0.0045 | 0.0000 | 0.0187 | 0.0186 |
| C@0.01 | 0.2115 | 0.0000 | 0.0426 | 0.0000 | 0.1824 | 0.1810 |
| C@0.03 | 0.4808 | 0.0000 | 0.1122 | 0.0000 | 0.5135 | 0.5092 |
| S1@0.001 | 0.0192 | 0.0000 | 0.0007 | 0.0000 | 0.0187 | 0.0186 |
| S1@0.01 | 0.0577 | 0.0000 | 0.0069 | 0.0000 | 0.1865 | 0.1851 |
| S1@0.03 | 0.1346 | 0.0000 | 0.0205 | 0.0000 | 0.5551 | 0.5510 |
| u@0.001 | 0.0385 | 0.0000 | 0.0039 | 0.0000 | 0.0080 | 0.0079 |
| u@0.01 | 0.1731 | 0.0000 | 0.0390 | 0.0000 | 0.0787 | 0.0781 |
| u@0.03 | 0.4231 | 0.0000 | 0.1116 | 0.0000 | 0.2275 | 0.2257 |
