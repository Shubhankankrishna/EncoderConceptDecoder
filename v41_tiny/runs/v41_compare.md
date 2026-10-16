# v4.1 registered comparisons

Differences are left minus right. Item-level paired SE and training-seed SE are separate.

## pin_v41_minus_noR

Pairs: v41_pin_s0_6M / ref_vs_pin_v4_s0_6M, v41_pin_s1_6M / ref_vs_pin_v4_s1_6M, v41_pin_s2_6M / ref_vs_pin_v4_s2_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | +0.0547 | 0.0036 | 0.0074 | 3 |
| heldin/act_opt_b2 | +0.1476 | 0.0156 | 0.0112 | 3 |
| heldin/ntp_action | -0.1489 | 0.0042 | 0.0070 | 3 |
| heldout/act_opt | +0.0555 | 0.0033 | 0.0037 | 3 |
| heldout/act_opt_b2 | +0.1667 | 0.0153 | 0.0118 | 3 |
| heldout/ntp_action | -0.1605 | 0.0039 | 0.0087 | 3 |
| long/act_opt | +0.0492 | 0.0042 | 0.0068 | 3 |
| long/act_opt_b2 | +0.1545 | 0.0204 | 0.0074 | 3 |
| long/ntp_action | -0.0665 | 0.0069 | 0.0120 | 3 |
| roll_temp1/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/complete | -0.0065 | 0.0029 | 0.0034 | 3 |
| roll_temp1/pair_ok | +0.0306 | 0.0128 | 0.0125 | 3 |
| roll_greedy/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/complete | +0.1706 | 0.0200 | 0.0392 | 3 |
| roll_greedy/pair_ok | +0.1170 | 0.0184 | 0.0239 | 3 |

## pin_v41_minus_LLM

Pairs: v41_pin_s0_6M / ref_vs_pin_llm_s0_6M, v41_pin_s1_6M / ref_vs_pin_llm_s1_6M, v41_pin_s2_6M / ref_vs_pin_llm_s2_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | -0.0869 | 0.0033 | 0.0106 | 3 |
| heldin/act_opt_b2 | -0.1505 | 0.0114 | 0.0203 | 3 |
| heldin/ntp_action | +0.1442 | 0.0035 | 0.0151 | 3 |
| heldout/act_opt | -0.0776 | 0.0032 | 0.0125 | 3 |
| heldout/act_opt_b2 | -0.1228 | 0.0115 | 0.0240 | 3 |
| heldout/ntp_action | +0.1406 | 0.0034 | 0.0176 | 3 |
| long/act_opt | -0.0648 | 0.0042 | 0.0054 | 3 |
| long/act_opt_b2 | -0.0911 | 0.0155 | 0.0185 | 3 |
| long/ntp_action | -1.1751 | 0.0302 | 0.4081 | 3 |
| roll_temp1/task | -0.0013 | 0.0013 | 0.0013 | 3 |
| roll_temp1/comp | -0.0039 | 0.0023 | 0.0023 | 3 |
| roll_temp1/complete | +0.0000 | 0.0045 | 0.0045 | 3 |
| roll_temp1/pair_ok | -0.1483 | 0.0144 | 0.0607 | 3 |
| roll_greedy/task | -0.0078 | 0.0032 | 0.0060 | 3 |
| roll_greedy/comp | -0.0221 | 0.0053 | 0.0143 | 3 |
| roll_greedy/complete | -0.2148 | 0.0168 | 0.0045 | 3 |
| roll_greedy/pair_ok | -0.1836 | 0.0183 | 0.0659 | 3 |

## pin_stack_minus_v41

Pairs: v41_stack_pin_s0_6M / v41_pin_s0_6M, v41_stack_pin_s1_6M / v41_pin_s1_6M, v41_stack_pin_s2_6M / v41_pin_s2_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | -0.0306 | 0.0034 | 0.0106 | 3 |
| heldin/act_opt_b2 | -0.0735 | 0.0126 | 0.0353 | 3 |
| heldin/ntp_action | +0.0794 | 0.0034 | 0.0344 | 3 |
| heldout/act_opt | -0.0327 | 0.0029 | 0.0129 | 3 |
| heldout/act_opt_b2 | -0.0867 | 0.0129 | 0.0467 | 3 |
| heldout/ntp_action | +0.0831 | 0.0031 | 0.0338 | 3 |
| long/act_opt | -0.0354 | 0.0039 | 0.0100 | 3 |
| long/act_opt_b2 | -0.0738 | 0.0180 | 0.0464 | 3 |
| long/ntp_action | +0.0511 | 0.0065 | 0.0281 | 3 |
| intervention/core_off/act_opt | -0.0830 | 0.0046 | 0.0505 | 3 |
| intervention/core_off/act_opt_b2 | -0.1289 | 0.0161 | 0.0282 | 3 |
| intervention/core_off/ntp_action | +3.2177 | 0.0283 | 4.0747 | 3 |
| intervention/M1/act_opt | -0.0335 | 0.0028 | 0.0133 | 3 |
| intervention/M1/act_opt_b2 | -0.0889 | 0.0130 | 0.0500 | 3 |
| intervention/M1/ntp_action | +0.0836 | 0.0031 | 0.0332 | 3 |
| intervention/loop_off/act_opt | -0.0373 | 0.0030 | 0.0138 | 3 |
| intervention/loop_off/act_opt_b2 | -0.1006 | 0.0130 | 0.0508 | 3 |
| intervention/loop_off/ntp_action | +0.0841 | 0.0031 | 0.0338 | 3 |
| intervention/u_off/act_opt | -0.0767 | 0.0039 | 0.0170 | 3 |
| intervention/u_off/act_opt_b2 | -0.1606 | 0.0163 | 0.0577 | 3 |
| intervention/u_off/ntp_action | +0.2223 | 0.0062 | 0.0795 | 3 |
| intervention/c_off/act_opt | -0.0351 | 0.0032 | 0.0120 | 3 |
| intervention/c_off/act_opt_b2 | -0.0833 | 0.0137 | 0.0434 | 3 |
| intervention/c_off/ntp_action | +0.1008 | 0.0035 | 0.0362 | 3 |
| intervention/core_shared_off/act_opt | -0.0361 | 0.0029 | 0.0132 | 3 |
| intervention/core_shared_off/act_opt_b2 | -0.0883 | 0.0132 | 0.0492 | 3 |
| intervention/core_shared_off/ntp_action | +0.0962 | 0.0034 | 0.0386 | 3 |
| intervention/enc_shared_off/act_opt | -0.0322 | 0.0030 | 0.0125 | 3 |
| intervention/enc_shared_off/act_opt_b2 | -0.0817 | 0.0129 | 0.0434 | 3 |
| intervention/enc_shared_off/ntp_action | +0.0788 | 0.0032 | 0.0329 | 3 |
| intervention/R_local/act_opt | +0.0610 | 0.0041 | 0.0231 | 3 |
| intervention/R_local/act_opt_b2 | +0.0011 | 0.0114 | 0.0302 | 3 |
| intervention/R_local/ntp_action | -1.1776 | 0.0234 | 1.3839 | 3 |
| intervention/core_off_R_local/act_opt | -0.0200 | 0.0046 | 0.0297 | 3 |
| intervention/core_off_R_local/act_opt_b2 | -0.0389 | 0.0120 | 0.0248 | 3 |
| intervention/core_off_R_local/ntp_action | +2.8700 | 0.0306 | 2.1551 | 3 |
| intervention/engram_off/act_opt | -0.0397 | 0.0032 | 0.0102 | 3 |
| intervention/engram_off/act_opt_b2 | -0.1189 | 0.0130 | 0.0420 | 3 |
| intervention/engram_off/ntp_action | +0.0823 | 0.0035 | 0.0275 | 3 |
| roll_temp1/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/complete | +0.0000 | 0.0045 | 0.0000 | 3 |
| roll_temp1/pair_ok | -0.0222 | 0.0132 | 0.0134 | 3 |
| roll_greedy/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/complete | -0.1276 | 0.0196 | 0.0440 | 3 |
| roll_greedy/pair_ok | -0.0659 | 0.0189 | 0.0140 | 3 |

## pin_loopoff_minus_v41

Pairs: v41_loopoff_pin_s0_6M / v41_pin_s0_6M, v41_loopoff_pin_s1_6M / v41_pin_s1_6M, v41_loopoff_pin_s2_6M / v41_pin_s2_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | +0.0013 | 0.0028 | 0.0062 | 3 |
| heldin/act_opt_b2 | +0.0046 | 0.0082 | 0.0064 | 3 |
| heldin/ntp_action | +0.0036 | 0.0025 | 0.0130 | 3 |
| heldout/act_opt | +0.0048 | 0.0027 | 0.0072 | 3 |
| heldout/act_opt_b2 | +0.0072 | 0.0087 | 0.0190 | 3 |
| heldout/ntp_action | +0.0027 | 0.0024 | 0.0104 | 3 |
| long/act_opt | +0.0061 | 0.0035 | 0.0095 | 3 |
| long/act_opt_b2 | +0.0174 | 0.0124 | 0.0111 | 3 |
| long/ntp_action | +0.0066 | 0.0048 | 0.0162 | 3 |
| intervention/core_off/act_opt | -0.0024 | 0.0037 | 0.0247 | 3 |
| intervention/core_off/act_opt_b2 | +0.0350 | 0.0125 | 0.0017 | 3 |
| intervention/core_off/ntp_action | -1.0925 | 0.0245 | 1.0943 | 3 |
| intervention/M1/act_opt | +0.0036 | 0.0027 | 0.0058 | 3 |
| intervention/M1/act_opt_b2 | +0.0083 | 0.0087 | 0.0197 | 3 |
| intervention/M1/ntp_action | +0.0026 | 0.0024 | 0.0107 | 3 |
| intervention/loop_off/act_opt | +0.0040 | 0.0027 | 0.0070 | 3 |
| intervention/loop_off/act_opt_b2 | +0.0044 | 0.0090 | 0.0202 | 3 |
| intervention/loop_off/ntp_action | +0.0026 | 0.0024 | 0.0102 | 3 |
| intervention/u_off/act_opt | -0.0121 | 0.0030 | 0.0095 | 3 |
| intervention/u_off/act_opt_b2 | -0.0050 | 0.0091 | 0.0247 | 3 |
| intervention/u_off/ntp_action | +0.0322 | 0.0030 | 0.0256 | 3 |
| intervention/c_off/act_opt | +0.0033 | 0.0027 | 0.0055 | 3 |
| intervention/c_off/act_opt_b2 | +0.0128 | 0.0087 | 0.0140 | 3 |
| intervention/c_off/ntp_action | +0.0021 | 0.0025 | 0.0115 | 3 |
| intervention/core_shared_off/act_opt | +0.0040 | 0.0028 | 0.0071 | 3 |
| intervention/core_shared_off/act_opt_b2 | +0.0039 | 0.0092 | 0.0247 | 3 |
| intervention/core_shared_off/ntp_action | +0.0064 | 0.0025 | 0.0104 | 3 |
| intervention/enc_shared_off/act_opt | +0.0031 | 0.0027 | 0.0071 | 3 |
| intervention/enc_shared_off/act_opt_b2 | +0.0033 | 0.0095 | 0.0228 | 3 |
| intervention/enc_shared_off/ntp_action | +0.0052 | 0.0025 | 0.0089 | 3 |
| intervention/R_local/act_opt | +0.0387 | 0.0035 | 0.0253 | 3 |
| intervention/R_local/act_opt_b2 | -0.0094 | 0.0096 | 0.0073 | 3 |
| intervention/R_local/ntp_action | -1.2931 | 0.0220 | 1.2965 | 3 |
| intervention/core_off_R_local/act_opt | -0.0074 | 0.0038 | 0.0049 | 3 |
| intervention/core_off_R_local/act_opt_b2 | +0.0244 | 0.0101 | 0.0164 | 3 |
| intervention/core_off_R_local/ntp_action | -1.4367 | 0.0275 | 0.8204 | 3 |
| intervention/engram_off/act_opt | +0.0015 | 0.0028 | 0.0063 | 3 |
| intervention/engram_off/act_opt_b2 | -0.0056 | 0.0092 | 0.0315 | 3 |
| intervention/engram_off/ntp_action | -0.0041 | 0.0026 | 0.0034 | 3 |
| roll_temp1/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_temp1/complete | +0.0039 | 0.0039 | 0.0039 | 3 |
| roll_temp1/pair_ok | +0.0056 | 0.0133 | 0.0143 | 3 |
| roll_greedy/task | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/comp | +0.0000 | 0.0000 | 0.0000 | 3 |
| roll_greedy/complete | -0.0312 | 0.0185 | 0.0772 | 3 |
| roll_greedy/pair_ok | -0.0449 | 0.0179 | 0.0143 | 3 |

## text_v41_minus_noR

Pairs: v41_text_s0_5M / ref_vs_text_v4_s0, v41_text_s1_5M / ref_vs_text_v4_s1

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| row_ntp | -0.0107 | 0.0018 | 0.0034 | 2 |
| probe/lr_carry3 | +0.0000 | 0.0000 | 0.0000 | 2 |
| probe/lr_multi4 | +0.0000 | 0.0000 | 0.0000 | 2 |
| probe/lr_recall1 | +0.0026 | 0.0058 | 0.0026 | 2 |

## text_v41_minus_LLM

Pairs: v41_text_s0_5M / ref_vs_text_llm_s0, v41_text_s1_5M / ref_vs_text_llm_s1

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| row_ntp | -0.0177 | 0.0040 | 0.0011 | 2 |
| probe/lr_carry3 | +0.0000 | 0.0000 | 0.0000 | 2 |
| probe/lr_multi4 | +0.0000 | 0.0000 | 0.0000 | 2 |
| probe/lr_recall1 | +0.0026 | 0.0058 | 0.0026 | 2 |

## fresh_LLM_minus_reference

Pairs: fresh_pin_llm_s0_6M / ref_vs_pin_llm_s0_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | +0.0000 | 0.0000 | unavailable | 1 |
| heldin/act_opt_b2 | +0.0000 | 0.0000 | unavailable | 1 |
| heldin/ntp_action | +0.0000 | 0.0000 | unavailable | 1 |
| heldout/act_opt | +0.0000 | 0.0000 | unavailable | 1 |
| heldout/act_opt_b2 | +0.0000 | 0.0000 | unavailable | 1 |
| heldout/ntp_action | +0.0000 | 0.0000 | unavailable | 1 |
| long/act_opt | +0.0000 | 0.0000 | unavailable | 1 |
| long/act_opt_b2 | +0.0000 | 0.0000 | unavailable | 1 |
| long/ntp_action | +0.0000 | 0.0000 | unavailable | 1 |
| roll_temp1/task | +0.0000 | 0.0000 | unavailable | 1 |
| roll_temp1/comp | +0.0000 | 0.0000 | unavailable | 1 |
| roll_temp1/complete | +0.0000 | 0.0000 | unavailable | 1 |
| roll_temp1/pair_ok | +0.0000 | 0.0000 | unavailable | 1 |
| roll_greedy/task | +0.0000 | 0.0000 | unavailable | 1 |
| roll_greedy/comp | +0.0000 | 0.0000 | unavailable | 1 |
| roll_greedy/complete | +0.0000 | 0.0000 | unavailable | 1 |
| roll_greedy/pair_ok | +0.0000 | 0.0000 | unavailable | 1 |

## pin_v41_minus_fresh_LLM

Pairs: v41_pin_s0_6M / fresh_pin_llm_s0_6M

| Metric | Difference | Paired item SE | Seed SE | Seeds |
|---|---:|---:|---:|---:|
| heldin/act_opt | -0.0717 | 0.0058 | unavailable | 1 |
| heldin/act_opt_b2 | -0.1302 | 0.0197 | unavailable | 1 |
| heldin/ntp_action | +0.1196 | 0.0062 | unavailable | 1 |
| heldout/act_opt | -0.0643 | 0.0055 | unavailable | 1 |
| heldout/act_opt_b2 | -0.1067 | 0.0207 | unavailable | 1 |
| heldout/ntp_action | +0.1156 | 0.0061 | unavailable | 1 |
| long/act_opt | -0.0616 | 0.0079 | unavailable | 1 |
| long/act_opt_b2 | -0.1146 | 0.0277 | unavailable | 1 |
| long/ntp_action | -0.7663 | 0.0375 | unavailable | 1 |
| roll_temp1/task | +0.0000 | 0.0000 | unavailable | 1 |
| roll_temp1/comp | -0.0078 | 0.0055 | unavailable | 1 |
| roll_temp1/complete | -0.0078 | 0.0055 | unavailable | 1 |
| roll_temp1/pair_ok | -0.1220 | 0.0243 | unavailable | 1 |
| roll_greedy/task | -0.0039 | 0.0039 | unavailable | 1 |
| roll_greedy/comp | -0.0078 | 0.0055 | unavailable | 1 |
| roll_greedy/complete | -0.2227 | 0.0278 | unavailable | 1 |
| roll_greedy/pair_ok | -0.1388 | 0.0322 | unavailable | 1 |
| intervention/M1/act_opt | -0.0237 | 0.0058 | unavailable | 1 |
| intervention/M1/act_opt_b2 | -0.0333 | 0.0222 | unavailable | 1 |
| intervention/M1/ntp_action | +0.0480 | 0.0071 | unavailable | 1 |
| intervention/loop_off/act_opt | -0.0123 | 0.0065 | unavailable | 1 |
| intervention/loop_off/act_opt_b2 | +0.0783 | 0.0264 | unavailable | 1 |
| intervention/loop_off/ntp_action | -0.0735 | 0.0139 | unavailable | 1 |

