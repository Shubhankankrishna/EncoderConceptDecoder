# Long-action likelihood decomposition

Per-episode means on the same 64 long episodes. Before/after refers to six completed visits, the training horizon.

| Model | Seed | Phase | Optimal accuracy | All-vocabulary NLL | Action-conditioned NLL | Non-action penalty | END probability | Action probability mass |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v41_pin_s0_6M | 0 | before_visit6 | 0.7466 | 0.7485 | 0.7377 | 0.0108 | 0.0066 | 0.9930 |
| v41_pin_s0_6M | 0 | after_visit6 | 0.7418 | 1.2844 | 0.7592 | 0.5251 | 0.1399 | 0.8590 |
| v41_pin_s1_6M | 1 | before_visit6 | 0.7426 | 0.7542 | 0.7464 | 0.0077 | 0.0051 | 0.9945 |
| v41_pin_s1_6M | 1 | after_visit6 | 0.7514 | 1.2263 | 0.7286 | 0.4977 | 0.1310 | 0.8682 |
| v41_pin_s2_6M | 2 | before_visit6 | 0.7273 | 0.7615 | 0.7527 | 0.0088 | 0.0056 | 0.9940 |
| v41_pin_s2_6M | 2 | after_visit6 | 0.7403 | 1.2139 | 0.7559 | 0.4581 | 0.1389 | 0.8601 |
| ref_vs_pin_llm_s0_6M | 0 | before_visit6 | 0.8155 | 0.6213 | 0.6208 | 0.0005 | 0.0001 | 0.9995 |
| ref_vs_pin_llm_s0_6M | 0 | after_visit6 | 0.7953 | 2.9546 | 0.6676 | 2.2869 | 0.1899 | 0.8050 |
| ref_vs_pin_llm_s1_6M | 1 | before_visit6 | 0.8248 | 0.6033 | 0.6030 | 0.0003 | 0.0000 | 0.9997 |
| ref_vs_pin_llm_s1_6M | 1 | after_visit6 | 0.8182 | 5.3960 | 0.6191 | 4.7769 | 0.3210 | 0.6759 |
| ref_vs_pin_llm_s2_6M | 2 | before_visit6 | 0.8429 | 0.5727 | 0.5722 | 0.0005 | 0.0000 | 0.9995 |
| ref_vs_pin_llm_s2_6M | 2 | after_visit6 | 0.7387 | 2.9503 | 0.8387 | 2.1116 | 0.3134 | 0.6593 |
| ref_vs_pin_v4_s0_6M | 0 | before_visit6 | 0.6915 | 0.8952 | 0.8706 | 0.0246 | 0.0214 | 0.9783 |
| ref_vs_pin_v4_s0_6M | 0 | after_visit6 | 0.6791 | 1.2137 | 0.9096 | 0.3041 | 0.2148 | 0.7841 |
| ref_vs_pin_v4_s1_6M | 1 | before_visit6 | 0.6937 | 0.8919 | 0.8700 | 0.0218 | 0.0185 | 0.9811 |
| ref_vs_pin_v4_s1_6M | 1 | after_visit6 | 0.6942 | 1.2395 | 0.9039 | 0.3355 | 0.2259 | 0.7733 |
| ref_vs_pin_v4_s2_6M | 2 | before_visit6 | 0.7066 | 0.8939 | 0.8731 | 0.0208 | 0.0177 | 0.9819 |
| ref_vs_pin_v4_s2_6M | 2 | after_visit6 | 0.6892 | 1.2309 | 0.9125 | 0.3184 | 0.2191 | 0.7802 |
| exploratory_mps_plan_pin_s0_6M | 0 | before_visit6 | 0.7478 | 0.7539 | 0.7439 | 0.0100 | 0.0062 | 0.9934 |
| exploratory_mps_plan_pin_s0_6M | 0 | after_visit6 | 0.7387 | 1.2642 | 0.7637 | 0.5005 | 0.1357 | 0.8627 |
| exploratory_noplan_pin_s0_6M | 0 | before_visit6 | 0.8060 | 0.6516 | 0.6420 | 0.0095 | 0.0059 | 0.9936 |
| exploratory_noplan_pin_s0_6M | 0 | after_visit6 | 0.8107 | 1.0670 | 0.6664 | 0.4006 | 0.1110 | 0.8875 |
