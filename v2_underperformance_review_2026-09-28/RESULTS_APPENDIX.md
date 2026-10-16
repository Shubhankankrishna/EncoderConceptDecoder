# Numerical evidence appendix

Frozen logs: 2026-09-28T22:34:33.473495+05:30. NTP is nats per valid token. Reported training throughput counts sequence slots, including padding.

## All v2 training summaries

| Step | NTP | NCP | VQ | NextLat h | NextLat KL | Consistency | LB | z | Drop % | Load entropy | Route change % | Grad norm | Slots/s | Swap MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 8.8964 | 4.7626 | 2.9728 | 0.5308 | 0.0960 | 0.0029 | 1.0258 | 7.7679 | 12.79 | 2.7453 | 0.97 | 0.9036 | 1378.2 | 1400.5 |
| 50 | 8.2750 | 4.7121 | 2.9114 | 0.2427 | 0.0346 | 0.0319 | 1.0712 | 7.5325 | 27.95 | 2.6861 | 8.82 | 1.3389 | 1402.1 | 1368.5 |
| 75 | 7.4927 | 4.5263 | 3.0514 | 0.0420 | 0.0055 | 0.0368 | 1.0301 | 6.8109 | 20.88 | 2.7262 | 12.84 | 1.3342 | 1321.9 | 1169.3 |
| 100 | 6.5769 | 4.3504 | 3.1465 | 0.0348 | 0.0100 | 0.0437 | 1.0173 | 5.5948 | 14.41 | 2.7364 | 13.02 | 1.2067 | 1227.5 | 1161.3 |
| 125 | 5.7350 | 4.1283 | 3.1406 | 0.0616 | 0.0405 | 0.0383 | 1.0146 | 4.2880 | 12.65 | 2.7418 | 10.95 | 0.8901 | 1074.8 | 2143.9 |
| 150 | 5.1205 | 3.9440 | 3.1760 | 0.0948 | 0.1028 | 0.0310 | 1.0151 | 3.8073 | 13.36 | 2.7366 | 9.28 | 0.6982 | 1159.7 | 3165.4 |
| 175 | 4.6442 | 3.8407 | 3.1793 | 0.1174 | 0.1587 | 0.0331 | 1.0169 | 3.4044 | 13.41 | 2.7317 | 9.21 | 0.6530 | 1129.6 | 3879.4 |
| 200 | 4.2412 | 3.9166 | 3.1869 | 0.1295 | 0.2134 | 0.0314 | 1.0217 | 3.0106 | 14.68 | 2.7208 | 8.49 | 1.1779 | 1189.7 | 3993.5 |
| 225 | 4.0791 | 4.0193 | 2.9821 | 0.1372 | 0.2368 | 0.0264 | 1.0281 | 2.9749 | 16.40 | 2.7123 | 8.87 | 0.9983 | 1162.1 | 4230.1 |
| 250 | 3.9244 | 4.0925 | 3.0632 | 0.1419 | 0.2423 | 0.0217 | 1.0346 | 2.7131 | 17.76 | 2.6983 | 8.85 | 0.9189 | 1185.2 | 3895.1 |
| 275 | 3.7549 | 4.1337 | 3.1179 | 0.1385 | 0.2266 | 0.0230 | 1.0353 | 2.2769 | 16.60 | 2.7075 | 9.83 | 0.7533 | 1186.0 | 4715.2 |

Consistency, NCP, LB, and z are raw logged scalars before their loss weights. NCP is averaged over sampled passes; the next table is only the last optimizer step of each interval. VQ is codebook-only. Grad norm is the recorded preclip mean.

## Every logged per-pass NCP vector

| Step | Last K | NCP by pass |
|---:|---:|---|
| 25 | 8 | 4.7501, 4.7504, 4.7508, 4.7512, 4.7517, 4.7522, 4.7527, 4.7531 |
| 50 | 10 | 4.5903, 4.5908, 4.6010, 4.6152, 4.6294, 4.6419, 4.6523, 4.6608, 4.6676, 4.6731 |
| 75 | 12 | 4.3831, 4.3645, 4.3685, 4.3879, 4.4146, 4.4421, 4.4676, 4.4900, 4.5090, 4.5251, 4.5386, 4.5498 |
| 100 | 12 | 4.2128, 4.1955, 4.2019, 4.2251, 4.2562, 4.2881, 4.3171, 4.3421, 4.3630, 4.3802, 4.3944, 4.4059 |
| 125 | 10 | 3.9473, 3.9380, 3.9461, 3.9674, 3.9957, 4.0248, 4.0512, 4.0734, 4.0918, 4.1067 |
| 150 | 8 | 3.8518, 3.8407, 3.8451, 3.8631, 3.8896, 3.9181, 3.9444, 3.9669 |
| 175 | 8 | 3.7745, 3.7603, 3.7613, 3.7749, 3.7963, 3.8192, 3.8400, 3.8572 |
| 200 | 10 | 3.9875, 3.9722, 3.9731, 3.9873, 4.0091, 4.0311, 4.0496, 4.0639, 4.0746, 4.0826 |
| 225 | 8 | 4.0982, 4.0811, 4.0863, 4.1063, 4.1311, 4.1537, 4.1716, 4.1851 |
| 250 | 12 | 4.0914, 4.0776, 4.0842, 4.1022, 4.1241, 4.1436, 4.1587, 4.1696, 4.1774, 4.1830, 4.1872, 4.1901 |
| 275 | 9 | 4.0552, 4.0372, 4.0459, 4.0689, 4.0952, 4.1178, 4.1349, 4.1471, 4.1557 |

## Official early validation, all relevant points

| Run | Step | Hours | TinyStories | WikiText | Mean | Recall | Carry | Multi | Hop | Bind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_main | 185 | 1.0372 | 4.234765 | 5.943382 | 5.089073 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.046875 |
| v1_main | 265 | 1.5054 | 3.630569 | 5.495405 | 4.562987 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.031250 |
| v1_main | 365 | 2.0736 | 3.206009 | 5.038271 | 4.122140 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.085938 |
| v2 | 138 | 1.0455 | 5.797843 | 6.861666 | 6.329755 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.054688 |
| v2 | 267 | 2.0944 | 4.304051 | 5.968571 | 5.136311 | 0.007812 | 0.000000 | 0.000000 | 0.000000 | 0.007812 |

## Fixed-checkpoint language/depth and state trajectories

Eight fixed rows per language, IDs 19, 36, 37, 38, 43, 62, 89, 123. Each language score pools token-loss sums/counts. NCP and state summaries average eight B2 batches; they have a different denominator from language NTP. Eval mode has no capacity dropping.

| Checkpoint | R | TinyStories | WikiText | Mean NCP | Mean state norm | Mean update norm | Mean cosine to R1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 138 | 1 | 5.870554 | 6.996753 | 3.715498 | 28.472273 | 8.138757 | 1.000000 |
| 138 | 2 | 5.860082 | 6.977127 | 3.711894 | 23.515065 | 6.456193 | 0.987672 |
| 138 | 4 | 5.846886 | 6.953747 | 3.733289 | 24.021122 | 2.182312 | 0.934285 |
| 138 | 6 | 5.842068 | 6.943164 | 3.759346 | 24.222457 | 1.300626 | 0.887353 |
| 138 | 8 | 5.840462 | 6.938029 | 3.778332 | 24.301412 | 0.802842 | 0.856798 |
| 138 | 10 | 5.840053 | 6.935721 | 3.790298 | 24.338525 | 0.538424 | 0.838513 |
| 138 | 12 | 5.839928 | 6.934893 | 3.797454 | 24.357444 | 0.389535 | 0.827702 |
| 138 | 16 | 5.839975 | 6.935099 | 3.804165 | 24.372363 | 0.263432 | 0.817486 |
| 267 | 1 | 4.435553 | 6.095465 | 3.981841 | 25.626978 | 10.668858 | 1.000000 |
| 267 | 2 | 4.417016 | 6.086768 | 3.973476 | 28.226461 | 8.597631 | 0.956557 |
| 267 | 4 | 4.404205 | 6.081604 | 3.995712 | 33.146352 | 4.084324 | 0.830754 |
| 267 | 6 | 4.397627 | 6.079466 | 4.010657 | 34.522881 | 1.650432 | 0.772281 |
| 267 | 8 | 4.395487 | 6.078908 | 4.016923 | 34.723357 | 0.678765 | 0.753132 |
| 267 | 10 | 4.394719 | 6.078777 | 4.019343 | 34.746372 | 0.355121 | 0.747057 |
| 267 | 12 | 4.394434 | 6.078672 | 4.020400 | 34.751697 | 0.236880 | 0.744932 |
| 267 | 16 | 4.394206 | 6.078534 | 4.021677 | 34.750891 | 0.179897 | 0.743397 |

## Raw gradient agreement: every selected shared path

K10, capacity 1.25, four first microbatches at successive optimizer-step offsets from checkpoint 267. These are raw gradients, not Adam-preconditioned updates or full encoder parameter gradients. Norm/projection ratios are relative to NTP on the same parameter.

| Batch offset | Parameter | NextLat cosine | NextLat norm ratio | NextLat NTP-direction projection ratio | All-aux cosine |
|---:|---|---:|---:|---:|---:|
| 0 | concept_features | -0.162231 | 0.376316 | -0.061050 | -0.087325 |
| 0 | local_features | 0.020903 | 0.379582 | 0.007934 | 0.020903 |
| 0 | core_W_in | -0.437160 | 0.431580 | -0.188670 | -0.365950 |
| 0 | expert_w2 | -0.254131 | 0.570512 | -0.144985 | -0.237543 |
| 0 | decoder_Wu | -0.314074 | 0.496185 | -0.155839 | -0.314074 |
| 0 | carry_gain | -0.391293 | 0.469872 | -0.183858 | -0.349348 |
| 0 | router_layer2 | -0.006148 | 0.483695 | -0.002974 | -0.163257 |
| 32 | concept_features | -0.200304 | 0.438935 | -0.087920 | -0.117729 |
| 32 | local_features | 0.042112 | 0.350900 | 0.014777 | 0.042112 |
| 32 | core_W_in | -0.598654 | 0.420234 | -0.251575 | -0.575980 |
| 32 | expert_w2 | -0.461041 | 0.432697 | -0.199491 | -0.452104 |
| 32 | decoder_Wu | -0.501982 | 0.384585 | -0.193055 | -0.501982 |
| 32 | carry_gain | -0.569496 | 0.377098 | -0.214756 | -0.565942 |
| 32 | router_layer2 | -0.444925 | 0.277869 | -0.123631 | -0.508236 |
| 64 | concept_features | 0.066977 | 0.464332 | 0.031099 | 0.116560 |
| 64 | local_features | -0.033376 | 0.446588 | -0.014905 | -0.033376 |
| 64 | core_W_in | -0.386777 | 0.678168 | -0.262300 | -0.345568 |
| 64 | expert_w2 | -0.422018 | 0.761228 | -0.321252 | -0.416943 |
| 64 | decoder_Wu | -0.522567 | 0.642196 | -0.335591 | -0.522567 |
| 64 | carry_gain | -0.505349 | 0.750201 | -0.379113 | -0.489995 |
| 64 | router_layer2 | -0.559810 | 0.586460 | -0.328306 | -0.471985 |
| 96 | concept_features | -0.106959 | 0.503829 | -0.053889 | -0.035128 |
| 96 | local_features | -0.038991 | 0.485936 | -0.018947 | -0.038991 |
| 96 | core_W_in | -0.434691 | 0.611265 | -0.265711 | -0.392665 |
| 96 | expert_w2 | -0.384388 | 0.701341 | -0.269587 | -0.375886 |
| 96 | decoder_Wu | -0.452712 | 0.656274 | -0.297103 | -0.452712 |
| 96 | carry_gain | -0.467626 | 0.603193 | -0.282068 | -0.447760 |
| 96 | router_layer2 | -0.530220 | 0.636628 | -0.337553 | -0.435014 |

### Initial-state projection gradient presence

| Batch offset | NTP | NextLat h | NextLat KL | Final-window consistency | Weighted NCP norm | Weighted routing norm |
|---:|---|---|---|---|---:|---:|
| 0 | None | None | None | None | 0.00242288 | 0.00057275 |
| 32 | None | None | None | None | 0.00173874 | 0.00056021 |
| 64 | None | None | None | None | 0.00210475 | 0.00059725 |
| 96 | None | None | None | None | 0.00165132 | 0.00060058 |

`None` means no autograd path, not a rounded small value. All K in the configured range detach before final NTP.

## Real versus padding routing, every sampled batch row

| Batch offset | Row | Real chunks | PAD chunks | Real assignments dropped % | PAD assignments dropped % | Real chunks losing both experts % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 256 | 0 | 0.3076 | N/A | 0.0000 |
| 0 | 2 | 180 | 76 | 17.6597 | 59.0461 | 4.1944 |
| 32 | 1 | 214 | 42 | 3.8318 | 22.5000 | 0.0000 |
| 32 | 2 | 246 | 10 | 29.2378 | 66.6250 | 8.1809 |
| 64 | 1 | 178 | 78 | 0.0000 | 20.7692 | 0.0000 |
| 64 | 2 | 255 | 1 | 22.2794 | 85.0000 | 6.4706 |
| 96 | 1 | 234 | 22 | 2.2382 | 8.7500 | 0.0321 |
| 96 | 2 | 251 | 5 | 23.6355 | 37.7500 | 7.2311 |

A real chunk contains at least one non-PAD token. Percentages average the 4 layers × 10 passes for each fixed row. They do not imply that token targets are removed.

## Additional raw results

The complete capacity sweeps, individual-row losses, batch reversal, carry/attention/expert interventions, individual loss gradients, and next-latent PAD-pair fractions are in [checkpoint 138](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_000138_diagnostics.json) and [checkpoint 267](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/checkpoint_000267_diagnostics.json). No field is discarded from those records.

All v1 main and continuation train/eval records, all v2 records, guards, configurations, matched windows, and data continuity checks are in [snapshot.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/snapshot.json). The [earlier results appendix](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/RESULTS_APPENDIX.md) covers completed v1 evaluations, probes, ablations, and dataset quality. It is retained as context rather than used as an equal-training-budget score for early v2.
