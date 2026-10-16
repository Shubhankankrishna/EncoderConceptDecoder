# I23: Q5 search memory recovery — 10 October 2026

Q5 seed 921 iteration 0 exited at 09:56:45 UTC (15:26:45 IST). Torch 2.10 MPS reported 2.34 GiB of live tensors plus 17.78 GiB of other allocations at its 20.13 GiB limit. No Q5 iteration had yet been saved. Q1–Q4, the Q5 sizing pilot, and the greedy reference remain intact.

The original-code first-frontier probe reproduced retention: after one complete RH step on the original 64-problem frontier, 416 discarded tree nodes remained alive, with 120.8 MB of tensor allocations and 5.484 GB of driver allocations. Explicit cycle collection removed all nodes and reduced tensor allocations to 13.3 MB. `empty_cache()` reduced driver allocations to 4.209 GB, leaving substantial process-local cache allocations. Raising or disabling the MPS memory limit is not part of this repair.

The repair collects discarded trees at completed search boundaries. Q5 additionally serializes CPU problem records and per-problem NumPy RNG states between sequential worker processes. A worker completes entire RH steps and hands its state back once driver allocations reach 4 GB; unfinished problems continue in a fresh worker with the same configuration. This releases Torch 2.10's process-local Metal graph/shape caches. The search frontier remains 64. Architecture, weights, seeds, 512/4096 problem counts, 32-expansion budget, hop limits, gates, and baseline matching rules are unchanged. Worker wall time contributes to measured search time.

The watchdog was stopped operationally at 10:01 UTC to prevent blind retries. The Q6 waiting driver was also stopped before its shared pending source changed; QF remains waiting. This is not a user pause. `recover_q5_memory.py` owns validation and restoration under a file lock, with live state in `q5_memory_recovery.json`.

The first validation attempt proved exact CPU record/RNG/count parity, then GPU inspection caught the guard comparing a device string to `mps` while Torch returned `mps:0`. That attempt was stopped as one process group and archived. The corrected worker uses `torch.device(...).type`; validation was restarted before any official lock or production launch changed. The launcher also rejects any surviving Q5 trainer or search worker before starting a new Q5 driver.

Stopping that probe also exposed a reentrant `Popen.wait()` in the parent's signal handler. The handler now terminates the worker and unwinds the active wait before `finally` reaps it. The isolated real-worker SIGTERM test passed at 10:28 UTC: parent exit 143 and both PIDs gone. Attempt logs are archived in the corresponding repair backup directories.

Validation requires exact CPU parity against archived search records (including RNG state and verifier counts), repeated CPU worker handoffs, the existing Q5 and Q6 contracts, their complete combined smoke workflow in isolated CPU runs/spec/test directories, an isolated full 64-problem MPS frontier, and physical data/completed-output hashes. Only then may new Q5/Q6 locks and test records be published. The helper resumes Q5/Q6 through `launch_block.py`, restores the validated watchdog launcher, and checks the actual child process and renewed heartbeat. Completed historical locks and results are preserved.

The upstream PyTorch discussion describes process-local shape-cache retention in varying-shape MPS inference on Torch 2.10: https://github.com/pytorch/pytorch/issues/181213. Local measured allocation behavior above is the evidence for this experiment's repair.

Current completion/validation state is authoritative in `q5_memory_recovery.json`; this note does not claim validation or recovery has finished.
