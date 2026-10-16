# Recurrent S2 MPS evaluation recovery — 9 October 2026

The first `S2_K1_rec_s911_measure` failed at 06:57 UTC with MPS OOM
(2.46 GiB tensors + 17.65 GiB other allocations, cap 20.13 GiB) during
`self_states`. Training had completed all 12,000 steps. The watchdog resumed
the evaluator once at 06:58:55; its driver and child were stopped gracefully at
07:04:58 for diagnosis. No checkpoint or completed measurement was removed.

**The Q3 `stopped_by_user` status is the queue's generic SIGTERM status, used
here for an authorized operational repair. The human has not paused the run.**
See `recovery_events.jsonl` for the operational-stop record.

Two singleton jobs handle the repair:

- `probe_i22.py`: a full production-size scratch evaluation of the same
  checkpoint at recurrent batch 16. Results stay in an isolated temporary
  directory and must never be copied into production results. Its progress and
  observed MPS memory are in `i22_probe.json` and `i22_probe.out`.
- `revalidate_i22.py`: waits for successful probe completion and the completed
  CPU batch-parity check, backs up shared source/locks/test records and completed
  work hashes, then applies only the recurrent K=1 evaluation batch reduction.
  It stops waiting Q4–Q6 drivers operationally, reruns existing contracts/smokes
  on CPU in isolated temporary runs, and resumes each block using
  `launch_block.py` only after its physical hashes and test identities match.
  Q3 resumes first; the helper verifies its child relationship and renewed
  heartbeat. Q4–Q6 validation continues while Q3 uses MPS. QF keeps waiting.
  State, current child/log and resumed PIDs are in `i22_validation.json`.

Small CPU comparisons already passed: the trained recurrent K=1 checkpoint
gave identical greedy tokens and selected self states at batch 64 versus 16
on 20 standard and 20 hard14 items (`i22_batch_parity.json`). This does not
claim bit-identical MPS outputs; rare floating-point ties can differ.

Do not start another probe, validator, driver or evaluator while these jobs
are alive. Use actual process commands, parents and state timestamps. If a job
fails, inspect its recorded error and fix the concrete cause. The watchdog
continues running every 900 seconds and honors the operationally stopped
drivers while validation is active. An explicit human pause/cancellation still
takes precedence; stop the recovery chain as well if one is requested.

After the edit, completed Q2 deliberately retains its original historical lock
and test records, backed up with the original shared source. Its results remain
valid and must not be overwritten. `collect_health.py` separates completed
locks from pending locks. All pending affected blocks require new legitimate
contract and smoke validation; no hash or preflight is bypassed.

This is an active repair, not a confirmed production recovery, until
`i22_validation.json` records the renewed Q3 heartbeat. Notify once when the
repair starts, again upon actual recovery or a new failure, and stay quiet
while unchanged validation is progressing.

Confirmed 2026-10-09T09:47:56.294141+00:00: Q3 official recurrent K1 evaluation completed successfully and K8 training progressed; Q3–Q6 CPU contracts and full smoke workflows passed, all pending lock inputs match, and Q4–QF drivers are alive waiting for dependencies. All 93 protected completed artifacts were preserved.
