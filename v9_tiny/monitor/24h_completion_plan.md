# V9: a plan to target completion within 24 hours

Prepared 9 October 2026, about 07:45 IST. This is a proposed execution plan, not a confirmed finish time. The target is approximately 07:45 IST on 10 October, measured from this planning window. No production queue, scientific setting, lock, or completed result has been changed for this plan.

## Feasibility

The current serial queue on the M2 MacBook Air cannot reliably meet this target. The best route is to finish Q2 on the Mac while preparing 4–8 dedicated NVIDIA GPU workers for Q3 through QF. Use one independent job per physical GPU. Four GPUs are the initial sizing assumption; additional identical GPUs reduce the two waves of baseline jobs and provide recovery capacity. The GPU count is a planning assumption until hardware access and a spending limit are supplied.

A 24-hour completion is plausible only if setup, validation, and representative timing measurements pass promptly. Neither CUDA throughput nor production Q5 search time has been measured. The schedule below allocates time; it does not assert a measured speedup.

## Work remaining

At the latest check, Q2 has 13 fully trained and evaluated runs. Seed 932's always-open colourcount training is underway. Q1 and Q3's dense tier are complete.

| Block | Required remaining work |
|---|---|
| Q2 | Finish seed 932's always-open colourcount and VQ colourcount, then all three bind arms and their evaluations; finish the decisions and Q2 result |
| Q3 | K1/K8 recurrent models for seeds 911/912: four 12,000-step runs; three mandatory dense contingencies for seed 911: K8m1, K8mK, K4, also 12,000 steps each; 6,000-step recurrent replay; all measurements and register decisions |
| Q4 | Decoder-only and joint refitting for seeds 921/922: four 6,000-step runs, transplant curves, full measurements, and G3 decisions |
| Q5 | Two seed-specific expert-iteration jobs, each iteration 0 plus four iterations; six matched baseline jobs (RFT, GRPO, REINFORCE); development evaluations and GradCuit pilots |
| Q6 | Development choices and fixed sealed procedures; currently three permitted primary methods × two suites × three budgets × two seeds = 36 primary cells, plus the required token-value fits and answer-loop cells. Recalculate the method inventory from the unchanged register before dispatch |
| QF | Held-out greedy evaluations of every Q5 arm for both seeds, statistical analysis, final JSON and report |

Q3's three contingency arms are required by the already-observed dense register, not optional work to omit to hit a deadline. Q5/Q6 follow the registered gate-derived branches and sizing rules. The counts above do not authorize changing those rules.

## Proposed schedule

Times are elapsed hours from starting the accelerated plan. Q2 stays on the Mac while remote setup proceeds. Later stages retain their result dependencies.

| Window | Work and completion checkpoint |
|---|---|
| 0–3 | Finish Q2 locally. Provision workers, copy stable inputs, validate the CUDA execution path and dispatcher, and measure representative training, evaluation, and search costs on development fixtures |
| 3–7 | Finish Q3 on separate GPUs. First wave: four recurrent models. Second wave: three dense contingencies and recurrent replay. Evaluate each checkpoint on its assigned exclusive worker; consolidate the Q3 result |
| 7–9 | Run the four Q4 arms on four GPUs after the transplant inputs are ready; evaluate, apply G3, and write the model-selection result |
| 9–19 | Run the two Q5 seed branches independently. Once a seed's complete EI schedule exists, dispatch that seed's three matched baselines independently. With four GPUs, six baseline jobs need two waves; with six or more GPUs they can share one wave after both EI schedules finish. Run required evaluations and pilots; consolidate Q5 |
| 19–22 | Finish Q6 development choices first. Commit each seed's procedure before any held-out item is read. Dispatch permitted sealed cells across exclusive workers using fixed timing caches and identical hardware |
| 22–24 | Run QF's held-out evaluations, verify all expected artifacts and completed exposures, generate the report, and retain time for retrying an operational failure |

This schedule requires Q2 to fit roughly three hours and the accelerated Q3/Q4 portions to fit six hours combined. If Q2 takes four hours, the available buffer shrinks by an hour. Do not label a partially finished stage complete at hour 24.

## Compute and budget

Use full GPUs with enough VRAM for production-shaped evaluation. A6000 48 GB workers are a cost-oriented candidate; H100 workers are an alternative if benchmark results justify the price. Retain float32 math. GPU marketing throughput, especially lower-precision Tensor Core figures, is not a runtime estimate for this 2.45M-parameter, Python-controlled workload.

Lambda's published A6000 rate is $1.09 per GPU-hour. Four GPUs for a full 24 hours would be $104.64; eight would be $209.28, before applicable taxes and storage. Its four-GPU H100 rate is $4.09 per GPU-hour, or $392.64 for four GPUs for 24 hours. These are arithmetic cost ceilings at published rates, not confirmed availability or a purchase proposal. Workers can be released when their work is verified complete. Source: [Lambda pricing](https://lambda.ai/pricing), checked 9 October 2026.

Plan around 32 or more host CPU cores and ample RAM for four concurrent workers, or adequately resourced individual instances. Each worker needs its own compiler cache and four CPU execution threads initially. The current local dependency versions are torch 2.10.0, numpy 1.26.4, and tokenizers 0.22.1; use the corresponding CUDA-capable PyTorch distribution and record the complete environment.

## Required preparation before moving production work

1. **Preserve inputs and paths.** Copy completed checkpoints, the frozen code/data, spec/locks/tests, the teacher and replay dependencies, and TinyStories arrays. The measured local components total roughly a few GB. Mirror the recorded project paths on the remote filesystem so existing checkpoint/source references stay intact. Verify actual content hashes; do not rewrite completed results to repair paths. Copy Q2's final result only after it is committed locally.
2. **Validate CUDA timing.** `frozen/code/runtime.py` selects MPS or CPU by default; `V8_DEVICE=cuda` can select CUDA, but its `sync()` only synchronizes MPS. Add a separately recorded operational execution wrapper that synchronizes CUDA before timing. Keep frozen source files byte-identical. Validate all imported timing paths before the S4 sizing pilot and S5 budget calibration. [PyTorch documents CUDA synchronization](https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html).
3. **Preserve numerical settings.** Disable TF32 and mixed precision in the new execution environment, retain float32, and validate eager/compiled output and gradient tolerances, teacher/core ownership, data order, seeded CPU sampling, and stop/verifier behavior. Cross-device results need not be bit-identical; record that completed MPS runs and new CUDA runs used different devices. Keep comparisons paired where the register requires it. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
4. **Validate the dispatcher.** The current drivers write one block-wide status and log and intentionally serialize work. Launching four copies of a block driver would collide and create duplicates. Implement a coordinator with a unique owner for every stage, exclusive output directories, per-job statuses, immutable input hashes, and a verified final merge. Use existing stage implementations and resume checks. Run the applicable contracts and end-to-end smoke workflows in isolated directories; any changed locked implementation requires legitimate validation and refreshed affected locks. Preserve the validated Q4 repair.
5. **Coordinate recovery and handoff.** The current detached launcher uses macOS `caffeinate`; remote launch must use a Linux service/process supervisor. Before starting Q3–QF remotely, record their remote ownership and retire their waiting local drivers in a coordinated handoff. Update the watchdog so a remote PID is not tested as a local PID or restarted locally. Keep Q2's owner and watchdog on the Mac until it finishes. Do not stop or migrate the active Q2 training mid-phase.
6. **Keep timing comparisons fair.** Each S5 baseline and method must use a timing reference from the same GPU type and execution environment. Calibrate without competing work on the same physical GPU. Reuse a committed cache for all comparable methods; do not pool MPS and CUDA timing claims. Keep each seed's method evaluations on the same worker where possible.

## Measured feasibility checks

Before accepting the deadline, build a projection from warm, representative development runs and the actual number of remaining jobs. Check separately: steady-state updates/second, compilation/startup time, complete evaluation time, S4 iteration-0 search, later-iteration search under the permitted policy, verifier calls, baseline samples/second, and batched greedy time per item for both hard suites.

For each stage, estimate the longest dependency chain and the load on its busiest worker. Parallel GPU count divides independent jobs, not a sequential training trajectory. Include validation, file transfer, pilot fitting, model loading, final analysis, and retries.

- **At hour 2:** CUDA contracts and smoke checks must pass, and the projected remaining critical path must fit the time left with about two hours reserved for recovery. If it does not, revise the ETA immediately.
- **At the first complete Q5 iteration:** use its actual verifier calls to project all six matched baselines. A search-only pilot does not predict the full Q5 duration. If baseline queues exceed the allocation, add identical workers only within the user's compute/budget limit, or report that 24 hours is no longer credible.
- **Before Q6's first held-out read:** freeze the complete permitted method inventory, timing caches, and procedures. Do not choose methods or thresholds after seeing held-out results.
- **At hour 22:** verify the outstanding artifact inventory. The completion notification requires Q1 through QF and both final outputs to be verified complete.

## If only the current Mac is available

Continue the registered queue and recover failures promptly. Avoid concurrent MPS trainers, which share this machine's one GPU and increase memory pressure. Profile operational bottlenecks on development fixtures and test any proposed optimization before applying it. Adding inference caches, changing decoding, reducing seeds or steps, dropping required arms, lowering evaluation counts, or changing budgets is not part of this plan.

A useful progress report can be produced after 24 hours on the Mac, but completing the full registered experiment within that period is not a credible commitment. A smaller study would need an explicit scope decision and a separate, clearly labeled protocol.

## Next input needed

Which GPU hardware is available, or what maximum cloud-compute budget should the plan assume? The answer is needed to execute the accelerated plan. Planning has not rented hardware, uploaded the experiment, or changed the running queue.
