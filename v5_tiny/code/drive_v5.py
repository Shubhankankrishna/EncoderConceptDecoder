"""Unattended driver for Phases B-E.  Each stage is idempotent: it skips work whose output file already exists.

usage: python drive_v5.py --stage b --steps_b 3500
       python drive_v5.py --stage b_eval | c | c_eval | d | e
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))
PY = sys.executable

B_VARIANTS = [("B_v2_s0", ["--variant", "det", "--xm", "0"]),
              ("B_v2a_s0", ["--variant", "adapter", "--xm", "0"]),
              ("B_v3_s0", ["--variant", "z", "--xm", "0"]),
              ("B_v6_s0", ["--variant", "gram", "--xm", "1"])]


def run_parallel(jobs, limit=1):
    """jobs: list of (name, argv, done_file).  Runs those not done, at most `limit` at a time (MPS contention is
    superlinear: 2 concurrent jobs ran at 0.6 s/step each vs 0.17 solo), waits for all."""
    if limit < len(jobs):
        for i in range(0, len(jobs), limit):
            run_parallel(jobs[i:i + limit], limit=len(jobs))
        return
    procs = []
    for name, argv, done in jobs:
        if os.path.exists(done):
            print(f"skip {name} (done)", flush=True); continue
        logf = open(os.path.join(RUNS, f"{name}.out"), "a")
        procs.append((name, subprocess.Popen([PY] + argv, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)))
        print(f"start {name}: {' '.join(argv)}", flush=True)
        time.sleep(2)
    bad = []
    for name, p in procs:
        rc = p.wait()
        print(f"end {name} rc={rc}", flush=True)
        if rc != 0:
            bad.append(name)
    if bad:
        print(f"FAILED: {bad}", flush=True)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--steps_b", type=int, default=3500)
    ap.add_argument("--steps_c", type=int, default=1500)
    ap.add_argument("--teacher", default="A_ncp_s0")
    a = ap.parse_args()
    if a.stage == "b":
        run_parallel([(n, ["phase_b.py"] + v + ["--teacher", a.teacher, "--steps", str(a.steps_b), "--out", n],
                       os.path.join(RUNS, n, "core.pt")) for n, v in B_VARIANTS])
    elif a.stage == "v0":
        run_parallel([("A_v0_s0", ["phase_a.py", "--arch", "v0", "--seed", "0", "--steps", "18000", "--eval_every", "1000",
                                   "--out", "A_v0_s0"], os.path.join(RUNS, "A_v0_s0", "train_summary.json"))])
        run_parallel([("A_v0_s0_evala", ["eval_a.py", "--run", "A_v0_s0"], os.path.join(RUNS, "A_v0_s0", "eval_a_val.json"))])
    elif a.stage == "b_eval":
        for n, v in B_VARIANTS:
            run_parallel([(f"{n}_evalb", ["eval_b.py", "--run", n], os.path.join(RUNS, n, "eval_b_val_lam1.0.json"))])
    elif a.stage == "c":
        jobs = []
        for src, dst, v in (("B_v6_s0", "C_v6_s0", ["--variant", "gram", "--xm", "1"]),
                            ("B_v3_s0", "C_v3_s0", ["--variant", "z", "--xm", "1"])):   # spec C2 default (12:12)
            jobs.append((dst, ["phase_b.py"] + v + ["--teacher", a.teacher, "--init", src, "--distill", "0",
                                                    "--steps", str(a.steps_c), "--seed", "1", "--out", dst],
                         os.path.join(RUNS, dst, "core.pt")))
        run_parallel(jobs)
    elif a.stage == "c_eval":
        for n in ("B_v6_s0",):
            for lam in ("0.0", "0.1", "0.3", "0.6"):
                run_parallel([(f"{n}_blend{lam}", ["eval_b.py", "--run", n, "--lam", lam, "--no_control"],
                               os.path.join(RUNS, n, f"eval_b_val_lam{lam}.json"))])
        for n in ("C_v6_s0", "C_v3_s0"):
            run_parallel([(f"{n}_evalb", ["eval_b.py", "--run", n], os.path.join(RUNS, n, "eval_b_val_lam1.0.json"))])
    elif a.stage == "d":
        run_parallel([("D_v6_s0", ["phase_d.py", "--core_run", "C_v6_s0", "--out", "D_v6_s0"],
                       os.path.join(RUNS, "D_v6_s0", "phase_d.json"))])
    elif a.stage == "e":
        run_parallel([("E_v6_s0_hard", ["phase_e.py", "--core_run", "C_v6_s0", "--set", "hardgoal", "--out", "E_v6_s0_hard"],
                       os.path.join(RUNS, "E_v6_s0_hard", "phase_e.json")),
                      ("E_v6_s0_std", ["phase_e.py", "--core_run", "C_v6_s0", "--set", "standard", "--out", "E_v6_s0_std"],
                       os.path.join(RUNS, "E_v6_s0_std", "phase_e.json"))])
    elif a.stage == "slack":
        # recorded order (14:40): alpha=0 ablation, V4, V5, V3 beta_z=1; each followed by its evaluation
        R = lambda *p: os.path.join(RUNS, *p)
        # control for the C_v3 rescue (15:52): same continuation without XM
        run_parallel([("C_v3_noxm_s0", ["phase_b.py", "--variant", "z", "--xm", "0", "--teacher", a.teacher, "--init", "B_v3_s0",
                                        "--distill", "0", "--steps", str(a.steps_c), "--seed", "1", "--out", "C_v3_noxm_s0"],
                       R("C_v3_noxm_s0", "core.pt"))])
        run_parallel([("C_v3_noxm_s0_evalb", ["eval_b.py", "--run", "C_v3_noxm_s0", "--no_control"],
                       R("C_v3_noxm_s0", "eval_b_val_lam1.0.json"))])
        run_parallel([("D_v6_gated_s0", ["phase_d.py", "--core_run", "C_v6_s0", "--gate_kl", "--out", "D_v6_gated_s0"],
                       R("D_v6_gated_s0", "phase_d.json"))])                                     # post-hoc arm 16:02
        run_parallel([("V2_bestof11", ["v2_bestof.py"], R("B_v2_s0", "bestof11_T1.json"))])     # pre-registered 14:50
        for n in ("B_v3_s0", "B_v6_s0"):    # exact SEs for the corrected oracle/mixture (teacher-forced only)
            run_parallel([(f"{n}_nllfix", ["eval_b.py", "--run", n, "--no_control", "--tag", "nllfix"],
                           R(n, "eval_b_val_nllfix.json"))])
        run_parallel([("A_ncp_a0_s0", ["phase_a.py", "--arch", "ncp", "--alpha", "0", "--seed", "0", "--steps", "18000",
                                       "--eval_every", "1000", "--out", "A_ncp_a0_s0"], R("A_ncp_a0_s0", "train_summary.json"))])
        run_parallel([("A_ncp_a0_s0_evala", ["eval_a.py", "--run", "A_ncp_a0_s0"], R("A_ncp_a0_s0", "eval_a_val.json"))])
        for name, v in (("B_v5_s0", ["--variant", "xmonly", "--xm", "1"]),
                        ("B_v4_s0", ["--variant", "gram", "--xm", "0"]),
                        ("B_v3b1_s0", ["--variant", "z", "--xm", "0", "--beta_z", "1.0"])):
            run_parallel([(name, ["phase_b.py"] + v + ["--teacher", a.teacher, "--steps", str(a.steps_b), "--out", name],
                           R(name, "core.pt"))])
            run_parallel([(f"{name}_evalb", ["eval_b.py", "--run", name], R(name, "eval_b_val_lam1.0.json"))])
    else:
        raise SystemExit("unknown stage")


if __name__ == "__main__":
    main()
