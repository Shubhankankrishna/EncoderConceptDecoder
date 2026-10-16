"""v3 acceptance harness (CPU, float64, small config) — draft_v3.md r2 §3.6, §4.10, §5.5, §7.1 (C1–C7 repaired).

Every test prints PASS/FAIL; the process exits 1 if any test fails (review C6). Deliberate-defect mutants must make
their test FAIL (a test that cannot fail proves nothing).

  nice -n 19 python test_v3.py          # all tests
"""
import copy, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, torch.nn.functional as F

import model as M
from model import build_model, forward, chunk_masks, rms, PAIRS, SCHED
from hrm_encoder import engram_indices, Encoder

RES = {}


def check(name, ok, info=""):
    RES[name] = bool(ok)
    print(f"{name:58s} {'PASS' if ok else 'FAIL'} {info}", flush=True)


def perturb(m, scale=0.1, skip=("engram.table", "core.table", "core.shared.w2", "engram2.Wv")):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not any(s in n for s in skip):
                p.add_(torch.randn_like(p) * scale)


def batch(m, B=2, T=128, V=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, V, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long); doc[1, 64:] = 1
    ids[1, 60:64] = 0                                                  # one padded chunk at the end of doc 0 of row 1
    ids[0, 126:128] = 0                                                # a partial chunk at the end of row 0
    return ids, doc, engram_indices(ids, doc, m.ecfg)


def main():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)

    # ------------------------------------------------------------------ T0 parity with v3_design/v3_ref.py (same weights)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "v3_design"))
    import v3_ref as R
    torch.manual_seed(0)
    ref = R.HRMThinkerV3(d=64, V=64, enc_ff=128, dec_ff=128, core_ff=96, shared_ff=128, engram_rows=1021, small=True)
    perturb(ref, skip=("engram.table", "core.table", "core.shared.w2"))
    with torch.no_grad():                                              # open the zero-init parts too
        ref.core.table.add_(torch.randn_like(ref.core.table) * 0.1)
        ref.core.shared.w2.weight.add_(torch.randn_like(ref.core.shared.w2.weight) * 0.1)
        ref.enc.engram2.Wv.weight.add_(torch.randn_like(ref.enc.engram2.Wv.weight) * 0.1)
    m, ecfg, _ = build_model("test")
    sd = {k: v for k, v in ref.state_dict().items() if k != "enc.engram2.table.weight"}   # v3_ref aliases the table there
    missing, unexpected = m.load_state_dict(sd, strict=False)
    check("T0a_state_dict_transfers_from_v3_ref", not missing and not unexpected, f"{missing[:3]} {unexpected[:3]}")
    ref.eval(); m.eval()
    ids, doc, ei = batch(m)
    with torch.no_grad():
        for sched in ("3", "2a", "2b", "1"):
            lr_, o_ = ref(ids, doc, ei, sched=sched)
            lm_, a_ = forward(m, ids, doc, ei, R=sched)
            check(f"T0b_bit_parity_with_v3_ref_sched_{sched}", torch.equal(lr_, lm_) and torch.equal(o_["u"], a_["u"]),
                  f"max|d|={(lr_ - lm_).abs().max().item():.1e}")

    # ------------------------------------------------------------------ encoder (§3.6)
    torch.manual_seed(0)
    m, ecfg, _ = build_model("test"); perturb(m)
    m.eval()
    calls = [0]
    h = m.enc.engram.table.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    with torch.no_grad():
        m.enc(ids, doc, ei)
    h.remove()
    check("E6_one_engram_gather_per_forward (C4)", calls[0] == 1, f"calls={calls[0]}")
    base = Encoder(ecfg)                                                # v1 one-read encoder, same weights, 4-head F
    base_sd = {k: v for k, v in m.enc.state_dict().items() if not k.startswith("engram2.")}
    with torch.no_grad():
        m2 = copy.deepcopy(m)
        for li, kind in enumerate(ecfg.layer_pattern):
            if kind == "F":
                base.layers[li].attn = copy.deepcopy(m2.enc.layers[li].attn)
        base.load_state_dict(base_sd)
        t1, c1, _ = base(ids, doc, ei); t2, c2, _ = m.enc(ids, doc, ei)   # W_v2 still 0 (perturb skips it)
    check("E1_two_reads_bit_identical_to_one_read_at_Wv2_0", torch.equal(t1, t2) and torch.equal(c1, c2))
    with torch.no_grad():
        m.enc.engram2.Wv.weight.normal_(0, 0.2)
    T, p = 256, 100
    idsr = torch.randint(2, 64, (1, T)); docr = torch.zeros(1, T, dtype=torch.long)
    idsr2 = idsr.clone(); idsr2[0, p] = (idsr[0, p] + 7) % 64
    with torch.no_grad():
        _, _, a1 = m.enc(idsr, docr, engram_indices(idsr, docr, ecfg)); _, _, a2 = m.enc(idsr2, docr, engram_indices(idsr2, docr, ecfg))
    dL = (a1["L"] - a2["L"]).abs().amax(-1)[0]; dG = (a1["G"] - a2["G"]).abs().amax(-1)[0]
    reach = int(torch.nonzero(dL != 0).max()) - p
    check("E2_L_reach_63_with_read2_and_6head_F", reach == 63 and dG[-1] != 0, f"reach={reach}")
    with torch.no_grad():
        full = m.enc(ids, doc, ei)
        m3 = copy.deepcopy(m)
        m3.enc.engram.Wv.weight.zero_(); m3.enc.engram2.Wv.weight.zero_()
        zero = m3.enc(ids, doc, ei); off = m.enc(ids, doc, ei, engram_off=True)
        m4 = copy.deepcopy(m); m4.enc.engram2.Wv.weight.zero_()
        z2 = m4.enc(ids, doc, ei); off2 = m.enc(ids, doc, ei, engram_read2_off=True)
    check("E3_engram_off_equals_Wv_zero_forward", torch.equal(zero[0], off[0]) and torch.equal(zero[1], off[1]))
    check("E4_engram_read2_off_equals_Wv2_zero_forward", torch.equal(z2[0], off2[0]) and torch.equal(z2[1], off2[1]))
    with torch.no_grad():
        fo = m.enc(ids, doc, ei, F_off=True)
    check("E5_F_off_leaves_L_path_and_changes_concepts", torch.equal(fo[0], full[0]) and not torch.equal(fo[1], full[1]))

    # ------------------------------------------------------------------ core (§4.10)
    torch.manual_seed(0)
    m, ecfg, _ = build_model("test"); perturb(m); m.eval(); core = m.core
    with torch.no_grad():
        tok, C, _ = m.enc(ids, doc, ei); cdoc = doc[:, ::4]
        ref_core = copy.deepcopy(core)
        plain = core.run(C, cdoc, "3", table_off=True, shared_off=True)[0]
        check("C1_init_identity_table0_sharedw2_0", torch.equal(core.trajectory(C, cdoc, "3"), plain))
        core.table.add_(torch.randn_like(core.table) * 0.1)
        S3_t = core.trajectory(C, cdoc, "3")
        check("C1b_table_only_perturbation_changes_output (C6)", not torch.equal(S3_t, plain))
        core2 = copy.deepcopy(ref_core); core2.shared.w2.weight.add_(torch.randn_like(core2.shared.w2.weight) * 0.1)
        check("C1c_shared_only_perturbation_changes_output (C6)", not torch.equal(core2.trajectory(C, cdoc, "3"), plain))
        core.shared.w2.weight.add_(torch.randn_like(core.shared.w2.weight) * 0.1)
        S3 = core.trajectory(C, cdoc, "3"); S1 = core.trajectory(C, cdoc, "1")
        check("C1d_M1_differs_from_M3", not torch.allclose(core.out_norm(S1), core.out_norm(S3)))
        pp = 10; C2 = C.clone(); C2[:, pp] += 1.0
        S3b = core.trajectory(C2, cdoc, "3")
        check("C2_causal_over_concepts", torch.equal(S3[:, :pp], S3b[:, :pp]) and not torch.equal(S3[0, pp:], S3b[0, pp:]))
        C3 = C.clone(); C3[1, 5] += 1.0
        S3c = core.trajectory(C3, cdoc, "3")
        check("C3_document_isolation", torch.equal(S3[1, 16:], S3c[1, 16:]) and torch.equal(S3[0], S3c[0])
              and not torch.equal(S3[1, 5:16], S3c[1, 5:16]))
        perm = torch.tensor([1, 0])
        check("C6_batch_row_permutation_invariant", torch.equal(core.trajectory(C[perm], cdoc[perm], "3"), S3[perm]))
        # batched trajectories == separate trajectories (rows [main; shortcut]); float64 tolerance, not bitwise
        for sc in ("1", "2a", "2b"):
            Sm, _, Ss = core.run(C, cdoc, "3", short=sc)
            d1 = (Sm - S3).abs().max().item(); d2 = (Ss - core.trajectory(C, cdoc, sc)).abs().max().item()
            check(f"C7_batched_shortcut_{sc}_equals_separate", d1 < 1e-12 and d2 < 1e-12, f"{d1:.1e} {d2:.1e}")
        # C-half cache is exactly the v3_ref read
        l0 = core.layers[0]
        xr = R.gr_read_mod(l0.read_attn, [C, S3], [core.table[0, 0, 0], core.table[0, 0, 1]])
        xm = l0.read_mod(l0.read_attn, l0.c_half(l0.read_attn, C), S3, 1 + core.table[0, 0, 0], 1 + core.table[0, 0, 1])
        check("C8_C_half_cache_bit_identical_to_v3_ref_read", torch.equal(xr, xm))
        # table_off / shared_ffn_off switches equal zeroed parameters
        z = copy.deepcopy(core); z.table.zero_(); z.shared.w2.weight.zero_()
        check("C9_table_off_and_shared_off_equal_zeroed_params",
              torch.equal(z.trajectory(C, cdoc, "3"), core.run(C, cdoc, "3", table_off=True, shared_off=True)[0]))

    # ------------------------------------------------------------------ gradients: C2 split into untouched init and perturbed
    torch.manual_seed(0)
    m0, _, _ = build_model("test"); m0.train()
    names_zero_expected = ("core.shared.w1.weight", "core.shared.w3.weight", "enc.engram2.Wk.weight")
    loss0, st0 = ntp_loss(m0, ids, doc, ei, retain=True)
    loss0.backward()
    g = {n: (p.grad.norm().item() if p.grad is not None else 0.0) for n, p in m0.named_parameters()}
    reach_ok = all(s.grad is not None and s.grad.abs().sum() > 0 for s in st0["states"]) and st0["C"].grad.abs().sum() > 0
    check("G1_untouched_init_NTP_reaches_C_S1_S2_S3", reach_ok)
    check("G2_untouched_init_table_sharedw2_Wv2_get_gradient",
          g["core.table"] > 0 and g["core.shared.w2.weight"] > 0 and g["enc.engram2.Wv.weight"] > 0)
    check("G3_untouched_init_expected_zero_branches_are_zero",
          all(g[n] == 0 for n in names_zero_expected) and
          m0.core.table.grad[:, :, 6].abs().sum() == 0, str({n: g[n] for n in names_zero_expected}))
    dead = m0.core.table.grad[[PAIRS[(0, 1)], PAIRS[(0, 2)], PAIRS[(0, 3)]], 0, 1].abs().sum().item()
    check("G4_structurally_dead_gammaS_t0_layer1_attn", dead == 0)
    torch.manual_seed(0)
    mp, _, _ = build_model("test"); perturb(mp, skip=("engram.table",)); mp.train()
    lp, _ = ntp_loss(mp, ids, doc, ei)
    sc_s = shortcut_losses(mp, ids, doc, ei, "2b")
    (lp + sc_s).backward()
    nog = [n for n, p in mp.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    nog = [n for n in nog if "engram.table" not in n]
    check("G5_perturbed_every_parameter_gets_gradient", not nog, str(nog[:6]))

    # ------------------------------------------------------------------ C1: teacher detachment with a mutant
    def scons_grads(detach=True):
        torch.manual_seed(0)
        mm, _, _ = build_model("test"); perturb(mm, skip=("engram.table",)); mm.train()
        tok, C, _ = mm.enc(ids, doc, ei); cdoc = doc[:, ::4]
        S3, states, Ss = mm.core.run(C, cdoc, "3", short="1")
        S3.retain_grad(); Ss.retain_grad()
        teacher = rms(S3).detach() if detach else rms(S3)
        ((rms(Ss) - teacher) ** 2).mean().backward()
        return S3.grad, Ss.grad
    tg, sg = scons_grads(True)
    check("D1_scons_teacher_gets_zero_grad_student_nonzero (C1)",
          (tg is None or tg.abs().sum() == 0) and sg is not None and sg.abs().sum() > 0)
    tg_m, _ = scons_grads(False)
    check("D1m_mutant_without_detach_is_detected", tg_m is not None and tg_m.abs().sum() > 0,
          f"teacher grad {0 if tg_m is None else tg_m.norm().item():.2e}")
    torch.manual_seed(0)
    mm, _, _ = build_model("test"); perturb(mm, skip=("engram.table",)); mm.train()
    tok, C, _ = mm.enc(ids, doc, ei)
    S3, _, Ss = mm.core.run(C, doc[:, ::4], "3", short="1")
    ((rms(Ss) - rms(S3).detach()) ** 2).mean().backward()
    check("D2_scons_does_not_touch_out_norm_gain", mm.core.out_norm.gamma.grad is None or
          mm.core.out_norm.gamma.grad.abs().sum() == 0)

    # ------------------------------------------------------------------ C3: reuse parity on gradients + one AdamW step
    def step_with(reuse):
        torch.manual_seed(0)
        mm, _, _ = build_model("test"); perturb(mm, skip=("engram.table",)); mm.train()
        opt = torch.optim.AdamW(M.param_groups(mm)[0], lr=1e-3)
        tok, C, _ = mm.enc(ids, doc, ei); cdoc = doc[:, ::4]
        if reuse:
            S3, _, Ss = mm.core.run(C, cdoc, "3", short="2a")
        else:
            S3 = mm.core.trajectory(C, cdoc, "3"); Ss = mm.core.trajectory(C, cdoc, "2a")
        u, us = mm.core.out_norm(S3), mm.core.out_norm(Ss)
        loss = (u ** 2).mean() + mm.core.head(us, pred=False)[0].logsumexp(-1).mean() + ((rms(Ss) - rms(S3).detach()) ** 2).mean()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}
        opt.step()
        return loss.item(), grads, {n: p.detach().clone() for n, p in mm.named_parameters()}
    la, ga, pa = step_with(True); lb, gb, pb = step_with(False)
    dg = max((ga[n] - gb[n]).abs().max().item() for n in ga)
    dp = max((pa[n] - pb[n]).abs().max().item() for n in pa)
    check("R1_reuse_parity_loss_grads_and_one_AdamW_step (C3)", abs(la - lb) < 1e-12 and dg < 1e-12 and dp < 1e-12 and ga.keys() == gb.keys(),
          f"dl={abs(la-lb):.1e} dg={dg:.1e} dp={dp:.1e}")

    # ------------------------------------------------------------------ decoder (§5.5) on the full model
    torch.manual_seed(0)
    m, _, _ = build_model("test"); perturb(m, skip=("engram.table",)); m.eval()
    with torch.no_grad():
        l1, _ = forward(m, ids, doc, ei)
        ids2 = ids.clone(); ids2[0, 90] = (ids[0, 90] + 3) % 64
        l2, _ = forward(m, ids2, doc, engram_indices(ids2, doc, m.ecfg))
        check("P1_future_token_edit_leaves_earlier_logits", torch.equal(l1[0, :90], l2[0, :90]) and not torch.equal(l1[0, 90:], l2[0, 90:]))
        ids3 = ids.clone(); ids3[1, 20] = (ids[1, 20] + 3) % 64
        l3, _ = forward(m, ids3, doc, engram_indices(ids3, doc, m.ecfg))
        check("P2_other_document_edit_leaves_this_document", torch.equal(l1[1, 64:], l3[1, 64:]) and torch.equal(l1[0], l3[0]))
        _, a1 = forward(m, ids, doc, ei)
        P = a1["P"]; null = m.dec.plan.null + m.dec.plan.offset[(torch.arange(128) + 1) % 4]
        check("P3_first_three_positions_get_null_plan", torch.equal(P[1, 64:67], null[64:67]) and torch.equal(P[0, :3], null[:3])
              and not torch.equal(P[0, 3:7], null[3:7]))
        lu, _ = forward(m, ids, doc, ei, use_u=False); lc_, _ = forward(m, ids, doc, ei, use_c=False)
        check("P4_use_u_use_c_switches_change_logits", not torch.equal(lu, l1) and not torch.equal(lc_, l1))
        # decoder reach 70 without the core (core_off): edit 71 tokens back changes nothing, 70 changes
        T2 = 256; idr = torch.randint(2, 64, (1, T2)); dr = torch.zeros(1, T2, dtype=torch.long)
        a0, _ = forward(m, idr, dr, engram_indices(idr, dr, m.ecfg), core_off=True)
        idx = idr.clone(); idx[0, 100] = (idr[0, 100] + 5) % 64
        a1, _ = forward(m, idx, dr, engram_indices(idx, dr, m.ecfg), core_off=True)
        ch = (a0 - a1).abs().amax(-1)[0]
        dreach = int(torch.nonzero(ch != 0).max()) - 100
        check("P5_decoder_token_path_reach_70", dreach == 70, f"reach={dreach}")

    # ------------------------------------------------------------------ target masks and full-batch reductions
    nonempty, complete, pair_ok = chunk_masks(ids, doc)
    check("M1_masks_padded_and_partial_chunks", (not complete[1, 15]) and (not nonempty[1, 15]) and (not complete[0, 31])
          and nonempty[0, 31] and (not pair_ok[1, 14]) and (not pair_ok[1, 15]) and (not pair_ok[0, 30]))
    torch.manual_seed(0)
    m, _, _ = build_model("test"); perturb(m, skip=("engram.table",)); m.eval()
    with torch.no_grad():
        tok, C, _ = m.enc(ids, doc, ei)
        vq, n = m.core.vq_sum(C, complete, count_hits=False)
        u = m.core.out_norm(m.core.trajectory(C, doc[:, ::4], "3")); lc, _ = m.core.head(u)
        allpad = torch.zeros_like(pair_ok)
        z = m.core.ncp_sum(lc, n, allpad)
        check("M2_empty_pair_mask_gives_finite_zero", torch.isfinite(z) and z.item() == 0.0)
    # full-batch reduction: two micro-batches with unequal valid counts == one batch of both
    torch.manual_seed(0)
    m, _, _ = build_model("test"); perturb(m, skip=("engram.table",)); m.eval()
    idsA, docA, eiA = batch(m, seed=3); idsB, docB, eiB = batch(m, seed=4)
    idsB[:, 40:] = 0                                                   # far fewer valid targets in micro-batch B
    eiB = engram_indices(idsB, docB, m.ecfg)
    from data import target_mask
    def ntp_sum(i, d_, e_):
        lg, _ = forward(m, i, d_, e_)
        tgt = i[:, 1:].masked_fill(~target_mask(i, d_), -100)
        return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum"), \
            target_mask(i, d_).sum()
    with torch.no_grad():
        sA, nA = ntp_sum(idsA, docA, eiA); sB, nB = ntp_sum(idsB, docB, eiB)
        both = torch.cat([idsA, idsB]); bd = torch.cat([docA, docB])
        s2, n2 = ntp_sum(both, bd, engram_indices(both, bd, m.ecfg))
    full_batch = (sA + sB) / (nA + nB); per_micro = (sA / nA + sB / nB) / 2
    check("M3_full_batch_denominator_equals_one_big_batch", abs((full_batch - s2 / n2).item()) < 1e-10 and
          abs((full_batch - per_micro).item()) > 1e-4, f"nA={int(nA)} nB={int(nB)}")

    # ------------------------------------------------------------------ optimizer groups
    m, _, _ = build_model("test")
    groups, nd = M.param_groups(m)
    check("O1_table_in_no_decay_group", "core.table" in nd and "enc.engram.table.weight" in nd and "core.vocab.e" in nd)
    ids_all = [id(p) for g_ in groups for p in g_["params"]]
    check("O2_every_parameter_registered_once", len(ids_all) == len(set(ids_all)) == len(list(m.parameters())))
    t = M.param_table(build_model(engram_rows=11)[0])
    check("A1_compute_params_18986871_and_ledger_reconciles (C5)", t["compute"] == 18986871 and t["parts"]["eng2"] == 214018)

    # ------------------------------------------------------------------ mutants: each must be caught
    mut = {}
    # (a) leaked future token: decoder plan index shifted by one chunk -> P1 must fail
    import hrm_decoder as HD
    orig = HD.plan_index
    HD.plan_index = lambda doc, k=4: (lambda j, ok, r: ((j + 1).clamp(max=doc.shape[1] // k - 1), ok, r))(*orig(doc, k))
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            mm, _, _ = build_model("test"); perturb(mm, skip=("engram.table",)); mm.eval()
            a, _ = forward(mm, ids, doc, ei); b, _ = forward(mm, ids2, doc, engram_indices(ids2, doc, mm.ecfg))
        mut["future_leak"] = not torch.equal(a[0, :90], b[0, :90])
    finally:
        HD.plan_index = orig
    # (b) crossed document boundary: concept mask ignores documents -> C3 must fail
    orig_bm = M.build_masks
    def leaky(doc, window):
        s, f = orig_bm(doc, window)
        Tn = doc.shape[1]; i = torch.arange(Tn)
        return s, (i[None, :] <= i[:, None])[None, None].expand_as(f)
    M.build_masks = leaky
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            mm, _, _ = build_model("test"); perturb(mm); mm.eval()
            tk, Cc, _ = mm.enc(ids, doc, ei); cd = doc[:, ::4]
            Ca = Cc.clone(); Ca[1, 5] += 1.0
            mut["doc_cross"] = not torch.equal(mm.core.trajectory(Cc, cd)[1, 16:], mm.core.trajectory(Ca, cd)[1, 16:])
    finally:
        M.build_masks = orig_bm
    mut["teacher_detach_removed"] = RES["D1m_mutant_without_detach_is_detected"]
    # (c) detached reused prefix: shortcut from a detached S1 -> gradient parity must fail
    orig_run = M.ConceptCoreV3.run
    def det_run(self, C, cdoc, sched="3", short=None, **kw):
        if short != "2a":
            return orig_run(self, C, cdoc, sched, short, **kw)
        S, states, _ = orig_run(self, C, cdoc, sched, None, **kw)
        S_s = self.run(C, cdoc, [(1, 2)], **kw)[0] if False else None
        onep = 1 + self.table; cache = self.c_cache(C); _, mask = M.build_masks(cdoc, 1)
        S_s = self.one_pass(cache, states[0].detach(), cdoc, mask, onep, PAIRS[(1, 2)], C.shape[0], self.shared)
        return S, states, S_s
    M.ConceptCoreV3.run = det_run
    try:
        lc2, gc2, _ = step_with(True)
    finally:
        M.ConceptCoreV3.run = orig_run
    mut["detached_prefix"] = max((gc2[n] - gb[n]).abs().max().item() for n in gb if n in gc2) > 1e-9
    # (d) duplicated optimizer registration / (e) table left in the decay group
    m, _, _ = build_model("test")
    g_, nd_ = M.param_groups(m)
    dup = [dict(params=g_[0]["params"] + [g_[1]["params"][0]], weight_decay=0.1), g_[1]]
    ids_d = [id(p) for x in dup for p in x["params"]]
    mut["duplicate_registration"] = len(ids_d) != len(set(ids_d))
    src = open(M.__file__).read()                                      # mutant param_groups without the table rule
    ns = {}
    exec(compile(src.replace(' or n == "core.table":', ':'), "model_mut", "exec"), ns)
    _, nd_mut = ns["param_groups"](m)
    mut["table_decayed"] = not ("core.table" in nd_mut and "enc.engram.table.weight" in nd_mut and "core.vocab.e" in nd_mut)
    for k, v in mut.items():
        check(f"X_mutant_{k}_is_caught", v)

    torch.set_default_dtype(torch.float32)
    bad = [k for k, v in RES.items() if not v]
    print(f"\n{len(RES) - len(bad)}/{len(RES)} PASS" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


def ntp_loss(m, ids, doc, ei, retain=False):
    from data import target_mask
    tok, C, _ = m.enc(ids, doc, ei)
    if retain:
        C.retain_grad()
    cdoc = doc[:, ::4]
    S, states, _ = m.core.run(C, cdoc, "3")
    if retain:
        for s in states:
            s.retain_grad()
    u = m.core.out_norm(S); lc, cp = m.core.head(u)
    logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight)
    tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100), \
        dict(C=C, states=states)


def shortcut_losses(m, ids, doc, ei, sc):
    tok, C, _ = m.enc(ids, doc, ei); cdoc = doc[:, ::4]
    nonempty, complete, pair_ok = chunk_masks(ids, doc)
    vq, n = m.core.vq_sum(C, complete, count_hits=False)
    S3, _, Ss = m.core.run(C, cdoc, "3", short=sc)
    us = m.core.out_norm(Ss); lcs, _ = m.core.head(us, pred=False)
    return 0.05 * m.core.ncp_sum(lcs, n, pair_ok) / pair_ok.sum() + 0.1 * (((rms(Ss) - rms(S3).detach()) ** 2).mean(-1) * nonempty).sum() / nonempty.sum() + vq / complete.sum()


if __name__ == "__main__":
    main()
