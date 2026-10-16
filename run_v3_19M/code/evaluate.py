"""v3 evaluation (ported from v2's evaluate.py; language / items / truncation code unchanged so numbers stay comparable).
v3 changes: R is a schedule name ("1" | "2a" | "2b" | "3"); codebook health uses the r2 target masks (ncp_ce = masked
pairs: nonempty current, complete target, same doc) and also reports ncp at M = 1; loop_dynamics measures the v3 main
trajectory (norms, updates, angles, rho, relative updates, gradient of NTP at each pass state).

Evaluation benchmark (PROMPT section 5): language NTP (nats, bits/byte), item exact match (teacher-forced greedy),
interventions (core off, plan swap, u off, c off, loop depth R, context truncation), codebook health, telemetry.

Every function takes the model in eval mode (so ConceptVocab.vq_loss does not touch the restart counters).
"""
import math, os, subprocess
import numpy as np
import torch
import torch.nn.functional as F

from model import forward, chunk_masks, R_MAP
from data import Canon, make_batch, target_mask, pack_items, PAD

LN2 = math.log(2)


class Evaluator:
    def __init__(self, data_dir, ecfg, device, batch_rows=4):
        self.dir, self.ecfg, self.dev, self.B = data_dir, ecfg, device, batch_rows
        self.canon = Canon(data_dir)
        self.tbytes = torch.from_numpy(np.load(os.path.join(data_dir, "token_bytes.npy")))
        self.tbytes[1] = 1                          # EOS is a real target: counted as one byte (newline convention)
        self._items = {}

    # ------------------------------------------------------------------ helpers
    def _to(self, *xs):
        return [x.to(self.dev) for x in xs]

    def _batches(self, tok, doc):
        for s in range(0, len(tok), self.B):
            t, d = np.asarray(tok[s:s + self.B]), np.asarray(doc[s:s + self.B])
            n_real = len(t)
            if n_real < 2:                           # plan swap needs >= 2 rows: pad with a DIFFERENT row (not scored)
                j = 0 if s > 0 else min(1, len(tok) - 1)
                t, d = np.concatenate([t, np.asarray(tok[j:j + 1])]), np.concatenate([d, np.asarray(doc[j:j + 1])])
            yield make_batch(t, d, self.canon, self.ecfg), n_real

    def items(self, split):
        if split not in self._items:
            self._items[split] = torch.load(os.path.join(self.dir, "eval", f"{split}.pt"), weights_only=False)
        return self._items[split]

    def lang_rows(self, split, name, n=None):
        d = os.path.join(self.dir, "lang_eval")
        tok = np.load(os.path.join(d, f"{split}_{name}_tok.npy"))
        doc = np.load(os.path.join(d, f"{split}_{name}_doc.npy"))
        ctx = np.load(os.path.join(d, f"{split}_{name}_ctx16.npy"))
        return (tok[:n], doc[:n], ctx[:n]) if n else (tok, doc, ctx)

    # ------------------------------------------------------------------ language
    @torch.no_grad()
    def language(self, m, tok, doc, R="3", last=None, **kw):
        """Mean next-token loss (nats) over valid targets, bits per byte; `last` = score only the last k positions
        of each row (per-row losses returned too)."""
        tot, n, nbytes, per_row = 0.0, 0, 0, []
        for (ids, dc, ei), n_real in self._batches(tok, doc):
            ids_d, dc_d, ei_d = self._to(ids, dc, ei)
            logits, _ = forward(m, ids_d, dc_d, ei_d, R, **kw)
            nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids_d[:, 1:].reshape(-1),
                                  reduction="none").view(ids.shape[0], -1).cpu()
            mask = target_mask(ids, dc)
            if last is not None:
                keep = torch.zeros_like(mask); keep[:, -last:] = True; mask &= keep
            nll, mask = nll[:n_real], mask[:n_real]
            b = self.tbytes[ids[:n_real, 1:]]
            tot += (nll * mask).sum().item(); n += mask.sum().item(); nbytes += (b * mask).sum().item()
            per_row += ((nll * mask).sum(1) / mask.sum(1).clamp_min(1)).tolist()
        return dict(nats=tot / max(n, 1), bpb=tot / LN2 / max(nbytes, 1), tokens=int(n), per_row=per_row)

    @torch.no_grad()
    def truncation(self, m, tok, doc, ctx, rs=(72, 128, 256, 512, 1024), R="3", last=16):
        """Score only the last 16 positions; feed only the last r tokens (r multiple of 4, the cut row starts a new
        document at its first token). Also the plan-swap cost on the same positions at full context."""
        out = {}
        T = tok.shape[1]
        for r in rs:
            t, d = tok[:, T - r:], doc[:, T - r:]
            res = self.language(m, t, d, R=R, last=last)
            out[f"r{r}"] = res
        out["swap_r1024"] = self.language(m, tok, doc, R=R, last=last, plan_swap=True)
        per = {k: np.array(v["per_row"]) for k, v in out.items()}
        sel = ctx >= 512
        out["summary"] = dict(
            gain_beyond_70=out["r72"]["nats"] - out["r1024"]["nats"],
            swap_cost=out["swap_r1024"]["nats"] - out["r1024"]["nats"],
            rows=int(len(ctx)), rows_ctx_ge_512=int(sel.sum()),
            gain_beyond_70_ctx512=float((per["r72"][sel] - per["r1024"][sel]).mean()) if sel.any() else None,
            swap_cost_ctx512=float((per["swap_r1024"][sel] - per["r1024"][sel]).mean()) if sel.any() else None,
            curve={k: v["nats"] for k, v in out.items() if k.startswith("r")},
            curve_ctx512={k: float(per[k][sel].mean()) for k in out if k.startswith("r")} if sel.any() else None)
        for k in list(out):
            if k != "summary":
                out[k].pop("per_row", None)
        return out

    # ------------------------------------------------------------------ items (teacher-forced greedy exact match)
    @torch.no_grad()
    def item_accuracy(self, m, items, R="3", **kw):
        tok, doc, where = pack_items(items)
        preds = np.zeros(tok.shape, dtype=np.int64)
        r0 = 0
        for (ids, dc, ei), n_real in self._batches(tok, doc):
            ids_d, dc_d, ei_d = self._to(ids, dc, ei)
            logits, _ = forward(m, ids_d, dc_d, ei_d, R, **kw)
            preds[r0:r0 + n_real, 1:] = logits[:n_real, :-1].argmax(-1).cpu().numpy()
            r0 += n_real
        fam = {}
        for it, (row, off) in zip(items, where):
            f = fam.setdefault(it["family"], dict(n=0, exact=0, off=None))
            ok_all = True
            for s, e, _ in it["spans"]:
                hit = preds[row, off + s:off + e] == tok[row, off + s:off + e]
                ok_all &= bool(hit.all())
                if it["family"] in ("lr_carry3", "lr_multi4"):
                    f["off"] = hit.astype(float) if f["off"] is None else f["off"] + hit
                    f["n_off"] = f.get("n_off", 0) + 1
            f["n"] += 1; f["exact"] += int(ok_all)
            if it["family"] == "lr_hop":
                dkey = f"exact_h{it['meta']['depth']}"
                f.setdefault(dkey, [0, 0]); f[dkey][0] += int(ok_all); f[dkey][1] += 1
        out = {}
        for k, f in fam.items():
            out[k] = dict(n=f["n"], exact=f["exact"] / f["n"])
            if f["off"] is not None:
                out[k]["by_offset"] = (f["off"] / f["n_off"]).round(4).tolist()
            for kk, v in f.items():
                if kk.startswith("exact_h"):
                    out[k][kk] = v[0] / v[1]
        return out

    def select(self, split, families=None, n=None):
        it = self.items(split)
        if families is not None:
            it = [x for x in it if x["family"] in families]
        if n is not None:
            by, sel = {}, []
            for x in it:
                if by.get(x["family"], 0) < n:
                    sel.append(x); by[x["family"]] = by.get(x["family"], 0) + 1
            it = sel
        return it

    # ------------------------------------------------------------------ codebook health and telemetry
    @torch.no_grad()
    def codebook_health(self, m, tok, doc, R="3"):
        """Code usage / perplexity / VE over NONEMPTY chunks (v1's convention for these ratios); ncp_ce over the r2-masked
        pairs at M = R and ncp_ce_M1 at M = 1 (the endpoint the shortcut terms train)."""
        V = m.core.vocab
        counts = torch.zeros(V.S, V.N)
        se = 0.0
        ce = {R: [0.0, 0], "1": [0.0, 0]}
        cs_all = []
        for (ids, dc, ei), n_real in self._batches(tok, doc):
            ids_d, dc_d, ei_d = self._to(ids, dc, ei)
            tokv, c_hat, _ = m.enc(ids_d, dc_d, ei_d)
            cdoc = dc_d[:, ::4]
            nonempty, complete, pair_ok = chunk_masks(ids_d, dc_d)
            n = V.assign(c_hat)
            for Rk in ce:
                u, lc, _ = m.core(c_hat, cdoc, Rk)
                ce[Rk][0] += m.core.ncp_sum(lc[:n_real], n[:n_real], pair_ok[:n_real]).item()
                ce[Rk][1] += int(pair_ok[:n_real].sum())
            real = nonempty[:n_real]
            c = c_hat[:n_real][real]
            nn_ = n[:n_real][real]
            q = V.e[torch.arange(V.S, device=c.device), nn_].flatten(-2)
            counts += F.one_hot(nn_.cpu(), V.N).sum(0).float()
            se += (c - q).pow(2).sum().item()
            cs_all.append(c.cpu())
        C = torch.cat(cs_all)
        var = (C - C.mean(0)).pow(2).sum().item()
        p = counts / counts.sum(1, keepdim=True)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1)
        return dict(codes_used_frac_min=float((counts > 0).float().mean(1).min()),
                    codes_used_frac_mean=float((counts > 0).float().mean()),
                    perplexity_mean=float(ent.exp().mean()), var_explained=1 - se / var,
                    ncp_ce=ce[R][0] / max(ce[R][1], 1), ncp_ce_M1=ce["1"][0] / max(ce["1"][1], 1),
                    code_entropy=float(ent.mean()), n_concepts=int(len(C)), ncp_pairs=ce[R][1])

    @torch.no_grad()
    def telemetry(self, m, tok, doc):
        (ids, dc, ei), _ = next(iter(self._batches(tok[:self.B], doc[:self.B])))
        ids_d, dc_d, ei_d = self._to(ids, dc, ei)
        _, _, aux = m.enc(ids_d, dc_d, ei_d)
        mix = aux["mix"].float().mean((0, 1)).tolist()
        taus = {n: [round(x, 4) for x in p.exp().tolist()] for n, p in m.named_parameters() if n.endswith("log_tau")}
        out = dict(crc_mix_e0_L_G=[round(x, 4) for x in mix], engram_Wv_norm=m.enc.engram.Wv.weight.norm().item(),
                   engram2_Wv_norm=m.enc.engram2.Wv.weight.norm().item(),
                   shared_w2_norm=m.core.shared.w2.weight.norm().item() if m.core.shared is not None else None,
                   table_norm=m.core.table.norm().item() if m.core.table is not None else None,
                   engram_alpha_mean=aux["engram_alpha"].float().mean().item(), tau=taus)
        if str(self.dev).startswith("mps"):
            out.update(mps_alloc_gib=torch.mps.current_allocated_memory() / 2**30,
                       mps_driver_gib=torch.mps.driver_allocated_memory() / 2**30)
        out["swap"] = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
        return out

    @torch.no_grad()
    def loop_dynamics(self, m, items, R="3"):
        """v3 main trajectory per pass on the given items (draft §4.7): |S|, |dS|, angle between consecutive updates,
        rho = |S_M| / sum |dS_i|, relative updates |dS_i| / |dS_1|, turn of S, and the NTP gradient at each pass state.
        Diagnostics only (neither necessary nor sufficient for useful work)."""
        tok, doc, where = pack_items(items[:32])
        (ids, dc, ei), _ = next(iter(self._batches(tok[:2], doc[:2])))
        ids_d, dc_d, ei_d = self._to(ids, dc, ei)
        core = m.core
        with torch.enable_grad():
            tokv, c_hat, _ = m.enc(ids_d, dc_d, ei_d)
            cdoc = dc_d[:, ::4]
            S, states, _ = core.run(c_hat.detach(), cdoc, R_MAP.get(R, R))
            for s_ in states:
                s_.retain_grad()
            u = core.out_norm(S); lc, cp = core.head(u)
            logits, _ = m.dec(tokv.detach(), u, cp, dc_d, m.enc.embed.weight)
            msk = target_mask(ids_d, dc_d)
            F.cross_entropy(logits[:, :-1][msk], ids_d[:, 1:][msk]).backward()
        P = len(states)
        norms = [s.norm(dim=-1).mean().item() for s in states]
        d = [states[0]] + [states[i] - states[i - 1] for i in range(1, P)]
        steps = [x.norm(dim=-1).mean().item() for x in d]
        ang = [torch.rad2deg(torch.acos(F.cosine_similarity(d[i], d[i + 1], -1).clamp(-1, 1))).mean().item() for i in range(P - 1)]
        turn = [torch.rad2deg(torch.acos(F.cosine_similarity(states[i], states[i + 1], -1).clamp(-1, 1))).mean().item() for i in range(P - 1)]
        rho = (states[-1].norm(dim=-1) / sum(x.norm(dim=-1) for x in d).clamp_min(1e-9)).mean().item()
        grads = [s.grad.norm(dim=-1).mean().item() if s.grad is not None else 0.0 for s in states]
        m.zero_grad(set_to_none=True)
        return dict(sched=R, state_norm=norms, step=steps, consecutive_step_angle_deg=ang, turn_deg=turn, rho=rho,
                    rel_step=[x / max(steps[0], 1e-9) for x in steps[1:]], grad=grads,
                    grad_ratio_S1_over_SM=grads[0] / max(grads[-1], 1e-12))

    # ------------------------------------------------------------------ bundles
    def quick(self, m, R="3", n_lang=128, n_probe=128):
        """Hourly eval (PROMPT 6.3): validation language, 5 probes x 128, plan swap on lr_recall1, codebook health."""
        from probes import PROBES
        out = {}
        for name in ("tinystories", "wikitext"):
            tok, doc, _ = self.lang_rows("val", name, n_lang)
            r = self.language(m, tok, doc, R=R); r.pop("per_row")
            out[f"val_{name}"] = r
        out["val_lang_mean"] = (out["val_tinystories"]["nats"] + out["val_wikitext"]["nats"]) / 2
        probes = self.select("val", PROBES, n_probe)
        out["probes"] = self.item_accuracy(m, probes, R=R)
        rec = [x for x in probes if x["family"] == "lr_recall1"]
        out["recall1_plan_swap"] = self.item_accuracy(m, rec, R=R, plan_swap=True)["lr_recall1"]
        mt, md = self._maint()
        out["codebook"] = self.codebook_health(m, mt, md, R=R)
        out["ncp_M1"] = out["codebook"]["ncp_ce_M1"]
        out["telemetry"] = self.telemetry(m, mt, md)
        lp = self.loop_dynamics(m, probes, R=R)
        out["loop"] = dict(angle23=lp["consecutive_step_angle_deg"][-1] if lp["consecutive_step_angle_deg"] else None,
                           rho3=lp["rho"], rel_step=lp["rel_step"], state_norm=lp["state_norm"], grad_ratio=lp["grad_ratio_S1_over_SM"])
        return out

    def _maint(self, n=16):
        d = os.path.join(self.dir, "maintenance")
        tok = np.load(os.path.join(d, "tok.npy"), mmap_mode="r")
        doc = np.load(os.path.join(d, "doc.npy"), mmap_mode="r")
        idx = np.random.RandomState(123).choice(len(tok), n, replace=False)
        idx.sort()
        return np.asarray(tok[idx]), np.asarray(doc[idx])
