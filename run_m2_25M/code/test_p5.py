"""P5 (d = 384 config): rerun pipeline tests P1-P3 with the PROMPT 2.1 configuration at small V and T, float64,
randomised weights (so zero-initialised paths cannot hide a leak), exact equality.

Also usable on trained weights (section 7.1): `python test_p5.py --ckpt path.pt` loads a checkpoint (V = 8192,
T = 1024 row from the test split is replaced by random ids) and reruns P1-P3 without randomising.
"""
import argparse, sys
import torch
from model import build_model, forward, engram_indices


def run(m, ecfg, V, T, p=301, label="", reach=70):
    diff = lambda a, b: torch.nonzero((a - b).abs().amax(-1)[0] != 0).flatten()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, V, (1, T), generator=g)
    doc = torch.zeros(1, T, dtype=torch.long)
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7 - 2) % (V - 2) + 2
    f = lambda i, d, **kw: forward(m, i, d, engram_indices(i, d, ecfg), 4, **kw)
    with torch.no_grad():
        (a, xa), (b, xb) = f(ids, doc), f(ids2, doc)
        (a0, _), (b0, _) = f(ids, doc, core_off=True), f(ids2, doc, core_off=True)
    ch, ch0 = diff(a, b), diff(a0, b0)
    # P1 causality
    assert ch.min().item() == p, ("P1", ch.min().item())
    print(f"PASS P5/P1 {label}: causality - editing token {p} changes nothing before it (first change at t={ch.min().item()})")
    # P2 reach
    assert ch0.max().item() - p == reach and torch.equal(a0[0, p + reach + 1:], b0[0, p + reach + 1:]), ("P2", ch0.max().item() - p)
    assert ch.max().item() == T - 1
    first_plan = diff(xa["P"], xb["P"]).min().item()
    assert first_plan == 4 * (p // 4) + 3, first_plan
    print(f"PASS P5/P2 {label}: core OFF reach exactly {ch0.max().item() - p} (bit-identical from t={p + reach + 1}); core ON the edit "
          f"reaches the row end (t={ch.max().item()}); plan first changes at t={first_plan} = 4*floor(p/4)+3")
    # P3 document isolation (boundary at a multiple of 4)
    doc2 = torch.zeros(1, T, dtype=torch.long); doc2[0, T // 2:] = 1
    ids3 = ids.clone(); ids3[0, T // 2 - 3] = (ids[0, T // 2 - 3] + 5 - 2) % (V - 2) + 2
    with torch.no_grad():
        (c1, y1), (c2, y2) = f(ids, doc2), f(ids3, doc2)
    assert torch.equal(c1[0, T // 2:], c2[0, T // 2:]) and torch.equal(y1["u"][0, T // 8:], y2["u"][0, T // 8:])
    assert torch.equal(y1["c_hat"][0, T // 8:], y2["c_hat"][0, T // 8:])
    print(f"PASS P5/P3 {label}: two packed documents - editing document 0 (t={T // 2 - 3}) leaves document 1's logits, "
          f"concepts and core states bit-identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--V", type=int, default=96)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    if a.ckpt is None:
        m, ecfg, dcfg = build_model(V=a.V)
        with torch.no_grad():
            for n, q in m.named_parameters():
                q.add_(torch.randn_like(q) * 0.2)
        m.eval()
        print(f"P5 config: d={ecfg.d_model}, heads {ecfg.n_heads}/{ecfg.n_kv_heads}x{ecfg.head_dim}, d_ff={ecfg.d_ff}, "
              f"pattern {ecfg.layer_pattern}, Engram {ecfg.engram_rows}x{ecfg.engram_dim}, core S=48 N=128, "
              f"decoder {dcfg.layout} L={dcfg.n_layers} w={dcfg.window}; V={a.V}, T={a.T}, float64, randomised weights")
        run(m, ecfg, a.V, a.T, label="(random weights)")
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m, ecfg, dcfg = build_model(layout=ck.get("args", {}).get("layout", "confined"))
        sd = ck.get("model", ck)
        m.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in sd.items()})
        m.double().eval()
        if dcfg.layout == "stack":           # the stack control's reach is 63 + 4 x 10 = 103, not 70
            import types
            print(f"(stack decoder: expected core-off reach {dcfg.reach()})")
        run(m, ecfg, 8192, 1024, p=601, label=f"(trained weights {a.ckpt})", reach=dcfg.reach())
