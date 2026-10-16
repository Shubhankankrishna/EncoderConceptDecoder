"""Verify the compat shim: (1) forward bit-identical to the original in float64; (2) run hrm_encoder.mps_smoke_test."""
import torch, hrm_encoder as E
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
cfg = E.EncoderConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, rope_dims=8, engram_rows=1021, engram_dim=8)
enc = E.Encoder(cfg); E._randomize(enc); enc.eval()
ids = torch.randint(0, 64, (2, 256)); doc = torch.zeros(2, 256, dtype=torch.long); doc[1, 128:] = 1
ei = E.engram_indices(ids, doc, cfg)
with torch.no_grad(): a = enc(ids, doc, ei)
import compat
with torch.no_grad(): b = enc(ids, doc, ei)
assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2]["G"], b[2]["G"])
print("PASS compat shim: encoder forward bit-identical to the original (float64, tok, c_hat, G)")
torch.set_default_dtype(torch.float32)
E.mps_smoke_test()
