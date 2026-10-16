"""Run-folder compatibility shim (the reference files stay byte-identical; see DECISIONS.md D-001).

hrm_encoder.EncoderLayer defines a helper named `_apply(L, G, y, s)`, which shadows torch.nn.Module._apply(fn).
Module.to(device) / .float() / .double() call child._apply(fn), so moving any model that contains an EncoderLayer
raises TypeError. This shim renames the helper to `_write_streams` and rebinds forward to it. The arithmetic is
unchanged token for token (verified bit-exact against the original forward in float64, see runs/tests/)."""
import hrm_encoder as _E

if "_write_streams" not in _E.EncoderLayer.__dict__:
    _E.EncoderLayer._write_streams = _E.EncoderLayer.__dict__["_apply"]

    def _forward(self, L, G, doc, mask, rope):
        x = self.read_attn([L] if self.kind == "S" else [L, G])
        L, G = self._write_streams(L, G, self.attn(x, doc, mask, rope), self.write_attn(x))
        x = self.read_ffn([L] if self.kind == "S" else [L, G])
        L, G = self._write_streams(L, G, self.ffn(x), self.write_ffn(x))
        return L, G

    _orig_forward = _E.EncoderLayer.forward
    _E.EncoderLayer._orig_forward = _orig_forward
    _E.EncoderLayer.forward = _forward
    del _E.EncoderLayer._apply          # nn.Module._apply is visible again
