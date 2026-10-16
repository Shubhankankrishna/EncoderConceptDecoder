"""Throughput shims with identical arithmetic (V-006).

The encoder's ShortConv recomputes `same_doc_shift_mask(doc, i)` (pad + compare) for every conv and every shift:
3 shifts x 3 convs x 8 layers + Engram = 75 boolean masks per forward, all from the same `doc`. Inside one
Encoder.forward the masks depend only on (doc, i), so they are memoised for the duration of that call. The masks carry
no gradient, so the backward pass is unaffected. The memo is THREAD-LOCAL: the data-prefetch thread also calls
same_doc_shift_mask (engram_indices) concurrently and always takes the original path.
"""
import threading
import hrm_encoder as _E

_orig_mask = _E.same_doc_shift_mask
_tls = threading.local()


def _memo_mask(doc, i):
    memo = getattr(_tls, "memo", None)
    if memo is None:
        return _orig_mask(doc, i)
    key = (doc.data_ptr(), str(doc.device), tuple(doc.shape), i)
    m = memo.get(key)
    if m is None:
        m = memo[key] = _orig_mask(doc, i)
    return m


if not getattr(_E, "_speed_patched", False):
    _E.same_doc_shift_mask = _memo_mask
    _orig_forward = _E.Encoder.forward

    def _forward(self, ids, doc, eng_idx):
        _tls.memo = {}
        try:
            return _orig_forward(self, ids, doc, eng_idx)
        finally:
            _tls.memo = None

    _E.Encoder.forward = _forward
    _E._speed_patched = True
