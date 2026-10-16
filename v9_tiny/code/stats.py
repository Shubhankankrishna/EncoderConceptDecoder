"""V9 statistics (draft §5.3, §13.2): McNemar exact, paired item bootstrap, paired document bootstrap, Holm."""
import numpy as np
from scipy.stats import binom

RESAMPLES = 10000


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes a, b."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError('paired length mismatch')
    gains = int((a & ~b).sum()); losses = int((~a & b).sum()); n = gains + losses
    p = 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(gains, losses), n, .5)))
    return dict(gains=gains, losses=losses, discordant=n, p=p)


def paired_bootstrap(a, b, seed=0, resamples=RESAMPLES, alpha=.05):
    """Percentile interval of mean(a - b) over items; a and b may be (items,) or (seeds, items) for the pooled interval."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError('paired shape mismatch')
    d = a - b
    if d.ndim == 2:
        d = d.mean(0)  # pooled over seeds on the same items
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(d), size=(resamples, len(d)))
    means = d[idx].mean(1)
    return dict(delta=float(d.mean()), interval=[float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))], n=len(d), resamples=resamples)


def document_bootstrap(num_a, num_b, den, seed=0, resamples=RESAMPLES, alpha=.05):
    """Paired document bootstrap of the NLL difference (a - b) on identical documents and targets."""
    num_a = np.asarray(num_a, float); num_b = np.asarray(num_b, float); den = np.asarray(den, float)
    d = num_a - num_b
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(d), size=(resamples, len(d)))
    est = d[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    point = float(d.sum() / max(den.sum(), 1))
    # Sign-flip randomization p-value for the Holm family (exchangeable under the paired null).
    flips = rng.choice([-1.0, 1.0], size=(resamples, len(d)))
    null = (flips * d).sum(1) / max(den.sum(), 1)
    p = float(min(1.0, (1 + (np.abs(null) >= abs(point)).sum()) / (resamples + 1)))
    return dict(delta=point, interval=[float(np.quantile(est, alpha / 2)), float(np.quantile(est, 1 - alpha / 2))], p=p, documents=len(d), resamples=resamples)


def paired_signflip(a, b, seed=0, resamples=RESAMPLES):
    """Two-sided sign-flip randomization p-value for mean(a - b) over items; (seeds, items) arrays are averaged over seeds first."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.ndim == 2:
        d = d.mean(0)
    rng = np.random.RandomState(seed)
    point = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(resamples, len(d))) * d).mean(1)
    return float(min(1.0, (1 + (np.abs(null) >= abs(point) - 1e-12).sum()) / (resamples + 1)))


def holm(pvalues):
    """Holm step-down adjusted p-values for a dict name -> p."""
    names = sorted(pvalues, key=lambda k: pvalues[k]); m = len(names); out = {}; running = 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[k])); out[k] = running
    return out


def self_test():
    r = mcnemar_exact([1, 1, 1, 0, 0], [0, 0, 0, 0, 0])
    assert r['gains'] == 3 and r['losses'] == 0 and abs(r['p'] - .25) < 1e-12, r
    assert mcnemar_exact([1, 0], [1, 0])['p'] == 1.0
    b = paired_bootstrap(np.ones(100), np.zeros(100))
    assert b['delta'] == 1 and b['interval'] == [1.0, 1.0]
    rng = np.random.RandomState(1); x = rng.rand(500) < .6; y = rng.rand(500) < .4
    b = paired_bootstrap(x, y); assert b['interval'][0] > 0, b
    pooled = paired_bootstrap(np.stack([x, x]), np.stack([y, y])); assert abs(pooled['delta'] - b['delta']) < 1e-12
    h = holm(dict(a=.01, b=.04, c=.03)); assert abs(h['a'] - .03) < 1e-12 and abs(h['c'] - .06) < 1e-12 and abs(h['b'] - .06) < 1e-12, h
    assert paired_signflip(x, y) < .01 and paired_signflip(x, x) == 1.0
    den = np.full(200, 50.); na = rng.rand(200) * 100; nb = na + 5
    t = document_bootstrap(na, nb, den); assert abs(t['delta'] + .1) < 1e-12 and t['interval'][1] < 0 and t['p'] < .01, t
    return True


if __name__ == '__main__':
    print(self_test())
