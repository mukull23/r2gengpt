"""Statistics for comparing runs (research gap 12): seed aggregation and paired bootstrap tests."""
from collections import Counter

import numpy as np

from .labeler import CONDITIONS, positive_labels

NO_FINDING = CONDITIONS.index("No Finding")  # the last column; the 13 before it are findings

_TINY, _SMALL = 1e-15, 1e-9  # the constants of coco-caption's bleu_scorer


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu_stats(reference, hypothesis, n=4):
    """Per-study sufficient statistics: [hyp_len, ref_len, correct_1..n, guess_1..n]."""
    ref, hyp = reference.split(), hypothesis.split()
    correct = [sum(min(c, _ngrams(ref, k)[g]) for g, c in _ngrams(hyp, k).items()) for k in range(1, n + 1)]
    guess = [max(len(hyp) - k + 1, 0) for k in range(1, n + 1)]
    return np.array([len(hyp), len(ref), *correct, *guess], dtype=float)


def corpus_bleu(stats, n=4):
    """Corpus BLEU-n from summed statistics; matches pycocoevalcap's Bleu with one reference per study."""
    s = stats.sum(0) if stats.ndim == 2 else stats
    hyp_len, ref_len, correct, guess = s[0], s[1], s[2:2 + n], s[2 + n:2 + 2 * n]
    score = np.prod((correct + _TINY) / (guess + _SMALL)) ** (1.0 / n)
    ratio = (hyp_len + _TINY) / (ref_len + _SMALL)
    return float(score * (np.exp(1 - 1 / ratio) if ratio < 1 else 1.0))


def micro_f1(gt, pred):
    tp, fp, fn = (gt & pred).sum(), (~gt & pred).sum(), (gt & ~pred).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


class _Run:
    """Per-study quantities of one predictions file, so resampled metrics are cheap."""

    def __init__(self, refs, preds, keys):
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        gts = {k: [refs[k]] for k in keys}
        res = {k: [preds[k] or "."] for k in keys}
        self.bleu = np.stack([bleu_stats(refs[k], preds[k] or ".") for k in keys])
        self.rouge = np.asarray(Rouge().compute_score(gts, res)[1], dtype=float)
        self.cider = np.asarray(Cider().compute_score(gts, res)[1], dtype=float)
        self.gt = positive_labels([refs[k] for k in keys]).astype(bool)
        self.pred = positive_labels([preds[k] or "" for k in keys]).astype(bool)

    def metrics(self, idx):
        return {"BLEU_4": corpus_bleu(self.bleu[idx]), "ROUGE_L": float(self.rouge[idx].mean()),
                "CIDEr": float(self.cider[idx].mean()), "CE_F1": micro_f1(self.gt[idx], self.pred[idx]),
                "CE_F1_findings": micro_f1(self.gt[idx][:, :NO_FINDING], self.pred[idx][:, :NO_FINDING])}


def paired_bootstrap(refs, preds_a, preds_b, n_boot=1000, seed=0):
    """Paired bootstrap over studies (Koehn, 2004). For each metric returns A, B, B-A, its 95% CI and a two-sided p.

    CIDEr's IDF weights are computed once on the full set; resampling then averages per-study scores.
    """
    keys = sorted(set(refs) & set(preds_a) & set(preds_b))
    a, b = _Run(refs, preds_a, keys), _Run(refs, preds_b, keys)
    full = np.arange(len(keys))
    base_a, base_b = a.metrics(full), b.metrics(full)
    rng = np.random.default_rng(seed)
    deltas = {m: [] for m in base_a}
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        ma, mb = a.metrics(idx), b.metrics(idx)
        for m in deltas:
            deltas[m].append(mb[m] - ma[m])
    out = {}
    for m, d in deltas.items():
        d = np.asarray(d)
        d[np.abs(d) < 1e-9] = 0.0  # ties (e.g. two zero BLEU scores) must not count as wins
        p = 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (n_boot + 1)  # add-one: p is never exactly 0
        out[m] = {"A": base_a[m], "B": base_b[m], "delta": base_b[m] - base_a[m],
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": float(min(p, 1.0))}
    return {"n": len(keys), "n_boot": n_boot, "metrics": out}


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
