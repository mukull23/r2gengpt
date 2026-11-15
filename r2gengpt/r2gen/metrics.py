"""Metrics used by the paper, plus the extra analyses listed as research gaps.

- Language quality (paper): BLEU-1..4, METEOR, ROUGE-L, CIDEr from coco-caption.
- Clinical efficacy (paper, MIMIC-CXR only; here for every dataset): precision/recall/F1 of 14 CheXpert labels.
- Gap 7 (bias toward normal cases): metrics on normal vs abnormal studies and per finding.
- Gap 8 (hallucination): how often a generated report asserts findings that the reference does not have.
- Gap 8 (RadGraph F1): optional, needs `pip install radgraph`.
"""
import shutil
import subprocess

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from .labeler import CONDITIONS, label_reports

NO_FINDING = CONDITIONS.index("No Finding")


def _java_available():
    # macOS ships a /usr/bin/java stub that fails when no JDK is installed.
    if not shutil.which("java"):
        return False
    return subprocess.run(["java", "-version"], capture_output=True).returncode == 0


def compute_nlg_metrics(references, predictions):
    """references / predictions: {study_id: report_text}. Returns {metric: score}."""
    gts = {k: [v] for k, v in references.items()}
    res = {k: [predictions[k] or "."] for k in references}  # empty strings break the scorers
    scores = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    scores.update({f"BLEU_{i + 1}": s for i, s in enumerate(bleu)})
    if _java_available():
        from pycocoevalcap.meteor.meteor import Meteor
        scores["METEOR"], _ = Meteor().compute_score(gts, res)
    scores["ROUGE_L"], _ = Rouge().compute_score(gts, res)
    scores["CIDEr"], _ = Cider().compute_score(gts, res)
    return {k: float(v) for k, v in scores.items()}


# ---------------------------------------------------------------- clinical labels
class CheXbertLabeler:
    """CheXbert (Smit et al., 2020) through the `f1chexbert` package; downloads its weights on first use."""

    def __init__(self, device=None):
        from f1chexbert import F1CheXbert
        self.model = F1CheXbert(device=device)

    def __call__(self, reports):
        rows = [self.model.get_label(r, mode="classification") for r in reports]
        return np.array([[np.nan if v == "" else float(v) for v in row] for row in rows], dtype=float)


def get_labeler(name="rules", device=None):
    """'rules' is the built-in CheXpert-style labeler; 'chexbert' gives numbers comparable with the literature."""
    if name == "rules":
        return label_reports
    if name == "chexbert":
        return CheXbertLabeler(device)
    raise ValueError(f"unknown labeler {name!r}")


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def clinical_efficacy(gt, pred):
    """gt / pred: binary (N, 14). Uncertain and missing labels count as negative, as in R2Gen's CE script."""
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    tp, fp, fn = (gt & pred).sum(0), (~gt & pred).sum(0), (gt & ~pred).sum(0)
    out = {}
    out["CE_precision"], out["CE_recall"], out["CE_F1"] = _prf(tp.sum(), fp.sum(), fn.sum())  # micro (paper)
    # Without "No Finding", which dominates IU-Xray: does the model detect the actual findings?
    f = [i for i in range(len(CONDITIONS)) if i != NO_FINDING]
    out["CE_precision_findings"], out["CE_recall_findings"], out["CE_F1_findings"] = _prf(
        tp[f].sum(), fp[f].sum(), fn[f].sum())
    per_label = [_prf(*c) for c in zip(tp, fp, fn)]
    out["CE_precision_macro"], out["CE_recall_macro"], out["CE_F1_macro"] = map(float, np.mean(per_label, 0))
    ex = [_prf((g & p).sum(), (~g & p).sum(), (g & ~p).sum()) for g, p in zip(gt, pred)]  # example-based
    out["CE_precision_example"], out["CE_recall_example"], out["CE_F1_example"] = map(float, np.mean(ex, 0))
    out = {k: float(v) for k, v in out.items()}
    out["per_label"] = {c: {"precision": p, "recall": r, "f1": f, "support": int(s)}
                        for c, (p, r, f), s in zip(CONDITIONS, per_label, gt.sum(0))}
    return out


def hallucination_stats(gt, pred):
    """A finding is hallucinated if the generated report asserts it and the reference does not (No Finding excluded)."""
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    keep = [i for i in range(len(CONDITIONS)) if i != NO_FINDING]
    g, p = gt[:, keep], pred[:, keep]
    extra, missed = p & ~g, g & ~p
    return {
        "hallucinated_findings_per_report": float(extra.sum(1).mean()) if len(g) else 0.0,
        "reports_with_hallucination": float(extra.any(1).mean()) if len(g) else 0.0,
        "omitted_findings_per_report": float(missed.sum(1).mean()) if len(g) else 0.0,
        "reports_with_omission": float(missed.any(1).mean()) if len(g) else 0.0,
        "abnormal_reports_called_normal": float((g.any(1) & pred[:, NO_FINDING]).sum() / max(g.any(1).sum(), 1)),
    }


def radgraph_f1(references, predictions):
    """RadGraph F1 (Jain et al., 2021; Delbrouck et al., 2022). Needs `pip install radgraph`."""
    from radgraph import F1RadGraph
    keys = list(references)
    (simple, partial, complete), *_ = F1RadGraph(reward_level="all")(
        refs=[references[k] for k in keys], hyps=[predictions[k] or "." for k in keys])
    return {"RadGraph_F1_simple": float(simple), "RadGraph_F1_partial": float(partial),
            "RadGraph_F1_complete": float(complete)}


def full_report(references, predictions, labeler="rules", radgraph=False, device=None):
    """Every metric for an evaluation run. Returns (scores, per-study labels)."""
    keys = list(references)
    scores = compute_nlg_metrics(references, predictions)
    label_fn = get_labeler(labeler, device)
    gt_raw = label_fn([references[k] for k in keys])
    pred_raw = label_fn([predictions[k] or "" for k in keys])
    gt, pred = (gt_raw == 1), (pred_raw == 1)
    ce = clinical_efficacy(gt, pred)
    scores.update({k: v for k, v in ce.items() if k != "per_label"})
    scores["labeler"] = labeler
    scores.update(hallucination_stats(gt, pred))
    if radgraph:
        scores.update(radgraph_f1(references, predictions))

    # Gap 7: is the model biased toward normal reports?
    normal = gt[:, NO_FINDING]
    subsets = {}
    for name, mask in (("normal", normal), ("abnormal", ~normal)):
        ids = [k for k, m in zip(keys, mask) if m]
        if ids:
            sub = compute_nlg_metrics({k: references[k] for k in ids}, {k: predictions[k] for k in ids})
            sub.update({k: v for k, v in clinical_efficacy(gt[mask], pred[mask]).items() if k != "per_label"})
            subsets[name] = {"n": len(ids), **sub}
    labels = {k: {"reference": _names(g), "generated": _names(p)} for k, g, p in zip(keys, gt_raw, pred_raw)}
    return {"overall": scores, "per_label": ce["per_label"], "subsets": subsets}, labels


def _names(row):
    return {c: int(v) for c, v in zip(CONDITIONS, row) if not np.isnan(v)}
