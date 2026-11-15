"""A rule-based, CheXpert-style labeler for chest X-ray reports.

The paper scores clinical efficacy with the CheXpert labeler (Irvin et al., 2019): every report is tagged with
14 observations, and precision/recall/F1 compare the tags of generated and reference reports. The official
labeler needs NegBio and a Java/Python 2-era toolchain, so this module re-implements its idea (mention
extraction, then negation and uncertainty detection) with regular expressions. It works on raw reports and on
the cleaned lower-case text that the model generates.

It is an approximation. Report it as such, or use `--labeler chexbert` (see metrics.py) for numbers that are
comparable with the literature.

Each observation gets one of: 1 positive, 0 negative, -1 uncertain, nan not mentioned (CheXpert convention).
"""
import re

import numpy as np

CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

# Mentions that are abnormal in themselves ("effusion"). Their status comes from negation/uncertainty cues.
_MENTIONS = {
    "Enlarged Cardiomediastinum": [r"mediastinal (widening|enlargement)", r"wid(e|ened) mediastinum"],
    "Cardiomegaly": [r"cardiomegaly", r"cardiac enlargement", r"enlarged (heart|cardiac)"],
    "Lung Opacity": [r"opaci(ty|ties|fication|fied)", r"densit(y|ies)", r"infiltrat\w*", r"airspace (disease|process)",
                     r"haz(y|iness)", r"ground ?glass", r"interstitial (markings?|prominence|disease|thickening|pattern)",
                     r"reticul\w*"],
    "Lung Lesion": [r"nodul(e|es|ar)", r"mass(es)?", r"(?<!bony )lesions?", r"cavit(y|ies|ation|ary)", r"cysts?"],
    "Edema": [r"edema", r"(vascular|venous) congestion", r"pulmonary congestion", r"heart failure", r"chf",
              r"kerley", r"cephalization"],
    "Consolidation": [r"consolidat\w*"],
    "Pneumonia": [r"pneumoni\w*", r"infection", r"infectious process"],
    "Atelectasis": [r"atelecta\w*", r"collapse"],
    "Pneumothorax": [r"pneumothora(x|ces)"],
    "Pleural Effusion": [r"(?<!pericardial )effusions?", r"pleural fluid", r"blunting", r"blunted"],
    "Pleural Other": [r"pleural (thickening|scar\w*|plaques?|calcification\w*|abnormalit\w*)", r"fibrothorax"],
    "Fracture": [r"fractur\w*", r"fx"],
    "Support Devices": [r"pacemaker", r"pacer", r"defibrillator", r"aicd", r"catheter", r"picc", r"(central|venous|mid|arterial) line",
                        r"(endotracheal|tracheostomy|chest|feeding|enteric|nasogastric|ng|et|og) tubes?", r"tubes? and lines?",
                        r"lines? (and|or) tubes?", r"lines?", r"stents?", r"(surgical )?clips?", r"sternotomy", r"sternal wires",
                        r"(valve|joint) (replacement|prosthesis)", r"prosthe\w*", r"port(-a-cath)?", r"drains?", r"hardware",
                        r"leads?", r"devices?", r"swan", r"generator"],
}

# Anatomy whose status depends on a modifier: "heart is enlarged" is positive, "heart size is normal" negative.
_ANATOMY = {
    "Cardiomegaly": r"(heart|cardiac)( size| silhouette| shadow| contours?)?",
    "Enlarged Cardiomediastinum": r"(cardio)?mediastin\w*( silhouette| contours?| width)?",
}
_ABNORMAL_SIZE = re.compile(r"\b(enlarged|enlargement|large|increased|prominent|prominence|widened|widening|cardiomegaly)\b")
_NORMAL_SIZE = re.compile(r"\b(normal|within normal limits|unremarkable|stable|not enlarged|no enlargement)\b")
_BORDERLINE = re.compile(r"\b(upper limits? of normal|borderline|top normal|top-normal|mildly prominent)\b")

_UNCERTAIN_STRONG = re.compile(
    r"\b(cannot (be )?(excluded|ruled out|exclude|rule out)|can not be (excluded|ruled out)|not (be )?excluded|"
    r"rule out|r/o|versus|vs)\b")
_NEGATION_PRE = re.compile(
    r"\b(no|not|without|negative for|free of|clear of|absence of|absent|resolution of|resolved|removal of|"
    r"removed|rather than|exclude[sd]?|deny|denies|nor)\b")
_NEGATION_POST = re.compile(
    r"^\W*(\w+\W+){0,4}?(is|are|has|have|was|were)?\W*(not (seen|identified|present|visualized|visible|demonstrated|"
    r"evident|appreciated|noted)|absent|resolved|removed|cleared|no longer (seen|visible|present))\b")
_UNCERTAIN = re.compile(
    r"\b(may|might|could|possibl[ey]|probabl[ey]|likely|questionabl[ey]|question of|suspicious|suspected|suspect|"
    r"suggest\w*|concern\w* for|consider\w*|differential|indeterminate|equivocal|cannot|presum\w*)\b")
# "no change in the effusion" means the effusion is still there.
_NOT_NEGATION = re.compile(r"\b(no|without) (\w+ ){0,2}(change|changed|interval change|significant change)\b|\bnot changed\b")
_CLAUSE_BREAK = re.compile(r"\b(?:but|however|although|though|except|otherwise|whereas|which|aside from)\b|\band (?:there|the)\b")
_NEGATION_WINDOW = 10  # words between a negation cue and the mention it scopes over


def _compile(patterns):
    return re.compile(r"\b(" + "|".join(patterns) + r")\b")


_MENTION_RE = {c: _compile(p) for c, p in _MENTIONS.items()}
_ANATOMY_RE = {c: re.compile(r"\b" + p + r"\b") for c, p in _ANATOMY.items()}


def _clauses(report):
    text = report.lower().replace("x-xxxx", " ").replace("xxxx", " ")
    for sentence in re.split(r"[.;:\n]", text):
        sentence = _NOT_NEGATION.sub(" unchanged ", sentence)
        for clause in _CLAUSE_BREAK.split(sentence):
            if clause.strip():
                yield " ".join(clause.split())


def _status(clause, start, end):
    before, after = clause[:start], clause[end:]
    near_before = " ".join(before.split()[-_NEGATION_WINDOW:])
    if _UNCERTAIN_STRONG.search(near_before) or _UNCERTAIN_STRONG.search(after):
        return -1
    if _NEGATION_PRE.search(near_before) or _NEGATION_POST.search(after):
        return 0
    # Outside a negated list, "scarring or atelectasis" leaves both findings uncertain.
    prev, nxt = before.split()[-1:], after.split()[:3]
    if _UNCERTAIN.search(near_before) or _UNCERTAIN.search(" ".join(nxt)) or prev == ["or"] or nxt[:1] == ["or"]:
        return -1
    return 1


def _anatomy_status(clause, match):
    """The modifier closest after the anatomy decides: "heart size normal with large hernia" is negative."""
    window = " ".join(clause[match.start():].split()[:_NEGATION_WINDOW + 2])
    found = [(m.start(), kind, m) for kind, rx in (("borderline", _BORDERLINE), ("normal", _NORMAL_SIZE),
                                                   ("abnormal", _ABNORMAL_SIZE)) if (m := rx.search(window))]
    if not found:
        return 0 if _NORMAL_SIZE.search(clause[:match.start()]) else None  # "normal heart size"
    _, kind, m = min(found, key=lambda f: f[0])
    if kind == "borderline":
        return -1
    if kind == "normal":
        return 0
    return _status(clause, match.start() + m.start(), match.start() + m.end())


def _merge(old, new):
    """Positive beats uncertain beats negative across a report's mentions (CheXpert aggregation)."""
    if new is None:
        return old
    if old is None or np.isnan(old):
        return new
    rank = {1: 2, -1: 1, 0: 0}
    return new if rank[new] > rank[int(old)] else old


def label_report(report):
    """Return a length-14 float array in CONDITIONS order: 1, 0, -1 or nan."""
    out = {c: None for c in CONDITIONS}
    for clause in _clauses(report):
        for cond, rx in _MENTION_RE.items():
            for m in rx.finditer(clause):
                out[cond] = _merge(out[cond], _status(clause, m.start(), m.end()))
        for cond, rx in _ANATOMY_RE.items():
            for m in rx.finditer(clause):
                out[cond] = _merge(out[cond], _anatomy_status(clause, m))
    pathologies = [out[c] for c in CONDITIONS if c not in ("Support Devices", "No Finding")]
    out["No Finding"] = 0 if any(s in (1, -1) for s in pathologies) else 1
    return np.array([np.nan if out[c] is None else out[c] for c in CONDITIONS], dtype=float)


def label_reports(reports):
    return np.stack([label_report(r) for r in reports]) if reports else np.zeros((0, len(CONDITIONS)))


def positive_labels(reports):
    """Binary matrix (N, 14): 1 only for positive mentions, as in R2Gen's clinical-efficacy script."""
    return (label_reports(reports) == 1).astype(np.int64)
