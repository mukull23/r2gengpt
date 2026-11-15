"""Unit tests for the parts that need no model: labeler, metrics, statistics, data loading, token pooling.

    python -m pytest tests -q
"""
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2gen.builder import DEFAULT_DATA_DIR  # noqa: E402
from r2gen.data import (ReportDataset, clean_indication, clean_report_iu_xray, clean_report_mimic_cxr,  # noqa: E402
                        collate, load_iu_indications)
from r2gen.labeler import CONDITIONS, label_report  # noqa: E402
from r2gen.mappers import grid_tokens, pooling_matrix  # noqa: E402
from r2gen.metrics import clinical_efficacy, compute_nlg_metrics, hallucination_stats  # noqa: E402
from r2gen.stats import bleu_stats, corpus_bleu, paired_bootstrap  # noqa: E402

HAS_IU = os.path.exists(os.path.join(DEFAULT_DATA_DIR, "annotation.json"))


def labels(text):
    return {c: int(v) for c, v in zip(CONDITIONS, label_report(text)) if not np.isnan(v)}


# ---------------------------------------------------------------- labeler
@pytest.mark.parametrize("text, expected", [
    ("The heart size and pulmonary vascularity appear within normal limits. No pneumothorax or pleural effusion.",
     {"Cardiomegaly": 0, "Pneumothorax": 0, "Pleural Effusion": 0, "No Finding": 1}),
    ("The heart is enlarged. Small left pleural effusion.", {"Cardiomegaly": 1, "Pleural Effusion": 1}),
    ("Heart is not enlarged.", {"Cardiomegaly": 0}),
    ("Heart size normal with large hiatal hernia.", {"Cardiomegaly": 0}),
    ("Heart size is upper limits of normal.", {"Cardiomegaly": -1}),
    ("Pneumonia cannot be excluded.", {"Pneumonia": -1}),
    ("Possible left basilar atelectasis.", {"Atelectasis": -1}),
    ("Lingular scarring or atelectasis.", {"Atelectasis": -1}),
    ("No change in the right pleural effusion.", {"Pleural Effusion": 1}),
    ("The lungs are normally inflated without evidence of focal airspace disease pleural effusion or pneumothorax.",
     {"Lung Opacity": 0, "Pleural Effusion": 0, "Pneumothorax": 0, "No Finding": 1}),
    ("Left-sided pacemaker in place. Mild pulmonary edema.", {"Support Devices": 1, "Edema": 1, "No Finding": 0}),
    ("Old healed left rib fracture.", {"Fracture": 1}),
    ("No pericardial effusion.", {}),
])
def test_labeler(text, expected):
    got = labels(text)
    for cond, value in expected.items():
        assert got.get(cond) == value, (cond, got)
    if not expected:
        assert "Pleural Effusion" not in got


def test_labeler_same_on_cleaned_text():
    text = "The heart is enlarged. There is a small left pleural effusion, possibly with atelectasis. No pneumothorax."
    assert np.array_equal(label_report(text), label_report(clean_report_iu_xray(text)), equal_nan=True)


def test_no_finding_requires_no_positive_or_uncertain_pathology():
    assert labels("Support devices in place.")["No Finding"] == 1
    assert labels("Possible small effusion.")["No Finding"] == 0


# ---------------------------------------------------------------- metrics
def test_corpus_bleu_matches_coco():
    refs = {"a": "the heart is normal . no effusion .", "b": "small left pleural effusion .",
            "c": "lungs are clear . no pneumothorax ."}
    preds = {"a": "the heart is normal . no pleural effusion .", "b": "left effusion .", "c": "lungs clear ."}
    coco = compute_nlg_metrics(refs, preds)["BLEU_4"]
    ours = corpus_bleu(np.stack([bleu_stats(refs[k], preds[k]) for k in refs]))
    assert ours == pytest.approx(coco, rel=1e-9, abs=1e-12)


def test_clinical_efficacy():
    gt = np.zeros((2, 14), int)
    pred = np.zeros((2, 14), int)
    gt[0, 1] = gt[0, 9] = 1   # cardiomegaly + effusion
    pred[0, 1] = pred[0, 4] = 1   # cardiomegaly + edema (one hit, one false positive)
    gt[1, 13] = pred[1, 13] = 1  # both say no finding
    ce = clinical_efficacy(gt, pred)
    assert ce["CE_precision"] == pytest.approx(2 / 3)
    assert ce["CE_recall"] == pytest.approx(2 / 3)
    assert ce["CE_F1"] == pytest.approx(2 / 3)
    assert ce["per_label"]["Cardiomegaly"]["f1"] == 1.0 and ce["per_label"]["Pleural Effusion"]["recall"] == 0.0
    h = hallucination_stats(gt, pred)
    assert h["hallucinated_findings_per_report"] == 0.5 and h["reports_with_omission"] == 0.5


def test_paired_bootstrap_identical_runs_are_not_significant():
    refs = {str(i): "the heart is normal ." for i in range(20)}
    preds = {str(i): "the heart is enlarged ." for i in range(20)}
    out = paired_bootstrap(refs, preds, dict(preds), n_boot=100)
    for r in out["metrics"].values():
        assert r["delta"] == 0 and r["p"] == 1.0


# ---------------------------------------------------------------- mappers
def test_grid_tokens_and_pooling():
    feats = torch.randn(2, 50, 8)  # CLS + 7x7
    grid, side = grid_tokens(feats)
    assert side == 7 and grid.shape[1] == 49
    pool = pooling_matrix(49, 7, 16, torch.device("cpu"))
    assert pool.shape == (16, 49) and torch.allclose(pool.sum(1), torch.ones(16))
    assert torch.allclose(pool.sum(0).sum(), torch.tensor(16.0))
    pool1d = pooling_matrix(49, 7, 10, torch.device("cpu"))  # not a square: 1-D pooling
    assert pool1d.shape == (10, 49) and torch.allclose(pool1d.sum(1), torch.ones(10))


# ---------------------------------------------------------------- data
def test_cleaners():
    assert clean_report_iu_xray("1. Heart is normal. 2. No effusion..") == "heart is normal . no effusion ."
    assert clean_report_mimic_cxr("Heart size normal.\n\nNo effusion..") == "heart size normal . no effusion ."
    assert clean_indication("XXXX-year-old male with chest pain.") == "year-old male with chest pain"
    assert clean_indication("None.") == ""


@pytest.mark.skipif(not HAS_IU, reason="IU-Xray not downloaded")
def test_iu_indications_cover_the_prepared_split():
    ind = load_iu_indications(DEFAULT_DATA_DIR)
    with open(os.path.join(DEFAULT_DATA_DIR, "annotation.json")) as f:
        ids = [s["id"] for split in json.load(f).values() for s in split]
    assert all(i in ind for i in ids)
    assert sum(bool(ind[i]) for i in ids) / len(ids) > 0.9


def test_mimic_style_dataset(tmp_path):
    """A MIMIC-CXR-shaped package (one view, report with headers) loads through the same pipeline."""
    (tmp_path / "images" / "p10" / "s1").mkdir(parents=True)
    Image.new("L", (32, 32), 128).save(tmp_path / "images" / "p10" / "s1" / "view1.jpg")
    ann = {split: [{"id": "abc", "study_id": 1, "subject_id": 10, "report": "Moderate cardiomegaly.\n No effusion.",
                    "image_path": ["p10/s1/view1.jpg"], "split": split, "indication": "Dyspnea."}]
           for split in ("train", "val", "test")}
    (tmp_path / "annotation.json").write_text(json.dumps(ann))
    ds = ReportDataset(str(tmp_path), "test", image_size=32, max_views=2, dataset="mimic_cxr")
    item = ds[0]
    assert item["images"].shape == (2, 3, 32, 32)  # the single view is repeated
    assert item["report"] == "moderate cardiomegaly . no effusion ."
    assert item["indication"] == "dyspnea"
    assert item["labels"][CONDITIONS.index("Cardiomegaly")] == 1
    batch = collate([item, item])
    assert batch["images"].shape == (2, 2, 3, 32, 32) and batch["labels"].shape == (2, 14)
