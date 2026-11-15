"""IU-Xray and MIMIC-CXR loading, following the R2Gen / R2GenGPT preprocessing.

Both datasets use the R2Gen-style package: `annotation.json` with train/val/test lists of
{"id", "report", "image_path": [...]} entries and an `images/` folder. MIMIC-CXR needs credentialed
PhysioNet access; download the R2Gen/R2GenGPT version of its annotation into dataset/mimic_cxr/.
"""
import glob
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .labeler import label_report

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DATASETS = ("iu_xray", "mimic_cxr")


def clean_report_iu_xray(report):
    """Same cleaning as R2Gen (Chen et al., 2020), which R2GenGPT reuses."""
    report_cleaner = lambda t: (
        t.replace("..", ".").replace("..", ".").replace("..", ".")
        .replace("1. ", "").replace(". 2. ", ". ").replace(". 3. ", ". ")
        .replace(". 4. ", ". ").replace(". 5. ", ". ").replace(" 2. ", ". ")
        .replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ")
        .strip().lower().split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        r"[.,?;*!%^&_+():-\[\]{}]", "",
        t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower(),
    )
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
    return " . ".join(tokens) + " ."


def clean_report_mimic_cxr(report):
    """R2Gen's MIMIC-CXR cleaning (also used by R2GenGPT)."""
    report_cleaner = lambda t: (
        t.replace("\n", " ").replace("__", "_").replace("__", "_").replace("__", "_")
        .replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")
        .replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
        .replace("  ", " ").replace("  ", " ").replace("..", ".").replace("..", ".")
        .replace("..", ".").replace("..", ".").replace("..", ".").replace("..", ".")
        .replace("..", ".").replace("..", ".").replace("1. ", "").replace(". 2. ", ". ")
        .replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ").replace(" 2. ", ". ")
        .replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ")
        .strip().lower().split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        r"[.,?;*!%^&_+():-\[\]{}]", "",
        t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower(),
    )
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
    return " . ".join(tokens) + " ."


CLEANERS = {"iu_xray": clean_report_iu_xray, "mimic_cxr": clean_report_mimic_cxr}


def clean_indication(text):
    """Lower-case, drop the "XXXX" de-identification marks and stray punctuation."""
    text = re.sub(r"x*-?xxxx-?", " ", (text or "").lower())
    text = re.sub(r"[^a-z0-9 ,./-]", " ", text)
    text = " ".join(text.split()).strip(" .,;:-")
    return "" if text in ("", "none", "na", "n/a") else text


def load_iu_indications(data_dir):
    """INDICATION section of each IU-Xray study, from the original NLM XML reports (research gap 6).

    The prepared annotation ids ("CXR2384_IM-0942") are the XML parentImage ids without the view suffix.
    """
    out = {}
    for path in glob.glob(os.path.join(data_dir, "reports", "ecgen-radiology", "*.xml")):
        with open(path, encoding="utf-8") as f:
            xml = f.read()
        m = re.search(r'Label="INDICATION">(.*?)</AbstractText>', xml, re.S)
        indication = clean_indication(m.group(1)) if m else ""
        for image_id in re.findall(r'parentImage id="([^"]+)"', xml):
            out[image_id.rsplit("-", 1)[0]] = indication
    return out


def build_transform(image_size=224, train=False, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        # Mild augmentation only; flips would swap left/right anatomy.
        ops.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def load_views(paths, transform, max_views):
    """Stack up to max_views images; single-view studies repeat the view so every batch has the same shape."""
    images = [transform(Image.open(p).convert("RGB")) for p in paths[:max_views]]
    while len(images) < max_views:
        images.append(images[-1])
    return torch.stack(images)  # (views, 3, H, W)


class ReportDataset(Dataset):
    """One item per study: its views, the cleaned reference report, the indication and CheXpert-style labels."""

    def __init__(self, data_dir, split, image_size=224, max_views=2, limit=None, dataset="iu_xray",
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
        with open(os.path.join(data_dir, "annotation.json")) as f:
            self.samples = json.load(f)[split]
        if limit:
            self.samples = self.samples[:limit]
        self.image_dir = os.path.join(data_dir, "images")
        self.max_views = max_views
        self.clean = CLEANERS[dataset]
        self.transform = build_transform(image_size, train=(split == "train"), mean=mean, std=std)
        # IU-Xray indications come from the NLM XML files; MIMIC entries may carry an "indication" field.
        self.indications = load_iu_indications(data_dir) if dataset == "iu_xray" else {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        paths = [os.path.join(self.image_dir, p) for p in sample["image_path"]]
        report = self.clean(sample["report"])
        indication = clean_indication(sample.get("indication")) or self.indications.get(sample["id"], "")
        return {
            "id": str(sample["id"]),
            "images": load_views(paths, self.transform, self.max_views),
            "report": report,
            "indication": indication,
            "labels": torch.from_numpy(label_report(report) == 1).float(),  # positives, for --disease_prompt
        }


IUXrayDataset = ReportDataset  # backwards-compatible name


def collate(batch):
    return {
        "id": [b["id"] for b in batch],
        "images": torch.stack([b["images"] for b in batch]),  # (B, views, 3, H, W)
        "report": [b["report"] for b in batch],
        "indication": [b["indication"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch]),  # (B, 14)
    }
