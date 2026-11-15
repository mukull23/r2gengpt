"""Training and generation loops shared by train.py, evaluate.py and smoke_test.py."""
import contextlib

import torch
from tqdm import tqdm

from .labeler import positive_labels
from .metrics import clinical_efficacy, compute_nlg_metrics


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum=1, log_every=20, max_steps=None):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    bar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(bar):
        if max_steps is not None and step >= max_steps:
            break
        with autocast(device):
            loss = model(batch["images"].to(device), batch["report"], indications=batch["indication"],
                         labels=batch["labels"].to(device)) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total += loss.item() * grad_accum
        n += 1
        if step % log_every == 0:
            bar.set_postfix(loss=f"{total / n:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total / max(n, 1)


@torch.no_grad()
def generate_reports(model, loader, device, gen_kwargs):
    """Generate a report for every study. Returns ({id: reference}, {id: generated})."""
    model.eval()
    refs, preds = {}, {}
    for batch in tqdm(loader, desc="generate", leave=False):
        with autocast(device):
            outputs = model.generate(batch["images"].to(device), indications=batch["indication"], **gen_kwargs)
        for sid, ref, pred in zip(batch["id"], batch["report"], outputs):
            refs[sid], preds[sid] = ref, pred
    return refs, preds


def run_generation(model, loader, device, gen_kwargs):
    """Generate every report and score it with the language and clinical-efficacy metrics."""
    refs, preds = generate_reports(model, loader, device, gen_kwargs)
    scores = compute_nlg_metrics(refs, preds)
    keys = list(refs)
    ce = clinical_efficacy(positive_labels([refs[k] for k in keys]), positive_labels([preds[k] for k in keys]))
    scores.update({k: ce[k] for k in ("CE_F1", "CE_F1_findings", "CE_recall_findings")})
    return scores, refs, preds
