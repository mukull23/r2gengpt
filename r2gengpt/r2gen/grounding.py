"""Sentence-to-region grounding for generated reports (research gap 9).

Two saliency methods, both mapped back to the encoder's patch grid through the visual-token maps:
  attention - the LLM's attention from each report token to the visual tokens, averaged over layers and heads
  gradient  - |gradient x input| of the sentence's log-likelihood w.r.t. each visual token
"""
import re

import torch
import torch.nn.functional as F


def use_eager_attention(llm):
    """SDPA/flash kernels do not return attention weights; switch the LLM to the eager implementation."""
    if hasattr(llm, "set_attn_implementation"):
        try:
            llm.set_attn_implementation("eager")
            return
        except Exception:
            pass
    llm.config._attn_implementation = "eager"


def sentence_spans(report):
    """Character spans of the sentences of a cleaned report ("a b . c d .")."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^.]+\.?", report) if m.group().strip(" .")]


def ground_report(model, images, report, indications=None, method="attention"):
    """Return {"sentences": [...], "token_maps": (S+1, T), "grid": (S+1, V, side, side) or None, "layout"}.

    Row 0 is the whole report; rows 1.. are its sentences. images: (1, views, 3, H, W).
    """
    model.eval()
    tok = model.tokenizer
    enc = tok(report + tok.eos_token, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc.input_ids, enc.offset_mapping
    spans = sentence_spans(report)
    groups = [list(range(len(ids) - 1))]  # whole report (without </s>)
    groups += [[j for j, (s, _) in enumerate(offsets[:-1]) if a <= s < b] for a, b in spans]

    old_len, model.max_report_len = model.max_report_len, len(ids) + 1
    try:
        with torch.no_grad():
            visual = model.encode_images(images, return_maps=True)
            findings = None
            if model.disease_prompt:
                findings = (model.predict_findings(visual) > 0.5).tolist()
        contexts = model.contexts(1, indications, findings)
        if method == "attention":
            use_eager_attention(model.llm)
            with torch.no_grad():
                inputs = model.build_inputs(visual, [report], contexts)
                out = model.llm(inputs_embeds=inputs["inputs_embeds"], attention_mask=inputs["attention_mask"],
                                position_ids=inputs["position_ids"], output_attentions=True)
            att = torch.stack([a[0].float() for a in out.attentions]).mean((0, 1))  # (seq, seq)
            p, start, t = inputs["prefix_len"], inputs["image_starts"][0], visual["tokens"].shape[1]
            per_token = att[p - 1: p - 1 + len(ids), start: start + t]  # row j predicts report token j
            per_token = per_token / per_token.sum(-1, keepdim=True).clamp_min(1e-12)
            token_maps = torch.stack([per_token[g].mean(0) if g else torch.zeros(t, device=att.device)
                                      for g in groups])
        elif method == "gradient":
            tokens = visual["tokens"].detach().float().requires_grad_(True)
            inputs = model.build_inputs({**visual, "tokens": tokens}, [report], contexts)
            logits = model.llm(inputs_embeds=inputs["inputs_embeds"], attention_mask=inputs["attention_mask"],
                               position_ids=inputs["position_ids"]).logits[0].float()
            p = inputs["prefix_len"]
            target = torch.tensor(ids, device=logits.device)
            logp = F.log_softmax(logits[p - 1: p - 1 + len(ids)], -1).gather(1, target[:, None])[:, 0]
            rows = []
            for g in groups:
                if not g:
                    rows.append(torch.zeros(tokens.shape[1], device=tokens.device))
                    continue
                (grad,) = torch.autograd.grad(logp[g].sum(), tokens, retain_graph=True)
                rows.append((grad * tokens).sum(-1)[0].abs().detach())
            token_maps = torch.stack(rows)
        else:
            raise ValueError("method must be 'attention' or 'gradient'")
    finally:
        model.max_report_len = old_len

    views, side = visual["layout"]
    grid = None
    if side is not None:
        cells = token_maps.float() @ visual["maps"][0]  # (S+1, V*G)
        grid = cells.view(len(groups), views, side, side)
        grid = grid / grid.flatten(1).amax(1).clamp_min(1e-12)[:, None, None, None]
    return {"sentences": [report] + [report[a:b].strip() for a, b in spans],
            "token_maps": token_maps.cpu(), "grid": None if grid is None else grid.cpu(), "layout": (views, side)}


def upsample(grid, size):
    """(V, side, side) -> (V, H, W) bilinear heatmaps in [0, 1]."""
    up = F.interpolate(grid[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    return (up - up.amin((1, 2), keepdim=True)) / (up.amax((1, 2), keepdim=True) - up.amin((1, 2), keepdim=True)
                                                    ).clamp_min(1e-12)
