"""Train R2GenGPT on IU-Xray or MIMIC-CXR. Examples:
    python train.py --align delta --run_name iu_delta
    python train.py --dataset mimic_cxr --align deep --run_name mimic_deep
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from r2gen.builder import add_model_args, build_model, generation_kwargs, get_device, resolve_args
from r2gen.data import ReportDataset, collate
from r2gen.engine import run_generation, train_one_epoch
from r2gen.labeler import positive_labels


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    g = p.add_argument_group("training")
    g.add_argument("--output_dir", default="runs")
    g.add_argument("--run_name", default=None)
    g.add_argument("--epochs", type=int, default=None, help="Paper: 15 for IU-Xray, 3 for MIMIC-CXR.")
    g.add_argument("--batch_size", type=int, default=6, help="Paper: 6 per GPU on 4 GPUs.")
    g.add_argument("--grad_accum", type=int, default=1)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--val_limit", type=int, default=None, help="Score only the first N val studies each epoch.")
    g.add_argument("--train_limit", type=int, default=None, help="Use only the first N train studies (debugging).")
    g.add_argument("--select_metric", default="BLEU_4", help="Validation metric that picks best.pt.")
    g.add_argument("--seed", type=int, default=42)
    return resolve_args(p.parse_args(argv))


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    run_name = args.run_name or f"{args.dataset}_{args.align}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args, device)
    trainable, total = model.count_parameters()
    print(f"device={device} dataset={args.dataset} align={args.align} mapper={args.mapper} "
          f"trainable={trainable / 1e6:.2f}M ({100 * trainable / total:.3f}% of {total / 1e6:.0f}M)")
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "trainable_params": trainable, "total_params": total}, f, indent=1)

    data_kw = dict(image_size=args.image_size, max_views=args.max_views, dataset=args.dataset,
                   mean=args.image_mean, std=args.image_std)
    train_set = ReportDataset(args.data_dir, "train", limit=args.train_limit, **data_kw)
    val_set = ReportDataset(args.data_dir, "val", limit=args.val_limit, **data_kw)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, args.batch_size, collate_fn=collate, num_workers=args.num_workers)

    if args.disease_prompt and args.disease_pos_weight == "balanced":
        pos = torch.from_numpy(positive_labels([train_set.clean(s["report"]) for s in train_set.samples])).float()
        model.disease_pos_weight = ((len(pos) - pos.sum(0)) / pos.sum(0).clamp_min(1)).clamp(max=10.0)
        print("disease head pos_weight:", [round(w, 2) for w in model.disease_pos_weight.tolist()])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_accum)
        train_minutes = (time.time() - start) / 60
        scores, _, preds = run_generation(model, val_loader, device, generation_kwargs(args))
        record = {"epoch": epoch, "train_loss": loss, "train_minutes": train_minutes,
                  "minutes": (time.time() - start) / 60, **scores}
        print(json.dumps(record))
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

        ckpt = {"args": vars(args), "epoch": epoch, "val": scores, "state_dict": model.trainable_state_dict()}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if scores[args.select_metric] > best:
            best = scores[args.select_metric]
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "val_predictions_best.json"), "w") as f:
                json.dump(preds, f, indent=1)
            print(f"  new best {args.select_metric}={best:.4f} -> {out_dir}/best.pt")
    return out_dir


if __name__ == "__main__":
    main()
