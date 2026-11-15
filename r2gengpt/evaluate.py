"""Score a trained checkpoint on a test split: language metrics, clinical efficacy, normal-vs-abnormal
breakdown and hallucination statistics. Examples:
    python evaluate.py --checkpoint runs/iu_delta/best.pt
    python evaluate.py --checkpoint runs/mimic_deep/best.pt --dataset iu_xray      # cross-dataset (gap 10)
    python evaluate.py --checkpoint runs/iu_delta/best.pt --num_beams 1 --tag _beam1
"""
import argparse
import json
import os

from torch.utils.data import DataLoader

from r2gen.builder import DATASET_ROOT, generation_kwargs, get_device, load_checkpoint
from r2gen.data import DATASETS, ReportDataset, collate
from r2gen.engine import generate_reports
from r2gen.metrics import full_report

GENERATION_OVERRIDES = {"num_beams": int, "max_new_tokens": int, "min_new_tokens": int,
                        "repetition_penalty": float, "length_penalty": float}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=DATASETS, default=None, help="Default: the dataset it was trained on.")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--labeler", choices=("rules", "chexbert"), default="rules",
                   help="rules: built-in CheXpert-style labeler; chexbert: needs `pip install f1chexbert`.")
    p.add_argument("--radgraph", action="store_true", help="Also compute RadGraph F1 (`pip install radgraph`).")
    p.add_argument("--tag", default="", help="Suffix for the output file names.")
    for name, typ in GENERATION_OVERRIDES.items():
        p.add_argument(f"--{name}", type=typ, default=None, help="Override the checkpoint's setting.")
    return p.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    device = get_device()
    overrides = {k: getattr(cli, k) for k in GENERATION_OVERRIDES if getattr(cli, k) is not None}
    model, args = load_checkpoint(cli.checkpoint, device, overrides)
    dataset = cli.dataset or args.dataset
    data_dir = cli.data_dir or (args.data_dir if dataset == args.dataset else os.path.join(DATASET_ROOT, dataset))
    data = ReportDataset(data_dir, cli.split, args.image_size, args.max_views, cli.limit, dataset=dataset,
                         mean=args.image_mean, std=args.image_std)
    loader = DataLoader(data, cli.batch_size, collate_fn=collate, num_workers=cli.num_workers)
    refs, preds = generate_reports(model, loader, device, generation_kwargs(args))
    report, labels = full_report(refs, preds, labeler=cli.labeler, radgraph=cli.radgraph, device=device)
    report["setup"] = {"checkpoint": cli.checkpoint, "trained_on": args.dataset, "evaluated_on": dataset,
                       "split": cli.split, "n": len(refs), **generation_kwargs(args)}

    out_dir = os.path.dirname(os.path.abspath(cli.checkpoint))
    prefix = os.path.join(out_dir, f"{cli.split}{'' if dataset == args.dataset else '_' + dataset}{cli.tag}")
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(report, f, indent=1)
    with open(f"{prefix}_predictions.json", "w") as f:
        json.dump({k: {"reference": refs[k], "generated": preds[k], "labels": labels[k]} for k in refs}, f, indent=1)
    print(json.dumps(report["overall"], indent=1))
    print(f"saved {prefix}_metrics.json and {prefix}_predictions.json")
    return report


if __name__ == "__main__":
    main()
