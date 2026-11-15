"""Show which image regions each generated sentence relies on (research gap 9). Examples:
    python explain.py --checkpoint runs/iu_delta/best.pt --study CXR2384_IM-0942
    python explain.py --checkpoint runs/iu_delta/best.pt --images frontal.png lateral.png --method gradient

Writes <out>.png (rows: whole report, then each sentence; columns: views) and <out>.json.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from r2gen.builder import generation_kwargs, get_device, load_checkpoint
from r2gen.data import CLEANERS, build_transform, clean_indication, load_iu_indications, load_views
from r2gen.grounding import ground_report, upsample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="Image files of one study.")
    src.add_argument("--study", help="A study id from the checkpoint's dataset annotation.json.")
    p.add_argument("--indication", default=None)
    p.add_argument("--report", default=None, help="Explain this text instead of the model's own report.")
    p.add_argument("--method", choices=("attention", "gradient"), default="attention")
    p.add_argument("--out", default=None, help="Output path without extension (default: next to the checkpoint).")
    return p.parse_args()


def find_study(args, study_id):
    with open(os.path.join(args.data_dir, "annotation.json")) as f:
        for split, samples in json.load(f).items():
            for s in samples:
                if str(s["id"]) == study_id:
                    paths = [os.path.join(args.data_dir, "images", x) for x in s["image_path"]]
                    indication = s.get("indication") or (
                        load_iu_indications(args.data_dir).get(study_id, "") if args.dataset == "iu_xray" else "")
                    return paths, CLEANERS[args.dataset](s["report"]), clean_indication(indication)
    raise SystemExit(f"study {study_id} not found in {args.data_dir}/annotation.json")


def main():
    cli = parse_args()
    device = get_device()
    model, args = load_checkpoint(cli.checkpoint, device, attn_implementation="eager")
    reference = None
    if cli.study:
        paths, reference, indication = find_study(args, cli.study)
    else:
        paths, indication = cli.images, ""
    if cli.indication is not None:
        indication = clean_indication(cli.indication)
    transform = build_transform(args.image_size, mean=args.image_mean, std=args.image_std)
    images = load_views(paths, transform, args.max_views).unsqueeze(0).to(device)

    report = cli.report or model.generate(images, indications=[indication], **generation_kwargs(args))[0]
    result = ground_report(model, images, report, indications=[indication], method=cli.method)

    out = cli.out or os.path.join(os.path.dirname(os.path.abspath(cli.checkpoint)),
                                  f"explain_{cli.study or 'images'}_{cli.method}")
    summary = {"images": paths, "indication": indication, "reference": reference, "generated": report,
               "method": cli.method, "sentences": []}
    raw = [Image.open(x).convert("L").resize((args.image_size, args.image_size)) for x in paths[: args.max_views]]
    grid = result["grid"]
    for i, sentence in enumerate(result["sentences"]):
        entry = {"sentence": sentence, "visual_token_weights": [round(v, 5) for v in result["token_maps"][i].tolist()]}
        if grid is not None:
            g = grid[i]
            entry["peak_cell_per_view"] = [list(np.unravel_index(int(v.argmax()), v.shape)) for v in g]
        summary["sentences"].append(entry)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=1, default=int)

    if grid is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        heat = [upsample(g, (args.image_size, args.image_size)).numpy() for g in grid]
        rows, cols = len(heat), len(raw)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.imshow(raw[c], cmap="gray")
                ax.imshow(heat[r][min(c, heat[r].shape[0] - 1)], cmap="jet", alpha=0.4)
                ax.axis("off")
            title = "whole report" if r == 0 else result["sentences"][r]
            axes[r][0].set_title(title[:70] + ("..." if len(title) > 70 else ""), fontsize=8, loc="left")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=120)
        print(f"saved {out}.png")
    else:
        print("the encoder's tokens do not form a spatial grid; only token weights were saved")
    print(f"saved {out}.json")
    print("generated:", report)
    if reference:
        print("reference:", reference)


if __name__ == "__main__":
    main()
