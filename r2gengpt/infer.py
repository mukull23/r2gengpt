"""Generate a report for one study (one or more views). Example:
    python infer.py --checkpoint runs/iu_delta/best.pt --images frontal.png lateral.png
    python infer.py --checkpoint runs/iu_ctx/best.pt --images f.png l.png --indication "chest pain"
"""
import argparse

from r2gen.builder import generation_kwargs, get_device, load_checkpoint
from r2gen.data import build_transform, clean_indication, load_views
from r2gen.labeler import CONDITIONS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--indication", default="", help="Used by models trained with --prompt_context indication.")
    cli = p.parse_args()

    device = get_device()
    model, args = load_checkpoint(cli.checkpoint, device)
    transform = build_transform(args.image_size, mean=args.image_mean, std=args.image_std)
    images = load_views(cli.images, transform, args.max_views).unsqueeze(0).to(device)
    reports, probs = model.generate(images, indications=[clean_indication(cli.indication)],
                                    return_findings=True, **generation_kwargs(args))
    print(reports[0])
    if probs is not None:
        found = [f"{c} ({p:.2f})" for c, p in zip(CONDITIONS, probs[0].tolist()) if p > 0.5]
        print("predicted findings:", ", ".join(found) or "none")


if __name__ == "__main__":
    main()
