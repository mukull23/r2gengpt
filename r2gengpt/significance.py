"""Is run B better than run A? Paired bootstrap over test studies (research gap 12). Example:
    python significance.py runs/iu_shallow/test_predictions.json runs/iu_delta/test_predictions.json
"""
import argparse
import json

from r2gen.stats import paired_bootstrap


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {k: v["reference"] for k, v in data.items()}, {k: v["generated"] for k, v in data.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="Baseline *_predictions.json from evaluate.py")
    p.add_argument("b", help="Challenger *_predictions.json")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    cli = p.parse_args()

    refs_a, preds_a = load(cli.a)
    refs_b, preds_b = load(cli.b)
    shared = set(refs_a) & set(refs_b)
    assert shared and all(refs_a[k] == refs_b[k] for k in shared), "the two files must cover the same studies"
    result = paired_bootstrap(refs_a, preds_a, preds_b, cli.n_boot, cli.seed)
    result.update(a=cli.a, b=cli.b)

    print(f"{result['n']} paired studies, {cli.n_boot} bootstrap resamples (B - A)")
    print(f"{'metric':8s} {'A':>8s} {'B':>8s} {'delta':>8s}   95% CI               p")
    for m, r in result["metrics"].items():
        star = " *" if r["p"] < 0.05 else ""
        print(f"{m:8s} {r['A']:8.4f} {r['B']:8.4f} {r['delta']:+8.4f}   "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  {r['p']:.3f}{star}")
    if cli.out:
        with open(cli.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
