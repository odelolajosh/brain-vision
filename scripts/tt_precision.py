#!/usr/bin/env python3
"""
tt_precision.py — Per-fold tumour-tissue metrics aggregated across folds.

Fills the precision gap in Chapter 4, Section 4.6.4. Precision is not reported
by the test-evaluation notebook, so it is recomputed here from the saved
confusion matrices.

Run from the repo root:
    python scripts/tt_precision.py
    python scripts/tt_precision.py --class-index 0     # same for normal tissue
    python scripts/tt_precision.py --pooled            # add the pooled variant

Aggregation:
    Each fold is a distinct model evaluated on the SAME fixed test partition,
    so the five confusion matrices are not independent samples of the data and
    must not be summed — pooling would count every test pixel five times and
    understate the variance to zero. Metrics are therefore computed per fold
    and aggregated as median +/- population standard deviation, consistent with
    every other figure reported in the chapter.
"""

import argparse
import re
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "NT", 1: "TT", 2: "BV", 3: "BG"}
CLASS_FULL  = {0: "Normal tissue", 1: "Tumour tissue",
               2: "Blood vessel", 3: "Background"}
FOLD_RE = re.compile(r"fold[_-]?(\d+)", re.IGNORECASE)


def load_matrices(path: Path) -> dict:
    """Return {fold: confusion matrix} from a test_eval archive."""
    z = np.load(path, allow_pickle=True)
    out = {}
    for key in z.files:
        arr = np.asarray(z[key])
        m = FOLD_RE.search(key)
        if m and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            out[int(m.group(1))] = arr.astype(np.float64)
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                out.setdefault(i + 1, arr[i].astype(np.float64))
    return out


def metrics(cm: np.ndarray, k: int) -> dict:
    """Precision, recall, specificity and F1 for class index k."""
    n  = cm.sum()
    tp = cm[k, k]
    fp = cm[:, k].sum() - tp      # predicted k, actually something else
    fn = cm[k, :].sum() - tp      # actually k, predicted something else
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec  = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec * 100, "recall": rec * 100,
            "specificity": spec * 100, "f1": f1 * 100}


def agg(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return (np.median(vals), np.std(vals)) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description="Per-class metrics across folds")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--class-index", type=int, default=1,
                    help="0=NT 1=TT 2=BV 3=BG (default 1, tumour tissue)")
    ap.add_argument("--pooled", action="store_true",
                    help="Also show metrics from the summed matrix. Reported "
                         "for reference only — see the note in the docstring.")
    args = ap.parse_args()

    k = args.class_index
    cname = CLASS_NAMES.get(k, str(k))
    files = sorted(Path(args.results_dir).glob("test_eval_*.npz"))
    if not files:
        raise SystemExit(f"No test_eval_*.npz in {args.results_dir}/")

    print(f"Class: {CLASS_FULL.get(k, k)} ({cname})")
    print(f"Aggregation: median +/- population std across folds")
    print("=" * 78)

    summary = []

    for f in files:
        cfg = f.stem.replace("test_eval_", "")
        mats = load_matrices(f)
        if not mats:
            print(f"\n{cfg}: no confusion matrices recognised — skipped")
            continue

        rows = {fold: metrics(cm, k) for fold, cm in sorted(mats.items())}

        print(f"\n{cfg}")
        print(f"  {'Fold':>4} {'TP':>8} {'FP':>8} {'FN':>8} "
              f"{'Prec':>7} {'Recall':>7} {'Spec':>7} {'F1':>7}")
        print("  " + "-" * 62)
        for fold, m in rows.items():
            print(f"  {fold:>4} {m['tp']:>8,.0f} {m['fp']:>8,.0f} "
                  f"{m['fn']:>8,.0f} {m['precision']:>6.1f}% "
                  f"{m['recall']:>6.1f}% {m['specificity']:>6.1f}% "
                  f"{m['f1']:>6.1f}%")

        a = {key: agg([m[key] for m in rows.values()])
             for key in ("precision", "recall", "specificity", "f1")}
        print("  " + "-" * 62)
        print(f"  {'med':>4} {'':>8} {'':>8} {'':>8} "
              f"{a['precision'][0]:>6.1f}% {a['recall'][0]:>6.1f}% "
              f"{a['specificity'][0]:>6.1f}% {a['f1'][0]:>6.1f}%")
        print(f"  {'std':>4} {'':>8} {'':>8} {'':>8} "
              f"{a['precision'][1]:>6.1f}  {a['recall'][1]:>6.1f}  "
              f"{a['specificity'][1]:>6.1f}  {a['f1'][1]:>6.1f} ")

        if args.pooled:
            pooled = metrics(sum(mats.values()), k)
            print(f"  pooled (reference only): "
                  f"prec {pooled['precision']:.1f}%  "
                  f"recall {pooled['recall']:.1f}%  "
                  f"F1 {pooled['f1']:.1f}%")

        summary.append((cfg, a))

    # ── Chapter-ready table ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  {cname} METRICS — median +/- std across folds (%)")
    print("=" * 78)
    print(f"  {'Configuration':<36}{'Precision':>14}{'Recall':>14}{'F1':>14}")
    print("  " + "-" * 74)
    for cfg, a in summary:
        print(f"  {cfg:<36}"
              f"{a['precision'][0]:>8.1f} ± {a['precision'][1]:<4.1f}"
              f"{a['recall'][0]:>8.1f} ± {a['recall'][1]:<4.1f}"
              f"{a['f1'][0]:>8.1f} ± {a['f1'][1]:<4.1f}")

    # ── Markdown for the chapter ─────────────────────────────────────────────
    out = Path(args.results_dir) / f"{cname.lower()}_precision.md"
    with open(out, "w") as fh:
        fh.write(f"### {CLASS_FULL.get(k, k)} — precision, recall and F1\n\n")
        fh.write("Median ± population standard deviation across folds, "
                 "computed from the saved test confusion matrices. Each fold "
                 "is a distinct model evaluated on the same fixed test "
                 "partition, so per-fold metrics are aggregated rather than "
                 "the matrices being pooled.\n\n")
        fh.write("| Configuration | Precision (%) | Recall (%) | "
                 "Specificity (%) | F1 (%) |\n")
        fh.write("|---|---|---|---|---|\n")
        for cfg, a in summary:
            fh.write(f"| `{cfg}` | "
                     f"{a['precision'][0]:.1f} ± {a['precision'][1]:.1f} | "
                     f"{a['recall'][0]:.1f} ± {a['recall'][1]:.1f} | "
                     f"{a['specificity'][0]:.1f} ± {a['specificity'][1]:.1f} | "
                     f"{a['f1'][0]:.1f} ± {a['f1'][1]:.1f} |\n")
    print(f"\n  Markdown -> {out}")


if __name__ == "__main__":
    main()