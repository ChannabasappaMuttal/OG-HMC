# ============================================================
# compute_hvr.py
#
# Compute HVR (Hierarchical Violation Rate) on any saved
# checkpoint — no retraining needed.
#
# Usage:
#   # On the best checkpoint from a training run:
#   python compute_hvr.py
#
#   # On a specific checkpoint:
#   python compute_hvr.py --ckpt checkpoints/HierFashion_seed42/hierfashion-best.ckpt
#
#   # On a specific seed:
#   python compute_hvr.py --seed 123
#
#   # Skip threshold recalibration (faster, uses fixed thresholds):
#   python compute_hvr.py --no-calibrate
#
# Output:
#   Prints HVR table to console
#   Saves results/hvr_results.csv
# ============================================================

import os, sys, math, glob, argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from core.common import (
    load_hierarchy, build_adjacency_matrix, FashionDataset,
    make_train_val_split, compute_pos_weights,
    evaluate_hierarchical, calibrate_thresholds,
    compute_per_group_attr_f1,
    get_logger, set_seed,
    TRAIN_ANN, TRAIN_IMAGES, VAL_ANN, VAL_IMAGES,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
    BATCH_SIZE, NUM_WORKERS, DATASET_ROOT
)
from torch.utils.data import DataLoader

logger = get_logger("compute_hvr")


def find_best_checkpoint(seed=42):
    """Find the most recently modified checkpoint for a given seed."""
    patterns = [
        f"checkpoints/HierFashion_seed{seed}/*.ckpt",
        f"checkpoints/HierFashion/*.ckpt",
        "checkpoints/**/*.ckpt",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Pick the one with 'best' in the name, else most recent
            best = [m for m in matches if "best" in m]
            chosen = best[0] if best else sorted(matches, key=os.path.getmtime)[-1]
            logger.info(f"Auto-selected checkpoint: {chosen}")
            return chosen
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute HVR on a saved checkpoint")
    parser.add_argument("--ckpt",         type=str,  default=None,
                        help="Path to .ckpt file. Auto-detected if not given.")
    parser.add_argument("--seed",         type=int,  default=42,
                        help="Seed used during training (for checkpoint lookup).")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip threshold calibration (faster, less accurate).")
    parser.add_argument("--save",         type=str,
                        default="results/hvr_results.csv",
                        help="Path to save CSV results.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Hierarchy ──────────────────────────────────────────
    hierarchy, hj = load_hierarchy()
    adj = build_adjacency_matrix(hierarchy, hj)

    # ── Data ───────────────────────────────────────────────
    train_sub, val_sub = make_train_val_split(
        annotation_file=TRAIN_ANN, image_dir=TRAIN_IMAGES,
        hierarchy=hierarchy, train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM, val_fraction=0.20, seed=args.seed,
    )
    test_ds = FashionDataset(VAL_IMAGES, VAL_ANN, hierarchy, VAL_TRANSFORM)

    train_dl = DataLoader(train_sub, BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_sub,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # ── Load checkpoint ────────────────────────────────────
    ckpt_path = args.ckpt or find_best_checkpoint(args.seed)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"\nNo checkpoint found. Searched in checkpoints/")
        print("Run  python final.py  first to train a model, then re-run this script.")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")

    pos_weight = compute_pos_weights(train_dl, hierarchy, device)
    steps_est  = math.ceil(len(train_sub) / BATCH_SIZE)

    from final import HierFashionLightning, LR_IMPROVED
    model = HierFashionLightning.load_from_checkpoint(
        ckpt_path,
        hierarchy=hierarchy,
        adj_matrix=adj,
        pos_weight=pos_weight,
        learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_est,
    )
    model.to(device)
    model.eval()
    logger.info("Checkpoint loaded.")

    # ── Optional calibration ───────────────────────────────
    threshold_config = None
    if not args.no_calibrate:
        logger.info("Calibrating thresholds on val set...")
        threshold_config = calibrate_thresholds(
            model, val_dl, device, hierarchy,
            top_k_candidates=(3, 5, 7, 10, 15),
        )
    else:
        logger.info("Skipping calibration — using fixed thresholds.")

    # ── Evaluate (HVR is computed inside evaluate_hierarchical) ──
    logger.info("Running evaluation on test set...")
    os.makedirs("results", exist_ok=True)
    results = evaluate_hierarchical(
        model, test_dl, device, model.loss_fn,
        save_path=args.save,
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    # ── Print clean summary ────────────────────────────────
    metric_rows = results[~results["Head"].isin(["CompositeLoss", "HVR"])]
    hvr_row     = results[results["Head"] == "HVR"]

    print("\n" + "=" * 60)
    print(f"  HVR EVALUATION — checkpoint: {os.path.basename(ckpt_path)}")
    print("=" * 60)

    # Per-head F1 table
    cols = ["Head", "Macro_F1", "Micro_F1", "mAP"]
    print(metric_rows[cols].to_string(index=False))

    # HVR section
    if not hvr_row.empty:
        r = hvr_row.iloc[0]
        hvr        = r["L_bce"]
        hvr_sub    = r["L_consistency"]
        hvr_attr   = r["L_path"]
        viol_pct   = r["CHL"]

        print("\n" + "-" * 60)
        print("  HIERARCHICAL VIOLATION RATE")
        print("-" * 60)
        print(f"  HVR overall              : {hvr:.4f}  ({hvr*100:.1f}%)")
        print(f"  HVR_subcat  (V1)         : {hvr_sub:.4f}  ({hvr_sub*100:.1f}%)")
        print(f"    → images where a subcat is predicted without its parent category")
        print(f"  HVR_attr    (V2)         : {hvr_attr:.4f}  ({hvr_attr*100:.1f}%)")
        print(f"    → images where an attr is predicted without its parent group")
        print(f"  Attr viol. per prediction: {viol_pct:.4f}  ({viol_pct*100:.1f}%)")
        print(f"    → of all predicted attrs, this % lack a predicted parent group")

        # Benchmark context
        print()
        print("  Benchmark context (lower HVR = better hierarchy coherence):")
        print(f"    HVR < 0.05  → excellent  (strong hierarchy regularisation)")
        print(f"    HVR 0.05–0.15 → good     (minor violations)")
        print(f"    HVR 0.15–0.30 → moderate (consider raising lambda_consistency)")
        print(f"    HVR > 0.30  → high       (hierarchy loss may not be converging)")

        verdict = (
            "EXCELLENT" if hvr < 0.05 else
            "GOOD"      if hvr < 0.15 else
            "MODERATE"  if hvr < 0.30 else
            "HIGH"
        )
        print(f"\n  Your model: {verdict}  (HVR = {hvr:.4f})")
        print("-" * 60)

    # ── Per-group attribute F1 ──────────────────────────────
    group_prefix = "attr_group_"
    group_rows   = results[results["Head"].str.startswith(group_prefix)]
    if not group_rows.empty:
        print("\n" + "=" * 65)
        print("  ATTRIBUTE F1 BY GROUP")
        print("=" * 65)
        print(f"  {'Group':42s} {'n':>5} {'MacroF1':>8} {'MicroF1':>8} {'mAP':>7}")
        print("  " + "─" * 72)
        summary_keys = {f"{group_prefix}nickname_group",
                        f"{group_prefix}non_nickname_groups"}
        for _, r in group_rows.iterrows():
            if r["Head"] in summary_keys:
                continue
            name = r["Head"].replace(group_prefix, "").replace("_", " ")
            print(f"    {name:42s} {int(r['L_bce']):>5} "
                  f"{r['Macro_F1']:>8.4f} {r['Micro_F1']:>8.4f} {r['mAP']:>7.4f}")
        print("  " + "─" * 72)
        for key, label in [(f"{group_prefix}nickname_group",     "nickname (52% of attrs)"),
                           (f"{group_prefix}non_nickname_groups","non-nickname (48%)")]:
            r = group_rows[group_rows["Head"] == key]
            if not r.empty:
                r = r.iloc[0]
                print(f"  * {label:42s} {int(r['L_bce']):>5} "
                      f"{r['Macro_F1']:>8.4f} {r['Micro_F1']:>8.4f} {r['mAP']:>7.4f}")
        print("=" * 65)

    print(f"\nFull results saved → {args.save}")


if __name__ == "__main__":
    main()
