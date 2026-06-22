# ============================================================
# compute_hvr_all.py
#
# Compute HVR + per-group attribute F1 for ALL trained models
# (HierFashion + all 8 baselines) from saved checkpoints.
#
# Usage:
#   # Run on all models automatically:
#   python compute_hvr_all.py
#
#   # Run on specific models only:
#   python compute_hvr_all.py --models HierFashion Q2L CNNHier
#
#   # Skip threshold calibration (faster):
#   python compute_hvr_all.py --no-calibrate
#
#   # Save comparison table to custom path:
#   python compute_hvr_all.py --save results/hvr_comparison.csv
#
# Output:
#   - Per-model HVR + per-group F1 printed to console
#   - results/hvr_all_models.csv  (one row per model)
#   - results/hvr_per_group.csv   (per-group breakdown per model)
# ============================================================

import os, sys, math, glob, argparse
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.common import (
    load_hierarchy, build_adjacency_matrix, FashionDataset,
    make_train_val_split, compute_pos_weights,
    evaluate_hierarchical, calibrate_thresholds,
    compute_per_group_attr_f1, compute_hvr,
    get_logger, set_seed,
    TRAIN_ANN, TRAIN_IMAGES, VAL_ANN, VAL_IMAGES,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
    BATCH_SIZE, NUM_WORKERS, DATASET_ROOT
)
from torch.utils.data import DataLoader

logger = get_logger("compute_hvr_all")


# ============================================================
# MODEL REGISTRY
# Each entry: (display_name, ModelClass, ckpt_dir_pattern, build_kwargs_fn)
# build_kwargs_fn(hierarchy, adj, pos_weight) → dict for load_from_checkpoint
# ============================================================

# def _build_hierfashion(hierarchy, adj, pos_weight, steps):
#     from final import HierFashionLightning, LR_IMPROVED
#     return dict(hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight,
#                 learning_rate=LR_IMPROVED, steps_per_epoch=steps), HierFashionLightning

def _build_cnn_flat(hierarchy, adj, pos_weight, steps):
    from models.cnn_flat import CNNFlatLightning
    return dict(hierarchy=hierarchy, pos_weight=pos_weight), CNNFlatLightning

def _build_cnn_hier(hierarchy, adj, pos_weight, steps):
    from models.cnn_hier import CNNHierLightning
    return dict(hierarchy=hierarchy, pos_weight=pos_weight), CNNHierLightning

def _build_cnn_gat(hierarchy, adj, pos_weight, steps):
    from models.cnn_gat import CNNGATLightning
    return dict(hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight), CNNGATLightning

def _build_mlgcn(hierarchy, adj, pos_weight, steps):
    from models.ml_gcn import MLGCNLightning
    return dict(hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight), MLGCNLightning

def _build_q2l(hierarchy, adj, pos_weight, steps):
    from models.q2l import Q2LLightning
    return dict(hierarchy=hierarchy, pos_weight=pos_weight), Q2LLightning

def _build_resnet_asl(hierarchy, adj, pos_weight, steps):
    from models.resnet_asl import ResNetASLLightning
    return dict(hierarchy=hierarchy), ResNetASLLightning

def _build_tresnet(hierarchy, adj, pos_weight, steps):
    from models.tresnet import TResNetLightning
    return dict(hierarchy=hierarchy, pos_weight=pos_weight), TResNetLightning

def _build_vit_mlc(hierarchy, adj, pos_weight, steps):
    from models.vit_mlc import ViTMLCLightning
    return dict(hierarchy=hierarchy, pos_weight=pos_weight), ViTMLCLightning


MODEL_REGISTRY = {
    # name           ckpt_dir_glob                          builder_fn
    "CNNFlat":     ("models/checkpoints/cnn_flat*",                _build_cnn_flat),
    "CNNHier":     ("models/checkpoints/cnn_hier*",                _build_cnn_hier),
    "CNNGAT":      ("models/checkpoints/cnn_gat*",                 _build_cnn_gat),
    "MLGCN":       ("models/checkpoints/ml_gcn*",                  _build_mlgcn),
    "Q2L":         ("models/checkpoints/q2l*",                    _build_q2l),
    "ResNetASL":   ("models/checkpoints/resnet_asl*",              _build_resnet_asl),
    "TResNet":     ("models/checkpoints/tresnet*",                _build_tresnet),
    "ViTMLC":      ("models/checkpoints/vit_mlc*",                 _build_vit_mlc),
}


def find_checkpoint(ckpt_glob):
    """Find best checkpoint matching glob pattern."""
    matches = glob.glob(ckpt_glob + "/**/*.ckpt", recursive=True)
    if not matches:
        matches = glob.glob(ckpt_glob + "/*.ckpt")
    if not matches:
        return None
    # Prefer 'best' in name, else most recently modified
    best = [m for m in matches if "best" in os.path.basename(m)]
    return best[0] if best else sorted(matches, key=os.path.getmtime)[-1]


def evaluate_one_model(model_name, builder_fn, ckpt_path,
                       hierarchy, adj, pos_weight, steps,
                       val_dl, test_dl, device,
                       use_calibration=True):
    """Load checkpoint, calibrate, evaluate, return results df."""

    logger.info(f"\n{'='*55}\n  {model_name}\n{'='*55}")
    logger.info(f"  Checkpoint: {ckpt_path}")

    kwargs, ModelClass = builder_fn(hierarchy, adj, pos_weight, steps)

    try:
        model = ModelClass.load_from_checkpoint(ckpt_path, **kwargs)
    except Exception as e:
        logger.error(f"  Failed to load {model_name}: {e}")
        return None

    model.to(device)
    model.eval()

    threshold_config = None
    if use_calibration:
        logger.info(f"  Calibrating thresholds...")
        try:
            threshold_config = calibrate_thresholds(
                model, val_dl, device, hierarchy,
                top_k_candidates=(3, 5, 7, 10, 15),
            )
        except Exception as e:
            logger.warning(f"  Calibration failed ({e}) — using fixed thresholds")

    logger.info(f"  Evaluating on test set...")
    os.makedirs("results", exist_ok=True)
    try:
        results = evaluate_hierarchical(
            model, test_dl, device, model.loss_fn,
            save_path=f"results/hvr_{model_name}_test_results.csv",
            threshold_config=threshold_config,
            hierarchy=hierarchy,
        )
    except Exception as e:
        logger.error(f"  Evaluation failed: {e}")
        return None

    return results


def extract_summary(model_name, results):
    """Extract one-row summary from a results df."""
    overall = results[results["Head"] == "OverallModel"]
    hvr_row = results[results["Head"] == "HVR"]
    nick_row = results[results["Head"] == "attr_group_nickname_group"]
    non_nick = results[results["Head"] == "attr_group_non_nickname_groups"]
    attr_row = results[results["Head"] == "attributes"]

    row = {"Model": model_name}

    if not overall.empty:
        r = overall.iloc[0]
        row.update({
            "Overall_Macro_F1": r["Macro_F1"],
            "Overall_Micro_F1": r["Micro_F1"],
            "Overall_mAP":      r["mAP"],
        })

    if not attr_row.empty:
        r = attr_row.iloc[0]
        row.update({
            "Attr_Macro_F1": r["Macro_F1"],
            "Attr_Micro_F1": r["Micro_F1"],
            "Attr_mAP":      r["mAP"],
        })

    if not nick_row.empty:
        row["Nickname_Macro_F1"] = nick_row.iloc[0]["Macro_F1"]

    if not non_nick.empty:
        row["NonNickname_Macro_F1"] = non_nick.iloc[0]["Macro_F1"]

    if not hvr_row.empty:
        r = hvr_row.iloc[0]
        row.update({
            "HVR":              r["L_bce"],
            "HVR_subcat":       r["L_consistency"],
            "HVR_attr":         r["L_path"],
            "Attr_viol_pct":    r["CHL"],
        })

    return row


def extract_per_group(model_name, results):
    """Extract per-group F1 rows."""
    group_rows = results[results["Head"].str.startswith("attr_group_")]
    rows = []
    for _, r in group_rows.iterrows():
        group = r["Head"].replace("attr_group_", "").replace("_", " ")
        rows.append({
            "Model":      model_name,
            "Group":      group,
            "n_attrs":    int(r["L_bce"]) if pd.notna(r["L_bce"]) else None,
            "Macro_F1":   r["Macro_F1"],
            "Micro_F1":   r["Micro_F1"],
            "mAP":        r["mAP"],
        })
    return rows


def print_comparison_table(summary_rows):
    """Print a clean side-by-side comparison of all models."""
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows).set_index("Model")

    print("\n" + "=" * 90)
    print("  HVR + ATTRIBUTE F1 — ALL MODELS COMPARISON")
    print("=" * 90)

    cols_f1  = ["Overall_Macro_F1", "Attr_Macro_F1",
                "Nickname_Macro_F1", "NonNickname_Macro_F1"]
    cols_hvr = ["HVR", "HVR_subcat", "HVR_attr", "Attr_viol_pct"]

    available_f1  = [c for c in cols_f1  if c in df.columns]
    available_hvr = [c for c in cols_hvr if c in df.columns]

    print(f"\n  {'Model':15s}", end="")
    for c in available_f1:
        print(f"  {c.replace('_',' '):>20s}", end="")
    print()
    print("  " + "─" * (15 + 22 * len(available_f1)))
    for model, row in df.iterrows():
        print(f"  {model:15s}", end="")
        for c in available_f1:
            val = row.get(c)
            print(f"  {val:>20.4f}" if pd.notna(val) else f"  {'N/A':>20s}", end="")
        print()

    print(f"\n  {'Model':15s}", end="")
    for c in available_hvr:
        print(f"  {c.replace('_',' '):>15s}", end="")
    print()
    print("  " + "─" * (15 + 17 * len(available_hvr)))
    for model, row in df.iterrows():
        print(f"  {model:15s}", end="")
        for c in available_hvr:
            val = row.get(c)
            print(f"  {val:>15.4f}" if pd.notna(val) else f"  {'N/A':>15s}", end="")
        print()

    # Best model per metric
    print("\n  Best per metric:")
    for c in available_f1 + available_hvr:
        if c not in df.columns:
            continue
        series = df[c].dropna()
        if series.empty:
            continue
        # Lower is better for HVR metrics
        if "HVR" in c or "viol" in c:
            best_model = series.idxmin()
            best_val   = series.min()
            note = "(lower = better)"
        else:
            best_model = series.idxmax()
            best_val   = series.max()
            note = "(higher = better)"
        print(f"    {c:30s}: {best_model:15s}  {best_val:.4f}  {note}")

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute HVR for all trained models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to evaluate (default: all found)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip threshold calibration (faster)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str,
                        default="results/hvr_all_models.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Hierarchy & data ───────────────────────────────────
    hierarchy, hj = load_hierarchy()
    adj = build_adjacency_matrix(hierarchy, hj)

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

    pos_weight = compute_pos_weights(train_dl, hierarchy, device)
    steps      = math.ceil(len(train_sub) / BATCH_SIZE)

    # ── Determine which models to evaluate ─────────────────
    models_to_run = args.models or list(MODEL_REGISTRY.keys())

    # Filter to only models that have a checkpoint
    runnable = {}
    skipped  = []
    for name in models_to_run:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model '{name}' — skipping. "
                           f"Valid: {list(MODEL_REGISTRY.keys())}")
            continue
        ckpt_glob, builder_fn = MODEL_REGISTRY[name]
        ckpt = find_checkpoint(ckpt_glob)
        if ckpt is None:
            logger.warning(f"No checkpoint found for {name} "
                           f"(searched: {ckpt_glob}/**/*.ckpt) — skipping.")
            skipped.append(name)
        else:
            runnable[name] = (ckpt, builder_fn)

    if not runnable:
        print("\nNo checkpoints found. Train models first with:")
        print("  python final.py              # HierFashion")
        print("  python run_all_experiments.py  # all baselines")
        return

    print(f"\nModels with checkpoints: {list(runnable.keys())}")
    if skipped:
        print(f"Skipped (no checkpoint):  {skipped}")

    # ── Evaluate each model ─────────────────────────────────
    summary_rows  = []
    per_group_rows = []

    for model_name, (ckpt_path, builder_fn) in runnable.items():
        results = evaluate_one_model(
            model_name, builder_fn, ckpt_path,
            hierarchy, adj, pos_weight, steps,
            val_dl, test_dl, device,
            use_calibration=not args.no_calibrate,
        )
        if results is None:
            continue

        summary_rows.append(extract_summary(model_name, results))
        per_group_rows.extend(extract_per_group(model_name, results))

    # ── Print comparison table ──────────────────────────────
    print_comparison_table(summary_rows)

    # ── Save CSVs ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values("Overall_Macro_F1",
                                            ascending=False,
                                            ignore_index=True)
        summary_df.to_csv(args.save, index=False)
        print(f"\nSummary saved → {args.save}")

    if per_group_rows:
        pg_df = pd.DataFrame(per_group_rows)
        pg_path = args.save.replace(".csv", "_per_group.csv")
        pg_df.to_csv(pg_path, index=False)
        print(f"Per-group saved → {pg_path}")

    if skipped:
        print(f"\nNot evaluated (no checkpoint): {skipped}")
        print("Train them with: python run_all_experiments.py")


if __name__ == "__main__":
    main()
