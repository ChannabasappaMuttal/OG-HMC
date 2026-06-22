# ============================================================
# run_myntra_ablation.py
# ============================================================
# Runs ALL ablation studies on the Myntra dataset.
#
# Two groups — mirrors run_all_experiments.py exactly:
#
# GROUP 1 — Individual component removal (A1–A7):
#   A1  Ablation_NoGAT               replace GAT with plain MLP
#   A2  Ablation_NoCrossAttn         replace cross-attn with global pool
#   A3  Ablation_NoHierLoss          plain BCE per head, no CHL terms
#   A4  Ablation_NoLabelGuidedAttn   replace LGSA with global avg pool
#   A5  Ablation_NoConsistency       CHL without L_consistency
#   A6  Ablation_NoPath              CHL without L_path
#   A7  Ablation_NoHierMask          no attribute masking
#
# GROUP 2 — Incremental build-up (B2–B4):
#   B2  Incremental_B2_GAT           B1(CNNHier) + GAT
#   B3  Incremental_B3_LGSA          B2 + Label-Guided Spatial Attention
#   B4  Incremental_B4_CrossAttn     B3 + Cross-Attention (full arch, plain loss)
#   B5  HierFashion_Full             B4 + Hierarchical Loss = final model
#
# Output:
#   results/myntra_ablation_removal.csv      ← removal table
#   results/myntra_ablation_incremental.csv  ← incremental table
#   results/myntra_ablation_all.csv          ← combined
#   results/myntra_ablation_<name>_test_results.csv  ← per-model detail
#
# USAGE:
#   # Run all ablations (assumes annotations already built)
#   python run_myntra_ablation.py \
#       --hierfashion_root ~/version11/"HierFashion Code" \
#       --output_dir ~/version11/"HierFashion Code"/dataset/myntra_coco
#
#   # Run only specific groups
#   python run_myntra_ablation.py \
#       --hierfashion_root ... --output_dir ... \
#       --groups removal          # only A1-A7
#       --groups incremental      # only B2-B4 + full model
#       --groups removal incremental  # both (default)
#
#   # Run only specific ablations by name
#   python run_myntra_ablation.py \
#       --hierfashion_root ... --output_dir ... \
#       --only NoGAT NoCrossAttn HierFashion_Full
# ============================================================

import os
import sys
import math
import argparse
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("myntra_ablation")


# ─────────────────────────────────────────────────────────────
# PATH INJECTION  (must happen before any model imports)
# ─────────────────────────────────────────────────────────────

def inject_paths(hierfashion_root, output_dir):
    if hierfashion_root not in sys.path:
        sys.path.insert(0, hierfashion_root)
    import core.common as _cm
    _cm.DATASET_ROOT   = output_dir
    _cm.TRAIN_IMAGES   = os.path.join(output_dir, "train2020")
    _cm.VAL_IMAGES     = os.path.join(output_dir, "val2020")
    _cm.TRAIN_ANN      = os.path.join(output_dir, "instances_attributes_train2020.json")
    _cm.VAL_ANN        = os.path.join(output_dir, "instances_attributes_val2020.json")
    _cm.HIERARCHY_PATH = os.path.join(hierfashion_root,
                                       "hierarchy_outputs",
                                       "fashionpedia_hierarchy.json")
    log.info(f"Paths injected → DATASET_ROOT={output_dir}")


# ─────────────────────────────────────────────────────────────
# TRAINER + TRAIN_AND_EVAL HELPERS
# ─────────────────────────────────────────────────────────────

def _make_trainer(name):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    from core.common import EPOCHS, PATIENCE
    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1,
                dirpath=f"checkpoints/myntra_abl_{name}",
                filename=f"myntra_abl_{name}-{{epoch:02d}}-{{val_loss:.4f}}",
            ),
            EarlyStopping(monitor="val_loss", mode="min",
                          patience=PATIENCE, verbose=False),
        ],
        logger=CSVLogger("training_logs", name=f"myntra_abl_{name}"),
        enable_progress_bar=True,
    )


def train_and_eval(ModelClass, name, build_kwargs,
                   train_dl, val_dl, test_dl, device,
                   ckpt_extra_kwargs=None,
                   use_calibration=False, hierarchy=None):
    import torch
    from core.common import evaluate_hierarchical, analyze_model, calibrate_thresholds

    log.info(f"\n{'='*60}\nABLATION (Myntra): {name}\n{'='*60}")

    model = ModelClass(**build_kwargs)
    analyze_model(model, device)

    trainer = _make_trainer(name)
    trainer.fit(model, train_dl, val_dl)

    ckpt_cb   = trainer.checkpoint_callback
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        load_kw = {**build_kwargs, **(ckpt_extra_kwargs or {})}
        model   = ModelClass.load_from_checkpoint(best_ckpt, **load_kw)
        model.to(device)
    else:
        log.warning(f"No checkpoint for {name} — using last weights")

    threshold_config = None
    if use_calibration and hierarchy is not None:
        log.info(f"[{name}] Calibrating thresholds on val set...")
        threshold_config = calibrate_thresholds(
            model, val_dl, device, hierarchy,
            top_k_candidates=(3, 5, 7, 10),
        )

    os.makedirs("results", exist_ok=True)
    df = evaluate_hierarchical(
        model, test_dl, device, model.loss_fn,
        save_path=f"results/myntra_ablation_{name}_test_results.csv",
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    row = df[df["Head"] == "OverallModel"]
    if row.empty:
        return {"Model": name}
    r = row.iloc[0]
    return {
        "Model":           name,
        "Macro_F1":        r.get("Macro_F1"),
        "Micro_F1":        r.get("Micro_F1"),
        "Macro_Precision": r.get("Macro_Precision"),
        "Micro_Precision": r.get("Micro_Precision"),
        "Macro_Recall":    r.get("Macro_Recall"),
        "Micro_Recall":    r.get("Micro_Recall"),
        "mAP":             r.get("mAP"),
    }


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run ablation studies on the Myntra dataset")
    p.add_argument("--hierfashion_root", required=True,
                   help='Path to "HierFashion Code" directory')
    p.add_argument("--output_dir", required=True,
                   help="Myntra COCO output dir (must contain train2020/ etc.)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups", nargs="+",
                   choices=["removal", "incremental"],
                   default=["removal", "incremental"],
                   help="Which ablation groups to run")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these specific ablation names, e.g. "
                        "NoGAT NoCrossAttn HierFashion_Full")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    hierfashion_root = str(Path(args.hierfashion_root).expanduser().resolve())
    output_dir       = str(Path(args.output_dir).expanduser().resolve())

    # Validate annotations exist
    for fname in ["train2020", "val2020",
                  "instances_attributes_train2020.json",
                  "instances_attributes_val2020.json"]:
        p = Path(output_dir) / fname
        if not p.exists():
            log.error(f"Missing: {p}")
            log.error("Run run_fashion_product_dataset.py first to build annotations.")
            sys.exit(1)

    # Inject paths BEFORE any import of core.common / ablation modules
    inject_paths(hierfashion_root, output_dir)

    # Now safe to import everything
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from core.common import (
        load_hierarchy, build_adjacency_matrix,
        FashionDataset, make_train_val_split,
        compute_pos_weights, set_seed,
        TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
        TRAIN_TRANSFORM, VAL_TRANSFORM,
        BATCH_SIZE, NUM_WORKERS,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  seed: {args.seed}")

    # ── Data setup ────────────────────────────────────────────
    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix = build_adjacency_matrix(hierarchy, hierarchy_json)

    train_subset, val_subset = make_train_val_split(
        annotation_file=TRAIN_ANN, image_dir=TRAIN_IMAGES,
        hierarchy=hierarchy, train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM, val_fraction=0.20, seed=args.seed,
    )
    test_ds = FashionDataset(VAL_IMAGES, VAL_ANN, hierarchy, VAL_TRANSFORM)

    train_dl = DataLoader(train_subset, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_subset,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,      BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    pos_weight      = compute_pos_weights(train_dl, hierarchy, device)
    steps_per_epoch = math.ceil(len(train_subset) / BATCH_SIZE)

    # ── Helper to decide whether to run a given ablation ──────
    def _run(name):
        if args.only is not None:
            return name in args.only
        return True

    # ── Shared kwargs for ablations that need adj_matrix ──────
    kw_abl  = dict(hierarchy=hierarchy, adj_matrix=adj_matrix,
                   pos_weight=pos_weight)

    removal_results     = []
    incremental_results = []

    # ==========================================================
    # GROUP 1 — Individual Component Removal  (A1–A7)
    # ==========================================================
    if "removal" in args.groups:

        # A1 — No GAT
        if _run("NoGAT"):
            from ablation.ablation_no_gat import AblationNoGATLightning
            removal_results.append(train_and_eval(
                AblationNoGATLightning, "NoGAT", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A2 — No Cross-Attention
        if _run("NoCrossAttn"):
            from ablation.ablation_no_cross_attn import AblationNoCrossAttnLightning
            removal_results.append(train_and_eval(
                AblationNoCrossAttnLightning, "NoCrossAttn", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A3 — No Hierarchical Loss
        if _run("NoHierLoss"):
            from ablation.ablation_no_hier_loss import AblationNoHierLossLightning
            removal_results.append(train_and_eval(
                AblationNoHierLossLightning, "NoHierLoss", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A4 — No Label-Guided Spatial Attention
        if _run("NoLabelGuidedAttn"):
            from ablation.ablation_no_label_guided_attn import AblationNoLabelGuidedAttnLightning
            removal_results.append(train_and_eval(
                AblationNoLabelGuidedAttnLightning, "NoLabelGuidedAttn", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A5 — No Consistency Loss
        if _run("NoConsistency"):
            from ablation.ablation_no_consistency import AblationNoConsistencyLightning
            removal_results.append(train_and_eval(
                AblationNoConsistencyLightning, "NoConsistency", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A6 — No Path Coherence
        if _run("NoPath"):
            from ablation.ablation_no_path_coherence import AblationNoPathLightning
            removal_results.append(train_and_eval(
                AblationNoPathLightning, "NoPath", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # A7 — No Hierarchical Masking
        if _run("NoHierMask"):
            from ablation.ablation_no_hier_mask import AblationNoHierMaskLightning
            removal_results.append(train_and_eval(
                AblationNoHierMaskLightning, "NoHierMask", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

    # ==========================================================
    # GROUP 2 — Incremental Build-up  (B2 → B4 → B5=Full)
    # ==========================================================
    if "incremental" in args.groups:

        # B2 — GAT added
        if _run("Incremental_B2_GAT"):
            from ablation.incremental_b2_gat import IncrementalB2Lightning
            incremental_results.append(train_and_eval(
                IncrementalB2Lightning, "Incremental_B2_GAT", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # B3 — LGSA added
        if _run("Incremental_B3_LGSA"):
            from ablation.incremental_b3_lgsa import IncrementalB3Lightning
            incremental_results.append(train_and_eval(
                IncrementalB3Lightning, "Incremental_B3_LGSA", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # B4 — Cross-Attention added
        if _run("Incremental_B4_CrossAttn"):
            from ablation.incremental_b4_crossattn import IncrementalB4Lightning
            incremental_results.append(train_and_eval(
                IncrementalB4Lightning, "Incremental_B4_CrossAttn", kw_abl,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_abl,
            ))

        # B5 — Full HierFashion (hierarchical loss added = final model)
        if _run("HierFashion_Full"):
            from final import HierFashionLightning, LR_IMPROVED
            kw_full = dict(hierarchy=hierarchy, adj_matrix=adj_matrix,
                           pos_weight=pos_weight, learning_rate=LR_IMPROVED,
                           steps_per_epoch=steps_per_epoch)
            incremental_results.append(train_and_eval(
                HierFashionLightning, "HierFashion_Full", kw_full,
                train_dl, val_dl, test_dl, device,
                ckpt_extra_kwargs=kw_full,
                use_calibration=True, hierarchy=hierarchy,
            ))

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================
    os.makedirs("results", exist_ok=True)

    def _save(rows, path, sort_col="mAP"):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df.to_csv(path, index=False)
        log.info(f"\n{df.to_string(index=False)}")
        log.info(f"Saved → {path}")
        return df

    df_removal     = _save(removal_results,
                           "results/myntra_ablation_removal.csv")
    df_incremental = _save(incremental_results,
                           "results/myntra_ablation_incremental.csv")

    all_rows = removal_results + incremental_results
    if all_rows:
        import pandas as pd
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv("results/myntra_ablation_all.csv", index=False)

    log.info("\n" + "=" * 60)
    log.info("MYNTRA ABLATION STUDY COMPLETE")
    log.info("=" * 60)
    log.info("Output files:")
    log.info("  results/myntra_ablation_removal.csv")
    log.info("  results/myntra_ablation_incremental.csv")
    log.info("  results/myntra_ablation_all.csv")
    log.info("  results/myntra_ablation_<name>_test_results.csv")


if __name__ == "__main__":
    main()