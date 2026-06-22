# ============================================================
# run_all_experiments.py  (v2)
# Train + evaluate ALL models sequentially.
# Uses identical data splits, eval function, and checkpoint
# loading as individual model files.
# Produces: results/all_results_comparison.csv
#
# v2 changes:
#  - set_seed(42) called at startup for reproducibility
#  - calibrate_thresholds applied to HierFashion (main model only)
#  - Baselines/ablations still use fixed thresholds (fair comparison)
# ============================================================

import os, sys, math, warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from core.common import (
    load_hierarchy, build_adjacency_matrix,
    FashionDataset, make_train_val_split,
    compute_pos_weights, evaluate_hierarchical,
    analyze_model, get_logger, set_seed, calibrate_thresholds,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
    BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS
)

logger = get_logger("run_all")
os.makedirs("results", exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def make_trainer(name):
    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1,
                dirpath=f"checkpoints/{name}",
                filename=f"{name}-{{epoch:02d}}-{{val_loss:.4f}}"
            ),
            EarlyStopping(monitor="val_loss", mode="min",
                          patience=PATIENCE, verbose=False),
        ],
        logger=CSVLogger("training_logs", name=name),
        enable_progress_bar=True,
    )


def train_and_eval(ModelClass, model_name, build_kwargs,
                   train_dl, val_dl, test_dl, device,
                   ckpt_extra_kwargs=None, use_calibration=False,
                   hierarchy=None):
    """
    Train model, load best checkpoint, evaluate on test set.
    use_calibration=True: applies calibrate_thresholds on val_dl
                          before test evaluation (main model only).
    """
    logger.info(f"\n{'='*60}\nTRAINING: {model_name}\n{'='*60}")

    model   = ModelClass(**build_kwargs)
    analyze_model(model, device)

    trainer = make_trainer(model_name)
    trainer.fit(model, train_dl, val_dl)

    ckpt_cb   = trainer.checkpoint_callback
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        load_kwargs = build_kwargs.copy()
        if ckpt_extra_kwargs:
            load_kwargs.update(ckpt_extra_kwargs)
        model = ModelClass.load_from_checkpoint(best_ckpt, **load_kwargs)
        model.to(device)
    else:
        logger.warning(f"No checkpoint found for {model_name} — using last weights")

    threshold_config = None
    if use_calibration and hierarchy is not None:
        logger.info(f"[{model_name}] Calibrating thresholds on val set...")
        threshold_config = calibrate_thresholds(
            model, val_dl, device, hierarchy,
            top_k_candidates=(3, 5, 7, 10),
        )

    df = evaluate_hierarchical(
        model, test_dl, device, model.loss_fn,
        save_path=f"results/{model_name}_test_results.csv",
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    row = df[df["Head"] == "OverallModel"]
    if row.empty:
        return {"Model": model_name}
    r = row.iloc[0]
    return {
        "Model":           model_name,
        "Macro_F1":        r.get("Macro_F1"),
        "Micro_F1":        r.get("Micro_F1"),
        "Macro_Precision": r.get("Macro_Precision"),
        "Micro_Precision": r.get("Micro_Precision"),
        "Macro_Recall":    r.get("Macro_Recall"),
        "Micro_Recall":    r.get("Micro_Recall"),
        "mAP":             r.get("mAP"),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(42)   # reproducible data shuffle + weight init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  seed: 42")

    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix = build_adjacency_matrix(hierarchy, hierarchy_json)

    train_subset, val_subset = make_train_val_split(
        annotation_file=TRAIN_ANN, image_dir=TRAIN_IMAGES,
        hierarchy=hierarchy, train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM, val_fraction=0.20, seed=42,
    )
    test_ds = FashionDataset(VAL_IMAGES, VAL_ANN, hierarchy, VAL_TRANSFORM)

    train_dl = DataLoader(train_subset, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_subset,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,      BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    pos_weight = compute_pos_weights(train_dl, hierarchy, device)
    steps_per_epoch = math.ceil(len(train_subset) / BATCH_SIZE)
    all_overview    = []

    # ===========================================================
    # 1. MAIN MODEL — HierFashion v2
    # ===========================================================
    from final import HierFashionLightning, LR_IMPROVED
    _kw_hier_main = dict(
        hierarchy=hierarchy, adj_matrix=adj_matrix,
        pos_weight=pos_weight, learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_per_epoch,
    )
    all_overview.append(train_and_eval(
        HierFashionLightning, "HierFashion_v2",
        build_kwargs=_kw_hier_main,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_hier_main,
        use_calibration=True,   # calibrated thresholds for main model
        hierarchy=hierarchy,
    ))

    # ===========================================================
    # 2. BASELINES  (fixed thresholds — apples-to-apples)
    # ===========================================================

    from models.cnn_flat import CNNFlatLightning
    _kw_flat = dict(hierarchy=hierarchy, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        CNNFlatLightning, "CNNFlat",
        build_kwargs=_kw_flat,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_flat,
    ))

    from models.cnn_hier import CNNHierLightning
    _kw_chier = dict(hierarchy=hierarchy, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        CNNHierLightning, "CNNHier",
        build_kwargs=_kw_chier,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_chier,
    ))

    from models.ml_gcn import MLGCNLightning
    _kw_gcn = dict(hierarchy=hierarchy, adj_matrix=adj_matrix, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        MLGCNLightning, "MLGCN",
        build_kwargs=_kw_gcn,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_gcn,
    ))

    from models.cnn_gat import CNNGATLightning
    _kw_gat = dict(hierarchy=hierarchy, adj_matrix=adj_matrix, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        CNNGATLightning, "CNNGAT",
        build_kwargs=_kw_gat,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_gat,
    ))

    from models.resnet_asl import ResNetASLLightning
    _kw_asl = dict(hierarchy=hierarchy)
    all_overview.append(train_and_eval(
        ResNetASLLightning, "ResNetASL",
        build_kwargs=_kw_asl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_asl,
    ))

    from models.vit_mlc import ViTMLCLightning
    _kw_vit = dict(hierarchy=hierarchy, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        ViTMLCLightning, "ViTMLC",
        build_kwargs=_kw_vit,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_vit,
    ))

    from models.q2l import Q2LLightning
    _kw_q2l = dict(hierarchy=hierarchy, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        Q2LLightning, "Q2L",
        build_kwargs=_kw_q2l,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_q2l,
    ))

    from models.tresnet import TResNetLightning
    _kw_tres = dict(hierarchy=hierarchy, pos_weight=pos_weight)
    all_overview.append(train_and_eval(
        TResNetLightning, "TResNet",
        build_kwargs=_kw_tres,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_tres,
    ))

    # ===========================================================
    # 3. ABLATIONS — Individual Removal
    # ===========================================================

    _kw_abl = dict(hierarchy=hierarchy, adj_matrix=adj_matrix, pos_weight=pos_weight)

    from ablation.ablation_no_label_guided_attn import AblationNoLabelGuidedAttnLightning
    all_overview.append(train_and_eval(
        AblationNoLabelGuidedAttnLightning, "Ablation_NoLabelGuidedAttn",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_gat import AblationNoGATLightning
    all_overview.append(train_and_eval(
        AblationNoGATLightning, "Ablation_NoGAT",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_cross_attn import AblationNoCrossAttnLightning
    all_overview.append(train_and_eval(
        AblationNoCrossAttnLightning, "Ablation_NoCrossAttn",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_hier_loss import AblationNoHierLossLightning
    all_overview.append(train_and_eval(
        AblationNoHierLossLightning, "Ablation_NoHierLoss",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_consistency import AblationNoConsistencyLightning
    all_overview.append(train_and_eval(
        AblationNoConsistencyLightning, "Ablation_NoConsistency",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_path_coherence import AblationNoPathLightning
    all_overview.append(train_and_eval(
        AblationNoPathLightning, "Ablation_NoPath",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.ablation_no_hier_mask import AblationNoHierMaskLightning
    all_overview.append(train_and_eval(
        AblationNoHierMaskLightning, "Ablation_NoHierMask",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    # ===========================================================
    # 4. ABLATIONS — Incremental Build-up
    # ===========================================================

    from ablation.incremental_b2_gat import IncrementalB2Lightning
    all_overview.append(train_and_eval(
        IncrementalB2Lightning, "Incremental_B2_GAT",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.incremental_b3_lgsa import IncrementalB3Lightning
    all_overview.append(train_and_eval(
        IncrementalB3Lightning, "Incremental_B3_LGSA",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    from ablation.incremental_b4_crossattn import IncrementalB4Lightning
    all_overview.append(train_and_eval(
        IncrementalB4Lightning, "Incremental_B4_CrossAttn",
        build_kwargs=_kw_abl,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, device=device,
        ckpt_extra_kwargs=_kw_abl,
    ))

    # ===========================================================
    # 5. MERGED COMPARISON TABLE
    # ===========================================================
    comparison_df = pd.DataFrame(all_overview)
    comparison_df = comparison_df.sort_values("mAP", ascending=False).reset_index(drop=True)
    comparison_df.to_csv("results/all_results_comparison.csv", index=False)

    logger.info("\n" + "="*60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("="*60)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    logger.info("\nAll results saved to: results/")
