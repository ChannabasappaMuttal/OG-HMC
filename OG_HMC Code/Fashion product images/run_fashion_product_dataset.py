# ============================================================
# run_baselines_unified.py
#
# Trains ALL 7 base models on the Fashion Product Images dataset
# with EXACTLY the same pipeline as final.py (main model):
#
#   ✅ seed=42 globally via set_seed() before anything runs
#   ✅ FashionDataModule (Lightning DataModule) for data loading
#   ✅ val_fraction=0.20, seed=42 split — identical to main model
#   ✅ TRAIN_TRANSFORM / VAL_TRANSFORM from core/common.py
#   ✅ LearningRateMonitor callback added to every trainer
#   ✅ calibrate_thresholds() run on val set before test eval
#   ✅ evaluate_hierarchical() called with threshold_config
#   ✅ Results saved to results/baselines_comparison_seed42.csv
#
# USAGE:
#   python run_baselines_unified.py
#
#   Optional env vars:
#     HIERFASHION_SEED=42        (override seed, default 42)
#     HIERFASHION_MODELS=all     (comma-sep subset, e.g. cnn_flat,vit_mlc)
#
# Models trained (in order):
#   1. CNNFlat    — ResNet50 + flat BCE heads
#   2. CNNHier    — ResNet50 + hierarchical loss
#   3. MLGCN      — ResNet50 + GCN graph
#   4. CNNGAT     — ResNet50 + GAT graph
#   5. ResNetASL  — ResNet50 + Asymmetric Loss
#   6. ViTMLC     — ViT-B/16 + hierarchical heads
#   7. Q2L        — ResNet50 + Query2Label transformer
#   8. TResNet    — ResNet50 + multi-scale pooling
# ============================================================

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# ── Resolve paths ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core.common import (
    FashionDataset, make_train_val_split, compute_pos_weights,
    evaluate_hierarchical, calibrate_thresholds, set_seed, get_logger,
    load_hierarchy, build_adjacency_matrix, analyze_model,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
    BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS,
)

# ── Model imports ─────────────────────────────────────────────
from models.cnn_flat   import CNNFlatLightning
from models.cnn_hier   import CNNHierLightning
from models.ml_gcn     import MLGCNLightning
from models.cnn_gat    import CNNGATLightning
from models.resnet_asl import ResNetASLLightning
from models.vit_mlc    import ViTMLCLightning
from models.q2l        import Q2LLightning
from models.tresnet    import TResNetLightning

logger = get_logger("run_baselines_unified")

# ============================================================
# CONFIG
# ============================================================
SEED = int(os.environ.get("HIERFASHION_SEED", 42))
_MODELS_ENV = os.environ.get("HIERFASHION_MODELS", "all")
RUN_MODELS = (
    None  # None means run all
    if _MODELS_ENV.strip().lower() == "all"
    else set(m.strip().lower() for m in _MODELS_ENV.split(","))
)


def _should_run(name: str) -> bool:
    return RUN_MODELS is None or name.lower() in RUN_MODELS


# ============================================================
# SHARED DATAMODULE  (identical to final.py's FashionDataModule)
# ============================================================
class FashionDataModule(pl.LightningDataModule):
    """
    Same DataModule class used in final.py.
    Ensures identical data ordering, split, and transforms
    across all baselines and the main model.
    """

    def __init__(self, hierarchy, batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS, seed=42):
        super().__init__()
        self.hierarchy   = hierarchy
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed

    def setup(self, stage=None):
        self.train_subset, self.val_subset = make_train_val_split(
            annotation_file=TRAIN_ANN,
            image_dir=TRAIN_IMAGES,
            hierarchy=self.hierarchy,
            train_transform=TRAIN_TRANSFORM,
            val_transform=VAL_TRANSFORM,
            val_fraction=0.20,
            seed=self.seed,
        )
        self.test_ds = FashionDataset(
            VAL_IMAGES, VAL_ANN, self.hierarchy, VAL_TRANSFORM
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_subset, self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_subset, self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


# ============================================================
# TRAINER FACTORY  (identical flags to final.py)
# ============================================================
def _make_trainer(model_name: str, seed: int):
    use_gpu = torch.cuda.is_available()
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath=f"checkpoints/{model_name}_seed{seed}",
        filename=f"{model_name}-seed{seed}-{{epoch:02d}}-{{val_loss:.4f}}",
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            ckpt_cb,
            EarlyStopping(monitor="val_loss", mode="min",
                          patience=PATIENCE, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),  # matches final.py
        ],
        logger=CSVLogger("training_logs", name=f"{model_name}_seed{seed}"),
    )
    return trainer, ckpt_cb


# ============================================================
# UNIFIED TRAIN + CALIBRATE + EVAL
# ============================================================
def train_and_eval(
    ModelClass,
    model_name: str,
    build_kwargs: dict,
    data_module: FashionDataModule,
    device: str,
    seed: int,
):
    """
    Full pipeline mirroring final.py exactly:
      1.  Build model
      2.  trainer.fit(model, datamodule=data_module)
      3.  Load best checkpoint
      4.  calibrate_thresholds() on val set
      5.  evaluate_hierarchical() on test set with threshold_config
    """
    logger.info(f"\n{'='*60}\nTRAINING: {model_name}  (seed={seed})\n{'='*60}")

    model = ModelClass(**build_kwargs)
    analyze_model(model, device)

    trainer, ckpt_cb = _make_trainer(model_name, seed)

    # trainer.fit with datamodule — identical to final.py
    trainer.fit(model, datamodule=data_module)

    best_ckpt = ckpt_cb.best_model_path
    logger.info(f"[{model_name}] Best checkpoint: {best_ckpt}")

    if best_ckpt:
        model = ModelClass.load_from_checkpoint(best_ckpt, **build_kwargs)
    model.to(device)
    model.eval()

    # ── Calibrate on val set (same as final.py) ───────────────
    logger.info(f"[{model_name}] Calibrating thresholds on val set...")
    threshold_config = calibrate_thresholds(
        model,
        data_module.val_dataloader(),
        device,
        data_module.hierarchy,
        top_k_candidates=(3, 5, 7, 10, 15),
    )

    # ── Evaluate on held-out test set ─────────────────────────
    logger.info(f"[{model_name}] Running TEST evaluation on val2020...")
    os.makedirs("results", exist_ok=True)
    results_df = evaluate_hierarchical(
        model,
        data_module.test_dataloader(),
        device,
        model.loss_fn,
        save_path=f"results/{model_name}_seed{seed}_test_results.csv",
        threshold_config=threshold_config,
        hierarchy=data_module.hierarchy,
    )

    # ── Extract summary row ────────────────────────────────────
    row = results_df[results_df["Head"] == "OverallModel"]
    if row.empty:
        logger.warning(f"[{model_name}] No OverallModel row found in results.")
        return {"Model": model_name, "Seed": seed}

    r = row.iloc[0]
    summary = {
        "Model":           model_name,
        "Seed":            seed,
        "Macro_F1":        r.get("Macro_F1"),
        "Micro_F1":        r.get("Micro_F1"),
        "mAP":             r.get("mAP"),
        "Macro_Precision": r.get("Macro_Precision"),
        "Micro_Precision": r.get("Micro_Precision"),
        "Macro_Recall":    r.get("Macro_Recall"),
        "Micro_Recall":    r.get("Micro_Recall"),
    }
    logger.info(
        f"[{model_name}] Macro-F1={summary['Macro_F1']:.4f}  "
        f"Micro-F1={summary['Micro_F1']:.4f}  mAP={summary['mAP']:.4f}"
    )
    return summary


# ============================================================
# MAIN
# ============================================================
def main():

    # ── Step 0: Global seed — must be first, before any ops ───
    set_seed(SEED)
    logger.info(f"Global seed set to {SEED}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Step 1: Hierarchy + adjacency matrix ──────────────────
    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix = build_adjacency_matrix(hierarchy, hierarchy_json)

    # ── Step 2: Shared DataModule (same class as final.py) ────
    data_module = FashionDataModule(
        hierarchy=hierarchy,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    data_module.setup()

    logger.info(
        f"Dataset split — "
        f"train: {len(data_module.train_subset)}  "
        f"val: {len(data_module.val_subset)}  "
        f"test: {len(data_module.test_ds)}"
    )

    # ── Step 3: Positive weights from train set ───────────────
    pos_weight = compute_pos_weights(
        data_module.train_dataloader(), hierarchy, device
    )

    # ── Step 4: Collect results ───────────────────────────────
    all_results = []

    # ----------------------------------------------------------
    # 1. CNNFlat — ResNet50 + flat independent BCE heads
    # ----------------------------------------------------------
    if _should_run("cnn_flat"):
        kw = dict(hierarchy=hierarchy, pos_weight=pos_weight)
        all_results.append(
            train_and_eval(CNNFlatLightning, "CNNFlat", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 2. CNNHier — ResNet50 + hierarchical composite loss
    # ----------------------------------------------------------
    if _should_run("cnn_hier"):
        kw = dict(hierarchy=hierarchy, pos_weight=pos_weight)
        all_results.append(
            train_and_eval(CNNHierLightning, "CNNHier", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 3. ML-GCN — ResNet50 + GCN on label co-occurrence graph
    # ----------------------------------------------------------
    if _should_run("mlgcn") or _should_run("ml_gcn"):
        kw = dict(hierarchy=hierarchy, adj_matrix=adj_matrix,
                  pos_weight=pos_weight)
        all_results.append(
            train_and_eval(MLGCNLightning, "MLGCN", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 4. CNN-GAT — ResNet50 + GAT label graph, no spatial attn
    # ----------------------------------------------------------
    if _should_run("cnn_gat"):
        kw = dict(hierarchy=hierarchy, adj_matrix=adj_matrix,
                  pos_weight=pos_weight)
        all_results.append(
            train_and_eval(CNNGATLightning, "CNNGAT", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 5. ResNet-ASL — ResNet50 + Asymmetric Loss
    #    NOTE: ResNetASLLightning uses its own ASL loss and does
    #    NOT accept pos_weight — excluded intentionally.
    # ----------------------------------------------------------
    if _should_run("resnet_asl"):
        kw = dict(hierarchy=hierarchy)
        all_results.append(
            train_and_eval(ResNetASLLightning, "ResNetASL", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 6. ViT-MLC — ViT-B/16 + 4 hierarchical FC heads
    # ----------------------------------------------------------
    if _should_run("vit_mlc"):
        kw = dict(hierarchy=hierarchy, pos_weight=pos_weight)
        all_results.append(
            train_and_eval(ViTMLCLightning, "ViTMLC", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 7. Query2Label — ResNet50 + transformer decoder label queries
    # ----------------------------------------------------------
    if _should_run("q2l"):
        kw = dict(hierarchy=hierarchy, pos_weight=pos_weight)
        all_results.append(
            train_and_eval(Q2LLightning, "Q2L", kw,
                           data_module, device, SEED)
        )

    # ----------------------------------------------------------
    # 8. TResNet — ResNet50 + multi-scale pooling
    # ----------------------------------------------------------
    if _should_run("tresnet"):
        kw = dict(hierarchy=hierarchy, pos_weight=pos_weight)
        all_results.append(
            train_and_eval(TResNetLightning, "TResNet", kw,
                           data_module, device, SEED)
        )

    # ── Step 5: Print and save comparison table ───────────────
    if not all_results:
        logger.warning("No models were run. "
                       "Check the HIERFASHION_MODELS env var.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values("mAP", ascending=False).reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/baselines_comparison_seed{SEED}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"  ALL BASELINES COMPLETE  (seed={SEED})")
    print("=" * 70)
    print(df[["Model", "Macro_F1", "Micro_F1", "mAP",
              "Macro_Precision", "Macro_Recall"]].to_string(index=False))
    print(f"\n  Comparison table → {out_path}")
    print(f"  Per-model detail  → results/<model>_seed{SEED}_test_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()