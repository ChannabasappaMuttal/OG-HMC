# ============================================================
# BASELINE 2 — CNNHier
# ResNet50 → 4 separate FC heads + Composite Hierarchical Loss
# Hierarchy in LOSS only — no graph, no attention
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common import (
    make_train_val_split,
    ResNet50BackboneFlat, CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, get_logger,
    load_hierarchy, TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, DATASET_ROOT, analyze_model
)

logger = get_logger("cnn_hier")


class CNNHier(nn.Module):

    def __init__(self, hierarchy):
        super().__init__()
        self.backbone = ResNet50BackboneFlat()

        self.category_head    = nn.Linear(2048, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(2048, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(2048, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, hierarchy.num_attributes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "categories":    self.category_head(feat),
            "subcategories": self.subcategory_head(feat),
            "attr_groups":   self.group_head(feat),
            "attributes":    self.attribute_head(feat),
        }


class CNNHierLightning(pl.LightningModule):

    def __init__(self, hierarchy, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "pos_weight"])
        self.model   = CNNHier(hierarchy)
        self.loss_fn = CompositeHierarchicalLoss(
            pos_weight_cat=pos_weight["categories"],
            pos_weight_sub=pos_weight["subcategories"],
            pos_weight_group=pos_weight["attr_groups"],
            pos_weight_attr=pos_weight["attributes"],
        )
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, t   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, t)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, t   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, t)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        return {"optimizer": opt, "lr_scheduler": sched}


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hierarchy, hj = load_hierarchy()

    # ── 80/20 internal train/val split from train2020 ──────
    train_subset, val_subset = make_train_val_split(
        annotation_file=TRAIN_ANN, image_dir=TRAIN_IMAGES,
        hierarchy=hierarchy, train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM, val_fraction=0.20, seed=42,
    )
    # ── Test set = full val2020 — never seen during training ─
    test_ds  = FashionDataset(VAL_IMAGES, VAL_ANN, hierarchy, VAL_TRANSFORM)

    train_dl = DataLoader(train_subset, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_subset,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,      BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    pos_weight = compute_pos_weights(train_dl, hierarchy, device)
    model = CNNHierLightning(hierarchy, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/cnn_hier",
        filename="cnn_hier-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb,
                   EarlyStopping(monitor="val_loss", mode="min",
                                 patience=PATIENCE)],
        logger=CSVLogger("training_logs", name="CNNHier"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = CNNHierLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/cnn_hier_test_results.csv")
