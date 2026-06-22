# ============================================================
# BASELINE 6 — ViT-MLC
# ViT-B/16 → CLS token → 4 Hierarchical FC Heads
# Transformer backbone, no graph
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common import (
    make_train_val_split,
    CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, get_logger,
    load_hierarchy, TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("vit_mlc")


class ViTMLC(nn.Module):

    def __init__(self, hierarchy):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()                 # output: [B, 768]
        self.backbone = vit

        self.category_head    = nn.Linear(768, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(768, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(768, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, hierarchy.num_attributes),
        )

    def forward(self, x):
        feat = self.backbone(x)                   # [B, 768]
        return {
            "categories":    self.category_head(feat),
            "subcategories": self.subcategory_head(feat),
            "attr_groups":   self.group_head(feat),
            "attributes":    self.attribute_head(feat),
        }


class ViTMLCLightning(pl.LightningModule):

    def __init__(self, hierarchy, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "pos_weight"])
        self.model   = ViTMLC(hierarchy)
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
        x, t  = batch
        loss  = self.loss_fn(self(x), t)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, t  = batch
        loss  = self.loss_fn(self(x), t)
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
    model = ViTMLCLightning(hierarchy, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/vit_mlc",
        filename="vit_mlc-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="ViTMLC"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = ViTMLCLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/vit_mlc_test_results.csv")
