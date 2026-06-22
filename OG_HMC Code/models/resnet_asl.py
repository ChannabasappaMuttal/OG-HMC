# ============================================================
# BASELINE 5 — ResNet-ASL
# ResNet50 + Asymmetric Loss (ASL)
# Tian et al. "Asymmetric Loss for Multi-Label Classification"
# No graph, no attention — only improved loss function
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common import (
    make_train_val_split,
    ResNet50BackboneFlat, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, get_logger,
    load_hierarchy, TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("resnet_asl")


class AsymmetricLoss(nn.Module):
    """
    ASL: shifts negative probabilities and applies different gamma for
    positive/negative examples to combat label imbalance.
    """

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits, targets):
        logits  = logits.float()   # fp32 cast — prevents log(0) NaN under AMP fp16
        targets = targets.float()
        p    = torch.sigmoid(logits)
        p_m  = torch.clamp(p - self.clip, min=0)  # shifted negative probs

        loss_pos = targets       * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - p_m).clamp(min=self.eps))

        loss     = loss_pos + loss_neg

        p_t = p * targets + p_m * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        asl_w = (1 - p_t) ** gamma

        result = (-asl_w * loss).mean()
        # Safety net — return detached zero if still NaN (should not happen)
        if not torch.isfinite(result):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return result


class ASLHierarchicalLoss(nn.Module):
    """ASL applied per hierarchy head + consistency terms."""

    def __init__(self, lambda_consistency=0.1, lambda_path=0.05):
        super().__init__()
        self.asl    = AsymmetricLoss()
        self.lc     = lambda_consistency
        self.lp     = lambda_path

    def forward(self, predictions, targets, return_components=False):
        L_cat   = self.asl(predictions["categories"],    targets["categories"].float())
        L_sub   = self.asl(predictions["subcategories"], targets["subcategories"].float())
        L_group = self.asl(predictions["attr_groups"],   targets["attr_groups"].float())
        L_attr  = self.asl(predictions["attributes"],    targets["attributes"].float())
        L_bce   = (L_cat + L_sub + L_group + L_attr) / 4

        cp = torch.sigmoid(predictions["categories"])
        sp = torch.sigmoid(predictions["subcategories"])
        gp = torch.sigmoid(predictions["attr_groups"])
        ap = torch.sigmoid(predictions["attributes"])

        L_cs   = (torch.mean((sp.mean(1) - cp.mean(1)) ** 2) +
                  torch.mean((gp.mean(1) - sp.mean(1)) ** 2) +
                  torch.mean((ap.mean(1) - gp.mean(1)) ** 2)) / 3
        L_path = torch.mean((ap.mean(1) - cp.mean(1)) ** 2)

        CHL    = L_bce + self.lc * L_cs + self.lp * L_path
        if return_components:
            return CHL, L_bce, L_cs, L_path
        return CHL


class ResNetASL(nn.Module):

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


class ResNetASLLightning(pl.LightningModule):

    def __init__(self, hierarchy, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy"])
        self.model   = ResNetASL(hierarchy)
        self.loss_fn = ASLHierarchicalLoss()
        self.lr      = learning_rate

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
    model = ResNetASLLightning(hierarchy)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/resnet_asl",
        filename="resnet_asl-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="ResNetASL"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = ResNetASLLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/resnet_asl_test_results.csv")
