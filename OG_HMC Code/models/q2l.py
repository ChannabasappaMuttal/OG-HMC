# ============================================================
# BASELINE 7 — Q2L (Query2Label)
# Liu et al. "Query2Label: A Simple Transformer Way to
#             Multi-Label Classification", NeurIPS 2021
# ResNet50 backbone + Transformer decoder with label queries
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
    ResNet50Backbone, CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, get_logger,
    load_hierarchy, TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("q2l")


class Q2L(nn.Module):
    """
    Label-query based multi-label classifier.
    Each label has a learnable query embedding.
    Transformer decoder cross-attends over spatial feature map.
    """

    def __init__(self, hierarchy, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.hierarchy = hierarchy
        total_labels   = (hierarchy.num_categories + hierarchy.num_subcategories +
                          hierarchy.num_attr_groups + hierarchy.num_attributes)

        # Visual backbone
        self.backbone   = ResNet50Backbone()
        self.vis_proj   = nn.Conv2d(2048, d_model, 1)       # spatial proj

        # Label query embeddings (one per label across all 4 levels)
        self.query_embed = nn.Embedding(total_labels, d_model)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Separate classifiers per head (binary per label)
        self.cat_head  = nn.Linear(d_model, 1)
        self.sub_head  = nn.Linear(d_model, 1)
        self.grp_head  = nn.Linear(d_model, 1)
        self.attr_head = nn.Linear(d_model, 1)

        self._cat_end  = hierarchy.num_categories
        self._sub_end  = self._cat_end + hierarchy.num_subcategories
        self._grp_end  = self._sub_end + hierarchy.num_attr_groups
        # attr goes to total_labels

    def forward(self, x):
        feature_maps, _ = self.backbone(x)                     # [B, 2048, H, W]
        B, _, H, W      = feature_maps.shape

        # Spatial features as memory
        mem = self.vis_proj(feature_maps)                       # [B, d_model, H, W]
        mem = mem.flatten(2).permute(0, 2, 1)                   # [B, H*W, d_model]

        # Label queries
        Q   = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, L, d_model]

        # Decode
        dec = self.decoder(Q, mem)                              # [B, L, d_model]

        cats  = self.cat_head(dec[:, :self._cat_end, :]).squeeze(-1)
        subs  = self.sub_head(dec[:, self._cat_end:self._sub_end, :]).squeeze(-1)
        grps  = self.grp_head(dec[:, self._sub_end:self._grp_end, :]).squeeze(-1)
        attrs = self.attr_head(dec[:, self._grp_end:, :]).squeeze(-1)

        return {
            "categories":    cats,
            "subcategories": subs,
            "attr_groups":   grps,
            "attributes":    attrs,
        }


class Q2LLightning(pl.LightningModule):

    def __init__(self, hierarchy, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "pos_weight"])
        self.model   = Q2L(hierarchy)
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
    model = Q2LLightning(hierarchy, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/q2l",
        filename="q2l-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="Q2L"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = Q2LLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/q2l_test_results.csv")
