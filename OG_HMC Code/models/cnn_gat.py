# ============================================================
# BASELINE 4 — CNN-GAT
# ResNet50 + GAT label graph → concatenate → FC heads
# Has graph but NO label-guided spatial attention,
# NO cross-attention fusion
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
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common import (
    make_train_val_split,
    ResNet50BackboneFlat, CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, build_adjacency_matrix,
    get_logger, load_hierarchy,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("cnn_gat")


class CNNGAT(nn.Module):

    def __init__(self, hierarchy, adj_matrix):
        super().__init__()
        self.backbone = ResNet50BackboneFlat()
        N             = hierarchy.num_nodes

        # Label embeddings + GAT
        self.node_emb = nn.Parameter(torch.randn(N, 300))
        edge_index, _ = dense_to_sparse(adj_matrix)
        self.register_buffer("edge_index", edge_index)
        self.gat1     = GATConv(300, 64, heads=8, dropout=0.1)
        self.gat2     = GATConv(512, 512, heads=1, dropout=0.1)

        # Fusion: visual(2048) + graph_global(512) → 2560 → heads
        self.fusion   = nn.Sequential(
            nn.Linear(2048 + 512, 1024), nn.ReLU(), nn.Dropout(0.3)
        )
        self.category_head    = nn.Linear(1024, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(1024, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(1024, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, hierarchy.num_attributes),
        )

    def forward(self, x):
        visual     = self.backbone(x)                              # [B, 2048]

        node_x     = F.elu(self.gat1(self.node_emb, self.edge_index))
        node_x     = self.gat2(node_x, self.edge_index)           # [N, 512]
        graph_glob = node_x.mean(0).unsqueeze(0).expand(visual.size(0), -1)  # [B, 512]

        fused      = self.fusion(torch.cat([visual, graph_glob], dim=1))  # [B, 1024]

        return {
            "categories":    self.category_head(fused),
            "subcategories": self.subcategory_head(fused),
            "attr_groups":   self.group_head(fused),
            "attributes":    self.attribute_head(fused),
        }


class CNNGATLightning(pl.LightningModule):

    def __init__(self, hierarchy, adj_matrix, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "adj_matrix", "pos_weight"])
        self.model   = CNNGAT(hierarchy, adj_matrix)
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
    adj = build_adjacency_matrix(hierarchy, hj)

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
    model = CNNGATLightning(hierarchy, adj, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/cnn_gat",
        filename="cnn_gat-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="CNNGAT"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = CNNGATLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/cnn_gat_test_results.csv")
