# ============================================================
# BASELINE 3 — ML-GCN
# ResNet50 + GCN on label co-occurrence graph → attribute prediction
# Based on: Chen et al. "Multi-Label Image Recognition with
#           Graph Convolutional Networks", CVPR 2019
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
    ResNet50BackboneFlat, CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, build_adjacency_matrix,
    get_logger, load_hierarchy,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("ml_gcn")


class GraphConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        # x: [N, in_dim]  adj: [N, N] (row-normalised)
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return self.norm(F.leaky_relu(x, 0.2))


class MLGCN(nn.Module):

    def __init__(self, hierarchy, adj_matrix):
        super().__init__()
        self.backbone = ResNet50BackboneFlat()
        N             = hierarchy.num_nodes

        # Normalise adjacency
        D_inv = torch.diag(1.0 / (adj_matrix.sum(1) + 1e-8))
        adj_norm = torch.matmul(D_inv, adj_matrix)
        self.register_buffer("adj_norm", adj_norm)

        # GCN on node embeddings
        self.node_emb = nn.Parameter(torch.randn(N, 300))
        self.gc1      = GraphConvLayer(300, 512)
        self.gc2      = GraphConvLayer(512, 2048)

        # Per-node classifier: inner product with visual features
        self.category_head    = nn.Linear(2048, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(2048, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(2048, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Linear(2048, hierarchy.num_attributes)

    def forward(self, x):
        visual   = self.backbone(x)                    # [B, 2048]
        node_x   = self.gc1(self.node_emb, self.adj_norm)
        node_x   = self.gc2(node_x, self.adj_norm)    # [N, 2048]

        # Fuse by adding mean graph repr to visual
        graph_global = node_x.mean(0)                 # [2048]
        fused        = visual + graph_global.unsqueeze(0)  # [B, 2048]

        return {
            "categories":    self.category_head(fused),
            "subcategories": self.subcategory_head(fused),
            "attr_groups":   self.group_head(fused),
            "attributes":    self.attribute_head(fused),
        }


class MLGCNLightning(pl.LightningModule):

    def __init__(self, hierarchy, adj_matrix, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "adj_matrix", "pos_weight"])
        self.model   = MLGCN(hierarchy, adj_matrix)
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
    model = MLGCNLightning(hierarchy, adj, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/ml_gcn",
        filename="ml_gcn-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="MLGCN"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = MLGCNLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/ml_gcn_test_results.csv")
