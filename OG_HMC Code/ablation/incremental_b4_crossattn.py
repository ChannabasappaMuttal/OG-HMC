# ============================================================
# ABLATION B4 — Incremental Step 4
# ResNet50 + GAT + Label-Guided Spatial Attention
#             + Cross-Attention Fusion + 4 heads + plain BCE
# Adds: Cross-Attention Fusion over B3
# Missing only: Hierarchical Loss (B5 = final.py adds this)
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
    ResNet50Backbone, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, build_adjacency_matrix,
    get_logger, load_hierarchy,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("incremental_b4")


class PlainMultiHeadBCE(nn.Module):
    def __init__(self, pos_weight_cat=None, pos_weight_sub=None,
                 pos_weight_group=None, pos_weight_attr=None):
        super().__init__()
        self.bce_cat   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cat)
        self.bce_sub   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_sub)
        self.bce_group = nn.BCEWithLogitsLoss(pos_weight=pos_weight_group)
        self.bce_attr  = nn.BCEWithLogitsLoss(pos_weight=pos_weight_attr)

    def forward(self, predictions, targets, return_components=False):
        L = (self.bce_cat(predictions["categories"],    targets["categories"].float()) +
             self.bce_sub(predictions["subcategories"], targets["subcategories"].float()) +
             self.bce_group(predictions["attr_groups"], targets["attr_groups"].float()) +
             self.bce_attr(predictions["attributes"],   targets["attributes"].float())) / 4
        zero = torch.tensor(0.0, device=L.device)
        if return_components:
            return L, L, zero, zero
        return L


class IncrementalB4Lightning(pl.LightningModule):
    """
    Build-up step 4:
    ✅ ResNet50 backbone (with feature maps)
    ✅ GAT label graph (2-layer)
    ✅ Label-Guided Spatial Attention
    ✅ Cross-Attention Fusion
    ✅ 4 prediction heads
    ❌ No hierarchical loss  (B5 = final.py adds this)
    ❌ No hierarchical masking
    """

    def __init__(self, hierarchy, adj_matrix, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "adj_matrix", "pos_weight"])
        self.lr = learning_rate
        N       = hierarchy.num_nodes

        # Visual backbone
        self.backbone = ResNet50Backbone()

        # GAT label graph
        self.node_embeddings = nn.Parameter(torch.randn(N, 300))
        edge_index, _ = dense_to_sparse(adj_matrix)
        self.register_buffer("edge_index", edge_index)
        self.gat1  = GATConv(300, 64, heads=8, dropout=0.1)
        self.gat2  = GATConv(512, 512, heads=1, dropout=0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        # Label-Guided Spatial Attention
        self.label_proj = nn.Linear(512, 2048)
        self.attn_norm  = nn.LayerNorm(2048)
        nn.init.xavier_uniform_(self.label_proj.weight)

        # Cross-Attention Fusion
        self.visual_proj_ca = nn.Linear(2048, 512)
        self.graph_proj_ca  = nn.Linear(512, 512)
        self.cross_attn     = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        self.cross_out      = nn.Linear(512, 512)
        self.cross_norm     = nn.LayerNorm(512)

        # Visual to 512
        self.visual_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2))

        # Shared trunk: [512 + 512] → heads
        self.shared = nn.Sequential(
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
        )

        # 4 heads
        self.category_head    = nn.Linear(512, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(512, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(512, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Sequential(
            nn.Linear(512, 768), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(512, hierarchy.num_attributes),
        )

        self.loss_fn = PlainMultiHeadBCE(
            pos_weight_cat=pos_weight["categories"],
            pos_weight_sub=pos_weight["subcategories"],
            pos_weight_group=pos_weight["attr_groups"],
            pos_weight_attr=pos_weight["attributes"],
        )

    def _lgsa(self, feature_maps, label_emb, N):
        B, C, H, W = feature_maps.shape
        proj    = self.label_proj(label_emb)
        scores  = torch.einsum("bchw,nc->bnhw", feature_maps, proj)
        w       = F.softmax(scores.view(B, N, -1), dim=-1).view(B, N, H, W)
        attended = torch.einsum("bnhw,bchw->bnc", w, feature_maps).mean(dim=1)
        return self.attn_norm(attended)

    def _cross_attn(self, visual, graph_nodes):
        B = visual.size(0)
        q = self.visual_proj_ca(visual).unsqueeze(1)
        k = self.graph_proj_ca(graph_nodes).unsqueeze(0).expand(B, -1, -1)
        o, _ = self.cross_attn(q, k, k)
        return self.cross_norm(self.cross_out(o.squeeze(1)))

    def forward(self, images, targets=None):
        fmaps, _    = self.backbone(images)

        x           = self.node_embeddings
        x           = self.norm1(F.elu(self.gat1(x, self.edge_index)))
        graph_nodes = self.norm2(self.gat2(x, self.edge_index))

        attended    = self._lgsa(fmaps, graph_nodes, graph_nodes.size(0))
        visual_512  = self.visual_proj(attended)
        cross_feat  = self._cross_attn(attended, graph_nodes)

        fused       = self.shared(torch.cat([visual_512, cross_feat], dim=1))

        predictions = {
            "categories":    self.category_head(fused),
            "subcategories": self.subcategory_head(fused),
            "attr_groups":   self.group_head(fused),
            "attributes":    self.attribute_head(fused),
        }
        if targets is not None:
            return predictions, self.loss_fn(predictions, targets)
        return predictions

    def training_step(self, batch, _):
        x, t = batch; _, loss = self(x, t)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False); return loss

    def validation_step(self, batch, _):
        x, t = batch; _, loss = self(x, t)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False); return loss

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
    model = IncrementalB4Lightning(hierarchy, adj, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/incremental_b4_crossattn",
        filename="incremental_b4_crossattn-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="Incremental_B4_CrossAttn"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = IncrementalB4Lightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/incremental_b4_crossattn_test_results.csv")
