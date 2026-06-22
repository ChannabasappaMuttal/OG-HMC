# ============================================================
# ABLATION A2 — HierFashion without Cross-Attention Fusion
# Replace cross-attention with simple global graph average pooling + concat
# Tests: "How much does the cross-attention fusion contribute?"
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
    ResNet50Backbone, CompositeHierarchicalLoss, FashionDataset,
    evaluate_hierarchical, compute_pos_weights, build_adjacency_matrix,
     get_logger, load_hierarchy,
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_ANN, VAL_ANN,
    TRAIN_TRANSFORM, VAL_TRANSFORM, BATCH_SIZE, LR, EPOCHS, PATIENCE,
    NUM_WORKERS, analyze_model
)

logger = get_logger("ablation_no_cross_attn")


class Phase2LabelGraphModule(nn.Module):
    def __init__(self, num_nodes, adj_matrix, embed_dim=300, out_dim=512):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        # N2: L2-normalise rows at init to prevent exploding GAT inputs
        nn.init.xavier_uniform_(self.node_embeddings.data.unsqueeze(0))
        self.node_embeddings.data = self.node_embeddings.data.squeeze(0)
        with torch.no_grad():
            norms = self.node_embeddings.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.node_embeddings.data.div_(norms)
        edge_index, _ = dense_to_sparse(adj_matrix)
        self.register_buffer("edge_index", edge_index)
        self.gat1 = GATConv(embed_dim, 64, heads=8, dropout=0.1)
        self.gat2 = GATConv(512, out_dim, heads=1, dropout=0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self):
        # N1: force fp32 for GAT (AMP fp16 + sparse softmax = NaN)
        with torch.amp.autocast("cuda", enabled=False):
            x  = self.node_embeddings.float()
            ei = self.edge_index
            x  = self.norm1(F.elu(self.gat1(x, ei)))
            x  = self.norm2(self.gat2(x, ei))
        return x


class LabelGuidedSpatialAttention(nn.Module):
    def __init__(self, visual_dim=2048, label_dim=512, num_labels=363):
        super().__init__()
        self.num_labels = num_labels
        self.label_proj = nn.Linear(label_dim, visual_dim)
        self.norm       = nn.LayerNorm(visual_dim)
        nn.init.xavier_uniform_(self.label_proj.weight)

    def forward(self, feature_maps, label_embeddings):
        B, C, H, W = feature_maps.shape
        proj   = self.label_proj(label_embeddings)
        scores = torch.einsum("bchw,nc->bnhw", feature_maps, proj)
        w      = F.softmax(scores.view(B, self.num_labels, -1), dim=-1).view(B, self.num_labels, H, W)
        att    = torch.einsum("bnhw,bchw->bnc", w, feature_maps).mean(dim=1)
        return self.norm(att)


class HierarchicalPredictionHeads(nn.Module):
    def __init__(self, hierarchy):
        super().__init__()
        # Input is 2048 + 512 = 2560 without cross-attn projection
        self.shared = nn.Sequential(
            nn.Linear(2048 + 512, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
        )
        self.category_head    = nn.Linear(512, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(512, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(512, hierarchy.num_attr_groups)
        self.attribute_head   = nn.Sequential(
            nn.Linear(512, 768), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(512, hierarchy.num_attributes),
        )

    def forward(self, x):
        x = self.shared(x)
        return {"categories": self.category_head(x), "subcategories": self.subcategory_head(x),
                "attr_groups": self.group_head(x), "attributes": self.attribute_head(x)}


class AblationNoCrossAttnLightning(pl.LightningModule):

    def __init__(self, hierarchy, adj_matrix, pos_weight, learning_rate=LR):
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "adj_matrix", "pos_weight"])
        self.lr = learning_rate

        self.backbone          = ResNet50Backbone()
        self.graph_module      = Phase2LabelGraphModule(hierarchy.num_nodes, adj_matrix)
        self.label_guided_attn = LabelGuidedSpatialAttention(2048, 512, hierarchy.num_nodes)
        # ❌ NO cross-attention — replaced with global mean pooling of graph nodes
        self.heads             = HierarchicalPredictionHeads(hierarchy)
        self.loss_fn           = CompositeHierarchicalLoss(
            pos_weight_cat=pos_weight["categories"],
            pos_weight_sub=pos_weight["subcategories"],
            pos_weight_group=pos_weight["attr_groups"],
            pos_weight_attr=pos_weight["attributes"],
        )
        # N5: corrected group→attr mask (old cat→attr was all-ones → recall collapse)
        num_groups = hierarchy.num_attr_groups
        num_attrs  = hierarchy.num_attributes
        grp_attr_mask = torch.zeros(num_groups, num_attrs)
        for group_name, attr_ids in hierarchy.level_3.items():
            g_idx = hierarchy.group_id_to_idx.get(group_name)
            if g_idx is None: continue
            for attr_id in attr_ids:
                a_idx = hierarchy.attribute_id_to_idx.get(str(attr_id))
                if a_idx is not None: grp_attr_mask[g_idx, a_idx] = 1.0
        self.register_buffer("grp_attr_mask", grp_attr_mask)

    def forward(self, images, targets=None):
        fmaps, _        = self.backbone(images)
        graph_nodes     = self.graph_module()
        attended_visual = self.label_guided_attn(fmaps, graph_nodes)

        # ❌ Simple concat instead of cross-attention
        graph_global    = graph_nodes.mean(0).unsqueeze(0).expand(images.size(0), -1)
        fused           = torch.cat([attended_visual, graph_global], dim=1)   # [B, 2560]

        predictions     = self.heads(fused)

        # N6: gate attrs by GROUP predictions (not category)
        grp_probs = torch.sigmoid(predictions["attr_groups"])
        allowed   = torch.matmul((grp_probs > 0.3).float(), self.grp_attr_mask).clamp(0, 1)
        has_group = (allowed.sum(dim=1, keepdim=True) > 0).float()
        gate      = has_group * allowed + (1 - has_group)
        predictions["attributes"] = predictions["attributes"] * gate

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
    model = AblationNoCrossAttnLightning(hierarchy, adj, pos_weight)
    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath="checkpoints/ablation_no_cross_attn",
        filename="ablation_no_cross_attn-{epoch:02d}-{val_loss:.4f}"
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
        logger=CSVLogger("training_logs", name="Ablation_NoCrossAttn"),
    )
    trainer.fit(model, train_dl, val_dl)

    # ── Load best checkpoint before test evaluation ─────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        model = AblationNoCrossAttnLightning.load_from_checkpoint(best_ckpt, hierarchy=hierarchy, adj_matrix=adj, pos_weight=pos_weight)
        model.to(device)

    os.makedirs("results", exist_ok=True)
    evaluate_hierarchical(model, test_dl, device, model.loss_fn,
                          save_path="results/ablation_no_cross_attn_test_results.csv")
