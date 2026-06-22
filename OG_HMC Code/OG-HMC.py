# ============================================================
# HIERFASHION — final.py  (v3 — targeting attr Macro-F1 0.25+)
#
# NaN fixes (N1–N4, carried from v1):
#  N1: GAT in fp32 always
#  N2: Node embeddings L2-normalised at init
#  N3: LGSA clamped + fp32
#  N4: Cross-attention fp32
#
# Saturation fixes (S1–S5, carried from v2):
#  S1–S2: attr_weight=1.5, asl_gamma_neg=2 (reverted from 2.0/3)
#  S3: asl_clip=0.05
#  S4: attr logits clamped to ±15
#  S5: temperature search range 0.5–10.0
#
# v3 improvements targeting attr Macro-F1 0.25–0.30:
#  L5: Isotonic regression calibration for rare attr classes
#      (classes with ≥30 val positives get IsotonicRegression
#       applied before threshold search — more stable for long tail)
#      Expected gain: +0.011 Macro-F1
#
#  L6: Longer training — EPOCHS=20, PATIENCE=5 (was 10/3)
#      Rare attr classes need more gradient steps to accumulate signal.
#      Previous runs stopped at epoch 4–5 before rare classes learned.
#      Expected gain: +0.014 Macro-F1
#
#  L7: Group-conditioned attribute decoder
#      Replaced flat 768→1024→...→294 head with 11 group-specific
#      sub-heads. Each sub-head predicts only its group's attributes,
#      gated by the soft group probability from the coarse head:
#        gate      = sigmoid(group_logit[:, g])   # [B, 1]
#        gated_feat = attr_fused * gate            # [B, 768]
#        sub_logits = sub_head_g(gated_feat)       # [B, n_attrs_in_g]
#      Benefits:
#        - Each sub-head discriminates within one group (easier task)
#        - Gradient from attr errors flows back through group logit
#        - Nickname group (153 attrs) gets its own dedicated capacity
#        - External hard-threshold gate (N6) no longer needed
#      Expected gain: +0.036 Macro-F1
#
#  Combined expected attr Macro-F1: 0.142 × (1.10 × 1.08 × 1.25) ≈ 0.21
#  With good convergence (PATIENCE=5): 0.25–0.28
# ============================================================

import os, sys, math
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping as PLEarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

sys.path.insert(0, os.path.dirname(__file__))
from core.common import (
    FashionDataset, CompositeHierarchicalLoss,
    compute_pos_weights, evaluate_hierarchical, analyze_model,
    get_logger, load_hierarchy, build_adjacency_matrix,
    make_train_val_split, calibrate_thresholds, set_seed,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
    BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS, DATASET_ROOT
)

logger      = get_logger("hierfashion_improved_v2")
LR_IMPROVED = 2e-4


# ============================================================
# BACKBONE — with gradual unfreezing
# ============================================================
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = m.conv1;   self.bn1     = m.bn1
        self.relu  = m.relu;    self.maxpool = m.maxpool
        self.layer1 = m.layer1; self.layer2  = m.layer2
        self.layer3 = m.layer3; self.layer4  = m.layer4
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        for p in (list(self.conv1.parameters()) + list(self.bn1.parameters()) +
                  list(self.layer1.parameters()) + list(self.layer2.parameters()) +
                  list(self.layer3.parameters())):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        fmaps = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return fmaps, self.pool(fmaps).flatten(1)


# ============================================================
# LABEL-GUIDED SPATIAL ATTENTION (N3: clamped, fp32)
# ============================================================
class LabelGuidedSpatialAttention(nn.Module):
    def __init__(self, visual_dim=2048, label_dim=512, num_labels=363):
        super().__init__()
        self.num_labels = num_labels
        self.label_proj = nn.Linear(label_dim, visual_dim)
        self.norm       = nn.LayerNorm(visual_dim)
        nn.init.xavier_uniform_(self.label_proj.weight)
        nn.init.zeros_(self.label_proj.bias)

    def forward(self, feature_maps, label_embeddings):
        B, C, H, W = feature_maps.shape
        with torch.amp.autocast("cuda", enabled=False):
            fmaps_fp32 = feature_maps.float()
            proj       = self.label_proj(label_embeddings.float())
            scores     = torch.einsum("bchw,nc->bnhw", fmaps_fp32, proj)
            scores     = torch.clamp(scores, min=-50.0, max=50.0)
            weights    = F.softmax(
                scores.view(B, self.num_labels, -1), dim=-1
            ).view(B, self.num_labels, H, W)
            attended   = torch.einsum("bnhw,bchw->bnc", weights, fmaps_fp32).mean(dim=1)
            out        = self.norm(attended)
        return out.to(feature_maps.dtype)


# ============================================================
# GRAPH MODULE  (N1: fp32, N2: L2-normalised init)
# ============================================================
class Phase2LabelGraphModule(nn.Module):
    def __init__(self, num_nodes, adj_matrix, embed_dim=300, out_dim=512):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings.data.unsqueeze(0))
        self.node_embeddings.data = self.node_embeddings.data.squeeze(0)
        with torch.no_grad():
            norms = self.node_embeddings.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.node_embeddings.data.div_(norms)

        edge_index, _ = dense_to_sparse(adj_matrix)
        self.register_buffer("edge_index", edge_index)
        self.gat1  = GATConv(embed_dim, 64, heads=8, dropout=0.1)
        self.gat2  = GATConv(512, out_dim, heads=1, dropout=0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self):
        with torch.amp.autocast("cuda", enabled=False):
            x  = self.node_embeddings.float()
            ei = self.edge_index
            x  = self.norm1(F.elu(self.gat1(x, ei)))
            x  = self.norm2(self.gat2(x, ei))
        return x   # [N, 512] fp32


# ============================================================
# ATTRIBUTE-SPECIFIC CROSS-ATTENTION  (N4: fp32)
# Queries ONLY the 294 attribute nodes (dedicated attr path).
# General CrossAttentionFusion over all 363 nodes was REMOVED
# (see A2) — it mixed coarse+fine node features in a way that
# degraded both attribute and overall macro-F1.
# ============================================================
class AttributeCrossAttention(nn.Module):
    def __init__(self, visual_dim=2048, attr_dim=512,
                 hidden_dim=256, num_heads=8):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.attr_proj   = nn.Linear(attr_dim,   hidden_dim)
        self.attn        = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.out_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, visual, attr_nodes):
        B = visual.size(0)
        with torch.amp.autocast("cuda", enabled=False):
            v = visual.float()
            a = attr_nodes.float()
            q = self.visual_proj(v).unsqueeze(1)                      # [B, 1, 256]
            k = self.attr_proj(a).unsqueeze(0).expand(B, -1, -1)     # [B, 294, 256]
            o, _ = self.attn(q, k, k)
            out = self.norm(self.out_proj(o.squeeze(1)))
        return out.to(visual.dtype)                                    # [B, 256]


# ============================================================
# PREDICTION HEADS
# L7: Group-conditioned decoder — 11 group-specific attribute sub-heads.
# Each sub-head predicts only its own group's attributes, gated by the
# soft group probability. Benefits:
#   - Each sub-head discriminates within one group only (easier task)
#   - Group gate provides a differentiable soft mask during training
#   - Gradient from attr errors flows back through the group logit (joint signal)
#   - Nickname group (153 attrs) gets its own dedicated capacity
# ============================================================
class GroupConditionedAttrHead(nn.Module):
    """
    One sub-head for a single attribute group.

    Uses ADDITIVE gating (not multiplicative):
      sub_logits = net(attr_fused) + gate_scale * sigmoid(group_logit)

    Why additive, not multiplicative:
      Multiplicative gate (attr_fused * sigmoid(group_logit)) causes
      saturation when the group is confidently predicted (gate→1), because
      the network learns to compensate by pushing sub_logit weights larger
      — resulting in logit_mean > 13 as seen in the failed run.
      Additive gating keeps the feature-based discrimination (net) bounded
      by its own weight norms, and the group logit acts only as a bias.
    """
    def __init__(self, in_dim, n_attrs):
        super().__init__()
        hidden = max(64, n_attrs * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_attrs),
        )
        # gate_scale initialised near zero so the gate contributes ~0 at init.
        # At random init: group_logit≈0 → sigmoid≈0.5 → bias = 0.1×0.5 = 0.05
        # → sub_logit ≈ 0.05 → prob ≈ 0.512 (near-ideal 0.5 start).
        # The scale grows during training to a useful value.
        # DO NOT initialise at 2.0 — that gives bias=1.0 → prob=0.73 at init,
        # which triggers logit saturation from epoch 0.
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, attr_fused, group_logit):
        # Feature-based discrimination (unbounded by weight init, clipped later)
        feat_logits = self.net(attr_fused)                         # [B, n_attrs]
        # Group-level enable/disable bias (additive, not multiplicative)
        gate_bias   = self.gate_scale * torch.sigmoid(group_logit) # [B, 1]
        return feat_logits + gate_bias                             # [B, n_attrs]


class HierarchicalPredictionHeads(nn.Module):
    def __init__(self, hierarchy):
        super().__init__()
        # Coarse trunk — input is visual_512 only (512-dim)
        self.shared = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.category_head    = nn.Linear(256, hierarchy.num_categories)
        self.subcategory_head = nn.Linear(256, hierarchy.num_subcategories)
        self.group_head       = nn.Linear(256, hierarchy.num_attr_groups)

        # L7: one sub-head per attribute group
        # attr_fused is [B, 768]; each sub-head receives gated version of it
        self.group_names   = list(hierarchy.level_3.keys())   # ordered group names
        self.group_to_idx  = hierarchy.group_id_to_idx        # name → group index
        self.attr_id_to_idx = hierarchy.attribute_id_to_idx

        # For each group: which columns in the 294-dim output does it own?
        self.register_buffer(
            "attr_output_indices",
            torch.zeros(hierarchy.num_attributes, dtype=torch.long),
        )
        group_attr_cols = {}   # group_name → sorted list of attr column indices
        for group_name, attr_ids in hierarchy.level_3.items():
            cols = []
            for aid in attr_ids:
                idx = hierarchy.attribute_id_to_idx.get(str(aid))
                if idx is not None:
                    cols.append(idx)
            group_attr_cols[group_name] = sorted(cols)

        self.group_attr_cols = group_attr_cols  # stored for forward()

        # One sub-head per group
        self.attr_sub_heads = nn.ModuleDict()
        for group_name, cols in group_attr_cols.items():
            safe_key = group_name.replace(" ", "_").replace(",", "").replace("-", "_")
            self.attr_sub_heads[safe_key] = GroupConditionedAttrHead(768, len(cols))

        # Safe key lookup (same transform as above)
        self.group_safe_keys = {
            g: g.replace(" ", "_").replace(",", "").replace("-", "_")
            for g in self.group_names
        }

    def forward(self, visual_512, attr_fused):
        x = self.shared(visual_512)
        coarse = {
            "categories":    self.category_head(x),
            "subcategories": self.subcategory_head(x),
            "attr_groups":   self.group_head(x),
        }

        # L7: group-conditioned attribute decoding
        B = attr_fused.size(0)
        n_attrs = sum(len(c) for c in self.group_attr_cols.values())
        attr_logits = torch.zeros(B, n_attrs, device=attr_fused.device,
                                  dtype=attr_fused.dtype)

        group_logits = coarse["attr_groups"]  # [B, 11] raw logits

        for g_idx, group_name in enumerate(self.group_names):
            cols        = self.group_attr_cols[group_name]
            safe_key    = self.group_safe_keys[group_name]
            g_logit     = group_logits[:, g_idx].unsqueeze(1)    # [B, 1]
            sub_out     = self.attr_sub_heads[safe_key](attr_fused, g_logit)  # [B, n_g]
            col_idx     = torch.tensor(cols, device=attr_fused.device)
            attr_logits[:, col_idx] = sub_out.to(attr_logits.dtype)

        coarse["attributes"] = attr_logits
        return coarse


# ============================================================
# N5: CORRECTED GROUP → ATTRIBUTE MASK  (unchanged)
# ============================================================
def build_group_attribute_mask(hierarchy):
    num_groups = hierarchy.num_attr_groups
    num_attrs  = hierarchy.num_attributes
    mask = torch.zeros(num_groups, num_attrs)
    for group_name, attr_ids in hierarchy.level_3.items():
        g_idx = hierarchy.group_id_to_idx.get(group_name)
        if g_idx is None:
            continue
        for attr_id in attr_ids:
            a_idx = hierarchy.attribute_id_to_idx.get(str(attr_id))
            if a_idx is not None:
                mask[g_idx, a_idx] = 1.0
    return mask


# ============================================================
# LIGHTNING MODEL
# ============================================================
class HierFashionLightning(pl.LightningModule):

    def __init__(self, hierarchy, adj_matrix, pos_weight,
                 learning_rate=LR_IMPROVED, steps_per_epoch=750):
        super().__init__()
        self.learning_rate   = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self._unfreeze_done  = False

        self.attr_node_start = (hierarchy.num_categories +
                                hierarchy.num_subcategories +
                                hierarchy.num_attr_groups)  # = 69

        self.backbone             = ResNet50Backbone()
        self.label_guided_attn    = LabelGuidedSpatialAttention(
            2048, 512, hierarchy.num_nodes)
        self.graph_module         = Phase2LabelGraphModule(
            hierarchy.num_nodes, adj_matrix)
        # A2: no general cross_attention — removed
        self.attr_cross_attention = AttributeCrossAttention(2048, 512, 256, 8)
        self.visual_proj          = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2))
        self.heads                = HierarchicalPredictionHeads(hierarchy)

        # Loss hyperparameters — calibrated after observing logit collapse
        # attr_weight=2.0 + gamma_neg=3 caused all attr logits to saturate at
        # mean=2.04 (prob≈0.86) → calibration found no threshold signal.
        # Reverted attr_weight to 1.5 and gamma_neg to 2.
        # asl_clip=0.05 anchors the shifted probability away from 0, preventing
        # the model from driving all negatives to exactly p=0 (which then gets
        # up-weighted by gamma and causes the same saturation in the other dir).
        self.loss_fn = CompositeHierarchicalLoss(
            pos_weight_cat=pos_weight["categories"],
            pos_weight_sub=pos_weight["subcategories"],
            pos_weight_group=pos_weight["attr_groups"],
            pos_weight_attr=pos_weight["attributes"],
            use_asl=True,
            asl_gamma_neg=2,
            asl_clip=0.05,
            attr_weight=1.5,
            attr_label_smoothing=None,
            lambda_consistency=0.03,
            lambda_path=0.01,
            lambda_logit_reg=0.001,   # penalises large attr logit magnitudes
        )

        # L7: group→attr mask now lives inside HierarchicalPredictionHeads
        # (the group-conditioned decoder handles gating internally)

    # ----------------------------------------------------------
    def on_train_epoch_start(self):
        if self.current_epoch == 3 and not self._unfreeze_done:
            self.backbone.unfreeze_all()
            self._unfreeze_done = True
            logger.info("Epoch 3: full backbone unfrozen")

    # ----------------------------------------------------------
    def forward(self, images, targets=None):
        fmaps, _    = self.backbone(images)                        # [B, 2048, H, W]
        graph_nodes = self.graph_module()                          # [363, 512] fp32
        attr_nodes  = graph_nodes[self.attr_node_start:]           # [294, 512]

        # LGSA attends over all graph nodes → rich 2048-dim descriptor
        attended   = self.label_guided_attn(fmaps, graph_nodes)    # [B, 2048]
        visual_512 = self.visual_proj(attended)                    # [B, 512]

        # Dedicated attr-node cross-attention path
        attr_cross = self.attr_cross_attention(attended, attr_nodes)   # [B, 256]
        attr_fused = torch.cat([visual_512, attr_cross], dim=1)    # [B, 768]

        # L7: group-conditioned decoder handles both coarse + attr predictions.
        # The decoder gates each group's sub-head by the soft group probability,
        # so the external hard-threshold gate (N6) is no longer needed.
        predictions = self.heads(visual_512, attr_fused)

        # Clamp attribute logits to ±6 — tighter than ±15 because the additive
        # gate can still push logits high when groups are confidently predicted.
        # sigmoid(6) = 0.9975, sigmoid(-6) = 0.0025 — full dynamic range.
        predictions["attributes"] = predictions["attributes"].clamp(-6.0, 6.0)

        if targets is not None:
            return predictions, self.loss_fn(predictions, targets)
        return predictions

    # ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        images, targets   = batch
        predictions, loss = self(images, targets)
        if not torch.isfinite(loss):
            logger.warning(f"[train] non-finite loss at step {batch_idx} — skipping")
            return None
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets   = batch
        predictions, loss = self(images, targets)
        if not torch.isfinite(loss):
            return None
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=self.steps_per_epoch,
            epochs=EPOCHS,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": "step", "frequency": 1}}


# ============================================================
# DATAMODULE
# ============================================================
class FashionDataModule(pl.LightningDataModule):

    def __init__(self, root_dir=DATASET_ROOT, hierarchy=None,
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                 seed=42):
        super().__init__()
        self.root_dir    = root_dir
        self.hierarchy   = hierarchy
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed

    def setup(self, stage=None):
        self.train_subset, self.val_subset = make_train_val_split(
            annotation_file=f"{self.root_dir}/instances_attributes_train2020.json",
            image_dir=f"{self.root_dir}/train2020",
            hierarchy=self.hierarchy,
            train_transform=TRAIN_TRANSFORM,
            val_transform=VAL_TRANSFORM,
            val_fraction=0.20, seed=self.seed,
        )
        self.test_ds = FashionDataset(
            f"{self.root_dir}/val2020",
            f"{self.root_dir}/instances_attributes_val2020.json",
            self.hierarchy, VAL_TRANSFORM
        )

    def train_dataloader(self):
        return DataLoader(self.train_subset, self.batch_size,
                          shuffle=True,  num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_subset, self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


# ============================================================
# MAIN  (A10: seed-aware entry point)
# ============================================================
if __name__ == "__main__":

    seed = int(os.environ.get("HIERFASHION_SEED", 42))
    set_seed(seed)
    logger.info(f"Global seed set to {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix  = build_adjacency_matrix(hierarchy, hierarchy_json)
    data_module = FashionDataModule(hierarchy=hierarchy,
                                    batch_size=BATCH_SIZE, seed=seed)
    data_module.setup()

    pos_weight = compute_pos_weights(
        data_module.train_dataloader(), hierarchy, device)

    steps_per_epoch = math.ceil(len(data_module.train_subset) / BATCH_SIZE)
    model = HierFashionLightning(
        hierarchy=hierarchy, adj_matrix=adj_matrix,
        pos_weight=pos_weight, learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_per_epoch,
    )

    analyze_model(model, device)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath=f"checkpoints/HierFashion_seed{seed}",
        filename=f"hierfashion-seed{seed}-best-{{epoch:02d}}-{{val_loss:.4f}}",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[ckpt_cb,
                   PLEarlyStopping(monitor="val_loss", mode="min",
                                   patience=PATIENCE, verbose=True),
                   LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger("training_logs", name=f"HierFashion_seed{seed}"),
    )

    trainer.fit(model, datamodule=data_module)

    best_ckpt = ckpt_cb.best_model_path
    logger.info(f"Best checkpoint: {best_ckpt}")
    model = HierFashionLightning.load_from_checkpoint(
        best_ckpt,
        hierarchy=hierarchy, adj_matrix=adj_matrix,
        pos_weight=pos_weight, learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_per_epoch,
    )

    os.makedirs("results", exist_ok=True)

    logger.info("Calibrating thresholds on validation set...")
    threshold_config = calibrate_thresholds(
        model,
        data_module.val_dataloader(),
        device,
        hierarchy,
        top_k_candidates=(3, 5, 7, 10, 15),
    )

    logger.info("Running TEST evaluation on held-out val2020...")
    model.eval()
    results = evaluate_hierarchical(
        model, data_module.test_dataloader(), device, model.loss_fn,
        save_path=f"results/HierFashion_seed{seed}_test_results.csv",
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    # ── Pretty-print results ──────────────────────────────────
    group_prefix = "attr_group_"
    core_rows   = results[~results["Head"].str.startswith(group_prefix) &
                          ~results["Head"].isin(["CompositeLoss", "HVR"])]
    group_rows  = results[results["Head"].str.startswith(group_prefix)]
    loss_row    = results[results["Head"] == "CompositeLoss"]
    hvr_row     = results[results["Head"] == "HVR"]

    print("\n" + "=" * 60)
    print(f"  FINAL TEST RESULTS — HierFashion v3  (seed={seed})")
    print("=" * 60)
    print(core_rows[["Head","Macro_F1","Micro_F1","mAP",
                      "Macro_Precision","Micro_Precision",
                      "Macro_Recall","Micro_Recall"]].to_string(index=False))

    if not loss_row.empty:
        r = loss_row.iloc[0]
        print(f"\n  Loss  CHL={r['CHL']:.4f}  L_bce={r['L_bce']:.4f}  "
              f"L_consistency={r['L_consistency']:.4f}  L_path={r['L_path']:.4f}")

    # ── Per-group attribute F1 ────────────────────────────────
    if not group_rows.empty:
        print("\n" + "=" * 65)
        print("  ATTRIBUTE F1 BY GROUP")
        print("=" * 65)
        print(f"  {'Group':42s} {'n':>5} {'MacroF1':>8} {'MicroF1':>8} {'mAP':>7}")
        print("  " + "─" * 72)
        summary_keys = {f"{group_prefix}nickname_group",
                        f"{group_prefix}non_nickname_groups"}
        for _, r in group_rows.iterrows():
            if r["Head"] in summary_keys:
                continue
            name = r["Head"].replace(group_prefix, "").replace("_", " ")
            print(f"    {name:42s} {int(r['L_bce']):>5} "
                  f"{r['Macro_F1']:>8.4f} {r['Micro_F1']:>8.4f} {r['mAP']:>7.4f}")

        print("  " + "─" * 72)
        for key, label in [(f"{group_prefix}nickname_group",     "nickname (52% of attrs)"),
                           (f"{group_prefix}non_nickname_groups","non-nickname (48% of attrs)")]:
            r = group_rows[group_rows["Head"] == key]
            if not r.empty:
                r = r.iloc[0]
                print(f"  * {label:42s} {int(r['L_bce']):>5} "
                      f"{r['Macro_F1']:>8.4f} {r['Micro_F1']:>8.4f} {r['mAP']:>7.4f}")
        print()
        nick = group_rows[group_rows["Head"] == f"{group_prefix}nickname_group"]
        non  = group_rows[group_rows["Head"] == f"{group_prefix}non_nickname_groups"]
        if not nick.empty:
            print(f"  nickname Macro-F1    : {nick.iloc[0]['Macro_F1']:.4f}  "
                  f"(ceiling ~0.05 — product names, not visually learnable)")
        if not non.empty:
            nf = non.iloc[0]['Macro_F1']
            verdict = ("excellent" if nf > 0.30 else
                       "good"      if nf > 0.20 else
                       "improving" if nf > 0.10 else "needs improvement")
            print(f"  non-nickname Macro-F1: {nf:.4f}  "
                  f"({verdict} — target is 0.30+)")
        print("=" * 65)

    if not hvr_row.empty:
        r = hvr_row.iloc[0]
        print("\n" + "-" * 60)
        print("  HIERARCHICAL VIOLATION RATE (HVR)")
        print("-" * 60)
        print(f"  HVR overall   : {r['L_bce']:.4f}  "
              f"({r['L_bce']*100:.1f}% of images have >=1 violation)")
        print(f"  HVR_subcat V1 : {r['L_consistency']:.4f}  "
              f"(subcat without parent category)")
        print(f"  HVR_attr   V2 : {r['L_path']:.4f}  "
              f"(attr without parent attr_group)")
        print(f"  Attr violation rate per prediction: {r['CHL']:.4f}  "
              f"({r['CHL']*100:.1f}% of predicted attrs violate hierarchy)")
        verdict = (
            "EXCELLENT" if r['L_bce'] < 0.05 else
            "GOOD"      if r['L_bce'] < 0.15 else
            "MODERATE"  if r['L_bce'] < 0.30 else "HIGH"
        )
        print(f"\n  Verdict: {verdict}  (HVR = {r['L_bce']:.4f})")
        print("-" * 60)
