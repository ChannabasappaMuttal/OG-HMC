# ============================================================
# HIERFASHION — core/common.py
# Shared config, hierarchy, dataset, backbone, loss, evaluation
# ============================================================

import os
import json
import time
import logging
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score
)

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE    = 64
LR            = 2e-4    # default for all models — improved convergence
EPOCHS        = 4       # L6: increased from 10 — rare attr classes need more steps
PATIENCE      = 2        # L6: increased from 3 — gives model time past plateau
IMAGE_SIZE    = 224
NUM_WORKERS   = 4

DATASET_ROOT  = "/teamspace/studios/this_studio/dataset/fashionpedia"
TRAIN_IMAGES  = f"{DATASET_ROOT}/train2020"
VAL_IMAGES    = f"{DATASET_ROOT}/val2020"
TRAIN_ANN     = f"{DATASET_ROOT}/instances_attributes_train2020.json"
VAL_ANN       = f"{DATASET_ROOT}/instances_attributes_val2020.json"
HIERARCHY_PATH = "/teamspace/studios/this_studio/hierarchy_outputs/fashionpedia_hierarchy.json"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# LOGGING
# ============================================================

def get_logger(name="experiment"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/experiment_log.txt")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s — %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = get_logger()

# ============================================================
# HIERARCHY
# ============================================================

class FashionHierarchy:

    def __init__(self, hierarchy_json):
        self.hierarchy   = hierarchy_json
        self.level_1     = hierarchy_json["level_1_category"]
        self.level_2     = hierarchy_json["level_2_subcategory"]
        self.level_3     = hierarchy_json["level_3_attribute_group"]
        self.level_4     = hierarchy_json["level_4_attributes"]
        self.node_index  = hierarchy_json["node_index"]

        self.category_id_to_idx    = {str(k): i for i, k in enumerate(self.level_1)}
        self.subcategory_id_to_idx = {k: i for i, k in enumerate(self.level_2)}
        self.group_id_to_idx       = {k: i for i, k in enumerate(self.level_3)}
        self.attribute_id_to_idx   = {str(k): i for i, k in enumerate(self.level_4)}

        self.num_categories  = len(self.level_1)
        self.num_subcategories = len(self.level_2)
        self.num_attr_groups = len(self.level_3)
        self.num_attributes  = len(self.level_4)
        self.num_nodes       = len(self.node_index)

        # category_id → subcategory_name
        self.category_to_subcategory = {}
        for sub_name, cat_list in self.level_2.items():
            for cat_id in cat_list:
                self.category_to_subcategory[str(cat_id)] = sub_name

        # attribute_id → group_name
        self.attribute_to_group = {}
        for group_name, attr_list in self.level_3.items():
            for attr_id in attr_list:
                self.attribute_to_group[str(attr_id)] = group_name


def load_hierarchy(path=HIERARCHY_PATH):
    with open(path) as f:
        hierarchy_json = json.load(f)
    hierarchy = FashionHierarchy(hierarchy_json)
    logger.info(f"Hierarchy loaded — nodes: {hierarchy.num_nodes}, "
                f"cats: {hierarchy.num_categories}, "
                f"subcats: {hierarchy.num_subcategories}, "
                f"groups: {hierarchy.num_attr_groups}, "
                f"attrs: {hierarchy.num_attributes}")
    return hierarchy, hierarchy_json


def build_adjacency_matrix(hierarchy, hierarchy_json):
    node_index = hierarchy.node_index
    N = hierarchy.num_nodes
    adj = torch.zeros(N, N)

    # Level 1 ↔ Level 2
    for subcat, cat_ids in hierarchy_json["level_2_subcategory"].items():
        sub_node = node_index[f"subcat_{subcat}"]
        for cid in cat_ids:
            cat_node = node_index[f"cat_{cid}"]
            adj[sub_node, cat_node] = 1
            adj[cat_node, sub_node] = 1

    # Level 3 ↔ Level 4
    for group, attr_ids in hierarchy_json["level_3_attribute_group"].items():
        group_node = node_index[f"group_{group}"]
        for aid in attr_ids:
            attr_node = node_index[f"attr_{aid}"]
            adj[group_node, attr_node] = 1
            adj[attr_node, group_node] = 1

    adj += torch.eye(N)   # self-loops
    logger.info(f"Adjacency matrix built — shape: {adj.shape}, "
                f"non-zero edges: {int(adj.sum().item())}")
    return adj


def build_category_attribute_mask(hierarchy):
    mask = torch.zeros(hierarchy.num_categories, hierarchy.num_attributes)
    for attr_id, group_name in hierarchy.attribute_to_group.items():
        attr_idx = hierarchy.attribute_id_to_idx[str(attr_id)]
        for cat_id, _ in hierarchy.category_to_subcategory.items():
            cat_idx = hierarchy.category_id_to_idx[str(cat_id)]
            if group_name in hierarchy.group_id_to_idx:
                mask[cat_idx, attr_idx] = 1
    return mask

# ============================================================
# DATASET
# ============================================================

class FashionDataset(Dataset):

    def __init__(self, image_dir, annotation_file, hierarchy, transform=None):
        self.image_dir  = image_dir
        self.hierarchy  = hierarchy
        self.transform  = transform

        with open(annotation_file) as f:
            data = json.load(f)

        self.images      = data["images"]
        self.annotations = data["annotations"]

        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info    = self.images[idx]
        path    = os.path.join(self.image_dir, info["file_name"])
        image   = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        targets = {
            "categories":    torch.zeros(self.hierarchy.num_categories),
            "subcategories": torch.zeros(self.hierarchy.num_subcategories),
            "attr_groups":   torch.zeros(self.hierarchy.num_attr_groups),
            "attributes":    torch.zeros(self.hierarchy.num_attributes),
        }

        for ann in self.img_to_anns[info["id"]]:
            cat_id = str(ann["category_id"])

            if cat_id in self.hierarchy.category_id_to_idx:
                targets["categories"][self.hierarchy.category_id_to_idx[cat_id]] = 1

            if cat_id in self.hierarchy.category_to_subcategory:
                sub = self.hierarchy.category_to_subcategory[cat_id]
                if sub in self.hierarchy.subcategory_id_to_idx:
                    targets["subcategories"][self.hierarchy.subcategory_id_to_idx[sub]] = 1

            for attr_id in ann.get("attribute_ids", []):
                aid = str(attr_id)
                if aid in self.hierarchy.attribute_id_to_idx:
                    targets["attributes"][self.hierarchy.attribute_id_to_idx[aid]] = 1
                    group = self.hierarchy.attribute_to_group.get(aid)
                    if group and group in self.hierarchy.group_id_to_idx:
                        targets["attr_groups"][self.hierarchy.group_id_to_idx[group]] = 1

        return image, targets


# ============================================================
# TRAIN/VAL SPLITTER
# Splits train2020 into internal train (80%) and val (20%).
# val2020 is kept strictly as the TEST set — never used for
# any training decision (early stopping, LR scheduling, etc.)
# ============================================================

class FashionDatasetSubset(torch.utils.data.Dataset):
    """Wraps FashionDataset with an index subset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def make_train_val_split(annotation_file, image_dir, hierarchy,
                          train_transform, val_transform,
                          val_fraction=0.20, seed=42):
    """
    Split train2020 into:
      - internal train set (80%) — used for gradient updates
      - internal val set  (20%) — used for early stopping only

    val2020 (test set) is NOT touched here — kept for final eval.
    """
    import random
    full_ds = FashionDataset(image_dir, annotation_file,
                              hierarchy, train_transform)
    n       = len(full_ds)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    split    = int(n * (1 - val_fraction))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    # Val subset needs val_transform (no augmentation)
    val_ds_full = FashionDataset(image_dir, annotation_file,
                                  hierarchy, val_transform)

    train_subset = FashionDatasetSubset(full_ds,     train_idx)
    val_subset   = FashionDatasetSubset(val_ds_full, val_idx)

    logger.info(f"Train split : {len(train_subset):,} images  "
                f"({100*(1-val_fraction):.0f}%)")
    logger.info(f"Val split   : {len(val_subset):,} images  "
                f"({100*val_fraction:.0f}%)")
    return train_subset, val_subset


# ============================================================
# POS WEIGHT COMPUTATION
# ============================================================

def compute_pos_weights(dataloader, hierarchy, device):
    sizes = {"categories": hierarchy.num_categories,
             "subcategories": hierarchy.num_subcategories,
             "attr_groups": hierarchy.num_attr_groups,
             "attributes": hierarchy.num_attributes}
    sums  = {k: torch.zeros(sizes[k]) for k in sizes}
    total = 0

    for _, targets in dataloader:
        for k in sums:
            sums[k] += targets[k].sum(0)
        total += targets["categories"].size(0)

    pos_weights = {}
    for k, s in sums.items():
        neg = total - s
        w   = torch.clamp(neg / (s + 1), max=10)
        pos_weights[k] = w.to(device)
    return pos_weights


# ============================================================
# COMPOSITE HIERARCHICAL LOSS
# ============================================================

class CompositeHierarchicalLoss(nn.Module):

    def __init__(self,
                 pos_weight_cat=None, pos_weight_sub=None,
                 pos_weight_group=None, pos_weight_attr=None,
                 gamma=3.5, alpha=0.25,
                 lambda_consistency=0.1, lambda_path=0.05,
                 lambda_logit_reg=0.001,    # L2 penalty on attr logit magnitude
                 label_smoothing=0.05,
                 # ── attribute-specific params (defaults keep baselines unchanged) ──
                 attr_weight=1.0,           # 2.5 for main model, 1.0 for baselines
                 attr_label_smoothing=None, # 0.10 for main model, None=same as label_smoothing
                 use_asl=False,             # True for main model, False for baselines
                 asl_gamma_neg=2,           # ASL: moderate negative suppression (4 was too aggressive)
                 asl_gamma_pos=0,           # ASL: don't down-weight positives
                 asl_clip=0.0,              # ASL: no clip (clip=0.05 caused threshold collapse)
                 ):
        super().__init__()
        self.lambda_consistency   = lambda_consistency
        self.lambda_path          = lambda_path
        self.lambda_logit_reg     = lambda_logit_reg
        self.label_smoothing      = label_smoothing
        self.attr_label_smoothing = attr_label_smoothing if attr_label_smoothing is not None                                     else label_smoothing
        self.gamma                = gamma
        self.alpha                = alpha
        self.attr_weight          = attr_weight
        self.use_asl              = use_asl
        self.asl_gamma_neg        = asl_gamma_neg
        self.asl_gamma_pos        = asl_gamma_pos
        self.asl_clip             = asl_clip

        self.bce_cat   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cat)
        self.bce_sub   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_sub)
        self.bce_group = nn.BCEWithLogitsLoss(pos_weight=pos_weight_group)

    def _smooth(self, t, s=None):
        s = s if s is not None else self.label_smoothing
        return t * (1 - s) + 0.5 * s

    def _focal(self, logits, targets):
        logits  = logits.float()          # fp32 for stability under AMP
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        w   = self.alpha * (1 - pt) ** self.gamma
        return (w * bce).mean()

    def _asl(self, logits, targets):
        """Asymmetric Loss — fp32 cast prevents log(0) NaN under AMP fp16."""
        logits  = logits.float()          # cast to fp32 before log operations
        targets = targets.float()
        p       = torch.sigmoid(logits)
        p_m     = torch.clamp(p - self.asl_clip, min=0)
        loss_p  = targets       * torch.log(p.clamp(min=1e-6))
        loss_n  = (1 - targets) * torch.log((1 - p_m).clamp(min=1e-6))
        loss    = -(loss_p + loss_n)
        p_t     = p * targets + p_m * (1 - targets)
        gamma   = self.asl_gamma_pos * targets + self.asl_gamma_neg * (1 - targets)
        result  = ((1 - p_t) ** gamma * loss).mean()
        # Safety net — return detached zero if still NaN (should not happen)
        if not torch.isfinite(result):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return result

    def forward(self, predictions, targets, return_components=False):
        L_cat   = self.bce_cat(predictions["categories"].float(),
                               self._smooth(targets["categories"].float()))
        L_sub   = self.bce_sub(predictions["subcategories"].float(),
                               self._smooth(targets["subcategories"].float()))
        L_group = self.bce_group(predictions["attr_groups"].float(),
                                 self._smooth(targets["attr_groups"].float()))

        # Attribute loss — ASL (main model) or focal (baselines, use_asl=False)
        attr_t  = self._smooth(targets["attributes"].float(),
                               s=self.attr_label_smoothing)
        L_attr  = self._asl(predictions["attributes"], attr_t) if self.use_asl                   else self._focal(predictions["attributes"], attr_t)

        # Attribute reweighting — 2.5 for main model, 1.0 for baselines
        L_bce   = (L_cat + L_sub + L_group + self.attr_weight * L_attr) /                   (3 + self.attr_weight)

        cat_p   = torch.sigmoid(predictions["categories"])
        sub_p   = torch.sigmoid(predictions["subcategories"])
        grp_p   = torch.sigmoid(predictions["attr_groups"])
        att_p   = torch.sigmoid(predictions["attributes"])

        L_cs    = (torch.mean((sub_p.mean(1) - cat_p.mean(1)) ** 2) +
                   torch.mean((grp_p.mean(1) - sub_p.mean(1)) ** 2) +
                   torch.mean((att_p.mean(1) - grp_p.mean(1)) ** 2)) / 3
        L_path  = torch.mean((att_p.mean(1) - cat_p.mean(1)) ** 2)

        # Logit regularisation — penalises large attr logit magnitudes directly.
        # Prevents the group-conditioned decoder from pushing logits to ceiling.
        # Mean squared logit: encourages logits to stay near 0 by default.
        L_reg   = (predictions["attributes"].float() ** 2).mean()
        CHL     = (L_bce + self.lambda_consistency * L_cs
                   + self.lambda_path * L_path
                   + self.lambda_logit_reg * L_reg)

        if return_components:
            return CHL, L_bce, L_cs, L_path
        return CHL


# ============================================================
# SIMPLE BCE LOSS (for baselines)
# ============================================================

class FlatBCELoss(nn.Module):
    """Standard BCE for flat baselines that output a single vector."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.bce(logits, targets)


# ============================================================
# BACKBONE
# ============================================================

class ResNet50Backbone(nn.Module):
    """Returns (feature_maps [B,2048,H,W], pooled [B,2048])."""

    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        fmaps  = self.features(x)
        pooled = self.pool(fmaps).flatten(1)
        return fmaps, pooled


class ResNet50BackboneFlat(nn.Module):
    """Returns only pooled [B,2048] — for flat baselines."""

    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)


class ViTBackbone(nn.Module):
    """ViT-B/16 backbone returning [B,768] CLS token."""

    def __init__(self):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.backbone = vit
        self.backbone.heads = nn.Identity()

    def forward(self, x):
        return self.backbone(x)   # [B, 768]


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.counter  = 0
        self.best     = None
        self.stop     = False

    def step(self, val):
        if self.best is None or val < self.best:
            self.best    = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# EVALUATION — identical across all models
# ============================================================

# ============================================================
# THRESHOLD CALIBRATION HELPERS
# All tuning is done on the validation set and the chosen
# thresholds are then applied unchanged to the test set.
# ============================================================

def _search_global_threshold(probs, true, thresholds=None, min_threshold=0.10):
    """
    Sweep a range of global thresholds and return the one that
    maximises Micro-F1 over the full label matrix.

    Guards:
    - Never returns a threshold below min_threshold (avoids Recall=1 collapse)
    - If no threshold beats F1=0, returns 0.5 (safe neutral)
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)   # floor at 0.10

    best_t, best_f1 = 0.5, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in thresholds:
            if t < min_threshold:
                continue
            preds = (probs > t).astype(int)
            if preds.sum() == 0:
                continue
            # Guard: reject if recall is suspiciously perfect (threshold too low)
            rec = recall_score(true, preds, average="micro", zero_division=0)
            if rec > 0.99 and t < 0.40:
                continue
            f1 = f1_score(true, preds, average="micro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def _search_per_class_thresholds(probs, true, n_steps=50):
    """
    L5: Hybrid per-class threshold calibration.

    - Classes with >=30 val positives: isotonic regression recalibrates
      probabilities first, then F1-optimal threshold search.
    - Classes with 10-29 val positives: F1-optimal threshold on raw probs.
    - Classes with <10 val positives or 0: threshold=0.90 (suppress).

    Isotonic regression benefit: monotonically maps uncalibrated model probs
    to calibrated ones — critical for rare attributes where the model outputs
    near-uniform scores and the threshold search finds noise otherwise.
    """
    from sklearn.isotonic import IsotonicRegression

    n_classes   = probs.shape[1]
    per_class_t = np.full(n_classes, 0.50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(n_classes):
            pos_count = int(true[:, c].sum())
            if pos_count == 0:
                per_class_t[c] = 0.90
                continue
            if pos_count < 10:
                per_class_t[c] = 0.90
                continue

            p_c = probs[:, c].copy()
            y_c = true[:, c]

            # L5: isotonic recalibration for classes with enough positive examples
            if pos_count >= 30:
                try:
                    ir = IsotonicRegression(out_of_bounds="clip")
                    p_c = ir.fit_transform(p_c, y_c)
                except Exception:
                    pass

            # F1-optimal threshold on (possibly recalibrated) probs
            max_preds = max(pos_count * 5, 10)
            best_f1, best_t = 0.0, 0.50
            for t in np.linspace(0.05, 0.90, n_steps):
                preds_c = (p_c > t).astype(int)
                n_pred  = preds_c.sum()
                if n_pred == 0 or n_pred > max_preds:
                    continue
                f1_c = f1_score(y_c, preds_c, zero_division=0)
                if f1_c > best_f1:
                    best_f1, best_t = f1_c, t
            if best_f1 > 0.05:
                per_class_t[c] = best_t

    return per_class_t


def _apply_topk_filter(probs, preds, k_values=(3, 5, 7, 10, 15), true=None):
    """
    Top-K filtering: for each image, if more than K labels are predicted,
    keep only the K with highest probability.  Does NOT force exactly K
    predictions — images with fewer than K predictions are untouched.

    When `true` is provided, the best K is chosen by Micro-F1 on val.

    Args:
        probs    : np.ndarray [N, C]  sigmoid probabilities
        preds    : np.ndarray [N, C]  binary predictions (from per-class thresh)
        k_values : K candidates to search
        true     : np.ndarray [N, C] or None

    Returns:
        filtered_preds : np.ndarray [N, C]
        best_k         : int
    """
    def _apply_k(p, pr, k):
        out = pr.copy()
        for i in range(p.shape[0]):
            pos_idx = np.where(pr[i] == 1)[0]
            if len(pos_idx) > k:
                top_idx = pos_idx[np.argsort(p[i, pos_idx])[-k:]]
                out[i]  = 0
                out[i, top_idx] = 1
        return out

    best_k, best_f1 = k_values[-1], 0.0   # default to largest K (least aggressive)

    if true is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in k_values:
                filtered = _apply_k(probs, preds, k)
                f1 = f1_score(true, filtered, average="micro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_k = f1, k

    filtered_preds = _apply_k(probs, preds, best_k)
    return filtered_preds, best_k


def _apply_hierarchical_filter(attr_probs, group_probs, hierarchy):
    """
    Enforce hierarchical consistency using the CORRECT branch of the
    Fashionpedia hierarchy: attr_groups → attributes.

    In Fashionpedia, categories and attributes are SEPARATE branches —
    there is no direct category→attribute mapping.  The correct parent
    of an attribute is its attribute_group (level_3).

    Rule: if an attribute group is NOT predicted for an image, zero-out
    all attribute probabilities that belong to that group.

    This is well-defined, always reduces false positives, and never
    suppresses a correct attribute prediction unless its group was
    also missed (which would itself be a model error).

    Args:
        attr_probs  : np.ndarray [N, num_attrs]
        group_probs : np.ndarray [N, num_groups]  sigmoid probabilities
        hierarchy   : FashionHierarchy

    Returns:
        filtered_attr_probs : np.ndarray [N, num_attrs]
    """
    num_groups = hierarchy.num_attr_groups
    num_attrs  = hierarchy.num_attributes

    # Build [num_groups, num_attrs] membership mask
    group_attr_mask = np.zeros((num_groups, num_attrs), dtype=np.float32)
    for group_name, attr_ids in hierarchy.level_3.items():
        g_idx = hierarchy.group_id_to_idx.get(group_name)
        if g_idx is None:
            continue
        for attr_id in attr_ids:
            a_idx = hierarchy.attribute_id_to_idx.get(str(attr_id))
            if a_idx is not None:
                group_attr_mask[g_idx, a_idx] = 1.0

    # Per-image: find predicted groups (threshold 0.3 — already calibrated),
    # build union of allowed attrs, zero-out attrs outside that union.
    # Use 0.3 as a conservative threshold to avoid over-suppression.
    N        = attr_probs.shape[0]
    filtered = attr_probs.copy()
    group_binary = (group_probs > 0.30).astype(np.float32)  # [N, num_groups]

    # allowed[i] = union of group_attr_mask rows for predicted groups
    allowed = np.dot(group_binary, group_attr_mask)          # [N, num_attrs]
    allowed = np.clip(allowed, 0, 1)

    # Only apply where at least one group is predicted (avoid all-zero mask)
    has_group = (group_binary.sum(axis=1) > 0)               # [N]
    filtered[has_group] *= allowed[has_group]

    return filtered


def _fit_temperature(logits_dict, true_dict, keys,
                     temps=(0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)):
    """
    Find a single per-head temperature T that maximises val Micro-F1
    at threshold=0.5 after scaling: prob = sigmoid(logit / T).

    T < 1  → spreads probabilities out  (fixes under-confident model)
    T > 1  → squashes probabilities in  (fixes over-confident model)
    T = 1  → no change

    Returns dict: {head: best_T}
    """
    best_temps = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in keys:
            logits = logits_dict[k]
            true   = true_dict[k]
            best_t_val, best_T = 0.0, 1.0
            for T in temps:
                probs = 1.0 / (1.0 + np.exp(-logits / T))
                preds = (probs > 0.5).astype(int)
                if preds.sum() == 0:
                    continue
                # Skip temperatures that produce absurd positive rates
                # (e.g. T=0.1 on saturated logits → 88% predicted positive)
                pred_rate = preds.mean()
                pos_rate  = true.mean()
                if pos_rate < 0.15 and pred_rate > 0.30:
                    continue
                f1 = f1_score(true, preds, average="micro", zero_division=0)
                if f1 > best_t_val:
                    best_t_val, best_T = f1, T
            best_temps[k] = best_T
            logger.info(f"[Temperature] {k}: best T={best_T:.2f}  "
                        f"val Micro-F1@0.5={best_t_val:.4f}")
    return best_temps


def calibrate_thresholds(model, val_loader, device, hierarchy,
                          top_k_candidates=(3, 5, 7, 10, 15)):
    """
    Full post-hoc calibration pipeline on the VALIDATION set.

    Step 0: Temperature scaling  — rescale logits so sigmoid outputs
            are in a useful range before thresholding.
    Step 1: Global threshold search  — categories / subcategories / attr_groups
    Step 2: Per-class thresholds     — attributes
    Step 3: Top-K cap search         — attributes
    Step 4: Hierarchical filter      — applied at test time

    All choices are made on val — never on test.
    """
    model.eval()
    model.to(device)

    keys = ["categories", "subcategories", "attr_groups", "attributes"]
    all_logits  = {k: [] for k in keys}
    all_targets = {k: [] for k in keys}

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            preds  = model(images)
            for k in keys:
                all_logits[k].append(preds[k].detach().cpu())
                all_targets[k].append(targets[k].cpu())

    logits_dict = {}
    true_dict   = {}
    for k in keys:
        logits_dict[k] = torch.cat(all_logits[k]).numpy()
        true_dict[k]   = torch.cat(all_targets[k]).numpy()

    # ── Diagnostics ─────────────────────────────────────────────
    for k in keys:
        lg = logits_dict[k]
        t  = true_dict[k]
        p  = 1.0 / (1.0 + np.exp(-lg))
        logger.info(
            f"[Diag] {k}: logit_mean={lg.mean():.3f}  logit_std={lg.std():.3f}  "
            f"prob_mean={p.mean():.4f}  prob_p90={np.percentile(p,90):.4f}  "
            f"pos_rate={t.mean():.4f}  "
            f"preds@0.5={(p>0.5).mean():.4f}  preds@0.3={(p>0.3).mean():.4f}"
        )

    # ── Step 0: Temperature scaling ─────────────────────────────
    temps = _fit_temperature(logits_dict, true_dict, keys)

    # Apply temperature to get calibrated probs
    probs_dict = {}
    for k in keys:
        T = temps[k]
        probs_dict[k] = 1.0 / (1.0 + np.exp(-logits_dict[k] / T))

    config = {"_temperatures": temps}

    # ── Step 1: global threshold for coarse heads ───────────────
    for k in ["categories", "subcategories", "attr_groups"]:
        best_t, best_f1 = _search_global_threshold(probs_dict[k], true_dict[k])
        config[k] = {"mode": "global", "threshold": best_t}
        logger.info(f"[Calibration] {k}: T={temps[k]:.2f}  "
                    f"threshold={best_t:.2f}  val Micro-F1={best_f1:.4f}")

    # ── Step 2: per-class thresholds for attributes ─────────────
    per_class_t = _search_per_class_thresholds(
        probs_dict["attributes"], true_dict["attributes"])
    logger.info(
        f"[Calibration] attributes: T={temps['attributes']:.2f}  "
        f"per-class thresh mean={per_class_t.mean():.3f}  "
        f"min={per_class_t.min():.3f}  max={per_class_t.max():.3f}"
    )

    # ── Step 3: top-K search ────────────────────────────────────
    attr_probs_topk = probs_dict["attributes"].copy()
    attr_probs_topk = _apply_hierarchical_filter(
        attr_probs_topk, probs_dict["attr_groups"], hierarchy)
    attr_preds_topk = (attr_probs_topk > per_class_t[np.newaxis, :]).astype(int)
    _, best_k = _apply_topk_filter(
        attr_probs_topk, attr_preds_topk,
        k_values=top_k_candidates,
        true=true_dict["attributes"]
    )
    logger.info(f"[Calibration] attributes best top-K={best_k}")

    config["attributes"] = {
        "mode":        "per_class_topk",
        "per_class_t": per_class_t,
        "top_k":       best_k,
    }

    return config



def compute_hvr(pred_cats, pred_subcats, pred_groups, pred_attrs, hierarchy):
    """
    Hierarchical Violation Rate (HVR).

    Measures how often the model predicts a child label without predicting
    the required parent label.  Two violation types:

      V1 — Subcategory without parent category:
           predicted subcat X but no predicted category is a parent of X.

      V2 — Attribute without parent attr_group:
           predicted attribute A but its parent attr_group is not predicted.

    Args:
        pred_cats    : np.ndarray [N, num_categories]   binary predictions
        pred_subcats : np.ndarray [N, num_subcategories] binary predictions
        pred_groups  : np.ndarray [N, num_attr_groups]  binary predictions
        pred_attrs   : np.ndarray [N, num_attributes]   binary predictions
        hierarchy    : FashionHierarchy

    Returns dict with keys:
        HVR            : fraction of images with >=1 violation (either type)
        HVR_subcat     : fraction with >=1 V1 violation
        HVR_attr       : fraction with >=1 V2 violation
        violation_subcat_pct : fraction of *predicted* subcats that are violations
        violation_attr_pct   : fraction of *predicted* attrs that are violations
        n_images       : total images evaluated
    """
    import numpy as np

    N = pred_cats.shape[0]

    # Build subcat_idx → set of valid parent cat_idx
    subcat_parent_cats = {}
    for subcat_name, parent_cat_ids in hierarchy.level_2.items():
        si = hierarchy.subcategory_id_to_idx.get(subcat_name)
        if si is None:
            continue
        parents = set()
        for cid in parent_cat_ids:
            ci = hierarchy.category_id_to_idx.get(str(cid))
            if ci is not None:
                parents.add(ci)
        subcat_parent_cats[si] = parents

    # Build attr_idx → parent group_idx
    attr_parent_group = {}
    for group_name, group_attr_ids in hierarchy.level_3.items():
        gi = hierarchy.group_id_to_idx.get(group_name)
        if gi is None:
            continue
        for aid in group_attr_ids:
            ai = hierarchy.attribute_id_to_idx.get(str(aid))
            if ai is not None:
                attr_parent_group[ai] = gi

    v1_images        = 0   # images with >=1 V1
    v2_images        = 0   # images with >=1 V2
    v1_count_total   = 0   # total V1 violations across all images
    v2_count_total   = 0   # total V2 violations
    pred_subcat_total = 0  # total predicted subcats
    pred_attr_total   = 0  # total predicted attrs

    for i in range(N):
        cats_i    = set(np.where(pred_cats[i]    == 1)[0])
        subcats_i = set(np.where(pred_subcats[i] == 1)[0])
        groups_i  = set(np.where(pred_groups[i]  == 1)[0])
        attrs_i   = set(np.where(pred_attrs[i]   == 1)[0])

        # V1: subcat predicted but no parent cat predicted
        v1_here = 0
        for si in subcats_i:
            parents = subcat_parent_cats.get(si, set())
            if not parents.intersection(cats_i):
                v1_here += 1
        pred_subcat_total += len(subcats_i)
        v1_count_total    += v1_here
        if v1_here > 0:
            v1_images += 1

        # V2: attr predicted but its parent group not predicted
        v2_here = 0
        for ai in attrs_i:
            gi = attr_parent_group.get(ai)
            if gi is not None and gi not in groups_i:
                v2_here += 1
        pred_attr_total += len(attrs_i)
        v2_count_total  += v2_here
        if v2_here > 0:
            v2_images += 1

    any_violation = 0
    for i in range(N):
        cats_i    = set(np.where(pred_cats[i]    == 1)[0])
        subcats_i = set(np.where(pred_subcats[i] == 1)[0])
        groups_i  = set(np.where(pred_groups[i]  == 1)[0])
        attrs_i   = set(np.where(pred_attrs[i]   == 1)[0])
        v1 = any(
            not subcat_parent_cats.get(si, set()).intersection(cats_i)
            for si in subcats_i
        )
        v2 = any(
            (gi := attr_parent_group.get(ai)) is not None and gi not in groups_i
            for ai in attrs_i
        )
        if v1 or v2:
            any_violation += 1

    return {
        "HVR":                 any_violation / N if N > 0 else 0.0,
        "HVR_subcat":          v1_images     / N if N > 0 else 0.0,
        "HVR_attr":            v2_images     / N if N > 0 else 0.0,
        "violation_subcat_pct": v1_count_total / pred_subcat_total
                                if pred_subcat_total > 0 else 0.0,
        "violation_attr_pct":   v2_count_total / pred_attr_total
                                if pred_attr_total > 0 else 0.0,
        "n_images": N,
    }



def compute_per_group_attr_f1(attr_true, attr_pred, attr_prob, hierarchy):
    """
    Compute Macro-F1, Micro-F1, and mAP for each attribute group separately.

    This is the most informative attribute metric because:
    - The "nickname" group (153 attrs) is visually ambiguous product names
      and has a structural ceiling of ~0.03–0.05 F1.
    - Non-nickname groups (silhouette, neckline type, textile pattern, etc.)
      are genuine visual properties and can reach 0.30–0.45 F1.
    - Reporting the overall 294-class Macro-F1 hides this distinction.

    Args:
        attr_true  : np.ndarray [N, 294]  ground truth
        attr_pred  : np.ndarray [N, 294]  binary predictions
        attr_prob  : np.ndarray [N, 294]  probabilities
        hierarchy  : FashionHierarchy

    Returns:
        list of dicts, one per group, with keys:
            group, n_attrs, Macro_F1, Micro_F1, mAP,
            Macro_Precision, Macro_Recall, n_pos_examples
    """
    import numpy as np

    results = []

    # Build group → column indices
    group_cols = {}
    for group_name, attr_ids in hierarchy.level_3.items():
        cols = []
        for aid in attr_ids:
            idx = hierarchy.attribute_id_to_idx.get(str(aid))
            if idx is not None:
                cols.append(idx)
        group_cols[group_name] = sorted(cols)

    for group_name, cols in group_cols.items():
        if not cols:
            continue
        col_idx  = np.array(cols)
        g_true   = attr_true[:, col_idx]
        g_pred   = attr_pred[:, col_idx]
        g_prob   = attr_prob[:, col_idx]
        n_pos    = int(g_true.sum())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mac_f1  = f1_score(g_true, g_pred, average="macro",  zero_division=0)
            mic_f1  = f1_score(g_true, g_pred, average="micro",  zero_division=0)
            mac_pre = precision_score(g_true, g_pred, average="macro",  zero_division=0)
            mac_rec = recall_score(g_true, g_pred, average="macro",  zero_division=0)
            map_val = _safe_map(g_true, g_prob)

        results.append({
            "group":           group_name,
            "n_attrs":         len(cols),
            "n_pos_examples":  n_pos,
            "Macro_F1":        round(mac_f1,  4),
            "Micro_F1":        round(mic_f1,  4),
            "mAP":             round(map_val, 4),
            "Macro_Precision": round(mac_pre, 4),
            "Macro_Recall":    round(mac_rec, 4),
        })

    # Add nickname vs non-nickname summary
    nick_cols     = group_cols.get("nickname", [])
    non_nick_cols = [c for g, cols in group_cols.items()
                     if g != "nickname" for c in cols]

    for label, cols in [("nickname_group", nick_cols),
                        ("non_nickname_groups", non_nick_cols)]:
        if not cols:
            continue
        ci     = np.array(cols)
        g_true = attr_true[:, ci]
        g_pred = attr_pred[:, ci]
        g_prob = attr_prob[:, ci]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append({
                "group":           label,
                "n_attrs":         len(cols),
                "n_pos_examples":  int(g_true.sum()),
                "Macro_F1":        round(f1_score(g_true, g_pred, average="macro", zero_division=0), 4),
                "Micro_F1":        round(f1_score(g_true, g_pred, average="micro", zero_division=0), 4),
                "mAP":             round(_safe_map(g_true, g_prob), 4),
                "Macro_Precision": round(precision_score(g_true, g_pred, average="macro", zero_division=0), 4),
                "Macro_Recall":    round(recall_score(g_true, g_pred, average="macro", zero_division=0), 4),
            })

    return results


def evaluate_hierarchical(model, dataloader, device, loss_fn,
                           save_path=None, threshold_config=None,
                           hierarchy=None):
    """
    Full evaluation for HierFashion-style models.

    Args:
        model             : trained Lightning model (eval mode)
        dataloader        : test dataloader
        device            : 'cuda' or 'cpu'
        loss_fn           : CompositeHierarchicalLoss instance
        save_path         : optional CSV path
        threshold_config  : dict returned by calibrate_thresholds()
                            If None, falls back to fixed defaults (0.5).
        hierarchy         : FashionHierarchy — required for hierarchical filter

    Pipeline (when threshold_config is supplied):
      1. Global optimised threshold  — categories / subcategories / attr_groups
      2. Per-class thresholds        — attributes
      3. Top-K filtering             — attributes
      4. Hierarchical consistency    — attributes masked by predicted categories
    """
    model.eval()
    model.to(device)

    keys        = ["categories", "subcategories", "attr_groups", "attributes"]
    all_logits  = {k: [] for k in keys}
    all_targets = {k: [] for k in keys}

    total_CHL = total_bce = total_cs = total_path = total_b = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            for k in targets:
                targets[k] = targets[k].to(device)

            predictions = model(images)

            CHL, L_bce, L_cs, L_path = loss_fn(
                predictions, targets, return_components=True
            )
            total_CHL  += CHL.item()
            total_bce  += L_bce.item()
            total_cs   += L_cs.item()
            total_path += L_path.item()
            total_b    += 1

            for k in keys:
                all_logits[k].append(predictions[k].detach().cpu())
                all_targets[k].append(targets[k].detach().cpu())

    rows           = []
    all_true_total = []
    all_pred_total = []
    all_prob_total = []

    # Collect group probs for hierarchical filter (attrs filtered by attr_groups)
    group_probs_test = None

    for key in keys:
        logits = torch.cat(all_logits[key]).numpy()
        true   = torch.cat(all_targets[key]).numpy()

        # ── Apply temperature scaling if available ───────────────
        if threshold_config is not None and "_temperatures" in threshold_config:
            T = threshold_config["_temperatures"].get(key, 1.0)
            probs = 1.0 / (1.0 + np.exp(-logits / T))
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))

        if key == "attr_groups":
            group_probs_test = probs.copy()

        # ── Apply thresholding strategy ──────────────────────────
        if threshold_config is None:
            # Legacy fallback — fixed thresholds (pre-calibration behaviour)
            fixed = {"categories": 0.40, "subcategories": 0.30,
                     "attr_groups": 0.30, "attributes": 0.25}
            preds = (probs > fixed[key]).astype(int)

        elif key != "attributes":
            # ── Technique 1: global optimised threshold ──────────
            t     = threshold_config[key]["threshold"]
            preds = (probs > t).astype(int)
            logger.info(f"[Eval] {key}: global threshold={t:.2f}")

        else:
            cfg = threshold_config["attributes"]

            # ── Technique 4: hierarchical consistency filter ─────
            # Filter attrs by their parent attr_groups (correct Fashionpedia
            # hierarchy branch). Must run on raw probs before thresholding.
            if hierarchy is not None and group_probs_test is not None:
                probs = _apply_hierarchical_filter(
                    probs, group_probs_test, hierarchy)
                logger.info("[Eval] attributes: hierarchical group filter applied")

            # ── Technique 2: per-class thresholds ────────────────
            per_class_t = cfg["per_class_t"]
            preds       = (probs > per_class_t[np.newaxis, :]).astype(int)

            # ── Technique 3: top-K cap (uses K found during calibration) ──
            best_k = cfg["top_k"]
            for i in range(preds.shape[0]):
                pos_idx = np.where(preds[i] == 1)[0]
                if len(pos_idx) > best_k:
                    top_idx = pos_idx[np.argsort(probs[i, pos_idx])[-best_k:]]
                    preds[i] = 0
                    preds[i, top_idx] = 1

            logger.info(
                f"[Eval] attributes: per-class thresholds + "
                f"top-{best_k} filter applied"
            )

        all_true_total.append(true)
        all_pred_total.append(preds)
        all_prob_total.append(probs)

        row = {
            "Head":             key,
            "Macro_F1":         f1_score(true, preds, average="macro",  zero_division=0),
            "Micro_F1":         f1_score(true, preds, average="micro",  zero_division=0),
            "Macro_Precision":  precision_score(true, preds, average="macro",  zero_division=0),
            "Micro_Precision":  precision_score(true, preds, average="micro",  zero_division=0),
            "Macro_Recall":     recall_score(true, preds, average="macro",     zero_division=0),
            "Micro_Recall":     recall_score(true, preds, average="micro",     zero_division=0),
            "mAP":              _safe_map(true, probs),
            "L_bce": None, "L_consistency": None, "L_path": None, "CHL": None,
        }
        rows.append(row)
        _print_metrics(key, row)

    # Composite loss row
    nb = total_b
    rows.append({
        "Head": "CompositeLoss",
        "Macro_F1": None, "Micro_F1": None,
        "Macro_Precision": None, "Micro_Precision": None,
        "Macro_Recall": None, "Micro_Recall": None, "mAP": None,
        "L_bce": total_bce / nb, "L_consistency": total_cs / nb,
        "L_path": total_path / nb, "CHL": total_CHL / nb,
    })

    # Overall row
    y_true = np.concatenate(all_true_total, axis=1)
    y_pred = np.concatenate(all_pred_total, axis=1)
    y_prob = np.concatenate(all_prob_total, axis=1)

    rows.append({
        "Head":             "OverallModel",
        "Macro_F1":         f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "Micro_F1":         f1_score(y_true, y_pred, average="micro",  zero_division=0),
        "Macro_Precision":  precision_score(y_true, y_pred, average="macro",  zero_division=0),
        "Micro_Precision":  precision_score(y_true, y_pred, average="micro",  zero_division=0),
        "Macro_Recall":     recall_score(y_true, y_pred, average="macro",     zero_division=0),
        "Micro_Recall":     recall_score(y_true, y_pred, average="micro",     zero_division=0),
        "mAP":              _safe_map(y_true, y_prob),
        "L_bce": None, "L_consistency": None, "L_path": None, "CHL": None,
    })

    # ── Per-group attribute F1 ──────────────────────────────────
    if hierarchy is not None:
        attr_true_arr = all_true_total[keys.index("attributes")]
        attr_pred_arr = all_pred_total[keys.index("attributes")]
        attr_prob_arr = all_prob_total[keys.index("attributes")]

        group_results = compute_per_group_attr_f1(
            attr_true_arr, attr_pred_arr, attr_prob_arr, hierarchy)

        print("\n--- ATTRIBUTE F1 BY GROUP ---")
        print(f"  {'Group':50s}  {'n_attrs':>7}  {'Macro_F1':>8}  {'Micro_F1':>8}  {'mAP':>7}")
        print("  " + "─" * 86)

        for gr in group_results:
            marker = "  "
            if gr["group"] == "nickname_group":
                print("  " + "─" * 86)
                marker = "* "  # flag nickname as special
            elif gr["group"] == "non_nickname_groups":
                marker = "* "
            print(f"  {marker}{gr['group']:48s}  {gr['n_attrs']:>7}  "
                  f"{gr['Macro_F1']:>8.4f}  {gr['Micro_F1']:>8.4f}  {gr['mAP']:>7.4f}")

        print("  (* = summary rows — nickname vs non-nickname)")

        # Append per-group rows to dataframe
        for gr in group_results:
            rows.append({
                "Head":             f"attr_group_{gr['group'].replace(' ','_').replace(',','').replace('-','_')}",
                "Macro_F1":         gr["Macro_F1"],
                "Micro_F1":         gr["Micro_F1"],
                "Macro_Precision":  gr["Macro_Precision"],
                "Micro_Precision":  None,
                "Macro_Recall":     gr["Macro_Recall"],
                "Micro_Recall":     None,
                "mAP":              gr["mAP"],
                "L_bce":            gr["n_attrs"],
                "L_consistency":    gr["n_pos_examples"],
                "L_path":           None,
                "CHL":              None,
            })

    # ── HVR — Hierarchical Violation Rate ───────────────────────
    if hierarchy is not None:
        pred_cats_arr    = all_pred_total[keys.index("categories")]
        pred_subcats_arr = all_pred_total[keys.index("subcategories")]
        pred_groups_arr  = all_pred_total[keys.index("attr_groups")]
        pred_attrs_arr   = all_pred_total[keys.index("attributes")]

        hvr = compute_hvr(
            pred_cats_arr, pred_subcats_arr,
            pred_groups_arr, pred_attrs_arr,
            hierarchy,
        )

        print("\n--- HIERARCHICAL VIOLATION RATE (HVR) ---")
        print(f"  HVR (any violation)       : {hvr['HVR']:.4f}  "
              f"({hvr['HVR']*100:.1f}% of images have >=1 violation)")
        print(f"  HVR_subcat (V1)           : {hvr['HVR_subcat']:.4f}  "
              f"(subcat predicted without parent category)")
        print(f"  HVR_attr   (V2)           : {hvr['HVR_attr']:.4f}  "
              f"(attr predicted without parent attr_group)")
        print(f"  Violation rate per subcat : {hvr['violation_subcat_pct']:.4f}  "
              f"({hvr['violation_subcat_pct']*100:.1f}% of predicted subcats are violations)")
        print(f"  Violation rate per attr   : {hvr['violation_attr_pct']:.4f}  "
              f"({hvr['violation_attr_pct']*100:.1f}% of predicted attrs are violations)")
        print(f"  (evaluated on {hvr['n_images']:,} images)")

        # Append HVR row to results dataframe
        rows.append({
            "Head":             "HVR",
            "Macro_F1":         None,
            "Micro_F1":         None,
            "Macro_Precision":  None,
            "Micro_Precision":  None,
            "Macro_Recall":     None,
            "Micro_Recall":     None,
            "mAP":              None,
            "L_bce":            hvr["HVR"],
            "L_consistency":    hvr["HVR_subcat"],
            "L_path":           hvr["HVR_attr"],
            "CHL":              hvr["violation_attr_pct"],
        })

    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Results saved → {save_path}")
    return df


def evaluate_flat(model, dataloader, device, loss_fn, hierarchy,
                  head="attributes", save_path=None):
    """
    Evaluation for flat baselines that output a single logit vector
    of size num_attributes (or num_nodes).
    """
    model.eval()
    model.to(device)

    all_logits  = []
    all_targets = []
    total_loss  = 0
    total_b     = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            tgt    = targets[head].to(device)

            logits = model(images)
            loss   = loss_fn(logits, tgt.float())
            total_loss += loss.item()
            total_b    += 1

            all_logits.append(logits.detach().cpu())
            all_targets.append(tgt.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    true   = torch.cat(all_targets).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs > 0.25).astype(int)

    row = {
        "Head":             head,
        "Macro_F1":         f1_score(true, preds, average="macro",  zero_division=0),
        "Micro_F1":         f1_score(true, preds, average="micro",  zero_division=0),
        "Macro_Precision":  precision_score(true, preds, average="macro",  zero_division=0),
        "Micro_Precision":  precision_score(true, preds, average="micro",  zero_division=0),
        "Macro_Recall":     recall_score(true, preds, average="macro",     zero_division=0),
        "Micro_Recall":     recall_score(true, preds, average="micro",     zero_division=0),
        "mAP":              _safe_map(true, probs),
        "L_bce": total_loss / total_b,
        "L_consistency": None, "L_path": None, "CHL": None,
    }

    df = pd.DataFrame([row])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Results saved → {save_path}")
    _print_metrics(head, row)
    return df


def _safe_map(true, probs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return average_precision_score(true, probs, average="macro")
    except Exception:
        return 0.0


def _print_metrics(key, row):
    print(f"\n--- {key.upper()} ---")
    for k, v in row.items():
        if k != "Head" and v is not None:
            print(f"  {k:22s}: {v:.4f}")


# ============================================================
# MODEL COMPLEXITY
# ============================================================

def analyze_model(model, device, input_size=(1, 3, 224, 224)):
    try:
        from thop import profile, clever_format
    except ImportError:
        print("thop not installed — skipping FLOPs")
        return

    model.eval()
    model.to(device)
    dummy = torch.randn(*input_size).to(device)

    params     = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_fmt, _ = clever_format([flops, params], "%.3f")
        # Remove thop-injected buffers (total_ops, total_params) from every
        # module so they are NOT saved into the Lightning checkpoint and do
        # not cause "Unexpected key(s)" errors on load_from_checkpoint.
        for mod in model.modules():
            mod._buffers.pop("total_ops", None)
            mod._buffers.pop("total_params", None)
    except Exception:
        flops_fmt = "N/A"

    timings = []
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        for _ in range(100):
            t0 = time.time()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.time() - t0) * 1000)

    mean_ms    = float(np.mean(timings))
    throughput = 1000 / mean_ms

    print("\n==============================")
    print("  MODEL COMPLEXITY ANALYSIS")
    print("==============================")
    print(f"  Total params     : {params:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  FLOPs            : {flops_fmt}")
    print(f"  Inference time   : {mean_ms:.3f} ms/image")
    print(f"  Throughput       : {throughput:.2f} img/s")

    # ✅ Restore train mode so Lightning trainer starts correctly
    model.train()

    return params, trainable, flops_fmt, mean_ms, throughput


# ============================================================
# MULTI-SEED RUNNER — for statistical significance
# Run any model 3 times with different seeds, report mean ± std
# Usage: from core.common import run_with_seeds
# ============================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_with_seeds(build_model_fn, train_fn, eval_fn, seeds=(42, 123, 7)):
    """
    Run a model with multiple seeds and report mean ± std.

    Args:
        build_model_fn : callable() → Lightning model
        train_fn       : callable(model) → trained model
        eval_fn        : callable(model) → pd.DataFrame of metrics
        seeds          : tuple of ints

    Returns:
        summary DataFrame with mean ± std per metric
    """
    import pandas as pd
    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        set_seed(seed)
        model   = build_model_fn()
        model   = train_fn(model)
        df      = eval_fn(model)
        overall = df[df["Head"] == "OverallModel"].iloc[0].to_dict()
        overall["seed"] = seed
        all_results.append(overall)

    results_df = pd.DataFrame(all_results)
    metrics    = ["Macro_F1", "Micro_F1", "Macro_Precision",
                  "Micro_Precision", "Macro_Recall", "Micro_Recall", "mAP"]

    print("\n" + "="*60)
    print("MULTI-SEED RESULTS (OverallModel)")
    print("="*60)
    summary = {}
    for m in metrics:
        vals = results_df[m].dropna().values
        if len(vals) > 0:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[m] = f"{mean:.4f} ± {std:.4f}"
            print(f"  {m:22s}: {mean:.4f} ± {std:.4f}")

    return results_df, summary
