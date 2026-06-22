# HierFashion — Hierarchical Multi-Label Fashion Attribute Recognition

A 4-level hierarchical multi-label classification system for the **Fashionpedia** dataset.

---

## Architecture

```
Input Image
    │
    ▼
ResNet50 Backbone → Feature Maps
    │                     │
    │          Label-Guided Spatial Attention ◄── GAT refined label repr
    │                     │
    │              Attended Visual Features
    │                     │
    ▼                     ▼
            Cross-Attention Fusion Module
                          │
              ┌───────────┼───────────┬────────────┐
              ▼           ▼           ▼            ▼
         Category    Subcategory  Attr Group   Attribute
           Head         Head        Head         Head
              └───────────┴───────────┴────────────┘
                                │
                       Composite Hierarchical Loss
                    (BCE + Consistency + Path Coherence)
```

---

## Project Structure

```
HierFashion/
├── final.py                        # Main HierFashion model (full architecture)
├── run_all_experiments.py          # Train + evaluate ALL models in one script
│
├── core/
│   └── common.py                   # Shared config, dataset, backbone, loss, evaluation
│
├── models/                         # Baseline comparison models
│   ├── cnn_flat.py                 # B1: ResNet50 + FC (no hierarchy)
│   ├── cnn_hier.py                 # B2: ResNet50 + hierarchical loss only
│   ├── ml_gcn.py                   # B3: ML-GCN (Chen et al., CVPR 2019)
│   ├── cnn_gat.py                  # B4: ResNet50 + real GAT (no spatial attn)
│   ├── resnet_asl.py               # B5: ResNet50 + Asymmetric Loss
│   ├── vit_mlc.py                  # B6: ViT-B/16 + hierarchical heads
│   ├── q2l.py                      # B7: Query2Label (Liu et al., NeurIPS 2021)
│   └── tresnet.py                  # B8: TResNet (multi-scale pooling)
│
├── ablation/                       # Ablation study models
│   ├── ablation_no_gat.py          # A1: Replace GAT with MLP
│   ├── ablation_no_cross_attn.py   # A2: Replace cross-attention with simple concat
│   ├── ablation_no_hier_loss.py    # A3: Replace hierarchical loss with plain BCE
│   └── ablation_no_label_guided_attn.py  # A4: Replace label-guided attn with GAP
│
├── hierarchy_outputs/
│   ├── fashionpedia_hierarchy.json
│   ├── fashion_hierarchy.json
│   ├── fashionpedia_adjacency.npy
│   └── hierarchy_statistics.csv
│
├── results/                        # Auto-created; all CSV evaluation results
├── training_logs/                  # Lightning CSV logs per model
└── logs/                           # experiment_log.txt
```

---

## Hierarchy

| Level | Name             | Count |
|-------|------------------|-------|
| 1     | Categories       | 46    |
| 2     | Subcategories    | 12    |
| 3     | Attribute Groups | 11    |
| 4     | Attributes       | 294   |
| —     | Total nodes      | 363   |

---

## Training Parameters (identical across ALL models)

| Parameter          | Value              |
|--------------------|--------------------|
| Optimizer          | AdamW              |
| Learning rate      | 5e-5               |
| Weight decay       | 1e-4               |
| LR scheduler       | CosineAnnealingLR  |
| Batch size         | 64                 |
| Max epochs         | 25                 |
| Early stopping     | patience=5, val_loss |
| Gradient clipping  | 1.0                |
| Precision          | 16-mixed (GPU)     |
| Image size         | 224×224            |

---

## Evaluation Metrics (identical across ALL models)

Per head (categories, subcategories, attr_groups, attributes):
- Macro F1, Micro F1
- Macro Precision, Micro Precision
- Macro Recall, Micro Recall
- mAP (mean Average Precision)

Plus:
- Composite Loss breakdown: L_bce, L_consistency, L_path, CHL
- OverallModel: all metrics across all heads combined

Adaptive threshold search for `attributes` head (0.05–0.50).
Threshold for `categories` = 0.30, others = 0.25.

---

## Quick Start

### Train only the main model
```bash
python final.py
```

### Train a single baseline
```bash
python models/cnn_gat.py
python models/vit_mlc.py
# etc.
```

### Train a single ablation
```bash
python ablation/ablation_no_gat.py
```

### Train ALL models + generate comparison table
```bash
python run_all_experiments.py
```
Results saved to `results/all_results_comparison.csv`.

---

## Ablation Study Design

### Part 1 — Individual Component Removal (from full model)
Remove exactly ONE component at a time. Everything else stays identical to `final.py`.

| ID | File | What is removed | Replaced with | Tests |
|----|------|-----------------|---------------|-------|
| A1 | `ablation_no_label_guided_attn.py` | Label-Guided Spatial Attention | Global Avg Pool | Spatial attention contribution |
| A2 | `ablation_no_gat.py` | GAT (2-layer) | 2-layer MLP | Graph attention contribution |
| A3 | `ablation_no_cross_attn.py` | Cross-Attention Fusion | Simple mean concat | Cross-attention fusion contribution |
| A4 | `ablation_no_hier_loss.py` | Full Hierarchical Loss | Plain multi-head BCE | All hierarchy-aware loss terms |
| A5 | `ablation_no_consistency.py` | Consistency term (L_cs) only | Removed from CHL | Parent-child consistency loss |
| A6 | `ablation_no_path_coherence.py` | Path coherence term (L_path) only | Removed from CHL | End-to-end path coherence loss |
| A7 | `ablation_no_hier_mask.py` | Hierarchical masking on attrs | No mask applied | Category-guided attribute masking |

### Part 2 — Incremental Build-Up (add components one by one)
Start from the simplest model, add one component at a time until full model.

| ID | File | Components present | Δ over previous |
|----|------|--------------------|-----------------|
| B1 | `models/cnn_hier.py` | ResNet50 + 4 heads + BCE | baseline |
| B2 | `incremental_b2_gat.py` | B1 + GAT label graph | +GAT |
| B3 | `incremental_b3_lgsa.py` | B2 + Label-Guided Spatial Attn | +LGSA |
| B4 | `incremental_b4_crossattn.py` | B3 + Cross-Attention Fusion | +CrossAttn |
| B5 | `final.py` | B4 + Hierarchical Loss + HierMask | +CHL (full model) |

**Total ablation rows in paper table: 10 variants + full model = 11 rows**

---

## Dependencies

```
torch >= 2.0
torchvision
pytorch-lightning >= 2.0
torch-geometric
scikit-learn
pandas
numpy
Pillow
thop
```

Install:
```bash
pip install torch torchvision pytorch-lightning torch-geometric scikit-learn pandas numpy Pillow thop
```

---

## Dataset

Fashionpedia: https://fashionpedia.github.io/home/
Expected layout:
```
dataset/fashionpedia/
    train2020/
    val2020/
    instances_attributes_train2020.json
    instances_attributes_val2020.json
```
Update `DATASET_ROOT` in `core/common.py` if needed.
