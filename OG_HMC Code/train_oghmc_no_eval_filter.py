# ============================================================
# experiments/train_oghmc_no_eval_filter.py
#
# REVIEWER-RESPONSE EXPERIMENT — companion training script
# --------------------------------------------------------------
# A reviewer asked what happens to OG-HMC's metrics if the
# evaluation-time hierarchical filter, `_apply_hierarchical_filter()`
# (defined in core/common.py), is disabled at test time.
#
# To answer that cleanly we need a checkpoint of the *exact*,
# *unmodified* full OG-HMC model (the same architecture, the same
# loss, the same data pipeline as the paper's main model in
# `final.py`). This script produces that checkpoint.
#
# WHAT THIS SCRIPT DOES
#   - Trains the standard, full OG-HMC model (`HierFashionLightning`,
#     imported unmodified from the repo's main training script)
#     with no changes to architecture, loss, optimizer, schedule,
#     or data.
#   - Saves checkpoints normally (same `ModelCheckpoint` callback
#     pattern used everywhere else in the repo).
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
#   - It does NOT touch `_apply_hierarchical_filter()`.
#   - It does NOT change calibration, thresholds, or evaluation.
#     All of that happens in the companion script
#     `evaluate_oghmc_no_eval_filter.py`, which loads the checkpoint
#     produced here and runs the evaluation-time ablation.
#
# WHY A NEW SCRIPT INSTEAD OF JUST RE-USING `final.py` DIRECTLY
#   - `final.py` is a fine way to train this exact model, and you
#     are welcome to point the evaluation script at any checkpoint
#     it produces. This script exists only so the reviewer-response
#     experiment has its own clearly-labelled checkpoint directory
#     (`checkpoints/oghmc_no_eval_filter/seed<seed>/`) that will not
#     be confused with or overwritten by other runs, and so the
#     whole experiment can be reproduced with two commands.
#   - No existing repository file is modified. This file only
#     *imports* the existing model/loss/data code.
#
# REPOSITORY COMPATIBILITY NOTE (please read)
#   While wiring this up we found that, in the exact code bundle we
#   were given, the main script's import line
#       from core.common import (..., build_hierarchy_constraint_matrices, ...)
#   refers to a function that is NOT defined anywhere in
#   `core/common.py` in this bundle, and `CompositeHierarchicalLoss`
#   in this bundle's `core/common.py` does not accept the
#   `subcat_to_cat_matrix` / `attr_to_group_matrix` keyword arguments
#   that the main model passes to it. This is a pre-existing
#   inconsistency between the shipped `final.py` and the shipped
#   `core/common.py` — it would block `python final.py` itself,
#   completely independent of this experiment. We have NOT patched
#   `core/common.py` or `final.py` (we were told not to modify
#   existing files). Instead:
#     - If `build_hierarchy_constraint_matrices` is missing, this
#       script installs a small, deterministic, in-memory shim
#       (defined only in this file) that builds the two constraint
#       matrices from the hierarchy in the obvious way. No file on
#       disk is changed.
#     - If `CompositeHierarchicalLoss` does not accept the
#       constraint-matrix kwargs, that is a deeper mismatch we
#       cannot safely shim without changing loss behaviour, so this
#       script fails fast with a clear message instead of guessing.
#   If your real environment's `core/common.py` already defines
#   `build_hierarchy_constraint_matrices` and an updated
#   `CompositeHierarchicalLoss`, none of this matters: the shim is a
#   no-op and the script behaves exactly like `final.py`.
#
# USAGE
#   python experiments/train_oghmc_no_eval_filter.py
#   python experiments/train_oghmc_no_eval_filter.py --seed 123
#   python experiments/train_oghmc_no_eval_filter.py --seed 42 --epochs 25
#
# OUTPUT
#   checkpoints/oghmc_no_eval_filter/seed<seed>/*.ckpt
#   checkpoints/oghmc_no_eval_filter/seed<seed>/run_info.json
#   training_logs/OGHMC_NoEvalFilter_seed<seed>/...
# ============================================================

import os
import sys
import math
import json
import glob
import inspect
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

# ----------------------------------------------------------------
# Make the repository root importable (this file lives in
# <repo_root>/experiments/, so the repo root is one level up).
# ----------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core.common as core_common  # noqa: E402  (reuse, unmodified)
from core.common import (  # noqa: E402  (reuse, unmodified)
    compute_pos_weights, analyze_model, get_logger,
    load_hierarchy, build_adjacency_matrix, set_seed,
    BATCH_SIZE, EPOCHS, PATIENCE,
)

logger = get_logger("train_oghmc_no_eval_filter")


# ============================================================
# Compatibility shim — see header note above.
# This does NOT modify any file on disk; it only adds a missing
# attribute to the already-imported `core.common` module object
# in memory, for the duration of this process, and only if the
# real function isn't already there.
# ============================================================
def _shim_build_hierarchy_constraint_matrices(hierarchy):
    """
    Deterministic fallback for `core.common.build_hierarchy_constraint_matrices`.

    subcat_to_cat_matrix : [num_subcategories, num_categories]
        1 if the category is a valid parent of the subcategory
        (derived from hierarchy.level_2, the same source used
        everywhere else in this repo for subcat->category parentage).

    attr_to_group_matrix : [num_attributes, num_attr_groups]
        1 if the attribute belongs to that attribute group
        (derived from hierarchy.level_3, the same source used by
        `_apply_hierarchical_filter` and `build_group_attribute_mask`
        elsewhere in this repo).
    """
    subcat_to_cat = torch.zeros(hierarchy.num_subcategories, hierarchy.num_categories)
    for subcat_name, cat_ids in hierarchy.level_2.items():
        s_idx = hierarchy.subcategory_id_to_idx.get(subcat_name)
        if s_idx is None:
            continue
        for cid in cat_ids:
            c_idx = hierarchy.category_id_to_idx.get(str(cid))
            if c_idx is not None:
                subcat_to_cat[s_idx, c_idx] = 1.0

    attr_to_group = torch.zeros(hierarchy.num_attributes, hierarchy.num_attr_groups)
    for group_name, attr_ids in hierarchy.level_3.items():
        g_idx = hierarchy.group_id_to_idx.get(group_name)
        if g_idx is None:
            continue
        for aid in attr_ids:
            a_idx = hierarchy.attribute_id_to_idx.get(str(aid))
            if a_idx is not None:
                attr_to_group[a_idx, g_idx] = 1.0

    return subcat_to_cat, attr_to_group


def _ensure_constraint_matrix_builder(core_common_module):
    """Install the shim only if the real function is missing. No-op otherwise."""
    if not hasattr(core_common_module, "build_hierarchy_constraint_matrices"):
        logger.warning(
            "[compat-shim] core.common.build_hierarchy_constraint_matrices "
            "is missing in this environment. Installing an in-memory "
            "fallback (no files on disk are modified). See header comment "
            "in this script for details."
        )
        core_common_module.build_hierarchy_constraint_matrices = (
            _shim_build_hierarchy_constraint_matrices
        )


def _check_loss_accepts_constraint_matrices(core_common_module):
    """
    Fail fast (with a clear, actionable message) if the loss class in
    this environment cannot accept the constraint-matrix kwargs the
    main model passes to it. We deliberately do NOT patch the loss
    class itself — that would risk silently changing loss behaviour,
    which would violate "reuse ... losses ... unchanged".
    """
    sig = inspect.signature(core_common_module.CompositeHierarchicalLoss.__init__)
    missing = [
        p for p in ("subcat_to_cat_matrix", "attr_to_group_matrix")
        if p not in sig.parameters
    ]
    if missing:
        raise RuntimeError(
            "Pre-existing repository inconsistency detected (not caused by "
            "this experiment): core.common.CompositeHierarchicalLoss.__init__ "
            f"does not accept {missing}, but the main OG-HMC model "
            "(final.py / HierFashionLightning) constructs the loss with "
            "those keyword arguments. This would also block `python "
            "final.py` itself. Please update core/common.py in your "
            "environment so CompositeHierarchicalLoss accepts these "
            "arguments (we have not done so here, since we were asked not "
            "to modify existing files)."
        )


def _load_oghmc_main_module():
    """
    Dynamically load the repo's main OG-HMC script. The README calls
    it `final.py`; in this exact bundle the file on disk is named
    `final (4).py` (a duplicate-upload artifact), so we try the clean
    name first and fall back to a glob match at the repo root.
    Only existing files are read — nothing is written or modified.
    """
    candidate = os.path.join(REPO_ROOT, "final.py")
    if not os.path.exists(candidate):
        matches = sorted(glob.glob(os.path.join(REPO_ROOT, "final*.py")))
        if not matches:
            raise FileNotFoundError(
                f"Could not find the main OG-HMC script ('final.py') under "
                f"{REPO_ROOT}"
            )
        candidate = matches[0]
        logger.info(f"Using main OG-HMC script: {candidate}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("oghmc_main", candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules["oghmc_main"] = module
    spec.loader.exec_module(module)  # runs only module-level code; the
    # `if __name__ == "__main__":` block inside final.py does NOT run,
    # because `module.__name__` is "oghmc_main", not "__main__".
    return module, candidate


def main():
    parser = argparse.ArgumentParser(
        description="Train the standard, unmodified OG-HMC model "
                     "(checkpoint for the eval-time hierarchical-filter "
                     "ablation experiment)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override EPOCHS from core/common.py (default: "
                              "use the repo's standard value, unchanged).")
    parser.add_argument("--output-dir", type=str,
                         default="checkpoints/oghmc_no_eval_filter")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs or EPOCHS

    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    _ensure_constraint_matrix_builder(core_common)
    oghmc_main, main_path = _load_oghmc_main_module()
    _check_loss_accepts_constraint_matrices(core_common)

    HierFashionLightning = oghmc_main.HierFashionLightning
    FashionDataModule    = oghmc_main.FashionDataModule
    LR_IMPROVED           = oghmc_main.LR_IMPROVED
    logger.info(f"Loaded OG-HMC model classes from: {main_path}")

    # ── Hierarchy / graph structures — unmodified, reused as-is ────
    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix = build_adjacency_matrix(hierarchy, hierarchy_json)
    subcat_to_cat_matrix, attr_to_group_matrix = \
        core_common.build_hierarchy_constraint_matrices(hierarchy)

    # ── Data — unmodified, reused as-is (same 80/20 split logic, same
    #     transforms, same held-out test set as every other model) ──
    data_module = FashionDataModule(hierarchy=hierarchy,
                                     batch_size=BATCH_SIZE, seed=seed)
    data_module.setup()

    pos_weight = compute_pos_weights(
        data_module.train_dataloader(), hierarchy, device)
    steps_per_epoch = math.ceil(len(data_module.train_subset) / BATCH_SIZE)

    # ── Model — the standard, full OG-HMC model, unmodified ────────
    model = HierFashionLightning(
        hierarchy=hierarchy, adj_matrix=adj_matrix,
        pos_weight=pos_weight, learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_per_epoch,
        subcat_to_cat_matrix=subcat_to_cat_matrix,
        attr_to_group_matrix=attr_to_group_matrix,
    )
    analyze_model(model, device)

    ckpt_dir = os.path.join(args.output_dir, f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath=ckpt_dir,
        filename=f"oghmc-no-eval-filter-seed{seed}-{{epoch:02d}}-{{val_loss:.4f}}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            ckpt_cb,
            EarlyStopping(monitor="val_loss", mode="min",
                          patience=PATIENCE, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=CSVLogger("training_logs", name=f"OGHMC_NoEvalFilter_seed{seed}"),
    )

    logger.info("Training the standard, full OG-HMC model (unchanged) ...")
    trainer.fit(model, datamodule=data_module)

    best_ckpt = ckpt_cb.best_model_path
    logger.info(f"Best checkpoint: {best_ckpt}")

    # Pointer file so the companion evaluation script can auto-discover
    # this checkpoint and reconstruct an identical data split/model.
    run_info = {
        "seed": seed,
        "best_checkpoint": best_ckpt,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": BATCH_SIZE,
        "epochs_requested": epochs,
        "main_script_used": main_path,
    }
    with open(os.path.join(ckpt_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — standard, unmodified OG-HMC model")
    print("=" * 60)
    print(f"  Seed              : {seed}")
    print(f"  Best checkpoint    : {best_ckpt}")
    print(f"  run_info.json      : {os.path.join(ckpt_dir, 'run_info.json')}")
    print("\n  Next step:")
    print(f"    python experiments/evaluate_oghmc_no_eval_filter.py --seed {seed}")


if __name__ == "__main__":
    main()
