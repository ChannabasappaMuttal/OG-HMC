# ============================================================
# experiments/evaluate_oghmc_no_eval_filter.py
#
# REVIEWER-RESPONSE EXPERIMENT
# --------------------------------------------------------------
# Reviewer concern: "How much of OG-HMC's reported hierarchical
# consistency / performance comes from the evaluation-time
# hierarchical filter (`_apply_hierarchical_filter()` in
# core/common.py), as opposed to the model itself?"
#
# This script answers that with a single-variable ablation:
#   - Load a trained OG-HMC checkpoint (e.g. produced by
#     experiments/train_oghmc_no_eval_filter.py, or any other
#     checkpoint of the same model).
#   - Run the *normal* evaluation pipeline (temperature scaling,
#     global thresholds, per-class thresholds, top-K capping —
#     all unmodified, all reused from core/common.py) ...
#   - ... but with ONLY the `_apply_hierarchical_filter()` step
#     removed from the attribute branch at TEST time.
#   - Report Micro-F1, Macro-F1, mAP, and the three HVR variants,
#     and save a side-by-side comparison against the standard
#     (filter-ON) evaluation of the SAME checkpoint, so the only
#     thing that differs between the two result rows is whether
#     the filter ran.
#
# HOW THE FILTER IS DISABLED WITHOUT EDITING core/common.py
#   We were told not to edit core/common.py. Editing the shared
#   `evaluate_hierarchical()` function in place is exactly the kind
#   of change that should not happen there (other scripts —
#   final.py, run_all_experiments.py, compute_hvr.py, every
#   ablation/baseline — all call it and must keep their existing
#   behaviour). Monkey-patching `_apply_hierarchical_filter` to a
#   no-op globally would be invisible/implicit and could leak into
#   other code paths that import core.common in the same process.
#   Instead, this file defines `evaluate_hierarchical_no_filter()`,
#   a COPY of `core.common.evaluate_hierarchical()` with exactly one
#   change: the call to `_apply_hierarchical_filter()` in the
#   attributes branch is skipped. Every other line — temperature
#   scaling, per-class thresholds, top-K capping, metric computation,
#   per-group attribute F1, HVR — is the same code, and it calls the
#   same shared helper functions from core.common (compute_hvr,
#   compute_per_group_attr_f1, etc.) rather than re-implementing them.
#   The diff is marked inline with "REVIEWER EXPERIMENT CHANGE".
#
# USAGE
#   # Auto-discover the checkpoint trained by the companion script:
#   python experiments/evaluate_oghmc_no_eval_filter.py --seed 42
#
#   # Or point at any OG-HMC checkpoint explicitly:
#   python experiments/evaluate_oghmc_no_eval_filter.py \
#       --ckpt checkpoints/HierFashion_seed42/hierfashion-seed42-best-....ckpt \
#       --seed 42
#
# OUTPUT (under --output-dir, default "results/")
#   oghmc_WITH_eval_filter_seed<seed>_test_results.csv   (standard, unmodified)
#   oghmc_NO_eval_filter_seed<seed>_test_results.csv     (filter disabled)
#   reviewer_response_eval_filter_ablation_seed<seed>_summary.csv
#       -> Micro-F1, Macro-F1, mAP, HVRattr, HVR_subcat, HVR_any
#          for both variants, side by side.
# ============================================================

import os
import sys
import math
import json
import glob
import inspect
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core.common as core_common  # noqa: E402  (reuse, unmodified)
from core.common import (  # noqa: E402  (reuse, unmodified)
    get_logger, load_hierarchy, build_adjacency_matrix, set_seed,
    compute_pos_weights, calibrate_thresholds, evaluate_hierarchical,
    compute_hvr, compute_per_group_attr_f1, _safe_map, _print_metrics,
    BATCH_SIZE,
)

logger = get_logger("evaluate_oghmc_no_eval_filter")


# ============================================================
# Same compatibility shim as the training script. No-op if your
# environment's core/common.py already defines this function.
# See the header note in experiments/train_oghmc_no_eval_filter.py
# for the full explanation. No file on disk is modified.
# ============================================================
def _shim_build_hierarchy_constraint_matrices(hierarchy):
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
    if not hasattr(core_common_module, "build_hierarchy_constraint_matrices"):
        logger.warning(
            "[compat-shim] core.common.build_hierarchy_constraint_matrices "
            "is missing in this environment. Installing an in-memory "
            "fallback (no files on disk are modified)."
        )
        core_common_module.build_hierarchy_constraint_matrices = (
            _shim_build_hierarchy_constraint_matrices
        )


def _check_loss_accepts_constraint_matrices(core_common_module):
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
    spec.loader.exec_module(module)
    return module, candidate


def _find_checkpoint(seed, search_dir="checkpoints/oghmc_no_eval_filter"):
    """Auto-discover the checkpoint trained by the companion training
    script for this seed. Falls back to globbing for any .ckpt in the
    seed directory if run_info.json isn't present."""
    seed_dir = os.path.join(search_dir, f"seed{seed}")
    info_path = os.path.join(seed_dir, "run_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        ckpt = info.get("best_checkpoint")
        if ckpt and os.path.exists(ckpt):
            return ckpt
    matches = sorted(glob.glob(os.path.join(seed_dir, "*.ckpt")),
                      key=os.path.getmtime)
    if matches:
        return matches[-1]
    return None


# ============================================================
# COPIED EVALUATION FUNCTION — single change from
# core.common.evaluate_hierarchical(), marked inline below.
#
# This is a near-verbatim copy of core/common.py's
# `evaluate_hierarchical()`. It reuses the SAME shared helpers
# (compute_hvr, compute_per_group_attr_f1, _safe_map, _print_metrics)
# imported from core.common, unmodified. core/common.py itself is
# never edited.
# ============================================================
def evaluate_hierarchical_no_filter(model, dataloader, device, loss_fn,
                                     save_path=None, threshold_config=None,
                                     hierarchy=None):
    """
    Identical to core.common.evaluate_hierarchical(), EXCEPT that the
    evaluation-time hierarchical consistency filter
    (`_apply_hierarchical_filter`, which zeroes out attribute
    probabilities whose parent attribute-group was not predicted) is
    NOT applied. Temperature scaling, per-class thresholds, and the
    top-K cap are still applied exactly as in the standard pipeline —
    only the hierarchical filter step is removed, so this isolates
    its effect from everything else in calibration/evaluation.
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

    # Collect group probs — kept for parity with the standard function
    # (and for debugging), even though we no longer feed them into a
    # filter step below.
    group_probs_test = None

    for key in keys:
        logits = torch.cat(all_logits[key]).numpy()
        true   = torch.cat(all_targets[key]).numpy()

        if threshold_config is not None and "_temperatures" in threshold_config:
            T = threshold_config["_temperatures"].get(key, 1.0)
            probs = 1.0 / (1.0 + np.exp(-logits / T))
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))

        if key == "attr_groups":
            group_probs_test = probs.copy()

        if threshold_config is None:
            fixed = {"categories": 0.40, "subcategories": 0.30,
                     "attr_groups": 0.30, "attributes": 0.25}
            preds = (probs > fixed[key]).astype(int)

        elif key != "attributes":
            t     = threshold_config[key]["threshold"]
            preds = (probs > t).astype(int)
            logger.info(f"[Eval-NoFilter] {key}: global threshold={t:.2f}")

        else:
            cfg = threshold_config["attributes"]

            # ──────────────────────────────────────────────────────
            # REVIEWER EXPERIMENT CHANGE (the only behavioural diff
            # versus core.common.evaluate_hierarchical):
            #
            #   Standard pipeline would do here:
            #       probs = _apply_hierarchical_filter(
            #           probs, group_probs_test, hierarchy)
            #
            #   We intentionally SKIP that call. Everything else in
            #   this branch (per-class thresholds, top-K cap) is
            #   unchanged.
            # ──────────────────────────────────────────────────────
            logger.info(
                "[Eval-NoFilter] attributes: hierarchical group filter "
                "SKIPPED (disabled for reviewer ablation experiment)"
            )

            per_class_t = cfg["per_class_t"]
            preds       = (probs > per_class_t[np.newaxis, :]).astype(int)

            best_k = cfg["top_k"]
            for i in range(preds.shape[0]):
                pos_idx = np.where(preds[i] == 1)[0]
                if len(pos_idx) > best_k:
                    top_idx = pos_idx[np.argsort(probs[i, pos_idx])[-best_k:]]
                    preds[i] = 0
                    preds[i, top_idx] = 1

            logger.info(
                f"[Eval-NoFilter] attributes: per-class thresholds + "
                f"top-{best_k} filter applied (hierarchical filter OFF)"
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

    nb = total_b
    rows.append({
        "Head": "CompositeLoss",
        "Macro_F1": None, "Micro_F1": None,
        "Macro_Precision": None, "Micro_Precision": None,
        "Macro_Recall": None, "Micro_Recall": None, "mAP": None,
        "L_bce": total_bce / nb, "L_consistency": total_cs / nb,
        "L_path": total_path / nb, "CHL": total_CHL / nb,
    })

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

    if hierarchy is not None:
        attr_true_arr = all_true_total[keys.index("attributes")]
        attr_pred_arr = all_pred_total[keys.index("attributes")]
        attr_prob_arr = all_prob_total[keys.index("attributes")]

        group_results = compute_per_group_attr_f1(
            attr_true_arr, attr_pred_arr, attr_prob_arr, hierarchy)

        print("\n--- ATTRIBUTE F1 BY GROUP (hierarchical filter OFF) ---")
        print(f"  {'Group':50s}  {'n_attrs':>7}  {'Macro_F1':>8}  {'Micro_F1':>8}  {'mAP':>7}")
        print("  " + "─" * 86)
        for gr in group_results:
            marker = "  "
            if gr["group"] == "nickname_group":
                print("  " + "─" * 86)
                marker = "* "
            elif gr["group"] == "non_nickname_groups":
                marker = "* "
            print(f"  {marker}{gr['group']:48s}  {gr['n_attrs']:>7}  "
                  f"{gr['Macro_F1']:>8.4f}  {gr['Micro_F1']:>8.4f}  {gr['mAP']:>7.4f}")
        print("  (* = summary rows — nickname vs non-nickname)")

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

        print("\n--- HIERARCHICAL VIOLATION RATE (HVR) — filter OFF ---")
        print(f"  HVR (any violation)       : {hvr['HVR']:.4f}")
        print(f"  HVR_subcat (V1)           : {hvr['HVR_subcat']:.4f}")
        print(f"  HVR_attr   (V2)           : {hvr['HVR_attr']:.4f}")
        print(f"  (evaluated on {hvr['n_images']:,} images)")

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


def _extract_headline_metrics(df, label):
    """Pull Micro-F1 / Macro-F1 / mAP (overall + attributes-only) and
    the three HVR variants out of an evaluate_hierarchical-style
    results dataframe."""
    overall = df[df["Head"] == "OverallModel"].iloc[0]
    attrs   = df[df["Head"] == "attributes"].iloc[0]
    hvr_row = df[df["Head"] == "HVR"]
    hvr     = hvr_row.iloc[0] if not hvr_row.empty else None

    return {
        "Variant":             label,
        "Micro_F1_overall":    overall["Micro_F1"],
        "Macro_F1_overall":    overall["Macro_F1"],
        "mAP_overall":         overall["mAP"],
        "Micro_F1_attributes": attrs["Micro_F1"],
        "Macro_F1_attributes": attrs["Macro_F1"],
        "mAP_attributes":      attrs["mAP"],
        # Mapping out of the HVR row's repurposed columns
        # (see core.common.evaluate_hierarchical / compute_hvr):
        #   L_bce         -> HVR  (any violation)
        #   L_consistency -> HVR_subcat (V1: subcat without parent category)
        #   L_path        -> HVR_attr  (V2: attribute without parent group)
        "HVR_any":             hvr["L_bce"]         if hvr is not None else None,
        "HVR_subcat":          hvr["L_consistency"] if hvr is not None else None,
        "HVRattr":             hvr["L_path"]        if hvr is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OG-HMC checkpoint with the "
                     "evaluation-time hierarchical filter DISABLED, and "
                     "compare against the standard (filter-ON) pipeline."
    )
    parser.add_argument("--ckpt", type=str, default=None,
                         help="Path to a trained OG-HMC .ckpt file. If "
                              "omitted, auto-discovers the checkpoint "
                              "produced by train_oghmc_no_eval_filter.py "
                              "for --seed.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the seed used for training, so "
                              "the val/test split is reconstructed "
                              "identically.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--top-k-candidates", type=str, default="3,5,7,10,15",
                         help="Comma-separated top-K candidates used during "
                              "threshold calibration (same search space as "
                              "the standard pipeline).")
    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    top_k_candidates = tuple(int(x) for x in args.top_k_candidates.split(","))

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    _ensure_constraint_matrix_builder(core_common)
    oghmc_main, main_path = _load_oghmc_main_module()
    _check_loss_accepts_constraint_matrices(core_common)

    HierFashionLightning = oghmc_main.HierFashionLightning
    FashionDataModule    = oghmc_main.FashionDataModule
    LR_IMPROVED           = oghmc_main.LR_IMPROVED
    logger.info(f"Loaded OG-HMC model classes from: {main_path}")

    # ── Resolve checkpoint ───────────────────────────────────────
    ckpt_path = args.ckpt or _find_checkpoint(seed)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No checkpoint found. Either run "
            f"`python experiments/train_oghmc_no_eval_filter.py --seed {seed}` "
            "first, or pass --ckpt /path/to/checkpoint.ckpt explicitly."
        )
    logger.info(f"Using checkpoint: {ckpt_path}")

    # ── Hierarchy / graph / data — identical reconstruction to the
    #    training script (same seed -> same split) ──────────────
    hierarchy, hierarchy_json = load_hierarchy()
    adj_matrix = build_adjacency_matrix(hierarchy, hierarchy_json)
    subcat_to_cat_matrix, attr_to_group_matrix = \
        core_common.build_hierarchy_constraint_matrices(hierarchy)

    data_module = FashionDataModule(hierarchy=hierarchy,
                                     batch_size=batch_size, seed=seed)
    data_module.setup()

    pos_weight = compute_pos_weights(
        data_module.train_dataloader(), hierarchy, device)
    steps_per_epoch = math.ceil(len(data_module.train_subset) / batch_size)

    # ── Load the trained checkpoint — same model, unmodified ───────
    model = HierFashionLightning.load_from_checkpoint(
        ckpt_path,
        hierarchy=hierarchy, adj_matrix=adj_matrix,
        pos_weight=pos_weight, learning_rate=LR_IMPROVED,
        steps_per_epoch=steps_per_epoch,
        subcat_to_cat_matrix=subcat_to_cat_matrix,
        attr_to_group_matrix=attr_to_group_matrix,
    )
    model.to(device)
    model.eval()

    # ── Calibration — UNMODIFIED, standard pipeline. The filter is
    #    still used here (as designed) to pick the best top-K on the
    #    validation set; only the final TEST evaluation below skips it.
    logger.info("Calibrating thresholds on validation set "
                "(standard pipeline — filter active during calibration)...")
    threshold_config = calibrate_thresholds(
        model, data_module.val_dataloader(), device, hierarchy,
        top_k_candidates=top_k_candidates,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── (A) Standard evaluation — filter ON — reference baseline.
    #    Uses core.common.evaluate_hierarchical() UNMODIFIED. ───────
    logger.info("Running STANDARD evaluation (hierarchical filter ENABLED)...")
    df_with_filter = evaluate_hierarchical(
        model, data_module.test_dataloader(), device, model.loss_fn,
        save_path=os.path.join(
            args.output_dir, f"oghmc_WITH_eval_filter_seed{seed}_test_results.csv"),
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    # ── (B) Ablation evaluation — filter OFF — the reviewer's question.
    logger.info("Running ABLATION evaluation (hierarchical filter DISABLED)...")
    df_no_filter = evaluate_hierarchical_no_filter(
        model, data_module.test_dataloader(), device, model.loss_fn,
        save_path=os.path.join(
            args.output_dir, f"oghmc_NO_eval_filter_seed{seed}_test_results.csv"),
        threshold_config=threshold_config,
        hierarchy=hierarchy,
    )

    # ── Side-by-side summary ────────────────────────────────────
    summary_rows = [
        _extract_headline_metrics(df_with_filter, "with_eval_filter (standard)"),
        _extract_headline_metrics(df_no_filter, "without_eval_filter (reviewer ablation)"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        args.output_dir,
        f"reviewer_response_eval_filter_ablation_seed{seed}_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("  REVIEWER-RESPONSE EXPERIMENT — eval-time hierarchical filter")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\n  Full per-head results:")
    print(f"    {os.path.join(args.output_dir, f'oghmc_WITH_eval_filter_seed{seed}_test_results.csv')}")
    print(f"    {os.path.join(args.output_dir, f'oghmc_NO_eval_filter_seed{seed}_test_results.csv')}")
    print(f"  Summary:")
    print(f"    {summary_path}")


if __name__ == "__main__":
    main()
