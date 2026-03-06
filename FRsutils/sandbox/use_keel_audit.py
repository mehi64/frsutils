# """
# @file keel_audit_cli.py
# @brief KEEL audit runner without CLI args (edit CONFIG in this file).

# Copy to:
#   scripts/keel_audit_cli.py

# Edit CONFIG and run:
#   python scripts/keel_audit_cli.py
# """

# from __future__ import annotations

# import json
# import os
# from typing import Any, Dict

# from FRsutils.utils.dataset_utils.KEEL.KEEL_Audit_Utility import (
#     audit_keel_file,
#     audit_keel_folder,
#     audit_keel_root,
# )

# # ============================================================
# # ✅ EDIT CONFIG HERE
# # ============================================================
# CONFIG: Dict[str, Any] = {
#     "LEVEL": "root",  # "file" | "folder" | "root"
#     "TARGET_PATH": r"F:/Personal_Files/FRsutils/datasets/KEEL/imbalanced",
#     "OUT_DIR": r"F:/temp",
#     "SAVE": True,

#     "TOP_N_OFFENDERS": 20,
#     "PRINT_ROOT_EXEC_SUMMARY": True,
#     "PRINT_SAVED_PATH": True,
# }


# def _validate_config(cfg: Dict[str, Any]) -> None:
#     level = cfg.get("LEVEL")
#     if level not in {"file", "folder", "root"}:
#         raise ValueError('CONFIG["LEVEL"] must be one of: "file", "folder", "root".')
#     if cfg.get("SAVE", True) and (not cfg.get("OUT_DIR")):
#         raise ValueError('When CONFIG["SAVE"]=True you must set CONFIG["OUT_DIR"].')


# def _print_root_exec(root_json: Dict[str, Any], top_n: int) -> None:
#     exec_sum = root_json.get("executive_summary", {})
#     counters = exec_sum.get("issue_counters", {})
#     offenders = exec_sum.get("top_offenders", [])[: int(top_n)]

#     print("\n========== KEEL AUDIT EXEC SUMMARY ==========")
#     print("Counters:", json.dumps(counters, indent=2, ensure_ascii=False))

#     print("\nTop offenders:")
#     for i, o in enumerate(offenders, start=1):
#         print(f"{i:02d}. {o.get('folder')}::{o.get('dataset_key')} | score={o.get('severity_score_0_100')} | status={o.get('run_status')}")
#         br = o.get("severity_breakdown", {})
#         if br:
#             print("    breakdown:", br)
#         if o.get("blocked_reasons"):
#             print("    blocked:", o.get("blocked_reasons"))
#         if o.get("risky_reasons"):
#             print("    risky:", o.get("risky_reasons"))
#     print("============================================\n")


# def main() -> None:
#     _validate_config(CONFIG)

#     level = CONFIG["LEVEL"]
#     target = os.path.abspath(CONFIG["TARGET_PATH"])
#     out_dir = os.path.abspath(CONFIG["OUT_DIR"]) if CONFIG.get("OUT_DIR") else None
#     save = bool(CONFIG.get("SAVE", True))

#     if level == "file":
#         res = audit_keel_file(target, out_dir=out_dir, save=save)
#         if CONFIG.get("PRINT_SAVED_PATH", True) and save:
#             print("\nSaved:", res.get("saved_to"))
#         return

#     if level == "folder":
#         res = audit_keel_folder(target, out_dir=out_dir, save=save)
#         if CONFIG.get("PRINT_SAVED_PATH", True) and save:
#             print("\nSaved:", res.get("saved_to"))
#         return

#     res = audit_keel_root(target, out_dir=out_dir, save=save, top_n_offenders=int(CONFIG.get("TOP_N_OFFENDERS", 20)))
#     if CONFIG.get("PRINT_SAVED_PATH", True) and save:
#         print("\nSaved:", res.get("saved_to"))

#     if CONFIG.get("PRINT_ROOT_EXEC_SUMMARY", True):
#         if save and res.get("saved_to") and os.path.isfile(res["saved_to"]):
#             with open(res["saved_to"], "r", encoding="utf-8") as f:
#                 root_json = json.load(f)
#             _print_root_exec(root_json, int(CONFIG.get("TOP_N_OFFENDERS", 20)))
#         else:
#             _print_root_exec(res, int(CONFIG.get("TOP_N_OFFENDERS", 20)))


# if __name__ == "__main__":
#     main()


"""
@file use_keel_audit.py
@brief Config-driven runner for KEEL_Audit_Utility with oversampling-focused summaries.

Small helper script for running file/folder/root audits without command-line arguments.
All behavior is configured through the CONFIG dictionary below.

##############################################
# ✅ Quick Summary of Features
# - Configure LEVEL and TARGET_PATH directly in this file
# - Run file/folder/root audit without CLI parsing
# - Print compact executive summaries after saving JSON results
# - Highlight oversampling-risk counters, top offenders, and similar dataset profiles
#
# ✅ Summary Table of Design Patterns
# Category                Name                    Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern          Facade                  Thin runner over 3 public audit entry points
# Architecture            Config-Driven Script    Central CONFIG dictionary controls behavior
# Clean Code              SRP, Fail-Fast          Validation, execution, printing are separated
#
##############################################
# ✅ How to Use - Examples
##############################################
# 1) Set CONFIG["LEVEL"] = "root" and CONFIG["TARGET_PATH"] to a KEEL root folder.
# 2) Run: python use_keel_audit.py
# 3) Read the saved JSON path and executive summary printed to stdout.
##############################################
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from FRsutils.utils.dataset_utils.KEEL.KEEL_Audit_Utility import (
    audit_keel_file,
    audit_keel_folder,
    audit_keel_root,
)


CONFIG: Dict[str, Any] = {
    "LEVEL": "root",  # "file" | "folder" | "root"
    "TARGET_PATH": r"F:/Personal_Files/Datasets_research/Datasets/Datasets/KEEL/imbalanced/binary_class/IR_higher_than_9_Part_3",
    "OUT_DIR": r"F:/temp",
    "SAVE": True,
    "TOP_N_OFFENDERS": 20,
    "PRINT_ROOT_EXEC_SUMMARY": True,
    "PRINT_SAVED_PATH": True,
}


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    @brief Validate runner configuration.

    @param cfg Runner configuration dictionary.
    """
    level = cfg.get("LEVEL")
    if level not in {"file", "folder", "root"}:
        raise ValueError('CONFIG["LEVEL"] must be one of: "file", "folder", "root".')
    if cfg.get("SAVE", True) and (not cfg.get("OUT_DIR")):
        raise ValueError('When CONFIG["SAVE"]=True you must set CONFIG["OUT_DIR"].')


def _print_file_summary(file_json: Dict[str, Any]) -> None:
    """
    @brief Print a compact file-level oversampling summary.

    @param file_json Audit result payload.
    """
    osa = file_json.get("oversampling_audit", {})
    print("\n========== KEEL FILE AUDIT SUMMARY ==========")
    print(f"Relation: {file_json.get('relation')}")
    print(f"File: {file_json.get('file', {}).get('basename')}")
    print("Class info:", json.dumps(file_json.get("class_info", {}), indent=2, ensure_ascii=False))
    print("Oversampling recommendation:", json.dumps(osa.get("oversampling_recommendation", {}), indent=2, ensure_ascii=False))
    print("Synthetic interpolation risk:", json.dumps(osa.get("synthetic_interpolation_risk", {}), indent=2, ensure_ascii=False))
    print("============================================\n")


def _print_folder_summary(folder_json: Dict[str, Any], top_n: int) -> None:
    """
    @brief Print a compact folder-level executive summary.

    @param folder_json Folder audit payload.
    @param top_n Maximum number of datasets to print.
    """
    rows = folder_json.get("executive_dataset_summaries", [])[: int(top_n)]
    print("\n========== KEEL FOLDER AUDIT SUMMARY ==========")
    print(f"Folder: {folder_json.get('folder', {}).get('name')}")
    print(f"Detected datasets: {folder_json.get('datasets_detected', {}).get('count')}")
    for idx, row in enumerate(rows, start=1):
        rec = row.get("oversampling_summary", {}).get("dataset_recommendation", {})
        risk = row.get("oversampling_summary", {}).get("representative_file_synthetic_interpolation_risk", {})
        print(
            f"{idx:02d}. {row.get('dataset_key')} | status={row.get('run_readiness', {}).get('status')} | "
            f"oversampling={rec.get('status')} | severity={row.get('severity', {}).get('score_0_100')} | "
            f"risk={risk.get('score_0_100')}"
        )
        if rec.get("cautious_reasons"):
            print("    cautious:", rec.get("cautious_reasons"))
        if row.get("partition_stability_summary", {}).get("unstable_reasons"):
            print("    stability:", row.get("partition_stability_summary", {}).get("unstable_reasons"))
    print("==============================================\n")


def _print_root_exec(root_json: Dict[str, Any], top_n: int) -> None:
    """
    @brief Print a compact root-level executive summary.

    @param root_json Root audit payload.
    @param top_n Maximum number of offenders to print.
    """
    exec_sum = root_json.get("executive_summary", {})
    counters = exec_sum.get("issue_counters", {})
    offenders = exec_sum.get("top_offenders", [])[: int(top_n)]
    similar_pairs = exec_sum.get("similar_dataset_pairs", [])[: min(10, int(top_n))]

    print("\n========== KEEL AUDIT EXEC SUMMARY ==========")
    print("Counters:", json.dumps(counters, indent=2, ensure_ascii=False))

    print("\nTop offenders:")
    for i, o in enumerate(offenders, start=1):
        print(
            f"{i:02d}. {o.get('folder')}::{o.get('dataset_key')} | score={o.get('severity_score_0_100')} | "
            f"status={o.get('run_status')}"
        )
        if o.get("severity_breakdown"):
            print("    breakdown:", o.get("severity_breakdown"))
        if o.get("blocked_reasons"):
            print("    blocked:", o.get("blocked_reasons"))
        if o.get("risky_reasons"):
            print("    risky:", o.get("risky_reasons"))

    if similar_pairs:
        print("\nSimilar dataset profiles:")
        for pair in similar_pairs:
            a = pair.get("dataset_a", {})
            b = pair.get("dataset_b", {})
            print(
                f"- {a.get('folder')}::{a.get('dataset_key')} ~ {b.get('folder')}::{b.get('dataset_key')} | "
                f"distance={pair.get('profile_distance')}"
            )

    print("============================================\n")


def main() -> None:
    """
    @brief Execute the configured audit run.
    """
    _validate_config(CONFIG)

    level = CONFIG["LEVEL"]
    target = os.path.abspath(CONFIG["TARGET_PATH"])
    out_dir = os.path.abspath(CONFIG["OUT_DIR"]) if CONFIG.get("OUT_DIR") else None
    save = bool(CONFIG.get("SAVE", True))
    top_n = int(CONFIG.get("TOP_N_OFFENDERS", 20))

    if level == "file":
        res = audit_keel_file(target, out_dir=out_dir, save=save)
        if CONFIG.get("PRINT_SAVED_PATH", True) and save:
            print("\nSaved:", res.get("saved_to"))
        _print_file_summary(res)
        return

    if level == "folder":
        res = audit_keel_folder(target, out_dir=out_dir, save=save)
        if CONFIG.get("PRINT_SAVED_PATH", True) and save:
            print("\nSaved:", res.get("saved_to"))
        _print_folder_summary(res, top_n=top_n)
        return

    res = audit_keel_root(target, out_dir=out_dir, save=save, top_n_offenders=top_n)
    if CONFIG.get("PRINT_SAVED_PATH", True) and save:
        print("\nSaved:", res.get("saved_to"))

    if CONFIG.get("PRINT_ROOT_EXEC_SUMMARY", True):
        if save and res.get("saved_to") and os.path.isfile(res["saved_to"]):
            with open(res["saved_to"], "r", encoding="utf-8") as f:
                root_json = json.load(f)
            _print_root_exec(root_json, top_n)
        else:
            _print_root_exec(res, top_n)


if __name__ == "__main__":
    main()
