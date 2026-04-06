"""
@file use_keel_audit.py
@brief Config-driven runner for KEEL_Audit_Utility aligned with the current class-based architecture.

This script is intentionally kept lightweight and configuration-driven. It is designed
for direct use during development: edit the CONFIG dictionary in this file and run it.

It is aligned with the current KEEL_Audit_Utility public API and its architectural
metadata, including dataset families and category descriptions.

##############################################
# ✅ Quick Summary of Features
# - Run file/folder/root KEEL audits without CLI parsing
# - Select the active dataset family through CONFIG
# - Print compact summaries for file/folder/root runs
# - Optionally print supported families and active category metadata
# - Prefer local KEEL_Audit_Utility import, with FRsutils fallback
#
# ✅ Summary Table of Design Patterns
# Category                Name                     Usage & Where Applied
# ------------------------------------------------------------------------------------
# Design Pattern          Facade                   Uses the public API exposed by KEEL_Audit_Utility
# Design Pattern          Strategy-like Config     Runtime behavior controlled by CONFIG
# Design Pattern          Adapter-style Import     Local import first, project import fallback
# Clean Code              SRP, Fail-Fast           Validation, running, and printing are separated
#
##############################################
# ✅ How to Use - Examples
##############################################
# 1) Edit CONFIG["LEVEL"] and CONFIG["TARGET_PATH"].
# 2) Optionally set CONFIG["DATASET_FAMILY"] to one of the supported families.
# 3) Run: python use_keel_audit.py
# 4) Read the saved JSON path and printed executive summary.
##############################################
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class KEELAuditImportBridge:
    """
    @brief Import bridge for KEEL_Audit_Utility public API.

    This runner first tries to import the local file-level module, which is the most
    convenient mode during development. If that fails, it falls back to a project path.
    """

    @staticmethod
    def load_api() -> Dict[str, Any]:
        """
        @brief Load KEEL audit public API symbols.

        @return Mapping with callable public API entries.
        """
        try:
            from KEEL_Audit_Utility import (
                audit_keel_file,
                audit_keel_folder,
                audit_keel_root,
                list_supported_audit_families,
                describe_active_audit_categories,
            )
            return {
                "audit_keel_file": audit_keel_file,
                "audit_keel_folder": audit_keel_folder,
                "audit_keel_root": audit_keel_root,
                "list_supported_audit_families": list_supported_audit_families,
                "describe_active_audit_categories": describe_active_audit_categories,
                "import_source": "local_module",
            }
        except Exception:
            pass

        try:
            from FRsutils.utils.dataset_utils.KEEL.KEEL_Audit_Utility import (
                audit_keel_file,
                audit_keel_folder,
                audit_keel_root,
                list_supported_audit_families,
                describe_active_audit_categories,
            )
            return {
                "audit_keel_file": audit_keel_file,
                "audit_keel_folder": audit_keel_folder,
                "audit_keel_root": audit_keel_root,
                "list_supported_audit_families": list_supported_audit_families,
                "describe_active_audit_categories": describe_active_audit_categories,
                "import_source": "FRsutils_package",
            }
        except Exception as exc:
            raise ImportError(
                "Could not import KEEL_Audit_Utility. Make sure either the local file "
                "KEEL_Audit_Utility.py is next to this script, or the FRsutils package "
                "path is installed and importable."
            ) from exc


CONFIG: Dict[str, Any] = {
    "LEVEL": "root",  # "file" | "folder" | "root"
    "TARGET_PATH": r"F:/Personal_Files/Datasets_research/Datasets/Datasets/KEEL/imbalanced/binary_class/IR_higher_than_9_Part_3",
    "OUT_DIR": r"F:/temp",
    "SAVE": True,
    "TOP_N_OFFENDERS": 20,
    "DATASET_FAMILY": "imbalanced_oversampling",
    "PRINT_IMPORT_SOURCE": True,
    "PRINT_SUPPORTED_FAMILIES": False,
    "PRINT_ACTIVE_CATEGORIES": True,
    "PRINT_SAVED_PATH": True,
    "PRINT_FILE_SUMMARY": True,
    "PRINT_FOLDER_SUMMARY": True,
    "PRINT_ROOT_EXEC_SUMMARY": True,
}


class KEELAuditRunnerConfigValidator:
    """
    @brief Validation utilities for runner configuration.
    """

    @staticmethod
    def validate(cfg: Dict[str, Any]) -> None:
        """
        @brief Validate runner configuration.

        @param cfg Runner configuration dictionary.
        """
        level = cfg.get("LEVEL")
        if level not in {"file", "folder", "root"}:
            raise ValueError('CONFIG["LEVEL"] must be one of: "file", "folder", "root".')

        if cfg.get("SAVE", True) and (not cfg.get("OUT_DIR")):
            raise ValueError('When CONFIG["SAVE"] = True you must set CONFIG["OUT_DIR"].')

        family = cfg.get("DATASET_FAMILY")
        if not isinstance(family, str) or not family.strip():
            raise ValueError('CONFIG["DATASET_FAMILY"] must be a non-empty string.')


class KEELAuditSummaryPrinter:
    """
    @brief Printer for compact, human-readable summaries of audit payloads.
    """

    @staticmethod
    def _safe_json(data: Any) -> str:
        """
        @brief Convert a Python object to formatted JSON text.
        """
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def print_supported_families(rows: List[Dict[str, Any]]) -> None:
        """
        @brief Print supported dataset families.

        @param rows List of family metadata dictionaries.
        """
        print("\n========== SUPPORTED AUDIT FAMILIES ==========")
        for row in rows:
            print(f"- {row.get('key')}: {row.get('display_name')}")
            if row.get("description"):
                print(f"  description: {row.get('description')}")
        print("============================================\n")

    @staticmethod
    def print_active_categories(rows: List[Dict[str, Any]], dataset_family: str) -> None:
        """
        @brief Print active categories for the selected family.

        @param rows Category metadata rows.
        @param dataset_family Family key.
        """
        print("\n========== ACTIVE AUDIT CATEGORIES ==========")
        print(f"Dataset family: {dataset_family}")
        for idx, row in enumerate(rows, start=1):
            print(f"{idx:02d}. {row.get('display_name')} [{row.get('key')}]")
            if row.get("description"):
                print(f"    {row.get('description')}")
            owned_keys = row.get("owned_top_level_keys") or []
            if owned_keys:
                print(f"    owns: {owned_keys}")
        print("============================================\n")

    @staticmethod
    def print_file_summary(file_json: Dict[str, Any]) -> None:
        """
        @brief Print a compact file-level summary.

        @param file_json File-level audit payload.
        """
        osa = file_json.get("oversampling_audit", {})
        family = file_json.get("audit_family", {})

        print("\n========== KEEL FILE AUDIT SUMMARY ==========")
        print(f"Relation: {file_json.get('relation')}")
        print(f"File: {file_json.get('file', {}).get('basename')}")
        print(f"Dataset family: {family.get('display_name', family.get('key'))}")
        print("Class info:")
        print(KEELAuditSummaryPrinter._safe_json(file_json.get("class_info", {})))
        print("Oversampling recommendation:")
        print(KEELAuditSummaryPrinter._safe_json(osa.get("oversampling_recommendation", {})))
        print("Synthetic interpolation risk:")
        print(KEELAuditSummaryPrinter._safe_json(osa.get("synthetic_interpolation_risk", {})))
        print("============================================\n")

    @staticmethod
    def print_folder_summary(folder_json: Dict[str, Any], top_n: int) -> None:
        """
        @brief Print a compact folder-level executive summary.

        @param folder_json Folder-level audit payload.
        @param top_n Maximum number of datasets to print.
        """
        rows = folder_json.get("executive_dataset_summaries", [])[: int(top_n)]
        family = folder_json.get("audit_family", {})

        print("\n========== KEEL FOLDER AUDIT SUMMARY ==========")
        print(f"Folder: {folder_json.get('folder', {}).get('name')}")
        print(f"Detected datasets: {folder_json.get('datasets_detected', {}).get('count')}")
        print(f"Dataset family: {family.get('display_name', family.get('key'))}")

        for idx, row in enumerate(rows, start=1):
            rec = row.get("oversampling_summary", {}).get("dataset_recommendation", {})
            rep_rec = row.get("oversampling_summary", {}).get("representative_file_recommendation", {})
            risk = row.get("oversampling_summary", {}).get("representative_file_synthetic_interpolation_risk", {})
            status = row.get("run_readiness", {}).get("status")
            severity = row.get("severity", {}).get("score_0_100")

            print(
                f"{idx:02d}. {row.get('dataset_key')} | status={status} | "
                f"dataset_rec={rec.get('status')} | rep_file_rec={rep_rec.get('status')} | "
                f"severity={severity} | risk={risk.get('score_0_100')}"
            )
            if rec.get("cautious_reasons"):
                print("    dataset cautious:", rec.get("cautious_reasons"))
            if rep_rec.get("cautious_reasons"):
                print("    representative file cautious:", rep_rec.get("cautious_reasons"))
            unstable = row.get("partition_stability_summary", {}).get("unstable_reasons")
            if unstable:
                print("    stability:", unstable)

        print("==============================================\n")

    @staticmethod
    def print_root_summary(root_json: Dict[str, Any], top_n: int) -> None:
        """
        @brief Print a compact root-level executive summary.

        @param root_json Root-level audit payload.
        @param top_n Maximum number of offenders to print.
        """
        exec_sum = root_json.get("executive_summary", {})
        counters = exec_sum.get("issue_counters", {})
        offenders = exec_sum.get("top_offenders", [])[: int(top_n)]
        similar_pairs = exec_sum.get("similar_dataset_pairs", [])[: min(10, int(top_n))]
        family = root_json.get("audit_family", {})

        print("\n========== KEEL ROOT AUDIT EXEC SUMMARY ==========")
        print(f"Root: {root_json.get('root', {}).get('abs_path')}")
        print(f"Dataset family: {family.get('display_name', family.get('key'))}")
        print("Counters:")
        print(KEELAuditSummaryPrinter._safe_json(counters))

        print("\nTop offenders:")
        for i, row in enumerate(offenders, start=1):
            print(
                f"{i:02d}. {row.get('folder')}::{row.get('dataset_key')} | "
                f"score={row.get('severity_score_0_100')} | status={row.get('run_status')}"
            )
            if row.get("severity_breakdown"):
                print("    breakdown:", row.get("severity_breakdown"))
            if row.get("blocked_reasons"):
                print("    blocked:", row.get("blocked_reasons"))
            if row.get("risky_reasons"):
                print("    risky:", row.get("risky_reasons"))

        if similar_pairs:
            print("\nSimilar dataset profiles:")
            for pair in similar_pairs:
                a = pair.get("dataset_a", {})
                b = pair.get("dataset_b", {})
                print(
                    f"- {a.get('folder')}::{a.get('dataset_key')} ~ "
                    f"{b.get('folder')}::{b.get('dataset_key')} | "
                    f"distance={pair.get('profile_distance')}"
                )

        print("==================================================\n")


class KEELAuditRunner:
    """
    @brief Config-driven execution runner for KEEL audits.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        @brief Initialize the runner.

        @param config Runner configuration dictionary.
        """
        self.config = config
        self.api = KEELAuditImportBridge.load_api()

    def _print_architecture_metadata(self) -> None:
        """
        @brief Optionally print supported families and active category metadata.
        """
        dataset_family = self.config["DATASET_FAMILY"]

        if self.config.get("PRINT_IMPORT_SOURCE", True):
            print(f"\nImport source: {self.api.get('import_source')}")

        if self.config.get("PRINT_SUPPORTED_FAMILIES", False):
            rows = self.api["list_supported_audit_families"]()
            KEELAuditSummaryPrinter.print_supported_families(rows)

        if self.config.get("PRINT_ACTIVE_CATEGORIES", False):
            rows = self.api["describe_active_audit_categories"](dataset_family=dataset_family)
            KEELAuditSummaryPrinter.print_active_categories(rows, dataset_family)

    def run(self) -> None:
        """
        @brief Execute the configured audit run.
        """
        KEELAuditRunnerConfigValidator.validate(self.config)
        self._print_architecture_metadata()

        level = self.config["LEVEL"]
        target = os.path.abspath(self.config["TARGET_PATH"])
        out_dir = os.path.abspath(self.config["OUT_DIR"]) if self.config.get("OUT_DIR") else None
        save = bool(self.config.get("SAVE", True))
        top_n = int(self.config.get("TOP_N_OFFENDERS", 20))
        dataset_family = self.config.get("DATASET_FAMILY", "imbalanced_oversampling")

        if level == "file":
            res = self.api["audit_keel_file"](
                target,
                out_dir=out_dir,
                save=save,
                dataset_family=dataset_family,
            )
            if self.config.get("PRINT_SAVED_PATH", True) and save:
                print("\nSaved:", res.get("saved_to"))
            if self.config.get("PRINT_FILE_SUMMARY", True):
                KEELAuditSummaryPrinter.print_file_summary(res)
            return

        if level == "folder":
            res = self.api["audit_keel_folder"](
                target,
                out_dir=out_dir,
                save=save,
                dataset_family=dataset_family,
            )
            if self.config.get("PRINT_SAVED_PATH", True) and save:
                print("\nSaved:", res.get("saved_to"))
            if self.config.get("PRINT_FOLDER_SUMMARY", True):
                KEELAuditSummaryPrinter.print_folder_summary(res, top_n=top_n)
            return

        res = self.api["audit_keel_root"](
            target,
            out_dir=out_dir,
            save=save,
            top_n_offenders=top_n,
            dataset_family=dataset_family,
        )
        if self.config.get("PRINT_SAVED_PATH", True) and save:
            print("\nSaved:", res.get("saved_to"))
        if self.config.get("PRINT_ROOT_EXEC_SUMMARY", True):
            if save and res.get("saved_to") and os.path.isfile(res["saved_to"]):
                with open(res["saved_to"], "r", encoding="utf-8") as handle:
                    root_json = json.load(handle)
                KEELAuditSummaryPrinter.print_root_summary(root_json, top_n)
            else:
                KEELAuditSummaryPrinter.print_root_summary(res, top_n)


def main() -> None:
    """
    @brief Script entry point.
    """
    KEELAuditRunner(CONFIG).run()


if __name__ == "__main__":
    main()
