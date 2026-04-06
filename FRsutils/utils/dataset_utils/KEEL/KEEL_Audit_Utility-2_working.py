"""
@file KEEL_Audit_Utility.py
@brief KEEL Dataset Auditor Utility (File/Folder/Root level) with Level-3 Executive Summary.

A utility module for auditing KEEL .dat datasets before feeding them into FRsutils pipelines.
It generates structured metadata + data-quality diagnostics at 3 levels:

- Level 1: Single file audit (one .dat file)
- Level 2: Folder audit (a dataset folder with one or many .dat files and/or predefined folds)
- Level 3: Root audit (root folder containing many dataset folders) with executive summaries.

##############################################
# ✅ Quick Summary of Features
# - audit_keel_file(...)         -> Full audit of one .dat file + saves JSON
# - audit_keel_folder(...)       -> Audits all .dat files in a folder + detects CV structure
# - audit_keel_root(...)         -> Audits all dataset folders under a root + produces global summary
#
# High-ROI checks:
# - Schema consistency across folds (header drift / column order drift + drift reasons)
# - Duplicate rows within files
# - Leakage between train/test files of the same fold (identical rows or identical inputs) + leakage rates
# - Label conflicts: identical inputs mapped to multiple labels
# - Low-quality features: constant/near-constant, high missingness, high-cardinality nominal
# - Numeric sanity: parseability, integer integrity, out-of-range vs header, robust outlier-rate, heavy-tail flags
#
# Level-3 additions (executive):
# - Per dataset summary: n_samples (min/mean/max), n_features, n_classes, IR, min_class_count, CV type/K
# - Per dataset severity score (0..100) + breakdown + top offenders
# - Per dataset run readiness: SAFE/RISKY/BLOCKED with reasons

##############################################
# ✅ Summary Table of Design Patterns
# Category                Name                    Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern          Facade                  3 public entry-points (file/folder/root)
# Design Pattern          Adapter                 Optional reuse of parse_keel_file (sanity-check)
# Architecture            Layered Auditing        Level 1 -> Level 2 -> Level 3 composition
# Clean Code              SRP, DRY, Fail-Safe     Parsing/checking/summarizing split; folder/root catch errors

##############################################
# ✅ How to Use - Examples
##############################################
# 1) Audit a single file:
# from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import audit_keel_file
# res = audit_keel_file(".../iris0-5-1tra.dat", out_dir="out_audit")
#
# 2) Audit a dataset folder (maybe predefined folds):
# from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import audit_keel_folder
# folder_res = audit_keel_folder(".../iris0_folds", out_dir="out_audit")
#
# 3) Audit a root folder containing many dataset folders:
# from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import audit_keel_root
# root_res = audit_keel_root(".../KEEL_root", out_dir="out_audit")
##############################################
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Imports from project (best-effort)
# -----------------------------
def _import_keel_helpers():
    """
    @brief Best-effort import of existing FRsutils helpers (for reuse).

    Uses:
    - discover_keel_cv_folds (folder fold pairing)
    - parse_keel_file (optional sanity check for single-output datasets)

    @return (discover_keel_cv_folds, parse_keel_file) each can be None if import fails
    """
    discover = None
    parse = None

    for mod_name in [
        "FRsutils.utils.dataset_utils.KEEL_CV_Utility",
        "KEEL_CV_Utility",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["discover_keel_cv_folds"])
            discover = getattr(mod, "discover_keel_cv_folds")
            break
        except Exception:
            pass

    for mod_name in [
        "FRsutils.utils.dataset_utils.KEEL_DS_loader_utility",
        "KEEL_DS_loader_utility",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["parse_keel_file"])
            parse = getattr(mod, "parse_keel_file")
            break
        except Exception:
            pass

    return discover, parse


_DISCOVER_KEEL_CV_FOLDS, _PARSE_KEEL_FILE = _import_keel_helpers()


# -----------------------------
# Constants
# -----------------------------
_MISSING_TOKENS = {"?", "<null>"}
_SUSPICIOUS_NULL_LIKE = {"null", "nil", "none", "nan", "na"}


# -----------------------------
# JSON helpers
# -----------------------------
def _jsonify(obj: Any) -> Any:
    """
    @brief Convert non-JSON-native objects to JSON-friendly values.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _save_json(path: str, payload: Dict[str, Any]) -> str:
    """
    @brief Save audit payload as JSON.

    @param path Output path
    @param payload Dict payload
    @return absolute path of saved file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def _dataset_folder_name_from_path(folder_path: str) -> str:
    """
    @brief Return dataset folder name for stable output layout.
    """
    return os.path.basename(os.path.abspath(folder_path))


# -----------------------------
# KEEL parsing
# -----------------------------
def _read_text(file_path: str) -> str:
    """
    @brief Read a file as UTF-8 best-effort (replacement for bad bytes).
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _detect_encoding_issues(text: str) -> Dict[str, Any]:
    """
    @brief Detect encoding-related and control character issues.
    """
    has_replacement = "�" in text
    ctrl_count = 0
    for ch in text:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            ctrl_count += 1
    return {
        "has_replacement_char": bool(has_replacement),
        "control_chars_count": int(ctrl_count),
    }


def _parse_attribute_definition(defn: str) -> Tuple[str, Optional[float], Optional[float], Optional[set]]:
    """
    @brief Parse KEEL @attribute definition into (type, min, max, allowed_values).
    """
    defn_str = defn.strip()

    if defn_str.startswith("{") and defn_str.endswith("}"):
        inside = defn_str[1:-1]
        values = [v.strip() for v in inside.split(",") if v.strip()]
        return "nominal", None, None, set(values)

    low = defn_str.lower()
    if low.startswith("integer") or low.startswith("real"):
        t = "integer" if low.startswith("integer") else "real"
        m = re.search(r"\[(.*?),(.*?)\]", defn_str)
        if m:
            mn = float(m.group(1).strip())
            mx = float(m.group(2).strip())
            return t, mn, mx, None
        return t, None, None, None

    raise ValueError(f"Unknown KEEL attribute definition: {defn}")


def _parse_keel_header(lines: List[str]) -> Tuple[str, Dict[str, Any], List[str], List[str], int]:
    """
    @brief Parse header fields: relation, attributes, inputs, outputs, data start index.
    """
    relation = None
    attributes: Dict[str, Any] = {}
    inputs: List[str] = []
    outputs: List[str] = []
    data_start_idx = -1

    for idx, line in enumerate(lines):
        low = line.lower()

        if low.startswith("@relation"):
            parts = line.split()
            relation = parts[1] if len(parts) > 1 else None

        elif low.startswith("@attribute"):
            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) < 3:
                raise ValueError(f"Malformed @attribute line: {line}")
            name = parts[1].strip()
            defn = parts[2].strip()
            t, mn, mx, allowed = _parse_attribute_definition(defn)
            attributes[name] = {
                "type": t,
                "min_range_header": mn,
                "max_range_header": mx,
                "nominal_allowed_header": allowed,
            }

        elif low.startswith("@inputs"):
            rhs = line.split(" ", 1)[1] if " " in line else ""
            inputs = [x.strip() for x in rhs.split(",") if x.strip()]

        elif low.startswith("@outputs"):
            rhs = line.split(" ", 1)[1] if " " in line else ""
            outputs = [x.strip() for x in rhs.split(",") if x.strip()]

        elif low == "@data":
            data_start_idx = idx + 1
            break

    if data_start_idx < 0:
        raise ValueError("No @data section found in KEEL file.")
    if relation is None:
        relation = ""

    return relation, attributes, inputs, outputs, data_start_idx


def _split_csv_like(line: str) -> List[str]:
    """
    @brief Split KEEL data line by comma, trimming spaces.
    """
    return re.split(r"\s*,\s*", line.strip())


def _load_keel_data(lines: List[str], data_start_idx: int, attribute_names: List[str]) -> pd.DataFrame:
    """
    @brief Load KEEL @data into a DataFrame of raw strings.

    Fail-safe: lines with wrong column count are skipped and counted.
    """
    rows = []
    bad_lines = 0
    for ln in lines[data_start_idx:]:
        if not ln:
            continue
        parts = _split_csv_like(ln)
        if len(parts) != len(attribute_names):
            bad_lines += 1
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=attribute_names)
    df.attrs["bad_lines_column_mismatch"] = int(bad_lines)
    return df


def _is_missing_token(token: str) -> bool:
    """
    @brief Missing values are expressed either with ? or <null>.
    """
    if token is None:
        return True
    return str(token).strip() in _MISSING_TOKENS


def _is_suspicious_null_like(token: str) -> bool:
    """
    @brief Detect null-like tokens that are NOT defined as missing by spec.
    """
    if token is None:
        return False
    return str(token).strip().lower() in _SUSPICIOUS_NULL_LIKE


def _schema_signature(relation: str, attributes: Dict[str, Any], inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
    """
    @brief Create a comparable schema signature for drift detection.
    """
    attr_items = []
    for k, v in attributes.items():
        allowed = v.get("nominal_allowed_header")
        attr_items.append(
            {
                "name": k,
                "type": v.get("type"),
                "min": v.get("min_range_header"),
                "max": v.get("max_range_header"),
                "allowed": sorted(list(allowed)) if isinstance(allowed, set) else allowed,
            }
        )
    return {
        "relation": relation or "",
        "attributes_in_order": attr_items,
        "inputs": list(inputs),
        "outputs": list(outputs),
    }


def _diff_schema_signatures(base_sig: Dict[str, Any], other_sig: Dict[str, Any]) -> List[str]:
    """
    @brief Compute human-readable reasons for schema drift between signatures.
    """
    reasons: List[str] = []

    if (base_sig.get("relation") or "") != (other_sig.get("relation") or ""):
        reasons.append("relation_changed")
    if base_sig.get("inputs") != other_sig.get("inputs"):
        reasons.append("inputs_changed")
    if base_sig.get("outputs") != other_sig.get("outputs"):
        reasons.append("outputs_changed")

    base_attrs = base_sig.get("attributes_in_order") or []
    other_attrs = other_sig.get("attributes_in_order") or []
    base_names = [a.get("name") for a in base_attrs]
    other_names = [a.get("name") for a in other_attrs]
    if base_names != other_names:
        if len(base_names) != len(other_names):
            reasons.append("attribute_count_changed")
        reasons.append("attribute_order_or_names_changed")

    base_map = {a.get("name"): a for a in base_attrs if a.get("name") is not None}
    other_map = {a.get("name"): a for a in other_attrs if a.get("name") is not None}
    shared = sorted(list(set(base_map.keys()).intersection(set(other_map.keys()))))

    for n in shared:
        if base_map[n].get("type") != other_map[n].get("type"):
            reasons.append("attribute_type_changed")
            break

    for n in shared:
        b = base_map[n]
        o = other_map[n]
        if b.get("type") == "nominal":
            if (b.get("allowed") or []) != (o.get("allowed") or []):
                reasons.append("nominal_domain_changed")
                break
        if b.get("type") in ("real", "integer"):
            if (b.get("min") != o.get("min")) or (b.get("max") != o.get("max")):
                reasons.append("numeric_range_changed")
                break

    return sorted(list(set(reasons)))


def _row_hash(values: List[str]) -> str:
    """
    @brief Hash a row by joining tokens with a delimiter unlikely to occur in KEEL.
    """
    return "\x1f".join([str(v).strip() for v in values])


def _estimate_one_hot_dim(attributes: Dict[str, Any], nominal_inputs: List[str], observed: Dict[str, List[str]]) -> int:
    """
    @brief Estimate output dimension after one-hot for nominal inputs.
    """
    total = 0
    for c in nominal_inputs:
        allowed = attributes.get(c, {}).get("nominal_allowed_header")
        if isinstance(allowed, set) and len(allowed) > 0:
            total += len(allowed)
        else:
            total += len(observed.get(c, []))
    return int(total)


# -----------------------------
# Stats helpers
# -----------------------------
def _safe_float_array(series: pd.Series) -> Tuple[np.ndarray, List[str], int]:
    """
    @brief Convert raw string series to float array, excluding missing tokens.
    Returns (values, unknown_tokens_list, n_nonmissing).
    """
    unknown = []
    vals = []
    n_nonmissing = 0
    for x in series.astype(str).tolist():
        s = str(x).strip()
        if _is_missing_token(s):
            continue
        n_nonmissing += 1
        try:
            vals.append(float(s))
        except Exception:
            unknown.append(s)
    return np.asarray(vals, dtype=float), sorted(list(set(unknown))), int(n_nonmissing)


def _skewness(x: np.ndarray) -> float:
    """
    @brief Skewness via standardized third moment.
    """
    if x.size < 3:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """
    @brief Kurtosis via standardized fourth moment.
    """
    if x.size < 4:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 4))


def _robust_numeric_stats(x: np.ndarray) -> Dict[str, Any]:
    """
    @brief Numeric stats including MAD and sigma≈1.4826*MAD and tail/skew metrics.
    """
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "sigma_approx": float("nan"),
            "quantiles": {},
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "kurtosis_excess": float("nan"),
            "heavy_tail": False,
        }

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma_approx = float(1.4826 * mad)

    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    qv = np.quantile(x, qs).tolist()
    quantiles = {str(q): float(v) for q, v in zip(qs, qv)}

    skew = _skewness(x)
    kurt = _kurtosis(x)
    ex = float(kurt - 3.0) if np.isfinite(kurt) else float("nan")
    heavy = bool(np.isfinite(kurt) and kurt > 3.0)

    return {
        "n": int(x.size),
        "mean": mean,
        "std": float(std),
        "median": median,
        "mad": mad,
        "sigma_approx": sigma_approx,
        "quantiles": quantiles,
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "kurtosis_excess": ex,
        "heavy_tail": heavy,
    }


def _robust_outlier_rate(x: np.ndarray, *, z_threshold: float = 3.5) -> Dict[str, Any]:
    """
    @brief Robust outlier rate using robust z-score.
    """
    if x.size == 0:
        return {"z_threshold": float(z_threshold), "n_outliers": 0, "outlier_rate": float("nan")}
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    denom = 1.4826 * mad
    if denom <= 0:
        return {"z_threshold": float(z_threshold), "n_outliers": 0, "outlier_rate": 0.0}
    rz = np.abs(x - median) / denom
    n_out = int(np.sum(rz > float(z_threshold)))
    return {"z_threshold": float(z_threshold), "n_outliers": n_out, "outlier_rate": float(n_out / x.size)}


# -----------------------------
# CV naming detection
# -----------------------------
@dataclass(frozen=True)
class _FoldNameInfo:
    dataset_key: str
    k_folds: Optional[int]
    fold_index: Optional[int]
    split: Optional[str]  # "tra" | "tst" | None


_FOLD_PAT = re.compile(r"^(?P<base>.*?)-(?P<k>\d+)-(?P<i>\d+)(?P<split>tra|tst)\.dat$", re.IGNORECASE)


def _parse_fold_filename(basename: str) -> _FoldNameInfo:
    """
    @brief Parse typical KEEL predefined-fold filename like: iris0-5-1tra.dat
    """
    m = _FOLD_PAT.match(basename)
    if not m:
        low = basename.lower()
        if low.endswith("tra.dat"):
            return _FoldNameInfo(dataset_key=basename[:-7], k_folds=None, fold_index=None, split="tra")
        if low.endswith("tst.dat"):
            return _FoldNameInfo(dataset_key=basename[:-7], k_folds=None, fold_index=None, split="tst")
        return _FoldNameInfo(dataset_key=os.path.splitext(basename)[0], k_folds=None, fold_index=None, split=None)

    base = m.group("base")
    k = int(m.group("k"))
    i = int(m.group("i"))
    split = m.group("split").lower()
    return _FoldNameInfo(dataset_key=base, k_folds=k, fold_index=i, split=split)


# -----------------------------
# Feature quality checks
# -----------------------------
def _feature_missingness(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """
    @brief Per-feature missingness fraction.
    """
    out = {}
    n = len(df)
    if n == 0:
        return {c: float("nan") for c in cols}
    for c in cols:
        out[c] = float(sum(_is_missing_token(x) for x in df[c].astype(str).tolist()) / n)
    return out


def _constant_and_near_constant(df: pd.DataFrame, cols: List[str], *, near_constant_threshold: float = 0.999) -> Dict[str, Any]:
    """
    @brief Detect constant and near-constant features.
    """
    constant = []
    near_constant = []
    for c in cols:
        ser = df[c].astype(str).map(lambda x: str(x).strip())
        ser = ser[~ser.map(_is_missing_token)]
        if ser.empty:
            continue
        vc = ser.value_counts(dropna=False)
        n_unique = int(len(vc))
        top_value = str(vc.index[0])
        top_ratio = float(vc.iloc[0] / vc.sum())
        if n_unique == 1:
            constant.append(c)
        elif top_ratio >= float(near_constant_threshold):
            near_constant.append({"feature": c, "top_value": top_value, "top_ratio": top_ratio, "n_unique_nonmissing": n_unique})
    return {"constant_features": sorted(list(set(constant))), "near_constant_features": near_constant}


def _duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    @brief Duplicate row count and ratio.
    """
    n = int(len(df))
    if n == 0:
        return {"n_rows": 0, "n_duplicate_rows": 0, "duplicate_ratio": float("nan")}
    dup = int(df.duplicated(keep="first").sum())
    return {"n_rows": n, "n_duplicate_rows": dup, "duplicate_ratio": float(dup / n)}


def _high_cardinality_nominal(
    attributes: Dict[str, Any],
    nominal_inputs: List[str],
    observed: Dict[str, List[str]],
    *,
    high_cardinality_threshold: int = 50,
) -> Dict[str, Any]:
    """
    @brief Detect high-cardinality nominal inputs.
    """
    flagged = []
    for c in nominal_inputs:
        allowed = attributes.get(c, {}).get("nominal_allowed_header")
        header_card = len(allowed) if isinstance(allowed, set) else None
        obs_card = len(observed.get(c, []))
        card = header_card if (header_card is not None and header_card > 0) else obs_card
        if int(card) >= int(high_cardinality_threshold):
            flagged.append(
                {"feature": c, "cardinality_effective": int(card), "header_cardinality": header_card, "observed_cardinality": int(obs_card)}
            )
    return {"high_cardinality_threshold": int(high_cardinality_threshold), "high_cardinality_nominal": flagged}


def _label_conflicts(df: pd.DataFrame, inputs: List[str], outputs: List[str], *, max_rows: int = 200_000) -> Dict[str, Any]:
    """
    @brief Label conflicts: identical inputs with multiple labels (single-output only).

    Guard: if dataset is huge, this check may be expensive; it will be skipped and reported.
    """
    if len(outputs) != 1:
        return {"enabled": False, "reason": "multioutput_or_unknown", "n_conflicting_input_groups": 0}

    outc = outputs[0]
    if outc not in df.columns:
        return {"enabled": False, "reason": "output_not_in_df", "n_conflicting_input_groups": 0}

    if len(df) > int(max_rows):
        return {"enabled": False, "reason": f"too_large_gt_{max_rows}", "n_conflicting_input_groups": 0}

    sub = df[inputs + [outc]].astype(str).apply(lambda c: c.map(lambda x: str(x).strip()))
    sub = sub[~sub[outc].map(_is_missing_token)]
    if sub.empty:
        return {"enabled": True, "n_conflicting_input_groups": 0, "n_rows_in_conflicts": 0, "examples": []}

    grp = sub.groupby(inputs, dropna=False)[outc].nunique()
    conflict_keys = grp[grp > 1]
    n_groups = int(conflict_keys.shape[0])
    if n_groups == 0:
        return {"enabled": True, "n_conflicting_input_groups": 0, "n_rows_in_conflicts": 0, "examples": []}

    conflict_index = conflict_keys.index
    conflict_rows = sub.set_index(inputs).loc[conflict_index]
    n_rows = int(conflict_rows.shape[0])

    examples = []
    for k in list(conflict_index[:5]):
        if not isinstance(k, tuple):
            k = (k,)
        d = dict(zip(inputs, k))
        ys = sorted(list(set(conflict_rows.loc[k][outc].tolist())))
        examples.append({"inputs": d, "labels": ys})

    return {"enabled": True, "n_conflicting_input_groups": n_groups, "n_rows_in_conflicts": n_rows, "examples": examples}


# -----------------------------
# Level 1: audit file
# -----------------------------
def audit_keel_file(
    file_path: str,
    *,
    out_dir: Optional[str] = None,
    save: bool = True,
    tolerance: float = 1e-12,
    near_constant_threshold: float = 0.999,
    high_missingness_threshold: float = 0.30,
    high_cardinality_threshold: int = 50,
    robust_z_threshold: float = 3.5,
) -> Dict[str, Any]:
    """
    @brief Audit a single KEEL .dat file (Level 1).

    @param file_path Path to .dat file
    @param out_dir Output directory base for JSON results. If None and save=True, uses "<folder>/_audit".
    @param save Whether to save JSON result
    @return audit result dict
    """
    abs_path = os.path.abspath(file_path)
    folder = os.path.dirname(abs_path)
    basename = os.path.basename(abs_path)

    text = _read_text(abs_path)
    encoding_issues = _detect_encoding_issues(text)

    fold_info = _parse_fold_filename(basename)
    is_train = fold_info.split == "tra"
    is_test = fold_info.split == "tst"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("%")]
    relation, attributes, inputs, outputs, data_start_idx = _parse_keel_header(lines)

    attr_names = list(attributes.keys())
    df = _load_keel_data(lines, data_start_idx, attr_names)

    # defaults if header omitted inputs/outputs
    if not inputs and not outputs and len(attr_names) >= 1:
        outputs = [attr_names[-1]]
        inputs = attr_names[:-1]
    if not outputs and len(attr_names) >= 1:
        outputs = [attr_names[-1]]
    if not inputs:
        inputs = [c for c in attr_names if c not in outputs]

    n_total = len(attr_names)
    n_out = len(outputs)
    n_in = len(inputs)

    nominal_features = [k for k, v in attributes.items() if v["type"] == "nominal"]
    numeric_features = [k for k, v in attributes.items() if v["type"] in {"real", "integer"}]
    integer_features = [k for k, v in attributes.items() if v["type"] == "integer"]

    # missing + unknown
    missing_per_feature: Dict[str, int] = {}
    unknown_per_feature: Dict[str, List[str]] = {}
    suspicious_null_like_per_feature: Dict[str, List[str]] = {}

    for col in attr_names:
        ser = df[col].astype(str)
        missing_per_feature[col] = int(sum(_is_missing_token(x.strip()) for x in ser.tolist()))
        sus = sorted({x.strip() for x in ser.tolist() if _is_suspicious_null_like(x)})
        if sus:
            suspicious_null_like_per_feature[col] = sus

    missing_in_inputs = any(missing_per_feature.get(c, 0) > 0 for c in inputs)
    missing_in_outputs = any(missing_per_feature.get(c, 0) > 0 for c in outputs)

    # nominal observed vs header
    nominal_observed: Dict[str, List[str]] = {}
    nominal_not_in_header: Dict[str, List[str]] = {}
    nominal_in_header_not_observed: Dict[str, List[str]] = {}

    for col in nominal_features:
        allowed = attributes[col].get("nominal_allowed_header") or set()
        ser = df[col].astype(str).map(lambda x: x.strip())
        obs = sorted({x for x in ser.tolist() if (not _is_missing_token(x))})
        nominal_observed[col] = obs

        not_in_hdr = sorted([x for x in obs if x not in allowed]) if allowed else []
        if not_in_hdr:
            nominal_not_in_header[col] = not_in_hdr

        in_hdr_not_obs = sorted([x for x in allowed if x not in set(obs)]) if allowed else []
        if in_hdr_not_obs:
            nominal_in_header_not_observed[col] = in_hdr_not_obs

        unk = set()
        for x in obs:
            if ("?" in x) and (x != "?"):
                unk.add(x)
            if _is_suspicious_null_like(x):
                unk.add(x)
            if allowed and (x not in allowed):
                unk.add(x)
        if unk:
            unknown_per_feature[col] = sorted(list(unk))

    # numeric checks + stats
    numeric_minmax: Dict[str, Dict[str, Any]] = {}
    numeric_out_of_range: Dict[str, Dict[str, Any]] = {}
    numeric_minmax_mismatch: Dict[str, Dict[str, Any]] = {}
    numeric_stats: Dict[str, Dict[str, Any]] = {}
    numeric_parse: Dict[str, Dict[str, Any]] = {}
    numeric_outliers: Dict[str, Dict[str, Any]] = {}
    integer_integrity: Dict[str, Dict[str, Any]] = {}

    for col in numeric_features:
        x, unknown_tokens, n_nonmissing = _safe_float_array(df[col])
        if unknown_tokens:
            unknown_per_feature[col] = sorted(list(set(unknown_per_feature.get(col, [])) | set(unknown_tokens)))

        parse_rate = float(x.size / n_nonmissing) if n_nonmissing > 0 else float("nan")
        numeric_parse[col] = {"n_nonmissing": int(n_nonmissing), "n_parsed_numeric": int(x.size), "parse_rate": parse_rate}

        dmin = float(np.min(x)) if x.size > 0 else float("nan")
        dmax = float(np.max(x)) if x.size > 0 else float("nan")

        hmin = attributes[col].get("min_range_header")
        hmax = attributes[col].get("max_range_header")

        numeric_minmax[col] = {"data_min": dmin, "data_max": dmax, "header_min": hmin, "header_max": hmax}

        mismatch = {}
        if (hmin is not None) and np.isfinite(dmin):
            mismatch["min_mismatch"] = bool(not np.isclose(dmin, float(hmin), atol=tolerance, rtol=0.0))
        if (hmax is not None) and np.isfinite(dmax):
            mismatch["max_mismatch"] = bool(not np.isclose(dmax, float(hmax), atol=tolerance, rtol=0.0))
        if mismatch:
            numeric_minmax_mismatch[col] = mismatch

        if (hmin is not None) and (hmax is not None) and x.size > 0:
            lo = float(hmin) - tolerance
            hi = float(hmax) + tolerance
            mask = (x < lo) | (x > hi)
            if np.any(mask):
                bad_vals = sorted(list(set([float(v) for v in x[mask]])))[:50]
                numeric_out_of_range[col] = {"header_min": float(hmin), "header_max": float(hmax), "n_out_of_range": int(np.sum(mask)), "example_values": bad_vals}

        numeric_stats[col] = _robust_numeric_stats(x)
        numeric_outliers[col] = _robust_outlier_rate(x, z_threshold=robust_z_threshold)

    for col in integer_features:
        ser = df[col].astype(str).map(lambda x: str(x).strip())
        non_missing = ser[~ser.map(_is_missing_token)]
        if non_missing.empty:
            integer_integrity[col] = {"n_nonmissing": 0, "n_integer_like": 0, "integer_like_rate": float("nan")}
            continue
        n_nonmissing = int(len(non_missing))
        n_int_like = 0
        bad_tokens = set()
        for s in non_missing.tolist():
            try:
                v = float(s)
                if float(v).is_integer():
                    n_int_like += 1
            except Exception:
                bad_tokens.add(s)
        if bad_tokens:
            unknown_per_feature[col] = sorted(list(set(unknown_per_feature.get(col, [])) | bad_tokens))
        integer_integrity[col] = {"n_nonmissing": n_nonmissing, "n_integer_like": int(n_int_like), "integer_like_rate": float(n_int_like / n_nonmissing)}

    # class info
    class_info: Dict[str, Any] = {"output_features": outputs}
    if len(outputs) == 1 and outputs[0] in df.columns:
        y_raw = df[outputs[0]].astype(str).map(lambda x: x.strip())
        y_obs = [x for x in y_raw.tolist() if not _is_missing_token(x)]
        unique = sorted(list(set(y_obs)))
        class_type = "binary" if len(unique) == 2 else ("multiclass" if len(unique) > 2 else "unknown")

        counts = {k: int(v) for k, v in pd.Series(y_obs).value_counts().to_dict().items()}
        ir = None
        minority = None
        min_count = min(counts.values()) if counts else None
        if len(counts) >= 2:
            mx = max(counts.values())
            mn = min(counts.values())
            ir = float(mx / mn) if mn > 0 else float("inf")
            minority = min(counts, key=counts.get)

        class_info.update(
            {
                "class_type": class_type,
                "class_counts": counts,
                "imbalance_rate_ir": ir,
                "minority_class": minority,
                "n_classes": int(len(unique)),
                "min_class_count": min_count,
                "has_tiny_class_le_2": bool(min_count is not None and min_count <= 2),
            }
        )
    else:
        class_info["class_type"] = "multioutput"
        per_out = {}
        for outc in outputs:
            if outc not in df.columns:
                continue
            y_raw = df[outc].astype(str).map(lambda x: x.strip())
            y_obs = [x for x in y_raw.tolist() if not _is_missing_token(x)]
            counts = {k: int(v) for k, v in pd.Series(y_obs).value_counts().to_dict().items()}
            ir = None
            minority = None
            min_count = min(counts.values()) if counts else None
            if len(counts) >= 2:
                mx = max(counts.values())
                mn = min(counts.values())
                ir = float(mx / mn) if mn > 0 else float("inf")
                minority = min(counts, key=counts.get)
            per_out[outc] = {"class_counts": counts, "imbalance_rate_ir": ir, "minority_class": minority, "min_class_count": min_count, "has_tiny_class_le_2": bool(min_count is not None and min_count <= 2)}
        class_info["per_output"] = per_out

    # preprocess recommendations
    needs_one_hot = [c for c in inputs if c in nominal_features]

    needs_unit_interval = []
    for c in inputs:
        if c not in numeric_features:
            continue
        hmin = attributes[c].get("min_range_header")
        hmax = attributes[c].get("max_range_header")
        if (hmin is None) or (hmax is None):
            dmin = numeric_minmax.get(c, {}).get("data_min")
            dmax = numeric_minmax.get(c, {}).get("data_max")
            if np.isfinite(dmin) and np.isfinite(dmax):
                if (dmin < -tolerance) or (dmax > 1.0 + tolerance) or (not np.isclose(dmin, 0.0, atol=tolerance)) or (not np.isclose(dmax, 1.0, atol=tolerance)):
                    needs_unit_interval.append(c)
        else:
            if (not np.isclose(float(hmin), 0.0, atol=tolerance)) or (not np.isclose(float(hmax), 1.0, atol=tolerance)):
                needs_unit_interval.append(c)

    # unknown tokens summary
    unknown_any = bool(unknown_per_feature)
    unknown_tokens_global = sorted({t for lst in unknown_per_feature.values() for t in lst})

    # Quality checks
    duplicates_info = _duplicate_rows(df)
    conflicts_info = _label_conflicts(df, inputs=inputs, outputs=outputs)

    missingness_all = _feature_missingness(df, attr_names)
    high_missing = {k: v for k, v in missingness_all.items() if (isinstance(v, float) and np.isfinite(v) and v >= float(high_missingness_threshold))}

    const_info = _constant_and_near_constant(df, cols=inputs, near_constant_threshold=near_constant_threshold)

    nominal_quality = _high_cardinality_nominal(attributes, nominal_inputs=[c for c in inputs if c in nominal_features], observed=nominal_observed, high_cardinality_threshold=high_cardinality_threshold)

    one_hot_dim_est = _estimate_one_hot_dim(attributes, nominal_inputs=[c for c in inputs if c in nominal_features], observed=nominal_observed)

    # optional reuse sanity-check
    reuse_check = {"used_parse_keel_file": False}
    if _PARSE_KEEL_FILE is not None:
        try:
            meta, _, _, _ = _PARSE_KEEL_FILE(abs_path, one_hot_encode=False, normalize=False)
            reuse_check["used_parse_keel_file"] = True
            reuse_check["parse_keel_file_relation"] = meta.get("relation")
            reuse_check["parse_keel_file_num_instances"] = meta.get("num_instances")
            reuse_check["relation_matches"] = bool((meta.get("relation") or "") == (relation or ""))
            reuse_check["num_instances_matches"] = bool(int(meta.get("num_instances", -1)) == int(len(df)))
        except Exception as e:
            reuse_check["parse_keel_file_error"] = f"{type(e).__name__}: {e}"

    result: Dict[str, Any] = {
        "audit_version": "1.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 1,
        "relation": relation,
        "file": {"basename": basename, "folder": folder, "abs_path": abs_path, "split": fold_info.split, "is_train": bool(is_train), "is_test": bool(is_test)},
        "schema": {
            "n_features_total_including_outputs": int(n_total),
            "n_input_features": int(n_in),
            "n_output_features": int(n_out),
            "input_features": inputs,
            "output_features": outputs,
            "attributes": attributes,
            "schema_signature": _schema_signature(relation, attributes, inputs, outputs),
            "feature_types": {"numeric": numeric_features, "integer": integer_features, "nominal": nominal_features},
        },
        "dataset_shape": {"n_instances": int(len(df)), "n_bad_lines_column_mismatch": int(df.attrs.get("bad_lines_column_mismatch", 0))},
        "encoding_issues": encoding_issues,
        "data_quality": {
            "missing": {
                "missing_tokens": sorted(list(_MISSING_TOKENS)),
                "missing_in_inputs": bool(missing_in_inputs),
                "missing_in_outputs": bool(missing_in_outputs),
                "missing_per_feature_count": missing_per_feature,
                "missingness_fraction_per_feature": missingness_all,
                "high_missingness_threshold": float(high_missingness_threshold),
                "high_missingness_features": high_missing,
            },
            "unknown_tokens": {
                "has_unknown_tokens": bool(unknown_any),
                "unknown_tokens_global": unknown_tokens_global,
                "unknown_tokens_per_feature": unknown_per_feature,
                "suspicious_null_like_per_feature": suspicious_null_like_per_feature,
            },
            "duplicates": duplicates_info,
            "label_conflicts": conflicts_info,
            "nominal": {
                "observed_values_per_feature": nominal_observed,
                "values_in_data_not_in_header": nominal_not_in_header,
                "values_in_header_not_in_data": nominal_in_header_not_observed,
            },
            "numeric": {
                "minmax_data_vs_header": numeric_minmax,
                "minmax_mismatch_flags": numeric_minmax_mismatch,
                "out_of_range_vs_header": numeric_out_of_range,
                "numeric_parseability": numeric_parse,
                "integer_integrity": integer_integrity,
                "robust_outliers": numeric_outliers,
                "robust_z_threshold": float(robust_z_threshold),
            },
        },
        "class_info": class_info,
        "feature_quality": {"near_constant_threshold": float(near_constant_threshold), **const_info, **nominal_quality},
        "preprocess_recommendations": {
            "needs_unit_interval_scaling_0_1": sorted(list(set(needs_unit_interval))),
            "needs_one_hot_encoding": sorted(list(set(needs_one_hot))),
            "estimated_one_hot_output_dim": int(one_hot_dim_est),
        },
        "numeric_feature_stats": numeric_stats,
        "reuse_sanity_check": reuse_check,
    }

    if save:
        if out_dir is None:
            out_path = os.path.join(folder, "_audit", "files", f"{basename}.audit.json")
        else:
            ds_folder = _dataset_folder_name_from_path(folder)
            out_path = os.path.join(out_dir, ds_folder, "files", f"{basename}.audit.json")
        result["saved_to"] = _save_json(out_path, result)

    return result


def _safe_error_audit(file_path: str, error: Exception, *, out_dir: Optional[str], save: bool) -> Dict[str, Any]:
    """
    @brief Create an error audit payload for a file that cannot be audited.
    """
    abs_path = os.path.abspath(file_path)
    folder = os.path.dirname(abs_path)
    basename = os.path.basename(abs_path)
    fold_info = _parse_fold_filename(basename)

    payload = {
        "audit_version": "1.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 1,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "file": {"basename": basename, "folder": folder, "abs_path": abs_path, "split": fold_info.split, "is_train": bool(fold_info.split == "tra"), "is_test": bool(fold_info.split == "tst")},
    }

    if save:
        if out_dir is None:
            out_path = os.path.join(folder, "_audit", "files", f"{basename}.audit.json")
        else:
            ds_folder = _dataset_folder_name_from_path(folder)
            out_path = os.path.join(out_dir, ds_folder, "files", f"{basename}.audit.json")
        payload["saved_to"] = _save_json(out_path, payload)

    return payload


def _compute_leakage_metrics_from_hashes(train_file: str, test_file: str, *, inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
    """
    @brief Compute leakage between a train/test pair by intersection of hashes.
    """
    def load_hashes(fp: str, cols: List[str]) -> set:
        text = _read_text(fp)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("%")]
        relation, attributes, inps, outs, data_start = _parse_keel_header(lines)
        attr_names = list(attributes.keys())
        col_to_idx = {c: i for i, c in enumerate(attr_names)}
        idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
        hs = set()
        for ln in lines[data_start:]:
            parts = _split_csv_like(ln)
            if len(parts) != len(attr_names):
                continue
            picked = [parts[i].strip() for i in idxs]
            hs.add(_row_hash(picked))
        return hs

    cols_full = list(inputs) + [c for c in outputs if c not in inputs]
    ht_full = load_hashes(train_file, cols_full)
    hs_full = load_hashes(test_file, cols_full)
    inter_full = int(len(ht_full.intersection(hs_full)))

    ht_in = load_hashes(train_file, list(inputs))
    hs_in = load_hashes(test_file, list(inputs))
    inter_in = int(len(ht_in.intersection(hs_in)))

    return {
        "train_basename": os.path.basename(train_file),
        "test_basename": os.path.basename(test_file),
        "n_intersection_full_row": inter_full,
        "n_intersection_inputs_only": inter_in,
        "has_leakage_full_row": bool(inter_full > 0),
        "has_leakage_inputs_only": bool(inter_in > 0),
    }


def _class_presence_in_split(file_audit: Dict[str, Any]) -> Optional[set]:
    """
    @brief Extract class set from a level-1 file audit if available (single output).
    """
    try:
        ci = file_audit.get("class_info", {})
        if ci.get("class_type") == "multioutput":
            return None
        counts = ci.get("class_counts")
        if not isinstance(counts, dict):
            return None
        return set(counts.keys())
    except Exception:
        return None


def _safe_mean(nums: List[float]) -> float:
    vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _safe_min(nums: List[float]) -> float:
    vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(min(vals)) if vals else float("nan")


def _safe_max(nums: List[float]) -> float:
    vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(max(vals)) if vals else float("nan")


def _severity_score_from_flags(flags: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Compute a 0..100 severity score with breakdown.
    """
    weights = {
        "leakage_full_row": 60,
        "leakage_inputs_only": 40,
        "schema_drift": 40,
        "bad_lines": 30,
        "label_conflicts": 25,
        "nominal_values_not_in_header": 20,
        "numeric_out_of_range": 15,
        "missing_in_outputs": 20,
        "missing_in_inputs": 10,
        "tiny_class_le_2": 15,
        "high_missingness": 10,
        "constant_features": 5,
        "high_cardinality_nominal": 5,
        "multioutput": 15,
        "nonbinary_multiclass": 5,
    }

    breakdown = {}
    total = 0

    def add(name: str, cond: bool):
        nonlocal total
        if cond:
            w = int(weights.get(name, 0))
            breakdown[name] = w
            total += w

    add("leakage_full_row", bool(flags.get("has_leakage_full_row", False)))
    add("leakage_inputs_only", bool(flags.get("has_leakage_inputs_only", False)))
    add("schema_drift", bool(flags.get("has_schema_drift", False)))
    add("bad_lines", bool(flags.get("has_bad_lines", False)))
    add("label_conflicts", bool(flags.get("has_label_conflicts", False)))
    add("nominal_values_not_in_header", bool(flags.get("has_nominal_values_not_in_header", False)))
    add("numeric_out_of_range", bool(flags.get("has_numeric_out_of_range", False)))
    add("missing_in_outputs", bool(flags.get("has_missing_in_outputs", False)))
    add("missing_in_inputs", bool(flags.get("has_missing_in_inputs", False)))
    add("tiny_class_le_2", bool(flags.get("has_tiny_class_le_2", False)))
    add("high_missingness", bool(flags.get("has_high_missingness", False)))
    add("constant_features", bool(flags.get("has_constant_features", False)))
    add("high_cardinality_nominal", bool(flags.get("has_high_cardinality_nominal", False)))
    add("multioutput", bool(flags.get("is_multioutput", False)))
    add("nonbinary_multiclass", bool(flags.get("is_multiclass", False)))

    score = min(100, int(total))
    return {"score_0_100": score, "breakdown": breakdown, "weights": weights}


def _run_readiness(flags: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Compute SAFE/RISKY/BLOCKED with reasons.
    """
    blocked = []
    risky = []

    if flags.get("has_schema_drift", False):
        blocked.append("schema_drift")
    if flags.get("has_bad_lines", False):
        blocked.append("bad_lines_column_mismatch")
    if flags.get("has_leakage_full_row", False) or flags.get("has_leakage_inputs_only", False):
        blocked.append("train_test_leakage")
    if flags.get("is_multioutput", False):
        blocked.append("multioutput_target")
    if flags.get("has_missing_in_outputs", False):
        blocked.append("missing_in_outputs")

    if flags.get("has_missing_in_inputs", False):
        risky.append("missing_in_inputs")
    if flags.get("has_nominal_values_not_in_header", False):
        risky.append("nominal_values_not_in_header")
    if flags.get("has_numeric_out_of_range", False):
        risky.append("numeric_out_of_range_vs_header")
    if flags.get("has_tiny_class_le_2", False):
        risky.append("tiny_class_le_2")
    if flags.get("has_label_conflicts", False):
        risky.append("label_conflicts")
    if flags.get("has_high_missingness", False):
        risky.append("high_missingness_features")
    if flags.get("has_constant_features", False):
        risky.append("constant_features")
    if flags.get("has_high_cardinality_nominal", False):
        risky.append("high_cardinality_nominal")

    status = "SAFE"
    if blocked:
        status = "BLOCKED"
    elif risky:
        status = "RISKY"

    return {"status": status, "blocked_reasons": blocked, "risky_reasons": risky}


def _aggregate_dataset_key_summary(
    *,
    folder_name: str,
    dataset_key: str,
    group_basenames: List[str],
    per_file_by_name: Dict[str, Dict[str, Any]],
    leakage_reports: List[Dict[str, Any]],
    schema_drift_entry: Optional[Dict[str, Any]],
    class_presence_issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    @brief Build an executive dataset-level summary for a dataset_key.
    """
    audits = [per_file_by_name.get(bn) for bn in group_basenames]
    audits = [a for a in audits if isinstance(a, dict)]
    ok_audits = [a for a in audits if a.get("status") != "error"]

    rep = ok_audits[0] if ok_audits else None
    rep_outputs = rep.get("schema", {}).get("output_features", []) if rep else []
    rep_n_inputs = int(rep.get("schema", {}).get("n_input_features", 0)) if rep else 0
    rep_n_outputs = int(rep.get("schema", {}).get("n_output_features", 0)) if rep else 0

    train_ns, test_ns, all_ns = [], [], []
    bad_lines_any = False
    for a in ok_audits:
        n = int(a.get("dataset_shape", {}).get("n_instances", 0))
        all_ns.append(float(n))
        if a.get("file", {}).get("is_train"):
            train_ns.append(float(n))
        if a.get("file", {}).get("is_test"):
            test_ns.append(float(n))
        if int(a.get("dataset_shape", {}).get("n_bad_lines_column_mismatch", 0)) > 0:
            bad_lines_any = True

    agg_audits = [a for a in ok_audits if a.get("file", {}).get("is_train")] or ok_audits

    class_type = rep.get("class_info", {}).get("class_type") if rep else None
    n_classes = rep.get("class_info", {}).get("n_classes") if rep else None
    class_counts_agg: Dict[str, int] = {}
    min_class_count = None
    ir = None
    if rep_outputs and len(rep_outputs) == 1:
        for a in agg_audits:
            counts = a.get("class_info", {}).get("class_counts")
            if isinstance(counts, dict):
                for k, v in counts.items():
                    class_counts_agg[k] = int(class_counts_agg.get(k, 0) + int(v))
        if class_counts_agg:
            min_class_count = min(class_counts_agg.values())
            if len(class_counts_agg) >= 2:
                mx = max(class_counts_agg.values())
                mn = min(class_counts_agg.values())
                ir = float(mx / mn) if mn > 0 else float("inf")

    # Aggregate flags and quality
    constant_feats, high_missing_feats, high_card_nom = set(), set(), set()
    max_ohe_dim = 0

    flags = {
        "has_leakage_full_row": False,
        "has_leakage_inputs_only": False,
        "has_schema_drift": bool(schema_drift_entry is not None and schema_drift_entry.get("drift_files")),
        "has_bad_lines": bool(bad_lines_any),
        "has_label_conflicts": False,
        "has_nominal_values_not_in_header": False,
        "has_numeric_out_of_range": False,
        "has_missing_in_outputs": False,
        "has_missing_in_inputs": False,
        "has_tiny_class_le_2": False,
        "has_high_missingness": False,
        "has_constant_features": False,
        "has_high_cardinality_nominal": False,
        "is_multioutput": False,
        "is_multiclass": False,
    }

    heavy_tail_counts = []
    outlier_rates = []

    for a in agg_audits:
        flags["has_missing_in_inputs"] |= bool(a.get("data_quality", {}).get("missing", {}).get("missing_in_inputs", False))
        flags["has_missing_in_outputs"] |= bool(a.get("data_quality", {}).get("missing", {}).get("missing_in_outputs", False))

        nom_bad = a.get("data_quality", {}).get("nominal", {}).get("values_in_data_not_in_header", {})
        if isinstance(nom_bad, dict) and nom_bad:
            flags["has_nominal_values_not_in_header"] = True

        oor = a.get("data_quality", {}).get("numeric", {}).get("out_of_range_vs_header", {})
        if isinstance(oor, dict) and oor:
            flags["has_numeric_out_of_range"] = True

        lc = a.get("data_quality", {}).get("label_conflicts", {})
        if isinstance(lc, dict) and lc.get("enabled") and int(lc.get("n_conflicting_input_groups", 0)) > 0:
            flags["has_label_conflicts"] = True

        ci = a.get("class_info", {})
        if isinstance(ci, dict):
            if ci.get("class_type") == "multioutput":
                flags["is_multioutput"] = True
            if ci.get("class_type") == "multiclass":
                flags["is_multiclass"] = True
            if bool(ci.get("has_tiny_class_le_2", False)):
                flags["has_tiny_class_le_2"] = True

        cf = a.get("feature_quality", {}).get("constant_features", [])
        if isinstance(cf, list) and cf:
            constant_feats |= set(cf)

        hm = a.get("data_quality", {}).get("missing", {}).get("high_missingness_features", {})
        if isinstance(hm, dict) and hm:
            flags["has_high_missingness"] = True
            high_missing_feats |= set(hm.keys())

        hcn = a.get("feature_quality", {}).get("high_cardinality_nominal", [])
        if isinstance(hcn, list) and hcn:
            flags["has_high_cardinality_nominal"] = True
            high_card_nom |= set([x.get("feature") for x in hcn if isinstance(x, dict) and x.get("feature")])

        try:
            max_ohe_dim = max(max_ohe_dim, int(a.get("preprocess_recommendations", {}).get("estimated_one_hot_output_dim", 0)))
        except Exception:
            pass

        nstats = a.get("numeric_feature_stats", {})
        if isinstance(nstats, dict) and nstats:
            ht = 0
            for _, st in nstats.items():
                if isinstance(st, dict) and bool(st.get("heavy_tail", False)):
                    ht += 1
            heavy_tail_counts.append(float(ht))

        ro = a.get("data_quality", {}).get("numeric", {}).get("robust_outliers", {})
        if isinstance(ro, dict) and ro:
            rates = []
            for _, rr in ro.items():
                if isinstance(rr, dict):
                    r = rr.get("outlier_rate")
                    if isinstance(r, (int, float)) and np.isfinite(r):
                        rates.append(float(r))
            if rates:
                outlier_rates.append(float(sum(rates) / len(rates)))

    flags["has_constant_features"] = bool(len(constant_feats) > 0)

    # Leakage aggregation for this dataset_key
    leaks = [x for x in leakage_reports if x.get("dataset_key") == dataset_key]
    worst_full, worst_inputs = 0, 0
    worst_full_rate, worst_inputs_rate = 0.0, 0.0
    worst_fold = None
    for l in leaks:
        nf = int(l.get("n_intersection_full_row", 0))
        ni = int(l.get("n_intersection_inputs_only", 0))
        rf = l.get("rate_full_row_over_test")
        ri = l.get("rate_inputs_only_over_test")
        rf = float(rf) if isinstance(rf, (int, float)) and np.isfinite(rf) else 0.0
        ri = float(ri) if isinstance(ri, (int, float)) and np.isfinite(ri) else 0.0
        if (nf > worst_full) or (ni > worst_inputs) or (rf > worst_full_rate) or (ri > worst_inputs_rate):
            worst_fold = l.get("fold_index")
        worst_full = max(worst_full, nf)
        worst_inputs = max(worst_inputs, ni)
        worst_full_rate = max(worst_full_rate, rf)
        worst_inputs_rate = max(worst_inputs_rate, ri)

    flags["has_leakage_full_row"] = bool(worst_full > 0)
    flags["has_leakage_inputs_only"] = bool(worst_inputs > 0)

    drift_reasons = schema_drift_entry.get("drift_reasons_summary", {}) if isinstance(schema_drift_entry, dict) else {}

    severity = _severity_score_from_flags(flags)
    readiness = _run_readiness(flags)

    return {
        "folder": folder_name,
        "dataset_key": dataset_key,
        "shape_summary": {
            "n_instances_all": {"min": _safe_min(all_ns), "mean": _safe_mean(all_ns), "max": _safe_max(all_ns)},
            "n_instances_train": {"min": _safe_min(train_ns), "mean": _safe_mean(train_ns), "max": _safe_max(train_ns)},
            "n_instances_test": {"min": _safe_min(test_ns), "mean": _safe_mean(test_ns), "max": _safe_max(test_ns)},
        },
        "schema_summary": {"n_input_features": rep_n_inputs, "n_output_features": rep_n_outputs, "estimated_one_hot_dim_max": int(max_ohe_dim)},
        "class_summary": {
            "class_type": class_type,
            "n_classes": n_classes,
            "class_counts_train_aggregate": class_counts_agg,
            "imbalance_rate_ir_train_aggregate": ir,
            "min_class_count_train_aggregate": min_class_count,
        },
        "leakage_summary": {
            "worst_fold_index": worst_fold,
            "worst_n_intersection_full_row": int(worst_full),
            "worst_n_intersection_inputs_only": int(worst_inputs),
            "worst_rate_full_row_over_test": float(worst_full_rate),
            "worst_rate_inputs_only_over_test": float(worst_inputs_rate),
        },
        "schema_drift_summary": {"has_schema_drift": bool(flags["has_schema_drift"]), "drift_reasons_summary": drift_reasons},
        "cv_class_presence_issues": [x for x in class_presence_issues if x.get("dataset_key") == dataset_key],
        "feature_quality_summary": {
            "constant_features_union": sorted(list(constant_feats)),
            "high_missingness_features_union": sorted(list(high_missing_feats)),
            "high_cardinality_nominal_union": sorted(list(high_card_nom)),
        },
        "tail_outlier_summary": {
            "heavy_tail_feature_count_mean": _safe_mean(heavy_tail_counts),
            "heavy_tail_feature_count_max": _safe_max(heavy_tail_counts),
            "outlier_rate_mean_over_numeric_features_mean": _safe_mean(outlier_rates),
            "outlier_rate_mean_over_numeric_features_max": _safe_max(outlier_rates),
        },
        "flags": flags,
        "run_readiness": readiness,
        "severity": severity,
    }


# -----------------------------
# Level 2: audit folder
# -----------------------------
def audit_keel_folder(folder_path: str, *, out_dir: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    """
    @brief Audit a folder containing KEEL .dat files (Level 2) + executive summaries.
    """
    abs_folder = os.path.abspath(folder_path)
    if not os.path.isdir(abs_folder):
        raise FileNotFoundError(f"Folder not found: {abs_folder}")

    folder_name = _dataset_folder_name_from_path(abs_folder)

    files = sorted([f for f in os.listdir(abs_folder) if f.lower().endswith(".dat")])
    dat_abs = [os.path.join(abs_folder, f) for f in files]

    groups: Dict[str, List[_FoldNameInfo]] = {}
    for f in files:
        info = _parse_fold_filename(f)
        groups.setdefault(info.dataset_key, []).append(info)

    dataset_keys = sorted(groups.keys())
    dataset_summaries = []
    folder_has_cv = False
    cv_k_values = set()

    file_audits = []
    per_file_by_name: Dict[str, Dict[str, Any]] = {}

    for fp in dat_abs:
        try:
            a = audit_keel_file(fp, out_dir=out_dir, save=save)
        except Exception as e:
            a = _safe_error_audit(fp, e, out_dir=out_dir, save=save)
        file_audits.append(a)
        per_file_by_name[os.path.basename(fp)] = a

    schema_drift = []
    leakage_reports = []
    split_class_presence_issues = []

    for ds_key in dataset_keys:
        infos = groups[ds_key]
        splits = sorted({i.split for i in infos if i.split is not None})
        looks_like_folds = ("tra" in splits) or ("tst" in splits)

        group_basenames = [f for f in files if _parse_fold_filename(f).dataset_key == ds_key]

        base_sig = None
        base_file = None
        for bn in group_basenames:
            a = per_file_by_name.get(bn)
            if not a or a.get("status") == "error":
                continue
            sig = a.get("schema", {}).get("schema_signature")
            if sig is None:
                continue
            base_sig = sig
            base_file = bn
            break

        drift_files = []
        drift_reasons_hist: Dict[str, int] = {}
        drift_entry = None
        if base_sig is not None:
            for bn in group_basenames:
                a = per_file_by_name.get(bn)
                if not a or a.get("status") == "error":
                    continue
                sig = a.get("schema", {}).get("schema_signature")
                if sig is None:
                    continue
                if sig != base_sig:
                    reasons = _diff_schema_signatures(base_sig, sig)
                    drift_files.append({"file": bn, "differs_from": base_file, "drift_reasons": reasons})
                    for r in reasons:
                        drift_reasons_hist[r] = int(drift_reasons_hist.get(r, 0) + 1)

        if drift_files:
            drift_entry = {"dataset_key": ds_key, "base_file": base_file, "drift_files": drift_files, "drift_reasons_summary": drift_reasons_hist}
            schema_drift.append(drift_entry)

        ds_sum: Dict[str, Any] = {
            "dataset_key": ds_key,
            "n_files": int(len(infos)),
            "has_train_test_splits": bool(looks_like_folds),
            "schema_drift": {"has_drift": bool(len(drift_files) > 0), "drift_files": drift_files, "drift_reasons_summary": drift_reasons_hist},
        }

        if looks_like_folds:
            folder_has_cv = True
            ks = sorted({i.k_folds for i in infos if i.k_folds is not None})
            k = ks[0] if ks else None
            if k is not None:
                cv_k_values.add(int(k))

            fold_indices = sorted({i.fold_index for i in infos if i.fold_index is not None})
            ds_sum["cv"] = {"kind": "predefined_folds", "k_folds_inferred": k, "fold_indices_found": fold_indices}

            problems = []
            if k is not None:
                expected = list(range(1, int(k) + 1))
                missing_folds = sorted([x for x in expected if x not in set(fold_indices)])
                if missing_folds:
                    problems.append({"missing_fold_indices": missing_folds})

            by_fold: Dict[int, set] = {}
            for i in infos:
                if i.fold_index is None or i.split is None:
                    continue
                by_fold.setdefault(int(i.fold_index), set()).add(i.split)
            missing_pairs = sorted([fi for fi, ss in by_fold.items() if not (("tra" in ss) and ("tst" in ss))])
            if missing_pairs:
                problems.append({"folds_missing_tra_or_tst": missing_pairs})

            if _DISCOVER_KEEL_CV_FOLDS is not None:
                try:
                    pairs = _DISCOVER_KEEL_CV_FOLDS(abs_folder)
                    ds_sum["cv"]["discover_keel_cv_folds_pairs"] = int(len(pairs))
                except Exception as e:
                    ds_sum["cv"]["discover_keel_cv_folds_error"] = f"{type(e).__name__}: {e}"

            ds_sum["cv"]["naming_validation_problems"] = problems

            # leakage + class presence per fold
            rep_bn = base_file if base_file is not None else (group_basenames[0] if group_basenames else None)
            rep_a = per_file_by_name.get(rep_bn) if rep_bn else None
            rep_inputs = rep_a.get("schema", {}).get("input_features") if rep_a else []
            rep_outputs = rep_a.get("schema", {}).get("output_features") if rep_a else []

            for fi in sorted([x for x in fold_indices if x is not None]):
                tra_bn = f"{ds_key}-{k}-{fi}tra.dat" if k is not None else None
                tst_bn = f"{ds_key}-{k}-{fi}tst.dat" if k is not None else None

                if tra_bn not in per_file_by_name:
                    cand = [bn for bn in group_basenames if bn.lower().endswith(f"-{fi}tra.dat".lower())]
                    tra_bn = cand[0] if cand else tra_bn
                if tst_bn not in per_file_by_name:
                    cand = [bn for bn in group_basenames if bn.lower().endswith(f"-{fi}tst.dat".lower())]
                    tst_bn = cand[0] if cand else tst_bn

                if tra_bn in per_file_by_name and tst_bn in per_file_by_name:
                    tra_fp = os.path.join(abs_folder, tra_bn)
                    tst_fp = os.path.join(abs_folder, tst_bn)
                    train_n = int(per_file_by_name[tra_bn].get("dataset_shape", {}).get("n_instances", 0))
                    test_n = int(per_file_by_name[tst_bn].get("dataset_shape", {}).get("n_instances", 0))

                    try:
                        leak = _compute_leakage_metrics_from_hashes(tra_fp, tst_fp, inputs=rep_inputs, outputs=rep_outputs)
                        leak["dataset_key"] = ds_key
                        leak["fold_index"] = int(fi)
                        leak["train_n_rows"] = train_n
                        leak["test_n_rows"] = test_n
                        leak["rate_full_row_over_test"] = float(leak["n_intersection_full_row"] / test_n) if test_n > 0 else float("nan")
                        leak["rate_inputs_only_over_test"] = float(leak["n_intersection_inputs_only"] / test_n) if test_n > 0 else float("nan")
                        leakage_reports.append(leak)
                    except Exception as e:
                        leakage_reports.append({"dataset_key": ds_key, "fold_index": int(fi), "train_basename": tra_bn, "test_basename": tst_bn, "train_n_rows": train_n, "test_n_rows": test_n, "error": f"{type(e).__name__}: {e}"})

                    tra_set = _class_presence_in_split(per_file_by_name.get(tra_bn, {}))
                    tst_set = _class_presence_in_split(per_file_by_name.get(tst_bn, {}))
                    if (tra_set is not None) and (tst_set is not None):
                        missing_in_train = sorted(list(tst_set - tra_set))
                        missing_in_test = sorted(list(tra_set - tst_set))
                        if missing_in_train or missing_in_test:
                            split_class_presence_issues.append({"dataset_key": ds_key, "fold_index": int(fi), "train_basename": tra_bn, "test_basename": tst_bn, "classes_missing_in_train": missing_in_train, "classes_missing_in_test": missing_in_test})
        else:
            ds_sum["cv"] = {"kind": "none"}

        dataset_summaries.append(ds_sum)

    # folder-level lists
    files_with_errors = []
    datasets_with_missing = []
    datasets_with_nonbinary = []
    datasets_with_numeric_out_of_range = []
    datasets_with_nominal_unknown = []
    datasets_with_leakage = []
    datasets_with_label_conflicts = []
    datasets_with_constant_features = []
    datasets_with_high_missingness = []
    datasets_with_high_cardinality_nominal = []
    datasets_with_tiny_class = []

    for a in file_audits:
        if a.get("status") == "error":
            files_with_errors.append({"file": a["file"]["basename"], "error": a["error"]})
            continue

        ds_name = a.get("relation") or a["file"]["basename"]

        if a["data_quality"]["missing"]["missing_in_inputs"] or a["data_quality"]["missing"]["missing_in_outputs"]:
            datasets_with_missing.append(ds_name)

        ci = a.get("class_info", {})
        if ci.get("class_type") not in (None, "binary"):
            datasets_with_nonbinary.append(ds_name)

        if ci.get("has_tiny_class_le_2") is True:
            datasets_with_tiny_class.append(ds_name)

        num_oor = a["data_quality"]["numeric"]["out_of_range_vs_header"]
        if num_oor:
            datasets_with_numeric_out_of_range.append({"dataset": ds_name, "file": a["file"]["basename"], "features": sorted(list(num_oor.keys()))})

        nom_bad = a["data_quality"]["nominal"]["values_in_data_not_in_header"]
        if nom_bad:
            datasets_with_nominal_unknown.append({"dataset": ds_name, "file": a["file"]["basename"], "features": sorted(list(nom_bad.keys()))})

        lc = a["data_quality"]["label_conflicts"]
        if lc.get("enabled") and lc.get("n_conflicting_input_groups", 0) > 0:
            datasets_with_label_conflicts.append({"dataset": ds_name, "file": a["file"]["basename"], "n_conflicting_groups": lc["n_conflicting_input_groups"]})

        cf = a.get("feature_quality", {}).get("constant_features", [])
        if cf:
            datasets_with_constant_features.append({"dataset": ds_name, "file": a["file"]["basename"], "features": cf})

        hm = a["data_quality"]["missing"].get("high_missingness_features", {})
        if hm:
            datasets_with_high_missingness.append({"dataset": ds_name, "file": a["file"]["basename"], "features": list(hm.keys())})

        hcn = a.get("feature_quality", {}).get("high_cardinality_nominal", [])
        if hcn:
            datasets_with_high_cardinality_nominal.append({"dataset": ds_name, "file": a["file"]["basename"], "features": [x.get("feature") for x in hcn if isinstance(x, dict)]})

    for leak in leakage_reports:
        if leak.get("has_leakage_full_row") or leak.get("has_leakage_inputs_only"):
            datasets_with_leakage.append(leak)

    # dataset_key executive summaries
    dataset_key_summaries = []
    for ds_key in dataset_keys:
        group_basenames = [f for f in files if _parse_fold_filename(f).dataset_key == ds_key]
        drift_entry = next((x for x in schema_drift if x.get("dataset_key") == ds_key), None)
        dataset_key_summaries.append(
            _aggregate_dataset_key_summary(
                folder_name=folder_name,
                dataset_key=ds_key,
                group_basenames=group_basenames,
                per_file_by_name=per_file_by_name,
                leakage_reports=leakage_reports,
                schema_drift_entry=drift_entry,
                class_presence_issues=split_class_presence_issues,
            )
        )

    result: Dict[str, Any] = {
        "audit_version": "1.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 2,
        "folder": {"abs_path": abs_folder, "name": folder_name, "n_dat_files": int(len(files)), "dat_files": files},
        "datasets_detected": {"count": int(len(dataset_keys)), "keys": dataset_keys, "has_cv_any": bool(folder_has_cv), "cv_k_values_inferred": sorted(list(cv_k_values)), "per_dataset": dataset_summaries},
        "schema_consistency": {"has_schema_drift_any": bool(len(schema_drift) > 0), "schema_drift_datasets": schema_drift},
        "cv_leakage": {"has_leakage_any": bool(any((x.get("has_leakage_full_row") or x.get("has_leakage_inputs_only")) for x in leakage_reports)), "leakage_reports": leakage_reports},
        "cv_class_presence": {"has_class_presence_issues_any": bool(len(split_class_presence_issues) > 0), "issues": split_class_presence_issues},
        "executive_dataset_summaries": dataset_key_summaries,
        "high_level_flags": {
            "has_errors_any": bool(len(files_with_errors) > 0),
            "has_missing_any": bool(len(datasets_with_missing) > 0),
            "has_nonbinary_any": bool(len(datasets_with_nonbinary) > 0),
            "has_numeric_out_of_range_any": bool(len(datasets_with_numeric_out_of_range) > 0),
            "has_nominal_unknown_any": bool(len(datasets_with_nominal_unknown) > 0),
            "has_schema_drift_any": bool(len(schema_drift) > 0),
            "has_leakage_any": bool(len(datasets_with_leakage) > 0),
            "has_label_conflicts_any": bool(len(datasets_with_label_conflicts) > 0),
            "has_constant_features_any": bool(len(datasets_with_constant_features) > 0),
            "has_high_missingness_any": bool(len(datasets_with_high_missingness) > 0),
            "has_high_cardinality_nominal_any": bool(len(datasets_with_high_cardinality_nominal) > 0),
            "has_tiny_class_any": bool(len(datasets_with_tiny_class) > 0),
        },
        "high_level_lists": {
            "files_with_errors": files_with_errors,
            "datasets_with_missing": sorted(list(set(datasets_with_missing))),
            "datasets_with_nonbinary_or_multioutput": sorted(list(set(datasets_with_nonbinary))),
            "datasets_with_tiny_class_le_2": sorted(list(set(datasets_with_tiny_class))),
            "datasets_with_numeric_out_of_range": datasets_with_numeric_out_of_range,
            "datasets_with_nominal_values_not_in_header": datasets_with_nominal_unknown,
            "datasets_with_schema_drift": schema_drift,
            "datasets_with_leakage": datasets_with_leakage,
            "datasets_with_label_conflicts": datasets_with_label_conflicts,
            "datasets_with_constant_features": datasets_with_constant_features,
            "datasets_with_high_missingness": datasets_with_high_missingness,
            "datasets_with_high_cardinality_nominal": datasets_with_high_cardinality_nominal,
        },
    }

    if save:
        if out_dir is None:
            out_path = os.path.join(abs_folder, "_audit", "folder.audit.json")
        else:
            out_path = os.path.join(out_dir, folder_name, "folder.audit.json")
        result["saved_to"] = _save_json(out_path, result)

    return result


# -----------------------------
# Level 3: audit root
# -----------------------------
def audit_keel_root(root_folder: str, *, out_dir: Optional[str] = None, save: bool = True, top_n_offenders: int = 20) -> Dict[str, Any]:
    """
    @brief Audit a root folder containing dataset folders (Level 3) + executive summaries.
    """
    abs_root = os.path.abspath(root_folder)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Root folder not found: {abs_root}")

    subdirs = sorted([os.path.join(abs_root, d) for d in os.listdir(abs_root) if os.path.isdir(os.path.join(abs_root, d)) and not d.startswith(".")])
    out_base = out_dir or os.path.join(abs_root, "_audit")

    per_folder = []
    cv_k_union = set()
    has_cv_flags = []

    # global lists
    global_errors = []
    global_missing = []
    global_nonbinary = []
    global_tiny_class = []
    global_num_oor = []
    global_nom_unknown = []
    global_schema_drift = []
    global_leakage = []
    global_label_conflicts = []
    global_constant = []
    global_high_missing = []
    global_high_card_nom = []

    executive_rows = []

    for d in subdirs:
        fr = audit_keel_folder(d, out_dir=out_base, save=save)
        per_folder.append(fr)

        has_cv_flags.append(bool(fr["datasets_detected"]["has_cv_any"]))
        for k in fr["datasets_detected"].get("cv_k_values_inferred", []):
            cv_k_union.add(int(k))

        folder_name = fr.get("folder", {}).get("name", os.path.basename(d))

        for it in fr["high_level_lists"]["files_with_errors"]:
            global_errors.append({"folder": folder_name, **it})

        for ds in fr["high_level_lists"]["datasets_with_missing"]:
            global_missing.append({"folder": folder_name, "dataset": ds})

        for ds in fr["high_level_lists"]["datasets_with_nonbinary_or_multioutput"]:
            global_nonbinary.append({"folder": folder_name, "dataset": ds})

        for ds in fr["high_level_lists"]["datasets_with_tiny_class_le_2"]:
            global_tiny_class.append({"folder": folder_name, "dataset": ds})

        for it in fr["high_level_lists"]["datasets_with_numeric_out_of_range"]:
            global_num_oor.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_nominal_values_not_in_header"]:
            global_nom_unknown.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_schema_drift"]:
            global_schema_drift.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_leakage"]:
            global_leakage.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_label_conflicts"]:
            global_label_conflicts.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_constant_features"]:
            global_constant.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_high_missingness"]:
            global_high_missing.append({"folder": folder_name, **it})

        for it in fr["high_level_lists"]["datasets_with_high_cardinality_nominal"]:
            global_high_card_nom.append({"folder": folder_name, **it})

        for row in fr.get("executive_dataset_summaries", []):
            executive_rows.append(row)

    all_same_cv = (len(set(has_cv_flags)) <= 1)

    offenders = []
    for r in executive_rows:
        sev = r.get("severity", {}).get("score_0_100")
        if isinstance(sev, (int, float)):
            offenders.append(
                {
                    "folder": r.get("folder"),
                    "dataset_key": r.get("dataset_key"),
                    "severity_score_0_100": int(sev),
                    "run_status": r.get("run_readiness", {}).get("status"),
                    "severity_breakdown": r.get("severity", {}).get("breakdown", {}),
                    "blocked_reasons": r.get("run_readiness", {}).get("blocked_reasons", []),
                    "risky_reasons": r.get("run_readiness", {}).get("risky_reasons", []),
                }
            )
    offenders_sorted = sorted(offenders, key=lambda x: x.get("severity_score_0_100", 0), reverse=True)
    top_offenders = offenders_sorted[: int(top_n_offenders)]

    issue_counters = {
        "blocked": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "BLOCKED")),
        "risky": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "RISKY")),
        "safe": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "SAFE")),
        "has_leakage": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_leakage_inputs_only") or r.get("flags", {}).get("has_leakage_full_row"))),
        "has_schema_drift": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_schema_drift"))),
        "has_missing_outputs": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_missing_in_outputs"))),
        "has_label_conflicts": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_label_conflicts"))),
    }

    result: Dict[str, Any] = {
        "audit_version": "1.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 3,
        "root": {"abs_path": abs_root, "n_dataset_folders": int(len(subdirs)), "dataset_folders": subdirs},
        "cv_consistency": {"all_same_cv_type": bool(all_same_cv), "has_cv_flags_per_folder": has_cv_flags, "cv_k_values_union": sorted(list(cv_k_union))},
        "executive_summary": {"issue_counters": issue_counters, "top_offenders": top_offenders, "per_dataset": executive_rows},
        "global_summary": {
            "has_errors_any": bool(len(global_errors) > 0),
            "files_with_errors": global_errors,
            "has_missing_any": bool(len(global_missing) > 0),
            "datasets_with_missing": global_missing,
            "has_nonbinary_any": bool(len(global_nonbinary) > 0),
            "datasets_not_binary_or_multioutput": global_nonbinary,
            "has_tiny_class_any": bool(len(global_tiny_class) > 0),
            "datasets_with_tiny_class_le_2": global_tiny_class,
            "datasets_with_numeric_out_of_range": global_num_oor,
            "datasets_with_nominal_values_not_in_header": global_nom_unknown,
            "has_schema_drift_any": bool(len(global_schema_drift) > 0),
            "datasets_with_schema_drift": global_schema_drift,
            "has_leakage_any": bool(len(global_leakage) > 0),
            "datasets_with_leakage": global_leakage,
            "has_label_conflicts_any": bool(len(global_label_conflicts) > 0),
            "datasets_with_label_conflicts": global_label_conflicts,
            "datasets_with_constant_features": global_constant,
            "datasets_with_high_missingness": global_high_missing,
            "datasets_with_high_cardinality_nominal": global_high_card_nom,
        },
    }

    if save:
        out_path = os.path.join(out_base, "root.audit.json")
        result["saved_to"] = _save_json(out_path, result)

    return result


__all__ = ["audit_keel_file", "audit_keel_folder", "audit_keel_root"]
