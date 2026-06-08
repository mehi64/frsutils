"""
@file KEEL_Audit_Utility.py
@brief KEEL Dataset Auditor Utility (File/Folder/Root level)

A utility module for auditing KEEL .dat datasets before feeding them into FRsutils pipelines.
It generates structured metadata + data-quality diagnostics at 3 levels:

- Level 1: Single file audit (one .dat file)
- Level 2: Folder audit (a dataset folder with one or many .dat files and/or predefined folds)
- Level 3: Root audit (root folder containing many dataset folders)

##############################################
# ✅ Quick Summary of Features
# - audit_keel_file(...)         -> Full audit of one .dat file + saves JSON
# - audit_keel_folder(...)       -> Audits all .dat files in a folder + detects CV structure
# - audit_keel_root(...)         -> Audits all dataset folders under a root + produces global summary
#
# Core checks (high ROI for Level 3):
# - Schema consistency across folds (header drift / column order drift)
# - Duplicate rows within files (and ratio)
# - Leakage between train/test files of the same fold (identical rows or identical inputs)
# - Label conflicts: identical inputs mapped to multiple labels
# - Low-quality features: constant/near-constant, high missingness, high-cardinality nominal
#
# Robust stats per numeric feature:
# - mean, std, median, MAD, sigma≈1.4826*MAD, quantiles, skewness, kurtosis, heavy-tail flag
# - robust outlier-rate via robust z-score

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

# 2) Audit a dataset folder (maybe predefined folds):
# from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import audit_keel_folder
# folder_res = audit_keel_folder(".../iris0_folds", out_dir="out_audit")

# 3) Audit a root folder containing many dataset folders:
# from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import audit_keel_root
# root_res = audit_keel_root(".../KEEL_imbalanced_root", out_dir="out_audit")

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

    # discover_keel_cv_folds
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

    # parse_keel_file
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
# tokens that are often "almost missing" but user defined only ? and <null> as missing.
# we keep these as "unknown/suspicious" unless user later decides otherwise:
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


# -----------------------------
# KEEL parsing (raw, no encoding)
# -----------------------------
def _read_text(file_path: str) -> str:
    """
    @brief Read a file as UTF-8 best-effort (replacement for bad bytes).

    @param file_path Path to file
    @return file content as string
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_nonempty_lines(file_path: str) -> List[str]:
    """
    @brief Read file and return stripped non-empty lines excluding KEEL comments (% ...).
    """
    text = _read_text(file_path)
    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("%"):
            continue
        out.append(s)
    return out


def _detect_encoding_issues(text: str) -> Dict[str, Any]:
    """
    @brief Detect encoding-related and control character issues.

    Reports:
    - presence of Unicode replacement char (�)
    - count of control chars excluding common whitespace

    @param text Raw file text
    @return dict with flags
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

    Supported:
    - integer [min,max]
    - real [min,max]
    - {v1,v2,...}  (nominal)
    """
    defn_str = defn.strip()

    # nominal
    if defn_str.startswith("{") and defn_str.endswith("}"):
        inside = defn_str[1:-1]
        values = [v.strip() for v in inside.split(",") if v.strip()]
        return "nominal", None, None, set(values)

    low = defn_str.lower()

    # numeric with optional range
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
    @brief Parse header fields: relation, attributes, inputs, outputs, and data section start index.

    @return (relation, attributes, inputs, outputs, data_start_idx)
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
            # allow spaces inside: split max 2 pieces after name
            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) < 3:
                raise ValueError(f"Malformed @attribute line: {line}")
            name = parts[1].strip()
            defn = parts[2].strip()
            t, mn, mx, allowed = _parse_attribute_definition(defn)
            attributes[name] = {
                "type": t,  # real/integer/nominal
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
    @brief Split a KEEL data line by comma while trimming spaces around tokens.
    """
    return re.split(r"\s*,\s*", line.strip())


def _load_keel_data(lines: List[str], data_start_idx: int, attribute_names: List[str]) -> pd.DataFrame:
    """
    @brief Load KEEL @data section into a DataFrame of raw strings.
    """
    rows = []
    bad_lines = 0
    for ln in lines[data_start_idx:]:
        if not ln:
            continue
        parts = _split_csv_like(ln)
        if len(parts) != len(attribute_names):
            bad_lines += 1
            # Keep going: we want audit to report issues rather than crash.
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=attribute_names)
    df.attrs["bad_lines_column_mismatch"] = int(bad_lines)
    return df


def _is_missing_token(token: str) -> bool:
    """
    @brief Missing values are expressed either with ? or <null> tokens.
    """
    if token is None:
        return True
    t = str(token).strip()
    return t in _MISSING_TOKENS


def _is_suspicious_null_like(token: str) -> bool:
    """
    @brief Detect null-like tokens that are NOT defined as missing by spec here.
    We report them as unknown_tokens so the user can decide.
    """
    if token is None:
        return False
    t = str(token).strip().lower()
    return t in _SUSPICIOUS_NULL_LIKE


def _schema_signature(relation: str, attributes: Dict[str, Any], inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
    """
    @brief Create a comparable schema signature for drift detection across files/folds.

    @param relation @relation value
    @param attributes Parsed attributes dict
    @param inputs Header inputs list
    @param outputs Header outputs list
    @return dict signature
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


def _row_hash(values: List[str]) -> str:
    """
    @brief Hash a row by joining tokens with a delimiter unlikely to occur in KEEL.
    """
    return "\x1f".join([str(v).strip() for v in values])


def _estimate_one_hot_dim(attributes: Dict[str, Any], nominal_inputs: List[str], observed: Dict[str, List[str]]) -> int:
    """
    @brief Estimate number of columns after one-hot encoding of nominal input features.

    Prefers header domain size if available, otherwise observed unique size.
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
# Robust stats helpers
# -----------------------------
def _safe_float_array(series: pd.Series) -> Tuple[np.ndarray, List[str], int]:
    """
    @brief Convert raw string series to float array, excluding missing tokens.
    Returns (values, unknown_tokens_list, n_nonmissing).

    @param series Raw string series
    @return (float array, unknown tokens, number of non-missing tokens)
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
    @brief Compute skewness using population moments (simple and dependency-free).
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
    @brief Compute kurtosis using population moments (simple and dependency-free).
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
    @brief Compute numeric stats including MAD and sigma≈1.4826*MAD and tail/skew metrics.
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
    @brief Compute robust outlier rate using robust z-score.

    robust_z = |x - median| / (1.4826 * MAD)

    @param x numeric array
    @param z_threshold outlier threshold (default 3.5)
    @return dict with outlier count + rate
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
        # fallback: detect train/test at end
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
    @brief Compute per-feature missingness fraction for selected columns.
    """
    out = {}
    n = len(df)
    if n == 0:
        return {c: float("nan") for c in cols}
    for c in cols:
        out[c] = float(sum(_is_missing_token(x) for x in df[c].astype(str).tolist()) / n)
    return out


def _constant_and_near_constant(
    df: pd.DataFrame,
    cols: List[str],
    *,
    near_constant_threshold: float = 0.999,
) -> Dict[str, Any]:
    """
    @brief Detect constant and near-constant features.

    Constant: only one non-missing unique value
    Near-constant: most frequent non-missing value ratio >= threshold

    @param df raw data frame
    @param cols columns to evaluate
    @param near_constant_threshold threshold for near-constant
    @return dict {constant: [...], near_constant: [{feature, top_value, top_ratio, n_unique}]}
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
            near_constant.append(
                {"feature": c, "top_value": top_value, "top_ratio": top_ratio, "n_unique_nonmissing": n_unique}
            )
    return {"constant_features": sorted(list(set(constant))), "near_constant_features": near_constant}


def _label_conflicts(
    df: pd.DataFrame,
    inputs: List[str],
    outputs: List[str],
) -> Dict[str, Any]:
    """
    @brief Detect label conflicts: identical inputs but multiple labels.

    Only computed for single-output datasets.

    @param df raw data frame
    @param inputs input feature names
    @param outputs output feature names
    @return dict with counts + example groups
    """
    if len(outputs) != 1:
        return {"enabled": False, "reason": "multioutput_or_unknown", "n_conflicting_input_groups": 0}

    outc = outputs[0]
    if outc not in df.columns:
        return {"enabled": False, "reason": "output_not_in_df", "n_conflicting_input_groups": 0}

    # Consider only rows where outputs are not missing
    sub = df[inputs + [outc]].astype(str).applymap(lambda x: str(x).strip())
    sub = sub[~sub[outc].map(_is_missing_token)]
    if sub.empty:
        return {"enabled": True, "n_conflicting_input_groups": 0, "n_rows_in_conflicts": 0, "examples": []}

    grp = sub.groupby(inputs, dropna=False)[outc].nunique()
    conflict_keys = grp[grp > 1]
    n_groups = int(conflict_keys.shape[0])
    if n_groups == 0:
        return {"enabled": True, "n_conflicting_input_groups": 0, "n_rows_in_conflicts": 0, "examples": []}

    # Count rows involved
    conflict_index = conflict_keys.index
    conflict_rows = sub.set_index(inputs).loc[conflict_index]
    n_rows = int(conflict_rows.shape[0])

    # Examples (up to 5 groups)
    examples = []
    # conflict_index can be Index or MultiIndex
    for k in list(conflict_index[:5]):
        if not isinstance(k, tuple):
            k = (k,)
        d = dict(zip(inputs, k))
        ys = sorted(list(set(conflict_rows.loc[k][outc].tolist())))
        examples.append({"inputs": d, "labels": ys})

    return {
        "enabled": True,
        "n_conflicting_input_groups": n_groups,
        "n_rows_in_conflicts": n_rows,
        "examples": examples,
    }


def _duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    @brief Compute duplicate rows inside one file.

    @param df raw data frame
    @return dict with duplicates count and ratio
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
    @brief Detect high-cardinality nominal input features (one-hot explosion risk).

    @param attributes header attributes map
    @param nominal_inputs nominal input feature names
    @param observed observed unique values map
    @param high_cardinality_threshold threshold for warning
    @return dict with flagged features
    """
    flagged = []
    for c in nominal_inputs:
        allowed = attributes.get(c, {}).get("nominal_allowed_header")
        header_card = len(allowed) if isinstance(allowed, set) else None
        obs_card = len(observed.get(c, []))
        card = header_card if (header_card is not None and header_card > 0) else obs_card
        if int(card) >= int(high_cardinality_threshold):
            flagged.append(
                {
                    "feature": c,
                    "cardinality_effective": int(card),
                    "header_cardinality": header_card,
                    "observed_cardinality": int(obs_card),
                }
            )
    return {"high_cardinality_threshold": int(high_cardinality_threshold), "high_cardinality_nominal": flagged}


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

    Extracts:
    - relation, filename, folder path
    - train/test status from basename (tra/tst)
    - inputs/outputs from header
    - attribute types from header (real/integer/nominal)
    - nominal allowed values (header) and observed values (data)
    - missing tokens in inputs vs outputs
    - unknown tokens (e.g., '??', 'nil', or invalid numeric strings)
    - numeric min/max from data and compare to header range + out-of-range detection
    - class distribution, IR, minority class
    - preprocess recommendations: normalization to [0,1], one-hot encoding (+ estimated OHE dimension)
    - robust numeric stats: mean/std/median/MAD/sigma_approx/quantiles/skewness/kurtosis/heavy-tail
    - quality checks: duplicates, label conflicts, constant/near-constant, missingness, high-cardinality nominal

    @param file_path Path to .dat file
    @param out_dir Output directory for JSON (if save=True). If None and save=True, uses "<folder>/_audit"
    @param save Whether to save JSON result
    @param tolerance Numeric comparison tolerance for range mismatch/out-of-range checks
    @param near_constant_threshold Threshold for near-constant features
    @param high_missingness_threshold Threshold for missingness warnings per feature
    @param high_cardinality_threshold Threshold for high-cardinality nominal warnings
    @param robust_z_threshold Robust z-score threshold for outlier rate
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

    lines = [ln for ln in text.splitlines()]
    # Remove blank and comments
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("%")]

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

    # feature counts
    n_total = len(attr_names)
    n_out = len(outputs)
    n_in = len(inputs)

    nominal_features = [k for k, v in attributes.items() if v["type"] == "nominal"]
    numeric_features = [k for k, v in attributes.items() if v["type"] in {"real", "integer"}]
    integer_features = [k for k, v in attributes.items() if v["type"] == "integer"]

    # missing + unknown tracking
    missing_per_feature: Dict[str, int] = {}
    unknown_per_feature: Dict[str, List[str]] = {}
    suspicious_null_like_per_feature: Dict[str, List[str]] = {}

    for col in attr_names:
        ser = df[col].astype(str)
        missing_per_feature[col] = int(sum(_is_missing_token(x.strip()) for x in ser.tolist()))
        # unknown null-like tokens
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

        # unknown tokens in nominal = observed values not in header OR suspicious null-like OR contains '?' but not missing
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
            # include tokens like "??" for numeric columns
            unknown_per_feature[col] = sorted(list(set(unknown_per_feature.get(col, [])) | set(unknown_tokens)))

        parse_rate = float(x.size / n_nonmissing) if n_nonmissing > 0 else float("nan")
        numeric_parse[col] = {
            "n_nonmissing": int(n_nonmissing),
            "n_parsed_numeric": int(x.size),
            "parse_rate": parse_rate,
        }

        if x.size > 0:
            dmin = float(np.min(x))
            dmax = float(np.max(x))
        else:
            dmin = float("nan")
            dmax = float("nan")

        hmin = attributes[col].get("min_range_header")
        hmax = attributes[col].get("max_range_header")

        numeric_minmax[col] = {
            "data_min": dmin,
            "data_max": dmax,
            "header_min": hmin,
            "header_max": hmax,
        }

        # mismatch vs header
        mismatch = {}
        if (hmin is not None) and np.isfinite(dmin):
            mismatch["min_mismatch"] = bool(not np.isclose(dmin, float(hmin), atol=tolerance, rtol=0.0))
        if (hmax is not None) and np.isfinite(dmax):
            mismatch["max_mismatch"] = bool(not np.isclose(dmax, float(hmax), atol=tolerance, rtol=0.0))
        if mismatch:
            numeric_minmax_mismatch[col] = mismatch

        # out-of-range vs header
        if (hmin is not None) and (hmax is not None) and x.size > 0:
            lo = float(hmin) - tolerance
            hi = float(hmax) + tolerance
            mask = (x < lo) | (x > hi)
            if np.any(mask):
                bad_vals = sorted(list(set([float(v) for v in x[mask]])))[:50]  # cap
                numeric_out_of_range[col] = {
                    "header_min": float(hmin),
                    "header_max": float(hmax),
                    "n_out_of_range": int(np.sum(mask)),
                    "example_values": bad_vals,
                }

        numeric_stats[col] = _robust_numeric_stats(x)
        numeric_outliers[col] = _robust_outlier_rate(x, z_threshold=robust_z_threshold)

    # integer integrity: how many non-missing values are integer-like
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
        integer_integrity[col] = {
            "n_nonmissing": n_nonmissing,
            "n_integer_like": int(n_int_like),
            "integer_like_rate": float(n_int_like / n_nonmissing),
        }

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
        min_count = None
        if len(counts) >= 1:
            min_count = min(counts.values())
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
        # multi-output (or weird header)
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
            min_count = None
            if len(counts) >= 1:
                min_count = min(counts.values())
            if len(counts) >= 2:
                mx = max(counts.values())
                mn = min(counts.values())
                ir = float(mx / mn) if mn > 0 else float("inf")
                minority = min(counts, key=counts.get)
            per_out[outc] = {
                "class_counts": counts,
                "imbalance_rate_ir": ir,
                "minority_class": minority,
                "min_class_count": min_count,
                "has_tiny_class_le_2": bool(min_count is not None and min_count <= 2),
            }
        class_info["per_output"] = per_out

    # preprocess recommendations
    needs_one_hot = [c for c in inputs if c in nominal_features]

    needs_unit_interval = []
    for c in inputs:
        if c not in numeric_features:
            continue
        hmin = attributes[c].get("min_range_header")
        hmax = attributes[c].get("max_range_header")
        # if header range is not exactly [0,1] => recommend scaling to [0,1]
        if (hmin is None) or (hmax is None):
            # fall back to data min/max
            dmin = numeric_minmax.get(c, {}).get("data_min")
            dmax = numeric_minmax.get(c, {}).get("data_max")
            if np.isfinite(dmin) and np.isfinite(dmax):
                if (dmin < -tolerance) or (dmax > 1.0 + tolerance) or (not np.isclose(dmin, 0.0, atol=tolerance)) or (
                    not np.isclose(dmax, 1.0, atol=tolerance)
                ):
                    needs_unit_interval.append(c)
        else:
            if (not np.isclose(float(hmin), 0.0, atol=tolerance)) or (not np.isclose(float(hmax), 1.0, atol=tolerance)):
                needs_unit_interval.append(c)

    # unknown tokens summary
    unknown_any = bool(unknown_per_feature)
    unknown_tokens_global = sorted({t for lst in unknown_per_feature.values() for t in lst})

    # Quality checks (top ROI)
    duplicates_info = _duplicate_rows(df)
    conflicts_info = _label_conflicts(df, inputs=inputs, outputs=outputs)

    missingness_all = _feature_missingness(df, attr_names)
    high_missing = {k: v for k, v in missingness_all.items() if (np.isfinite(v) and v >= float(high_missingness_threshold))}

    const_info = _constant_and_near_constant(df, cols=inputs, near_constant_threshold=near_constant_threshold)

    nominal_quality = _high_cardinality_nominal(
        attributes,
        nominal_inputs=[c for c in inputs if c in nominal_features],
        observed=nominal_observed,
        high_cardinality_threshold=high_cardinality_threshold,
    )

    one_hot_dim_est = _estimate_one_hot_dim(
        attributes,
        nominal_inputs=[c for c in inputs if c in nominal_features],
        observed=nominal_observed,
    )

    # optional reuse: parse_keel_file sanity-check (single-output only)
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
        "audit_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 1,
        "relation": relation,
        "file": {
            "basename": basename,
            "folder": folder,
            "abs_path": abs_path,
            "split": fold_info.split,
            "is_train": bool(is_train),
            "is_test": bool(is_test),
        },
        "schema": {
            "n_features_total_including_outputs": int(n_total),
            "n_input_features": int(n_in),
            "n_output_features": int(n_out),
            "input_features": inputs,
            "output_features": outputs,
            "attributes": attributes,
            "schema_signature": _schema_signature(relation, attributes, inputs, outputs),
            "feature_types": {
                "numeric": numeric_features,
                "integer": integer_features,
                "nominal": nominal_features,
            },
        },
        "dataset_shape": {
            "n_instances": int(len(df)),
            "n_bad_lines_column_mismatch": int(df.attrs.get("bad_lines_column_mismatch", 0)),
        },
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
        "feature_quality": {
            "near_constant_threshold": float(near_constant_threshold),
            **const_info,
            **nominal_quality,
        },
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
            ds_folder_name = os.path.basename(folder)
            out_path = os.path.join(out_dir, ds_folder_name, "files", f"{basename}.audit.json")

        result["saved_to"] = _save_json(out_path, result)

    return result


def _safe_error_audit(file_path: str, error: Exception, *, out_dir: Optional[str], save: bool) -> Dict[str, Any]:
    """
    @brief Create an error audit payload for a file that cannot be audited.

    @param file_path file path
    @param error caught exception
    @param out_dir output folder
    @param save save JSON
    @return dict payload
    """
    abs_path = os.path.abspath(file_path)
    folder = os.path.dirname(abs_path)
    basename = os.path.basename(abs_path)
    fold_info = _parse_fold_filename(basename)

    payload = {
        "audit_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 1,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "file": {
            "basename": basename,
            "folder": folder,
            "abs_path": abs_path,
            "split": fold_info.split,
            "is_train": bool(fold_info.split == "tra"),
            "is_test": bool(fold_info.split == "tst"),
        },
    }

    if save:
        if out_dir is None:
            out_path = os.path.join(folder, "_audit", "files", f"{basename}.audit.json")
        else:
            ds_folder_name = os.path.basename(folder)
            out_path = os.path.join(out_dir, ds_folder_name, "files", f"{basename}.audit.json")

        payload["saved_to"] = _save_json(out_path, payload)

    return payload


def _compute_leakage_metrics(
    train_file: str,
    test_file: str,
    *,
    inputs: List[str],
    outputs: List[str],
) -> Dict[str, Any]:
    """
    @brief Compute leakage between a train/test pair by intersection of hashes.

    Leakage modes:
    - full_row: identical complete rows exist in both
    - inputs_only: identical input vectors exist in both (even if label differs)

    @param train_file train path
    @param test_file test path
    @param inputs input feature names (from header)
    @param outputs output feature names (from header)
    @return dict with counts
    """
    # Re-read quickly and hash data rows.
    def load_hashes(fp: str, cols: List[str]) -> set:
        text = _read_text(fp)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("%")]
        relation, attributes, inps, outs, data_start = _parse_keel_header(lines)
        attr_names = list(attributes.keys())
        # Map to indices
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

    # full row = inputs + outputs, in file order provided
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


# -----------------------------
# Level 2: audit folder
# -----------------------------
def audit_keel_folder(
    folder_path: str,
    *,
    out_dir: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    @brief Audit a folder containing KEEL .dat files (Level 2).

    Reports:
    - how many datasets detected
    - whether predefined folds exist
    - if yes, which CV kind (k inferred from name pattern) and whether filenames are consistent
    - schema drift across folds/files (header mismatch)
    - leakage/overlap between train/test within same fold
    - runs Level-1 audit for each .dat file and stores per-file JSON

    @param folder_path Folder path containing .dat files
    @param out_dir Output directory for JSON. If None and save=True, uses "<folder>/_audit"
    @param save Whether to save JSON outputs
    @return folder audit dict
    """
    abs_folder = os.path.abspath(folder_path)
    if not os.path.isdir(abs_folder):
        raise FileNotFoundError(f"Folder not found: {abs_folder}")

    files = sorted([f for f in os.listdir(abs_folder) if f.lower().endswith(".dat")])
    dat_abs = [os.path.join(abs_folder, f) for f in files]

    # group by dataset_key
    groups: Dict[str, List[_FoldNameInfo]] = {}
    for f in files:
        info = _parse_fold_filename(f)
        groups.setdefault(info.dataset_key, []).append(info)

    dataset_keys = sorted(groups.keys())

    dataset_summaries = []
    folder_has_cv = False
    cv_k_values = set()

    # audit every file (level 1) first (so we can aggregate later)
    file_audits = []
    per_file_by_name = {}
    for fp in dat_abs:
        try:
            a = audit_keel_file(fp, out_dir=out_dir or os.path.join(abs_folder, "_audit"), save=save)
        except Exception as e:
            a = _safe_error_audit(fp, e, out_dir=out_dir or os.path.join(abs_folder, "_audit"), save=save)
        file_audits.append(a)
        per_file_by_name[os.path.basename(fp)] = a

    # schema drift + leakage checks per dataset key
    schema_drift = []
    leakage_reports = []
    split_class_presence_issues = []

    for ds_key in dataset_keys:
        infos = groups[ds_key]
        splits = sorted({i.split for i in infos if i.split is not None})
        looks_like_folds = ("tra" in splits) or ("tst" in splits)

        ds_sum: Dict[str, Any] = {"dataset_key": ds_key, "n_files": int(len(infos)), "has_train_test_splits": looks_like_folds}

        # schema signature comparison
        sigs = {}
        for i in infos:
            a = per_file_by_name.get(
                f"{ds_key}-{i.k_folds}-{i.fold_index}{i.split}.dat" if (i.k_folds and i.fold_index and i.split) else None,
                None
            )
            # fallback by basename search
            if a is None:
                # find by actual filename
                # i.dataset_key is base; we need actual file name
                pass

        # Build map by actual file basenames in this group
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
            sigs[bn] = sig
            if base_sig is None:
                base_sig = sig
                base_file = bn
        drift_files = []
        for bn, sig in sigs.items():
            if sig != base_sig:
                drift_files.append({"file": bn, "differs_from": base_file})
        if drift_files:
            schema_drift.append({"dataset_key": ds_key, "base_file": base_file, "drift_files": drift_files})
        ds_sum["schema_drift"] = {"has_drift": bool(len(drift_files) > 0), "drift_files": drift_files}

        if looks_like_folds:
            folder_has_cv = True
            # infer k if present
            ks = sorted({i.k_folds for i in infos if i.k_folds is not None})
            k = ks[0] if ks else None
            if k is not None:
                cv_k_values.add(int(k))

            # validate fold completeness if k and fold_index exist
            fold_indices = sorted({i.fold_index for i in infos if i.fold_index is not None})
            ds_sum["cv"] = {
                "kind": "predefined_folds",
                "k_folds_inferred": k,
                "fold_indices_found": fold_indices,
            }

            # require tra+tst per fold index
            problems = []
            if k is not None:
                expected = list(range(1, int(k) + 1))
                missing_folds = sorted([x for x in expected if x not in set(fold_indices)])
                if missing_folds:
                    problems.append({"missing_fold_indices": missing_folds})

            # check each fold has tra+tst
            by_fold: Dict[int, set] = {}
            for i in infos:
                if i.fold_index is None or i.split is None:
                    continue
                by_fold.setdefault(int(i.fold_index), set()).add(i.split)
            missing_pairs = sorted([fi for fi, ss in by_fold.items() if not (("tra" in ss) and ("tst" in ss))])
            if missing_pairs:
                problems.append({"folds_missing_tra_or_tst": missing_pairs})

            # optional reuse: discover_keel_cv_folds validation (expects folder is one dataset)
            if _DISCOVER_KEEL_CV_FOLDS is not None:
                try:
                    pairs = _DISCOVER_KEEL_CV_FOLDS(abs_folder)
                    ds_sum["cv"]["discover_keel_cv_folds_pairs"] = int(len(pairs))
                except Exception as e:
                    ds_sum["cv"]["discover_keel_cv_folds_error"] = f"{type(e).__name__}: {e}"

            ds_sum["cv"]["naming_validation_problems"] = problems

            # Leakage and class-presence per fold
            # pick a representative file to get inputs/outputs (base schema)
            rep_bn = base_file if base_file is not None else (group_basenames[0] if group_basenames else None)
            rep_a = per_file_by_name.get(rep_bn) if rep_bn else None
            rep_inputs = rep_a.get("schema", {}).get("input_features") if rep_a else []
            rep_outputs = rep_a.get("schema", {}).get("output_features") if rep_a else []

            for fi in sorted([x for x in fold_indices if x is not None]):
                tra_bn = f"{ds_key}-{k}-{fi}tra.dat" if k is not None else None
                tst_bn = f"{ds_key}-{k}-{fi}tst.dat" if k is not None else None

                # fallback search if naming is irregular
                if tra_bn not in per_file_by_name:
                    cand = [bn for bn in group_basenames if bn.lower().endswith(f"-{fi}tra.dat".lower())]
                    tra_bn = cand[0] if cand else tra_bn
                if tst_bn not in per_file_by_name:
                    cand = [bn for bn in group_basenames if bn.lower().endswith(f"-{fi}tst.dat".lower())]
                    tst_bn = cand[0] if cand else tst_bn

                if tra_bn in per_file_by_name and tst_bn in per_file_by_name:
                    tra_fp = os.path.join(abs_folder, tra_bn)
                    tst_fp = os.path.join(abs_folder, tst_bn)

                    try:
                        leak = _compute_leakage_metrics(tra_fp, tst_fp, inputs=rep_inputs, outputs=rep_outputs)
                        leak["dataset_key"] = ds_key
                        leak["fold_index"] = int(fi)
                        leakage_reports.append(leak)
                    except Exception as e:
                        leakage_reports.append(
                            {
                                "dataset_key": ds_key,
                                "fold_index": int(fi),
                                "train_basename": tra_bn,
                                "test_basename": tst_bn,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )

                    # class presence issues (single output only)
                    tra_set = _class_presence_in_split(per_file_by_name.get(tra_bn, {}))
                    tst_set = _class_presence_in_split(per_file_by_name.get(tst_bn, {}))
                    if (tra_set is not None) and (tst_set is not None):
                        missing_in_train = sorted(list(tst_set - tra_set))
                        missing_in_test = sorted(list(tra_set - tst_set))
                        if missing_in_train or missing_in_test:
                            split_class_presence_issues.append(
                                {
                                    "dataset_key": ds_key,
                                    "fold_index": int(fi),
                                    "train_basename": tra_bn,
                                    "test_basename": tst_bn,
                                    "classes_missing_in_train": missing_in_train,
                                    "classes_missing_in_test": missing_in_test,
                                }
                            )
        else:
            ds_sum["cv"] = {"kind": "none"}

        dataset_summaries.append(ds_sum)

    # folder-level quick aggregation (useful for level 3)
    datasets_with_missing = []
    datasets_with_nonbinary = []
    datasets_with_numeric_out_of_range = []
    datasets_with_nominal_unknown = []
    datasets_with_schema_drift = []
    datasets_with_leakage = []
    datasets_with_label_conflicts = []
    datasets_with_constant_features = []
    datasets_with_high_missingness = []
    datasets_with_high_cardinality_nominal = []
    datasets_with_tiny_class = []
    files_with_errors = []

    for a in file_audits:
        if a.get("status") == "error":
            files_with_errors.append({"file": a["file"]["basename"], "error": a["error"]})
            continue

        ds_name = a.get("relation") or a["file"]["basename"]

        miss_in = a["data_quality"]["missing"]["missing_in_inputs"]
        miss_out = a["data_quality"]["missing"]["missing_in_outputs"]
        if miss_in or miss_out:
            datasets_with_missing.append(ds_name)

        ci = a.get("class_info", {})
        if ci.get("class_type") not in (None, "binary"):
            # multioutput/multiclass/unknown
            datasets_with_nonbinary.append(ds_name)

        if ci.get("has_tiny_class_le_2") is True:
            datasets_with_tiny_class.append(ds_name)

        num_oor = a["data_quality"]["numeric"]["out_of_range_vs_header"]
        if num_oor:
            datasets_with_numeric_out_of_range.append(
                {"dataset": ds_name, "file": a["file"]["basename"], "features": sorted(list(num_oor.keys()))}
            )

        nom_bad = a["data_quality"]["nominal"]["values_in_data_not_in_header"]
        if nom_bad:
            datasets_with_nominal_unknown.append(
                {"dataset": ds_name, "file": a["file"]["basename"], "features": sorted(list(nom_bad.keys()))}
            )

        lc = a["data_quality"]["label_conflicts"]
        if lc.get("enabled") and lc.get("n_conflicting_input_groups", 0) > 0:
            datasets_with_label_conflicts.append(
                {"dataset": ds_name, "file": a["file"]["basename"], "n_conflicting_groups": lc["n_conflicting_input_groups"]}
            )

        cf = a.get("feature_quality", {}).get("constant_features", [])
        if cf:
            datasets_with_constant_features.append({"dataset": ds_name, "file": a["file"]["basename"], "features": cf})

        hm = a["data_quality"]["missing"].get("high_missingness_features", {})
        if hm:
            datasets_with_high_missingness.append({"dataset": ds_name, "file": a["file"]["basename"], "features": list(hm.keys())})

        hcn = a.get("feature_quality", {}).get("high_cardinality_nominal", [])
        if hcn:
            datasets_with_high_cardinality_nominal.append({"dataset": ds_name, "file": a["file"]["basename"], "features": [x["feature"] for x in hcn]})

        # schema drift flag already per dataset key; we will aggregate later
    if schema_drift:
        datasets_with_schema_drift = schema_drift

    # leakage summary
    for leak in leakage_reports:
        if leak.get("has_leakage_full_row") or leak.get("has_leakage_inputs_only"):
            datasets_with_leakage.append(leak)

    result: Dict[str, Any] = {
        "audit_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 2,
        "folder": {"abs_path": abs_folder, "n_dat_files": int(len(files)), "dat_files": files},
        "datasets_detected": {
            "count": int(len(dataset_keys)),
            "keys": dataset_keys,
            "has_cv_any": bool(folder_has_cv),
            "cv_k_values_inferred": sorted(list(cv_k_values)),
            "per_dataset": dataset_summaries,
        },
        "schema_consistency": {
            "has_schema_drift_any": bool(len(schema_drift) > 0),
            "schema_drift_datasets": schema_drift,
        },
        "cv_leakage": {
            "has_leakage_any": bool(any((x.get("has_leakage_full_row") or x.get("has_leakage_inputs_only")) for x in leakage_reports)),
            "leakage_reports": leakage_reports,
        },
        "cv_class_presence": {
            "has_class_presence_issues_any": bool(len(split_class_presence_issues) > 0),
            "issues": split_class_presence_issues,
        },
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
            "datasets_with_schema_drift": datasets_with_schema_drift,
            "datasets_with_leakage": datasets_with_leakage,
            "datasets_with_label_conflicts": datasets_with_label_conflicts,
            "datasets_with_constant_features": datasets_with_constant_features,
            "datasets_with_high_missingness": datasets_with_high_missingness,
            "datasets_with_high_cardinality_nominal": datasets_with_high_cardinality_nominal,
        },
    }

    if save:
        folder_name = os.path.basename(abs_folder)
        if out_dir is None:
            out_path = os.path.join(abs_folder, "_audit", "folder.audit.json")
        else:
            out_path = os.path.join(out_dir, folder_name, "folder.audit.json")

        result["saved_to"] = _save_json(out_path, result)

    return result


# -----------------------------
# Level 3: audit root
# -----------------------------
def audit_keel_root(
    root_folder: str,
    *,
    out_dir: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    @brief Audit a root folder containing many dataset folders (Level 3).

    Produces:
    - list of dataset folders
    - whether all are same CV type
    - global summary (important flags + lists):
      - datasets with missing values
      - datasets with numeric out-of-range (dataset + feature names)
      - datasets with nominal values not in header (dataset + feature names)
      - datasets that are not binary class (multiclass or multioutput)
      - schema drift datasets
      - leakage datasets
      - label-conflict datasets
      - constant/high-missing/high-cardinality features

    @param root_folder Root folder path
    @param out_dir Output directory for JSON. If None and save=True, uses "<root>/_audit"
    @param save Whether to save JSON outputs
    @return root audit dict
    """
    abs_root = os.path.abspath(root_folder)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Root folder not found: {abs_root}")

    subdirs = sorted(
        [
            os.path.join(abs_root, d)
            for d in os.listdir(abs_root)
            if os.path.isdir(os.path.join(abs_root, d)) and not d.startswith(".")
        ]
    )

    if out_dir is None:
        out_dir = os.path.join(abs_root, "_audit")

    per_folder = []
    cv_k_union = set()
    has_cv_flags = []

    # Global aggregations
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

    for d in subdirs:
        fr = audit_keel_folder(d, out_dir=out_dir, save=save)
        per_folder.append(fr)

        has_cv_flags.append(bool(fr["datasets_detected"]["has_cv_any"]))
        for k in fr["datasets_detected"].get("cv_k_values_inferred", []):
            cv_k_union.add(int(k))

        folder_name = os.path.basename(d)

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

    all_same_cv = (len(set(has_cv_flags)) <= 1)

    result: Dict[str, Any] = {
        "audit_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": 3,
        "root": {"abs_path": abs_root, "n_dataset_folders": int(len(subdirs)), "dataset_folders": subdirs},
        "cv_consistency": {
            "all_same_cv_type": bool(all_same_cv),
            "has_cv_flags_per_folder": has_cv_flags,
            "cv_k_values_union": sorted(list(cv_k_union)),
        },
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
        out_base = out_dir or os.path.join(abs_root, "_audit")
        out_path = os.path.join(out_base, "root.audit.json")
        result["saved_to"] = _save_json(out_path, result)

    return result


__all__ = [
    "audit_keel_file",
    "audit_keel_folder",
    "audit_keel_root",
]
