# SPDX-License-Identifier: BSD-3-Clause
"""KEEL Dataset Auditor Utility with logically grouped class-based helpers.

This archived module is retained for historical reference and is not part of the stable public API.
"""

from __future__ import annotations

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_MISSING_TOKENS = {"?", "<null>"}
_SUSPICIOUS_NULL_LIKE = {"null", "nil", "none", "nan", "na"}

class _FoldNameInfo:
    dataset_key: str
    k_folds: Optional[int]
    fold_index: Optional[int]
    split: Optional[str]

class ProjectHelperImportBridge:
    """
    @brief Best-effort bridge to optional existing project helpers.
    """

    @staticmethod
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


class JsonArtifactIO:
    """
    @brief JSON serialization and persistence helpers.
    """

    @staticmethod
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

    @staticmethod
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


class PathLayoutHelper:
    """
    @brief Path-layout helpers for naming stable output locations.
    """

    @staticmethod
    def _dataset_folder_name_from_path(folder_path: str) -> str:
        """
        @brief Return dataset folder name for stable output layout.
        """
        return os.path.basename(os.path.abspath(folder_path))


class KEELParsingHelper:
    """
    @brief Parsing and low-level KEEL file helpers.
    """

    @staticmethod
    def _read_text(file_path: str) -> str:
        """
        @brief Read a file as UTF-8 best-effort (replacement for bad bytes).
        """
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _split_csv_like(line: str) -> List[str]:
        """
        @brief Split KEEL data line by comma, trimming spaces.
        """
        return re.split(r"\s*,\s*", line.strip())

    @staticmethod
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

    @staticmethod
    def _is_missing_token(token: str) -> bool:
        """
        @brief Missing values are expressed either with ? or <null>.
        """
        if token is None:
            return True
        return str(token).strip() in _MISSING_TOKENS

    @staticmethod
    def _is_suspicious_null_like(token: str) -> bool:
        """
        @brief Detect null-like tokens that are NOT defined as missing by spec.
        """
        if token is None:
            return False
        return str(token).strip().lower() in _SUSPICIOUS_NULL_LIKE

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _row_hash(values: List[str]) -> str:
        """
        @brief Hash a row by joining tokens with a delimiter unlikely to occur in KEEL.
        """
        return "\x1f".join([str(v).strip() for v in values])

    @staticmethod
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

    @staticmethod
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


class NumericStatisticsHelper:
    """
    @brief Reusable numeric/statistical helper computations.
    """

    @staticmethod
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

    @staticmethod
    def _normalize_numeric_frame(df_num: pd.DataFrame) -> pd.DataFrame:
        """
        @brief Min-max normalize numeric columns to [0, 1] with safe constant handling.

        @param df_num Numeric dataframe.
        @return Normalized numeric dataframe.
        """
        if df_num.empty:
            return df_num.copy()

        res = df_num.copy()
        for col in res.columns:
            ser = pd.to_numeric(res[col], errors="coerce")
            if ser.dropna().empty:
                res[col] = 0.0
                continue
            mn = float(ser.min())
            mx = float(ser.max())
            if np.isclose(mx, mn):
                res[col] = 0.0
            else:
                res[col] = (ser - mn) / (mx - mn)
        return res.fillna(0.0)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _safe_mean(nums: List[float]) -> float:
        vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    @staticmethod
    def _safe_min(nums: List[float]) -> float:
        vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
        return float(min(vals)) if vals else float("nan")

    @staticmethod
    def _safe_max(nums: List[float]) -> float:
        vals = [x for x in nums if isinstance(x, (int, float)) and np.isfinite(x)]
        return float(max(vals)) if vals else float("nan")

    @staticmethod
    def _summarize_numeric_series(values: List[float]) -> Dict[str, Any]:
        """
        @brief Summarize a numeric series with min/mean/max/std.
        """
        vals = [float(x) for x in values if isinstance(x, (int, float)) and np.isfinite(x)]
        if not vals:
            return {"n": 0, "min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }


class OversamplingAnalysisHelper:
    """
    @brief Oversampling-oriented neighborhood and topology diagnostics.
    """

    @staticmethod
    def _build_knn_ready_matrix(
        df: pd.DataFrame,
        *,
        attributes: Dict[str, Any],
        inputs: List[str],
        outputs: List[str],
        max_rows: int = 10000,
    ) -> Dict[str, Any]:
        """
        @brief Build a kNN-ready design matrix for oversampling-oriented diagnostics.

        @param df Input dataframe parsed from KEEL.
        @param attributes Parsed KEEL attributes.
        @param inputs Input feature names.
        @param outputs Output feature names.
        @param max_rows Maximum rows for expensive neighborhood diagnostics.
        @return Dict with enabled flag, reason, matrix, labels, and helper metadata.
        """
        if len(outputs) != 1:
            return {"enabled": False, "reason": "multioutput_or_unknown"}

        outc = outputs[0]
        if outc not in df.columns:
            return {"enabled": False, "reason": "output_not_in_df"}

        y_raw = df[outc].astype(str).map(lambda x: x.strip())
        valid_mask = ~y_raw.map(_is_missing_token)
        work_df = df.loc[valid_mask, inputs].copy()
        y = y_raw.loc[valid_mask].reset_index(drop=True)
        work_df = work_df.reset_index(drop=True)

        if work_df.empty:
            return {"enabled": False, "reason": "no_valid_rows_after_target_filter"}
        if len(work_df) > int(max_rows):
            return {"enabled": False, "reason": f"too_large_gt_{max_rows}", "n_rows": int(len(work_df))}

        numeric_cols = [c for c in inputs if attributes.get(c, {}).get("type") in {"real", "integer"}]
        nominal_cols = [c for c in inputs if attributes.get(c, {}).get("type") == "nominal"]

        parts: List[pd.DataFrame] = []
        if numeric_cols:
            num_df = work_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            parts.append(_normalize_numeric_frame(num_df))

        if nominal_cols:
            nom_df = work_df[nominal_cols].astype(str).apply(lambda c: c.map(lambda x: str(x).strip()))
            nom_df = nom_df.replace({tok: "__MISSING__" for tok in _MISSING_TOKENS})
            parts.append(pd.get_dummies(nom_df, prefix=nominal_cols, dummy_na=False))

        if not parts:
            return {"enabled": False, "reason": "no_input_features_available"}

        x_df = pd.concat(parts, axis=1)
        if x_df.empty:
            return {"enabled": False, "reason": "empty_transformed_matrix"}

        x = x_df.to_numpy(dtype=float, copy=False)
        class_counts = {k: int(v) for k, v in y.value_counts().to_dict().items()}
        minority_label = min(class_counts, key=class_counts.get)
        majority_label = max(class_counts, key=class_counts.get)

        return {
            "enabled": True,
            "reason": None,
            "x": x,
            "y": y.to_numpy(dtype=object),
            "feature_names": list(x_df.columns),
            "n_rows": int(x.shape[0]),
            "n_features_transformed": int(x.shape[1]),
            "class_counts": class_counts,
            "minority_label": minority_label,
            "majority_label": majority_label,
            "numeric_cols": numeric_cols,
            "nominal_cols": nominal_cols,
        }

    @staticmethod
    def _topology_label_from_majority_count(majority_count: int, k_neighbors: int) -> str:
        """
        @brief Map neighborhood majority count to a minority-topology category.
        """
        if majority_count <= 1:
            return "safe"
        if majority_count < (k_neighbors / 2.0):
            return "borderline"
        if majority_count < k_neighbors:
            return "rare"
        return "outlier"

    @staticmethod
    def _oversampling_knn_diagnostics(
        df: pd.DataFrame,
        *,
        attributes: Dict[str, Any],
        inputs: List[str],
        outputs: List[str],
        k_neighbors: int = 5,
        feasibility_ks: Optional[List[int]] = None,
        max_rows: int = 10000,
    ) -> Dict[str, Any]:
        """
        @brief Compute oversampling-oriented neighborhood diagnostics.

        @param df Input dataframe.
        @param attributes Parsed KEEL attributes.
        @param inputs Input features.
        @param outputs Output features.
        @param k_neighbors Neighborhood size used for topology/hardness summaries.
        @param feasibility_ks Neighbor values to test for SMOTE-family feasibility.
        @param max_rows Maximum rows for neighborhood analysis.
        @return Oversampling diagnostics dict.
        """
        ks = sorted(set(feasibility_ks or [3, 5, 7]))
        prep = _build_knn_ready_matrix(df, attributes=attributes, inputs=inputs, outputs=outputs, max_rows=max_rows)
        if not prep.get("enabled", False):
            return {
                "enabled": False,
                "reason": prep.get("reason", "unknown"),
                "knn_feasibility": {str(k): {"is_feasible": False, "is_fragile": False, "required_minority_count": int(k + 1)} for k in ks},
            }

        x = prep["x"]
        y = prep["y"]
        minority_label = prep["minority_label"]
        class_counts = prep["class_counts"]
        minority_mask = (y == minority_label)
        minority_idx = np.where(minority_mask)[0]
        n_minority = int(minority_mask.sum())
        n_total = int(len(y))

        knn_feasibility = {}
        for k in ks:
            feasible = bool(n_minority > k)
            fragile = bool(feasible and n_minority <= (2 * k + 1))
            knn_feasibility[str(k)] = {
                "is_feasible": feasible,
                "is_fragile": fragile,
                "required_minority_count": int(k + 1),
                "observed_minority_count": int(n_minority),
            }

        max_neighbors = int(min(max(k_neighbors + 1, 2), max(1, n_total - 1)))
        if max_neighbors <= 1:
            return {
                "enabled": False,
                "reason": "not_enough_rows_for_neighbors",
                "knn_feasibility": knn_feasibility,
            }

        nn = NearestNeighbors(n_neighbors=max_neighbors, metric="euclidean")
        nn.fit(x)
        neigh_idx = nn.kneighbors(return_distance=False)
        neigh_idx = neigh_idx[:, 1:max_neighbors]
        effective_k = int(neigh_idx.shape[1])
        if effective_k <= 0:
            return {
                "enabled": False,
                "reason": "not_enough_neighbors_after_self_removal",
                "knn_feasibility": knn_feasibility,
            }

        disagreement_all = []
        minority_majority_ratios = []
        minority_labels = []
        minority_noise_flags = []

        for i in range(n_total):
            n_labels = y[neigh_idx[i]]
            disagree_ratio = float(np.mean(n_labels != y[i]))
            disagreement_all.append(disagree_ratio)

            if minority_mask[i]:
                maj_count = int(np.sum(n_labels != minority_label))
                maj_ratio = float(maj_count / effective_k)
                minority_majority_ratios.append(maj_ratio)
                minority_labels.append(_topology_label_from_majority_count(maj_count, effective_k))
                minority_noise_flags.append(bool(maj_ratio >= 0.80))

        minority_topology_counts = {name: 0 for name in ["safe", "borderline", "rare", "outlier"]}
        for lab in minority_labels:
            minority_topology_counts[lab] = int(minority_topology_counts.get(lab, 0) + 1)
        minority_topology_ratios = {
            k: float(v / n_minority) if n_minority > 0 else float("nan")
            for k, v in minority_topology_counts.items()
        }

        centroid_min = np.mean(x[minority_mask], axis=0) if n_minority > 0 else np.zeros(x.shape[1])
        centroid_maj = np.mean(x[~minority_mask], axis=0) if np.any(~minority_mask) else np.zeros(x.shape[1])
        centroid_distance = float(np.linalg.norm(centroid_min - centroid_maj))
        pooled_std = float(np.mean(np.std(x, axis=0))) if x.shape[1] > 0 else float("nan")
        separability = float(centroid_distance / pooled_std) if (isinstance(pooled_std, float) and np.isfinite(pooled_std) and pooled_std > 0.0) else float("nan")

        hardness_all_mean = float(np.mean(disagreement_all)) if disagreement_all else float("nan")
        hardness_minority_mean = float(np.mean([disagreement_all[i] for i in minority_idx])) if len(minority_idx) > 0 else float("nan")
        overlap_minority_mean = float(np.mean(minority_majority_ratios)) if minority_majority_ratios else float("nan")
        overlap_minority_max = float(np.max(minority_majority_ratios)) if minority_majority_ratios else float("nan")
        mixed_neighborhood_rate = float(np.mean([r > 0.0 for r in minority_majority_ratios])) if minority_majority_ratios else float("nan")
        noise_rate = float(np.mean(minority_noise_flags)) if minority_noise_flags else float("nan")

        interpolation_components = {
            "boundary_crossing_risk": float(overlap_minority_mean) if np.isfinite(overlap_minority_mean) else 0.0,
            "outlier_seed_risk": float(minority_topology_ratios.get("outlier", 0.0) + 0.5 * minority_topology_ratios.get("rare", 0.0)),
            "noise_seed_risk": float(noise_rate) if np.isfinite(noise_rate) else 0.0,
            "minority_hardness_risk": float(hardness_minority_mean) if np.isfinite(hardness_minority_mean) else 0.0,
        }
        interpolation_risk_score = int(round(min(100.0, 100.0 * (0.35 * interpolation_components["boundary_crossing_risk"] + 0.25 * interpolation_components["outlier_seed_risk"] + 0.20 * interpolation_components["noise_seed_risk"] + 0.20 * interpolation_components["minority_hardness_risk"]))))

        recommendation = {"status": "GOOD_FOR_STANDARD_OVERSAMPLING", "reasons": []}
        if not knn_feasibility.get("3", {}).get("is_feasible", False):
            recommendation["status"] = "NOT_SUITABLE_FOR_NEIGHBOR_OVERSAMPLING"
            recommendation["reasons"].append("minority_class_too_small_for_k3")
        else:
            if knn_feasibility.get("5", {}).get("is_fragile", False) or knn_feasibility.get("7", {}).get("is_fragile", False):
                recommendation["reasons"].append("neighbor_feasibility_fragile")
            if np.isfinite(overlap_minority_mean) and overlap_minority_mean >= 0.50:
                recommendation["reasons"].append("high_minority_overlap")
            if np.isfinite(noise_rate) and noise_rate >= 0.25:
                recommendation["reasons"].append("high_noise_suspicion")
            if interpolation_risk_score >= 70:
                recommendation["reasons"].append("high_interpolation_risk")
            if recommendation["reasons"]:
                recommendation["status"] = "USE_WITH_CAUTION"
            if recommendation["status"] == "GOOD_FOR_STANDARD_OVERSAMPLING" and knn_feasibility.get("5", {}).get("is_feasible", False):
                recommendation["recommended_neighbor_k"] = 5
            elif knn_feasibility.get("3", {}).get("is_feasible", False):
                recommendation["recommended_neighbor_k"] = 3

        return {
            "enabled": True,
            "reason": None,
            "n_rows_used": int(n_total),
            "n_features_transformed": int(prep["n_features_transformed"]),
            "class_counts": class_counts,
            "minority_label": minority_label,
            "effective_k_for_topology": int(effective_k),
            "knn_feasibility": knn_feasibility,
            "minority_topology": {"counts": minority_topology_counts, "ratios": minority_topology_ratios},
            "overlap_hardness": {
                "minority_majority_neighbor_ratio_mean": overlap_minority_mean,
                "minority_majority_neighbor_ratio_max": overlap_minority_max,
                "minority_mixed_neighborhood_rate": mixed_neighborhood_rate,
                "instance_hardness_mean": hardness_all_mean,
                "instance_hardness_minority_mean": hardness_minority_mean,
                "centroid_distance": centroid_distance,
                "separability_proxy": separability,
            },
            "noise_suspicion": {
                "minority_noise_suspicion_count": int(sum(minority_noise_flags)),
                "minority_noise_suspicion_rate": noise_rate,
                "rule": "minority_point_with_majority_ratio_ge_0_80_in_local_neighborhood",
            },
            "synthetic_interpolation_risk": {"score_0_100": int(interpolation_risk_score), **interpolation_components},
            "oversampling_recommendation": recommendation,
        }


class FeatureQualityHelper:
    """
    @brief Feature-quality and label-integrity helper computations.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
        """
        @brief Duplicate row count and ratio.
        """
        n = int(len(df))
        if n == 0:
            return {"n_rows": 0, "n_duplicate_rows": 0, "duplicate_ratio": float("nan")}
        dup = int(df.duplicated(keep="first").sum())
        return {"n_rows": n, "n_duplicate_rows": dup, "duplicate_ratio": float(dup / n)}

    @staticmethod
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

    @staticmethod
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


class SplitAuditHelper:
    """
    @brief Train/test leakage, class presence, and partition stability helpers.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _compute_partition_stability(ok_audits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        @brief Aggregate fold stability for imbalance and oversampling-oriented metrics.
        """
        trains = [a for a in ok_audits if a.get("file", {}).get("is_train")] or ok_audits
        irs, mins = [], []
        overlap_vals, hardness_vals, risk_vals = [], [], []
        safe_vals, borderline_vals, rare_vals, outlier_vals = [], [], [], []
        feasible_k3, feasible_k5 = [], []

        for a in trains:
            ci = a.get("class_info", {})
            ir = ci.get("imbalance_rate_ir")
            if isinstance(ir, (int, float)) and np.isfinite(ir):
                irs.append(float(ir))
            mc = ci.get("min_class_count")
            if isinstance(mc, (int, float)) and np.isfinite(mc):
                mins.append(float(mc))

            osa = a.get("oversampling_audit", {})
            if not osa.get("enabled", False):
                continue
            oh = osa.get("overlap_hardness", {})
            mt = osa.get("minority_topology", {}).get("ratios", {})
            risk = osa.get("synthetic_interpolation_risk", {}).get("score_0_100")

            for val, bucket in [
                (oh.get("minority_majority_neighbor_ratio_mean"), overlap_vals),
                (oh.get("instance_hardness_minority_mean"), hardness_vals),
            ]:
                if isinstance(val, (int, float)) and np.isfinite(val):
                    bucket.append(float(val))
            if isinstance(risk, (int, float)) and np.isfinite(risk):
                risk_vals.append(float(risk))

            for key, bucket in [("safe", safe_vals), ("borderline", borderline_vals), ("rare", rare_vals), ("outlier", outlier_vals)]:
                val = mt.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    bucket.append(float(val))

            k3 = osa.get("knn_feasibility", {}).get("3", {}).get("is_feasible")
            k5 = osa.get("knn_feasibility", {}).get("5", {}).get("is_feasible")
            if isinstance(k3, bool):
                feasible_k3.append(k3)
            if isinstance(k5, bool):
                feasible_k5.append(k5)

        unstable_reasons = []
        ir_summary = _summarize_numeric_series(irs)
        min_summary = _summarize_numeric_series(mins)
        overlap_summary = _summarize_numeric_series(overlap_vals)
        hardness_summary = _summarize_numeric_series(hardness_vals)
        risk_summary = _summarize_numeric_series(risk_vals)

        if ir_summary["n"] >= 2 and np.isfinite(ir_summary["std"]) and ir_summary["std"] >= 1.0:
            unstable_reasons.append("ir_variability_high")
        if min_summary["n"] >= 2 and np.isfinite(min_summary["min"]) and np.isfinite(min_summary["max"]) and (min_summary["max"] - min_summary["min"] >= 3.0):
            unstable_reasons.append("minority_count_variability_high")
        if overlap_summary["n"] >= 2 and np.isfinite(overlap_summary["std"]) and overlap_summary["std"] >= 0.15:
            unstable_reasons.append("overlap_variability_high")
        if hardness_summary["n"] >= 2 and np.isfinite(hardness_summary["std"]) and hardness_summary["std"] >= 0.10:
            unstable_reasons.append("hardness_variability_high")
        if risk_summary["n"] >= 2 and np.isfinite(risk_summary["std"]) and risk_summary["std"] >= 12.0:
            unstable_reasons.append("interpolation_risk_variability_high")
        if feasible_k3 and (not all(feasible_k3)):
            unstable_reasons.append("k3_feasibility_inconsistent_across_folds")
        if feasible_k5 and (not all(feasible_k5)):
            unstable_reasons.append("k5_feasibility_inconsistent_across_folds")

        return {
            "is_stable": bool(len(unstable_reasons) == 0),
            "unstable_reasons": unstable_reasons,
            "ir": ir_summary,
            "min_class_count": min_summary,
            "minority_overlap": overlap_summary,
            "minority_hardness": hardness_summary,
            "interpolation_risk": risk_summary,
            "minority_topology": {
                "safe": _summarize_numeric_series(safe_vals),
                "borderline": _summarize_numeric_series(borderline_vals),
                "rare": _summarize_numeric_series(rare_vals),
                "outlier": _summarize_numeric_series(outlier_vals),
            },
            "knn_feasibility_consistency": {
                "k3_all_folds_feasible": bool(feasible_k3 and all(feasible_k3)),
                "k5_all_folds_feasible": bool(feasible_k5 and all(feasible_k5)),
            },
        }


class DecisionSupportHelper:
    """
    @brief Severity, readiness, and recommendation logic.
    """

    @staticmethod
    def _dominant_severity_drivers(breakdown: Dict[str, int], *, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        @brief Return the largest contributors to severity.
        """
        items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
        return [{"driver": k, "weight": int(v)} for k, v in items[: int(top_n)]]

    @staticmethod
    def _oversampling_dataset_recommendation(flags: Dict[str, Any], *, partition_stability: Optional[Dict[str, Any]] = None, representative_audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        @brief Build an oversampling-specific dataset recommendation.
        """
        blocked_reasons = []
        cautious_reasons = []
        preferred_methods = []

        rep_osa = (representative_audit or {}).get("oversampling_audit", {})
        rep_rec = rep_osa.get("oversampling_recommendation", {}) if isinstance(rep_osa, dict) else {}
        rep_knn = rep_osa.get("knn_feasibility", {}) if isinstance(rep_osa, dict) else {}

        if flags.get("has_schema_drift"):
            blocked_reasons.append("schema_drift")
        if flags.get("has_leakage_full_row") or flags.get("has_leakage_inputs_only"):
            blocked_reasons.append("train_test_leakage")
        if flags.get("is_multioutput"):
            blocked_reasons.append("multioutput_target")
        if not rep_knn.get("3", {}).get("is_feasible", False):
            blocked_reasons.append("neighbor_oversampling_not_feasible_for_k3")

        if flags.get("has_knn_feasibility_fragile"):
            cautious_reasons.append("neighbor_feasibility_fragile")
        if flags.get("has_high_overlap"):
            cautious_reasons.append("high_minority_overlap")
        if flags.get("has_high_hardness"):
            cautious_reasons.append("high_minority_hardness")
        if flags.get("has_high_noise_suspicion"):
            cautious_reasons.append("high_noise_suspicion")
        if flags.get("has_high_interpolation_risk"):
            cautious_reasons.append("high_interpolation_risk")
        if flags.get("has_risky_minority_topology"):
            cautious_reasons.append("risky_minority_topology")
        if flags.get("has_partition_instability"):
            cautious_reasons.append("partition_instability")
        if flags.get("has_missing_in_inputs"):
            cautious_reasons.append("missing_in_inputs")

        if blocked_reasons:
            status = "NOT_SUITABLE_FOR_STANDARD_OVERSAMPLING"
        elif cautious_reasons:
            status = "USE_WITH_CAUTION"
        else:
            status = "GOOD_FOR_STANDARD_OVERSAMPLING"

        if status == "NOT_SUITABLE_FOR_STANDARD_OVERSAMPLING":
            preferred_methods = ["baseline_without_oversampling", "manual_review", "data_cleaning_before_resampling"]
        elif status == "USE_WITH_CAUTION":
            preferred_methods = ["k3_neighbor_methods", "filtered_or_borderline_aware_oversamplers", "subgroup_analysis"]
        else:
            preferred_methods = ["SMOTE_family_standard_benchmark", "k5_neighbor_methods"]

        if isinstance(partition_stability, dict) and partition_stability.get("is_stable") is False:
            preferred_methods.append("report_fold_instability_explicitly")

        return {
            "status": status,
            "blocked_reasons": blocked_reasons,
            "cautious_reasons": cautious_reasons,
            "preferred_actions": preferred_methods,
            "representative_file_recommendation": rep_rec,
        }

    @staticmethod
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
            "knn_feasibility_fragile": 10,
            "high_overlap": 12,
            "high_hardness": 10,
            "high_noise_suspicion": 10,
            "high_interpolation_risk": 15,
            "risky_minority_topology": 10,
            "partition_instability": 10,
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
        add("knn_feasibility_fragile", bool(flags.get("has_knn_feasibility_fragile", False)))
        add("high_overlap", bool(flags.get("has_high_overlap", False)))
        add("high_hardness", bool(flags.get("has_high_hardness", False)))
        add("high_noise_suspicion", bool(flags.get("has_high_noise_suspicion", False)))
        add("high_interpolation_risk", bool(flags.get("has_high_interpolation_risk", False)))
        add("risky_minority_topology", bool(flags.get("has_risky_minority_topology", False)))
        add("partition_instability", bool(flags.get("has_partition_instability", False)))

        score = min(100, int(total))
        return {"score_0_100": score, "breakdown": breakdown, "weights": weights}

    @staticmethod
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
        if flags.get("has_knn_feasibility_fragile", False):
            risky.append("neighbor_feasibility_fragile")
        if flags.get("has_high_overlap", False):
            risky.append("high_minority_overlap")
        if flags.get("has_high_hardness", False):
            risky.append("high_minority_hardness")
        if flags.get("has_high_noise_suspicion", False):
            risky.append("high_noise_suspicion")
        if flags.get("has_high_interpolation_risk", False):
            risky.append("high_interpolation_risk")
        if flags.get("has_risky_minority_topology", False):
            risky.append("risky_minority_topology")
        if flags.get("has_partition_instability", False):
            risky.append("partition_instability")

        status = "SAFE"
        if blocked:
            status = "BLOCKED"
        elif risky:
            status = "RISKY"

        return {"status": status, "blocked_reasons": blocked, "risky_reasons": risky}


class DatasetExecutiveSummaryBuilder:
    """
    @brief Dataset-level executive summary assembly.
    """

    @staticmethod
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
            "has_knn_feasibility_fragile": False,
            "has_high_overlap": False,
            "has_high_hardness": False,
            "has_high_noise_suspicion": False,
            "has_high_interpolation_risk": False,
            "has_risky_minority_topology": False,
            "has_partition_instability": False,
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

            osa = a.get("oversampling_audit", {})
            if isinstance(osa, dict) and osa.get("enabled", False):
                kf = osa.get("knn_feasibility", {})
                if kf.get("5", {}).get("is_fragile") or kf.get("7", {}).get("is_fragile"):
                    flags["has_knn_feasibility_fragile"] = True

                oh = osa.get("overlap_hardness", {})
                overlap_mean = oh.get("minority_majority_neighbor_ratio_mean")
                hardness_mean = oh.get("instance_hardness_minority_mean")
                if isinstance(overlap_mean, (int, float)) and np.isfinite(overlap_mean) and overlap_mean >= 0.50:
                    flags["has_high_overlap"] = True
                if isinstance(hardness_mean, (int, float)) and np.isfinite(hardness_mean) and hardness_mean >= 0.45:
                    flags["has_high_hardness"] = True

                ns_rate = osa.get("noise_suspicion", {}).get("minority_noise_suspicion_rate")
                if isinstance(ns_rate, (int, float)) and np.isfinite(ns_rate) and ns_rate >= 0.25:
                    flags["has_high_noise_suspicion"] = True

                risk_score = osa.get("synthetic_interpolation_risk", {}).get("score_0_100")
                if isinstance(risk_score, (int, float)) and np.isfinite(risk_score) and risk_score >= 70:
                    flags["has_high_interpolation_risk"] = True

                topo = osa.get("minority_topology", {}).get("ratios", {})
                rare_ratio = topo.get("rare")
                outlier_ratio = topo.get("outlier")
                rare_ratio = float(rare_ratio) if isinstance(rare_ratio, (int, float)) and np.isfinite(rare_ratio) else 0.0
                outlier_ratio = float(outlier_ratio) if isinstance(outlier_ratio, (int, float)) and np.isfinite(outlier_ratio) else 0.0
                if (rare_ratio + outlier_ratio) >= 0.50:
                    flags["has_risky_minority_topology"] = True

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

        partition_stability = _compute_partition_stability(ok_audits)
        flags["has_partition_instability"] = bool(not partition_stability.get("is_stable", True))

        drift_reasons = schema_drift_entry.get("drift_reasons_summary", {}) if isinstance(schema_drift_entry, dict) else {}

        severity = _severity_score_from_flags(flags)
        severity["dominant_drivers"] = _dominant_severity_drivers(severity.get("breakdown", {}))
        readiness = _run_readiness(flags)
        oversampling_recommendation = _oversampling_dataset_recommendation(
            flags,
            partition_stability=partition_stability,
            representative_audit=rep,
        )

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
            "partition_stability_summary": partition_stability,
            "oversampling_summary": {
                "representative_file_recommendation": (rep.get("oversampling_audit", {}).get("oversampling_recommendation") if rep else {}),
                "representative_file_overlap_hardness": (rep.get("oversampling_audit", {}).get("overlap_hardness") if rep else {}),
                "representative_file_topology": (rep.get("oversampling_audit", {}).get("minority_topology") if rep else {}),
                "representative_file_synthetic_interpolation_risk": (rep.get("oversampling_audit", {}).get("synthetic_interpolation_risk") if rep else {}),
                "dataset_recommendation": oversampling_recommendation,
            },
            "flags": flags,
            "run_readiness": readiness,
            "severity": severity,
        }


class ErrorAuditPayloadBuilder:
    """
    @brief Error payload construction for failed audits.
    """

    @staticmethod
    def _safe_error_audit(file_path: str, error: Exception, *, out_dir: Optional[str], save: bool) -> Dict[str, Any]:
        """
        @brief Create an error audit payload for a file that cannot be audited.
        """
        abs_path = os.path.abspath(file_path)
        folder = os.path.dirname(abs_path)
        basename = os.path.basename(abs_path)
        fold_info = _parse_fold_filename(basename)

        payload = {
            "audit_version": "1.3",
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


class FileAuditWorkflow:
    """
    @brief Level-1 single-file KEEL audit workflow.
    """

    @staticmethod
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
        oversampling_analysis_max_rows: int = 10000,
        oversampling_feasibility_ks: Optional[List[int]] = None,
        oversampling_topology_k: int = 5,
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

        oversampling_audit = _oversampling_knn_diagnostics(
            df,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            k_neighbors=oversampling_topology_k,
            feasibility_ks=oversampling_feasibility_ks or [3, 5, 7],
            max_rows=oversampling_analysis_max_rows,
        )
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
            "audit_version": "1.3",
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
            "oversampling_audit": oversampling_audit,
        }

        if save:
            if out_dir is None:
                out_path = os.path.join(folder, "_audit", "files", f"{basename}.audit.json")
            else:
                ds_folder = _dataset_folder_name_from_path(folder)
                out_path = os.path.join(out_dir, ds_folder, "files", f"{basename}.audit.json")
            result["saved_to"] = _save_json(out_path, result)

        return result


class FolderAuditWorkflow:
    """
    @brief Level-2 folder/fold KEEL audit workflow.
    """

    @staticmethod
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
        datasets_with_knn_fragility = []
        datasets_with_high_overlap = []
        datasets_with_high_hardness = []
        datasets_with_high_noise_suspicion = []
        datasets_with_high_interpolation_risk = []
        datasets_with_risky_topology = []
        datasets_with_partition_instability = []

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

        for row in dataset_key_summaries:
            ds_key = row.get("dataset_key")
            flags = row.get("flags", {})
            if flags.get("has_knn_feasibility_fragile"):
                datasets_with_knn_fragility.append(ds_key)
            if flags.get("has_high_overlap"):
                datasets_with_high_overlap.append(ds_key)
            if flags.get("has_high_hardness"):
                datasets_with_high_hardness.append(ds_key)
            if flags.get("has_high_noise_suspicion"):
                datasets_with_high_noise_suspicion.append(ds_key)
            if flags.get("has_high_interpolation_risk"):
                datasets_with_high_interpolation_risk.append(ds_key)
            if flags.get("has_risky_minority_topology"):
                datasets_with_risky_topology.append(ds_key)
            if flags.get("has_partition_instability"):
                datasets_with_partition_instability.append(ds_key)

        result: Dict[str, Any] = {
            "audit_version": "1.3",
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
                "has_knn_fragility_any": bool(len(datasets_with_knn_fragility) > 0),
                "has_high_overlap_any": bool(len(datasets_with_high_overlap) > 0),
                "has_high_hardness_any": bool(len(datasets_with_high_hardness) > 0),
                "has_high_noise_suspicion_any": bool(len(datasets_with_high_noise_suspicion) > 0),
                "has_high_interpolation_risk_any": bool(len(datasets_with_high_interpolation_risk) > 0),
                "has_risky_topology_any": bool(len(datasets_with_risky_topology) > 0),
                "has_partition_instability_any": bool(len(datasets_with_partition_instability) > 0),
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
                "datasets_with_knn_fragility": sorted(list(set(datasets_with_knn_fragility))),
                "datasets_with_high_overlap": sorted(list(set(datasets_with_high_overlap))),
                "datasets_with_high_hardness": sorted(list(set(datasets_with_high_hardness))),
                "datasets_with_high_noise_suspicion": sorted(list(set(datasets_with_high_noise_suspicion))),
                "datasets_with_high_interpolation_risk": sorted(list(set(datasets_with_high_interpolation_risk))),
                "datasets_with_risky_topology": sorted(list(set(datasets_with_risky_topology))),
                "datasets_with_partition_instability": sorted(list(set(datasets_with_partition_instability))),
            },
        }

        if save:
            if out_dir is None:
                out_path = os.path.join(abs_folder, "_audit", "folder.audit.json")
            else:
                out_path = os.path.join(out_dir, folder_name, "folder.audit.json")
            result["saved_to"] = _save_json(out_path, result)

        return result


class RootAuditWorkflow:
    """
    @brief Level-3 root-level KEEL audit workflow.
    """

    @staticmethod
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
        global_knn_fragility = []
        global_high_overlap = []
        global_high_hardness = []
        global_high_noise_suspicion = []
        global_high_interpolation_risk = []
        global_risky_topology = []
        global_partition_instability = []

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

            for ds in fr["high_level_lists"].get("datasets_with_knn_fragility", []):
                global_knn_fragility.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_high_overlap", []):
                global_high_overlap.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_high_hardness", []):
                global_high_hardness.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_high_noise_suspicion", []):
                global_high_noise_suspicion.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_high_interpolation_risk", []):
                global_high_interpolation_risk.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_risky_topology", []):
                global_risky_topology.append({"folder": folder_name, "dataset": ds})
            for ds in fr["high_level_lists"].get("datasets_with_partition_instability", []):
                global_partition_instability.append({"folder": folder_name, "dataset": ds})

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

        profile_vectors = []
        for r in executive_rows:
            flags = r.get("flags", {})
            overlap = r.get("oversampling_summary", {}).get("representative_file_overlap_hardness", {}).get("minority_majority_neighbor_ratio_mean")
            hardness = r.get("oversampling_summary", {}).get("representative_file_overlap_hardness", {}).get("instance_hardness_minority_mean")
            risk = r.get("oversampling_summary", {}).get("representative_file_synthetic_interpolation_risk", {}).get("score_0_100")
            ir = r.get("class_summary", {}).get("imbalance_rate_ir_train_aggregate")
            min_count = r.get("class_summary", {}).get("min_class_count_train_aggregate")
            vec = [
                float(overlap) if isinstance(overlap, (int, float)) and np.isfinite(overlap) else 0.0,
                float(hardness) if isinstance(hardness, (int, float)) and np.isfinite(hardness) else 0.0,
                (float(risk) / 100.0) if isinstance(risk, (int, float)) and np.isfinite(risk) else 0.0,
                min(float(ir) / 20.0, 1.0) if isinstance(ir, (int, float)) and np.isfinite(ir) else 0.0,
                min(float(min_count) / 20.0, 1.0) if isinstance(min_count, (int, float)) and np.isfinite(min_count) else 0.0,
                1.0 if flags.get("has_partition_instability") else 0.0,
            ]
            profile_vectors.append({"folder": r.get("folder"), "dataset_key": r.get("dataset_key"), "vector": np.asarray(vec, dtype=float)})

        similar_dataset_pairs = []
        for i in range(len(profile_vectors)):
            for j in range(i + 1, len(profile_vectors)):
                v1 = profile_vectors[i]["vector"]
                v2 = profile_vectors[j]["vector"]
                dist = float(np.linalg.norm(v1 - v2))
                if dist <= 0.15:
                    similar_dataset_pairs.append({
                        "dataset_a": {"folder": profile_vectors[i]["folder"], "dataset_key": profile_vectors[i]["dataset_key"]},
                        "dataset_b": {"folder": profile_vectors[j]["folder"], "dataset_key": profile_vectors[j]["dataset_key"]},
                        "profile_distance": dist,
                    })
        similar_dataset_pairs = sorted(similar_dataset_pairs, key=lambda x: x.get("profile_distance", 1e9))[: int(max(10, top_n_offenders))]

        issue_counters = {
            "blocked": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "BLOCKED")),
            "risky": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "RISKY")),
            "safe": int(sum(1 for r in executive_rows if r.get("run_readiness", {}).get("status") == "SAFE")),
            "has_leakage": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_leakage_inputs_only") or r.get("flags", {}).get("has_leakage_full_row"))),
            "has_schema_drift": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_schema_drift"))),
            "has_missing_outputs": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_missing_in_outputs"))),
            "has_label_conflicts": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_label_conflicts"))),
            "has_knn_fragility": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_knn_feasibility_fragile"))),
            "has_high_overlap": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_high_overlap"))),
            "has_high_noise_suspicion": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_high_noise_suspicion"))),
            "has_high_interpolation_risk": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_high_interpolation_risk"))),
            "has_partition_instability": int(sum(1 for r in executive_rows if r.get("flags", {}).get("has_partition_instability"))),
        }

        result: Dict[str, Any] = {
            "audit_version": "1.3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "level": 3,
            "root": {"abs_path": abs_root, "n_dataset_folders": int(len(subdirs)), "dataset_folders": subdirs},
            "cv_consistency": {"all_same_cv_type": bool(all_same_cv), "has_cv_flags_per_folder": has_cv_flags, "cv_k_values_union": sorted(list(cv_k_union))},
            "executive_summary": {"issue_counters": issue_counters, "top_offenders": top_offenders, "similar_dataset_pairs": similar_dataset_pairs, "per_dataset": executive_rows},
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
                "datasets_with_knn_fragility": global_knn_fragility,
                "datasets_with_high_overlap": global_high_overlap,
                "datasets_with_high_hardness": global_high_hardness,
                "datasets_with_high_noise_suspicion": global_high_noise_suspicion,
                "datasets_with_high_interpolation_risk": global_high_interpolation_risk,
                "datasets_with_risky_topology": global_risky_topology,
                "datasets_with_partition_instability": global_partition_instability,
            },
        }

        if save:
            out_path = os.path.join(out_base, "root.audit.json")
            result["saved_to"] = _save_json(out_path, result)

        return result

# -----------------------------
# Optional helper imports
# -----------------------------
_DISCOVER_KEEL_CV_FOLDS, _PARSE_KEEL_FILE = ProjectHelperImportBridge._import_keel_helpers()


# -----------------------------
# Backing aliases for grouped helper methods
# -----------------------------
_jsonify = JsonArtifactIO._jsonify
_save_json = JsonArtifactIO._save_json
_dataset_folder_name_from_path = PathLayoutHelper._dataset_folder_name_from_path

_read_text = KEELParsingHelper._read_text
_detect_encoding_issues = KEELParsingHelper._detect_encoding_issues
_parse_attribute_definition = KEELParsingHelper._parse_attribute_definition
_parse_keel_header = KEELParsingHelper._parse_keel_header
_split_csv_like = KEELParsingHelper._split_csv_like
_load_keel_data = KEELParsingHelper._load_keel_data
_is_missing_token = KEELParsingHelper._is_missing_token
_is_suspicious_null_like = KEELParsingHelper._is_suspicious_null_like
_schema_signature = KEELParsingHelper._schema_signature
_diff_schema_signatures = KEELParsingHelper._diff_schema_signatures
_row_hash = KEELParsingHelper._row_hash
_estimate_one_hot_dim = KEELParsingHelper._estimate_one_hot_dim
_parse_fold_filename = KEELParsingHelper._parse_fold_filename

_safe_float_array = NumericStatisticsHelper._safe_float_array
_normalize_numeric_frame = NumericStatisticsHelper._normalize_numeric_frame
_skewness = NumericStatisticsHelper._skewness
_kurtosis = NumericStatisticsHelper._kurtosis
_robust_numeric_stats = NumericStatisticsHelper._robust_numeric_stats
_robust_outlier_rate = NumericStatisticsHelper._robust_outlier_rate
_safe_mean = NumericStatisticsHelper._safe_mean
_safe_min = NumericStatisticsHelper._safe_min
_safe_max = NumericStatisticsHelper._safe_max
_summarize_numeric_series = NumericStatisticsHelper._summarize_numeric_series

_build_knn_ready_matrix = OversamplingAnalysisHelper._build_knn_ready_matrix
_topology_label_from_majority_count = OversamplingAnalysisHelper._topology_label_from_majority_count
_oversampling_knn_diagnostics = OversamplingAnalysisHelper._oversampling_knn_diagnostics

_feature_missingness = FeatureQualityHelper._feature_missingness
_constant_and_near_constant = FeatureQualityHelper._constant_and_near_constant
_duplicate_rows = FeatureQualityHelper._duplicate_rows
_high_cardinality_nominal = FeatureQualityHelper._high_cardinality_nominal
_label_conflicts = FeatureQualityHelper._label_conflicts

_compute_leakage_metrics_from_hashes = SplitAuditHelper._compute_leakage_metrics_from_hashes
_class_presence_in_split = SplitAuditHelper._class_presence_in_split
_compute_partition_stability = SplitAuditHelper._compute_partition_stability

_dominant_severity_drivers = DecisionSupportHelper._dominant_severity_drivers
_oversampling_dataset_recommendation = DecisionSupportHelper._oversampling_dataset_recommendation
_severity_score_from_flags = DecisionSupportHelper._severity_score_from_flags
_run_readiness = DecisionSupportHelper._run_readiness

_aggregate_dataset_key_summary = DatasetExecutiveSummaryBuilder._aggregate_dataset_key_summary
_safe_error_audit = ErrorAuditPayloadBuilder._safe_error_audit

_procedural_audit_keel_file = FileAuditWorkflow.audit_keel_file
_procedural_audit_keel_folder = FolderAuditWorkflow.audit_keel_folder
_procedural_audit_keel_root = RootAuditWorkflow.audit_keel_root


@dataclass(frozen=True)
class AuditCategoryDescriptor:
    """
    @brief Lightweight descriptor for one logical family of audit analyses.
    """
    key: str
    title: str
    description: str
    supported_levels: Tuple[int, ...]
    owned_top_level_keys: Tuple[str, ...]
    folder_summary_keys: Tuple[str, ...] = ()
    root_summary_keys: Tuple[str, ...] = ()


class BaseAuditCategory:
    """
    @brief Base class for one logical audit category.
    """
    descriptor: AuditCategoryDescriptor

    def build_metadata(self) -> Dict[str, Any]:
        d = self.descriptor
        return {
            "key": d.key,
            "title": d.title,
            "description": d.description,
            "supported_levels": list(d.supported_levels),
            "owned_top_level_keys": list(d.owned_top_level_keys),
            "folder_summary_keys": list(d.folder_summary_keys),
            "root_summary_keys": list(d.root_summary_keys),
        }


class StructureFormatAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="structure_format",
        title="بررسی ساختار و فرمت فایل KEEL",
        description="خواندن هدر، ساخت schema، تشخیص نوع ویژگی‌ها، شکل فایل و مشکلات پایه‌ی encoding/format.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("relation", "file", "schema", "dataset_shape", "encoding_issues"),
        folder_summary_keys=("schema_consistency",),
        root_summary_keys=("cv_consistency",),
    )


class MissingTokenAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="missing_tokens",
        title="بررسی missing و tokenهای مشکوک",
        description="تحلیل missingness، missing در ورودی/خروجی، missing زیاد و tokenهای null-like یا ناشناخته.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("data_quality",),
        folder_summary_keys=("high_level_flags", "high_level_lists"),
        root_summary_keys=("global_summary", "executive_summary"),
    )


class DomainConsistencyAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="domain_consistency",
        title="سازگاری دامنه‌ها و schema",
        description="سازگاری nominal و numeric با header، مقادیر خارج از بازه، parseability و drift بین foldها.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("data_quality",),
        folder_summary_keys=("schema_consistency", "high_level_flags", "high_level_lists"),
        root_summary_keys=("global_summary",),
    )


class NumericDistributionAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="numeric_distribution",
        title="آمار عددی، tails و outlier",
        description="آمار مقاوم، outlier rate و رفتار توزیع featureهای عددی.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("numeric_feature_stats",),
        folder_summary_keys=("executive_dataset_summaries",),
        root_summary_keys=("executive_summary",),
    )


class FeatureQualityAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="feature_quality",
        title="کیفیت featureها و آمادگی preprocessing",
        description="constant/near-constant، cardinality بالا و توصیه‌های preprocessing.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("feature_quality", "preprocess_recommendations"),
        folder_summary_keys=("executive_dataset_summaries", "high_level_lists"),
        root_summary_keys=("global_summary",),
    )


class DuplicateLabelAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="duplicate_and_label_integrity",
        title="تکراری‌ها و سلامت برچسب",
        description="duplicate rows، label conflict و موارد مشابه مربوط به integrity رکوردها.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("data_quality",),
        folder_summary_keys=("cv_leakage", "high_level_lists"),
        root_summary_keys=("global_summary",),
    )


class ClassImbalanceAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="class_imbalance",
        title="تحلیل target و عدم‌توازن",
        description="نوع مسئله، شمارش کلاس‌ها، IR، minority class و tiny class.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("class_info",),
        folder_summary_keys=("cv_class_presence", "executive_dataset_summaries"),
        root_summary_keys=("global_summary", "executive_summary"),
    )


class OversamplingSuitabilityAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="oversampling_suitability",
        title="تحلیل مخصوص oversampling",
        description="feasibility روش‌های همسایگی‌محور، topology اقلیت، overlap، hardness، noise و recommendation.",
        supported_levels=(1, 2, 3),
        owned_top_level_keys=("oversampling_audit",),
        folder_summary_keys=("executive_dataset_summaries", "high_level_lists"),
        root_summary_keys=("global_summary", "executive_summary"),
    )


class SplitShiftLeakageAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="split_shift_leakage",
        title="تحلیل split، shift و leakage",
        description="leakage، class presence و drift بین train/test یا foldها.",
        supported_levels=(2, 3),
        owned_top_level_keys=(),
        folder_summary_keys=("schema_consistency", "cv_leakage", "cv_class_presence", "executive_dataset_summaries"),
        root_summary_keys=("global_summary", "executive_summary", "cv_consistency"),
    )


class SeverityRecommendationAuditCategory(BaseAuditCategory):
    descriptor = AuditCategoryDescriptor(
        key="severity_recommendation",
        title="severity، readiness و recommendation",
        description="severity score، drivers، run readiness و خلاصه‌های اجرایی برای تصمیم‌گیری.",
        supported_levels=(2, 3),
        owned_top_level_keys=(),
        folder_summary_keys=("executive_dataset_summaries", "high_level_flags", "high_level_lists"),
        root_summary_keys=("global_summary", "executive_summary"),
    )


class BaseKEELAuditFamily:
    """
    @brief Base class for one KEEL dataset family audit implementation.
    """
    family_key: str = "base"
    display_name: str = "Base KEEL Audit Family"
    description: str = "Base family. Override in subclasses."

    def build_categories(self) -> List[BaseAuditCategory]:
        raise NotImplementedError

    def _build_architecture_block(self) -> Dict[str, Any]:
        categories = [cat.build_metadata() for cat in self.build_categories()]
        return {
            "family_key": self.family_key,
            "display_name": self.display_name,
            "description": self.description,
            "categories": categories,
        }

    def _attach_architecture_metadata(self, result: Dict[str, Any], level: int) -> Dict[str, Any]:
        result["audit_family"] = {
            "key": self.family_key,
            "display_name": self.display_name,
            "description": self.description,
        }
        result["audit_architecture"] = self._build_architecture_block()
        result["audit_level"] = int(level)
        return result

    def audit_file(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def audit_folder(self, folder_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def audit_root(self, root_folder: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class _PlaceholderFutureFamily(BaseKEELAuditFamily):
    """
    @brief Placeholder family for future KEEL dataset categories.
    """
    def build_categories(self) -> List[BaseAuditCategory]:
        return []

    def audit_file(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Audit family '{self.family_key}' is planned but not implemented yet.")

    def audit_folder(self, folder_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Audit family '{self.family_key}' is planned but not implemented yet.")

    def audit_root(self, root_folder: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Audit family '{self.family_key}' is planned but not implemented yet.")


class ImbalancedOversamplingAuditFamily(BaseKEELAuditFamily):
    """
    @brief Active family for the current project focus: imbalanced classification datasets used in oversampling research.
    """
    family_key = "imbalanced_oversampling"
    display_name = "Imbalanced Classification for Oversampling"
    description = (
        "Family focused on KEEL imbalanced classification datasets that are used "
        "for oversampling benchmarks and data-quality auditing before modeling."
    )

    def build_categories(self) -> List[BaseAuditCategory]:
        return [
            StructureFormatAuditCategory(),
            MissingTokenAuditCategory(),
            DomainConsistencyAuditCategory(),
            NumericDistributionAuditCategory(),
            FeatureQualityAuditCategory(),
            DuplicateLabelAuditCategory(),
            ClassImbalanceAuditCategory(),
            OversamplingSuitabilityAuditCategory(),
            SplitShiftLeakageAuditCategory(),
            SeverityRecommendationAuditCategory(),
        ]

    def audit_file(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        result = _procedural_audit_keel_file(file_path, **kwargs)
        return self._attach_architecture_metadata(result, level=1)

    def audit_folder(self, folder_path: str, **kwargs: Any) -> Dict[str, Any]:
        result = _procedural_audit_keel_folder(folder_path, **kwargs)
        return self._attach_architecture_metadata(result, level=2)

    def audit_root(self, root_folder: str, **kwargs: Any) -> Dict[str, Any]:
        result = _procedural_audit_keel_root(root_folder, **kwargs)
        return self._attach_architecture_metadata(result, level=3)


class StandardClassificationAuditFamily(_PlaceholderFutureFamily):
    family_key = "standard_classification"
    display_name = "Standard Classification"
    description = "Planned family for standard KEEL supervised classification datasets."


class MissingValueClassificationAuditFamily(_PlaceholderFutureFamily):
    family_key = "classification_missing_values"
    display_name = "Classification with Missing Values"
    description = "Planned family for KEEL classification datasets with missing values."


class MultiInstanceClassificationAuditFamily(_PlaceholderFutureFamily):
    family_key = "multi_instance_classification"
    display_name = "Multi-Instance Classification"
    description = "Planned family for KEEL multi-instance classification datasets."


class MultiLabelClassificationAuditFamily(_PlaceholderFutureFamily):
    family_key = "multi_label_classification"
    display_name = "Multi-Label Classification"
    description = "Planned family for KEEL multi-label classification datasets."


class ClassificationWithClassNoiseAuditFamily(_PlaceholderFutureFamily):
    family_key = "classification_class_noise"
    display_name = "Classification with Class Noise"
    description = "Planned family for KEEL classification datasets with class noise."


class ClassificationWithAttributeNoiseAuditFamily(_PlaceholderFutureFamily):
    family_key = "classification_attribute_noise"
    display_name = "Classification with Attribute Noise"
    description = "Planned family for KEEL classification datasets with attribute noise."


class SemiSupervisedClassificationAuditFamily(_PlaceholderFutureFamily):
    family_key = "semi_supervised_classification"
    display_name = "Semi-Supervised Classification"
    description = "Planned family for KEEL semi-supervised classification datasets."


class RegressionAuditFamily(_PlaceholderFutureFamily):
    family_key = "regression"
    display_name = "Regression"
    description = "Planned family for KEEL regression datasets."


class TimeSeriesAuditFamily(_PlaceholderFutureFamily):
    family_key = "time_series"
    display_name = "Time Series"
    description = "Planned family for KEEL time-series datasets."


class UnsupervisedAuditFamily(_PlaceholderFutureFamily):
    family_key = "unsupervised"
    display_name = "Unsupervised"
    description = "Planned family for KEEL unsupervised datasets."


class LowQualityAuditFamily(_PlaceholderFutureFamily):
    family_key = "low_quality"
    display_name = "Low Quality"
    description = "Planned family for KEEL low-quality datasets."


class KEELAuditFamilyRegistry:
    """
    @brief Registry of available audit families.
    """
    def __init__(self) -> None:
        self._families: Dict[str, BaseKEELAuditFamily] = {}

    def register(self, family: BaseKEELAuditFamily) -> None:
        self._families[family.family_key] = family

    def get(self, family_key: str) -> BaseKEELAuditFamily:
        if family_key not in self._families:
            known = ", ".join(sorted(self._families))
            raise KeyError(f"Unknown KEEL audit family: '{family_key}'. Known families: {known}")
        return self._families[family_key]

    def list_families(self) -> List[Dict[str, str]]:
        return [
            {
                "key": fam.family_key,
                "display_name": fam.display_name,
                "description": fam.description,
            }
            for fam in self._families.values()
        ]


class KEELAuditFacade:
    """
    @brief Facade that exposes the stable public API while routing to the chosen family.
    """
    def __init__(self, registry: Optional[KEELAuditFamilyRegistry] = None) -> None:
        self.registry = registry or _DEFAULT_AUDIT_REGISTRY

    def audit_file(self, file_path: str, *, dataset_family: str = "imbalanced_oversampling", **kwargs: Any) -> Dict[str, Any]:
        family = self.registry.get(dataset_family)
        return family.audit_file(file_path, **kwargs)

    def audit_folder(self, folder_path: str, *, dataset_family: str = "imbalanced_oversampling", **kwargs: Any) -> Dict[str, Any]:
        family = self.registry.get(dataset_family)
        return family.audit_folder(folder_path, **kwargs)

    def audit_root(self, root_folder: str, *, dataset_family: str = "imbalanced_oversampling", **kwargs: Any) -> Dict[str, Any]:
        family = self.registry.get(dataset_family)
        return family.audit_root(root_folder, **kwargs)

    def list_supported_families(self) -> List[Dict[str, str]]:
        return self.registry.list_families()


_DEFAULT_AUDIT_REGISTRY = KEELAuditFamilyRegistry()
for _family in [
    ImbalancedOversamplingAuditFamily(),
    StandardClassificationAuditFamily(),
    MissingValueClassificationAuditFamily(),
    MultiInstanceClassificationAuditFamily(),
    MultiLabelClassificationAuditFamily(),
    ClassificationWithClassNoiseAuditFamily(),
    ClassificationWithAttributeNoiseAuditFamily(),
    SemiSupervisedClassificationAuditFamily(),
    RegressionAuditFamily(),
    TimeSeriesAuditFamily(),
    UnsupervisedAuditFamily(),
    LowQualityAuditFamily(),
]:
    _DEFAULT_AUDIT_REGISTRY.register(_family)


_DEFAULT_AUDIT_FACADE = KEELAuditFacade()


def list_supported_audit_families() -> List[Dict[str, str]]:
    """
    @brief Return the registered KEEL dataset families supported by the architecture.
    """
    return _DEFAULT_AUDIT_FACADE.list_supported_families()


def describe_active_audit_categories(dataset_family: str = "imbalanced_oversampling") -> List[Dict[str, Any]]:
    """
    @brief Return category metadata for one dataset family.
    """
    family = _DEFAULT_AUDIT_REGISTRY.get(dataset_family)
    return [cat.build_metadata() for cat in family.build_categories()]


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
    oversampling_analysis_max_rows: int = 10000,
    oversampling_feasibility_ks: Optional[List[int]] = None,
    oversampling_topology_k: int = 5,
    dataset_family: str = "imbalanced_oversampling",
) -> Dict[str, Any]:
    """
    @brief Public API for file-level KEEL audit.
    """
    return _DEFAULT_AUDIT_FACADE.audit_file(
        file_path,
        dataset_family=dataset_family,
        out_dir=out_dir,
        save=save,
        tolerance=tolerance,
        near_constant_threshold=near_constant_threshold,
        high_missingness_threshold=high_missingness_threshold,
        high_cardinality_threshold=high_cardinality_threshold,
        robust_z_threshold=robust_z_threshold,
        oversampling_analysis_max_rows=oversampling_analysis_max_rows,
        oversampling_feasibility_ks=oversampling_feasibility_ks,
        oversampling_topology_k=oversampling_topology_k,
    )


def audit_keel_folder(
    folder_path: str,
    *,
    out_dir: Optional[str] = None,
    save: bool = True,
    dataset_family: str = "imbalanced_oversampling",
) -> Dict[str, Any]:
    """
    @brief Public API for folder-level KEEL audit.
    """
    return _DEFAULT_AUDIT_FACADE.audit_folder(
        folder_path,
        dataset_family=dataset_family,
        out_dir=out_dir,
        save=save,
    )


def audit_keel_root(
    root_folder: str,
    *,
    out_dir: Optional[str] = None,
    save: bool = True,
    top_n_offenders: int = 20,
    dataset_family: str = "imbalanced_oversampling",
) -> Dict[str, Any]:
    """
    @brief Public API for root-level KEEL audit.
    """
    return _DEFAULT_AUDIT_FACADE.audit_root(
        root_folder,
        dataset_family=dataset_family,
        out_dir=out_dir,
        save=save,
        top_n_offenders=top_n_offenders,
    )


__all__ = [
    "audit_keel_file",
    "audit_keel_folder",
    "audit_keel_root",
    "list_supported_audit_families",
    "describe_active_audit_categories",
    "KEELAuditFacade",
    "KEELAuditFamilyRegistry",
    "BaseKEELAuditFamily",
    "BaseAuditCategory",
    "ImbalancedOversamplingAuditFamily",
]
