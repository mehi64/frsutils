import pytest

from FRsutils.utils.dataset_utils.KEEL_Audit_Utility import (
    audit_keel_file,
    audit_keel_folder,
    audit_keel_root,
)


def _write(p, text: str):
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_audit_keel_file_quality_checks(tmp_path):
    dat = """@relation iris0
@attribute f1 real [0,10]
@attribute color {red, green}
@attribute class {A,B}
@inputs f1, color
@outputs class
@data
1, red, A
?, green, B
11, blue, A
<null>, red, ?
1, red, A
"""
    fp = _write(tmp_path / "iris0-5-1tra.dat", dat)

    res = audit_keel_file(fp, out_dir=str(tmp_path / "out"), save=True)

    assert res["level"] == 1
    assert res["relation"] == "iris0"
    assert res["file"]["is_train"] is True

    # missing
    assert res["data_quality"]["missing"]["missing_in_inputs"] is True
    assert res["data_quality"]["missing"]["missing_in_outputs"] is True

    # nominal: blue not in header
    nom_bad = res["data_quality"]["nominal"]["values_in_data_not_in_header"]
    assert "color" in nom_bad
    assert "blue" in nom_bad["color"]

    # numeric out-of-range: 11 outside [0,10]
    oor = res["data_quality"]["numeric"]["out_of_range_vs_header"]
    assert "f1" in oor
    assert oor["f1"]["n_out_of_range"] >= 1

    # duplicates: last row duplicates first row
    dup = res["data_quality"]["duplicates"]
    assert dup["n_duplicate_rows"] >= 1

    # label conflicts should be 0 in this example
    assert res["data_quality"]["label_conflicts"]["n_conflicting_input_groups"] == 0

    # preprocess recommendations
    assert "color" in res["preprocess_recommendations"]["needs_one_hot_encoding"]
    assert "f1" in res["preprocess_recommendations"]["needs_unit_interval_scaling_0_1"]


def test_audit_keel_folder_detects_schema_drift_and_leakage(tmp_path):
    # fold 1 tra/tst exist; fold 2 tra/tst exist but with schema drift (extra attribute)
    tra1 = """@relation ds
@attribute f1 real [0,1]
@attribute class {A,B}
@inputs f1
@outputs class
@data
0.1, A
0.2, B
"""
    tst1 = """@relation ds
@attribute f1 real [0,1]
@attribute class {A,B}
@inputs f1
@outputs class
@data
0.1, A
0.3, B
"""

    # schema drift: extra attribute f2
    tra2 = """@relation ds
@attribute f1 real [0,1]
@attribute f2 real [0,1]
@attribute class {A,B}
@inputs f1, f2
@outputs class
@data
0.1, 0.1, A
0.2, 0.2, B
"""
    tst2 = """@relation ds
@attribute f1 real [0,1]
@attribute f2 real [0,1]
@attribute class {A,B}
@inputs f1, f2
@outputs class
@data
0.1, 0.1, A
"""

    _write(tmp_path / "ds-2-1tra.dat", tra1)
    _write(tmp_path / "ds-2-1tst.dat", tst1)
    _write(tmp_path / "ds-2-2tra.dat", tra2)
    _write(tmp_path / "ds-2-2tst.dat", tst2)

    res = audit_keel_folder(str(tmp_path), out_dir=str(tmp_path / "out"), save=True)

    assert res["level"] == 2
    assert res["datasets_detected"]["has_cv_any"] is True

    # schema drift should be detected for dataset_key "ds"
    assert res["schema_consistency"]["has_schema_drift_any"] is True

    # leakage: train and test share input 0.1 for fold 1
    assert res["cv_leakage"]["has_leakage_any"] is True
    leaks = res["cv_leakage"]["leakage_reports"]
    assert any(l.get("fold_index") == 1 and l.get("n_intersection_inputs_only", 0) >= 1 for l in leaks)


def test_audit_keel_root_aggregates_new_lists(tmp_path):
    d1 = tmp_path / "ds1"
    d2 = tmp_path / "ds2"
    d1.mkdir()
    d2.mkdir()

    ok = """@relation ok
@attribute f1 real [0,1]
@attribute class {A,B}
@inputs f1
@outputs class
@data
0.1, A
0.2, B
"""
    # missing + multiclass + constant feature
    bad = """@relation bad
@attribute f1 real [0,1]
@attribute f2 real [0,1]
@attribute class {A,B,C}
@inputs f1, f2
@outputs class
@data
0.1, 0.5, A
0.2, 0.5, ?
"""

    _write(d1 / "ok.dat", ok)
    _write(d2 / "bad.dat", bad)

    res = audit_keel_root(str(tmp_path), out_dir=str(tmp_path / "out"), save=True)

    assert res["level"] == 3
    assert res["global_summary"]["has_missing_any"] is True
    assert res["global_summary"]["has_nonbinary_any"] is True

    # constant feature (f2) should appear
    consts = res["global_summary"]["datasets_with_constant_features"]
    assert any("features" in it and "f2" in it["features"] for it in consts)
