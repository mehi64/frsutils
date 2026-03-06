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

    assert res["data_quality"]["missing"]["missing_in_inputs"] is True
    assert res["data_quality"]["missing"]["missing_in_outputs"] is True

    nom_bad = res["data_quality"]["nominal"]["values_in_data_not_in_header"]
    assert "color" in nom_bad and "blue" in nom_bad["color"]

    oor = res["data_quality"]["numeric"]["out_of_range_vs_header"]
    assert "f1" in oor and oor["f1"]["n_out_of_range"] >= 1

    dup = res["data_quality"]["duplicates"]
    assert dup["n_duplicate_rows"] >= 1


def test_audit_keel_folder_exec_summary_and_leakage_rates(tmp_path):
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
    _write(tmp_path / "ds-2-1tra.dat", tra1)
    _write(tmp_path / "ds-2-1tst.dat", tst1)

    res = audit_keel_folder(str(tmp_path), out_dir=str(tmp_path / "out"), save=True)

    assert res["level"] == 2
    assert res["cv_leakage"]["has_leakage_any"] is True

    leaks = res["cv_leakage"]["leakage_reports"]
    l1 = [x for x in leaks if x.get("fold_index") == 1][0]
    assert "rate_inputs_only_over_test" in l1
    assert l1["rate_inputs_only_over_test"] >= 0.5  # 1 overlap / test size 2

    exec_rows = res["executive_dataset_summaries"]
    assert len(exec_rows) >= 1
    assert exec_rows[0]["severity"]["score_0_100"] >= 40  # leakage => high score


def test_audit_keel_root_top_offenders(tmp_path):
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
    tra = """@relation leak
@attribute f1 real [0,1]
@attribute class {A,B}
@inputs f1
@outputs class
@data
0.1, A
0.2, B
"""
    tst = """@relation leak
@attribute f1 real [0,1]
@attribute class {A,B}
@inputs f1
@outputs class
@data
0.1, A
"""

    _write(d1 / "ok.dat", ok)
    _write(d2 / "leak-2-1tra.dat", tra)
    _write(d2 / "leak-2-1tst.dat", tst)

    res = audit_keel_root(str(tmp_path), out_dir=str(tmp_path / "out"), save=True, top_n_offenders=5)
    assert res["level"] == 3
    assert "executive_summary" in res
    assert len(res["executive_summary"]["top_offenders"]) >= 1
