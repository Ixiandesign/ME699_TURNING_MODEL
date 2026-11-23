"""Regression tests against the course workbooks plus basic physics checks."""
import math
from typing import Any

import pytest

import turning_model as tm

TI64 = tm.MATERIALS["ti64"]  # the workbooks are Ti-6Al-4V
# Workbook "Deflection Calculator Example": sharp-tool feed-force constants
SHARP_P = {**tm.DEFAULTS, "rc": 0.8, "f": 0.2, "ap": 0.4}
SHARP_MAT = {**TI64, "kf1": 5966.0, "mf": -0.382}


def forces_99(p, mat, ap):
    """Force sums on the workbook's 99-interval kappa grid."""
    k_min, k_max = tm.kappa_limits(p["rc"], p["f"], ap)
    return tm.integrate_forces(p["rc"], p["f"], ap, mat, dkappa=(k_max - k_min) / 99)


def test_kappa_limits_match_workbook():
    k_min, k_max = tm.kappa_limits(0.8, 0.2, 0.4)
    assert k_min == pytest.approx(-0.12508, abs=1e-5)
    assert k_max == pytest.approx(1.0472, abs=1e-4)


def test_chip_thickness_zero_at_kappa_min_and_positive_after():
    k_min, k_max = tm.kappa_limits(0.8, 0.2, 0.4)
    assert tm.uncut_chip_thickness(k_min, 0.8, 0.2) == pytest.approx(0.0, abs=1e-4)
    assert tm.uncut_chip_thickness(k_max, 0.8, 0.2) > 0.1


def test_forces_match_deflection_workbook():
    F = forces_99(SHARP_P, SHARP_MAT, 0.4)
    assert F["Fc"] == pytest.approx(189.46, rel=1e-3)
    assert F["Fr"] == pytest.approx(74.929, rel=1e-3)
    assert F["Fa"] == pytest.approx(50.086, rel=1e-3)


def test_forces_match_nose_radius_workbook():
    F = forces_99({**tm.DEFAULTS, "rc": 0.8, "f": 0.3}, TI64, 0.663)  # homework kc/kf constants
    assert F["Fc"] == pytest.approx(425.09, rel=1e-3)
    assert F["Fr"] == pytest.approx(117.85, rel=1e-3)
    assert F["Fa"] == pytest.approx(102.61, rel=1e-3)


def test_step_size_convergence():
    coarse = tm.integrate_forces(0.8, 0.2, 0.4, SHARP_MAT, dkappa=0.005)
    fine = tm.integrate_forces(0.8, 0.2, 0.4, SHARP_MAT, dkappa=0.0025)
    assert coarse["Fr"] == pytest.approx(fine["Fr"], rel=0.01)
    assert coarse["Fc"] == pytest.approx(fine["Fc"], rel=0.01)


def test_cantilever_hand_case():
    # 100 N on a 100 mm steel (210 GPa) bar, I = 1000 mm^4 -> 100*1e6/(3*210e3*1000)
    d = tm.cantilever_deflection(100.0, 100.0, 210e3, 1000.0)
    assert d == pytest.approx(1e8 / 6.3e8)


def test_circular_second_moment():
    assert tm.circular_I(16.0) == pytest.approx(math.pi * 16**4 / 64)


def test_deflection_loop_matches_workbook():
    # Undamped successive substitution, as in the workbook's rev2..rev10 sheets
    one = tm.solve_deflection(SHARP_P, SHARP_MAT, relax=1.0, max_iter=1)
    assert one["ap_true"] == pytest.approx(0.11955, rel=0.02)
    ten = tm.solve_deflection(SHARP_P, SHARP_MAT, relax=1.0, max_iter=10)
    assert ten["ap_true"] == pytest.approx(0.20229, rel=0.02)


def test_damped_loop_reaches_same_fixed_point():
    res = tm.solve_deflection(SHARP_P, SHARP_MAT)
    assert res["converged"]
    assert res["ap_true"] == pytest.approx(0.2023, rel=0.02)
    # fixed point: ap_true = ap_nom - deflection(ap_true)
    assert res["ap_true"] + res["deflection"] == pytest.approx(SHARP_P["ap"], abs=1e-5)


def test_true_ap_never_negative():
    soft = {**SHARP_P, "k_machine": 1.0, "ap": 0.05}
    assert tm.solve_deflection(soft, SHARP_MAT)["ap_true"] >= 0.0


def test_diameters():
    res = tm.solve_deflection(SHARP_P, SHARP_MAT)
    assert res["d_zero"] == pytest.approx(SHARP_P["d"] - 2 * SHARP_P["ap"])
    assert res["d_final"] == pytest.approx(SHARP_P["d"] - 2 * res["ap_true"])
    assert res["d_final"] > res["d_zero"]  # deflection leaves material behind


def test_default_material_is_al7075():
    assert tm.MATERIALS["al7075"]["E_GPa"] == 71.7


def test_al7075_homework_case_is_sane():
    mat = tm.MATERIALS["al7075"]
    res = tm.solve_deflection(tm.DEFAULTS, mat)
    assert res["converged"]
    assert 0 < res["ap_true"] < res["ap_nom"]
    # lower modulus: beam compliance rises by E_Ti/E_Al vs the Ti workpiece
    ti = tm.compliance(tm.DEFAULTS, TI64)[0]
    assert tm.compliance(tm.DEFAULTS, mat)[0] == pytest.approx(ti * 114 / 71.7)


def test_optimiser_hits_target_diameter():
    target = 15.5
    ap_nom, res = tm.solve_nominal_ap(SHARP_P, SHARP_MAT, target)
    assert res["d_final"] == pytest.approx(target, abs=1e-4)
    assert ap_nom > (SHARP_P["d"] - target) / 2  # must over-command to cover deflection


# ------------------------------------------------------ Part 3 additions
AL = tm.MATERIALS["al7075"]
P3 = tm.PART3
PLAN: dict[str, Any] = dict(f_rough=0.1, f_finish=0.05, rpm=2000.0)


def test_stepped_beam_limits():
    E, L = 71_700.0, 100.0
    uniform = 1 / (3 * E * tm.circular_I(8.0)) * L**3
    assert tm.compliance({**P3, "L": L, "d": 8.0, "d_bar": 8.0}, AL)[0] == pytest.approx(uniform)
    # turned section as long as the whole overhang -> uniform bar of the turned diameter
    assert tm.compliance({**P3, "L": L, "d": 8.0, "d_bar": 9.5, "Ls": L}, AL)[0] == pytest.approx(uniform)
    # a short thin tip on a fat bar is stiffer than a thin bar all the way
    short = tm.compliance({**P3, "L": L, "d": 8.0, "d_bar": 9.5, "Ls": 10.0}, AL)[0]
    assert uniform > short > L**3 / (3 * E * tm.circular_I(9.5))


def test_edge_radius_cutoff_and_ploughing():
    base = tm.integrate_forces(0.8, 0.05, 0.05, AL)
    cut = tm.integrate_forces(0.8, 0.05, 0.05, AL, re=0.01)         # h < h_min elements do not cut
    assert 0.9 * base["Fc"] < cut["Fc"] <= base["Fc"]
    plough = tm.integrate_forces(0.8, 0.05, 0.05, AL, re=0.01, pl=0.5)
    assert plough["Fr"] > cut["Fr"]
    assert tm.integrate_forces(0.8, 0.05, 0.05, AL, re=0.01, pl=0.0)["Fr"] == pytest.approx(cut["Fr"])


def test_kinematic_ra_matches_workbook():
    assert tm.kinematic_ra(0.3, 0.8) == pytest.approx(4.6875)       # Nose Radius workbook C15


def test_flash_temperature_limits():
    kw: dict[str, Any] = dict(mat=AL, b=0.24, w=0.1, beta=0.2)
    lo = tm.flash_temperature(Fc=10.0, vc=60.0, **kw)
    assert lo["dT"] > 0 and lo["Pe"] > 0
    assert tm.flash_temperature(Fc=20.0, vc=60.0, **kw)["dT"] == pytest.approx(2 * lo["dT"], rel=1e-6)
    # fast source: Jaeger asymptote dT = 2 q l / (K sqrt(pi L))
    hi = tm.flash_temperature(Fc=10.0, vc=60000.0, **kw)
    q, l, K = hi["q"], 0.05e-3, AL["k_th"]
    assert hi["dT"] == pytest.approx(2 * q * l / (K * math.sqrt(math.pi * hi["L"])), rel=0.03)
    # slow source: stationary band, finite and larger than the fast-source scaling would give
    assert math.isfinite(tm.flash_temperature(Fc=10.0, vc=0.01, **kw)["dT"])


def test_plan_meets_tolerance_without_undersize():
    plan = tm.plan_passes(P3, AL, **PLAN)
    s = plan["summary"]
    assert s["in_tolerance"]
    assert abs(s["d_final"] - P3["d_target"]) <= P3["tol"]
    diam = [r["d_after"] for r in plan["passes"]]
    assert all(a >= b for a, b in zip(diam, diam[1:]))              # monotone: material only leaves
    assert min(diam) >= P3["d_target"] - 1e-9                       # set point at target -> never undersize
    assert s["time_s"] > 0 and s["n_passes"] == len(plan["passes"])


def test_plan_fixed_pass_count_and_sensitivity():
    nom = tm.plan_passes(P3, AL, **PLAN)
    n = nom["summary"]["n_passes"]
    assert tm.plan_passes(P3, AL, n_fixed=n - 2, **PLAN)["summary"]["n_passes"] == n - 2
    hi = tm.plan_passes(P3, {**AL, "kc1": AL["kc1"] * 1.2, "kf1": AL["kf1"] * 1.2}, n_fixed=n, **PLAN)
    assert hi["summary"]["d_final"] >= nom["summary"]["d_final"]    # more force -> more material left
    rows = tm.sensitivity(P3, AL, rel=0.2, **PLAN)
    assert {r["case"] for r in rows} >= {"force +20%", "force -20%"}


def test_calibration_recovers_force_scale():
    truth = {**AL, "kc1": AL["kc1"] * 1.25, "kf1": AL["kf1"] * 0.8}
    data = []
    for f, ap in [(0.05, 0.05), (0.1, 0.1), (0.15, 0.2)]:
        F = tm._forces({**P3, "f": f}, ap, truth)
        data.append(dict(f=f, ap=ap, Fc=F["Fc"], Fr=F["Fr"]))
    fit = tm.calibrate_forces(P3, AL, data)
    assert fit["kc1"] == pytest.approx(truth["kc1"], rel=1e-6)
    assert fit["kf1"] == pytest.approx(truth["kf1"], rel=1e-6)
    err = tm.compare(dict(Fc=110.0), dict(Fc=100.0))
    assert err["Fc"] == pytest.approx(10.0)


def test_strategy_search_respects_constraints():
    best = tm.search_strategy(P3, AL)
    s = best["summary"]
    assert s["in_tolerance"] and P3["ra_band"][0] <= s["Ra"] <= P3["ra_band"][1]
    assert s["T_max"] < tm.T_CRIT_ALUMINIUM


def test_solver_converges_with_edge_radius_cutoff():
    # hard h_min cutoff makes F(ap) jumpy; the fixed-point loop alone cycles here
    for f, ap in [(0.1, 0.01), (0.05, 0.7938)]:
        res = tm.solve_deflection({**P3, "f": f, "ap": ap}, AL)
        assert res["converged"]
        assert abs(res["ap_true"] + res["deflection"] - ap) < 2e-3   # residual below 2 um


# ------------------------------------------------------------------- UI helpers
def test_force_profile_matches_workbook_table():
    k_min, k_max = tm.kappa_limits(0.8, 0.3, 0.663)
    prof = tm.force_profile(0.8, 0.3, 0.663, TI64, dkappa=(k_max - k_min) / 99)
    assert len(prof["kappa"]) == 100                                  # workbook rows 35..134
    assert prof["h"][0] == pytest.approx(2.6643e-4, rel=1e-3)         # C35
    assert prof["dFc"][0] == pytest.approx(0.019374, rel=1e-3)        # D35
    assert prof["dFf"][1] == pytest.approx(0.448837, rel=1e-3)        # E36
    assert prof["Fc_cum"][-1] == pytest.approx(425.09, rel=1e-3)      # H134
    assert prof["Fr_cum"][-1] == pytest.approx(117.85, rel=1e-3)      # I134
    assert prof["Fa_cum"][-1] == pytest.approx(102.61, rel=1e-3)      # J134
    tot = tm.integrate_forces(0.8, 0.3, 0.663, TI64, dkappa=(k_max - k_min) / 99)
    assert prof["Fc_cum"][-1] == pytest.approx(tot["Fc"]) and prof["Fr_cum"][-1] == pytest.approx(tot["Fr"])


def test_ui_fixed_setup_not_editable():
    import turning_ui as ui
    keys = {k for k, _, _ in ui.SINGLE_FIELDS} | {k for k, _, _ in ui.PART3_FIELDS}
    fixed = {"d", "d_bar", "L", "Ls", "rc", "re", "d_target", "tol", "ra_lo", "ra_hi",
             "E_GPa", "kc1", "mc", "kf1", "mf", "sigma_y", "rho", "cp", "k_th"}
    assert not keys & fixed                                              # exam geometry and 7075 are fixed
    assert ui.MAT is tm.MATERIALS["al7075"]
    assert "9.525" in ui.fixed_setup() and "7.9375" in ui.fixed_setup() and "flank 7" in ui.fixed_setup()


def test_ui_parse_single():
    import turning_ui as ui
    vals = {k: d for k, _, d in ui.SINGLE_FIELDS}
    p, target = ui.parse_single(vals)
    assert p["rc"] == P3["rc"] and p["re"] == P3["re"] and p["L"] == P3["L"] and p["Ls"] == P3["Ls"]
    assert p["vc"] == pytest.approx(math.pi * P3["d"] * 2000 / 1000) and target is None   # rpm -> vc
    assert ui.parse_single({**vals, "target": "9.2"})[1] == 9.2
    with pytest.raises(ValueError, match="Feed"):
        ui.parse_single({**vals, "f": "abc"})


def test_ui_run_single_matches_model():
    import turning_ui as ui
    vals = {k: d for k, _, d in ui.SINGLE_FIELDS}
    p, target = ui.parse_single(vals)
    out = ui.run_single(p, AL, target)
    assert out["res"]["ap_true"] == pytest.approx(tm.solve_deflection(p, AL)["ap_true"])
    assert "true ap" in out["text"]
    out_t = ui.run_single(p, AL, 9.2)                                 # target diameter -> solves nominal ap
    assert out_t["res"]["d_final"] == pytest.approx(9.2, abs=1e-4) and "command nominal ap" in out_t["text"]
    with pytest.raises(ValueError):
        ui.run_single({**p, "ap": 5.0}, AL, None)                     # ap > 2*rc


def test_ui_parse_and_run_part3_fixed():
    import turning_ui as ui
    vals = {k: d for k, _, d in ui.PART3_FIELDS}
    p, cfg = ui.parse_part3({**vals, "mode": "fixed", "f_rough": "0.1", "f_finish": "0.05", "rpm": "2000"})
    assert p["d_target"] == P3["d_target"] and p["ra_band"] == P3["ra_band"]
    assert cfg["mode"] == "fixed" and cfg["rpm"] == 2000.0
    _, cfg_s = ui.parse_part3({**vals, "mode": "search", "feeds": "0.05, 0.1", "rpms": "2000,3000"})
    assert cfg_s["feeds"] == (0.05, 0.1) and cfg_s["rpms"] == (2000.0, 3000.0)
    out = ui.run_part3(p, AL, cfg)
    ref = tm.plan_passes(p, AL, 0.1, 0.05, 2000.0)
    assert out["plan"]["summary"]["d_final"] == pytest.approx(ref["summary"]["d_final"])
    assert "sensitivity" in out["text"] and "final diameter" in out["text"]


def test_ui_plots_draw_expected_axes():
    from matplotlib.figure import Figure

    import turning_ui as ui
    p = {**P3, "f": 0.05, "ap": 0.25}
    fig = Figure()
    ui.plot_single(fig, p, AL, tm.solve_deflection(p, AL))
    assert len(fig.axes) == 6
    assert all(ax.has_data() for ax in fig.axes)
    plan = tm.plan_passes(P3, AL, 0.1, 0.05, 2000.0)
    fig2 = Figure()
    ui.plot_plan(fig2, P3, AL, plan)
    assert len(fig2.axes) == 6
    assert all(ax.has_data() for ax in fig2.axes)


def test_ui_ignores_stale_tcl_environment(monkeypatch):
    import os

    import turning_ui as ui
    monkeypatch.setenv("TCL_LIBRARY", r"C:\nowhere\other-python\tcl8.6")
    monkeypatch.setenv("TK_LIBRARY", r"C:\nowhere\other-python\tk8.6")
    ui.use_own_tcl()
    assert os.path.isfile(os.path.join(os.environ["TCL_LIBRARY"], "init.tcl"))
    assert os.path.isdir(os.environ["TK_LIBRARY"])


def test_ui_prefers_venv_tcl_over_base(monkeypatch, tmp_path):
    import os
    import sys

    import turning_ui as ui
    for name in ("tcl8.6", "tk8.6"):
        (tmp_path / "tcl" / name).mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))               # a venv with its own Tcl/Tk copy
    ui.use_own_tcl()
    assert os.environ["TCL_LIBRARY"] == str(tmp_path / "tcl" / "tcl8.6")
    assert os.environ["TK_LIBRARY"] == str(tmp_path / "tcl" / "tk8.6")
