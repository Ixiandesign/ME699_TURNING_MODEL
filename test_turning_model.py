"""Regression tests against the course workbooks plus basic physics checks."""
import math
from dataclasses import replace

import pytest

import turning_model as tm

TI64 = tm.MATERIALS["ti64"]  # workbooks are Ti-6Al-4V
# Workbook "Deflection Calculator Example": sharp-tool feed-force constants
SHARP = tm.Inputs(**{**TI64, "kf1": 5966.0, "mf": -0.382}, rc_mm=0.8, f_mm=0.2, ap_mm=0.4)


def forces_99(inp: tm.Inputs, ap_mm: float) -> tm.Forces:
    """Force sums on the workbook's 99-interval kappa grid."""
    k_min, k_max = tm.kappa_limits(inp.rc_mm, inp.f_mm, ap_mm)
    return tm.integrate_forces(inp, ap_mm, dkappa=(k_max - k_min) / 99)


def test_kappa_limits_match_workbook():
    k_min, k_max = tm.kappa_limits(0.8, 0.2, 0.4)
    assert k_min == pytest.approx(-0.12508, abs=1e-5)
    assert k_max == pytest.approx(1.0472, abs=1e-4)


def test_chip_thickness_zero_at_kappa_min_and_positive_after():
    k_min, k_max = tm.kappa_limits(0.8, 0.2, 0.4)
    assert tm.uncut_chip_thickness(k_min, 0.8, 0.2) == pytest.approx(0.0, abs=1e-4)
    assert tm.uncut_chip_thickness(k_max, 0.8, 0.2) > 0.1


def test_forces_match_deflection_workbook():
    F = forces_99(SHARP, 0.4)
    assert F.Fc_N == pytest.approx(189.46, rel=1e-3)
    assert F.Fr_N == pytest.approx(74.929, rel=1e-3)
    assert F.Fa_N == pytest.approx(50.086, rel=1e-3)


def test_forces_match_nose_radius_workbook():
    inp = tm.Inputs(**TI64, rc_mm=0.8, f_mm=0.3, ap_mm=0.663)  # homework kc/kf constants
    F = forces_99(inp, 0.663)
    assert F.Fc_N == pytest.approx(425.09, rel=1e-3)
    assert F.Fr_N == pytest.approx(117.85, rel=1e-3)
    assert F.Fa_N == pytest.approx(102.61, rel=1e-3)


def test_step_size_convergence():
    coarse = tm.integrate_forces(SHARP, 0.4, dkappa=0.005)
    fine = tm.integrate_forces(SHARP, 0.4, dkappa=0.0025)
    assert coarse.Fr_N == pytest.approx(fine.Fr_N, rel=0.01)
    assert coarse.Fc_N == pytest.approx(fine.Fc_N, rel=0.01)


def test_cantilever_hand_case():
    # 100 N on a 100 mm steel (210 GPa) bar, I = 1000 mm^4 -> 100*1e6/(3*210e3*1000)
    d = tm.cantilever_deflection(100.0, 100.0, 210e3, 1000.0)
    assert d == pytest.approx(1e8 / 6.3e8)


def test_circular_second_moment():
    assert tm.circular_I(16.0) == pytest.approx(math.pi * 16**4 / 64)


def test_deflection_loop_matches_workbook():
    # Undamped successive substitution, as in the workbook's rev2..rev10 sheets
    one = tm.solve_deflection(SHARP, relax=1.0, max_iter=1)
    assert one.ap_true_mm == pytest.approx(0.11955, rel=0.02)
    ten = tm.solve_deflection(SHARP, relax=1.0, max_iter=10)
    assert ten.ap_true_mm == pytest.approx(0.20229, rel=0.02)


def test_damped_loop_reaches_same_fixed_point():
    res = tm.solve_deflection(SHARP)
    assert res.converged
    assert res.ap_true_mm == pytest.approx(0.2023, rel=0.02)
    # fixed point: ap_true = ap_nom - deflection(ap_true)
    assert res.ap_true_mm + res.deflection_mm == pytest.approx(SHARP.ap_mm, abs=1e-5)


def test_true_ap_never_negative():
    soft = replace(SHARP, k_machine_N_mm=1.0, ap_mm=0.05)
    assert tm.solve_deflection(soft).ap_true_mm >= 0.0


def test_diameters():
    res = tm.solve_deflection(SHARP)
    assert res.d_zero_mm == pytest.approx(SHARP.d_mm - 2 * SHARP.ap_mm)
    assert res.d_final_mm == pytest.approx(SHARP.d_mm - 2 * res.ap_true_mm)
    assert res.d_final_mm > res.d_zero_mm  # deflection leaves material behind


def test_default_material_is_al6061():
    inp = tm.Inputs()
    assert inp.material == "Al6061" and inp.E_GPa == 70.0
    assert tm.MATERIALS["al6061"] == {
        "material": inp.material, "E_GPa": inp.E_GPa, "kc1": inp.kc1, "mc": inp.mc,
        "kf1": inp.kf1, "mf": inp.mf}


def test_al6061_homework_case_is_sane():
    res = tm.solve_deflection(tm.Inputs())
    assert res.converged
    assert 0 < res.ap_true_mm < res.ap_nom_mm
    # softer + lower modulus: beam compliance rises by E_Ti/E_Al vs the Ti workpiece
    ti = tm.compliance_mm_per_N(tm.Inputs(**TI64))[0]
    assert tm.compliance_mm_per_N(tm.Inputs())[0] == pytest.approx(ti * 114 / 70)


def test_optimiser_hits_target_diameter():
    target = 15.5
    ap_nom, res = tm.solve_nominal_ap(SHARP, target)
    assert res.d_final_mm == pytest.approx(target, abs=1e-4)
    assert ap_nom > (SHARP.d_mm - target) / 2  # must over-command to cover deflection
