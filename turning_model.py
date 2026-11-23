"""Finish-turning model (ME699 homework Parts 1-2, ME599/699 final exam Part 3).

Part 1: cantilever beam deflection, point load at the free end.
Part 2: finish-turning forces with a nose-radius tool, the resulting
workpiece + machine deflection, and the true vs nominal (radial) depth of cut.
Part 3: multi-pass strategy to a target diameter (spring passes), surface finish,
flash temperature, cycle time, calibration / sensitivity helpers.

Units: mm, N, N/mm^2 (= MPa), rad. Only the kc/kf power laws use micrometres
(h_um = 1000 * h_mm). Inputs are two plain dicts:
  p   process: rc, f, ap, d, L, k_machine, vc, re, pl (+ optional d_bar, Ls)  (see DEFAULTS, PART3)
  mat material: name, E_GPa, kc1, mc, kf1, mf, sigma_y, rho, cp, k_th (see MATERIALS)

Usage: uv run turning_model.py [--material ti64 --rc 0.8 --f 0.18 --ap 0.25 --target 15.5 --plot]
       uv run turning_model.py --part3 [--f-rough 0.1 --f-finish 0.05 --rpm 2000]
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

DKAPPA = 0.005   # target kappa step (rad)

DEFAULTS: dict[str, Any] = dict(
    rc=0.8,           # tool nose radius
    f=0.18,           # feed per revolution (axial)
    ap=0.25,          # nominal depth of cut (radial)
    d=16.0,           # workpiece diameter before the pass
    L=160.0,          # overhang length (L/D = 10)
    k_machine=50_000.0,  # machine / tool-holder stiffness (N/mm)
    vc=60.0,          # cutting speed (m/min)
    re=0.0,           # cutting-edge radius (mm); 0 = ideally sharp edge
    pl=0.0,           # ploughing scale (0 = off, uncalibrated; see integrate_forces)
)

# Final exam Part 3: 3/8 in AA7075 bar, 1/4 in section turned at the free end to 5/16 in.
PART3: dict[str, Any] = {
    **DEFAULTS,
    "d": 9.525, "d_bar": 9.525,     # start diameter; un-turned bar diameter upstream
    "L": 101.6,                     # overhang = whole 4 in bar (conservative: worst-case stiffness)
    "Ls": 6.35,                     # length of the turned section at the free end
    "rc": 0.8, "re": 0.010,         # tool nose radius, edge radius (exam statement)
    "d_target": 7.9375, "tol": 0.0127,   # final diameter and +/- tolerance on it
    "ra_band": (0.1, 0.4),          # required Ra range (um)
    "beta": 0.2,                    # fraction of cutting power entering the workpiece (placeholder)
    "w_contact": 0.1,               # effective thermal contact width (mm) (placeholder)
}

# Lathe settings available for Part 3 (placeholders until the course list is supplied).
LATHE: dict[str, tuple[float, ...]] = dict(
    rpm=(500.0, 1000.0, 1500.0, 2000.0, 3000.0),
    feed=(0.03, 0.05, 0.075, 0.1, 0.15),
)

# No reliable critical temperature is available for AA7075: this is the lower edge of the
# artificial-aging / over-aging range of 7075-T6 (literature placeholder, indicative only).
T_CRIT_ALUMINIUM = 150.0   # deg C
T_SOLIDUS_7075 = 477.0     # deg C, upper bound (melting begins)

# kc = kc1 * h_um**mc, kf = kf1 * h_um**mf (N/mm^2, h in um)
MATERIALS: dict[str, dict[str, Any]] = {
    # kc/kf: power-law fit of the course 7075 Fc/Ff data (Cutting Force Data.xlsx, 3 mm width).
    # Elastic/thermal values are literature placeholders for 7075-T6.
    "al7075": dict(name="AA7075-T6", E_GPa=71.7, kc1=2534.0, mc=-0.149, kf1=2126.0, mf=-0.168,
                   sigma_y=503.0, rho=2810.0, cp=960.0, k_th=130.0),   # MPa, kg/m^3, J/kg/K, W/m/K
    # Homework / Nose Radius workbook constants
    "ti64": dict(name="Ti-6Al-4V", E_GPa=114.0, kc1=4500.0, mc=-0.175, kf1=18250.0, mf=-0.611,
                 sigma_y=880.0, rho=4430.0, cp=526.0, k_th=6.7),
}


# ---------------------------------------------------------------- Part 1: beam

def circular_I(d: float) -> float:
    """Second moment of area of a solid round bar (mm^4)."""
    return math.pi * d**4 / 64


def cantilever_deflection(F: float, L: float, E: float, I: float) -> float:
    """Tip deflection (mm) of a cantilever with a point load F at the free end (E in N/mm^2)."""
    return F * L**3 / (3 * E * I)


# ------------------------------------------------------------ Part 2: geometry

def kappa_limits(rc: float, f: float, ap: float) -> tuple[float, float]:
    """Engaged nose-arc limits (rad): kappa_min from the feed, kappa_max from ap."""
    if not 0 < ap <= 2 * rc:
        raise ValueError("ap must be in (0, 2*rc]")
    k_min = -math.acos((rc - f**2 / (8 * rc)) / rc)
    k_max = math.acos((rc - ap) / rc)
    return k_min, k_max


def uncut_chip_thickness(kappa, rc: float, f: float):
    """Uncut chip thickness h(kappa) in mm (homework formula)."""
    c = np.cos(kappa - np.pi / 2)
    return rc + f * c - np.sqrt(rc**2 + f**2 * (c**2 - 1))


def force_profile(rc: float, f: float, ap: float, mat: dict[str, Any],
                  dkappa: float = DKAPPA, re: float = 0.0, pl: float = 0.0,
                  hmin_ratio: float = 0.3) -> dict[str, np.ndarray]:
    """Per-element table over the engaged arc, like the workbook's rows 35-134.

    kappa (rad), h (mm), dFc, dFf (N, normal to the edge), dFr, dFa (radial / axial parts of dFf)
    and the running sums Fc_cum, Fr_cum, Fa_cum. dF = k(h) * h * ds with ds = rc * dkappa.

    Edge radius re (mm): elements thinner than h_min = hmin_ratio*re form no chip and add no
    cutting force. Ploughing (pl > 0): every engaged element also presses the edge into the
    surface with pl * sigma_y * re * ds, normal to the cutting edge (pl = calibration knob).
    """
    k_min, k_max = kappa_limits(rc, f, ap)
    n = max(1, round((k_max - k_min) / dkappa))
    kappa = np.linspace(k_min, k_max, n + 1)      # grid includes both end points
    ds = rc * (k_max - k_min) / n                 # arc length per element (mm)
    h = np.maximum(uncut_chip_thickness(kappa, rc, f), 1e-9)   # mm
    h_um = h * 1000                               # kc/kf are defined for h in um
    cuts = h >= hmin_ratio * re                   # sub-h_min elements plough instead of cutting
    dFc = np.where(cuts, mat["kc1"] * h_um ** mat["mc"] * h * ds, 0.0)
    dFf = np.where(cuts, mat["kf1"] * h_um ** mat["mf"] * h * ds, 0.0)   # normal to the cutting edge
    dFf = dFf + pl * mat.get("sigma_y", 0.0) * re * ds                   # ploughing (off by default)
    dFr, dFa = np.cos(kappa) * dFf, np.sin(kappa) * dFf
    return dict(kappa=kappa, h=h, dFc=dFc, dFf=dFf, dFr=dFr, dFa=dFa,
                Fc_cum=np.cumsum(dFc), Fr_cum=np.cumsum(dFr), Fa_cum=np.cumsum(dFa))


def integrate_forces(rc: float, f: float, ap: float, mat: dict[str, Any],
                     dkappa: float = DKAPPA, re: float = 0.0, pl: float = 0.0,
                     hmin_ratio: float = 0.3) -> dict[str, float]:
    """Total force over the engaged arc (the end of force_profile's running sums).

    Returns Fc (tangential), Fr (radial), Fa (axial) in N and mean chip thickness h_avg in mm.
    """
    if ap <= 1e-9:
        return dict(Fc=0.0, Fr=0.0, Fa=0.0, h_avg=0.0)
    prof = force_profile(rc, f, ap, mat, dkappa, re, pl, hmin_ratio)
    return dict(Fc=float(prof["Fc_cum"][-1]), Fr=float(prof["Fr_cum"][-1]),
                Fa=float(prof["Fa_cum"][-1]), h_avg=float(prof["h"].mean()))


# ----------------------------------------------- Part 2: deflection + iteration

def compliance(p: dict[str, Any], mat: dict[str, Any]) -> tuple[float, float]:
    """(beam, machine) radial compliance at the tool tip, mm/N.

    Uniform bar of diameter d, unless p has Ls: then a stepped cantilever whose last Ls mm
    (the turned section, diameter d) hang off an un-turned bar of diameter d_bar. The load sits
    at the free end, the worst case, since the cut starts there.
    """
    E = mat["E_GPa"] * 1000
    if p.get("Ls"):
        L, Ls = p["L"], p["Ls"]
        beam = ((L**3 - Ls**3) / circular_I(p.get("d_bar", p["d"])) + Ls**3 / circular_I(p["d"])) / (3 * E)
    else:
        beam = cantilever_deflection(1.0, p["L"], E, circular_I(p["d"]))
    return beam, 1.0 / p["k_machine"]


def _forces(p: dict[str, Any], ap: float, mat: dict[str, Any]) -> dict[str, float]:
    """integrate_forces with the edge-radius / ploughing settings taken from p."""
    return integrate_forces(p["rc"], p["f"], ap, mat, re=p.get("re", 0.0), pl=p.get("pl", 0.0))


def solve_deflection(p: dict[str, Any], mat: dict[str, Any], relax: float = 0.5,
                     max_iter: int = 200, tol: float = 1e-6) -> dict[str, Any]:
    """Iterate ap_true = ap_nom - deflection(F_radial(ap_true)).

    relax=1 is the workbook's plain successive substitution (it oscillates
    towards the answer); 0.5 damps that and reaches the same fixed point.
    """
    c_beam, c_mach = compliance(p, mat)
    ap, history, converged = p["ap"], [], False
    for _ in range(max_iter):
        defl = _forces(p, ap, mat)["Fr"] * (c_beam + c_mach)
        history.append(defl)
        new = ap + relax * (max(p["ap"] - defl, 0.0) - ap)
        converged = abs(new - ap) < tol
        ap = new
        if converged:
            break
    if not converged and relax < 1:    # relax=1 stays the plain workbook iteration, as asked
        # The edge-radius cutoff and the discrete arc grid make F(ap) jump slightly, so the plain
        # iteration can cycle. ap + deflection(ap) rises with ap: bisect for ap + deflection = ap_nom.
        lo, hi = 0.0, p["ap"]
        for _ in range(60):
            ap = (lo + hi) / 2
            if ap + _forces(p, ap, mat)["Fr"] * (c_beam + c_mach) > p["ap"]:
                hi = ap
            else:
                lo = ap
        ap, converged = (lo + hi) / 2, True
    F = _forces(p, ap, mat)
    beam, machine = F["Fr"] * c_beam, F["Fr"] * c_mach
    return dict(F, ap_nom=p["ap"], ap_true=ap, beam=beam, machine=machine,
                deflection=beam + machine, history=history, converged=converged,
                d_zero=p["d"] - 2 * p["ap"],     # ap is radial -> diameter drops by 2*ap
                d_final=p["d"] - 2 * ap)


def solve_nominal_ap(p: dict[str, Any], mat: dict[str, Any], d_target: float,
                     tol: float = 1e-5, max_iter: int = 200) -> tuple[float, dict[str, Any]]:
    """Nominal ap to command so the deflected cut lands on d_target (replaces Goal Seek)."""
    ap_target = (p["d"] - d_target) / 2
    ap_nom = ap_target
    res = solve_deflection({**p, "ap": ap_nom}, mat)
    for _ in range(max_iter):
        err = ap_target - res["ap_true"]
        if abs(err) < tol:
            break
        ap_nom += err        # add back what the tool failed to remove
        res = solve_deflection({**p, "ap": ap_nom}, mat)
    return ap_nom, res


# --------------------------------- Part 3: roughness, heat, passes, validation

def kinematic_ra(f: float, rc: float) -> float:
    """Kinematic Ra (um), f and rc in mm, with the workbook's 4/3 factor."""
    return f**2 / (32 * rc) * 1000 * 4 / 3


def _band_integral(Z: float) -> float:
    """int_0^Z exp(z) K0(z) dz, using K0(z) = int_0^inf exp(-z cosh t) dt (no Bessel library)."""
    dt = 0.001
    t = (np.arange(40_000) + 0.5) * dt            # midpoint rule on t in (0, 40)
    c = np.cosh(t) - 1
    return float(np.sum(-np.expm1(-Z * c) / c) * dt)


def flash_temperature(Fc: float, vc: float, mat: dict[str, Any], b: float, w: float,
                      beta: float = 0.2, T0: float = 20.0) -> dict[str, float]:
    """Peak surface temperature of the workpiece under a moving band heat source (Jaeger).

    Heat input Q = beta * Fc * vc enters the workpiece over a band of width w (mm, along the
    cutting direction) and length b (mm, engaged edge). Fc in N, vc in m/min.
    Pe = V*w/alpha; the Jaeger parameter is L = V*(w/2)/(2*alpha) = Pe/4. The maximum
    is at the trailing edge of the band: dT = q/(pi*K) * (2*alpha/V) * int_0^2L e^z K0(z) dz,
    which tends to 2*q*l/(K*sqrt(pi*L)) for fast sources and to the stationary band for slow ones.
    """
    V = max(vc, 1e-9) / 60                              # m/s
    K = mat["k_th"]
    alpha = K / (mat["rho"] * mat["cp"])                # m^2/s
    q = beta * Fc * V / (b * w * 1e-6)                  # W/m^2 into the workpiece
    L = V * (w / 2 * 1e-3) / (2 * alpha)
    dT = q / (math.pi * K) * (2 * alpha / V) * _band_integral(2 * L)
    return dict(q=q, L=L, Pe=V * w * 1e-3 / alpha, dT=dT, T=T0 + dT,
                depth=2e3 * math.sqrt(alpha * w * 1e-3 / V))   # thermal penetration depth (mm)


def plan_passes(p: dict[str, Any], mat: dict[str, Any], f_rough: float, f_finish: float,
                rpm: float, allow: float = 0.05, margin: float = 0.5, max_passes: int = 60,
                n_fixed: int | None = None) -> dict[str, Any]:
    """Turn from p["d"] to p["d_target"] by repeated passes at a fixed set point (the target).

    Every pass is commanded to the target diameter, so the tool can never cut undersize: deflection
    only leaves stock, which the next pass (a "spring pass") removes, shrinking the error each time
    (the steady-state balance of cutting force and deflection, applied pass after pass). Passes
    use f_rough while the radial stock left exceeds `allow` mm, then f_finish. Stops once
    |d - target| <= margin * tol (margin < 1 leaves room for model error), or after n_fixed
    passes if given (used to test a plan against a perturbed model).
    """
    d_target, tol = p["d_target"], p["tol"]
    L_cut = p.get("Lcut", p.get("Ls", 0.0))              # length turned per pass (mm)
    d, rows, stalled = p["d"], [], False
    while len(rows) < (n_fixed if n_fixed is not None else max_passes):
        err = d - d_target
        if err <= 1e-9 or (n_fixed is None and err <= margin * tol):
            break
        rough = err / 2 > allow
        f = f_rough if rough else f_finish
        res = solve_deflection({**p, "d": d, "f": f, "ap": err / 2}, mat)
        vc = math.pi * d * rpm / 1000                    # m/min, at the diameter being cut
        ap = res["ap_true"]
        if ap > 1e-9:
            k_min, k_max = kappa_limits(p["rc"], f, ap)
            heat = flash_temperature(res["Fc"], vc, mat, p["rc"] * (k_max - k_min),
                                     p["w_contact"], p["beta"])
        else:
            heat = dict(T=20.0, Pe=0.0)
        rows.append(dict(n=len(rows) + 1, kind="rough" if rough else "finish", f=f, rpm=rpm, vc=vc,
                         d_before=d, ap_nom=err / 2, ap_true=ap, Fc=res["Fc"], Fr=res["Fr"],
                         deflection=res["deflection"], d_after=res["d_final"],
                         Ra=kinematic_ra(f, p["rc"]), T=heat["T"], Pe=heat["Pe"],
                         t_s=L_cut / (f * rpm) * 60, mrr=1000 * vc * f * ap))
        if ap <= 1e-7:                                   # nothing left that the tool removes
            stalled = True
            break
        d = res["d_final"]
    f_last = rows[-1]["f"] if rows else f_finish
    t_s = sum(r["t_s"] for r in rows)
    volume = math.pi / 4 * (p["d"] ** 2 - d**2) * L_cut   # removed volume (mm^3)
    summary = dict(n_passes=len(rows), d_final=d, error=d - d_target, stalled=stalled,
                   in_tolerance=abs(d - d_target) <= tol, time_s=t_s, volume=volume,
                   productivity=volume / (t_s / 60) if t_s else 0.0,   # mm^3/min
                   Ra=kinematic_ra(f_last, p["rc"]), T_max=max((r["T"] for r in rows), default=20.0))
    return dict(passes=rows, summary=summary)


def search_strategy(p: dict[str, Any], mat: dict[str, Any], rpms: tuple[float, ...] = LATHE["rpm"],
                    feeds: tuple[float, ...] = LATHE["feed"], **plan_kw: Any) -> dict[str, Any]:
    """Fastest plan over the lathe settings that meets tolerance, Ra band and temperature limit."""
    best: dict[str, Any] | None = None
    lo, hi = p["ra_band"]
    for rpm in rpms:
        for f_finish in feeds:
            for f_rough in (x for x in feeds if x >= f_finish):
                plan = plan_passes(p, mat, f_rough, f_finish, rpm, **plan_kw)
                s = plan["summary"]
                if (s["in_tolerance"] and not s["stalled"] and lo <= s["Ra"] <= hi
                        and s["T_max"] < T_CRIT_ALUMINIUM
                        and (best is None or s["time_s"] < best["summary"]["time_s"])):
                    best = {**plan, "config": dict(f_rough=f_rough, f_finish=f_finish, rpm=rpm)}
    if best is None:
        raise ValueError("no lathe setting meets tolerance, Ra band and temperature limit")
    return best


def calibrate_forces(p: dict[str, Any], mat: dict[str, Any], data: list[dict[str, float]]) -> dict[str, Any]:
    """Least-squares scale of kc1 (on Fc) and kf1 (on Fradial) to measured forces.

    data rows: dict(f=feed, ap=depth, Fc=measured, Fr=measured). Fits Fc ~ s_c*Fc_model,
    Fr ~ s_f*Fr_model (exact for pl = 0, where force is linear in kc1/kf1).
    """
    pred = [_forces({**p, "f": r["f"]}, r["ap"], mat) for r in data]
    sc = sum(r["Fc"] * m["Fc"] for r, m in zip(data, pred)) / sum(m["Fc"] ** 2 for m in pred)
    sf = sum(r["Fr"] * m["Fr"] for r, m in zip(data, pred)) / sum(m["Fr"] ** 2 for m in pred)
    return {**mat, "kc1": mat["kc1"] * sc, "kf1": mat["kf1"] * sf}


def compare(pred: dict[str, float], meas: dict[str, float]) -> dict[str, float]:
    """Percent error of each predicted quantity against its measurement."""
    return {k: 100 * (pred[k] - meas[k]) / meas[k] for k in meas if k in pred}


def sensitivity(p: dict[str, Any], mat: dict[str, Any], rel: float = 0.2, **plan_kw: Any) -> list[dict[str, Any]]:
    """Run the nominal plan, then replay its pass count on a model with one input changed by +/-rel.

    err_um is the final diameter error of that replay; n_replan is the pass count a re-plan needs.
    """
    n = plan_passes(p, mat, **plan_kw)["summary"]["n_passes"]
    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = [("nominal", p, mat)]
    for sign in (+1, -1):
        s = 1 + sign * rel
        tag = f"{sign * rel:+.0%}"
        cases += [(f"force {tag}", p, {**mat, "kc1": mat["kc1"] * s, "kf1": mat["kf1"] * s}),
                  (f"E {tag}", p, {**mat, "E_GPa": mat["E_GPa"] * s}),
                  (f"machine stiffness {tag}", {**p, "k_machine": p["k_machine"] * s}, mat)]
    rows = []
    for name, pp, mm in cases:
        fixed = plan_passes(pp, mm, n_fixed=n, **plan_kw)["summary"]
        replan = plan_passes(pp, mm, **plan_kw)["summary"]
        rows.append(dict(case=name, d_final=fixed["d_final"], err_um=1000 * fixed["error"],
                         in_tolerance=fixed["in_tolerance"], n_replan=replan["n_passes"]))
    return rows


# --------------------------------------------------------------------- report

def report(p: dict[str, Any], mat: dict[str, Any], res: dict[str, Any]) -> str:
    rpm = 1000 * p["vc"] / (math.pi * p["d"])
    torque = res["Fc"] * p["d"] / 2 / 1000                      # N*m
    power = torque * 2 * math.pi * rpm / 60                     # W
    Ra = kinematic_ra(p["f"], p["rc"])                          # um
    k_min, k_max = kappa_limits(p["rc"], p["f"], res["ap_true"])
    lines = [
        f"material {mat['name']} (E={mat['E_GPa']:g} GPa)",
        f"rc={p['rc']} mm  f={p['f']} mm/rev  d={p['d']} mm  L={p['L']} mm (L/D={p['L'] / p['d']:.1f})",
        f"nominal ap        {res['ap_nom']:.4f} mm",
        f"true ap           {res['ap_true']:.4f} mm  ({res['ap_true'] / res['ap_nom']:.0%} of nominal)"
        + ("" if res["converged"] else "  [NOT CONVERGED]"),
        f"deflection        {res['deflection'] * 1000:.1f} um  (beam {res['beam'] * 1000:.1f} + machine {res['machine'] * 1000:.1f})",
        f"zero-defl. dia.   {res['d_zero']:.4f} mm",
        f"predicted dia.    {res['d_final']:.4f} mm  (oversize {1000 * (res['d_final'] - res['d_zero']):.1f} um)",
        f"forces            Fc {res['Fc']:.1f} N   Fradial {res['Fr']:.1f} N   Faxial {res['Fa']:.1f} N",
        f"engaged arc       {math.degrees(k_min):.2f} to {math.degrees(k_max):.2f} deg, h_avg {res['h_avg'] * 1000:.1f} um",
        f"rpm {rpm:.0f}  torque {torque:.2f} N*m  power {power:.0f} W",
        f"kinematic Ra {Ra:.2f} um",
    ]
    if mat["name"] == "Ti-6Al-4V":   # Sandvik constants are Ti-6Al-4V only
        lines.append(f"Sandvik Fc check {2300 * res['h_avg']**-0.15 * p['f'] * res['ap_true']:.1f} N")
    if res["deflection"] > 0.1 * res["h_avg"]:
        lines.append("WARNING: deflection > 10% of average chip thickness (chatter risk)")
    if not 0.1 * p["rc"] <= p["ap"] <= p["rc"] / 3:
        lines.append("note: ap outside the 1/10-1/3 of rc finishing guideline")
    return "\n".join(lines)


def plot(p: dict[str, Any], mat: dict[str, Any], res: dict[str, Any]) -> None:
    from turning_ui import use_own_tcl     # ignore a stale machine-wide TCL_LIBRARY before the Tk backend loads
    use_own_tcl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    k_min, k_max = kappa_limits(p["rc"], p["f"], res["ap_true"])
    kappa = np.linspace(k_min, k_max, 200)
    ax[0].plot(np.degrees(kappa), uncut_chip_thickness(kappa, p["rc"], p["f"]) * 1000)
    ax[0].set(xlabel="kappa (deg)", ylabel="uncut chip thickness (um)", title="Chip thickness over engaged arc")

    ax[1].plot(range(1, len(res["history"]) + 1), np.array(res["history"]) * 1000, "o-")
    ax[1].set(xlabel="iteration", ylabel="deflection (um)", title="Deflection convergence")

    noms = np.linspace(0.05, 0.5, 30)
    trues = [solve_deflection({**p, "ap": a}, mat)["ap_true"] for a in noms]
    ax[2].plot(noms, trues, label="predicted")
    ax[2].plot(noms, noms, "--", label="no deflection")
    ax[2].set(xlabel="nominal ap (mm)", ylabel="true ap (mm)", title="True vs nominal depth of cut")
    ax[2].legend()
    fig.tight_layout()
    plt.show()


def report_plan(p: dict[str, Any], mat: dict[str, Any], plan: dict[str, Any]) -> str:
    """Pass table plus summary for plan_passes / search_strategy output."""
    s = plan["summary"]
    lines = [f"material {mat['name']}  {p['d']} -> {p['d_target']} mm +/-{p['tol']} mm, "
             f"overhang {p['L']} mm, turned length {p.get('Ls', 0)} mm, re {1000 * p['re']:.0f} um",
             "pass kind    f    rpm  vc   ap_nom ap_true   Fc    Fr   defl   d_after   Ra    T    t",
             "     (mm/rev)       (m/min) (um)   (um)     (N)   (N)   (um)    (mm)    (um) (C)  (s)"]
    for r in plan["passes"]:
        lines.append(f"{r['n']:>3}  {r['kind']:<6}{r['f']:>5.3f}{r['rpm']:>6.0f}{r['vc']:>6.1f}"
                     f"{1000 * r['ap_nom']:>8.1f}{1000 * r['ap_true']:>7.1f}{r['Fc']:>7.2f}{r['Fr']:>6.2f}"
                     f"{1000 * r['deflection']:>7.1f}{r['d_after']:>10.4f}{r['Ra']:>6.2f}{r['T']:>6.1f}{r['t_s']:>6.2f}")
    lines += [f"final diameter {s['d_final']:.4f} mm (error {1000 * s['error']:+.1f} um, "
              f"{'within' if s['in_tolerance'] else 'OUTSIDE'} +/-{1000 * p['tol']:.1f} um)",
              f"{s['n_passes']} passes, {s['time_s']:.1f} s, removed {s['volume']:.1f} mm^3 "
              f"({s['productivity']:.0f} mm^3/min)",
              f"finish Ra {s['Ra']:.2f} um (band {p['ra_band'][0]}-{p['ra_band'][1]}), "
              f"peak workpiece surface temperature {s['T_max']:.1f} C "
              f"(aging-range placeholder {T_CRIT_ALUMINIUM:.0f} C)"]
    if s["stalled"]:
        lines.append("WARNING: stalled - remaining stock is too thin to remove (minimum chip thickness)")
    return "\n".join(lines)


def report_sensitivity(rows: list[dict[str, Any]]) -> str:
    """Text table for sensitivity() rows."""
    lines = ["sensitivity (nominal pass count replayed on a perturbed model)",
             f"{'case':<26}{'d_final':>9}{'err um':>8}  in tol  re-plan passes"]
    lines += [f"{r['case']:<26}{r['d_final']:>9.4f}{r['err_um']:>8.1f}  {str(r['in_tolerance']):<6}  {r['n_replan']}"
              for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    for name in DEFAULTS:
        if name != "vc":
            parser.add_argument(f"--{name}", type=float, default=DEFAULTS[name])
    parser.add_argument("--material", choices=MATERIALS, default="al7075")
    parser.add_argument("--E", type=float, help="Young's modulus (GPa); overrides the material's")
    parser.add_argument("--target", type=float, help="desired final diameter (mm); solves for nominal ap")
    parser.add_argument("--part3", action="store_true", help="exam Part 3 bar: plan the passes (searches lathe settings)")
    parser.add_argument("--f-rough", type=float, help="with --part3: roughing feed (mm/rev)")
    parser.add_argument("--f-finish", type=float, help="with --part3: finishing feed (mm/rev)")
    parser.add_argument("--rpm", type=float, help="with --part3: spindle speed")
    parser.add_argument("--plot", action="store_true")
    a = parser.parse_args()
    p = {**DEFAULTS, **{k: getattr(a, k) for k in DEFAULTS if k != "vc"}}
    mat = {**MATERIALS[a.material], **({"E_GPa": a.E} if a.E else {})}

    if a.part3:
        p = {**PART3, "pl": a.pl}
        if a.f_rough and a.f_finish and a.rpm:
            plan = plan_passes(p, mat, a.f_rough, a.f_finish, a.rpm)
        else:
            plan = search_strategy(p, mat)
            print("fastest setting meeting tolerance, Ra and temperature:", plan["config"], "\n")
        print(report_plan(p, mat, plan))
        kw = plan["config"] if "config" in plan else dict(f_rough=a.f_rough, f_finish=a.f_finish, rpm=a.rpm)
        print("\n" + report_sensitivity(sensitivity(p, mat, **kw)))
        return

    if a.target is not None:
        ap_nom, res = solve_nominal_ap(p, mat, a.target)
        p["ap"] = ap_nom
        print(f"target diameter {a.target} mm -> command nominal ap = {ap_nom:.4f} mm\n")
    else:
        res = solve_deflection(p, mat)
    print(report(p, mat, res))
    if a.plot:
        plot(p, mat, res)


if __name__ == "__main__":
    main()
