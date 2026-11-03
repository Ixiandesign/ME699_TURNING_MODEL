"""Finish-turning deflection model (ME699 homework, Parts 1 and 2).

Part 1: cantilever beam deflection, point load at the free end.
Part 2: finish-turning forces with a nose-radius tool, the resulting
workpiece + machine deflection, and the true vs nominal (radial) depth of cut.

Units: mm, N, N/mm^2 (= MPa), rad. Only the kc/kf power laws use micrometres
(h_um = 1000 * h_mm). Inputs are two plain dicts:
  p   process: rc, f, ap, d, L, k_machine, vc  (see DEFAULTS)
  mat material: name, E_GPa, kc1, mc, kf1, mf  (see MATERIALS)

Usage: python turning_model.py [--material ti64 --rc 0.8 --f 0.18 --ap 0.25 --target 15.5 --plot]
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
)

# kc = kc1 * h_um**mc, kf = kf1 * h_um**mf (N/mm^2, h in um)
MATERIALS: dict[str, dict[str, Any]] = {
    # E from the course AA6061 thermal sheet; kc/kf power-law fit of the course
    # 7075 Fc/Ff data (Cutting Force Data.xlsx, 3 mm width) used as an Al proxy,
    # since the course has no 6061 force data. 7075 is harder -> forces run high.
    "al6061": dict(name="Al6061", E_GPa=70.0, kc1=2534.0, mc=-0.149, kf1=2126.0, mf=-0.168),
    # Homework / Nose Radius workbook constants
    "ti64": dict(name="Ti-6Al-4V", E_GPa=114.0, kc1=4500.0, mc=-0.175, kf1=18250.0, mf=-0.611),
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


def integrate_forces(rc: float, f: float, ap: float, mat: dict[str, Any],
                     dkappa: float = DKAPPA) -> dict[str, float]:
    """Sum dF = k(h) * h * ds over the engaged arc (ds = rc * dkappa).

    Returns Fc (tangential), Fr (radial), Fa (axial) in N and mean chip thickness h_avg in mm.
    """
    if ap <= 1e-9:
        return dict(Fc=0.0, Fr=0.0, Fa=0.0, h_avg=0.0)
    k_min, k_max = kappa_limits(rc, f, ap)
    n = max(1, round((k_max - k_min) / dkappa))
    kappa = np.linspace(k_min, k_max, n + 1)      # grid includes both end points
    ds = rc * (k_max - k_min) / n                 # arc length per element (mm)
    h = np.maximum(uncut_chip_thickness(kappa, rc, f), 1e-9)   # mm
    h_um = h * 1000                               # kc/kf are defined for h in um
    dFc = mat["kc1"] * h_um ** mat["mc"] * h * ds
    dFf = mat["kf1"] * h_um ** mat["mf"] * h * ds  # normal to the cutting edge
    return dict(Fc=float(dFc.sum()),
                Fr=float((np.cos(kappa) * dFf).sum()),
                Fa=float((np.sin(kappa) * dFf).sum()),
                h_avg=float(h.mean()))


# ----------------------------------------------- Part 2: deflection + iteration

def compliance(p: dict[str, Any], mat: dict[str, Any]) -> tuple[float, float]:
    """(beam, machine) radial compliance at the tool tip, mm/N."""
    beam = cantilever_deflection(1.0, p["L"], mat["E_GPa"] * 1000, circular_I(p["d"]))
    return beam, 1.0 / p["k_machine"]


def solve_deflection(p: dict[str, Any], mat: dict[str, Any], relax: float = 0.5,
                     max_iter: int = 200, tol: float = 1e-6) -> dict[str, Any]:
    """Iterate ap_true = ap_nom - deflection(F_radial(ap_true)).

    relax=1 is the workbook's plain successive substitution (it oscillates
    towards the answer); 0.5 damps that and reaches the same fixed point.
    """
    c_beam, c_mach = compliance(p, mat)
    ap, history, converged = p["ap"], [], False
    for _ in range(max_iter):
        defl = integrate_forces(p["rc"], p["f"], ap, mat)["Fr"] * (c_beam + c_mach)
        history.append(defl)
        new = ap + relax * (max(p["ap"] - defl, 0.0) - ap)
        converged = abs(new - ap) < tol
        ap = new
        if converged:
            break
    F = integrate_forces(p["rc"], p["f"], ap, mat)
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


# --------------------------------------------------------------------- report

def report(p: dict[str, Any], mat: dict[str, Any], res: dict[str, Any]) -> str:
    rpm = 1000 * p["vc"] / (math.pi * p["d"])
    torque = res["Fc"] * p["d"] / 2 / 1000                      # N*m
    power = torque * 2 * math.pi * rpm / 60                     # W
    Ra = p["f"]**2 / (32 * p["rc"]) * 1000 * 4 / 3              # um, workbook's 4/3 factor
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


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    for name in DEFAULTS:
        if name != "vc":
            parser.add_argument(f"--{name}", type=float, default=DEFAULTS[name])
    parser.add_argument("--material", choices=MATERIALS, default="al6061")
    parser.add_argument("--E", type=float, help="Young's modulus (GPa); overrides the material's")
    parser.add_argument("--target", type=float, help="desired final diameter (mm); solves for nominal ap")
    parser.add_argument("--plot", action="store_true")
    a = parser.parse_args()
    p = {**DEFAULTS, **{k: getattr(a, k) for k in DEFAULTS if k != "vc"}}
    mat = {**MATERIALS[a.material], **({"E_GPa": a.E} if a.E else {})}

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
