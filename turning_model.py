"""Finish-turning deflection model (ME699 homework, Parts 1 and 2).

Part 1: cantilever beam deflection, point load at the free end.
Part 2: Ti-6Al-4V finish-turning forces with a nose-radius tool, the resulting
workpiece + machine deflection, and the true vs nominal (radial) depth of cut.

Units: mm, N, N/mm^2 (= MPa), rad. Lengths/forces carry _mm/_N suffixes.
Only the kc/kf power laws use micrometres (h_um = 1000 * h_mm).

Usage: python turning_model.py [--rc 0.8 --f 0.18 --ap 0.25 --target 15.5 --plot]
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Inputs:
    """Process, material and machine inputs (defaults: homework hints + example workbook)."""
    rc_mm: float = 0.8            # tool nose radius
    f_mm: float = 0.18            # feed per revolution (axial)
    ap_mm: float = 0.25           # nominal depth of cut (radial)
    d_mm: float = 16.0            # workpiece diameter before the pass
    L_mm: float = 160.0           # overhang length (L/D = 10)
    E_GPa: float = 114.0          # Ti-6Al-4V Young's modulus
    k_machine_N_mm: float = 50_000.0  # machine/tool-holder stiffness
    vc_m_min: float = 60.0        # cutting speed
    kc1: float = 4500.0           # specific cutting force at h = 1 um (N/mm^2)
    mc: float = -0.175            # kc = kc1 * h_um**mc
    kf1: float = 18250.0          # specific feed force at h = 1 um (N/mm^2)
    mf: float = -0.611            # kf = kf1 * h_um**mf
    dkappa: float = 0.005         # target kappa step (rad)


@dataclass(frozen=True)
class Forces:
    """Forces summed over the engaged nose arc."""
    Fc_N: float       # tangential cutting force
    Fr_N: float       # radial (thrust) force -> deflects the workpiece
    Fa_N: float       # axial (feed direction) force
    h_avg_mm: float   # mean uncut chip thickness over the arc
    kappa_min: float
    kappa_max: float


@dataclass(frozen=True)
class DeflectionResult:
    ap_nom_mm: float
    ap_true_mm: float
    beam_mm: float        # workpiece bending at the tool
    machine_mm: float     # machine/tool-holder compliance
    forces: Forces        # evaluated at the converged true depth of cut
    history_mm: list[float]   # total deflection at each pass
    converged: bool
    d_zero_mm: float      # final diameter with zero deflection
    d_final_mm: float     # predicted final diameter

    @property
    def deflection_mm(self) -> float:
        return self.beam_mm + self.machine_mm


# ---------------------------------------------------------------- Part 1: beam

def circular_I(d_mm: float) -> float:
    """Second moment of area of a solid round bar (mm^4)."""
    return math.pi * d_mm**4 / 64


def cantilever_deflection(F_N: float, L_mm: float, E_N_mm2: float, I_mm4: float) -> float:
    """Tip deflection (mm) of a cantilever with a point load F at the free end."""
    return F_N * L_mm**3 / (3 * E_N_mm2 * I_mm4)


# ------------------------------------------------------------ Part 2: geometry

def kappa_limits(rc_mm: float, f_mm: float, ap_mm: float) -> tuple[float, float]:
    """Engaged nose-arc limits (rad): kappa_min from the feed, kappa_max from ap."""
    if not 0 < ap_mm <= 2 * rc_mm:
        raise ValueError("ap must be in (0, 2*rc]")
    k_min = -math.acos((rc_mm - f_mm**2 / (8 * rc_mm)) / rc_mm)
    k_max = math.acos((rc_mm - ap_mm) / rc_mm)
    return k_min, k_max


def uncut_chip_thickness(kappa, rc_mm: float, f_mm: float):
    """Uncut chip thickness h(kappa) in mm (homework formula)."""
    c = np.cos(kappa - np.pi / 2)
    return rc_mm + f_mm * c - np.sqrt(rc_mm**2 + f_mm**2 * (c**2 - 1))


def integrate_forces(inp: Inputs, ap_mm: float, dkappa: float | None = None) -> Forces:
    """Sum dF = k(h) * h * ds over the engaged arc (ds = rc * dkappa)."""
    if ap_mm <= 1e-9:
        return Forces(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k_min, k_max = kappa_limits(inp.rc_mm, inp.f_mm, ap_mm)
    n = max(1, round((k_max - k_min) / (dkappa or inp.dkappa)))
    kappa = np.linspace(k_min, k_max, n + 1)      # grid includes both end points
    ds_mm = inp.rc_mm * (k_max - k_min) / n       # arc length per element
    h_mm = np.maximum(uncut_chip_thickness(kappa, inp.rc_mm, inp.f_mm), 1e-9)
    h_um = h_mm * 1000                            # kc/kf are defined for h in um
    dFc = inp.kc1 * h_um**inp.mc * h_mm * ds_mm
    dFf = inp.kf1 * h_um**inp.mf * h_mm * ds_mm   # normal to the cutting edge
    return Forces(
        Fc_N=float(dFc.sum()),
        Fr_N=float((np.cos(kappa) * dFf).sum()),
        Fa_N=float((np.sin(kappa) * dFf).sum()),
        h_avg_mm=float(h_mm.mean()),
        kappa_min=k_min,
        kappa_max=k_max,
    )


# ----------------------------------------------- Part 2: deflection + iteration

def compliance_mm_per_N(inp: Inputs) -> tuple[float, float]:
    """(beam, machine) radial compliance at the tool tip."""
    beam = cantilever_deflection(1.0, inp.L_mm, inp.E_GPa * 1000, circular_I(inp.d_mm))
    return beam, 1.0 / inp.k_machine_N_mm


def solve_deflection(inp: Inputs, relax: float = 0.5, max_iter: int = 200,
                     tol: float = 1e-6) -> DeflectionResult:
    """Iterate ap_true = ap_nom - deflection(F_radial(ap_true)).

    relax=1 is the workbook's plain successive substitution (it oscillates
    towards the answer); 0.5 damps that and reaches the same fixed point.
    """
    c_beam, c_mach = compliance_mm_per_N(inp)
    ap, history, converged = inp.ap_mm, [], False
    for _ in range(max_iter):
        defl = integrate_forces(inp, ap).Fr_N * (c_beam + c_mach)
        history.append(defl)
        new = ap + relax * (max(inp.ap_mm - defl, 0.0) - ap)
        converged = abs(new - ap) < tol
        ap = new
        if converged:
            break
    F = integrate_forces(inp, ap)
    return DeflectionResult(
        ap_nom_mm=inp.ap_mm, ap_true_mm=ap,
        beam_mm=F.Fr_N * c_beam, machine_mm=F.Fr_N * c_mach,
        forces=F, history_mm=history, converged=converged,
        d_zero_mm=inp.d_mm - 2 * inp.ap_mm,      # ap is radial -> diameter drops by 2*ap
        d_final_mm=inp.d_mm - 2 * ap,
    )


def solve_nominal_ap(inp: Inputs, d_target_mm: float, tol: float = 1e-5,
                     max_iter: int = 200) -> tuple[float, DeflectionResult]:
    """Nominal ap to command so the deflected cut lands on d_target (replaces Goal Seek)."""
    ap_target = (inp.d_mm - d_target_mm) / 2
    ap_nom = ap_target
    res = solve_deflection(replace(inp, ap_mm=ap_nom))
    for _ in range(max_iter):
        err = ap_target - res.ap_true_mm
        if abs(err) < tol:
            break
        ap_nom += err        # add back what the tool failed to remove
        res = solve_deflection(replace(inp, ap_mm=ap_nom))
    return ap_nom, res


# --------------------------------------------------------------------- report

def report(inp: Inputs, res: DeflectionResult) -> str:
    F = res.forces
    rpm = 1000 * inp.vc_m_min / (math.pi * inp.d_mm)
    torque_Nm = F.Fc_N * inp.d_mm / 2 / 1000
    power_W = torque_Nm * 2 * math.pi * rpm / 60
    Ra_um = inp.f_mm**2 / (32 * inp.rc_mm) * 1000 * 4 / 3   # workbook's 4/3 factor
    sandvik_N = 2300 * F.h_avg_mm**-0.15 * inp.f_mm * res.ap_true_mm
    lines = [
        f"rc={inp.rc_mm} mm  f={inp.f_mm} mm/rev  d={inp.d_mm} mm  L={inp.L_mm} mm (L/D={inp.L_mm / inp.d_mm:.1f})",
        f"nominal ap        {res.ap_nom_mm:.4f} mm",
        f"true ap           {res.ap_true_mm:.4f} mm  ({res.ap_true_mm / res.ap_nom_mm:.0%} of nominal)"
        + ("" if res.converged else "  [NOT CONVERGED]"),
        f"deflection        {res.deflection_mm * 1000:.1f} um  (beam {res.beam_mm * 1000:.1f} + machine {res.machine_mm * 1000:.1f})",
        f"zero-defl. dia.   {res.d_zero_mm:.4f} mm",
        f"predicted dia.    {res.d_final_mm:.4f} mm  (oversize {1000 * (res.d_final_mm - res.d_zero_mm):.1f} um)",
        f"forces            Fc {F.Fc_N:.1f} N   Fradial {F.Fr_N:.1f} N   Faxial {F.Fa_N:.1f} N",
        f"engaged arc       {math.degrees(F.kappa_min):.2f} to {math.degrees(F.kappa_max):.2f} deg, h_avg {F.h_avg_mm * 1000:.1f} um",
        f"rpm {rpm:.0f}  torque {torque_Nm:.2f} N*m  power {power_W:.0f} W",
        f"kinematic Ra {Ra_um:.2f} um   Sandvik Fc check {sandvik_N:.1f} N",
    ]
    if res.deflection_mm > 0.1 * F.h_avg_mm:
        lines.append("WARNING: deflection > 10% of average chip thickness (chatter risk)")
    if not 0.1 * inp.rc_mm <= inp.ap_mm <= inp.rc_mm / 3:
        lines.append("note: ap outside the 1/10-1/3 of rc finishing guideline")
    return "\n".join(lines)


def plot(inp: Inputs, res: DeflectionResult) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    F = res.forces
    kappa = np.linspace(F.kappa_min, F.kappa_max, 200)
    ax[0].plot(np.degrees(kappa), uncut_chip_thickness(kappa, inp.rc_mm, inp.f_mm) * 1000)
    ax[0].set(xlabel="kappa (deg)", ylabel="uncut chip thickness (um)", title="Chip thickness over engaged arc")

    ax[1].plot(range(1, len(res.history_mm) + 1), np.array(res.history_mm) * 1000, "o-")
    ax[1].set(xlabel="iteration", ylabel="deflection (um)", title="Deflection convergence")

    noms = np.linspace(0.05, 0.5, 30)
    trues = [solve_deflection(replace(inp, ap_mm=a)).ap_true_mm for a in noms]
    ax[2].plot(noms, trues, label="predicted")
    ax[2].plot(noms, noms, "--", label="no deflection")
    ax[2].set(xlabel="nominal ap (mm)", ylabel="true ap (mm)", title="True vs nominal depth of cut")
    ax[2].legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    d = Inputs()
    for name, default in [("rc", d.rc_mm), ("f", d.f_mm), ("ap", d.ap_mm), ("d", d.d_mm),
                          ("L", d.L_mm), ("E", d.E_GPa), ("k", d.k_machine_N_mm)]:
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--target", type=float, help="desired final diameter (mm); solves for nominal ap")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    inp = replace(d, rc_mm=a.rc, f_mm=a.f, ap_mm=a.ap, d_mm=a.d, L_mm=a.L, E_GPa=a.E,
                  k_machine_N_mm=a.k)

    if a.target is not None:
        ap_nom, res = solve_nominal_ap(inp, a.target)
        inp = replace(inp, ap_mm=ap_nom)
        print(f"target diameter {a.target} mm -> command nominal ap = {ap_nom:.4f} mm\n")
    else:
        res = solve_deflection(inp)
    print(report(inp, res))
    if a.plot:
        plot(inp, res)


if __name__ == "__main__":
    main()
