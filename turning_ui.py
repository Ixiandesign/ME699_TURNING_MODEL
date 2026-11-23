"""Desktop UI for turning_model.py (tkinter + matplotlib). Run: python turning_ui.py

Two tabs share one editable material table:
  Single cut  - forces, deflection, true vs nominal depth of cut (the workbook + deflection model)
  Part 3 plan - multi-pass plan for the exam bar, Ra, temperature, cycle time, sensitivity
Parsing, running and plotting are plain functions (no tkinter), so they are testable headless.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Any

import numpy as np
from matplotlib.figure import Figure

import turning_model as tm

# (key, label, default text). Defaults come straight from the model's dicts.
SINGLE_FIELDS = [
    ("rc", "Nose radius rc (mm)", str(tm.DEFAULTS["rc"])),
    ("re", "Edge radius re (mm, 0 = sharp)", str(tm.DEFAULTS["re"])),
    ("pl", "Ploughing scale pl (0 = off)", str(tm.DEFAULTS["pl"])),
    ("f", "Feed f (mm/rev)", str(tm.DEFAULTS["f"])),
    ("ap", "Nominal radial depth ap (mm)", str(tm.DEFAULTS["ap"])),
    ("d", "Workpiece diameter d (mm)", str(tm.DEFAULTS["d"])),
    ("L", "Overhang length L (mm)", str(tm.DEFAULTS["L"])),
    ("k_machine", "Machine stiffness (N/mm)", str(tm.DEFAULTS["k_machine"])),
    ("vc", "Cutting speed vc (m/min)", str(tm.DEFAULTS["vc"])),
    ("target", "Target diameter (mm, optional)", ""),
]
_list = lambda xs: ", ".join(f"{x:g}" for x in xs)  # noqa: E731
PART3_FIELDS = [
    ("mode", "Mode (search / fixed)", "search"),
    ("d", "Start diameter d (mm)", str(tm.PART3["d"])),
    ("d_bar", "Un-turned bar diameter (mm)", str(tm.PART3["d_bar"])),
    ("L", "Overhang length L (mm)", str(tm.PART3["L"])),
    ("Ls", "Turned length Ls (mm)", str(tm.PART3["Ls"])),
    ("rc", "Nose radius rc (mm)", str(tm.PART3["rc"])),
    ("re", "Edge radius re (mm)", str(tm.PART3["re"])),
    ("pl", "Ploughing scale pl (0 = off)", str(tm.PART3["pl"])),
    ("k_machine", "Machine stiffness (N/mm)", str(tm.PART3["k_machine"])),
    ("d_target", "Target diameter (mm)", str(tm.PART3["d_target"])),
    ("tol", "Tolerance +/- (mm)", str(tm.PART3["tol"])),
    ("ra_lo", "Ra minimum (um)", str(tm.PART3["ra_band"][0])),
    ("ra_hi", "Ra maximum (um)", str(tm.PART3["ra_band"][1])),
    ("beta", "Heat fraction into workpiece", str(tm.PART3["beta"])),
    ("w_contact", "Thermal contact width (mm)", str(tm.PART3["w_contact"])),
    ("allow", "Rough/finish stock threshold (mm)", "0.05"),
    ("margin", "Stop at margin x tolerance", "0.5"),
    ("rpms", "Search: spindle speeds (rpm)", _list(tm.LATHE["rpm"])),
    ("feeds", "Search: feeds (mm/rev)", _list(tm.LATHE["feed"])),
    ("f_rough", "Fixed: roughing feed (mm/rev)", "0.1"),
    ("f_finish", "Fixed: finishing feed (mm/rev)", "0.05"),
    ("rpm", "Fixed: spindle speed (rpm)", "2000"),
]
MATERIAL_FIELDS = [
    ("E_GPa", "Young's modulus E (GPa)"), ("kc1", "kc1 (N/mm^2)"), ("mc", "mc"),
    ("kf1", "kf1 (N/mm^2)"), ("mf", "mf"), ("sigma_y", "Yield strength (MPa)"),
    ("rho", "Density (kg/m^3)"), ("cp", "Specific heat (J/kg/K)"), ("k_th", "Conductivity (W/m/K)"),
]


# ------------------------------------------------------------------ parsing

def _num(vals: dict[str, str], key: str, label: str) -> float:
    """float(vals[key]) with an error that names the field."""
    try:
        return float(vals[key])
    except (KeyError, ValueError):
        raise ValueError(f"{label}: '{vals.get(key, '')}' is not a number") from None


def _numbers(vals: dict[str, str], key: str, label: str) -> tuple[float, ...]:
    """Comma-separated list of numbers."""
    try:
        out = tuple(float(x) for x in vals[key].split(",") if x.strip())
    except (KeyError, ValueError):
        raise ValueError(f"{label}: '{vals.get(key, '')}' is not a comma-separated list of numbers") from None
    if not out:
        raise ValueError(f"{label}: enter at least one value")
    return out


def material_strings(mat: dict[str, Any]) -> dict[str, str]:
    """Material numbers as entry text."""
    return {k: str(float(mat[k])) for k, _ in MATERIAL_FIELDS}


def parse_material(preset: str, vals: dict[str, str]) -> dict[str, Any]:
    """Preset material with the edited numbers applied."""
    mat = dict(tm.MATERIALS[preset])
    for key, label in MATERIAL_FIELDS:
        mat[key] = _num(vals, key, label)
    return mat


def parse_single(vals: dict[str, str]) -> tuple[dict[str, Any], float | None]:
    """Process dict for the single-cut tab, and the optional target diameter."""
    labels = {k: lab for k, lab, _ in SINGLE_FIELDS}
    p = {**tm.DEFAULTS, **{k: _num(vals, k, labels[k]) for k in tm.DEFAULTS}}
    target = _num(vals, "target", labels["target"]) if vals.get("target", "").strip() else None
    return p, target


def parse_part3(vals: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process dict for the exam bar, and the plan settings (mode, feeds, rpm lists...)."""
    labels = {k: lab for k, lab, _ in PART3_FIELDS}
    keys = ["d", "d_bar", "L", "Ls", "rc", "re", "pl", "k_machine", "d_target", "tol", "beta", "w_contact"]
    p = {**tm.PART3, **{k: _num(vals, k, labels[k]) for k in keys},
         "ra_band": (_num(vals, "ra_lo", labels["ra_lo"]), _num(vals, "ra_hi", labels["ra_hi"]))}
    cfg: dict[str, Any] = dict(mode=vals["mode"].strip(), allow=_num(vals, "allow", labels["allow"]),
                               margin=_num(vals, "margin", labels["margin"]))
    if cfg["mode"] == "search":
        cfg.update(rpms=_numbers(vals, "rpms", labels["rpms"]), feeds=_numbers(vals, "feeds", labels["feeds"]))
    elif cfg["mode"] == "fixed":
        cfg.update({k: _num(vals, k, labels[k]) for k in ("f_rough", "f_finish", "rpm")})
    else:
        raise ValueError("Mode: use 'search' or 'fixed'")
    return p, cfg


# ------------------------------------------------------------------ running

def run_single(p: dict[str, Any], mat: dict[str, Any], target: float | None) -> dict[str, Any]:
    """Solve one cut (optionally for a target diameter) and format the report."""
    p = dict(p)
    head = ""
    if target is not None:
        ap_nom, res = tm.solve_nominal_ap(p, mat, target)
        p["ap"] = ap_nom
        head = f"target diameter {target} mm -> command nominal ap = {ap_nom:.4f} mm\n\n"
    else:
        res = tm.solve_deflection(p, mat)
    table = ["", "pass  deflection [um]  % of nominal ap"]
    table += [f"{i:>4}  {d * 1000:>15.1f}  {d / p['ap']:>14.0%}" for i, d in enumerate(res["history"][:12], 1)]
    return dict(text=head + tm.report(p, mat, res) + "\n".join(table), p=p, res=res)


def run_part3(p: dict[str, Any], mat: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Plan the passes (searching the lathe settings or with fixed ones), then the sensitivity table."""
    plan_kw = dict(allow=cfg["allow"], margin=cfg["margin"])
    if cfg["mode"] == "search":
        plan = tm.search_strategy(p, mat, cfg["rpms"], cfg["feeds"], **plan_kw)
        kw = plan["config"]
        head = f"fastest setting meeting tolerance, Ra and temperature: {kw}\n\n"
    else:
        kw = dict(f_rough=cfg["f_rough"], f_finish=cfg["f_finish"], rpm=cfg["rpm"])
        plan = tm.plan_passes(p, mat, **kw, **plan_kw)
        head = ""
    rows = tm.sensitivity(p, mat, **kw, **plan_kw)
    text = head + tm.report_plan(p, mat, plan) + "\n\n" + tm.report_sensitivity(rows)
    return dict(text=text, p=p, plan=plan, sensitivity=rows)


# ------------------------------------------------------------------ figures

def _profile(p: dict[str, Any], mat: dict[str, Any], f: float, ap: float) -> dict[str, np.ndarray] | None:
    """Workbook-style element table (100 rows, 99 intervals) for the engaged arc, or None if no cut."""
    if ap <= 1e-9:
        return None
    k_min, k_max = tm.kappa_limits(p["rc"], f, ap)
    return tm.force_profile(p["rc"], f, ap, mat, dkappa=(k_max - k_min) / 99,
                            re=p.get("re", 0.0), pl=p.get("pl", 0.0))


def plot_single(fig: Figure, p: dict[str, Any], mat: dict[str, Any], res: dict[str, Any]) -> None:
    """Six panels: the workbook's chip-thickness / element-force / cumulative-force columns, plus deflection."""
    fig.clear()
    ax = fig.subplots(2, 3)
    prof = _profile(p, mat, p["f"], res["ap_true"])
    if prof is not None:
        deg = np.degrees(prof["kappa"])
        ax[0, 0].plot(deg, prof["h"] * 1000)
        ax[0, 1].plot(deg, prof["dFc"], label="dFc")
        ax[0, 1].plot(deg, prof["dFf"], label="dFf")
        ax[0, 2].plot(deg, prof["Fc_cum"], label="sum Fc")
        ax[0, 2].plot(deg, prof["Fr_cum"], label="sum Fradial")
        ax[0, 2].plot(deg, prof["Fa_cum"], label="sum Faxial")
        ax[0, 1].legend()
        ax[0, 2].legend()
    ax[0, 0].set(xlabel="kappa (deg)", ylabel="uncut chip thickness (um)", title="Chip thickness over engaged arc")
    ax[0, 1].set(xlabel="kappa (deg)", ylabel="force per element (N)", title="Element forces")
    ax[0, 2].set(xlabel="kappa (deg)", ylabel="cumulative force (N)", title="Force sums over the arc")

    ax[1, 0].plot(range(1, len(res["history"]) + 1), np.array(res["history"]) * 1000, "o-")
    ax[1, 0].set(xlabel="iteration", ylabel="deflection (um)", title="Deflection convergence")

    top = min(1.5 * p["ap"], 2 * p["rc"] * 0.98)
    noms = np.linspace(0.1 * top, top, 20)
    trues = [tm.solve_deflection({**p, "ap": a}, mat)["ap_true"] for a in noms]
    ax[1, 1].plot(noms, trues, label="predicted")
    ax[1, 1].plot(noms, noms, "--", label="no deflection")
    ax[1, 1].plot([res["ap_nom"]], [res["ap_true"]], "ro", label="this cut")
    ax[1, 1].set(xlabel="nominal ap (mm)", ylabel="true ap (mm)", title="True vs nominal depth of cut")
    ax[1, 1].legend()

    ax[1, 2].bar(["Fc", "Fradial", "Faxial"], [res["Fc"], res["Fr"], res["Fa"]])
    ax[1, 2].set(ylabel="force (N)", title="Forces at the true depth of cut")
    fig.tight_layout()


def plot_plan(fig: Figure, p: dict[str, Any], mat: dict[str, Any], plan: dict[str, Any]) -> None:
    """Six panels: convergence to the target, depths, forces, deflection, temperature, chip thickness."""
    fig.clear()
    ax = fig.subplots(2, 3)
    rows = plan["passes"]
    n = np.array([r["n"] for r in rows])
    above = np.maximum([1000 * (r["d_after"] - p["d_target"]) for r in rows], 1e-3)
    ax[0, 0].semilogy(n, above, "o-")
    ax[0, 0].axhline(1000 * p["tol"], ls="--", c="r", label="tolerance")
    ax[0, 0].set(xlabel="pass", ylabel="diameter above target (um)", title="Spring-pass convergence")
    ax[0, 0].legend()

    ax[0, 1].bar(n - 0.2, [1000 * r["ap_nom"] for r in rows], 0.4, label="commanded")
    ax[0, 1].bar(n + 0.2, [1000 * r["ap_true"] for r in rows], 0.4, label="true")
    ax[0, 1].set(xlabel="pass", ylabel="radial depth (um)", title="Commanded vs true depth")
    ax[0, 1].legend()

    ax[0, 2].plot(n, [r["Fc"] for r in rows], "o-", label="Fc")
    ax[0, 2].plot(n, [r["Fr"] for r in rows], "o-", label="Fradial")
    ax[0, 2].set(xlabel="pass", ylabel="force (N)", title="Forces per pass")
    ax[0, 2].legend()

    ax[1, 0].plot(n, [1000 * r["deflection"] for r in rows], "o-")
    ax[1, 0].set(xlabel="pass", ylabel="deflection (um)", title="Workpiece deflection per pass")

    ax[1, 1].plot(n, [r["T"] for r in rows], "o-", label="peak surface T")
    ax[1, 1].axhline(tm.T_CRIT_ALUMINIUM, ls="--", c="r", label="aging-range placeholder")
    ax[1, 1].set(xlabel="pass", ylabel="temperature (C)", title="Flash temperature per pass")
    ax[1, 1].legend()

    for r, name in ((rows[0], "first pass"), (rows[-1], "last pass")):
        prof = _profile(p, mat, r["f"], r["ap_true"])
        if prof is not None:
            ax[1, 2].plot(np.degrees(prof["kappa"]), prof["h"] * 1000, label=name)
    ax[1, 2].set(xlabel="kappa (deg)", ylabel="uncut chip thickness (um)", title="Chip thickness over engaged arc")
    ax[1, 2].legend()
    fig.tight_layout()


# ------------------------------------------------------------------- window

def use_own_tcl() -> None:
    """Point Tcl/Tk at this interpreter's own libraries.

    A leftover TCL_LIBRARY / TK_LIBRARY from another Python install (different Tcl version)
    makes tk.Tk() fail, so the window never opens. Call before creating the root window.
    """
    for var, prefix in (("TCL_LIBRARY", "tcl8."), ("TK_LIBRARY", "tk8.")):
        found = sorted(glob.glob(os.path.join(sys.base_prefix, "tcl", prefix + "*")))
        if found:
            os.environ[var] = found[-1]


def build_ui(root: Any) -> dict[str, Any]:
    """Create the widgets on `root`; returns the run callbacks, figures and output widgets."""
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root.title("Finish-turning model")
    root.geometry(f"{min(1500, int(root.winfo_screenwidth() * 0.95))}x{int(root.winfo_screenheight() * 0.88)}+0+0")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # material table shared by both tabs (same StringVars, so an edit shows up everywhere)
    preset = tk.StringVar(value="al7075")
    mat_vars = {k: tk.StringVar(value=v) for k, v in material_strings(tm.MATERIALS["al7075"]).items()}

    def load_preset(*_: Any) -> None:
        for k, v in material_strings(tm.MATERIALS[preset.get()]).items():
            mat_vars[k].set(v)

    def material() -> dict[str, Any]:
        return parse_material(preset.get(), {k: v.get() for k, v in mat_vars.items()})

    def make_tab(title: str, fields: list[tuple[str, str, str]], run: Any) -> dict[str, Any]:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        form = ttk.Frame(tab)
        form.pack(side="left", fill="y", padx=8, pady=8)
        vars_: dict[str, tk.StringVar] = {}
        r = 0
        ttk.Label(form, text="Material").grid(row=r, column=0, sticky="w")
        box = ttk.Combobox(form, textvariable=preset, values=list(tm.MATERIALS), width=14, state="readonly")
        box.grid(row=r, column=1)
        box.bind("<<ComboboxSelected>>", load_preset)
        for key, label in MATERIAL_FIELDS:
            r += 1
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(form, textvariable=mat_vars[key], width=16).grid(row=r, column=1)
        r += 1
        ttk.Separator(form).grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        for key, label, default in fields:
            r += 1
            vars_[key] = tk.StringVar(value=default)
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            if key == "mode":
                ttk.Combobox(form, textvariable=vars_[key], values=["search", "fixed"], width=14,
                             state="readonly").grid(row=r, column=1)
            else:
                ttk.Entry(form, textvariable=vars_[key], width=16).grid(row=r, column=1)
        ttk.Button(form, text="Run", command=run).grid(row=r + 1, column=0, columnspan=2, pady=8, sticky="ew")
        out = ttk.Frame(tab)
        out.pack(side="left", fill="both", expand=True)
        box_frame = ttk.Frame(out)                        # text + scrollbar (reports run ~35 lines)
        box_frame.pack(fill="x")
        text = tk.Text(box_frame, height=16, font=("Consolas", 9), wrap="none")
        scroll = ttk.Scrollbar(box_frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="x", expand=True)
        fig = Figure(figsize=(9, 5))
        canvas = FigureCanvasTkAgg(fig, master=out)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return dict(vars=vars_, text=text, fig=fig, canvas=canvas)

    def show(tab: dict[str, Any], msg: str) -> None:
        tab["text"].delete("1.0", "end")
        tab["text"].insert("1.0", msg)

    def run_tab(tab: dict[str, Any], parse: Any, run: Any, plot: Any) -> None:
        """Parse the form, run the model, show text and figures; errors go to the text box."""
        show(tab, "running...")
        root.config(cursor="watch")
        root.update_idletasks()
        try:
            vals = {k: v.get() for k, v in tab["vars"].items()}
            p, extra = parse(vals)
            mat = material()
            out = run(p, mat, extra)
            show(tab, out["text"])
            plot(tab["fig"], out["p"], mat, out.get("res") or out["plan"])
            tab["canvas"].draw()
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            show(tab, f"Error: {e}")
        finally:
            root.config(cursor="")

    tabs: dict[str, dict[str, Any]] = {}
    run_single_cb = lambda: run_tab(tabs["single"], parse_single, run_single, plot_single)  # noqa: E731
    run_part3_cb = lambda: run_tab(tabs["part3"], parse_part3, run_part3, plot_plan)  # noqa: E731
    tabs["single"] = make_tab("Single cut", SINGLE_FIELDS, run_single_cb)
    tabs["part3"] = make_tab("Part 3 plan", PART3_FIELDS, run_part3_cb)
    root.bind("<Return>", lambda *_: (run_single_cb, run_part3_cb)[notebook.index("current")]())
    run_single_cb()                                   # show something on start-up
    return dict(run_single=run_single_cb, run_part3=run_part3_cb, tabs=tabs, notebook=notebook)


def main() -> None:
    import tkinter as tk

    use_own_tcl()
    root = tk.Tk()
    build_ui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
