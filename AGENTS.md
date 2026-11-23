# ME699 finish-turning deflection model

Python model for the ME699 homework and ME599/699 final exam Part 3: cantilever beam deflection + finish-turning
forces (default AA7075-T6, `--material ti64` for the homework/workbook set) -> true vs nominal depth of cut, plus a
multi-pass plan to a target diameter (Ra, flash temperature, cycle time, sensitivity). Reference physics and the
example workbook live in `ME699_context/` (start at `INDEX.md`; not in the repo at the moment). `MODEL_NOTES.md` explains
the method; lathe settings, thermal constants and the critical temperature are placeholders until the course values are supplied.

## Repo rules
- Keep folder structure, organization and file count simple and minimal.
- Comment code, but not excessively.
- CPU-only NumPy; no GPU/CUDA, no solver libraries. Physics must come from the course context.
- Units: mm, N, N/mm^2, rad. Only convert to um inside the kc/kf power laws.
- No dataclasses/classes: inputs are plain dicts (`p` process, `mat` material), results are plain dicts.
- Notebooks: `deflection_model.ipynb` (inputs + model cell), `deflection_model_expanded.ipynb` (one cell per model step, plain NumPy, cross-checked against the module).
- Tooling: pyright (Python LSP) and Context7 MCP (`.mcp.json`) are installed.

## Commands
- `python -m pytest -q`
- `python turning_model.py [--plot]`
- `python turning_model.py --part3 [--f-rough 0.1 --f-finish 0.05 --rpm 2000]`
- `pyright turning_model.py test_turning_model.py`
