# ME699 finish-turning deflection model

Python model for the ME699 homework: cantilever beam deflection + finish-turning
forces (default Al6061, `--material ti64` for the homework/workbook set) -> true vs nominal depth of cut. Reference physics and the
example workbook live in `ME699_context/` (start at `INDEX.md`).

## Repo rules
- Keep folder structure, organization and file count simple and minimal.
- Comment code, but not excessively.
- CPU-only NumPy; no GPU/CUDA, no solver libraries. Physics must come from the course context.
- Units: mm, N, N/mm^2, rad. Only convert to um inside the kc/kf power laws.
- Tooling: pyright (Python LSP) and Context7 MCP (`.mcp.json`) are installed.

## Commands
- `python -m pytest -q`
- `python turning_model.py [--plot]`
- `pyright turning_model.py test_turning_model.py`
