# Create .venv with the project's requirements and run Tcl/Tk from inside it.
# Usage (from the project folder):  .\setup_venv.ps1      then:  .\.venv\Scripts\python turning_ui.py
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m venv .venv
.\.venv\Scripts\python -m pip install -q -r requirements.txt

# A venv has no Tcl/Tk of its own: copy the base install's libraries into it.
$base = & .\.venv\Scripts\python -c "import sys; print(sys.base_prefix)"
Copy-Item "$base\tcl" .venv\tcl -Recurse -Force

# sitecustomize runs at start-up of every interpreter in the venv (UI, matplotlib --plot, notebooks)
# and points Tcl/Tk at that copy, ignoring a machine-wide TCL_LIBRARY / TK_LIBRARY from another Python.
@'
import glob, os, sys
for var, name in (("TCL_LIBRARY", "tcl8."), ("TK_LIBRARY", "tk8.")):
    found = sorted(glob.glob(os.path.join(sys.prefix, "tcl", name + "*")))
    if found:
        os.environ[var] = found[-1]
'@ | Set-Content .venv\Lib\site-packages\sitecustomize.py -Encoding utf8

Write-Host "Done. Run: .\.venv\Scripts\python turning_ui.py"
