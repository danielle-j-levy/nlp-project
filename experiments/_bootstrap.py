"""Path + working-directory setup shared by every row script.

Importing this makes the shared modules in ../core importable and pins the
working directory to the repo root so that data/ paths resolve regardless of
where the script was launched from.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
os.chdir(ROOT)


# Every emit() call records here, so a notebook / driver can collect the
# computed numbers without parsing stdout.
RESULTS = {}


def emit(row, published, computed):
    RESULTS[row] = {"published": published, "computed": computed}
    print(f"\n{row}")
    print("  published:", json.dumps(published))
    print("  computed :", json.dumps(computed))
    return computed
