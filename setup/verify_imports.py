"""
RT-FSAS Package Verification Script
Run: python setup/verify_imports.py
"""

packages = [
    ("torch",            "import torch; v = torch.__version__"),
    ("torch_geometric",  "import torch_geometric; v = torch_geometric.__version__"),
    ("faiss",            "import faiss; v = 'ok'"),
    ("google-generativeai", "import google.generativeai; v = 'ok'"),
    ("statsbombpy",      "from statsbombpy import sb; v = 'ok'"),
    ("pandas",           "import pandas as pd; v = pd.__version__"),
    ("numpy",            "import numpy as np; v = np.__version__"),
    ("scikit-learn",     "import sklearn; v = sklearn.__version__"),
    ("matplotlib",       "import matplotlib; v = matplotlib.__version__"),
    ("plotly",           "import plotly; v = plotly.__version__"),
    ("dash",             "import dash; v = dash.__version__"),
    ("tqdm",             "import tqdm; v = tqdm.__version__"),
    ("joblib",           "import joblib; v = joblib.__version__"),
    ("scipy",            "import scipy; v = scipy.__version__"),
]

print()
print("RT-FSAS Package Verification")
print("=" * 45)

passed, failed = 0, 0
for name, code in packages:
    try:
        loc = {}
        exec(code, {}, loc)
        ver = loc.get("v", "ok")
        print(f"  OK   {name:<25} {ver}")
        passed += 1
    except Exception as e:
        print(f"  FAIL {name:<25} {e}")
        failed += 1

print("=" * 45)
print(f"  {passed} passed, {failed} failed")
if failed == 0:
    print("  All packages ready. Environment is good!")
else:
    print("  Fix the failed packages before continuing.")
print()
