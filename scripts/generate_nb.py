import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell('# RT-FSAS: Data Exploration\nChecking La Liga 2015/16 dataset for graph construction requirements.'),
    nbf.v4.new_code_cell('import json, os\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndata_dir = r"d:\\NUST\\6th sem\\Machine Learning\\project\\open-data\\data"'),
    nbf.v4.new_markdown_cell('## 1. Load Match List (La Liga 2015/16)'),
    nbf.v4.new_code_cell('with open(os.path.join(data_dir, "competitions.json"), encoding="utf-8") as f:\n    comps = json.load(f)\nla_liga_1516 = [c for c in comps if c["competition_id"] == 11 and c["season_id"] == 27]\nprint(f"La Liga 15/16 Found: {bool(la_liga_1516)}")'),
    nbf.v4.new_markdown_cell('## 2. Inspect Sample Match and Event Schema'),
    nbf.v4.new_code_cell('matches_file = os.path.join(data_dir, "matches", "11", "27.json")\nwith open(matches_file, encoding="utf-8") as f:\n    matches = json.load(f)\nprint(f"Total matches in season: {len(matches)}")\nsample_match_id = matches[0]["match_id"]\n\nwith open(os.path.join(data_dir, "events", f"{sample_match_id}.json"), encoding="utf-8") as f:\n    events = json.load(f)\nprint(f"Events in sample match: {len(events)}")\n\nfor e in events:\n    if "location" in e:\n        print("Sample Event:", e["type"]["name"], e["location"])\n        break'),
    nbf.v4.new_markdown_cell('## 3. Check for Three-Sixty Data (Freeze Frames)'),
    nbf.v4.new_code_cell('ts_file = os.path.join(data_dir, "three-sixty", f"{sample_match_id}.json")\nhas_ts = os.path.exists(ts_file)\nprint(f"Three-sixty available: {has_ts}\\nNOTE: If False, we must fallback to position estimation for off-ball players.")'),
    nbf.v4.new_markdown_cell('## 4. Profile Season (Events & Types)'),
    nbf.v4.new_code_cell('types_counter = {}\nfor m in matches:\n    ev_file = os.path.join(data_dir, "events", f"{m[\'match_id\']}.json")\n    if os.path.exists(ev_file):\n        with open(ev_file, encoding="utf-8") as f:\n            evs = json.load(f)\n        for e in evs:\n            t = e["type"]["name"]\n            types_counter[t] = types_counter.get(t, 0) + 1\n\ntop_types = sorted(types_counter.items(), key=lambda x: x[1], reverse=True)[:5]\nprint("Top Event Types:", top_types)')
]

with open('notebooks/01_data_exploration.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 01 created successfully.")
