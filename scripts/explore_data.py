import json
import os
import pandas as pd
from statsbombpy import sb
import matplotlib.pyplot as plt

def main():
    print("--- 1.5a: Loading La Liga 2015/16 Match List ---")
    data_dir = r"d:\NUST\6th sem\Machine Learning\project\open-data\data"

    # From open-data/data/competitions.json, La Liga is competition_id=11, season_id=27
    with open(os.path.join(data_dir, "competitions.json"), encoding='utf-8') as f:
        comps = json.load(f)
        
    la_liga_1516 = [c for c in comps if c['competition_id'] == 11 and c['season_id'] == 27]
    print(f"La Liga 2015/16 found: {bool(la_liga_1516)}")

    # Load matches
    matches_file = os.path.join(data_dir, "matches", "11", "27.json")
    if os.path.exists(matches_file):
        with open(matches_file, encoding='utf-8') as f:
            matches = json.load(f)
        print(f"Total matches found: {len(matches)}")
        sample_match_id = matches[0]['match_id']
        print(f"Sample match ID to use: {sample_match_id}")
    else:
        print("Matches file not found!")
        return

    print("\n--- 1.5b: Inspect Event Schema ---")
    events_file = os.path.join(data_dir, "events", f"{sample_match_id}.json")
    if os.path.exists(events_file):
        with open(events_file, encoding='utf-8') as f:
            events = json.load(f)
        print(f"Total events in sample match: {len(events)}")
        
        # Look at the first valid event that has location
        for e in events:
            if 'location' in e:
                print("Sample event with location keys:", list(e.keys()))
                print(f"Location example: {e['location']}")
                print(f"Type: {e['type']['name']}")
                break
    else:
        print("Events file not found!")
    
    print("\n--- 1.5c: Check three-sixty (freeze-frame) availability ---")
    three_sixty_dir = os.path.join(data_dir, "three-sixty")
    ts_file = os.path.join(three_sixty_dir, f"{sample_match_id}.json")
    has_ts = os.path.exists(ts_file)
    print(f"Three-sixty data available for sample match: {has_ts}")
    if has_ts:
        with open(ts_file, encoding='utf-8') as f:
            ts_data = json.load(f)
        print(f"Number of freeze frames in sample match: {len(ts_data)}")
        
    print("\n--- 1.5d: Profile Full Season ---")
    event_counts = []
    types_counter = {}
    matches_dir = os.path.join(data_dir, "events")
    
    for match in matches:
        mid = match['match_id']
        ef = os.path.join(matches_dir, f"{mid}.json")
        if os.path.exists(ef):
            with open(ef, encoding='utf-8') as f:
                evs = json.load(f)
            event_counts.append(len(evs))
            for e in evs:
                t = e['type']['name']
                types_counter[t] = types_counter.get(t, 0) + 1
    
    if event_counts:
        print(f"Avg events per match: {sum(event_counts) // len(event_counts)}")
        print("Top 5 event types:")
        top_types = sorted(types_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        for t, c in top_types:
            print(f"  {t}: {c}")

    print("\n--- 1.5e: Data Visualization readiness ---")
    print("Data structures verified. We can use matplotlib to plot the pitch and events.")

if __name__ == '__main__':
    main()
