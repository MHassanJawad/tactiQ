import os
import json
import torch
from tqdm import tqdm
import warnings
from typing import Iterator, Tuple, Any, Dict

# Use relative imports properly as a script or module
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.graph_builder import GraphBuilder

warnings.filterwarnings("ignore")

# We map the most common StatsBomb events to integer classes so the GNN can learn to predict them
EVENT_CLASSES = {
    "Pass": 0, "Ball Receipt*": 1, "Carry": 2, "Pressure": 3, 
    "Duel": 4, "Clearance": 5, "Foul Committed": 6, 
    "Interception": 7, "Dribble": 8, "Shot": 9
}


def _iter_consecutive_pairs_streaming(events_file: str) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Stream-parse a StatsBomb events JSON array without loading the full file into RAM.
    Yields (events[i], events[i+1]) for i = 0 .. n-2 — same pairs as the old json.load loop.
    """
    import ijson

    with open(events_file, "rb") as f:
        it = ijson.items(f, "item")
        try:
            prev = next(it)
        except StopIteration:
            return
        for curr in it:
            yield prev, curr
            prev = curr


def _iter_consecutive_pairs_json_load(events_file: str) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Fallback: load entire file (high RAM). Prefer streaming."""
    with open(events_file, encoding="utf-8") as f:
        events = json.load(f)
    for i in range(len(events) - 1):
        yield events[i], events[i + 1]


def iter_consecutive_event_pairs(events_file: str) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Prefer ijson streaming to avoid MemoryError on very large match event files.
    Falls back to json.load only if ijson is not installed.
    """
    try:
        import ijson  # noqa: F401
    except ImportError:
        return _iter_consecutive_pairs_json_load(events_file)
    return _iter_consecutive_pairs_streaming(events_file)


def build_datasets(data_dir, out_dir, connect_radius=25.0, train_match_count=15):
    """
    Parses full match files to generate thousands of PyG graphs.
    Saves a smaller 'train' subset and a 'full' set for the FAISS index.
    """
    matches_file = os.path.join(data_dir, "matches", "11", "27.json")
    if not os.path.exists(matches_file):
        print(f"Error: Could not find matches file at {matches_file}")
        return

    with open(matches_file, encoding='utf-8') as f:
        matches = json.load(f)
        
    builder = GraphBuilder(connect_radius=connect_radius)
    
    train_graphs = []
    full_graphs = []
    
    # Select the first N matches for the training set to keep CPU training fast
    train_match_ids = set([m['match_id'] for m in matches[:train_match_count]])
    
    print(f"Processing {len(matches)} total matches...")
    print(f"({train_match_count} matches reserved for GNN training, all 38 for FAISS index)")

    try:
        import ijson  # noqa: F401
        print("Using ijson streaming for event files (low RAM).")
    except ImportError:
        print("Warning: ijson not installed; using json.load (may MemoryError on huge files). pip install ijson")

    skipped_matches = []

    for match in tqdm(matches, desc="Parsing Matches"):
        mid = match['match_id']
        events_file = os.path.join(data_dir, "events", f"{mid}.json")

        if not os.path.exists(events_file):
            continue

        try:
            pair_iter = iter_consecutive_event_pairs(events_file)
            for curr_ev, next_ev in pair_iter:
                # We can only build spatial graphs if the event has a location
                if "location" not in curr_ev:
                    continue

                next_type = next_ev.get("type", {}).get("name", "Unknown")
                # Class 10 is 'Other / Unknown'
                label = EVENT_CLASSES.get(next_type, 10)

                try:
                    graph = builder.build_from_event(curr_ev, next_action_label=label)
                except Exception:
                    continue

                full_graphs.append(graph)
                if mid in train_match_ids:
                    train_graphs.append(graph)

        except MemoryError:
            skipped_matches.append(mid)
            print(f"\nMemoryError on match_id={mid} — skipped file {events_file}. Install ijson: pip install ijson")
            continue
        except json.JSONDecodeError as e:
            skipped_matches.append(mid)
            print(f"\nJSONDecodeError match_id={mid}: {e}")
            continue
                
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "la_liga_2015_16_train.pt")
    full_path = os.path.join(out_dir, "la_liga_2015_16_full.pt")
    
    if skipped_matches:
        print(f"\nSkipped {len(skipped_matches)} match(es) due to errors (see logs above).")

    print(f"\nFinal Statistics:")
    print(f"  Training Set: {len(train_graphs):,} graphs")
    print(f"  Full Index Set: {len(full_graphs):,} graphs")
    
    print(f"\nSaving to disk...")
    torch.save(train_graphs, train_path)
    torch.save(full_graphs, full_path)
    
    print("Done! Dataset fully cached.")

if __name__ == '__main__':
    data_directory = r"d:\NUST\6th sem\Machine Learning\project\open-data\data"
    output_directory = r"d:\NUST\6th sem\Machine Learning\project\graphs"
    
    build_datasets(data_dir=data_directory, out_dir=output_directory)
