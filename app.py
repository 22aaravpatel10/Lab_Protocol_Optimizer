"""
Lab Protocol Optimizer — Flask Application
==========================================
Serves the three-layer UI and provides JSON API endpoints.
"""
from __future__ import annotations
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from typing import Optional
import numpy as np

from src.deck import (
    DeckLayout, LabwareItem, LABWARE_TYPES,
    DECK_COLS, DECK_ROWS, PITCH_X, PITCH_Y,
    position_index_to_xy, arm_travel_distance
)
from src.protocol import (
    Protocol, StepType, build_media_change_protocol
)
from src.optimizer import (
    Job, task_matrix_to_jobs, build_distance_matrix,
    run_optimization, PIPETTE_CAPACITY
)

app = Flask(__name__)

# ── In-memory state (single-session) ─────────────────────────────────────────
_deck = DeckLayout(name="Biomek i7 Layout")
_protocol: Optional[Protocol] = None


# ── Page ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Layer 1: Deck API ─────────────────────────────────────────────────────────

@app.route("/api/deck", methods=["GET"])
def get_deck():
    return jsonify(_deck.to_dict())


@app.route("/api/deck/place", methods=["POST"])
def place_labware():
    d = request.json
    sample_map_raw = d.get("sample_map", {})
    sample_map = {int(k): v for k, v in sample_map_raw.items()} if sample_map_raw else {}
    item = LabwareItem(
        labware_id=d["labware_id"],
        labware_type=d["labware_type"],
        deck_position=int(d["deck_position"]),
        role=d.get("role", "generic"),
        label=d.get("label", ""),
        sample_map=sample_map,
    )
    ok = _deck.place(item)
    if not ok:
        return jsonify({"error": f"Position {d['deck_position']} already occupied"}), 409
    return jsonify({"ok": True, "deck": _deck.to_dict()})


@app.route("/api/deck/remove", methods=["POST"])
def remove_labware():
    pos = int(request.json["deck_position"])
    item = _deck.remove(pos)
    return jsonify({"ok": True, "removed": item.to_dict() if item else None,
                    "deck": _deck.to_dict()})


@app.route("/api/deck/move", methods=["POST"])
def move_labware():
    d = request.json
    ok = _deck.move(int(d["from_pos"]), int(d["to_pos"]))
    if not ok:
        return jsonify({"error": "Move failed — target occupied or source empty"}), 409
    return jsonify({"ok": True, "deck": _deck.to_dict()})


@app.route("/api/deck/clear", methods=["POST"])
def clear_deck():
    global _deck
    _deck = DeckLayout(name="Biomek i7 Layout")
    return jsonify({"ok": True, "deck": _deck.to_dict()})


@app.route("/api/deck/sample_map", methods=["POST"])
def update_sample_map():
    """
    Set or update sample names for wells on a labware item.
    Body: { deck_position: int, sample_map: {well_index: sample_name} }
    """
    d = request.json
    pos = int(d["deck_position"])
    item = _deck.positions.get(pos)
    if item is None:
        return jsonify({"error": f"No labware at position {pos}"}), 404
    new_map = {int(k): v for k, v in d.get("sample_map", {}).items()}
    item.sample_map.update(new_map)
    return jsonify({"ok": True, "labware": item.to_dict()})


@app.route("/api/deck/sample_map/clear", methods=["POST"])
def clear_sample_map():
    """Clear all sample names from a labware position."""
    pos = int(request.json["deck_position"])
    item = _deck.positions.get(pos)
    if item is None:
        return jsonify({"error": f"No labware at position {pos}"}), 404
    item.sample_map.clear()
    return jsonify({"ok": True, "labware": item.to_dict()})


@app.route("/api/deck/distances", methods=["GET"])
def deck_distances():
    """Return arm travel distance matrix between all occupied positions."""
    dm = _deck.distance_matrix_mm()
    result = {f"{a},{b}": round(v, 2) for (a, b), v in dm.items()}
    return jsonify(result)


@app.route("/api/deck/presets/<preset_name>", methods=["POST"])
def load_preset(preset_name: str):
    """Load a preset deck layout."""
    global _deck
    _deck = DeckLayout(name=f"Preset: {preset_name}")
    presets = _get_presets()
    if preset_name not in presets:
        return jsonify({"error": f"Unknown preset '{preset_name}'"}), 404
    for item_data in presets[preset_name]:
        # Convert sample_map keys to int (JSON/dict may store them as strings)
        d = dict(item_data)
        if "sample_map" in d and d["sample_map"]:
            d["sample_map"] = {int(k): v for k, v in d["sample_map"].items()}
        item = LabwareItem(**d)
        _deck.place(item)
    return jsonify({"ok": True, "deck": _deck.to_dict()})


@app.route("/api/deck/presets", methods=["GET"])
def list_presets():
    return jsonify({"presets": list(_get_presets().keys())})


def _get_presets():
    # Cherry-pick source plate sample map:
    # 8 compounds across 24 wells (A1-C8), 3 concentration stocks per compound
    cherry_src_map = {}
    compounds = ["Erlotinib","Gefitinib","Imatinib","Sorafenib",
                 "Vemurafenib","Crizotinib","Osimertinib","Lapatinib"]
    concs = ["10mM","1mM","100µM"]
    for ci, cmp in enumerate(compounds):
        for si, stock in enumerate(concs):
            well_idx = ci * 3 + si   # A1..A3=Erlotinib, A4..A6=Gefitinib, etc.
            cherry_src_map[well_idx] = f"{cmp} {stock}"

    # Destination plate well map: 8 compounds × 10 dose points (columns 1-10),
    # duplicated in rows A-D (rep 1) and rows E-H (rep 2)
    cherry_dst_map = {}
    dose_labels = ["10µM","3µM","1µM","300nM","100nM","30nM","10nM","3nM","1nM","0nM"]
    for ci, cmp in enumerate(compounds):
        for di, dose in enumerate(dose_labels):
            # Rep 1: rows A-D (rows 0-3), compound→row, dose→col
            row1 = ci // 2
            col1 = (ci % 2) * 10 + di   # compounds pair into double-row cols
            # simpler: row = compound index mod 4, col pair
            row1 = ci % 4
            col1 = (ci // 4) * 10 + di
            idx1 = row1 * 12 + col1
            cherry_dst_map[idx1] = f"{cmp}@{dose} rep1"
            # Rep 2: rows E-H (rows 4-7)
            row2 = row1 + 4
            idx2 = row2 * 12 + col1
            cherry_dst_map[idx2] = f"{cmp}@{dose} rep2"

    return {
        "media_change_96": [
            {"labware_id": "dirty_tips",  "labware_type": "tiprack_200ul",  "deck_position": 1,  "role": "tips_dirty",  "label": "Dirty Tips"},
            {"labware_id": "clean_tips",  "labware_type": "tiprack_200ul",  "deck_position": 2,  "role": "tips_clean",  "label": "Clean Tips"},
            {"labware_id": "waste",       "labware_type": "waste_trough",   "deck_position": 3,  "role": "waste",       "label": "Waste"},
            {"labware_id": "reservoir",   "labware_type": "reservoir_12",   "deck_position": 10, "role": "reservoir",   "label": "Fresh Media"},
            {"labware_id": "source",      "labware_type": "96well_plate",   "deck_position": 19, "role": "source",      "label": "Source (Old Media)"},
            {"labware_id": "destination", "labware_type": "96well_plate",   "deck_position": 23, "role": "destination", "label": "Destination"},
        ],
        "serial_dilution": [
            {"labware_id": "tips_a",      "labware_type": "tiprack_200ul",  "deck_position": 1,  "role": "tips_clean",  "label": "Tips A"},
            {"labware_id": "tips_b",      "labware_type": "tiprack_200ul",  "deck_position": 2,  "role": "tips_clean",  "label": "Tips B"},
            {"labware_id": "waste",       "labware_type": "waste_trough",   "deck_position": 3,  "role": "waste",       "label": "Waste"},
            {"labware_id": "diluent",     "labware_type": "reservoir_12",   "deck_position": 10, "role": "reservoir",   "label": "Diluent"},
            {"labware_id": "compound",    "labware_type": "tube_rack_15ml", "deck_position": 11, "role": "source",      "label": "Compound"},
            {"labware_id": "plate1",      "labware_type": "96well_plate",   "deck_position": 20, "role": "destination", "label": "Assay Plate 1"},
            {"labware_id": "plate2",      "labware_type": "96well_plate",   "deck_position": 21, "role": "destination", "label": "Assay Plate 2"},
        ],
        "dose_response_cherry_pick": [
            # Tips
            {"labware_id": "tips1",    "labware_type": "tiprack_200ul",  "deck_position": 1,  "role": "tips_clean", "label": "Tips 1"},
            {"labware_id": "tips2",    "labware_type": "tiprack_200ul",  "deck_position": 2,  "role": "tips_clean", "label": "Tips 2"},
            {"labware_id": "tips3",    "labware_type": "tiprack_200ul",  "deck_position": 3,  "role": "tips_clean", "label": "Tips 3"},
            # Waste
            {"labware_id": "waste",    "labware_type": "waste_trough",   "deck_position": 4,  "role": "waste",      "label": "Waste"},
            # DMSO diluent reservoir
            {"labware_id": "dmso",     "labware_type": "reservoir_12",   "deck_position": 5,  "role": "reservoir",  "label": "DMSO Diluent"},
            # Compound stock source plate (8 compounds × 3 stocks = 24 wells used)
            {"labware_id": "stocks",   "labware_type": "96well_plate",   "deck_position": 14, "role": "source",
             "label": "Compound Stocks", "sample_map": {str(k): v for k,v in cherry_src_map.items()}},
            # Intermediate dilution plate (acoustic-to-acoustic, mid-deck)
            {"labware_id": "intermed", "labware_type": "96well_plate",   "deck_position": 23, "role": "generic",    "label": "Intermediate Dilutions"},
            # Two assay destination plates (far side of deck)
            {"labware_id": "assay1",   "labware_type": "96well_plate",   "deck_position": 33, "role": "destination",
             "label": "Assay Plate 1 (rep1)", "sample_map": {str(k): v for k,v in cherry_dst_map.items() if "rep1" in v}},
            {"labware_id": "assay2",   "labware_type": "96well_plate",   "deck_position": 34, "role": "destination",
             "label": "Assay Plate 2 (rep2)", "sample_map": {str(k): v for k,v in cherry_dst_map.items() if "rep2" in v}},
        ],
    }


# ── Layer 2: Protocol API ─────────────────────────────────────────────────────

@app.route("/api/protocol/media_change", methods=["POST"])
def build_media_change():
    """Build and simulate a media-change protocol from the current deck."""
    global _protocol
    d = request.json or {}

    sources = _deck.get_by_role("source")
    dests   = _deck.get_by_role("destination")
    dirty   = _deck.get_by_role("tips_dirty")
    clean   = _deck.get_by_role("tips_clean")
    wastes  = _deck.get_by_role("waste")
    reservoirs = _deck.get_by_role("reservoir")

    errors = []
    if not sources:   errors.append("No source plate assigned (role='source')")
    if not dests:     errors.append("No destination plate assigned (role='destination')")
    if not dirty:     errors.append("No dirty tip rack assigned (role='tips_dirty')")
    if not clean:     errors.append("No clean tip rack assigned (role='tips_clean')")
    if not wastes:    errors.append("No waste assigned (role='waste')")
    if not reservoirs: errors.append("No reservoir assigned (role='reservoir')")
    if errors:
        return jsonify({"errors": errors}), 400

    n_wells = int(d.get("n_wells", 96))
    asp_vol = float(d.get("aspirate_vol", 200))
    disp_vol = float(d.get("dispense_vol", 200))
    n_channels = int(d.get("n_channels", 8))

    # Build well groups sized to the pipette head capacity
    well_groups = [
        list(range(i, min(i + n_channels, n_wells)))
        for i in range(0, n_wells, n_channels)
    ]

    _protocol = build_media_change_protocol(
        deck=_deck,
        source_id=sources[0].labware_id,
        dest_id=dests[0].labware_id,
        dirty_tips_id=dirty[0].labware_id,
        clean_tips_id=clean[0].labware_id,
        waste_id=wastes[0].labware_id,
        reservoir_id=reservoirs[0].labware_id,
        well_groups=well_groups,
        aspirate_vol=asp_vol,
        dispense_vol=disp_vol,
        n_channels=n_channels,
    )

    state, step_log = _protocol.simulate()
    return jsonify({
        "protocol": _protocol.to_dict(),
        "simulation": {
            "step_log": step_log,
            "final_state": state.to_dict(),
            "total_time_s": round(state.elapsed_time_s, 2),
            "total_travel_mm": round(state.total_travel_mm, 2),
        }
    })


@app.route("/api/protocol/custom", methods=["POST"])
def build_custom_protocol():
    """Build a protocol from a step list sent by the frontend."""
    global _protocol
    data = request.json
    proto_name = data.get("name", "Custom Protocol")
    steps_raw = data.get("steps", [])

    _protocol = Protocol(proto_name, _deck)

    for s in steps_raw:
        stype = s.get("type")
        if stype == "PICK_TIPS":
            _protocol.pick_tips(s["tip_labware_id"], s["tip_positions"], s.get("description", ""))
        elif stype == "ASPIRATE":
            _protocol.aspirate(s["labware_id"], s["well_indices"], s["volume_ul"],
                               s.get("flow_rate", 100.0), s.get("description", ""))
        elif stype == "DISPENSE":
            _protocol.dispense(s["labware_id"], s["well_indices"], s["volume_ul"],
                               s.get("flow_rate", 150.0), s.get("description", ""))
        elif stype == "DISCARD_TIPS":
            _protocol.discard_tips(s["tip_labware_id"], s["waste_labware_id"],
                                   s["tip_positions"], s.get("description", ""))
        elif stype == "PARK_TIPS":
            _protocol.park_tips(s["tip_labware_id"], s["tip_positions"], s.get("description", ""))
        elif stype == "MIX":
            _protocol.mix(s["labware_id"], s["well_indices"], s["volume_ul"],
                          s.get("n_cycles", 3), s.get("description", ""))
        elif stype == "DELAY":
            _protocol.delay(s["delay_s"], s.get("description", ""))

    state, step_log = _protocol.simulate()
    return jsonify({
        "protocol": _protocol.to_dict(),
        "simulation": {
            "step_log": step_log,
            "final_state": state.to_dict(),
            "total_time_s": round(state.elapsed_time_s, 2),
            "total_travel_mm": round(state.total_travel_mm, 2),
        }
    })


@app.route("/api/protocol/simulate", methods=["POST"])
def simulate_protocol():
    """Re-simulate the current protocol."""
    if _protocol is None:
        return jsonify({"error": "No protocol loaded"}), 404
    state, step_log = _protocol.simulate()
    return jsonify({
        "simulation": {
            "step_log": step_log,
            "final_state": state.to_dict(),
            "total_time_s": round(state.elapsed_time_s, 2),
            "total_travel_mm": round(state.total_travel_mm, 2),
        }
    })


# ── Layer 3: Optimizer API ────────────────────────────────────────────────────

@app.route("/api/optimize", methods=["POST"])
def optimize():
    """
    Run the CVRP optimizer on a transfer task matrix.
    Body JSON:
      task_matrix: 2D array (n_src × n_dst) of volumes
      src_labware_id: string
      dst_labware_id: string
      alpha: float (plate-level cost weight, default 1.0)
      beta: float (deck travel weight, default 1/300)
      time_limit_s: int (solver time limit, default 10)
    """
    d = request.json
    T = np.array(d["task_matrix"], dtype=float)

    # Resolve deck positions from labware IDs
    src_id = d.get("src_labware_id", "")
    dst_id = d.get("dst_labware_id", "")
    src_item = _find_labware(src_id)
    dst_item = _find_labware(dst_id)

    src_pos = src_item.deck_position if src_item else 19
    dst_pos = dst_item.deck_position if dst_item else 23
    src_cols = src_item.cols if src_item else 12
    dst_cols = dst_item.cols if dst_item else 12

    alpha = float(d.get("alpha", 1.0))
    beta  = float(d.get("beta", 1.0 / 300.0))
    time_limit = int(d.get("time_limit_s", 10))
    n_channels = int(d.get("n_channels", 8))

    jobs = task_matrix_to_jobs(
        T, src_id, dst_id, src_pos, dst_pos,
        src_cols=src_cols, dst_cols=dst_cols,
        src_sample_map=src_item.sample_map if src_item else None,
        dst_sample_map=dst_item.sample_map if dst_item else None,
    )

    if not jobs:
        return jsonify({"error": "Task matrix has no non-zero entries"}), 400

    report = run_optimization(jobs, alpha=alpha, beta=beta, time_limit_s=time_limit,
                               n_channels=n_channels)
    summary = report.summary()

    # Include job list for visualization
    summary["jobs"] = [j.to_dict() for j in jobs]

    # Include distance matrix (as flat list for frontend heatmap)
    dm = report.distance_matrix
    summary["distance_matrix"] = {
        "data": dm.tolist(),
        "n": int(dm.shape[0]),
    }

    return jsonify(summary)


@app.route("/api/optimize/random", methods=["POST"])
def optimize_random():
    """
    Generate a random task matrix and optimize it.
    Useful for benchmarking and demos.
    """
    d = request.json or {}
    n_src = int(d.get("n_src", 96))
    n_dst = int(d.get("n_dst", 96))
    density = float(d.get("density", 0.3))  # fraction of wells with transfers
    vol = float(d.get("volume_ul", 100.0))
    seed = d.get("seed", 42)
    rng = np.random.default_rng(seed)

    T = np.zeros((n_src, n_dst))
    mask = rng.random((n_src, n_dst)) < density
    T[mask] = vol

    src_item = next(iter(item for item in _deck.positions.values()
                         if item and item.role == "source"), None)
    dst_item = next(iter(item for item in _deck.positions.values()
                         if item and item.role == "destination"), None)
    src_pos = src_item.deck_position if src_item else 19
    dst_pos = dst_item.deck_position if dst_item else 23
    src_cols = src_item.cols if src_item else 12
    dst_cols = dst_item.cols if dst_item else 12

    alpha = float(d.get("alpha", 1.0))
    beta  = float(d.get("beta", 1.0 / 300.0))
    time_limit = int(d.get("time_limit_s", 10))
    n_channels = int(d.get("n_channels", 8))

    src_item2 = next((i for i in _deck.positions.values() if i and i.role=="source"), None)
    dst_item2 = next((i for i in _deck.positions.values() if i and i.role=="destination"), None)
    jobs = task_matrix_to_jobs(T, "source", "destination", src_pos, dst_pos,
                                src_cols=src_cols, dst_cols=dst_cols,
                                src_sample_map=src_item2.sample_map if src_item2 else None,
                                dst_sample_map=dst_item2.sample_map if dst_item2 else None)
    report = run_optimization(jobs, alpha=alpha, beta=beta, time_limit_s=time_limit,
                               n_channels=n_channels)
    summary = report.summary()
    summary["jobs"] = [j.to_dict() for j in jobs]
    summary["task_matrix"] = T.tolist()
    dm = report.distance_matrix
    summary["distance_matrix"] = {"data": dm.tolist(), "n": int(dm.shape[0])}
    return jsonify(summary)


@app.route("/api/optimize/cherry_pick", methods=["POST"])
def optimize_cherry_pick():
    """
    Build and optimize a cherry-pick task matrix from the current deck.
    Generates a realistic sparse multi-source → multi-destination transfer map
    using the named samples on the source plate.

    The task matrix is deliberately non-trivial:
    - Each source well maps to 2-4 destination wells (replicates + dose points)
    - Transfers are scattered non-contiguously across both plates
    - This maximises CVRP optimization headroom vs row-major baseline

    Math: the distance matrix D_extended[i,j] is built exactly as in the
    random benchmark — well-adjacency term + deck-travel term. The sparsity
    and non-contiguous structure of cherry-pick tasks is what makes CVRP
    shine vs greedy/LAP: neighbours in the distance matrix are jobs that share
    a nearby source well OR a nearby destination well, and the CVRP solver
    batches these optimally within the n_channels capacity constraint.
    """
    d = request.json or {}
    alpha = float(d.get("alpha", 1.0))
    beta  = float(d.get("beta", 1.0 / 300.0))
    time_limit = int(d.get("time_limit_s", 10))
    n_channels = int(d.get("n_channels", 8))
    seed = d.get("seed", 42)

    src_item = next((i for i in _deck.positions.values() if i and i.role=="source"), None)
    dst_item = next((i for i in _deck.positions.values() if i and i.role=="destination"), None)

    if not src_item or not dst_item:
        return jsonify({"error": "Need source and destination plates on deck"}), 400

    n_src = src_item.cols * src_item.rows
    n_dst = dst_item.cols * dst_item.rows
    rng = np.random.default_rng(seed)

    # Build a sparse cherry-pick task matrix:
    # Each named source well transfers to 2-4 scattered destination wells.
    # Unnamed wells have a 15% chance of being included (background transfers).
    T = np.zeros((n_src, n_dst))

    # Ensure sample_map has integer keys (preset JSON may have stored string keys)
    int_sample_map = {int(k): v for k, v in src_item.sample_map.items()}
    src_item.sample_map = int_sample_map  # fix in-place for job labelling too

    named_wells = list(int_sample_map.keys())
    if named_wells:
        for src_well in named_wells:
            # Each named sample → 2 to 4 destination wells (dose replicates)
            n_hits = rng.integers(2, 5)
            dst_wells = rng.choice(n_dst, size=int(n_hits), replace=False)
            vol = float(rng.choice([50.0, 100.0, 200.0]))
            for dw in dst_wells:
                T[int(src_well), int(dw)] = vol

    # Background sparse transfers from unnamed wells
    unnamed = [i for i in range(n_src) if i not in int_sample_map]
    for src_well in unnamed:
        if rng.random() < 0.15:
            n_hits = rng.integers(1, 3)
            dst_wells = rng.choice(n_dst, size=int(n_hits), replace=False)
            for dw in dst_wells:
                T[int(src_well), int(dw)] = 100.0

    jobs = task_matrix_to_jobs(
        T,
        src_item.labware_id, dst_item.labware_id,
        src_item.deck_position, dst_item.deck_position,
        src_cols=src_item.cols, dst_cols=dst_item.cols,
        src_sample_map=src_item.sample_map,
        dst_sample_map=dst_item.sample_map,
    )

    if not jobs:
        return jsonify({"error": "No transfers generated — add named samples to source plate"}), 400

    report = run_optimization(jobs, alpha=alpha, beta=beta, time_limit_s=time_limit,
                               n_channels=n_channels)
    summary = report.summary()
    summary["jobs"] = [j.to_dict() for j in jobs]
    summary["task_matrix"] = T.tolist()
    dm = report.distance_matrix
    summary["distance_matrix"] = {"data": dm.tolist(), "n": int(dm.shape[0])}
    summary["preset"] = "dose_response_cherry_pick"
    return jsonify(summary)


def _find_labware(labware_id: str):
    for item in _deck.positions.values():
        if item and item.labware_id == labware_id:
            return item
    return None


if __name__ == "__main__":
    app.run(debug=True, port=5050)
