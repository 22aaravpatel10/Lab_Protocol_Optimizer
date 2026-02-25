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
    item = LabwareItem(
        labware_id=d["labware_id"],
        labware_type=d["labware_type"],
        deck_position=int(d["deck_position"]),
        role=d.get("role", "generic"),
        label=d.get("label", ""),
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
        item = LabwareItem(**item_data)
        _deck.place(item)
    return jsonify({"ok": True, "deck": _deck.to_dict()})


@app.route("/api/deck/presets", methods=["GET"])
def list_presets():
    return jsonify({"presets": list(_get_presets().keys())})


def _get_presets():
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
        src_cols=src_cols, dst_cols=dst_cols
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

    jobs = task_matrix_to_jobs(T, "source", "destination", src_pos, dst_pos,
                                src_cols=src_cols, dst_cols=dst_cols)
    report = run_optimization(jobs, alpha=alpha, beta=beta, time_limit_s=time_limit,
                               n_channels=n_channels)
    summary = report.summary()
    summary["jobs"] = [j.to_dict() for j in jobs]
    summary["task_matrix"] = T.tolist()
    dm = report.distance_matrix
    summary["distance_matrix"] = {"data": dm.tolist(), "n": int(dm.shape[0])}
    return jsonify(summary)


def _find_labware(labware_id: str):
    for item in _deck.positions.values():
        if item and item.labware_id == labware_id:
            return item
    return None


if __name__ == "__main__":
    app.run(debug=True, port=5050)
