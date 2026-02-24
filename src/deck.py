"""
Layer 1 — Biomek i7 Deck Layout Engine
=======================================
Approximated from Beckman Coulter documentation and ANSI/SLAS SBS standards.

The Biomek i7 has a 9×5 grid of ALP (Automated Labware Positioner) positions.
Each position accommodates one SBS-footprint labware (127.76 × 85.48 mm).

Approximated deck coordinate system (origin = front-left corner):
  - X axis: left→right (along 9-position axis), ~140 mm pitch
  - Y axis: front→back (along 5-position axis), ~100 mm pitch

Grid positions are labeled 1–45 (row-major, left-to-right, front-to-back).
Physical coordinates are the CENTER of each deck position in mm.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import math

# ── Deck geometry (approximated from Beckman i7 specs + SBS standard) ────────
# 45 positions in a 9-column × 5-row grid
# SBS plate: 127.76 × 85.48 mm; pitch adds ~12 mm clearance each side
DECK_COLS = 9
DECK_ROWS = 5
PITCH_X = 140.0   # mm center-to-center between columns
PITCH_Y = 100.0   # mm center-to-center between rows
ORIGIN_X = 70.0   # mm – x-center of position (1,1) from deck left edge
ORIGIN_Y = 50.0   # mm – y-center of position (1,1) from deck front edge

# SBS 96-well plate internal geometry
WELL_COLS = 12
WELL_ROWS = 8
WELL_PITCH = 9.0       # mm center-to-center
PLATE_OFFSET_X = 14.38  # mm from plate left edge to A1 well center (ANSI/SLAS 4-2004)
PLATE_OFFSET_Y = 11.24  # mm from plate top edge to A1 well center


def deck_position_to_xy(col: int, row: int) -> Tuple[float, float]:
    """Return (x, y) center-mm of a deck grid position (1-indexed col, row)."""
    x = ORIGIN_X + (col - 1) * PITCH_X
    y = ORIGIN_Y + (row - 1) * PITCH_Y
    return x, y


def position_index_to_colrow(pos: int) -> Tuple[int, int]:
    """Convert 1-based position index (1–45) to (col, row)."""
    pos -= 1
    return (pos % DECK_COLS) + 1, (pos // DECK_COLS) + 1


def position_index_to_xy(pos: int) -> Tuple[float, float]:
    col, row = position_index_to_colrow(pos)
    return deck_position_to_xy(col, row)


def arm_travel_distance(pos_a: int, pos_b: int) -> float:
    """
    Euclidean distance (mm) the arm travels between two deck positions.
    Used as the spatial cost term added to the CVRP distance matrix (Layer 3).
    """
    xa, ya = position_index_to_xy(pos_a)
    xb, yb = position_index_to_xy(pos_b)
    return math.hypot(xb - xa, yb - ya)


def well_center_xy(deck_pos: int, well_row: int, well_col: int) -> Tuple[float, float]:
    """
    Absolute (x, y) mm of a specific well within a plate at a deck position.
    well_row: 0-indexed (A=0 … H=7)
    well_col: 0-indexed (1=0 … 12=11)
    """
    px, py = position_index_to_xy(deck_pos)
    # Plate center → A1 offset (plate is SBS: 127.76 × 85.48 mm)
    plate_x0 = px - 127.76 / 2 + PLATE_OFFSET_X
    plate_y0 = py - 85.48 / 2 + PLATE_OFFSET_Y
    wx = plate_x0 + well_col * WELL_PITCH
    wy = plate_y0 + well_row * WELL_PITCH
    return wx, wy


# ── Labware catalogue ─────────────────────────────────────────────────────────
LABWARE_TYPES = {
    "96well_plate":   {"label": "96-Well Plate",   "cols": 12, "rows": 8,  "color": "#4a9eff", "icon": "🧫"},
    "384well_plate":  {"label": "384-Well Plate",  "cols": 24, "rows": 16, "color": "#7b68ee", "icon": "🧫"},
    "tiprack_50ul":   {"label": "Tip Rack 50µL",   "cols": 12, "rows": 8,  "color": "#50c878", "icon": "🔩"},
    "tiprack_200ul":  {"label": "Tip Rack 200µL",  "cols": 12, "rows": 8,  "color": "#3cb371", "icon": "🔩"},
    "tiprack_1000ul": {"label": "Tip Rack 1000µL", "cols": 12, "rows": 8,  "color": "#2e8b57", "icon": "🔩"},
    "reservoir_12":   {"label": "12-Col Reservoir", "cols": 12, "rows": 1, "color": "#ffa07a", "icon": "🧪"},
    "reservoir_1":    {"label": "Single Reservoir", "cols": 1,  "rows": 1,  "color": "#ff6347", "icon": "🧪"},
    "waste_trough":   {"label": "Waste Trough",    "cols": 1,  "rows": 1,  "color": "#808080", "icon": "🗑️"},
    "tube_rack_15ml": {"label": "Tube Rack 15mL",  "cols": 6,  "rows": 4,  "color": "#dda0dd", "icon": "🧬"},
    "tube_rack_50ml": {"label": "Tube Rack 50mL",  "cols": 4,  "rows": 2,  "color": "#da70d6", "icon": "🧬"},
}


@dataclass
class LabwareItem:
    labware_id: str          # unique ID within a deck layout
    labware_type: str        # key into LABWARE_TYPES
    deck_position: int       # 1–45
    role: str = "generic"    # e.g. "source", "destination", "tips_clean", "tips_dirty", "waste", "reservoir"
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = LABWARE_TYPES[self.labware_type]["label"]

    @property
    def xy(self) -> Tuple[float, float]:
        return position_index_to_xy(self.deck_position)

    @property
    def cols(self) -> int:
        return LABWARE_TYPES[self.labware_type]["cols"]

    @property
    def rows(self) -> int:
        return LABWARE_TYPES[self.labware_type]["rows"]

    def to_dict(self) -> dict:
        lw = LABWARE_TYPES[self.labware_type]
        col, row = position_index_to_colrow(self.deck_position)
        x, y = self.xy
        return {
            "labware_id": self.labware_id,
            "labware_type": self.labware_type,
            "deck_position": self.deck_position,
            "deck_col": col,
            "deck_row": row,
            "x_mm": round(x, 2),
            "y_mm": round(y, 2),
            "role": self.role,
            "label": self.label,
            "display_label": lw["label"],
            "color": lw["color"],
            "icon": lw["icon"],
            "well_cols": lw["cols"],
            "well_rows": lw["rows"],
        }


@dataclass
class DeckLayout:
    name: str = "Untitled Layout"
    positions: Dict[int, Optional[LabwareItem]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize all 45 positions as empty
        for i in range(1, DECK_COLS * DECK_ROWS + 1):
            if i not in self.positions:
                self.positions[i] = None

    def place(self, item: LabwareItem) -> bool:
        """Place labware on the deck. Returns False if position occupied."""
        pos = item.deck_position
        if pos < 1 or pos > 45:
            raise ValueError(f"Position {pos} out of range 1–45")
        if self.positions[pos] is not None:
            return False
        self.positions[pos] = item
        return True

    def remove(self, deck_position: int) -> Optional[LabwareItem]:
        item = self.positions.get(deck_position)
        self.positions[deck_position] = None
        return item

    def move(self, from_pos: int, to_pos: int) -> bool:
        if self.positions[to_pos] is not None:
            return False
        item = self.positions[from_pos]
        if item is None:
            return False
        item.deck_position = to_pos
        self.positions[to_pos] = item
        self.positions[from_pos] = None
        return True

    def get_by_role(self, role: str) -> list:
        return [item for item in self.positions.values()
                if item is not None and item.role == role]

    def distance_matrix_mm(self) -> Dict[Tuple[int, int], float]:
        """Pre-compute arm travel distances (mm) between all occupied positions."""
        occupied = [pos for pos, item in self.positions.items() if item is not None]
        dm = {}
        for a in occupied:
            for b in occupied:
                dm[(a, b)] = arm_travel_distance(a, b)
        return dm

    def to_dict(self) -> dict:
        grid = []
        for row in range(1, DECK_ROWS + 1):
            for col in range(1, DECK_COLS + 1):
                pos = (row - 1) * DECK_COLS + col
                item = self.positions.get(pos)
                x, y = deck_position_to_xy(col, row)
                grid.append({
                    "position": pos,
                    "col": col,
                    "row": row,
                    "x_mm": round(x, 2),
                    "y_mm": round(y, 2),
                    "occupied": item is not None,
                    "labware": item.to_dict() if item else None,
                })
        return {
            "name": self.name,
            "deck_cols": DECK_COLS,
            "deck_rows": DECK_ROWS,
            "pitch_x": PITCH_X,
            "pitch_y": PITCH_Y,
            "grid": grid,
            "labware_types": LABWARE_TYPES,
        }
