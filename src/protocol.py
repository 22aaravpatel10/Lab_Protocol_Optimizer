"""
Layer 2 — Protocol Sequence Designer
=====================================
Models a stateful liquid handling protocol as a sequence of Steps.

Each Step represents a discrete arm action:
  PICK_TIPS    – pick up a set of tips from a tip rack position
  ASPIRATE     – draw liquid from wells at a labware position
  DISPENSE     – deliver liquid to wells at a labware position
  DISCARD_TIPS – eject tips into waste
  PARK_TIPS    – return tips to the rack (for reuse later)
  MIX          – aspirate+dispense in-place (e.g., for mixing)
  DELAY        – wait N seconds

The ProtocolState tracks:
  - Which tip positions have been used, parked, or are clean
  - Current arm position (deck position index)
  - Time elapsed (ms) with physics-based arm movement cost

The Protocol validates that tip state constraints are satisfied at each step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set
import math

from src.deck import DeckLayout, arm_travel_distance, WELL_PITCH


# ── Step Types ────────────────────────────────────────────────────────────────

class StepType(str, Enum):
    PICK_TIPS    = "PICK_TIPS"
    ASPIRATE     = "ASPIRATE"
    DISPENSE     = "DISPENSE"
    DISCARD_TIPS = "DISCARD_TIPS"
    PARK_TIPS    = "PARK_TIPS"
    MIX          = "MIX"
    DELAY        = "DELAY"


# Arm speed: ~300 mm/s (approximated from Biomek i-series spec sheets)
ARM_SPEED_MM_PER_S = 300.0
# Z-axis move time per operation (aspirate/dispense): ~0.5 s (approximated)
Z_MOVE_TIME_S = 0.5
# Tip pickup time: ~1.0 s
TIP_PICKUP_TIME_S = 1.0
# Tip discard/park time: ~0.8 s
TIP_DISCARD_TIME_S = 0.8
# Default flow rates (µL/s)
DEFAULT_ASPIRATE_RATE = 100.0
DEFAULT_DISPENSE_RATE = 150.0


@dataclass
class WellSelection:
    """A set of wells selected in a single multichannel operation (up to 8)."""
    labware_id: str          # which labware
    well_indices: List[int]  # 0-based flat indices (row-major: row*cols + col)
    volume_ul: float         # volume per well

    @property
    def n_channels(self) -> int:
        return len(self.well_indices)

    def to_dict(self) -> dict:
        return {
            "labware_id": self.labware_id,
            "well_indices": self.well_indices,
            "volume_ul": self.volume_ul,
            "n_channels": self.n_channels,
        }


@dataclass
class ProtocolStep:
    step_id: str
    step_type: StepType
    description: str = ""

    # For PICK_TIPS / PARK_TIPS / DISCARD_TIPS
    tip_labware_id: Optional[str] = None   # which tip rack
    tip_positions: List[int] = field(default_factory=list)  # which tip slots

    # For ASPIRATE / DISPENSE / MIX
    well_selection: Optional[WellSelection] = None
    flow_rate_ul_s: float = DEFAULT_ASPIRATE_RATE

    # For DELAY
    delay_s: float = 0.0

    # Computed fields (set by simulator)
    arm_travel_mm: float = 0.0
    estimated_time_s: float = 0.0
    deck_position: Optional[int] = None    # where the arm goes for this step

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "tip_labware_id": self.tip_labware_id,
            "tip_positions": self.tip_positions,
            "well_selection": self.well_selection.to_dict() if self.well_selection else None,
            "flow_rate_ul_s": self.flow_rate_ul_s,
            "delay_s": self.delay_s,
            "arm_travel_mm": round(self.arm_travel_mm, 2),
            "estimated_time_s": round(self.estimated_time_s, 3),
            "deck_position": self.deck_position,
        }


# ── Tip State ─────────────────────────────────────────────────────────────────

class TipStatus(str, Enum):
    CLEAN   = "clean"
    IN_USE  = "in_use"
    DIRTY   = "dirty"    # used but parked back
    EMPTY   = "empty"    # used and discarded


@dataclass
class TipState:
    """Tracks every tip slot across all tip racks on the deck."""
    slots: Dict[Tuple[str, int], TipStatus] = field(default_factory=dict)

    def initialize_rack(self, labware_id: str, n_tips: int):
        for i in range(n_tips):
            self.slots[(labware_id, i)] = TipStatus.CLEAN

    def pick(self, labware_id: str, positions: List[int]) -> List[str]:
        """Pick tips. Returns list of error messages (empty = success)."""
        errors = []
        for p in positions:
            status = self.slots.get((labware_id, p))
            if status is None:
                errors.append(f"Tip slot {p} on {labware_id} does not exist.")
            elif status == TipStatus.EMPTY:
                errors.append(f"Tip slot {p} on {labware_id} is empty (already discarded).")
            elif status == TipStatus.IN_USE:
                errors.append(f"Tip slot {p} on {labware_id} is already in use.")
            elif status in (TipStatus.CLEAN, TipStatus.DIRTY):
                self.slots[(labware_id, p)] = TipStatus.IN_USE
        return errors

    def discard(self, labware_id: str, positions: List[int]):
        for p in positions:
            self.slots[(labware_id, p)] = TipStatus.EMPTY

    def park(self, labware_id: str, positions: List[int]):
        for p in positions:
            if self.slots.get((labware_id, p)) == TipStatus.IN_USE:
                self.slots[(labware_id, p)] = TipStatus.DIRTY

    def count_by_status(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in TipStatus}
        for s in self.slots.values():
            counts[s.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            f"{lid}:{pos}": status.value
            for (lid, pos), status in self.slots.items()
        }


# ── Protocol State Machine ────────────────────────────────────────────────────

@dataclass
class ProtocolState:
    """Live state during protocol simulation."""
    current_deck_pos: int = 1       # arm starts at position 1
    tips_mounted: bool = False
    mounted_tip_rack: Optional[str] = None
    mounted_tip_positions: List[int] = field(default_factory=list)
    tip_state: TipState = field(default_factory=TipState)
    elapsed_time_s: float = 0.0
    total_travel_mm: float = 0.0
    step_log: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def move_arm_to(self, target_deck_pos: int) -> float:
        """Move arm to target position; return travel distance in mm."""
        dist = arm_travel_distance(self.current_deck_pos, target_deck_pos)
        travel_time = dist / ARM_SPEED_MM_PER_S
        self.elapsed_time_s += travel_time
        self.total_travel_mm += dist
        self.current_deck_pos = target_deck_pos
        return dist

    def to_dict(self) -> dict:
        return {
            "current_deck_pos": self.current_deck_pos,
            "tips_mounted": self.tips_mounted,
            "mounted_tip_rack": self.mounted_tip_rack,
            "mounted_tip_positions": self.mounted_tip_positions,
            "tip_state": self.tip_state.to_dict(),
            "elapsed_time_s": round(self.elapsed_time_s, 3),
            "total_travel_mm": round(self.total_travel_mm, 2),
            "errors": self.errors,
        }


# ── Protocol Class ────────────────────────────────────────────────────────────

class Protocol:
    def __init__(self, name: str, deck: DeckLayout):
        self.name = name
        self.deck = deck
        self.steps: List[ProtocolStep] = []
        self._step_counter = 0

    def _next_id(self) -> str:
        self._step_counter += 1
        return f"step_{self._step_counter:03d}"

    # ── Builder helpers ────────────────────────────────────────────────────

    def pick_tips(self, tip_labware_id: str, positions: List[int], description: str = "") -> ProtocolStep:
        lw = self._get_labware(tip_labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.PICK_TIPS,
            tip_labware_id=tip_labware_id,
            tip_positions=positions,
            deck_position=lw.deck_position if lw else None,
            description=description or f"Pick {len(positions)} tips from {tip_labware_id}",
        )
        self.steps.append(step)
        return step

    def aspirate(self, labware_id: str, well_indices: List[int], volume_ul: float,
                 flow_rate: float = DEFAULT_ASPIRATE_RATE, description: str = "") -> ProtocolStep:
        lw = self._get_labware(labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.ASPIRATE,
            well_selection=WellSelection(labware_id, well_indices, volume_ul),
            flow_rate_ul_s=flow_rate,
            deck_position=lw.deck_position if lw else None,
            description=description or f"Aspirate {volume_ul}µL from {labware_id} wells {well_indices}",
        )
        self.steps.append(step)
        return step

    def dispense(self, labware_id: str, well_indices: List[int], volume_ul: float,
                 flow_rate: float = DEFAULT_DISPENSE_RATE, description: str = "") -> ProtocolStep:
        lw = self._get_labware(labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.DISPENSE,
            well_selection=WellSelection(labware_id, well_indices, volume_ul),
            flow_rate_ul_s=flow_rate,
            deck_position=lw.deck_position if lw else None,
            description=description or f"Dispense {volume_ul}µL to {labware_id} wells {well_indices}",
        )
        self.steps.append(step)
        return step

    def discard_tips(self, tip_labware_id: str, waste_labware_id: str,
                     positions: List[int], description: str = "") -> ProtocolStep:
        lw = self._get_labware(waste_labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.DISCARD_TIPS,
            tip_labware_id=tip_labware_id,
            tip_positions=positions,
            deck_position=lw.deck_position if lw else None,
            description=description or f"Discard tips to waste",
        )
        self.steps.append(step)
        return step

    def park_tips(self, tip_labware_id: str, positions: List[int], description: str = "") -> ProtocolStep:
        lw = self._get_labware(tip_labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.PARK_TIPS,
            tip_labware_id=tip_labware_id,
            tip_positions=positions,
            deck_position=lw.deck_position if lw else None,
            description=description or f"Park tips on {tip_labware_id}",
        )
        self.steps.append(step)
        return step

    def mix(self, labware_id: str, well_indices: List[int], volume_ul: float,
            n_cycles: int = 3, description: str = "") -> ProtocolStep:
        lw = self._get_labware(labware_id)
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.MIX,
            well_selection=WellSelection(labware_id, well_indices, volume_ul),
            deck_position=lw.deck_position if lw else None,
            description=description or f"Mix {n_cycles}× {volume_ul}µL in {labware_id}",
        )
        step._mix_cycles = n_cycles
        self.steps.append(step)
        return step

    def delay(self, seconds: float, description: str = "") -> ProtocolStep:
        step = ProtocolStep(
            step_id=self._next_id(),
            step_type=StepType.DELAY,
            delay_s=seconds,
            description=description or f"Delay {seconds}s",
        )
        self.steps.append(step)
        return step

    # ── Simulation ─────────────────────────────────────────────────────────

    def simulate(self) -> Tuple[ProtocolState, List[dict]]:
        """
        Run a physics-based simulation of the protocol.
        Returns (final_state, step_log) with timing and arm travel per step.
        """
        state = ProtocolState()

        # Initialize tip racks
        for pos, item in self.deck.positions.items():
            if item and "tiprack" in item.labware_type:
                n = item.cols * item.rows
                state.tip_state.initialize_rack(item.labware_id, n)

        step_log = []

        for step in self.steps:
            entry = {"step_id": step.step_id, "type": step.step_type.value,
                     "description": step.description, "errors": []}
            t_start = state.elapsed_time_s

            target = step.deck_position
            if target is not None:
                travel = state.move_arm_to(target)
                step.arm_travel_mm = travel
            else:
                travel = 0.0

            if step.step_type == StepType.PICK_TIPS:
                errs = state.tip_state.pick(step.tip_labware_id, step.tip_positions)
                entry["errors"].extend(errs)
                state.tips_mounted = len(errs) == 0
                state.mounted_tip_rack = step.tip_labware_id
                state.mounted_tip_positions = step.tip_positions
                state.elapsed_time_s += TIP_PICKUP_TIME_S

            elif step.step_type == StepType.ASPIRATE:
                if not state.tips_mounted:
                    entry["errors"].append("No tips mounted — cannot aspirate.")
                ws = step.well_selection
                asp_time = ws.volume_ul / step.flow_rate_ul_s + Z_MOVE_TIME_S
                state.elapsed_time_s += asp_time

            elif step.step_type == StepType.DISPENSE:
                if not state.tips_mounted:
                    entry["errors"].append("No tips mounted — cannot dispense.")
                ws = step.well_selection
                disp_time = ws.volume_ul / step.flow_rate_ul_s + Z_MOVE_TIME_S
                state.elapsed_time_s += disp_time

            elif step.step_type == StepType.DISCARD_TIPS:
                state.tip_state.discard(step.tip_labware_id, step.tip_positions)
                state.tips_mounted = False
                state.mounted_tip_rack = None
                state.mounted_tip_positions = []
                state.elapsed_time_s += TIP_DISCARD_TIME_S

            elif step.step_type == StepType.PARK_TIPS:
                state.tip_state.park(step.tip_labware_id, step.tip_positions)
                state.tips_mounted = False
                state.mounted_tip_rack = None
                state.mounted_tip_positions = []
                state.elapsed_time_s += TIP_DISCARD_TIME_S

            elif step.step_type == StepType.MIX:
                if not state.tips_mounted:
                    entry["errors"].append("No tips mounted — cannot mix.")
                ws = step.well_selection
                n_cycles = getattr(step, "_mix_cycles", 3)
                mix_time = n_cycles * 2 * (ws.volume_ul / step.flow_rate_ul_s + Z_MOVE_TIME_S)
                state.elapsed_time_s += mix_time

            elif step.step_type == StepType.DELAY:
                state.elapsed_time_s += step.delay_s

            step.estimated_time_s = state.elapsed_time_s - t_start
            entry["arm_travel_mm"] = round(travel, 2)
            entry["step_time_s"] = round(step.estimated_time_s, 3)
            entry["cumulative_time_s"] = round(state.elapsed_time_s, 3)
            entry["arm_position"] = state.current_deck_pos
            entry["tip_counts"] = state.tip_state.count_by_status()
            if entry["errors"]:
                state.errors.extend(entry["errors"])
            step_log.append(entry)

        return state, step_log

    def _get_labware(self, labware_id: str):
        for item in self.deck.positions.values():
            if item and item.labware_id == labware_id:
                return item
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


# ── Media Change Protocol Factory ─────────────────────────────────────────────

def build_media_change_protocol(deck: DeckLayout,
                                 source_id: str,
                                 dest_id: str,
                                 dirty_tips_id: str,
                                 clean_tips_id: str,
                                 waste_id: str,
                                 reservoir_id: str,
                                 well_groups: List[List[int]],
                                 aspirate_vol: float = 200.0,
                                 dispense_vol: float = 200.0) -> Protocol:
    """
    Build the canonical media-change protocol:
      For each well group (up to 8 wells):
        1. Pick dirty tips
        2. Aspirate old media from wells
        3. Move to waste, discard liquid (tip still on)
        4. Park dirty tips
        5. Pick clean tips
        6. Aspirate fresh media from reservoir
        7. Dispense to wells
        8. Park clean tips
    """
    proto = Protocol("Media Change", deck)

    for i, wells in enumerate(well_groups):
        group_desc = f"Group {i+1} (wells {wells})"
        tip_slots = list(range(i * len(wells), (i + 1) * len(wells)))

        proto.pick_tips(dirty_tips_id, tip_slots,
                        description=f"[{group_desc}] Pick dirty tips")
        proto.aspirate(source_id, wells, aspirate_vol,
                       description=f"[{group_desc}] Aspirate old media")
        proto.dispense(waste_id, [0] * len(wells), aspirate_vol,
                       description=f"[{group_desc}] Dump waste")
        proto.park_tips(dirty_tips_id, tip_slots,
                        description=f"[{group_desc}] Park dirty tips")
        proto.pick_tips(clean_tips_id, tip_slots,
                        description=f"[{group_desc}] Pick clean tips")
        proto.aspirate(reservoir_id, [0] * len(wells), dispense_vol,
                       description=f"[{group_desc}] Aspirate fresh media")
        proto.dispense(dest_id, wells, dispense_vol,
                       description=f"[{group_desc}] Dispense fresh media")
        proto.park_tips(clean_tips_id, tip_slots,
                        description=f"[{group_desc}] Park clean tips")

    return proto
