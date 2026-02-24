"""
Layer 3 — CVRP Worklist Optimizer
===================================
Implements the optimization from Wu, Wang & Coley (2025) "Optimization of
Robotic Liquid Handling as a Capacitated Vehicle Routing Problem"
DOI: 10.1039/D5DD00233H

Extended with a spatial deck-travel cost term (the paper's Equation 1 gains
an additional component from Layer 1's real arm travel distances).

Paper's formulation recap
--------------------------
- A "job" = (source_well, destination_well) with a transfer volume
- Task matrix T[a,b] = volume to transfer from well a → well b
- Distance matrix D' captures the cost of doing job_i then job_j:
    D'[i,j] = D'_src[i,j] + D'_dst[i,j]
  where D'_src[i,j] = 0 if src wells are same-column adjacent (pipettable
  simultaneously), else 1 (requires separate arm move).
- The 8-channel pipette = vehicle with capacity 8
- One "cycle" = one vehicle route from depot (dummy job 0) and back
- Objective: minimise sum of D'[i,j] * x[i,j,k] over all route edges

Our Extension (Spatial Cost Term)
-----------------------------------
When two consecutive jobs visit *different deck positions*, the arm must
physically travel between those positions. We add this cost:

    D_extended[i,j] = alpha * D'_plate[i,j]  +  beta * D_deck[i,j]

where:
  D'_plate[i,j] = paper's original distance (well adjacency cost, 0 or 1+)
  D_deck[i,j]   = Euclidean mm distance between deck positions of jobs i and j
  alpha          = weight for within-plate cost (default 1.0, ~1 s per unit)
  beta           = weight for deck travel cost (default 1/300, ~1 s per 300mm)

This matches the paper's Eq. 1 structure: total time = pipetting time + arm move time.
Arm moves between deck positions are NOT negligible for cross-deck protocols
(e.g., tips at pos 1, source at pos 5, destination at pos 9).

Solver: Google OR-Tools CVRP with
  - First solution: PATH_CHEAPEST_ARC
  - Metaheuristic:  GUIDED_LOCAL_SEARCH
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import time
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.deck import DeckLayout, arm_travel_distance, WELL_PITCH

# ── Cost weights ─────────────────────────────────────────────────────────────
ALPHA = 1.0          # weight for plate-level well adjacency cost (dimensionless, ~1 s/unit)
BETA  = 1.0 / 300.0  # weight for deck travel (mm→s at 300 mm/s arm speed)
SCALE = 1000         # OR-Tools needs integer costs; multiply floats by SCALE

PIPETTE_CAPACITY = 8  # 8-channel head


# ── Job definition ────────────────────────────────────────────────────────────

@dataclass
class Job:
    """A single liquid transfer: aspirate from src_well → dispense to dst_well."""
    job_id: int
    src_labware_id: str
    dst_labware_id: str
    src_well: int        # 0-based flat index in source plate
    dst_well: int        # 0-based flat index in destination plate
    src_deck_pos: int    # deck position of source labware
    dst_deck_pos: int    # deck position of destination labware
    volume_ul: float
    src_row: int = 0
    src_col: int = 0
    dst_row: int = 0
    dst_col: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "src_labware_id": self.src_labware_id,
            "dst_labware_id": self.dst_labware_id,
            "src_well": self.src_well,
            "dst_well": self.dst_well,
            "src_deck_pos": self.src_deck_pos,
            "dst_deck_pos": self.dst_deck_pos,
            "volume_ul": self.volume_ul,
            "src_row": self.src_row,
            "src_col": self.src_col,
            "dst_row": self.dst_row,
            "dst_col": self.dst_col,
        }


# ── Task Matrix → Job list ────────────────────────────────────────────────────

def task_matrix_to_jobs(T: np.ndarray,
                         src_labware_id: str, dst_labware_id: str,
                         src_deck_pos: int, dst_deck_pos: int,
                         src_cols: int = 12, dst_cols: int = 12) -> List[Job]:
    """
    Convert a transfer task matrix T[n_src_wells × n_dst_wells] to a Job list.
    T[a, b] = volume to transfer from src well a to dst well b (0 = no transfer).
    """
    jobs = []
    job_id = 1
    rows, cols = T.shape
    idxs = np.argwhere(T > 0)
    for (a, b) in idxs:
        vol = float(T[a, b])
        jobs.append(Job(
            job_id=job_id,
            src_labware_id=src_labware_id,
            dst_labware_id=dst_labware_id,
            src_well=int(a),
            dst_well=int(b),
            src_deck_pos=src_deck_pos,
            dst_deck_pos=dst_deck_pos,
            volume_ul=vol,
            src_row=int(a) // src_cols,
            src_col=int(a) % src_cols,
            dst_row=int(b) // dst_cols,
            dst_col=int(b) % dst_cols,
        ))
        job_id += 1
    return jobs


# ── Distance Matrix ───────────────────────────────────────────────────────────

def _well_distance_1d(r_a: int, c_a: int, r_b: int, c_b: int,
                      n_channels: int = 8) -> float:
    """
    Paper's well adjacency cost (D_src or D_dst component).
    Two wells can be pipetted simultaneously (cost=0) if they share a column
    and their row indices differ by < n_channels. Otherwise cost=1.
    This is a simplified version of the paper's Eq.1 binary adjacency term.
    """
    if c_a == c_b and abs(r_a - r_b) < n_channels:
        return 0.0
    return 1.0


def build_distance_matrix(jobs: List[Job],
                            alpha: float = ALPHA,
                            beta: float = BETA) -> np.ndarray:
    """
    Build the extended distance matrix D_extended for OR-Tools.

    Size: (N+1) × (N+1) where index 0 = depot (dummy job).
    D_extended[i, j] = alpha * D'_plate[i,j] + beta * D_deck[i,j]

    D'_plate[i,j] = src_well_cost(i→j) + dst_well_cost(i→j)  [paper's Eq.2-4]
    D_deck[i,j]   = arm travel mm between last deck pos of job i and first of job j

    For multi-plate protocols: the arm goes src→dst for each job.
    Between jobs, the arm travels from dst_pos of job i to src_pos of job j.
    """
    N = len(jobs)
    D = np.zeros((N + 1, N + 1), dtype=float)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ji = jobs[i]
            jj = jobs[j]

            # Paper's plate-level cost: well adjacency on source + destination plates
            src_cost = _well_distance_1d(ji.src_row, ji.src_col,
                                          jj.src_row, jj.src_col)
            dst_cost = _well_distance_1d(ji.dst_row, ji.dst_col,
                                          jj.dst_row, jj.dst_col)
            plate_cost = src_cost + dst_cost

            # Spatial deck cost: arm travels from dst of job i → src of job j
            # (plus any tip rack visits are handled at the protocol level)
            deck_mm = arm_travel_distance(ji.dst_deck_pos, jj.src_deck_pos)

            D[i + 1, j + 1] = alpha * plate_cost + beta * deck_mm

    # Depot (index 0) has zero distance to/from all jobs
    # (matches paper: dummy node has all-zero distances)
    return D


def _to_int_matrix(D: np.ndarray, scale: int = SCALE) -> List[List[int]]:
    """Scale float matrix to integers for OR-Tools."""
    return [[int(round(v * scale)) for v in row] for row in D.tolist()]


# ── OR-Tools CVRP Solver ─────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    method: str
    routes: List[List[int]]          # list of job_id lists (1-indexed, matching jobs list)
    total_cost: float
    total_travel_mm: float
    n_cycles: int
    solve_time_s: float
    job_order: List[int]             # flat sequence of job indices (0-based) across all cycles
    distance_matrix: Optional[np.ndarray] = None
    improvement_vs_baseline: Optional[float] = None  # % improvement over row-major

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "routes": self.routes,
            "total_cost": round(self.total_cost, 4),
            "total_travel_mm": round(self.total_travel_mm, 2),
            "n_cycles": self.n_cycles,
            "solve_time_s": round(self.solve_time_s, 3),
            "job_order": self.job_order,
            "improvement_vs_baseline": (
                round(self.improvement_vs_baseline, 2)
                if self.improvement_vs_baseline is not None else None
            ),
        }


def solve_cvrp(jobs: List[Job],
               distance_matrix: np.ndarray,
               capacity: int = PIPETTE_CAPACITY,
               time_limit_s: int = 10,
               alpha: float = ALPHA,
               beta: float = BETA) -> OptimizationResult:
    """
    Solve the CVRP using Google OR-Tools.
    Returns optimized routes (cycles) over the job list.

    Depot = index 0 (dummy job with zero distances).
    Each vehicle (cycle) has capacity = 8 (pipette channels).
    Each job has demand = 1 (one well pair per tip).
    """
    N = len(jobs)
    if N == 0:
        return OptimizationResult("cvrp", [], 0.0, 0.0, 0, 0.0, [])

    t0 = time.time()

    D_int = _to_int_matrix(distance_matrix, SCALE)

    # OR-Tools data model
    data = {
        "distance_matrix": D_int,
        "num_vehicles": N,          # upper bound; OR-Tools will use as few as needed
        "depot": 0,
        "demands": [0] + [1] * N,  # depot demand = 0; each job demand = 1
        "vehicle_capacities": [capacity] * N,
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"],
    )
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return data["distance_matrix"][from_node][to_node]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Capacity constraint
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        return data["demands"][from_node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,                             # null capacity slack
        data["vehicle_capacities"],    # vehicle capacities
        True,                          # start cumul at zero
        "Capacity",
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(search_params)
    solve_time = time.time() - t0

    if not solution:
        # Fallback to row-major baseline if solver fails
        return _row_major_baseline(jobs, distance_matrix, solve_time)

    # Extract routes
    routes = []
    job_order = []
    total_cost = 0.0

    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node - 1)  # convert back to 0-based job index
                job_order.append(node - 1)
            idx = solution.Value(routing.NextVar(idx))
        if route:
            routes.append(route)

    # Compute total cost from distance matrix
    for route in routes:
        prev = 0  # depot
        for job_idx in route:
            node = job_idx + 1
            total_cost += distance_matrix[prev, node]
            prev = node
        total_cost += distance_matrix[prev, 0]  # return to depot

    # Deck travel for the optimized order
    total_mm = _compute_deck_travel(job_order, jobs)

    result = OptimizationResult(
        method="cvrp_or_tools",
        routes=routes,
        total_cost=total_cost,
        total_travel_mm=total_mm,
        n_cycles=len(routes),
        solve_time_s=solve_time,
        job_order=job_order,
        distance_matrix=distance_matrix,
    )

    # Compare with row-major baseline
    baseline = _row_major_baseline(jobs, distance_matrix, 0.0)
    if baseline.total_cost > 0:
        result.improvement_vs_baseline = (
            (baseline.total_cost - total_cost) / baseline.total_cost * 100
        )

    return result


# ── Baseline Methods ──────────────────────────────────────────────────────────

def _row_major_baseline(jobs: List[Job],
                         D: np.ndarray,
                         solve_time: float) -> OptimizationResult:
    """Row-major sorting baseline: execute jobs in task matrix order."""
    N = len(jobs)
    order = list(range(N))
    routes = [order[i:i+PIPETTE_CAPACITY] for i in range(0, N, PIPETTE_CAPACITY)]
    cost = _route_cost(routes, D)
    mm = _compute_deck_travel(order, jobs)
    return OptimizationResult("row_major", routes, cost, mm, len(routes), solve_time, order)


def _greedy_baseline(jobs: List[Job], D: np.ndarray) -> OptimizationResult:
    """Greedy nearest-neighbour baseline."""
    t0 = time.time()
    N = len(jobs)
    unvisited = set(range(N))
    order = []
    routes = []
    current_route = []
    current = 0  # depot

    while unvisited:
        # Find nearest unvisited job
        best, best_cost = -1, float("inf")
        for j in unvisited:
            c = D[current, j + 1]
            if c < best_cost:
                best_cost, best = c, j
        current_route.append(best)
        order.append(best)
        unvisited.remove(best)
        current = best + 1
        if len(current_route) >= PIPETTE_CAPACITY:
            routes.append(current_route)
            current_route = []
            current = 0  # return to depot

    if current_route:
        routes.append(current_route)

    cost = _route_cost(routes, D)
    mm = _compute_deck_travel(order, jobs)
    return OptimizationResult("greedy", routes, cost, mm, len(routes),
                               time.time() - t0, order)


def _lap_baseline(jobs: List[Job], D: np.ndarray) -> OptimizationResult:
    """
    Long-Axis Prioritized (LAP) baseline from the paper.
    Prioritizes same-column jobs to maximize parallelization.
    """
    t0 = time.time()
    N = len(jobs)
    # Group by source column (long axis for 96-well = 12 cols)
    from collections import defaultdict
    col_groups: Dict[int, List[int]] = defaultdict(list)
    for i, job in enumerate(jobs):
        col_groups[job.src_col].append(i)

    order = []
    routes = []
    current_route = []

    for col in sorted(col_groups.keys()):
        for ji in col_groups[col]:
            current_route.append(ji)
            order.append(ji)
            if len(current_route) >= PIPETTE_CAPACITY:
                routes.append(current_route)
                current_route = []

    if current_route:
        routes.append(current_route)

    cost = _route_cost(routes, D)
    mm = _compute_deck_travel(order, jobs)
    return OptimizationResult("lap", routes, cost, mm, len(routes),
                               time.time() - t0, order)


def _route_cost(routes: List[List[int]], D: np.ndarray) -> float:
    """Compute total distance for a set of routes through the distance matrix."""
    total = 0.0
    for route in routes:
        prev = 0
        for ji in route:
            total += D[prev, ji + 1]
            prev = ji + 1
        total += D[prev, 0]
    return total


def _compute_deck_travel(job_order: List[int], jobs: List[Job]) -> float:
    """Total arm travel in mm for an ordered job sequence (dst→src between jobs)."""
    mm = 0.0
    for k in range(1, len(job_order)):
        prev = jobs[job_order[k - 1]]
        curr = jobs[job_order[k]]
        mm += arm_travel_distance(prev.dst_deck_pos, curr.src_deck_pos)
    return mm


# ── Full Optimization Run ─────────────────────────────────────────────────────

@dataclass
class OptimizationReport:
    jobs: List[Job]
    distance_matrix: np.ndarray
    cvrp: OptimizationResult
    row_major: OptimizationResult
    greedy: OptimizationResult
    lap: OptimizationResult
    alpha: float
    beta: float

    def summary(self) -> dict:
        return {
            "n_jobs": len(self.jobs),
            "alpha": self.alpha,
            "beta": self.beta,
            "methods": {
                "cvrp_or_tools": self.cvrp.to_dict(),
                "row_major": self.row_major.to_dict(),
                "greedy": self.greedy.to_dict(),
                "lap": self.lap.to_dict(),
            },
            "improvements": {
                "vs_row_major_%": self.cvrp.improvement_vs_baseline,
                "vs_greedy_%": (
                    round((self.greedy.total_cost - self.cvrp.total_cost)
                          / self.greedy.total_cost * 100, 2)
                    if self.greedy.total_cost > 0 else None
                ),
                "vs_lap_%": (
                    round((self.lap.total_cost - self.cvrp.total_cost)
                          / self.lap.total_cost * 100, 2)
                    if self.lap.total_cost > 0 else None
                ),
            },
            "deck_travel_savings_mm": {
                "vs_row_major": round(self.row_major.total_travel_mm - self.cvrp.total_travel_mm, 2),
                "vs_greedy": round(self.greedy.total_travel_mm - self.cvrp.total_travel_mm, 2),
            }
        }


def run_optimization(jobs: List[Job],
                     alpha: float = ALPHA,
                     beta: float = BETA,
                     time_limit_s: int = 10) -> OptimizationReport:
    """Run the full optimization and all baselines. Return a comparison report."""
    D = build_distance_matrix(jobs, alpha=alpha, beta=beta)

    cvrp_result    = solve_cvrp(jobs, D, time_limit_s=time_limit_s, alpha=alpha, beta=beta)
    row_major      = _row_major_baseline(jobs, D, 0.0)
    greedy         = _greedy_baseline(jobs, D)
    lap            = _lap_baseline(jobs, D)

    return OptimizationReport(
        jobs=jobs,
        distance_matrix=D,
        cvrp=cvrp_result,
        row_major=row_major,
        greedy=greedy,
        lap=lap,
        alpha=alpha,
        beta=beta,
    )
