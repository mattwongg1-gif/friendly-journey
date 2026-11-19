"""OR-Tools model for one driver's week.

The week is modelled as one continuous tour that starts at the driver's home:
  * Working time is a single clock running across the driver's available days
    (day k covers [B[k-1], B[k]) where B is the running total of daily limits).
    A store visit may not straddle the end of a day.
  * After vehicle_cap stores the driver must go home to restock. Restocks are
    optional copies of the home node (OR-Tools "reload" pattern) with a load
    dimension that resets to zero there.
  * The driver does not go home at night; the next day starts where the last
    one ended.
  * Objective: minimise total drive minutes, plus a penalty for every store
    left out (huge for required stores, small for optional look-ahead ones).

The solver gives the order. schedule.simulate() then lays that order onto
real days and is the final word on which day each stop happens.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HOME_LOC = 0  # matrix index of the driver's home


@dataclass
class SolverSite:
    loc: int            # row/col in the duration matrix (1..n; 0 is home)
    service_mins: int
    penalty: int        # objective cost of not visiting


@dataclass
class SolveResult:
    # ("site", i) where i indexes the input sites list, or ("restock", None)
    sequence: List[Tuple[str, Optional[int]]]
    dropped: List[int]
    status: str


def solve_driver_week(
    matrix: np.ndarray,
    sites: Sequence[SolverSite],
    vehicle_cap: Optional[int],
    day_budgets: Sequence[int],
    restock_mins: int,
    return_home: bool,
    time_limit_s: int = 20,
) -> SolveResult:
    n = len(sites)
    if n == 0 or not day_budgets:
        return SolveResult([], list(range(n)), "empty")

    use_cap = bool(vehicle_cap) and vehicle_cap > 0 and n > vehicle_cap
    n_reload = math.ceil(n / vehicle_cap) if use_cap else 0

    # Node layout: 0 start | 1..R restocks | R+1..R+n sites | R+n+1 end
    START = 0
    RELOAD = range(1, n_reload + 1)
    SITE0 = n_reload + 1
    END = n_reload + n + 1
    n_nodes = END + 1

    loc = [HOME_LOC] * (n_reload + 1) + [s.loc for s in sites] + [HOME_LOC]
    service = [0] + [restock_mins] * n_reload + [int(s.service_mins) for s in sites] + [0]

    m = np.asarray(matrix, dtype=int)
    travel = m[np.ix_(loc, loc)].copy()
    if not return_home:
        travel[:, END] = 0  # the week can end anywhere
    travel_l = travel.tolist()
    is_reload = [False] * n_nodes
    for r in RELOAD:
        is_reload[r] = True

    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, [START], [END])
    routing = pywrapcp.RoutingModel(manager)

    # Arc cost = drive minutes (+ restock minutes so idle restocks are never free).
    def cost_cb(i_idx, j_idx):
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        return travel_l[i][j] + (restock_mins if is_reload[j] else 0)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_cb))

    # Working-clock time dimension.
    def time_cb(i_idx, j_idx):
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        return service[i] + travel_l[i][j]

    bounds = np.cumsum(day_budgets).tolist()
    horizon = int(bounds[-1])
    routing.AddDimension(
        routing.RegisterTransitCallback(time_cb),
        int(max(day_budgets)),  # slack = waiting, e.g. ending a day early
        horizon,
        True,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Smallest drive into each site from any other node, used below.
    min_in = []
    for j in range(n_nodes):
        col = [travel_l[i][j] for i in range(n_nodes) if i != j and loc[i] != loc[j]]
        min_in.append(min(col) if col else 0)

    for k, s in enumerate(sites):
        node = SITE0 + k
        idx = manager.NodeToIndex(node)
        cv = time_dim.CumulVar(idx)
        svc = service[node]
        cv.SetRange(0, max(0, horizon - svc))
        for b in bounds[:-1]:
            # Service must finish by the end of the day ...
            lo, hi = b - svc + 1, b - 1
            if lo <= hi:
                cv.RemoveInterval(int(lo), int(hi))
            # ... and an arrival just after the day starts would mean the drive
            # began the night before.
            if min_in[node] > 0:
                cv.RemoveInterval(int(b), int(b + min_in[node] - 1))

    # Vehicle capacity with restocks at home.
    if use_cap:
        demand = [0] * n_nodes
        for k in range(n):
            demand[SITE0 + k] = 1
        for r in RELOAD:
            demand[r] = -vehicle_cap

        def demand_cb(i_idx):
            return demand[manager.IndexToNode(i_idx)]

        routing.AddDimension(
            routing.RegisterUnaryTransitCallback(demand_cb),
            vehicle_cap,  # slack lets the load drop to anything at a restock
            vehicle_cap,
            True,
            "Load",
        )
        load_dim = routing.GetDimensionOrDie("Load")
        for k in range(n):
            load_dim.SlackVar(manager.NodeToIndex(SITE0 + k)).SetValue(0)
        for r in RELOAD:
            routing.AddDisjunction([manager.NodeToIndex(r)], 0)

    for k, s in enumerate(sites):
        routing.AddDisjunction([manager.NodeToIndex(SITE0 + k)], int(s.penalty))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return _greedy_fallback(m, sites, vehicle_cap if use_cap else None)

    seq: List[Tuple[str, Optional[int]]] = []
    visited = set()
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if is_reload[node]:
            seq.append(("restock", None))
        elif SITE0 <= node < END:
            seq.append(("site", node - SITE0))
            visited.add(node - SITE0)
        index = sol.Value(routing.NextVar(index))

    # A restock as the very last action is pointless.
    while seq and seq[-1][0] == "restock":
        seq.pop()
    dropped = [k for k in range(n) if k not in visited]
    return SolveResult(seq, dropped, "ok")


def _greedy_fallback(matrix, sites, cap) -> SolveResult:
    """Nearest-neighbour order with restocks; used only if OR-Tools finds nothing."""
    remaining = set(range(len(sites)))
    cur, load, seq = HOME_LOC, 0, []
    while remaining:
        if cap and load >= cap:
            seq.append(("restock", None))
            cur, load = HOME_LOC, 0
        k = min(remaining, key=lambda i: matrix[cur][sites[i].loc])
        seq.append(("site", k))
        remaining.discard(k)
        cur, load = sites[k].loc, load + 1
    return SolveResult(seq, [], "greedy_fallback")
