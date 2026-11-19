"""Plan a week for every driver. Pure pandas: no Spark or Foundry imports,
so it can be unit tested anywhere."""
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import Columns, PlannerConfig
from .due_sites import parse_geopoint, select_due_sites
from .schedule import simulate
from .solver import HOME_LOC, SolverSite, solve_driver_week
from .travel import TravelProvider

log = logging.getLogger(__name__)

ROUTE_COLUMNS = [
    "route_id", "driver_id", "date", "week_start", "drive_time_mins", "work_time_mins",
    "total_time_mins", "max_work_mins", "over_budget_mins", "stop_count", "restock_count",
    "start_location", "end_location", "geometry", "planned_at",
]
STOP_COLUMNS = [
    "stop_id", "route_id", "driver_id", "date", "site_id", "seq_ind", "profit",
    "arrive_min", "depart_min", "drive_mins_from_prev", "restock_before", "required",
]
UNSCHEDULED_COLUMNS = [
    "week_start", "driver_id", "site_id", "required", "next_due_date", "days_overdue", "reason",
]
TARGET_COLUMNS = ["site_id", "targ_visit_date", "route_id", "driver_id", "week_start"]


def week_bounds(today: date) -> Tuple[date, date]:
    monday = today - timedelta(days=today.weekday())
    return monday, monday + timedelta(days=6)


def _as_bool(v, default: bool) -> bool:
    """Booleans from typed columns or CSV text ("false", "No", "0", "")."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null", "none"):
            return default
        return s in ("true", "t", "yes", "y", "1")
    return bool(v)


def available_days(
    driver_id: str,
    plan_from: date,
    week_end: date,
    availability: Optional[pd.DataFrame],
    cfg: PlannerConfig,
    cols: Columns = Columns(),
) -> List[Tuple[date, int]]:
    """Dates the driver works between plan_from and week_end, with minute limits.

    The availability table holds exceptions: a row with is_available = false
    removes a default work day; a row with is_available = true adds a day (e.g.
    a Saturday) or overrides max_work_mins. No row means the default schedule.
    """
    overrides: Dict[date, dict] = {}
    if availability is not None and len(availability):
        mine = availability[availability[cols.avail_driver] == driver_id]
        for rec in mine.to_dict("records"):
            d = pd.Timestamp(rec[cols.avail_date]).date()
            overrides[d] = rec

    days = []
    d = plan_from
    while d <= week_end:
        works = d.weekday() in cfg.default_work_days
        budget = cfg.default_max_work_mins
        if d in overrides:
            rec = overrides[d]
            works = _as_bool(rec.get(cols.avail_flag), default=True)
            mx = rec.get(cols.avail_max_mins)
            if isinstance(mx, str):
                mx = mx.strip() or None
            if mx is not None and not pd.isna(mx):
                budget = int(float(mx))
        if works and budget > 0:
            days.append((d, budget))
        d += timedelta(days=1)
    return days


def plan_driver_week(
    driver_id: str,
    home: Tuple[float, float],
    vehicle_cap: Optional[int],
    due: pd.DataFrame,
    days: Sequence[Tuple[date, int]],
    provider: TravelProvider,
    cfg: PlannerConfig,
    week_start: date,
    planned_at: str,
):
    routes, stops, unscheduled, targets = [], [], [], []
    due = due.reset_index(drop=True)

    def unsched(rows: pd.DataFrame, reason: str):
        for r in rows.to_dict("records"):
            unscheduled.append({
                "week_start": week_start, "driver_id": driver_id, "site_id": r["site_id"],
                "required": bool(r["required"]), "next_due_date": r["next_due_date"],
                "days_overdue": int(r["days_overdue"]), "reason": reason,
            })

    if due.empty:
        return routes, stops, unscheduled, targets
    if not days:
        unsched(due, "no_available_days")
        return routes, stops, unscheduled, targets

    coords = [home] + list(zip(due["lat"], due["lon"]))
    matrix = provider.duration_matrix_mins(coords)
    sites = [
        SolverSite(loc=i + 1, service_mins=int(r.service_mins), penalty=int(r.penalty))
        for i, r in enumerate(due.itertuples())
    ]
    budgets = [b for _, b in days]

    res = solve_driver_week(
        matrix, sites, vehicle_cap, budgets, cfg.restock_mins,
        cfg.return_home_at_end, cfg.solver_time_limit_s,
    )
    log.info("driver %s: %d due, solver %s, dropped %d", driver_id, len(sites), res.status, len(res.dropped))

    day_plans, overflow = simulate(res.sequence, matrix, sites, budgets, cfg.restock_mins, cfg.return_home_at_end)

    for k in res.dropped:
        row = due.iloc[[k]]
        unsched(row, "week_full" if row["required"].iat[0] else "optional_not_worth_it")
    for k in overflow:
        unsched(due.iloc[[k]], "week_full")

    def loc_coord(loc: int) -> Tuple[float, float]:
        return coords[loc]

    def gp(loc: int) -> str:
        lat, lon = loc_coord(loc)
        return f"{lat:.6f},{lon:.6f}"

    for dp in day_plans:
        d, budget = days[dp.day_idx]
        route_id = f"{driver_id}-{d:%Y%m%d}"
        path = [loc_coord(dp.start_loc)] + [loc_coord(e.loc) for e in dp.events]
        geom = provider.line_geometry(path)

        seq, restock_pending, n_restock = 0, False, 0
        for e in dp.events:
            if e.kind == "restock":
                restock_pending, n_restock = True, n_restock + 1
                continue
            if e.kind != "site":
                continue
            seq += 1
            r = due.iloc[e.site_idx]
            stops.append({
                "stop_id": f"{route_id}-{seq:03d}", "route_id": route_id, "driver_id": driver_id,
                "date": d, "site_id": r["site_id"], "seq_ind": seq, "profit": r["profit"],
                "arrive_min": e.arrive_min, "depart_min": e.depart_min,
                "drive_mins_from_prev": e.drive_mins, "restock_before": restock_pending,
                "required": bool(r["required"]),
            })
            targets.append({"site_id": r["site_id"], "targ_visit_date": d, "route_id": route_id,
                            "driver_id": driver_id, "week_start": week_start})
            restock_pending = False

        routes.append({
            "route_id": route_id, "driver_id": driver_id, "date": d, "week_start": week_start,
            "drive_time_mins": dp.drive_mins, "work_time_mins": dp.work_mins,
            "total_time_mins": dp.used, "max_work_mins": budget,
            "over_budget_mins": dp.over_budget_mins, "stop_count": seq, "restock_count": n_restock,
            "start_location": gp(dp.start_loc), "end_location": gp(dp.end_loc),
            "geometry": json.dumps(geom) if geom else None, "planned_at": planned_at,
        })
    return routes, stops, unscheduled, targets


def plan_week(
    drivers: pd.DataFrame,
    territories: pd.DataFrame,
    sites: pd.DataFrame,
    availability: Optional[pd.DataFrame],
    provider: TravelProvider,
    today: date,
    cfg: PlannerConfig = PlannerConfig(),
    cols: Columns = Columns(),
):
    """Returns (routes, stops, unscheduled, site_targets) DataFrames for the
    week containing `today`. Days before `today` are not planned, so a rerun
    mid-week only replans the rest of the week."""
    week_start, week_end = week_bounds(today)
    planned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    due = select_due_sites(sites, week_start, cfg, cols)
    terr_to_driver = dict(zip(territories[cols.territory_pk], territories[cols.territory_driver]))
    due["driver_id"] = due["territory_id"].map(terr_to_driver)

    all_routes, all_stops, all_unsched, all_targets = [], [], [], []

    def bulk_unsched(rows: pd.DataFrame, reason: str):
        for r in rows.to_dict("records"):
            all_unsched.append({
                "week_start": week_start, "driver_id": r.get("driver_id"), "site_id": r["site_id"],
                "required": bool(r["required"]), "next_due_date": r["next_due_date"],
                "days_overdue": int(r["days_overdue"]), "reason": reason,
            })

    bad = due[due["problem"].notna()]
    for problem, grp in bad.groupby("problem"):
        bulk_unsched(grp, problem)
    due = due[due["problem"].isna()]
    bulk_unsched(due[due["driver_id"].isna()], "no_driver_for_territory")
    due = due[due["driver_id"].notna()]

    known = set(drivers[cols.driver_pk])
    bulk_unsched(due[~due["driver_id"].isin(known)], "driver_not_found")

    # Plan drivers region by region so road data loaded for one driver is
    # reused by their neighbours.
    def region(rec):
        h = parse_geopoint(rec.get(cols.driver_home))
        return (0, int(h[0] // 2), int(h[1] // 2)) if h else (1, 0, 0)

    attempted, failures = 0, []
    for drv in sorted(drivers.to_dict("records"), key=region):
        driver_id = drv[cols.driver_pk]
        mine = due[due["driver_id"] == driver_id]
        if mine.empty:
            continue
        home = parse_geopoint(drv.get(cols.driver_home))
        if home is None:
            bulk_unsched(mine, "driver_missing_home")
            continue
        cap = drv.get(cols.driver_cap)
        cap = int(cap) if cap is not None and not pd.isna(cap) and int(cap) > 0 else None

        days = available_days(driver_id, max(today, week_start), week_end, availability, cfg, cols)
        attempted += 1
        try:
            r, s, u, t = plan_driver_week(driver_id, home, cap, mine, days, provider, cfg, week_start, planned_at)
        except Exception as e:
            log.exception("planning failed for driver %s", driver_id)
            failures.append(e)
            bulk_unsched(mine, "planning_error")
            continue
        all_routes += r
        all_stops += s
        all_unsched += u
        all_targets += t

    # One bad driver is reported in unscheduled_sites; if every driver failed
    # (e.g. OSRM is down) fail the build instead of writing an empty plan.
    if attempted and len(failures) == attempted:
        raise RuntimeError(f"Planning failed for all {attempted} drivers") from failures[0]

    return (
        pd.DataFrame(all_routes, columns=ROUTE_COLUMNS),
        pd.DataFrame(all_stops, columns=STOP_COLUMNS),
        pd.DataFrame(all_unsched, columns=UNSCHEDULED_COLUMNS),
        pd.DataFrame(all_targets, columns=TARGET_COLUMNS),
    )
