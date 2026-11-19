"""Tunable settings for the weekly route planner.

Everything a planner might want to change lives here, so the transform code
does not need editing to change a rule.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PlannerConfig:
    # --- Which days a driver works -------------------------------------------
    # Used when the availability dataset has no row for a driver/date.
    # 0 = Monday ... 6 = Sunday.
    default_work_days: Tuple[int, ...] = (0, 1, 2, 3, 4)
    # Minutes of drive + service + restock time allowed per working day.
    default_max_work_mins: int = 480
    # Local time zone used to decide what "today" and "this week" are.
    timezone: str = "America/Chicago"

    # --- Store selection --------------------------------------------------------
    # A store is required this week if last_visit_date + visit_freq weeks
    # falls on or before Sunday of the planning week (overdue stores included).
    # Stores due within this many days *after* the week are optional: the
    # solver adds them only if they are cheap to fit in.
    lookahead_days: int = 0
    # Service minutes used when Site.avg_service_mins is null.
    default_service_mins: int = 20

    # --- Route shape ---------------------------------------------------------
    # Minutes spent at home reloading merchandise after hitting vehicle_cap.
    restock_mins: int = 30
    # Whether the driver must be back home at the end of the week.
    return_home_at_end: bool = True

    # --- Objective weights (units: minutes of driving) -----------------------
    # Cost of leaving a required store unvisited. Large, so stores are only
    # dropped when the week is genuinely full.
    required_drop_penalty: int = 100_000
    # Extra penalty per day overdue, so the most overdue stores are kept first.
    penalty_per_overdue_day: int = 1_000
    # Extra penalty if the store was targeted on a past plan but not visited.
    missed_target_penalty: int = 20_000
    # An optional (look-ahead) store is added only if it costs fewer than this
    # many extra driving minutes.
    optional_drop_penalty: int = 45

    # --- Solver --------------------------------------------------------------
    solver_time_limit_s: int = 20

    # --- Straight-line fallback (used if OSRM is unavailable) ----------------
    fallback_speed_kmh: float = 50.0
    fallback_circuity: float = 1.3


# Column names in the datasets that back each object type. Change these if
# your backing datasets use different names.
@dataclass
class Columns:
    driver_pk: str = "driver_id"
    driver_home: str = "home"            # geopoint string "lat,lon"
    driver_cap: str = "vehicle_cap"

    territory_pk: str = "territory_id"
    territory_driver: str = "driver_id"

    site_pk: str = "site_id"
    site_location: str = "location"      # geopoint string "lat,lon"
    site_territory: str = "territory_id"
    site_service: str = "avg_service_mins"
    site_freq_weeks: str = "visit_freq"
    site_last_visit: str = "last_visit_date"
    site_last_target: str = "last_targ_visit_date"
    # Optional: expected profit per visit, copied onto Stop.profit if present.
    site_profit: str = "expected_profit"

    avail_driver: str = "driver_id"
    avail_date: str = "date"
    avail_flag: str = "is_available"
    avail_max_mins: str = "max_work_mins"
