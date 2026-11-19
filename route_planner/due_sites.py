"""Decide which stores each driver must visit in the planning week."""
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from .config import Columns, PlannerConfig


def parse_geopoint(value) -> Optional[Tuple[float, float]]:
    """Foundry geopoint strings are "lat,lon". Returns (lat, lon) or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        lat_s, lon_s = str(value).split(",")
        lat, lon = float(lat_s), float(lon_s)
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _as_date(v) -> Optional[date]:
    if isinstance(v, str):
        v = v.strip() or None  # CSV uploads give text; blank = no date
    if v is None or (not isinstance(v, date) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.Timestamp(v).date()


def select_due_sites(
    sites: pd.DataFrame,
    week_start: date,
    cfg: PlannerConfig,
    cols: Columns = Columns(),
) -> pd.DataFrame:
    """Return one row per store that is due (or optionally due) this week.

    visit_freq is weeks between visits, so the next due date is
    last_visit_date + 7 * visit_freq days. Stores never visited are due now.

    Output columns: site_id, territory_id, lat, lon, service_mins, required,
    next_due_date, days_overdue, missed_target, penalty, profit, problem.
    `problem` is set (and the store is not routable) when its data is bad.
    """
    week_end = week_start + timedelta(days=6)
    optional_until = week_end + timedelta(days=cfg.lookahead_days)

    rows: List[dict] = []
    for rec in sites.to_dict("records"):
        freq = rec.get(cols.site_freq_weeks)
        last_visit = _as_date(rec.get(cols.site_last_visit))
        last_target = _as_date(rec.get(cols.site_last_target))

        if freq is None or pd.isna(freq) or int(freq) <= 0:
            continue  # no visit schedule set: never auto-planned
        freq = int(freq)

        next_due = (last_visit + timedelta(weeks=freq)) if last_visit else week_start
        if next_due > optional_until:
            continue
        required = next_due <= week_end

        # Targeted on an earlier plan, but no visit recorded since then.
        missed_target = bool(
            last_target
            and last_target < week_start
            and (last_visit is None or last_visit < last_target)
        )
        days_overdue = max(0, (week_start - next_due).days)

        if required:
            penalty = (
                cfg.required_drop_penalty
                + cfg.penalty_per_overdue_day * days_overdue
                + (cfg.missed_target_penalty if missed_target else 0)
            )
        else:
            penalty = cfg.optional_drop_penalty

        loc = parse_geopoint(rec.get(cols.site_location))
        service = rec.get(cols.site_service)
        service = int(service) if service is not None and not pd.isna(service) else cfg.default_service_mins
        profit = rec.get(cols.site_profit)

        rows.append(
            {
                "site_id": rec[cols.site_pk],
                "territory_id": rec.get(cols.site_territory),
                "lat": loc[0] if loc else None,
                "lon": loc[1] if loc else None,
                "service_mins": max(0, service),
                "required": required,
                "next_due_date": next_due,
                "days_overdue": days_overdue,
                "missed_target": missed_target,
                "penalty": int(penalty),
                "profit": None if profit is None or pd.isna(profit) else float(profit),
                "problem": None if loc else "missing_or_invalid_location",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "site_id", "territory_id", "lat", "lon", "service_mins", "required",
            "next_due_date", "days_overdue", "missed_target", "penalty", "profit", "problem",
        ],
    )
