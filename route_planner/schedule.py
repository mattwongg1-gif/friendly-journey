"""Lay the solver's visiting order onto the driver's real working days."""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .solver import HOME_LOC, SolverSite


@dataclass
class Event:
    kind: str                 # "site" | "restock" | "home"
    site_idx: Optional[int]   # index into the sites list for kind == "site"
    loc: int                  # matrix index
    drive_mins: int           # driving to get here
    arrive_min: int           # minutes after the start of this day's shift
    depart_min: int


@dataclass
class DayPlan:
    day_idx: int
    start_loc: int
    budget: int
    events: List[Event] = field(default_factory=list)
    used: int = 0

    @property
    def drive_mins(self) -> int:
        return sum(e.drive_mins for e in self.events)

    @property
    def work_mins(self) -> int:
        return sum(e.depart_min - e.arrive_min for e in self.events)

    @property
    def over_budget_mins(self) -> int:
        return max(0, self.used - self.budget)

    @property
    def end_loc(self) -> int:
        return self.events[-1].loc if self.events else self.start_loc


def simulate(
    sequence: Sequence[Tuple[str, Optional[int]]],
    matrix: np.ndarray,
    sites: Sequence[SolverSite],
    day_budgets: Sequence[int],
    restock_mins: int,
    return_home: bool,
) -> Tuple[List[DayPlan], List[int]]:
    """Walk the order, starting a new day whenever the next step won't fit.

    Rules:
      * A store visit (drive + service) must fit in the current day, unless the
        day is still empty (a single oversized stop is allowed, and flagged).
      * Before a restock the driver may drive home at the end of a day and
        restock first thing the next morning.
      * The final drive home goes on the last working day; any overrun shows up
        as over_budget_mins.
    Returns (days that have events, site indices that did not fit).
    """
    days: List[DayPlan] = []
    if not day_budgets:
        return days, [i for kind, i in sequence if kind == "site"]

    day = DayPlan(0, HOME_LOC, int(day_budgets[0]))
    days.append(day)
    cur = HOME_LOC
    unscheduled: List[int] = []

    def next_day() -> Optional[DayPlan]:
        nonlocal day
        if day.day_idx + 1 >= len(day_budgets):
            return None
        day = DayPlan(day.day_idx + 1, cur, int(day_budgets[day.day_idx + 1]))
        days.append(day)
        return day

    out_of_days = False
    for kind, si in sequence:
        if out_of_days:
            if kind == "site":
                unscheduled.append(si)
            continue

        if kind == "restock":
            drive = int(matrix[cur][HOME_LOC])
            if day.used + drive + restock_mins <= day.budget or not day.events:
                _add(day, Event("restock", None, HOME_LOC, drive, day.used + drive, day.used + drive + restock_mins))
                cur = HOME_LOC
                continue
            if drive > 0 and day.used + drive <= day.budget:
                # Drive home tonight, restock tomorrow morning.
                _add(day, Event("home", None, HOME_LOC, drive, day.used + drive, day.used + drive))
                cur = HOME_LOC
                drive = 0
            if next_day() is None:
                out_of_days = True
                continue
            _add(day, Event("restock", None, HOME_LOC, drive, drive, drive + restock_mins))
            cur = HOME_LOC
            continue

        s = sites[si]
        drive = int(matrix[cur][s.loc])
        need = drive + int(s.service_mins)
        if day.used + need > day.budget and day.events:
            if next_day() is None:
                out_of_days = True
                unscheduled.append(si)
                continue
            drive = int(matrix[cur][s.loc])  # unchanged location, same drive
        _add(day, Event("site", si, s.loc, drive, day.used + drive, day.used + drive + int(s.service_mins)))
        cur = s.loc

    if return_home and cur != HOME_LOC:
        # Put the drive home on the last day that has work.
        last = next((d for d in reversed(days) if d.events), days[0])
        drive = int(matrix[cur][HOME_LOC])
        _add(last, Event("home", None, HOME_LOC, drive, last.used + drive, last.used + drive))

    return [d for d in days if d.events], unscheduled


def _add(day: DayPlan, ev: Event) -> None:
    day.events.append(ev)
    day.used = ev.depart_min
