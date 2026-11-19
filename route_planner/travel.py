"""Drive-time matrices and route lines.

OsrmProvider talks to an OSRM server (open-source routing on OpenStreetMap
data). HaversineProvider is a no-network fallback for tests or outages.
Coordinates are (lat, lon) everywhere in this package; conversion to OSRM's
and GeoJSON's lon,lat order happens only inside this module.
"""
import logging
import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)
LatLon = Tuple[float, float]


class TravelProvider:
    def duration_matrix_mins(self, coords: Sequence[LatLon]) -> np.ndarray:
        """Square int matrix: minutes to drive from coords[i] to coords[j]."""
        raise NotImplementedError

    def line_geometry(self, coords: Sequence[LatLon]) -> Optional[dict]:
        """GeoJSON LineString following the stops in order, or None."""
        raise NotImplementedError


def _dedupe_consecutive(coords: Sequence[LatLon]) -> List[LatLon]:
    out: List[LatLon] = []
    for c in coords:
        if not out or (round(c[0], 6), round(c[1], 6)) != (round(out[-1][0], 6), round(out[-1][1], 6)):
            out.append(c)
    return out


class HaversineProvider(TravelProvider):
    """Straight-line distance x circuity factor / average speed."""

    def __init__(self, avg_speed_kmh: float = 50.0, circuity: float = 1.3):
        self.speed = avg_speed_kmh
        self.circuity = circuity

    def km_matrix(self, coords: Sequence[LatLon]) -> np.ndarray:
        lat = np.radians([c[0] for c in coords])
        lon = np.radians([c[1] for c in coords])
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def duration_matrix_mins(self, coords):
        km = self.km_matrix(coords) * self.circuity
        return np.ceil(km / self.speed * 60).astype(int)

    def line_geometry(self, coords):
        pts = _dedupe_consecutive(coords)
        if len(pts) < 2:
            return None
        return {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in pts]}


class OsrmProvider(TravelProvider):
    """Calls an OSRM server's /table and /route services.

    session: a requests.Session (in Foundry, the client from the imported
    Source). max_table_size must not exceed the server's --max-table-size
    (OSRM default is 100 locations per request).
    """

    def __init__(
        self,
        base_url: str,
        session=None,
        profile: str = "driving",
        max_table_size: int = 100,
        max_route_waypoints: int = 100,
        timeout_s: int = 60,
        retries: int = 3,
        fallback: Optional[TravelProvider] = None,
    ):
        if session is None:
            import requests
            session = requests.Session()
        self.base = base_url.rstrip("/")
        self.session = session
        self.profile = profile
        self.max_table = max(2, max_table_size)
        self.max_route = max(2, max_route_waypoints)
        self.timeout = timeout_s
        self.retries = retries
        self.fallback = fallback or HaversineProvider()

    # -- HTTP --------------------------------------------------------------
    @staticmethod
    def _coord_str(coords: Sequence[LatLon]) -> str:
        return ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)

    def _get(self, service: str, coords: Sequence[LatLon], params: dict) -> dict:
        url = f"{self.base}/{service}/v1/{self.profile}/{self._coord_str(coords)}"
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise RuntimeError(f"OSRM {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                if data.get("code") != "Ok":
                    # 4xx-style errors (NoRoute, TooBig, InvalidQuery) won't fix themselves.
                    raise ValueError(f"OSRM {service} error {data.get('code')}: {data.get('message')}")
                return data
            except ValueError:
                raise
            except Exception as e:  # network / 5xx: retry with backoff
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OSRM {service} failed after {self.retries} attempts: {last_err}")

    # -- Matrix ------------------------------------------------------------
    def duration_matrix_mins(self, coords):
        n = len(coords)
        out = np.full((n, n), np.nan)
        # Each request carries at most max_table locations: block I as sources,
        # block J as destinations.
        block = self.max_table if n <= self.max_table else self.max_table // 2
        starts = list(range(0, n, block))
        for si in starts:
            I = list(range(si, min(si + block, n)))
            for sj in starts:
                J = list(range(sj, min(sj + block, n)))
                if I == J:
                    locs = [coords[k] for k in I]
                    params = {"annotations": "duration"}
                else:
                    locs = [coords[k] for k in I] + [coords[k] for k in J]
                    params = {
                        "annotations": "duration",
                        "sources": ";".join(str(k) for k in range(len(I))),
                        "destinations": ";".join(str(len(I) + k) for k in range(len(J))),
                    }
                data = self._get("table", locs, params)
                durs = np.array(data["durations"], dtype=float)  # seconds; null -> nan
                out[np.ix_(I, J)] = durs

        missing = np.isnan(out)
        if missing.any():
            log.warning("OSRM returned %d unroutable pairs; using straight-line estimate", int(missing.sum()))
            out[missing] = self.fallback.duration_matrix_mins(coords)[missing] * 60
        np.fill_diagonal(out, 0)
        return np.ceil(out / 60).astype(int)

    # -- Geometry ------------------------------------------------------------
    def line_geometry(self, coords):
        pts = _dedupe_consecutive(coords)
        if len(pts) < 2:
            return None
        line: List[List[float]] = []
        step = self.max_route - 1
        try:
            for s in range(0, len(pts) - 1, step):
                chunk = pts[s : s + self.max_route]
                data = self._get("route", chunk, {"overview": "full", "geometries": "geojson"})
                seg = data["routes"][0]["geometry"]["coordinates"]
                line.extend(seg if not line else seg[1:])
        except Exception as e:
            log.warning("OSRM route geometry failed (%s); drawing straight lines", e)
            return self.fallback.line_geometry(pts)
        return {"type": "LineString", "coordinates": line}
