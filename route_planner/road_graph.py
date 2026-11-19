"""Road routing inside the transform, with no routing server.

build_road_graph() turns OpenStreetMap extracts (.osm.pbf files, e.g. from
Geofabrik) into a compact directed graph: node coordinates plus edges with
drive seconds. RoadGraphProvider runs Dijkstra on that graph (scipy) to get
drive-time matrices and the road-following line for each route.
"""
import logging
import re
from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .travel import HaversineProvider, TravelProvider, _dedupe_consecutive

log = logging.getLogger(__name__)
EARTH_R = 6_371_000.0

# Default free-flow speeds (km/h) by OSM highway type, used when a road has
# no usable maxspeed tag. Roughly US-urban/rural averages; tune to taste.
DEFAULT_SPEEDS_KMH: Dict[str, float] = {
    "motorway": 100, "motorway_link": 60,
    "trunk": 85, "trunk_link": 50,
    "primary": 65, "primary_link": 45,
    "secondary": 55, "secondary_link": 40,
    "tertiary": 45, "tertiary_link": 35,
    "unclassified": 40, "residential": 35, "living_street": 15,
    "service": 15, "road": 30,
}
SKIP_SERVICE = {"driveway", "drive-through", "emergency_access"}
NO_ACCESS = {"no", "private"}
# Real-world travel is slower than free-flow (lights, turns, traffic).
DEFAULT_SPEED_FACTOR = 0.85


def _parse_maxspeed(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph)?", v.split(";")[0])
    if not m:
        return None
    kmh = float(m.group(1)) * (1.609 if m.group(2) else 1.0)
    return kmh if 5 <= kmh <= 140 else None


def _oneway(tags, highway: str) -> int:
    """1 = forward only, -1 = reverse only, 0 = both directions."""
    v = tags.get("oneway")
    if v in ("yes", "true", "1"):
        return 1
    if v == "-1":
        return -1
    if v in ("no", "false", "0"):
        return 0
    if highway in ("motorway", "motorway_link") or tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 0


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_road_graph(
    osm_paths: Iterable[str],
    speeds_kmh: Dict[str, float] = DEFAULT_SPEEDS_KMH,
    speed_factor: float = DEFAULT_SPEED_FACTOR,
    keep_largest_component: bool = True,
) -> Dict[str, np.ndarray]:
    """Read drivable roads from one or more OSM files (.pbf or .osm).

    Returns arrays: lat, lon (per node) and u, v, secs (per directed edge).
    Several files (e.g. neighbouring states) are merged on OSM node id.
    """
    import osmium

    ids, lats, lons = array("q"), array("d"), array("d")
    way_len, way_kmh, way_dir = array("q"), array("f"), array("b")

    class Collector(osmium.SimpleHandler):
        def way(self, w):
            t = w.tags
            hw = t.get("highway")
            if hw not in speeds_kmh:
                return
            if t.get("area") == "yes" or t.get("access") in NO_ACCESS or t.get("motor_vehicle") in NO_ACCESS:
                return
            if hw == "service" and t.get("service") in SKIP_SERVICE:
                return
            pts = []
            for n in w.nodes:
                if not n.location.valid():
                    return
                pts.append((n.ref, n.location.lat, n.location.lon))
            if len(pts) < 2:
                return
            for ref, la, lo in pts:
                ids.append(ref)
                lats.append(la)
                lons.append(lo)
            way_len.append(len(pts))
            way_kmh.append((_parse_maxspeed(t.get("maxspeed")) or speeds_kmh[hw]) * speed_factor)
            way_dir.append(_oneway(t, hw))

    for p in osm_paths:
        log.info("reading %s", p)
        Collector().apply_file(p, locations=True)

    if not way_len:
        raise ValueError("No drivable roads found in the OSM files")

    ids_a = np.frombuffer(ids, dtype=np.int64)
    lat_a = np.frombuffer(lats, dtype=np.float64)
    lon_a = np.frombuffer(lons, dtype=np.float64)
    lens = np.frombuffer(way_len, dtype=np.int64)
    way_of_pt = np.repeat(np.arange(len(lens)), lens)

    _, inv = np.unique(ids_a, return_inverse=True)
    n_nodes = int(inv.max()) + 1
    node_lat = np.empty(n_nodes)
    node_lon = np.empty(n_nodes)
    node_lat[inv] = lat_a
    node_lon[inv] = lon_a

    # Consecutive points of the same way form segments.
    i = np.nonzero(way_of_pt[:-1] == way_of_pt[1:])[0]
    w = way_of_pt[i]
    a, b = inv[i], inv[i + 1]
    metres = _haversine_m(lat_a[i], lon_a[i], lat_a[i + 1], lon_a[i + 1])
    kmh = np.frombuffer(way_kmh, dtype=np.float32)[w].astype(float)
    secs = np.maximum(metres / (kmh / 3.6), 0.1)
    direction = np.frombuffer(way_dir, dtype=np.int8)[w]

    fwd = direction >= 0
    rev = direction <= 0
    u = np.concatenate([a[fwd], b[rev]])
    v = np.concatenate([b[fwd], a[rev]])
    s = np.concatenate([secs[fwd], secs[rev]])

    # Keep the fastest of any duplicate edges (overlapping files, double-mapped ways).
    keep = u != v
    u, v, s = u[keep], v[keep], s[keep]
    order = np.lexsort((s, v, u))
    u, v, s = u[order], v[order], s[order]
    first = np.ones(len(u), dtype=bool)
    first[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    u, v, s = u[first], v[first], s[first]

    if keep_largest_component:
        g = csr_matrix((s, (u, v)), shape=(n_nodes, n_nodes))
        _, labels = connected_components(g, directed=True, connection="strong")
        main = labels == np.bincount(labels).argmax()
        remap = np.full(n_nodes, -1)
        remap[main] = np.arange(int(main.sum()))
        e = main[u] & main[v]
        u, v, s = remap[u[e]], remap[v[e]], s[e]
        node_lat, node_lon = node_lat[main], node_lon[main]

    log.info("road graph: %d nodes, %d edges", len(node_lat), len(u))
    return {
        "lat": node_lat.astype(np.float64),
        "lon": node_lon.astype(np.float64),
        "u": u.astype(np.int32),
        "v": v.astype(np.int32),
        "secs": s.astype(np.float32),
    }


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


class _Sub:
    """The part of the graph around one driver's stores."""

    def __init__(self, g, lat, lon, bbox):
        self.g, self.lat, self.lon, self.bbox = g, lat, lon, bbox
        _, labels = connected_components(g, directed=True, connection="strong")
        self.snap_ids = np.nonzero(labels == np.bincount(labels).argmax())[0]
        self.tree = cKDTree(_xyz(lat[self.snap_ids], lon[self.snap_ids]))

    def contains(self, coords):
        la0, la1, lo0, lo1 = self.bbox
        return all(la0 <= a <= la1 and lo0 <= b <= lo1 for a, b in coords)


class RoadGraphProvider(TravelProvider):
    """Drive times and route lines from a road graph made by build_road_graph().

    For each driver, only roads within buffer_km of the home and stores are
    used, which keeps Dijkstra fast. Stores are snapped to the nearest road;
    the gap is charged at snap_speed_kmh (parking lots, driveways).
    """

    def __init__(
        self,
        graph: Dict[str, np.ndarray],
        buffer_km: float = 25.0,
        snap_speed_kmh: float = 15.0,
        max_snap_m: float = 3000.0,
        batch: int = 32,
        fallback: Optional[TravelProvider] = None,
    ):
        self.lat, self.lon = graph["lat"], graph["lon"]
        self.u, self.v, self.secs = graph["u"], graph["v"], graph["secs"]
        self.buffer_km = buffer_km
        self.snap_mps = snap_speed_kmh / 3.6
        self.max_snap_m = max_snap_m
        self.batch = batch
        self.fallback = fallback or HaversineProvider()
        self._sub: Optional[_Sub] = None

    # -- graph clipping ------------------------------------------------------
    def _subgraph(self, coords: Sequence[Tuple[float, float]]) -> _Sub:
        if self._sub is not None and self._sub.contains(coords):
            return self._sub
        la = [c[0] for c in coords]
        lo = [c[1] for c in coords]
        dlat = self.buffer_km / 111.0
        dlon = self.buffer_km / (111.0 * max(0.2, np.cos(np.radians(np.mean(la)))))
        bbox = (min(la) - dlat, max(la) + dlat, min(lo) - dlon, max(lo) + dlon)
        mask = (self.lat >= bbox[0]) & (self.lat <= bbox[1]) & (self.lon >= bbox[2]) & (self.lon <= bbox[3])
        keep = np.nonzero(mask)[0]
        if len(keep) < 2:
            raise ValueError("No roads near these locations: is the OSM extract for the right region?")
        remap = np.full(len(self.lat), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        e = mask[self.u] & mask[self.v]
        g = csr_matrix((self.secs[e].astype(float), (remap[self.u[e]], remap[self.v[e]])),
                       shape=(len(keep), len(keep)))
        self._sub = _Sub(g, self.lat[keep], self.lon[keep], bbox)
        return self._sub

    def _snap(self, sub: _Sub, coords):
        d, k = sub.tree.query(_xyz([c[0] for c in coords], [c[1] for c in coords]))
        metres = d * EARTH_R  # chord ~ arc at these distances
        far = metres > self.max_snap_m
        if far.any():
            log.warning("%d location(s) are over %.0f m from the nearest road", int(far.sum()), self.max_snap_m)
        return sub.snap_ids[k], metres / self.snap_mps

    # -- TravelProvider ------------------------------------------------------
    def duration_matrix_mins(self, coords):
        sub = self._subgraph(coords)
        nodes, snap_s = self._snap(sub, coords)
        uniq, inv = np.unique(nodes, return_inverse=True)
        n = len(coords)
        secs = np.empty((n, n))
        for s0 in range(0, len(uniq), self.batch):
            src = uniq[s0 : s0 + self.batch]
            dist = dijkstra(sub.g, directed=True, indices=src)[:, nodes]  # (batch, n)
            for r, row in enumerate(range(s0, s0 + len(src))):
                secs[inv == row] = dist[r]
        secs = secs + snap_s[:, None] + snap_s[None, :]
        bad = ~np.isfinite(secs)
        if bad.any():
            log.warning("%d pairs have no road route; using straight-line estimate", int(bad.sum()))
            secs[bad] = self.fallback.duration_matrix_mins(coords)[bad] * 60
        np.fill_diagonal(secs, 0)
        return np.ceil(secs / 60).astype(int)

    def line_geometry(self, coords):
        pts = _dedupe_consecutive(coords)
        if len(pts) < 2:
            return None
        sub = self._subgraph(pts)
        nodes, _ = self._snap(sub, pts)
        src = np.unique(nodes[:-1])
        _, pred = dijkstra(sub.g, directed=True, indices=src, return_predecessors=True)
        row_of = {int(s): r for r, s in enumerate(src)}

        line: List[List[float]] = [[pts[0][1], pts[0][0]]]
        for leg in range(len(pts) - 1):
            a, b = int(nodes[leg]), int(nodes[leg + 1])
            p = pred[row_of[a]]
            path, cur = [], b
            while cur != a and cur >= 0:
                path.append(cur)
                cur = p[cur]
            if cur != a:  # unreachable: straight segment
                line.append([pts[leg + 1][1], pts[leg + 1][0]])
                continue
            path.append(a)
            path.reverse()
            line.extend([float(sub.lon[x]), float(sub.lat[x])] for x in path)
            line.append([pts[leg + 1][1], pts[leg + 1][0]])
        # Drop exact repeats produced by snapping.
        out = [line[0]]
        for c in line[1:]:
            if c != out[-1]:
                out.append(c)
        return {"type": "LineString", "coordinates": out}


def save_graph(graph: Dict[str, np.ndarray], fileobj) -> None:
    np.savez_compressed(fileobj, **graph)


def load_graph(fileobj) -> Dict[str, np.ndarray]:
    with np.load(fileobj) as z:
        return {k: z[k] for k in ("lat", "lon", "u", "v", "secs")}
