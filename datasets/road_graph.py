"""Build road graphs from OpenStreetMap extracts, one per state.

Input: a dataset holding the .osm.pbf files you uploaded, one per state
(e.g. https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf).

Output: a dataset holding
  <state>.npz   one road graph per uploaded file
  index.json    which ~28 km grid cells each state has roads in
weekly_routes uses index.json to load only the states each driver needs, and
joins neighbouring states at the border (roads crossing it are in both files).

Files are processed one at a time, so memory depends on the largest state,
not the total upload. Rebuild after uploading fresher OSM files (monthly is
plenty).
"""
import json
import logging
import os
import re
import shutil
import tempfile

from transforms.api import Input, Output, transform

from myproject.route_planner.road_graph import build_road_graph, coverage_cells, save_graph

log = logging.getLogger(__name__)

FOLDER = "/YourOrg/YourProject/route_planning"
OSM_FILES_DS = f"{FOLDER}/osm_extracts"
CHUNK = 16 * 1024 * 1024


def _state_name(path: str) -> str:
    """'illinois-latest.osm.pbf' -> 'illinois'"""
    name = os.path.basename(path)
    name = re.sub(r"\.(osm\.pbf|pbf|osm)$", "", name)
    return re.sub(r"-latest$", "", name)


@transform(
    graph_out=Output(f"{FOLDER}/road_graph"),
    osm_files=Input(OSM_FILES_DS),
)
def compute(graph_out, osm_files):
    fs = osm_files.filesystem()
    paths = sorted(f.path for f in fs.ls() if re.search(r"\.(osm\.pbf|pbf|osm)$", f.path))
    if not paths:
        raise ValueError(f"No .osm.pbf files found in {OSM_FILES_DS}")

    out_fs = graph_out.filesystem()
    index = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths:
            state = _state_name(p)
            if state in index:
                raise ValueError(f"Two files give the state name '{state}'; rename one")
            local = os.path.join(tmp, os.path.basename(p))
            with fs.open(p, "rb") as src, open(local, "wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK)

            # Keep every road (no island removal) so border roads can join
            # up with the neighbouring state later.
            graph = build_road_graph([local], keep_largest_component=False)
            os.remove(local)

            npz = os.path.join(tmp, f"{state}.npz")
            with open(npz, "wb") as fh:
                save_graph(graph, fh)
            with open(npz, "rb") as src, out_fs.open(f"{state}.npz", "wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK)
            os.remove(npz)

            index[state] = coverage_cells(graph["lat"], graph["lon"])
            log.info("%s: %d nodes, %d edges, %d cells", state, len(graph["lat"]), len(graph["u"]), len(index[state]))
            del graph

    with out_fs.open("index.json", "w") as fh:
        json.dump(index, fh)
