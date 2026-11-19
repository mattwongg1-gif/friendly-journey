"""Build the road graph from OpenStreetMap extracts.

Input: a dataset holding one or more .osm.pbf files, uploaded by hand
(e.g. https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf).
Upload every state your territories touch; overlapping borders are merged.

Output: a dataset holding one file, road_graph.npz, read by weekly_routes.
Rebuild when you upload fresher OSM files (monthly is plenty).
"""
import os
import re
import shutil
import tempfile

from transforms.api import Input, Output, transform

from myproject.route_planner.road_graph import build_road_graph, save_graph

FOLDER = "/YourOrg/YourProject/route_planning"
OSM_FILES_DS = f"{FOLDER}/osm_extracts"


@transform(
    graph_out=Output(f"{FOLDER}/road_graph"),
    osm_files=Input(OSM_FILES_DS),
)
def compute(graph_out, osm_files):
    fs = osm_files.filesystem()
    paths = [f.path for f in fs.ls() if re.search(r"\.(osm\.pbf|pbf|osm)$", f.path)]
    if not paths:
        raise ValueError(f"No .osm.pbf files found in {OSM_FILES_DS}")

    with tempfile.TemporaryDirectory() as tmp:
        local = []
        for p in paths:
            dst = os.path.join(tmp, os.path.basename(p))
            with fs.open(p, "rb") as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            local.append(dst)

        graph = build_road_graph(local)

        npz = os.path.join(tmp, "road_graph.npz")
        with open(npz, "wb") as fh:
            save_graph(graph, fh)
        with open(npz, "rb") as src, graph_out.filesystem().open("road_graph.npz", "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
