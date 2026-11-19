"""Weekly route plan: run every Monday by a Foundry schedule.

Reads the datasets backing Driver, Territory and Site plus a driver
availability table, plans the week for every driver, and writes:
  route_plan         -> backs the Route object type   (one row per driver-day)
  stop_plan          -> backs the Stop object type    (one row per store visit)
  unscheduled_sites  -> due stores that did not fit, with the reason
  site_targets       -> planned visit date per store (feeds last_targ_visit_date)

route_plan and stop_plan keep history: each run replaces only the days it
plans (today through Sunday) and keeps every other row from earlier runs.
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from transforms.api import Input, Output, incremental, transform

from myproject.route_planner import Columns, HaversineProvider, PlannerConfig, plan_week
from myproject.route_planner.road_graph import RegionalRoadGraphs, load_graph
from myproject.route_planner.planner import (
    ROUTE_COLUMNS, STOP_COLUMNS, TARGET_COLUMNS, UNSCHEDULED_COLUMNS, week_bounds,
)

# ---- Edit these ------------------------------------------------------------
CFG = PlannerConfig()        # rules: work days, minutes per day, restock time ...
COLS = Columns()             # column names in your backing datasets
FOLDER = "/YourOrg/YourProject/route_planning"   # where outputs are written
DRIVER_DS = "/YourOrg/YourProject/ontology/driver"       # path or RID
TERRITORY_DS = "/YourOrg/YourProject/ontology/territory"
SITE_DS = "/YourOrg/YourProject/ontology/site"
# Optional. None = everyone works the default days (Mon-Fri, 480 min).
AVAILABILITY_DS = None   # e.g. f"{FOLDER}/driver_availability"
# Optional. None = straight-line drive estimates. Set to the road graph
# dataset built by datasets/road_graph.py (one file per state) for real
# road times and lines.
ROAD_GRAPH_DS = None     # e.g. f"{FOLDER}/road_graph"
# -----------------------------------------------------------------------------

ROUTE_SCHEMA = T.StructType([
    T.StructField("route_id", T.StringType()),
    T.StructField("driver_id", T.StringType()),
    T.StructField("date", T.DateType()),
    T.StructField("week_start", T.DateType()),
    T.StructField("drive_time_mins", T.IntegerType()),
    T.StructField("work_time_mins", T.IntegerType()),
    T.StructField("total_time_mins", T.IntegerType()),
    T.StructField("max_work_mins", T.IntegerType()),
    T.StructField("over_budget_mins", T.IntegerType()),
    T.StructField("stop_count", T.IntegerType()),
    T.StructField("restock_count", T.IntegerType()),
    T.StructField("start_location", T.StringType()),   # geopoint "lat,lon"
    T.StructField("end_location", T.StringType()),     # geopoint "lat,lon"
    T.StructField("geometry", T.StringType()),         # geoshape: GeoJSON LineString
    T.StructField("planned_at", T.StringType()),
])
STOP_SCHEMA = T.StructType([
    T.StructField("stop_id", T.StringType()),
    T.StructField("route_id", T.StringType()),
    T.StructField("driver_id", T.StringType()),
    T.StructField("date", T.DateType()),
    T.StructField("site_id", T.StringType()),
    T.StructField("seq_ind", T.IntegerType()),
    T.StructField("profit", T.DecimalType(18, 2)),
    T.StructField("arrive_min", T.IntegerType()),          # minutes after shift start
    T.StructField("depart_min", T.IntegerType()),
    T.StructField("drive_mins_from_prev", T.IntegerType()),
    T.StructField("restock_before", T.BooleanType()),       # went home to reload first
    T.StructField("required", T.BooleanType()),
])
UNSCHEDULED_SCHEMA = T.StructType([
    T.StructField("week_start", T.DateType()),
    T.StructField("driver_id", T.StringType()),
    T.StructField("site_id", T.StringType()),
    T.StructField("required", T.BooleanType()),
    T.StructField("next_due_date", T.DateType()),
    T.StructField("days_overdue", T.IntegerType()),
    T.StructField("reason", T.StringType()),
])
TARGET_SCHEMA = T.StructType([
    T.StructField("site_id", T.StringType()),
    T.StructField("targ_visit_date", T.DateType()),
    T.StructField("route_id", T.StringType()),
    T.StructField("driver_id", T.StringType()),
    T.StructField("week_start", T.DateType()),
])


def _to_spark(spark, pdf: pd.DataFrame, columns, schema):
    """pandas -> Spark with NaN -> null and numpy scalars -> Python types."""
    from decimal import Decimal

    decimal_cols = {f.name for f in schema.fields if isinstance(f.dataType, T.DecimalType)}

    def clean(col, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, np.generic):
            v = v.item()
        if col in decimal_cols:
            return Decimal(str(round(float(v), 2)))
        return v

    rows = [tuple(clean(c, r[c]) for c in columns) for r in pdf.to_dict("records")]
    return spark.createDataFrame(rows, schema)


def _merge_history(output, new_df, schema, plan_from, week_end):
    """Keep earlier rows outside the replanned window, then replace the output."""
    previous = output.dataframe("previous", schema)  # empty on the first run
    kept = previous.filter((F.col("date") < F.lit(plan_from)) | (F.col("date") > F.lit(week_end)))
    merged = kept.unionByName(new_df).localCheckpoint(eager=True)
    output.set_mode("replace")
    output.write_dataframe(merged)


def _provider(road_graph):
    fallback = HaversineProvider(CFG.fallback_speed_kmh, CFG.fallback_circuity)
    if road_graph is None:
        return fallback
    fs = road_graph.filesystem()
    with fs.open("index.json", "r") as fh:
        index = json.load(fh)

    def load_state(name):
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, f"{name}.npz")
            with fs.open(f"{name}.npz", "rb") as src, open(local, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            return load_graph(local)

    # Loads only the states near each driver; joins neighbours at borders.
    return RegionalRoadGraphs(index, load_state, fallback=fallback)


_inputs = {"drivers": Input(DRIVER_DS), "territories": Input(TERRITORY_DS), "sites": Input(SITE_DS)}
if AVAILABILITY_DS:
    _inputs["availability"] = Input(AVAILABILITY_DS)
if ROAD_GRAPH_DS:
    _inputs["road_graph"] = Input(ROAD_GRAPH_DS)


@incremental(snapshot_inputs=list(_inputs), semantic_version=1)
@transform(
    routes_out=Output(f"{FOLDER}/route_plan"),
    stops_out=Output(f"{FOLDER}/stop_plan"),
    unscheduled_out=Output(f"{FOLDER}/unscheduled_sites"),
    targets_out=Output(f"{FOLDER}/site_targets"),
    **_inputs,
)
def compute(ctx, routes_out, stops_out, unscheduled_out, targets_out,
            drivers, territories, sites, availability=None, road_graph=None):
    today = datetime.now(ZoneInfo(CFG.timezone)).date()
    week_start, week_end = week_bounds(today)
    plan_from = max(today, week_start)

    provider = _provider(road_graph)

    routes, stops, unscheduled, targets = plan_week(
        drivers.dataframe().toPandas(),
        territories.dataframe().toPandas(),
        sites.dataframe().toPandas(),
        availability.dataframe().toPandas() if availability is not None else None,
        provider,
        today,
        CFG,
        COLS,
    )

    spark = ctx.spark_session
    _merge_history(routes_out, _to_spark(spark, routes, ROUTE_COLUMNS, ROUTE_SCHEMA),
                   ROUTE_SCHEMA, plan_from, week_end)
    _merge_history(stops_out, _to_spark(spark, stops, STOP_COLUMNS, STOP_SCHEMA),
                   STOP_SCHEMA, plan_from, week_end)

    # These two only describe the current week.
    unscheduled_out.set_mode("replace")
    unscheduled_out.write_dataframe(_to_spark(spark, unscheduled, UNSCHEDULED_COLUMNS, UNSCHEDULED_SCHEMA))
    targets_out.set_mode("replace")
    targets_out.write_dataframe(_to_spark(spark, targets, TARGET_COLUMNS, TARGET_SCHEMA))
