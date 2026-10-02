"""Download and audit ODbL OpenStreetMap airport geometry, without labels.

Usage:
  python osm_features.py --audit
  python osm_features.py --fetch --airport LFPG --year 2025
  python osm_features.py --fetch-all

Fetches are serial, cached, and limited to one attempt per airport/snapshot.
Historical snapshots precede the respective challenge year. This script never
loads taxi-time targets, trains a model, or changes a competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ENDPOINT = "https://overpass.private.coffee/api/interpreter"
LICENSE_URL = "https://www.openstreetmap.org/copyright"
OVERPASS_DOC = "https://wiki.openstreetmap.org/wiki/Overpass_API"
SNAPSHOTS = {2025: "2025-01-01T00:00:00Z", 2026: "2026-01-01T00:00:00Z"}
# Tight airport boxes, in Overpass south/west/north/east order. They exclude
# nearby aerodromes where practical; all actual stand matches are audited.
BBOX = {
    "EDDF": (49.98, 8.48, 50.08, 8.67),
    "EDDM": (48.31, 11.72, 48.40, 11.88),
    "EGLL": (51.44, -0.52, 51.50, -0.39),
    "EHAM": (52.28, 4.69, 52.36, 4.84),
    "LEBL": (41.26, 2.04, 41.34, 2.13),
    "LEMD": (40.45, -3.63, 40.55, -3.49),
    "LFPG": (48.96, 2.47, 49.05, 2.65),
    "LIRF": (41.76, 12.20, 41.84, 12.28),
    "LTFM": (41.22, 28.68, 41.32, 28.85),
    "LSZH": (47.42, 8.51, 47.50, 8.61),
}
RUNWAY_TOKEN = re.compile(r"(?<!\d)(0[1-9]|[12]\d|3[0-6])([LCR]?)(?!\d)")


def query(airport: str, year: int) -> str:
    south, west, north, east = BBOX[airport]
    bbox = f"{south},{west},{north},{east}"
    return (f'[out:json][timeout:60][date:"{SNAPSHOTS[year]}"];'
            f'(node["aeroway"="parking_position"]({bbox});'
            f'way["aeroway"="parking_position"]({bbox});'
            f'way["aeroway"="runway"]({bbox}););out geom;')


def raw_path(output_dir: Path, airport: str, year: int) -> Path:
    return output_dir / "raw" / f"{airport}_{year}.json"


def fetch(output_dir: Path, airport: str, year: int, timeout: int = 45) -> dict:
    path = raw_path(output_dir, airport, year)
    if path.exists():
        return {"airport": airport, "year": year, "status": "cached", "file": str(path)}
    q = query(airport, year)
    body = urllib.parse.urlencode({"data": q}).encode("ascii")
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"User-Agent": "PRC-Data-Challenge-2026-OSM-Research/1.0",
                 "Content-Type": "application/x-www-form-urlencoded"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            http_status = response.status
        data = json.loads(payload)
        if "elements" not in data or not data["elements"]:
            raise ValueError("Overpass response has no map elements")
        if "remark" in data:
            raise ValueError(f'Overpass remark: {data["remark"]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        meta = {"airport": airport, "year": year, "snapshot": SNAPSHOTS[year],
                "endpoint": ENDPOINT, "query": q, "license": "ODbL 1.0",
                "license_url": LICENSE_URL, "documentation": OVERPASS_DOC,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": http_status, "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "elements": len(data["elements"]),
                "osm_base": data.get("osm3s", {}).get("timestamp_osm_base")}
        path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"airport": airport, "year": year, "status": "downloaded", **meta}
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"airport": airport, "year": year, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}


def canonical_ref(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    ref = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    for prefix in ("PARKINGPOSITION", "PARKINGSTAND", "STAND"):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
    return ref


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.degrees(math.atan2(x, y)) % 360


def location(element: dict) -> tuple[float, float] | None:
    if element["type"] == "node":
        return float(element["lat"]), float(element["lon"])
    geometry = element.get("geometry", [])
    if geometry:
        # The parking-position way is normally drawn towards the wheel stop.
        return float(geometry[-1]["lat"]), float(geometry[-1]["lon"])
    return None


def parse_map(path: Path) -> tuple[dict[str, tuple[float, float]],
                                    dict[str, tuple[float, float]], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stand_candidates: dict[str, list[tuple[float, float]]] = {}
    runways: dict[str, tuple[float, float, float]] = {}
    for element in raw.get("elements", []):
        tags = element.get("tags", {})
        kind = tags.get("aeroway")
        if kind == "parking_position":
            pos = location(element)
            for ref in re.split(r"[;/]", tags.get("ref", "")):
                key = canonical_ref(ref)
                if key and pos:
                    stand_candidates.setdefault(key, []).append(pos)
        elif kind == "runway" and element["type"] == "way":
            geom = element.get("geometry", [])
            if len(geom) < 2:
                continue
            first = (float(geom[0]["lat"]), float(geom[0]["lon"]))
            last = (float(geom[-1]["lat"]), float(geom[-1]["lon"]))
            length = haversine_m(*first, *last)
            if length < 400:
                continue
            azimuth = bearing_deg(*first, *last)
            for match in RUNWAY_TOKEN.finditer(tags.get("ref", "")):
                key = match.group(1) + match.group(2)
                direction = (int(match.group(1)) * 10) % 360
                difference = abs((azimuth - direction + 180) % 360 - 180)
                threshold = first if difference <= 90 else last
                if key not in runways or length > runways[key][2]:
                    runways[key] = (*threshold, length)
    stands: dict[str, tuple[float, float]] = {}
    ambiguous = 0
    for key, positions in stand_candidates.items():
        anchor = positions[0]
        if all(haversine_m(*anchor, *pos) < 100 for pos in positions[1:]):
            stands[key] = (float(np.mean([p[0] for p in positions])),
                           float(np.mean([p[1] for p in positions])))
        else:
            ambiguous += 1
    runway_coords = {key: (entry[0], entry[1]) for key, entry in runways.items()}
    summary = {"osm_elements": len(raw.get("elements", [])),
               "stand_refs": len(stands), "ambiguous_stand_refs_excluded": ambiguous,
               "runway_heads": len(runway_coords)}
    return stands, runway_coords, summary


def local_keys(cache_dir: Path, year: int) -> pd.DataFrame:
    source = cache_dir / ("features.parquet" if year == 2025 else "ranking_features.parquet")
    frame = pl.scan_parquet(str(source)).select(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).collect().to_pandas()
    frame["ADEP_mvt"] = frame.ADEP_mvt.astype("string")
    frame["STAND_mvt"] = frame.STAND_mvt.astype("string")
    frame["RUNWAY_mvt"] = frame.RUNWAY_mvt.astype("string")
    return frame


def audit(output_dir: Path, cache_dir: Path) -> dict:
    report = {"purpose": "Label-free OSM stand/runway match audit",
              "snapshots": SNAPSHOTS, "license": "ODbL 1.0",
              "license_url": LICENSE_URL, "source_documentation": OVERPASS_DOC,
              "airports": {}}
    for year in SNAPSHOTS:
        frame = local_keys(cache_dir, year)
        frame["stand_key"] = frame.STAND_mvt.map(canonical_ref)
        frame["runway_key"] = frame.RUNWAY_mvt.map(canonical_ref)
        summary = {"rows": len(frame), "source": str(cache_dir / (
            "features.parquet" if year == 2025 else "ranking_features.parquet")),
            "airport": {}}
        rows = []
        for airport, airport_frame in frame.groupby("ADEP_mvt", sort=True, observed=True):
            path = raw_path(output_dir, str(airport), year)
            entry = {"rows": int(len(airport_frame)),
                     "stand_present_rows": int(airport_frame.STAND_mvt.notna().sum()),
                     "runway_present_rows": int(airport_frame.RUNWAY_mvt.notna().sum()),
                     "unique_stands": int(airport_frame.stand_key.nunique()),
                     "map_cached": path.exists()}
            if path.exists():
                stands, runways, map_summary = parse_map(path)
                entry.update(map_summary)
                stand_hit = airport_frame.stand_key.isin(stands)
                runway_hit = airport_frame.runway_key.isin(runways)
                entry["stand_match_rows"] = int(stand_hit.sum())
                entry["runway_match_rows"] = int(runway_hit.sum())
                entry["both_match_rows"] = int((stand_hit & runway_hit).sum())
                entry["stand_match_fraction"] = float(stand_hit.mean())
                entry["both_match_fraction"] = float((stand_hit & runway_hit).mean())
                pairs = airport_frame[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
                                       "stand_key", "runway_key"]].drop_duplicates().copy()
                pairs["osm_stand_lat"] = pairs.stand_key.map(lambda k: stands.get(k, (np.nan, np.nan))[0])
                pairs["osm_stand_lon"] = pairs.stand_key.map(lambda k: stands.get(k, (np.nan, np.nan))[1])
                pairs["osm_runway_threshold_lat"] = pairs.runway_key.map(
                    lambda k: runways.get(k, (np.nan, np.nan))[0])
                pairs["osm_runway_threshold_lon"] = pairs.runway_key.map(
                    lambda k: runways.get(k, (np.nan, np.nan))[1])
                pairs["osm_straight_line_m"] = [
                    haversine_m(a, b, c, d) if all(np.isfinite(z) for z in (a, b, c, d)) else np.nan
                    for a, b, c, d in zip(pairs.osm_stand_lat, pairs.osm_stand_lon,
                                           pairs.osm_runway_threshold_lat,
                                           pairs.osm_runway_threshold_lon)]
                pairs["snapshot"] = SNAPSHOTS[year]
                rows.append(pairs)
            summary["airport"][str(airport)] = entry
        report["airports"][str(year)] = summary
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(
                output_dir / f"crosswalk_{year}.parquet", index=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/osm"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--fetch-all", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--airport", choices=tuple(BBOX))
    parser.add_argument("--year", type=int, choices=tuple(SNAPSHOTS))
    args = parser.parse_args()
    if args.fetch and (args.airport is None or args.year is None):
        parser.error("--fetch requires --airport and --year")
    if args.fetch_all and args.fetch:
        parser.error("Choose either --fetch or --fetch-all")
    if args.fetch or args.fetch_all:
        targets = ([(args.airport, args.year)] if args.fetch else
                   [(airport, year) for year in SNAPSHOTS for airport in BBOX])
        results = []
        for i, (airport, year) in enumerate(targets):
            if i:
                time.sleep(1.5)
            result = fetch(args.output_dir, airport, year)
            results.append(result)
            print(json.dumps(result), flush=True)
            if result["status"] == "failed":
                print("Stopping after one failed public API request; cached maps remain reusable.", flush=True)
                break
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "fetch_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.audit or not (args.fetch or args.fetch_all):
        report = audit(args.output_dir, args.cache_dir)
        print(json.dumps({year: {airport: info["map_cached"] for airport, info in part["airport"].items()}
                          for year, part in report["airports"].items()}, indent=2))


if __name__ == "__main__":
    main()
