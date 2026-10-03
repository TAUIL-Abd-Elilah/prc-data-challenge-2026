"""Build the unchanged NOAA weather input inside an isolated clean run.

Only pure processing functions are called from the original downloader.
The original downloader's main(), source-checkout output path and existing
weather artifacts are never used. All real modes require published source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

import download_noaa_weather as original

SOURCE = Path(__file__).resolve().parent
SPEC = SOURCE / "reports/clean_replication_weather_spec.json"
PINS = {
    "download_noaa_weather.py": "bbb8fe3dc64250fbc41519ba716913af3349d5724fc65e388d399a3b1d380525",
    "WEATHER_SOURCES.md": "da6c1f472813bcb95c8223054aa44e1086af8c69ea471cc1e4793b5e5f53ff0f",
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "requirements-lock.txt": "2d201a484bc8c8f7027e56f608da51114958dd5c628ab32a5f2018a437086254",
}
PROTOCOL = "parents/weather_protocol.json"
RECEIPT = "parents/weather_producer_receipt.json"
WEATHER = "data/external/weather.parquet"
MANIFEST = "data/external/weather_sources.json"
ROWS = 102480


class NOAARedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urlsplit(newurl)
        if destination.scheme != "https" or destination.hostname != "www.ncei.noaa.gov":
            raise ValueError("NOAA redirect refused before following an outside URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def root_path(value: Path) -> Path:
    root = value.resolve(strict=True)
    if (not root.is_dir() or root == SOURCE or root in SOURCE.parents
            or SOURCE in root.parents):
        raise ValueError("Run root must exist and be disjoint from the source checkout")
    return root


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=False)
    if root not in path.parents:
        raise ValueError("Output/input path escapes the isolated run root")
    return path


def write_new(root: Path, name: str, payload: bytes) -> None:
    path = inside(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Check again after parent creation, including existing junctions/symlinks.
    if inside(root, name) != path:
        raise ValueError("Output parent changed")
    with path.open("xb") as handle:
        handle.write(payload)
    if path.read_bytes() != payload:
        raise ValueError("Exclusive output readback differs")


def source_snapshot(published: str | None = None) -> dict:
    actual = {name: sha(SOURCE / name) for name in PINS}
    if actual != PINS:
        raise ValueError("Original processing or documented source bytes changed")
    actual.update({"replica_weather.py": sha(Path(__file__)),
                   "reports/clean_replication_weather_spec.json": sha(SPEC)})
    spec = read(SPEC)
    if spec.get("original_source_sha256") != PINS:
        raise ValueError("Prospective weather source contract changed")
    if published is not None:
        if published != actual["replica_weather.py"]:
            raise ValueError("Published weather wrapper SHA differs")
        for name, digest in actual.items():
            data = subprocess.check_output(["git", "-C", str(SOURCE), "show", "HEAD:" + name])
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Source/spec must be committed before real execution")
    return actual


def inputs() -> list[dict]:
    return [{"airport": airport, "station_id": station,
             "station_name": label, "year": year,
             "url": f"{original.BASE_URL}/{year}/parquet/GHCNh_{station}_{year}.parquet",
             "path": f"data/external/noaa-station-years/GHCNh_{station}_{year}.parquet"}
            for airport, (station, label) in original.STATIONS.items()
            for year in (2025, 2026)]


def memory_guard() -> None:
    if psutil.virtual_memory().available < 10 * 2 ** 30:
        raise MemoryError("Ten GiB available physical RAM required before weather value reads")


def prepare(root: Path, published: str) -> dict:
    before = source_snapshot(published)
    value = {"schema_version": 1, "status": "prepared_before_downloads_or_value_reads",
             "source_sha256": before, "station_years": inputs(),
             "license": "CC0-1.0", "expected_rows": ROWS,
             "original_processing_unchanged": True,
             "competition_labels_read": False, "leaderboard_used": False}
    if source_snapshot(published) != before:
        raise ValueError("Source changed during preparation")
    write_new(root, PROTOCOL, json_bytes(value))
    return {"protocol_sha256": sha(inside(root, PROTOCOL)), "station_years": 20}


def protocol(root: Path, published: str, published_protocol: str) -> tuple[dict, dict]:
    before = source_snapshot(published)
    path = inside(root, PROTOCOL)
    value = read(path)
    expected = {"schema_version": 1, "status": "prepared_before_downloads_or_value_reads",
                "source_sha256": before, "station_years": inputs(),
                "license": "CC0-1.0", "expected_rows": ROWS,
                "original_processing_unchanged": True,
                "competition_labels_read": False, "leaderboard_used": False}
    if sha(path) != published_protocol or value != expected:
        raise ValueError("Published prepared weather protocol differs")
    return value, before


def grid_for(raw_path: Path, item: dict) -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(raw_path, columns=original.RAW_COLUMNS)
    raw = table.to_pandas()
    hourly = original.weather_hours(raw, item["year"])
    grid = pd.DataFrame(index=original.hour_grid(item["year"]))
    grid.index.name = "weather_hour_utc"
    grid = grid.join(hourly, how="left").reset_index()
    grid.insert(0, "airport", item["airport"])
    grid.insert(1, "wx_station_id", item["station_id"])
    grid["wx_obs_count"] = grid["wx_obs_count"].fillna(0).astype("int16")
    grid["wx_fog_proxy"] = ((grid["wx_visibility_km"] <= 1.0)
                            | grid["wx_fog_reported"].fillna(False))
    grid["wx_deicing_proxy"] = (grid["wx_temperature_c"] <= 3.0) & (
        (grid["wx_precip_mm"] > 0) | grid["wx_snow_reported"].fillna(False)
        | grid["wx_freezing_precip_reported"].fillna(False)
        | (grid["wx_snow_depth_mm"] > 0))
    for column in [c for c in grid if c.endswith("_reported") or c.endswith("_proxy")]:
        grid[column] = grid[column].fillna(False).astype(bool)
    details = {k: item[k] for k in ("airport", "station_id", "station_name", "year", "url")}
    details.update({"sha256": sha(raw_path), "source_records": len(raw),
                    "observed_hours": int((grid.wx_obs_count > 0).sum()),
                    "expected_hours": len(grid)})
    return grid, details


def weather_table(paths: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames, details = [], []
    for path, item in zip(paths, inputs(), strict=True):
        frame, detail = grid_for(path, item)
        frames.append(frame)
        details.append(detail)
    result = pd.concat(frames, ignore_index=True)
    if (len(result) != ROWS or result.duplicated(["airport", "weather_hour_utc"]).any()
            or set(result.airport) != set(original.STATIONS)):
        raise ValueError("Weather grid schema/airport/hour coverage differs")
    return result, details


def download_station(url: str, path: Path) -> None:
    """Bound transient retries and preserve failed attempt bytes separately."""
    destination = urlsplit(url)
    if destination.scheme != "https" or destination.hostname != "www.ncei.noaa.gov":
        raise ValueError("Only fixed official NOAA URLs are allowed")
    for attempt in range(1, 4):
        attempted = path.with_name(f"{path.stem}_attempt_{attempt:02d}{path.suffix}")
        request = Request(url, headers={"User-Agent": "PRC-clean-replica-weather/1.0"})
        try:
            with build_opener(NOAARedirectHandler()).open(request, timeout=45) as response, attempted.open("xb") as handle:
                final = urlsplit(response.geturl())
                if final.scheme != "https" or final.hostname != "www.ncei.noaa.gov":
                    raise ValueError("NOAA download redirected outside the allowlist")
                total = 0
                for block in iter(lambda: response.read(2 ** 20), b""):
                    total += len(block)
                    if total > 64 * 2 ** 20:
                        raise ValueError("Station-year file exceeds the documented download bound")
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        except (HTTPError, URLError, TimeoutError) as error:
            print(json.dumps({"transient_download_error": type(error).__name__,
                              "attempt": attempt, "maximum_attempts": 3}), flush=True)
            if attempt == 3:
                raise
            time.sleep(1 if attempt == 1 else 3)
            continue
        os.link(attempted, path)  # Canonical staged input exists only after a complete response.
        return
    raise RuntimeError("No complete NOAA station-year response")


def build(root: Path, published: str, published_protocol: str) -> dict:
    memory_guard()
    _, before = protocol(root, published, published_protocol)
    names = [item["path"] for item in inputs()] + [WEATHER, MANIFEST, RECEIPT]
    if any(inside(root, name).exists() for name in names):
        raise FileExistsError("Weather outputs already exist; use a fresh run root")
    stage = inside(root, ".replica-weather-stage")
    stage.mkdir(exist_ok=False)
    local = []
    for index, item in enumerate(inputs()):
        path = stage / f"source_{index:02d}.parquet"
        download_station(item["url"], path)
        local.append(path)
        if source_snapshot(published) != before:
            raise ValueError("Processing source changed during download")
        print(json.dumps({"downloaded_station_year": index + 1, "sha256": sha(path)}), flush=True)
    weather, sources = weather_table(local)
    output = stage / "weather.parquet"
    pq.write_table(pa.Table.from_pandas(weather, preserve_index=False), output, compression="zstd")
    if not pd.read_parquet(output).equals(weather):
        raise ValueError("Weather table readback differs")
    manifest = {"dataset": "NOAA NCEI Global Historical Climatology Network-hourly (GHCNh)",
                "dataset_doi": "https://doi.org/10.25921/jp3d-3v19", "license": "CC0-1.0",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "output_rows": ROWS, "sources": sources}
    manifest_bytes = json_bytes(manifest)
    receipt = {"schema_version": 1, "status": "complete", "license": "CC0-1.0",
               "source_sha256": before, "protocol_sha256": published_protocol,
               "output_rows": ROWS, "source_urls": [s["url"] for s in sources],
               "station_year_files": {item["path"]: sha(path)
                                      for item, path in zip(inputs(), local, strict=True)},
               "output_sha256": {WEATHER: sha(output),
                                  MANIFEST: hashlib.sha256(manifest_bytes).hexdigest()},
               "original_processing_unchanged": True, "source_recheck_before_after": True,
               "competition_labels_read": False, "leaderboard_used": False,
               "snapshot_scope": "independent current NOAA acquisition; no original-byte claim"}
    if source_snapshot(published) != before or sha(inside(root, PROTOCOL)) != published_protocol:
        raise ValueError("Source/protocol changed during weather construction")
    for item, path in zip(inputs(), local, strict=True):
        destination = inside(root, item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, inside(root, item["path"]))
    destination = inside(root, WEATHER)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(output, inside(root, WEATHER))
    write_new(root, MANIFEST, manifest_bytes)
    write_new(root, RECEIPT, json_bytes(receipt))
    return verify(root, published, published_protocol)


def verify(root: Path, published: str, published_protocol: str) -> dict:
    memory_guard()
    _, before = protocol(root, published, published_protocol)
    receipt_path = inside(root, RECEIPT)
    receipt_sha = sha(receipt_path)
    receipt = read(receipt_path)
    paths = [inside(root, item["path"]) for item in inputs()]
    hashes = {item["path"]: sha(path) for item, path in zip(inputs(), paths, strict=True)}
    outputs = {name: sha(inside(root, name)) for name in (WEATHER, MANIFEST)}
    expected_receipt = {"schema_version": 1, "status": "complete", "license": "CC0-1.0",
                        "source_sha256": before, "protocol_sha256": published_protocol,
                        "output_rows": ROWS, "source_urls": [item["url"] for item in inputs()],
                        "station_year_files": hashes, "output_sha256": outputs,
                        "original_processing_unchanged": True,
                        "source_recheck_before_after": True,
                        "competition_labels_read": False, "leaderboard_used": False,
                        "snapshot_scope": "independent current NOAA acquisition; no original-byte claim"}
    if receipt != expected_receipt:
        raise ValueError("Weather producer receipt/source/output bytes differ")
    rebuilt, sources = weather_table(paths)
    manifest = read(inside(root, MANIFEST))
    if (not pd.read_parquet(inside(root, WEATHER)).equals(rebuilt)
            or manifest.get("sources") != sources or manifest.get("license") != "CC0-1.0"
            or manifest.get("output_rows") != ROWS):
        raise ValueError("Weather saved source/table processing replay failed")
    if (source_snapshot(published) != before or sha(receipt_path) != receipt_sha
            or sha(inside(root, PROTOCOL)) != published_protocol
            or any(sha(inside(root, name)) != digest for name, digest in {**hashes, **outputs}.items())):
        raise ValueError("Weather input/output/source changed during replay")
    return {"status": "verified", "rows": ROWS, "receipt_sha256": receipt_sha,
            "weather_sha256": outputs[WEATHER], "station_years": len(paths)}


def self_test() -> dict:
    source_snapshot()
    raw = pd.DataFrame({name: [None, None, None] for name in original.RAW_COLUMNS})
    raw["DATE"] = ["2025-01-01T00:05:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:55:00Z"]
    raw["temperature"] = [1, 3, 9999]
    raw["wind_direction"] = [350, 10, 25]
    raw["precipitation"] = [.2, .7, .4]
    raw["pres_wx_MW1"] = ["SN", "FG", "RA"]
    hourly = original.weather_hours(raw, 2025)
    assert len(hourly) == 1 and hourly.wx_temperature_c.iloc[0] == 2
    assert hourly.wx_wind_direction_deg.iloc[0] == 25
    assert hourly.wx_precip_mm.iloc[0] == .7
    assert hourly.wx_snow_reported.iloc[0] and hourly.wx_fog_reported.iloc[0]
    assert len(inputs()) == 20 and all(urlsplit(x["url"]).hostname == "www.ncei.noaa.gov" for x in inputs())
    assert len(original.hour_grid(2025)) == 8760 and len(original.hour_grid(2026)) == 1488
    redirect = NOAARedirectHandler()
    try:
        redirect.redirect_request(Request(inputs()[0]["url"]), None, 302, "Found", {}, "https://example.com/weather")
    except ValueError:
        pass
    else:
        raise AssertionError("Outside redirect was not refused before following")
    allowed = redirect.redirect_request(Request(inputs()[0]["url"]), None, 302, "Found", {}, inputs()[1]["url"])
    assert allowed.full_url == inputs()[1]["url"]
    try:
        root_path(SOURCE)
    except ValueError:
        pass
    else:
        raise AssertionError("Source-root refusal failed")
    return {"fixed_source_and_stations": "passed", "synthetic_hourly_processing": "passed",
            "redirect_refused_before_following": True,
            "source_root_refused": True, "real_values_or_network_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("plan", "self-test", "prepare", "build", "verify"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--published-source-sha256")
    parser.add_argument("--published-protocol-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        source_snapshot()
        result = {"spec": str(SPEC), "scope": "unchanged isolated NOAA weather only",
                  "steps": ["prepare", "publish protocol", "build", "verify"],
                  "station_years": inputs(), "competition_raw_manifest": "separate input-seal prerequisite"}
    elif args.mode == "self-test":
        result = self_test()
    else:
        if args.run_root is None or args.published_source_sha256 is None:
            parser.error("Real modes require isolated --run-root and published source SHA")
        root = root_path(args.run_root)
        if args.mode == "prepare":
            result = prepare(root, args.published_source_sha256)
        else:
            if args.published_protocol_sha256 is None:
                parser.error("Build/verify require the published prepared protocol SHA")
            action = build if args.mode == "build" else verify
            result = action(root, args.published_source_sha256, args.published_protocol_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
