"""Build hourly airport weather features from NOAA NCEI GHCNh station files.

The GHCNh dataset is CC0-1.0. See data/external/README.md for the station
selection, source citation, units, aggregation decisions, and limitations.

Run from any directory with::

    python download_noaa_weather.py

Requires pandas and pyarrow, already listed in requirements.txt. Downloads one
small station-year Parquet file per airport and year, then writes only the
derived hourly feature table and a source manifest to data/external/.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "external"
BASE_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/access/by-year"
)
STATIONS = {
    "EDDF": ("GMI0000EDDF", "Frankfurt Main"),
    "EDDM": ("GMI0000EDDM", "Munich"),
    "EGLL": ("UKI0000EGLL", "London Heathrow"),
    "EHAM": ("NLMU0029295", "Amsterdam Schiphol"),
    "LEBL": ("SPUP00000Y9", "Port de Barcelona - ZAL Prat (4.6 km from LEBL)"),
    "LEMD": ("SPMU0098221", "Madrid Barajas"),
    "LFPG": ("FRI0000LFPG", "Paris Charles de Gaulle"),
    "LIRF": ("ITMU0016242", "Rome Fiumicino"),
    "LTFM": ("TUI0000LTFM", "Istanbul New Airport"),
    "LSZH": ("SZM00006664", "Zuerich-Affoltern (4.3 km from LSZH)"),
}
RAW_COLUMNS = [
    "DATE",
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "precipitation",
    "snow_depth",
    "visibility",
    "ceiling_height",
    "pres_wx_MW1",
    "pres_wx_AU1",
    "pres_wx_AW1",
]
NUMERIC_COLUMNS = {
    "temperature": (-60, 60),
    "dew_point_temperature": (-80, 50),
    "relative_humidity": (0, 100),
    "wind_speed": (0, 100),
    "wind_direction": (0, 360),
    "wind_gust": (0, 120),
    "precipitation": (0, 250),
    "snow_depth": (0, 10000),
    "visibility": (0, 100),
    "ceiling_height": (0, 20000),
}
AGGREGATIONS = {
    "temperature": "median",
    "dew_point_temperature": "median",
    "relative_humidity": "median",
    "wind_speed": "median",
    "wind_gust": "max",
    "precipitation": "max",
    "snow_depth": "max",
    "visibility": "min",
    "ceiling_height": "min",
    "wx_snow_reported": "max",
    "wx_fog_reported": "max",
    "wx_mist_reported": "max",
    "wx_rain_reported": "max",
    "wx_freezing_precip_reported": "max",
}
RENAME = {
    "temperature": "wx_temperature_c",
    "dew_point_temperature": "wx_dewpoint_c",
    "relative_humidity": "wx_relative_humidity_pct",
    "wind_speed": "wx_wind_speed_mps",
    "wind_direction": "wx_wind_direction_deg",
    "wind_gust": "wx_wind_gust_mps",
    "precipitation": "wx_precip_mm",
    "snow_depth": "wx_snow_depth_mm",
    "visibility": "wx_visibility_km",
    "ceiling_height": "wx_ceiling_m",
}


def download(url: str) -> bytes:
    """Fetch a small public NOAA file with bounded retries."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "PRC-Data-Challenge-weather-research/1.0"})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt * 2)
    raise RuntimeError(f"Cannot download {url}: {last_error}") from last_error


def weather_hours(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    raw = raw.loc[raw["DATE"].notna()]
    if year == 2026:
        raw = raw.loc[raw["DATE"].dt.month.isin((1, 7))]

    for name, (minimum, maximum) in NUMERIC_COLUMNS.items():
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
        raw.loc[~raw[name].between(minimum, maximum), name] = np.nan

    present_weather = raw[["pres_wx_MW1", "pres_wx_AU1", "pres_wx_AW1"]].fillna("").astype(str)
    raw["wx_snow_reported"] = present_weather.apply(lambda c: c.str.contains("SN", case=False)).any(axis=1)
    raw["wx_fog_reported"] = present_weather.apply(lambda c: c.str.contains("FG", case=False)).any(axis=1)
    raw["wx_mist_reported"] = present_weather.apply(lambda c: c.str.contains("BR", case=False)).any(axis=1)
    raw["wx_rain_reported"] = present_weather.apply(lambda c: c.str.contains("RA|DZ", case=False, regex=True)).any(axis=1)
    raw["wx_freezing_precip_reported"] = present_weather.apply(lambda c: c.str.contains("FZ", case=False)).any(axis=1)
    raw["weather_hour_utc"] = raw["DATE"].dt.floor("h")

    grouped = raw.groupby("weather_hour_utc", sort=True)
    hourly = grouped.agg(AGGREGATIONS)
    hourly["wx_obs_count"] = grouped.size()
    # A direction is circular, so use the last valid observation in the hour.
    direction = raw.loc[raw["wind_direction"].notna()].sort_values("DATE")
    direction = direction.drop_duplicates("weather_hour_utc", keep="last")
    hourly["wind_direction"] = direction.set_index("weather_hour_utc")["wind_direction"]
    return hourly.rename(columns=RENAME)


def hour_grid(year: int) -> pd.DatetimeIndex:
    if year == 2025:
        return pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC")
    jan = pd.date_range("2026-01-01", "2026-02-01", freq="h", inclusive="left", tz="UTC")
    jul = pd.date_range("2026-07-01", "2026-08-01", freq="h", inclusive="left", tz="UTC")
    return jan.append(jul)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_airports: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for airport, (station_id, station_name) in STATIONS.items():
        for year in (2025, 2026):
            url = f"{BASE_URL}/{year}/parquet/GHCNh_{station_id}_{year}.parquet"
            payload = download(url)
            table = pq.read_table(pa.BufferReader(payload), columns=RAW_COLUMNS)
            raw = table.to_pandas()
            hourly = weather_hours(raw, year)
            grid = pd.DataFrame(index=hour_grid(year))
            grid.index.name = "weather_hour_utc"
            grid = grid.join(hourly, how="left").reset_index()
            grid.insert(0, "airport", airport)
            grid.insert(1, "wx_station_id", station_id)
            grid["wx_obs_count"] = grid["wx_obs_count"].fillna(0).astype("int16")
            grid["wx_fog_proxy"] = (grid["wx_visibility_km"] <= 1.0) | grid["wx_fog_reported"].fillna(False)
            grid["wx_deicing_proxy"] = (grid["wx_temperature_c"] <= 3.0) & (
                (grid["wx_precip_mm"] > 0)
                | grid["wx_snow_reported"].fillna(False)
                | grid["wx_freezing_precip_reported"].fillna(False)
                | (grid["wx_snow_depth_mm"] > 0)
            )
            for column in [c for c in grid.columns if c.endswith("_reported") or c.endswith("_proxy")]:
                grid[column] = grid[column].fillna(False).astype(bool)
            all_airports.append(grid)
            sources.append(
                {
                    "airport": airport,
                    "station_id": station_id,
                    "station_name": station_name,
                    "year": year,
                    "url": url,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_records": len(raw),
                    "observed_hours": int((grid["wx_obs_count"] > 0).sum()),
                    "expected_hours": len(grid),
                }
            )
            print(f"{airport} {year}: {len(raw):,} observations, {sources[-1]['observed_hours']:,}/{len(grid):,} hours")

    weather = pd.concat(all_airports, ignore_index=True)
    if weather.duplicated(["airport", "weather_hour_utc"]).any():
        raise ValueError("Duplicate airport-hour keys")
    if len(weather) != 10 * (8760 + 744 + 744):
        raise ValueError(f"Unexpected output row count: {len(weather)}")
    pq.write_table(pa.Table.from_pandas(weather, preserve_index=False), OUT_DIR / "weather.parquet", compression="zstd")
    manifest = {
        "dataset": "NOAA NCEI Global Historical Climatology Network-hourly (GHCNh)",
        "dataset_doi": "https://doi.org/10.25921/jp3d-3v19",
        "license": "CC0-1.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "output_rows": len(weather),
        "sources": sources,
    }
    (OUT_DIR / "weather_sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(weather):,} rows to {OUT_DIR / 'weather.parquet'}")


if __name__ == "__main__":
    main()
