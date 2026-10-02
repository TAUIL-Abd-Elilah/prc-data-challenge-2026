# Hourly weather for the PRC Data Challenge

`weather.parquet` contains hourly UTC weather features for the 10 challenge airports in all of 2025 and January/July 2026. Rebuild it with `python download_noaa_weather.py`. The script downloads 20 station-year Parquet files and records each exact URL and SHA-256 digest in `weather_sources.json`.

## Source and license

Source: NOAA National Centers for Environmental Information, **Global Historical Climatology Network-hourly (GHCNh)**, version 1.1.0. The [dataset metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01688) specifies the **CC0-1.0** public-domain dedication and the citation:

> Menne, Matthew J.; Noone, Simon; Casey, Nancy W.; Dunn, Robert H.; McNeill, Shelley; Kantor, Diana; Thorne, Peter W.; Orcutt, Karen; Cunningham, Sam; Risavi, Nicholas. 2023. Global Historical Climatology Network-Hourly (GHCNh). [10 station subsets, 2025 and January/July 2026, accessed 2 October 2026]. NOAA National Centers for Environmental Information. https://doi.org/10.25921/jp3d-3v19.

[NOAA's GHCNh product page](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) describes the observed station data and available elements. The [GHCNh documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf) defines units and station-year file structure. The [station list](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv) supplies station names, ICAO identifiers, and coordinates. This feature table is a derived subset; it is not an official NOAA product or an airport operational record.

## Stations

| Airport | GHCNh station | Station name / relationship |
|---|---|---|
| EDDF | GMI0000EDDF | Frankfurt Main |
| EDDM | GMI0000EDDM | Munich |
| EGLL | UKI0000EGLL | London Heathrow |
| EHAM | NLMU0029295 | Amsterdam Schiphol |
| LEBL | SPUP00000Y9 | Port de Barcelona - ZAL Prat, about 4.6 km from airport reference point |
| LEMD | SPMU0098221 | Madrid Barajas |
| LFPG | FRI0000LFPG | Paris Charles de Gaulle |
| LIRF | ITMU0016242 | Rome Fiumicino |
| LTFM | TUI0000LTFM | Istanbul New Airport |
| LSZH | SZM00006664 | Zuerich-Affoltern, about 4.3 km from airport reference point |

The Barcelona and Zürich substitutes are the nearest stations with both 2025 and 2026 files among those checked. Weather at those stations may differ from the airfield.

## Table and processing

Each row is identified by `airport` (ICAO code) and `weather_hour_utc` (UTC hour). All expected hours are retained, including hours without reports (`wx_obs_count = 0`). Join flight movement time rounded **down** to the UTC hour. Preserve missing weather values rather than treating them as zero.

Within each hour, the script takes medians for temperature, dew point, relative humidity and wind speed; the latest valid wind direction; maxima for gust, precipitation and snow depth; and minima for visibility and cloud ceiling. NOAA describes the precipitation field as a nominal hourly amount that can also appear in intermediate reports, so the script uses the maximum reported hourly value rather than adding multiple reports. Weather condition flags are true when a present-weather code in the source contains the corresponding `SN`, `FG`, `BR`, `RA`/`DZ`, or `FZ` text. A false flag means that condition was not reported in the available observations, not that it was definitely absent.

Units follow [NOAA's documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf): °C for temperature/dew point, percent for relative humidity, m/s for wind, degrees from true north for direction, mm for precipitation and snow depth, km for visibility, and m for cloud ceiling. `wx_fog_proxy` is true when visibility is at most 1 km or fog was reported. `wx_deicing_proxy` is true when temperature is at most 3 °C and precipitation, snow cover, reported snow, or freezing precipitation is present. These two proxy columns are modeling heuristics, not operational labels.

NOAA station records are observations, so coverage and some variables vary by station and time. In particular, present-weather codes, snow depth, precipitation, and visibility can be sparse. Check `weather_sources.json` for observed-hour coverage and validate any benefit with a time-based local holdout before relying on these features.
