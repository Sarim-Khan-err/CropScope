"""Build traceable price, district supply, and household-demand summaries."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of empty series")
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def price_analysis():
    path = ROOT / "data" / "pbs_onion_weekly.csv"
    with path.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    by_week = defaultdict(dict)
    urls = {}
    for row in source_rows:
        value = float(row["price_rs_per_kg"])
        if value <= 0:
            raise ValueError("A PBS price must be positive")
        by_week[row["week"]][row["city"]] = value
        urls[row["week"]] = row["source_url"]
    weeks = sorted(by_week)
    if len(weeks) < 20:
        raise ValueError("At least 20 weekly price observations are required")
    if any(len(by_week[week]) != 18 or "Pakistan" not in by_week[week] for week in weeks):
        raise ValueError("Every week must contain 17 surveyed city prices and the PBS national average")
    national = [by_week[week]["Pakistan"] for week in weeks]
    latest_week, previous_week = weeks[-1], weeks[-2]
    latest, previous = national[-1], national[-2]
    change_week = 100 * (latest / previous - 1)
    change_4week = 100 * (latest / national[-5] - 1)
    prior_4week_changes = [100 * (national[i] / national[i - 4] - 1) for i in range(4, len(national) - 1)]
    upper = percentile(prior_4week_changes, 0.9)
    lower = percentile(prior_4week_changes, 0.1)
    signal = "unusually fast rise" if change_4week > upper else "unusually fast fall" if change_4week < lower else "within past movement range"
    city_latest = []
    city_alerts = []
    for city in sorted(set(by_week[latest_week]) - {"Pakistan"}):
        price = by_week[latest_week][city]
        previous_price = by_week[previous_week][city]
        premium_pct = 100 * (price / latest - 1)
        historical_premiums = [100 * (by_week[week][city] / by_week[week]["Pakistan"] - 1)
                               for week in weeks[:-1]]
        threshold = percentile(historical_premiums, 0.9)
        entry = {"city": city, "price_rs_per_kg": price,
                 "weekly_change_pct": round(100 * (price / previous_price - 1), 1),
                 "vs_national_pct": round(premium_pct, 1)}
        city_latest.append(entry)
        if premium_pct > threshold and premium_pct > 0:
            city_alerts.append({**entry, "rule": "City price is above its own 90th-percentile historical premium over the national average.",
                                "historical_premium_threshold_pct": round(threshold, 1)})
    city_latest.sort(key=lambda item: item["price_rs_per_kg"], reverse=True)
    city_alerts.sort(key=lambda item: item["vs_national_pct"], reverse=True)
    return {
        "status": "loaded", "source": "Pakistan Bureau of Statistics, Weekly SPI Annex, Appendix A",
        "source_index_url": "https://www.pbs.gov.pk/price-statistics/",
        "latest_source_url": urls[latest_week], "latest_week": latest_week,
        "first_week": weeks[0], "weeks": len(weeks), "cities": 17,
        "latest_national_rs_per_kg": latest, "previous_national_rs_per_kg": previous,
        "weekly_change_pct": round(change_week, 1), "four_week_change_pct": round(change_4week, 1),
        "four_week_signal": signal,
        "range_4week_change_pct": [round(lower, 1), round(upper, 1)],
        "history": [{"week": week, "price_rs_per_kg": by_week[week]["Pakistan"], "source_url": urls[week]}
                    for week in weeks],
        "city_history": {city: [{"week": week, "price_rs_per_kg": by_week[week][city]}
                                for week in weeks]
                         for city in sorted(set(by_week[latest_week]) - {"Pakistan"})},
        "city_latest": city_latest, "city_alerts": city_alerts,
        "alert_note": "Alerts describe surveyed city retail prices, not district-wide shortages or future prices. The city premium threshold is calculated from its preceding 48 weekly observations."
    }


def price_pattern_reference():
    """Monthly price trajectory and city dispersion over the one observed year.

    This is descriptive statistics on the same 49-week PBS series used above,
    not a seasonal model: the archive spans exactly one calendar pass
    (2025-10 to 2026-10), so there is no second year to confirm any monthly
    pattern repeats rather than being a one-off move. It is never framed as a
    recurring season or as timing advice.
    """
    path = ROOT / "data" / "pbs_onion_weekly.csv"
    with path.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    by_week = defaultdict(dict)
    for row in source_rows:
        by_week[row["week"]][row["city"]] = float(row["price_rs_per_kg"])
    weeks = sorted(by_week)
    cities = sorted(set(by_week[weeks[0]]) - {"Pakistan"})

    by_month = defaultdict(list)
    for week in weeks:
        by_month[week[:7]].append(by_week[week]["Pakistan"])
    monthly = [{"month": month, "avg_rs_per_kg": round(statistics.mean(values), 2),
               "min_rs_per_kg": round(min(values), 2), "max_rs_per_kg": round(max(values), 2),
               "weeks": len(values)} for month, values in sorted(by_month.items())]

    city_stats = []
    for city in cities:
        series = [by_week[week][city] for week in weeks]
        premiums = [100 * (by_week[week][city] / by_week[week]["Pakistan"] - 1) for week in weeks]
        mean_price = statistics.mean(series)
        city_stats.append({
            "city": city, "mean_rs_per_kg": round(mean_price, 2),
            "mean_premium_pct": round(statistics.mean(premiums), 1),
            "coefficient_of_variation_pct": round(100 * statistics.pstdev(series) / mean_price, 1),
            "min_rs_per_kg": round(min(series), 2), "max_rs_per_kg": round(max(series), 2),
        })
    most_volatile = sorted(city_stats, key=lambda x: x["coefficient_of_variation_pct"], reverse=True)
    most_expensive = sorted(city_stats, key=lambda x: x["mean_premium_pct"], reverse=True)

    return {
        "status": "loaded", "coverage_note": "One observed year (2025-10-23 to 2026-10-08) — a monthly trajectory, not a confirmed recurring season.",
        "weeks": len(weeks), "months": len(monthly), "monthly": monthly,
        "most_volatile_cities": most_volatile[:5], "most_expensive_cities": most_expensive[:5],
        "least_expensive_cities": most_expensive[-5:][::-1],
        "source_index_url": "https://www.pbs.gov.pk/price-statistics/",
    }


def supply_demand_reference():
    with (ROOT / "data" / "district_onion_production.csv").open(newline="", encoding="utf-8") as file:
        districts = list(csv.DictReader(file))
    with (ROOT / "data" / "household_demand_reference.csv").open(newline="", encoding="utf-8") as file:
        demand = next(csv.DictReader(file))
    ranked = sorted(districts, key=lambda row: int(row["production_2023_24_t"]), reverse=True)
    top = [{"district": row["district"], "province": row["province"],
            "production_2023_24_t": int(row["production_2023_24_t"]),
            "production_2022_23_t": int(row["production_2022_23_t"]),
            "source_pdf_page": int(row["source_pdf_page"])} for row in ranked[:10]]
    return {
        "district_status": "loaded", "district_count": len(districts),
        "district_season": "2023-24", "top_districts": top,
        "districts": [{"district": row["district"], "province": row["province"],
                        "production_2023_24_t": int(row["production_2023_24_t"]),
                        "production_2022_23_t": int(row["production_2022_23_t"])}
                       for row in ranked],
        "district_source_url": districts[0]["source_url"],
        "monthly_household_reference_t": int(demand["monthly_household_reference_t"]),
        "household_kg_per_person_month": float(demand["onion_kg_per_person_per_month"]),
        "consumption_source_url": demand["consumption_source_url"],
        "population_source_url": demand["population_source_url"],
        "population_year": int(demand["population_year"]),
        "demand_note": demand["note"],
        "supply_note": "District figures are annual crop production, not monthly harvested or market-arrival volumes. The printed KP province total is one tonne below the sum of its district rows in the source publication."
    }


ROUTE_COORD_PATTERN = re.compile(r"driving/([\-0-9.]+),([\-0-9.]+);([\-0-9.]+),([\-0-9.]+)")


def road_and_season_reference():
    with (ROOT / "data" / "road_distance_reference.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with (ROOT / "data" / "district_onion_production.csv").open(newline="", encoding="utf-8") as file:
        sources = {row["district"]: row["province"] for row in csv.DictReader(file)}
    source_names = {row["source_district"] for row in rows}
    market_names = {row["market_city"] for row in rows}
    if not rows or len(rows) != len(source_names) * len(market_names):
        raise ValueError("Road-distance reference is incomplete")

    def viewer_url(route_url):
        # route_url is a raw OSRM API endpoint (JSON response, not a page).
        # Build the equivalent human-viewable OpenStreetMap directions page
        # from the same coordinates, using the same OSRM routing engine.
        # Same-city rows (0 km) never called OSRM, so route_url is blank.
        if not route_url:
            return ""
        match = ROUTE_COORD_PATTERN.search(route_url)
        if not match:
            raise ValueError(f"Could not parse coordinates from route_url: {route_url}")
        lon1, lat1, lon2, lat2 = match.groups()
        return f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={lat1}%2C{lon1}%3B{lat2}%2C{lon2}"

    return {
        "status": "route_distances_loaded", "route_count": len(rows),
        "routes": [{"source_district": row["source_district"], "source_province": sources[row["source_district"]],
                    "source_lat": float(row["source_lat"]), "source_lon": float(row["source_lon"]),
                    "market_city": row["market_city"],
                    "market_lat": float(row["market_lat"]), "market_lon": float(row["market_lon"]),
                    "road_distance_km": float(row["road_distance_km"]),
                    "model_drive_hours": float(row["model_drive_hours"]),
                    "retrieved_date": row["retrieved_date"], "route_url": row["route_url"],
                    "viewer_url": viewer_url(row["route_url"]),
                    # Leaflet expects [lat, lon]; OSRM's GeoJSON is [lon, lat].
                    "route_path": [[lat, lon] for lon, lat in json.loads(row["route_geometry_geojson"])]}
                   for row in rows],
        "osm_source_url": "https://www.openstreetmap.org/copyright",
        "route_note": "Road distances join district and market city centers using OpenStreetMap roads. They do not include farm pickup, market access, freight quotes, truck availability, spoilage, or storage capacity."
    }


def source_yield_reference():
    with (ROOT / "data" / "source_district_onion_yield_history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_district = defaultdict(list)
    for row in rows:
        by_district[row["district"]].append({
            "season": row["season"], "area_ha": int(row["area_ha"]),
            "production_t": int(row["production_t"]),
            "yield_t_per_ha": float(row["yield_t_per_ha"]),
        })
    if len(by_district) != 4 or any(len(series) < 10 for series in by_district.values()):
        raise ValueError("Cross-checked source district yield history is incomplete")
    return {
        "status": "historical_labels_loaded", "district_count": len(by_district),
        "observation_count": len(rows), "history": dict(by_district),
        "source_url": rows[0]["source_url"], "mirror_url": rows[0]["mirror_url"],
        "mirror_pull_date": rows[0]["mirror_pull_date"],
        "note": "PBS portal district yield history for the four selected source districts, downloaded through a Data Darbar mirror. Their 2023-24 area and output were cross-checked against the MNFSR district publication; Multan's 2024-25 output also matches Punjab Agriculture Department's published 132,258 tonnes. Other 2024-25 district rows lack that independent check here. The mirror omits some districts and 2017-18, so its rows are not national totals. Historical district yield does not replace onion field boundaries or current satellite observations."
    }


def trade_reference():
    with (ROOT / "data" / "onion_trade_fy2024_25.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_direction = defaultdict(list)
    for row in rows:
        by_direction[row["direction"]].append(row)
    if set(by_direction) != {"import", "export"}:
        raise ValueError("Trade reference must contain both import and export rows")

    def summarize(direction):
        country_rows = [r for r in by_direction[direction] if r["country"] != "ALL COUNTRIES"]
        total = next(r for r in by_direction[direction] if r["country"] == "ALL COUNTRIES")
        ranked = sorted(country_rows, key=lambda r: int(r["qty_kg_fy2024_25"]), reverse=True)
        total_kg = int(total["qty_kg_fy2024_25"])
        return {
            "total_t": round(total_kg / 1000),
            "total_pkr_thousands": round(float(total["value_pkr_thousands_fy2024_25"])),
            "top_countries": [{"country": r["country"], "t": round(int(r["qty_kg_fy2024_25"]) / 1000),
                               "share_pct": round(100 * int(r["qty_kg_fy2024_25"]) / total_kg, 1)}
                              for r in ranked[:5]],
            "source_url": total["source_url"],
        }

    imports, exports = summarize("import"), summarize("export")
    return {
        "status": "loaded", "fiscal_year": "FY2024-25 (July 2024 to June 2025)",
        "commodity": "Onions and shallots, HS 7031000",
        "imports": imports, "exports": exports,
        "net_position": "net importer by tonnage" if imports["total_t"] > exports["total_t"] else "net exporter by tonnage",
        "retrieved_date": rows[0]["retrieved_date"],
        "note": "Official PBS commodity-by-country trade report, one cumulative fiscal year. Annual granularity only — not matched to the weekly price series, not a trend (no second machine-readable year retrieved here), and not a supply or price driver claim. Values are PKR thousands as published, with no currency conversion."
    }


def satellite_catalog_reference():
    with (ROOT / "data" / "satellite_scene_reference.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_district = defaultdict(list)
    centers = {}
    for row in rows:
        by_district[row["source_district"]].append({
            "scene_id": row["scene_id"], "observed_at_utc": row["observed_at_utc"],
            "tile_cloud_cover_pct": float(row["tile_cloud_cover_pct"]),
            "scene_url": row["scene_url"],
        })
        centers[row["source_district"]] = {"lat": float(row["center_lat"]), "lon": float(row["center_lon"])}
    if len(by_district) != 4 or any(len(scenes) < 1 for scenes in by_district.values()):
        raise ValueError("Sentinel-2 catalog records are incomplete")
    return {
        "status": "scene_metadata_loaded", "scene_count": len(rows),
        "retrieved_date": rows[0]["retrieved_date"],
        "scenes": dict(by_district), "district_centers": centers, "catalog_url": "https://stac.dataspace.copernicus.eu/v1/",
        "note": "Official Copernicus Sentinel-2 L2A scene metadata near district-center points. Cloud cover is for the satellite tile, not an onion field. No reflectance, NDVI, crop mask, or yield inference is computed."
    }
