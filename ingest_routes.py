"""Collect road-distance references from OpenStreetMap Nominatim and OSRM.

Routes join the center of each leading onion-production district to major
surveyed city markets. They are geographic references, not shipment quotes.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEOCODES = ROOT / "data" / "osm_geocodes.json"
OUT = ROOT / "data" / "road_distance_reference.csv"
CENTER_OVERRIDES = {"Swat": "Mingora, Swat, Pakistan"}
MARKETS = {
    "Karachi": "Karachi, Sindh, Pakistan",
    "Lahore": "Lahore, Punjab, Pakistan",
    "Multan": "Multan, Punjab, Pakistan",
    "Peshawar": "Peshawar, Khyber Pakhtunkhwa, Pakistan",
    "Quetta": "Quetta, Balochistan, Pakistan",
}


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ProduceFlowPlanner/0.1 (educational research)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def source_centers():
    with (ROOT / "data" / "district_onion_production.csv").open(newline="", encoding="utf-8") as file:
        districts = list(csv.DictReader(file))
    top_by_province = {}
    for row in districts:
        province = row["province"]
        if province not in top_by_province or int(row["production_2023_24_t"]) > int(top_by_province[province]["production_2023_24_t"]):
            top_by_province[province] = row
    return {row["district"]: CENTER_OVERRIDES.get(row["district"],
            f"{row['district']}, {row['province']}, Pakistan")
            for row in top_by_province.values()}


def geocode_all(source_centers):
    cache = json.loads(GEOCODES.read_text(encoding="utf-8")) if GEOCODES.exists() else {}
    for query in sorted(set(source_centers.values()) | set(MARKETS.values())):
        if query in cache:
            continue
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
        results = get_json(url)
        if not results:
            raise ValueError(f"No OpenStreetMap geocode found for {query}")
        cache[query] = {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"]),
                        "osm_type": results[0]["osm_type"], "osm_id": results[0]["osm_id"], "query_url": url}
        GEOCODES.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print("Geocoded", query, flush=True)
        time.sleep(1.2)  # Nominatim usage policy: at most one request per second.
    return cache


def build_routes():
    sources = source_centers()
    geocodes = geocode_all(sources)
    rows = []
    for district, source_query in sources.items():
        source = geocodes[source_query]
        for market, market_query in MARKETS.items():
            destination = geocodes[market_query]
            if source_query == market_query:
                distance_km = 0.0
                duration_hours = 0.0
                route_url = ""
                geometry = []
            else:
                # "simplified" (Douglas-Peucker reduced), not "full" -- full
                # overview on a long route returns thousands of points, which
                # is both too large to store and unnecessary for display.
                route_url = ("https://router.project-osrm.org/route/v1/driving/"
                             f"{source['lon']},{source['lat']};{destination['lon']},{destination['lat']}"
                             "?overview=simplified&geometries=geojson")
                route = get_json(route_url)
                if route.get("code") != "Ok" or not route.get("routes"):
                    raise ValueError(f"No road route from {district} to {market}")
                distance_km = round(route["routes"][0]["distance"] / 1000, 1)
                duration_hours = round(route["routes"][0]["duration"] / 3600, 1)
                # GeoJSON order is [lon, lat]; kept as-is here, converted for Leaflet at read time.
                geometry = route["routes"][0]["geometry"]["coordinates"]
                time.sleep(0.3)
            rows.append({"source_district": district, "source_center": source_query,
                         "source_lat": source["lat"], "source_lon": source["lon"],
                         "market_city": market, "market_lat": destination["lat"], "market_lon": destination["lon"],
                         "road_distance_km": distance_km,
                         "model_drive_hours": duration_hours, "retrieved_date": date.today().isoformat(),
                         "route_url": route_url, "route_geometry_geojson": json.dumps(geometry),
                         "geocode_source": "OpenStreetMap Nominatim",
                         "route_source": "OpenStreetMap road network via OSRM"})
            print(district, "->", market, distance_km, "km", flush=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} road-distance references to {OUT}")


if __name__ == "__main__":
    build_routes()
