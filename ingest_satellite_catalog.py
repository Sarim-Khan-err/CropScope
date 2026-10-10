"""Record recent Sentinel-2 scene metadata near selected source districts.

Scene metadata is not onion-specific reflectance and does not estimate yield.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from ingest_routes import source_centers

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "satellite_scene_reference.csv"
STAC = "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items"


def build():
    geocodes = json.loads((ROOT / "data" / "osm_geocodes.json").read_text(encoding="utf-8"))
    today = date.today()
    start = today - timedelta(days=90)
    rows = []
    for district, center in source_centers().items():
        point = geocodes[center]
        lat, lon = point["lat"], point["lon"]
        params = {"bbox": f"{lon-.01:.5f},{lat-.01:.5f},{lon+.01:.5f},{lat+.01:.5f}",
                  "datetime": f"{start.isoformat()}T00:00:00Z/{today.isoformat()}T23:59:59Z",
                  "sortby": "-properties.datetime", "limit": "5"}
        url = STAC + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "ProduceFlowPlanner/0.1 (educational research)"})
        with urllib.request.urlopen(request, timeout=45) as response:
            features = json.load(response).get("features", [])
        if not features:
            raise ValueError(f"No recent Sentinel-2 L2A scenes found near {district}")
        for feature in features:
            scene_id = feature["id"]
            rows.append({"source_district": district, "center_lat": lat, "center_lon": lon,
                         "scene_id": scene_id, "observed_at_utc": feature["properties"]["datetime"],
                         "tile_cloud_cover_pct": feature["properties"].get("eo:cloud_cover", ""),
                         "scene_url": STAC + "/" + urllib.parse.quote(scene_id),
                         "query_url": url, "retrieved_date": today.isoformat()})
        print(district, len(features), "scene records", flush=True)
        time.sleep(0.3)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", len(rows), "Sentinel-2 catalog records to", OUT)


if __name__ == "__main__":
    build()
