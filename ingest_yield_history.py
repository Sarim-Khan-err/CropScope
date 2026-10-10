"""Extract cross-checked district onion yield history from a PBS portal mirror.

The source Parquet is a Data Darbar export of the PBS National Accounts crop
portal, pulled 2026-09-27. The four 2023-24 records are compared against the
separate MNFSR district crop table before any history is written.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".vendor"))
import duckdb  # noqa: E402

SOURCE = ROOT / "data" / "sources" / "pbs_crops_district_fy_mirror.parquet"
OUT = ROOT / "data" / "source_district_onion_yield_history.csv"
PORTAL_URL = "https://na.data.gov.pk/crops/home"
MIRROR_URL = "https://darbar.adaad.org/datasets/crops-district-fy/"


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("district", ""))


def build():
    with (ROOT / "data" / "district_onion_production.csv").open(newline="", encoding="utf-8") as file:
        official = list(csv.DictReader(file))
    top = {}
    for row in official:
        province = row["province"]
        if province not in top or int(row["production_2023_24_t"]) > int(top[province]["production_2023_24_t"]):
            top[province] = row
    by_name = {normalize(row["district"]): row for row in top.values()}
    query = f"SELECT district,fy,area_000ha,production_000t,yield_t_per_ha FROM '{SOURCE.as_posix()}' WHERE crop='Onion'"
    extracted = []
    for district, season, area, production, yield_value in duckdb.sql(query).fetchall():
        key = normalize(district)
        if key not in by_name or area is None or production is None or area <= 0:
            continue
        published = by_name[key]
        area_ha = round(area * 1000)
        production_t = round(production * 1000)
        if season == "2023-24":
            if abs(area_ha - int(published["area_2023_24_ha"])) > 1:
                raise ValueError(f"Area mismatch for {district}: {area_ha}")
            if abs(production_t - int(published["production_2023_24_t"])) > 1:
                raise ValueError(f"Production mismatch for {district}: {production_t}")
        extracted.append({"district": published["district"], "province": published["province"],
                          "season": season, "area_ha": area_ha, "production_t": production_t,
                          "yield_t_per_ha": round(float(yield_value), 2),
                          "source_url": PORTAL_URL, "mirror_url": MIRROR_URL,
                          "mirror_pull_date": "2026-09-27"})
    for row in top.values():
        if not any(item["district"] == row["district"] and item["season"] == "2023-24" for item in extracted):
            raise ValueError(f"Missing cross-check season for {row['district']}")
    extracted.sort(key=lambda item: (item["district"], item["season"]))
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(extracted[0]))
        writer.writeheader()
        writer.writerows(extracted)
    print(f"Wrote {len(extracted)} cross-checked district yield observations to {OUT}")


if __name__ == "__main__":
    build()
