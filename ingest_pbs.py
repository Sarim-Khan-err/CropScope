"""Download and normalize Pakistan Bureau of Statistics weekly onion prices.

Source index: https://www.pbs.gov.pk/price-statistics/
The index HTML is saved at data/pbs_price_statistics_page.html for reproducibility.
Run with --download only when internet access is available. --build works offline.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "pbs_price_statistics_page.html"
RAW = ROOT / "data" / "sources" / "pbs_weekly"
OUT = ROOT / "data" / "pbs_onion_weekly.csv"
BASE = "https://www.pbs.gov.pk/"


def archive_links():
    page = INDEX.read_text(encoding="utf-8")
    start = page.index("const data = [")
    end = page.index("let filteredData", start)
    items = re.findall(r"\{[^{}]*\}", page[start:end])
    records = []
    for item in items:
        date_match = re.search(r'date\s*:\s*"([^"]+)"', item)
        xlsx_match = re.search(r'annexureExcel\s*:\s*"([^"]+)"', item)
        if date_match and xlsx_match:
            date = datetime.strptime(date_match.group(1), "%d-%m-%Y").date().isoformat()
            url = BASE + xlsx_match.group(1).lstrip("/")
            records.append((date, url))
    return sorted(set(records))


def download_one(record):
    date, url = record
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{date}.xlsx"
    if path.exists() and path.stat().st_size > 1000:
        return path
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data collection)"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            if not content.startswith(b"PK"):
                raise ValueError(f"Non-XLSX response for {date}")
            path.write_bytes(content)
            return path
        except Exception as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {date}: {error}")


def read_onion_rows(path, date, url):
    sheet = load_workbook(path, read_only=True, data_only=True)["Appendix-A"]
    rows = list(sheet.iter_rows(values_only=True))
    values = {}
    for index, header in enumerate(rows):
        if header[0] != "SL.":
            continue
        if index + 1 >= len(rows):
            continue
        subheader = rows[index + 1]
        city_columns = []
        for column, name in enumerate(header):
            if not isinstance(name, str) or column + 1 >= len(subheader):
                continue
            if subheader[column + 1] != "AVG":
                continue
            city = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
            if city in {"National Average", "National Ave."}:
                city = "Pakistan"
            if city and city != "SL.":
                city_columns.append((city, column + 1))
        for data_row in rows[index + 2 :]:
            if data_row[0] == "SL.":
                break
            if len(data_row) > 2 and data_row[1] == "Onions" and data_row[2] == "1 Kg":
                for city, column in city_columns:
                    price = data_row[column]
                    if isinstance(price, (int, float)) and price > 0:
                        values[city] = round(float(price), 2)
                break
    if "Pakistan" not in values or len(values) < 18:
        raise ValueError(f"Expected 17 city prices and national average in {path.name}; got {sorted(values)}")
    return [{"week": date, "city": city, "price_rs_per_kg": price, "source_url": url}
            for city, price in sorted(values.items())]


def build(links):
    records = []
    failures = []
    for date, url in links:
        path = RAW / f"{date}.xlsx"
        if not path.exists():
            failures.append((date, "file absent"))
            continue
        try:
            records.extend(read_onion_rows(path, date, url))
        except Exception as exc:
            failures.append((date, str(exc)))
    if not records:
        raise ValueError("No verified PBS onion price rows were parsed.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["week", "city", "price_rs_per_kg", "source_url"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} verified prices across {len(set(row['week'] for row in records))} weeks to {OUT}")
    for date, reason in failures:
        print(f"Skipped {date}: {reason}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    links = archive_links()
    print(f"Found {len(links)} official weekly XLSX links in the saved PBS index.")
    if args.download:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(download_one, link): link for link in links}
            for future in as_completed(futures):
                date, _ = futures[future]
                try:
                    path = future.result()
                    print(f"Downloaded {date}: {path.stat().st_size} bytes")
                except Exception as exc:
                    print(f"Download failed {date}: {exc}")
    build(links)


if __name__ == "__main__":
    main()
