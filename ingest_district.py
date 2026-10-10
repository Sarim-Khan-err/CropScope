"""Extract the official onion district table and HIES household consumption rate."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "sources" / "mnfsr_district_crops_2024.pdf"
OUT = ROOT / "data" / "district_onion_production.csv"
SOURCE_URL = "https://mnfsr.gov.pk/SiteImage/Downloads/Cap_2024%20updated%20Final.pdf"
HIES = ROOT / "data" / "sources" / "pbs_hies_2024_25.pdf"
HIES_URL = "https://www.pbs.gov.pk/wp-content/uploads/2020/07/HIES-2024-25-Report-Final-2.pdf"
CENSUS_URL = "https://www.pbs.gov.pk/sites/default/files/population/2023/reports/National%20Census%20Report-2023.pdf"
PROVINCES = {"PUNJAB": "Punjab", "SINDH": "Sindh", "KPK": "Khyber Pakhtunkhwa", "BALOCHISTAN": "Balochistan"}


def extract_districts():
    records = []
    totals = {}
    province = None
    with pdfplumber.open(SOURCE) as pdf:
        for page_number in (94, 95, 96):
            tables = pdf.pages[page_number - 1].extract_tables()
            if len(tables) != 1:
                raise ValueError(f"Expected one table on source PDF page {page_number}")
            for row in tables[0]:
                name = re.sub(r"\s+", " ", (row[0] or "")).strip()
                if name in PROVINCES:
                    province = PROVINCES[name]
                    continue
                if name == "Total":
                    totals[province] = int(row[4])
                    continue
                if not province or name in {"PAKISTAN", "Province/ District", "Province/"}:
                    continue
                if len(row) < 5 or not all(re.fullmatch(r"\d+", str(value or "")) for value in row[1:5]):
                    continue
                records.append({"province": province, "district": name,
                                "area_2022_23_ha": int(row[1]), "area_2023_24_ha": int(row[2]),
                                "production_2022_23_t": int(row[3]), "production_2023_24_t": int(row[4]),
                                "source_pdf_page": page_number, "source_url": SOURCE_URL})
    for province, expected in totals.items():
        actual = sum(row["production_2023_24_t"] for row in records if row["province"] == province)
        if abs(actual - expected) > 1:
            raise ValueError(f"District sum for {province} is {actual}, source total is {expected}")
        if actual != expected:
            print(f"Source discrepancy: {province} district rows sum to {actual:,} t; printed province total is {expected:,} t. Individual rows are preserved.")
    if sum(totals.values()) != 2_304_176:
        raise ValueError("Province sums do not match official national total.")
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} district rows; 2023-24 source total {sum(totals.values()):,} tonnes")


def extract_household_benchmark():
    with pdfplumber.open(HIES) as pdf:
        text = pdf.pages[41].extract_text() or ""
    match = re.search(r"Onion\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if not match or tuple(map(float, match.groups()[-3:])) != (0.85, 0.85, 0.85):
        raise ValueError("Could not verify HIES 2024-25 onion consumption row.")
    population = 241_499_431  # PBS National Census Report 2023, section 4.1.1
    monthly_kg_per_person = float(match.group(6))
    output = ROOT / "data" / "household_demand_reference.csv"
    row = {"population_year": 2023, "population": population,
           "consumption_survey_year": "2024-25", "onion_kg_per_person_per_month": monthly_kg_per_person,
           "monthly_household_reference_t": round(population * monthly_kg_per_person / 1000),
           "population_source_url": CENSUS_URL, "consumption_source_url": HIES_URL,
           "note": "Census population multiplied by HIES national household consumption rate; excludes commercial, institutional, export and loss demand. Population and survey years differ."}
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote household consumption reference: {row['monthly_household_reference_t']:,} tonnes per month")


if __name__ == "__main__":
    extract_districts()
    extract_household_benchmark()
