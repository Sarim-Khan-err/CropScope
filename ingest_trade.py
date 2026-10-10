"""Extract onion (HS 7031000, "Onions and shallots") import/export rows from
PBS's official commodity-by-country trade report.

Source: Pakistan Bureau of Statistics (D-10) Imports/Exports by Commodities
and Countries, cumulative July 2024 to June 2025 (FY2024-25). Country rows
with zero quantity and value in both years are dropped.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".vendor"))
from openpyxl import load_workbook  # noqa: E402

FILES = {
    "import": (ROOT / "data" / "sources" / "trade" / "D-10_Import-FY2024-25.xlsx",
               "https://www.pbs.gov.pk/wp-content/uploads/2020/07/D-10_Import-0625-1.xlsx"),
    "export": (ROOT / "data" / "sources" / "trade" / "D-10_Export-FY2024-25.xlsx",
               "https://www.pbs.gov.pk/wp-content/uploads/2020/07/D-10_Export-0625.xlsx"),
}
OUT = ROOT / "data" / "onion_trade_fy2024_25.csv"
HS_CODE = "7031000"
RETRIEVED_DATE = "2026-10-10"


def onion_block(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    for i, row in enumerate(rows):
        hs = row[0]
        if isinstance(hs, (int, float)) and str(int(hs)) == HS_CODE:
            start = i
            break
    else:
        raise ValueError(f"HS code {HS_CODE} not found in {path}")
    block = [rows[start]]
    j = start + 1
    while j < len(rows) and rows[j][0] is None:
        block.append(rows[j])
        j += 1
    unit = block[0][2]
    if unit != "KG":
        raise ValueError(f"Expected KG unit for onions, got {unit}")
    return block


def build():
    out_rows = []
    for direction, (path, source_url) in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing cached source file: {path}")
        block = onion_block(path)
        total_row = block[0]
        out_rows.append({
            "direction": direction, "country": "ALL COUNTRIES",
            "qty_kg_fy2024_25": int(total_row[5]), "value_pkr_thousands_fy2024_25": round(float(total_row[6]), 3),
            "qty_kg_june2025": int(total_row[3]), "value_pkr_thousands_june2025": round(float(total_row[4]), 3),
            "source_url": source_url, "retrieved_date": RETRIEVED_DATE,
        })
        for row in block[1:]:
            qty_fy, value_fy = row[5] or 0, row[6] or 0
            qty_m, value_m = row[3] or 0, row[4] or 0
            if qty_fy == 0 and value_fy == 0 and qty_m == 0 and value_m == 0:
                continue
            out_rows.append({
                "direction": direction, "country": row[1],
                "qty_kg_fy2024_25": int(qty_fy), "value_pkr_thousands_fy2024_25": round(float(value_fy), 3),
                "qty_kg_june2025": int(qty_m), "value_pkr_thousands_june2025": round(float(value_m), 3),
                "source_url": source_url, "retrieved_date": RETRIEVED_DATE,
            })
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} onion trade rows (HS {HS_CODE}, FY2024-25) to {OUT}")


if __name__ == "__main__":
    build()
