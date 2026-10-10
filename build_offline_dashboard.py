"""Package the generated dashboard data into one local HTML file."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "app" / "index.html"
DATA = ROOT / "outputs" / "dashboard.json"
TARGET = ROOT / "dashboard_offline.html"


def build() -> Path:
    html = SOURCE.read_text(encoding="utf-8")
    payload = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    marker = "  <script>\n    const COLORS="
    if html.count(marker) != 1:
        raise ValueError("Could not locate dashboard script insertion point")
    html = html.replace(marker, f"  <script>window.__DASHBOARD_DATA__={payload};</script>\n" + marker, 1)
    TARGET.write_text(html, encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    print(build())
