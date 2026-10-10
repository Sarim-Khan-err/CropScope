from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from market_analysis import price_analysis, price_pattern_reference, road_and_season_reference, satellite_catalog_reference, source_yield_reference, supply_demand_reference, trade_reference
from evaluate_potato_satellite import build_result as potato_satellite_result

CROP_CLASSIFIER_CACHE = Path(__file__).resolve().parent / "outputs" / "crop_classifier_evaluation.json"


def crop_classifier_result():
    # Training takes ~2 minutes (gradient descent over 85k fields), too slow
    # to redo on every dashboard rebuild. Run `python evaluate_crop_classifier.py`
    # once (or after changing its method) to regenerate this cache.
    if not CROP_CLASSIFIER_CACHE.exists():
        raise FileNotFoundError(
            "outputs/crop_classifier_evaluation.json is missing. Run "
            "`python evaluate_crop_classifier.py` first to train and cache the "
            "satellite crop classifier results."
        )
    return json.loads(CROP_CLASSIFIER_CACHE.read_text(encoding="utf-8"))

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "onion_production.csv"
LATEST_NATIONAL = ROOT / "data" / "national_onion_latest.csv"
OUT = ROOT / "outputs"
PROVINCES = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"]


def read_records():
    with DATA.open(newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))
    if not records:
        raise ValueError("Production table contains no observations.")
    values = {}
    for row in records:
        season = row["season"].strip()
        province = row["province"].strip()
        value = float(row["production_thousand_tonnes"])
        if province not in PROVINCES or value < 0:
            raise ValueError(f"Invalid production row: {row}")
        values[(season, province)] = value * 1000
    seasons = sorted({season for season, _ in values}, key=lambda s: int(s[:4]))
    if any((season, province) not in values for season in seasons for province in PROVINCES):
        raise ValueError("Each season must contain all four provinces.")
    return seasons, values


def solve_linear(matrix, vector, ridge=1.0):
    """Solve a small ridge regression system with Gauss-Jordan elimination."""
    n = len(vector)
    a = [list(map(float, row)) + [float(vector[i])] for i, row in enumerate(matrix)]
    for i in range(1, n):
        a[i][i] += ridge
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            continue
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def national_feature(index, series):
    # Put production on the same numeric scale as time. Unscaled tonne values
    # made X'X poorly conditioned and the tiny ridge penalty effectively zero.
    scale = 1_000_000.0
    return [1.0, series[index - 1] / scale,
            statistics.mean(series[index - 3:index]) / scale,
            index / max(1, len(series) - 1)]


def fit_national_ridge(train_indices, series, ridge=0.05):
    rows = [national_feature(i, series) for i in train_indices if i >= 3]
    targets = [series[i] / 1_000_000.0 for i in train_indices if i >= 3]
    n = len(rows[0])
    xtx = [[sum(row[i] * row[j] for row in rows) for j in range(n)] for i in range(n)]
    xty = [sum(row[i] * y for row, y in zip(rows, targets)) for i in range(n)]
    # Standardized feature scales make this penalty meaningful and stable.
    return solve_linear(xtx, xty, ridge=ridge)


def predict_national_ridge(coefficients, index, series):
    return max(0.0, sum(a * b for a, b in zip(coefficients, national_feature(index, series)))) * 1_000_000.0


def select_ridge_penalty(index, series):
    """Select shrinkage using only earlier rolling validation seasons."""
    candidates = [0.01, 0.05, 0.15, 0.5, 1.0]
    inner_indices = range(6, index)
    if index <= 6:
        return 0.05
    scores = []
    for penalty in candidates:
        errors = []
        for validation_index in inner_indices:
            coefficients = fit_national_ridge(list(range(3, validation_index)), series, penalty)
            prediction = predict_national_ridge(coefficients, validation_index, series)
            errors.append(abs(series[validation_index] - prediction))
        if errors:
            scores.append((statistics.mean(errors), penalty))
    return min(scores)[1] if scores else 0.05


def run_pipeline():
    seasons, values = read_records()
    if len(seasons) < 8:
        raise ValueError("At least eight seasons are required for the rolling backtest.")
    OUT.mkdir(exist_ok=True)
    first_backtest_index = 6
    regional_history = []
    for season in seasons:
        row = {"season": season}
        for province in PROVINCES:
            row[province] = round(values[(season, province)])
        row["total"] = sum(row[p] for p in PROVINCES)
        regional_history.append(row)

    national_rows = [{"season": season,
                      "production_t": round(sum(values[(season, p)] for p in PROVINCES)),
                      "status": "reported; sum of province rows",
                      "source": "MNFSR Agricultural Statistics of Pakistan 2023-24, Table 35"}
                     for season in seasons]
    with LATEST_NATIONAL.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            national_rows.append({"season": row["season"],
                                  "production_t": round(float(row["production_thousand_tonnes"]) * 1000),
                                  "status": row["status"], "source": row["source_table"],
                                  "source_url": row["source_url"]})
    national_rows.sort(key=lambda row: int(row["season"][:4]))
    national_seasons = [row["season"] for row in national_rows]
    national_values = [float(row["production_t"]) for row in national_rows]

    errors_by_model = {"National ridge": [], "Last season": [], "Three-season mean": []}
    backtest_rows = []
    # Expanding-window validation uses only years before each held-out season.
    for index in range(first_backtest_index, len(national_values)):
        ridge_penalty = select_ridge_penalty(index, national_values)
        coefficients = fit_national_ridge(list(range(3, index)), national_values, ridge_penalty)
        predictions = {
            "National ridge": predict_national_ridge(coefficients, index, national_values),
            "Last season": national_values[index - 1],
            "Three-season mean": statistics.mean(national_values[index - 3:index]),
        }
        for model, predicted in predictions.items():
            actual = national_values[index]
            error = abs(actual - predicted)
            errors_by_model[model].append(error)
            backtest_rows.append({"season": national_seasons[index], "model": model,
                                  "actual_t": round(actual), "forecast_t": round(predicted),
                                  "absolute_error_t": round(error)})

    metrics = [{"model": model, "mae_t": round(statistics.mean(errors)),
                "wape_pct": round(100 * sum(errors) / sum(national_values[first_backtest_index:]), 1),
                "backtest_count": len(errors)} for model, errors in errors_by_model.items()]
    best_method = min(metrics, key=lambda item: item["mae_t"])["model"]
    final_ridge_penalty = select_ridge_penalty(len(national_values), national_values)
    if best_method == "National ridge":
        selected = predict_national_ridge(fit_national_ridge(list(range(3, len(national_values))), national_values, final_ridge_penalty), len(national_values), national_values)
    elif best_method == "Last season":
        selected = national_values[-1]
    else:
        selected = statistics.mean(national_values[-3:])
    ridge_value = predict_national_ridge(fit_national_ridge(list(range(3, len(national_values))), national_values, final_ridge_penalty), len(national_values), national_values)
    latest_start = int(national_seasons[-1][:4])
    next_season = f"{latest_start + 1}-{str((latest_start + 2) % 100).zfill(2)}"
    national_baseline = {"season": next_season, "baseline_t": round(selected),
                         "ridge_candidate_t": round(ridge_value), "selected_method": best_method,
                         "ridge_penalty": final_ridge_penalty,
                         "latest_input_status": national_rows[-1]["status"]}

    history = []
    history = regional_history

    dataset_info = {
        "status": "loaded",
        "source": "MNFSR Agricultural Statistics of Pakistan 2023-24 Table 35 (regional); Pakistan Economic Survey 2025-26 Table 2.3 (latest national totals)",
        "source_url": "https://mnfsr.gov.pk/SiteImage/Publication/ASP%20for%20Web%202024.pdf",
        "latest_source_url": "https://finance.gov.pk/survey/chapter_26/Complete_PES2025_26.pdf",
        "seasons": len(seasons), "first_season": seasons[0], "last_season": seasons[-1],
        "regions": len(PROVINCES), "observations": len(values),
        "unit_conversion": "Source values in thousand tonnes multiplied by 1,000.",
    }
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "dataset": dataset_info, "history": history, "national_history": national_rows,
        "national_baseline": national_baseline, "national_latest": national_rows[-1],
        "backtest": metrics, "backtest_rows": backtest_rows,
        "model_note": "Ridge regression uses prior national output, the trailing three-season mean, and a time trend. Its penalty is selected on earlier rolling seasons. An outer expanding-window backtest compares it with repeating last season and the three-season mean. The displayed next-season number is a historical benchmark from the lowest-MAE method, not a current crop measurement or a satellite-informed prediction. Province series ends at 2023-24; no later province shares are inferred.",
        "next_season": next_season,
        "price_status": "loaded",
        "price_analysis": price_analysis(),
        "price_pattern_reference": price_pattern_reference(),
        "supply_demand_reference": supply_demand_reference(),
        "road_and_season_reference": road_and_season_reference(),
        "source_yield_reference": source_yield_reference(),
        "satellite_catalog_reference": satellite_catalog_reference(),
        "trade_reference": trade_reference(),
        "crop_classifier": crop_classifier_result(),
        "potato_satellite": potato_satellite_result(),
        "price_source_url": "https://www.pbs.gov.pk/price-statistics/",
    }
    (OUT / "dashboard.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "national_baseline.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(national_baseline.keys()))
        writer.writeheader(); writer.writerow(national_baseline)
    with (OUT / "backtest_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
        writer.writeheader(); writer.writerows(metrics)
    with (OUT / "backtest_detail.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(backtest_rows[0].keys()))
        writer.writeheader(); writer.writerows(backtest_rows)
    from build_offline_dashboard import build
    build()
    return result


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "app"), **kwargs)

    def do_GET(self):
        if self.path == "/outputs/dashboard.json":
            payload = (OUT / "dashboard.json").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description="Run CropScope data pipeline and dashboard")
    parser.add_argument("--serve", action="store_true", help="serve the dashboard locally after generating real-data outputs")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    result = run_pipeline()
    print(f"Generated from {result['dataset']['observations']} official provincial observations and {len(result['national_history'])} national seasons.")
    print(f"National historical benchmark: {result['next_season']}; latest total status: {result['national_latest']['status']}.")
    if args.serve:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), AppHandler)
        print(f"Dashboard running at http://127.0.0.1:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()
