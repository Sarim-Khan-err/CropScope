"""Check two more onion-specific AI tasks against simple historical baselines.

Both trials reuse the expanding-window, no-future-leakage pattern already used
for the national ridge audit in run.py: a candidate is only trained on data
that would have existed strictly before the held-out observation. Neither
trial is wired into the dashboard; this script exists so the negative result
is reproducible rather than just asserted in the README.

1. District onion-yield trend: can a pooled ridge (district fixed effect +
   prior yield + trailing 3-season mean + time trend) beat last-season
   persistence for the four source districts with cross-checked yield
   history? All four districts share an identical 16-season list (verified
   below), so a position index is a safe calendar-aligned cutoff.

2. Weekly city onion price: can the same kind of pooled ridge (city fixed
   effect + prior week + trailing 4-week mean + time trend) beat week-to-week
   persistence for the 17 PBS-surveyed cities?
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def solve_linear(matrix, vector, ridge=1.0):
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
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def pooled_ridge_predict(train_rows, train_y, x_pred, ridge):
    nfeat = len(train_rows[0])
    xtx = [[sum(r[a] * r[b] for r in train_rows) for b in range(nfeat)] for a in range(nfeat)]
    xty = [sum(r[a] * y for r, y in zip(train_rows, train_y)) for a in range(nfeat)]
    coeffs = solve_linear(xtx, xty, ridge=ridge)
    return max(0.0, sum(a * b for a, b in zip(coeffs, x_pred)))


def district_yield_trial():
    with (ROOT / "data" / "source_district_onion_yield_history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_district = defaultdict(list)
    for row in rows:
        by_district[row["district"]].append(row)
    districts = sorted(by_district)
    for d in districts:
        by_district[d].sort(key=lambda r: r["season"])
    seasons_by_district = {d: [r["season"] for r in by_district[d]] for d in districts}
    if len({tuple(v) for v in seasons_by_district.values()}) != 1:
        raise ValueError("Districts do not share an identical season list; position-index cutoff is unsafe")
    series = {d: [float(r["yield_t_per_ha"]) for r in by_district[d]] for d in districts}
    n_seasons = len(seasons_by_district[districts[0]])
    n_districts = len(districts)
    min_train = 5

    def features(d_idx, prior, trail3, t_norm):
        return [1.0 if j == d_idx else 0.0 for j in range(n_districts)] + [prior, trail3, t_norm]

    persistence_errs, mean3_errs, ridge_errs = [], [], []
    detail = []
    for d_idx, d in enumerate(districts):
        s = series[d]
        for i in range(min_train, n_seasons):
            actual = s[i]
            last = s[i - 1]
            trail3 = statistics.mean(s[max(0, i - 3):i])
            persistence_errs.append(abs(actual - last))
            mean3_errs.append(abs(actual - trail3))
            train_rows, train_y = [], []
            for j, other in enumerate(districts):
                o = series[other]
                for k in range(1, i):
                    train_rows.append(features(j, o[k - 1], statistics.mean(o[max(0, k - 3):k]), k / max(1, n_seasons - 1)))
                    train_y.append(o[k])
            pred = pooled_ridge_predict(train_rows, train_y, features(d_idx, last, trail3, i / max(1, n_seasons - 1)), ridge=2.0)
            ridge_errs.append(abs(actual - pred))
            detail.append({"district": d, "season": seasons_by_district[d][i], "actual_t_per_ha": actual,
                           "last_season_t_per_ha": last, "trail3_t_per_ha": round(trail3, 2),
                           "pooled_ridge_pred_t_per_ha": round(pred, 2)})
    summary = [
        {"task": "District yield (t/ha)", "model": "Last season", "mae": round(statistics.mean(persistence_errs), 3), "n": len(persistence_errs)},
        {"task": "District yield (t/ha)", "model": "3-season mean", "mae": round(statistics.mean(mean3_errs), 3), "n": len(mean3_errs)},
        {"task": "District yield (t/ha)", "model": "Pooled ridge (district FE)", "mae": round(statistics.mean(ridge_errs), 3), "n": len(ridge_errs)},
    ]
    return summary, detail


def weekly_price_trial():
    with (ROOT / "data" / "pbs_onion_weekly.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_week = defaultdict(dict)
    for row in rows:
        by_week[row["week"]][row["city"]] = float(row["price_rs_per_kg"])
    weeks = sorted(by_week)
    cities = sorted(set(by_week[weeks[0]]) - {"Pakistan"})
    series = {c: [by_week[w][c] for w in weeks] for c in cities}
    n_weeks = len(weeks)
    n_cities = len(cities)
    min_train = 20
    scale = 100.0

    def features(c_idx, prior, trail4, t_norm):
        return [1.0 if j == c_idx else 0.0 for j in range(n_cities)] + [prior / scale, trail4 / scale, t_norm]

    def fit(cutoff, ridge):
        train_rows, train_y = [], []
        for j, c in enumerate(cities):
            s = series[c]
            for k in range(4, cutoff):
                train_rows.append(features(j, s[k - 1], statistics.mean(s[k - 4:k]), k / max(1, n_weeks - 1)))
                train_y.append(s[k] / scale)
        nfeat = len(train_rows[0])
        xtx = [[sum(r[a] * r[b] for r in train_rows) for b in range(nfeat)] for a in range(nfeat)]
        xty = [sum(r[a] * y for r, y in zip(train_rows, train_y)) for a in range(nfeat)]
        return solve_linear(xtx, xty, ridge=ridge)

    def predict(coeffs, c_idx, prior, trail4, t_norm):
        x = features(c_idx, prior, trail4, t_norm)
        return max(0.0, sum(a * b for a, b in zip(coeffs, x))) * scale

    def select_penalty(cutoff):
        candidates = [0.5, 1.0, 2.0, 5.0, 10.0]
        if cutoff <= min_train + 1:
            return 1.0
        scores = []
        for penalty in candidates:
            errs = []
            for val_i in range(min_train, cutoff):
                coeffs = fit(val_i, penalty)
                for j, c in enumerate(cities):
                    s = series[c]
                    pred = predict(coeffs, j, s[val_i - 1], statistics.mean(s[val_i - 4:val_i]), val_i / max(1, n_weeks - 1))
                    errs.append(abs(s[val_i] - pred))
            scores.append((statistics.mean(errs), penalty))
        return min(scores)[1]

    persistence_errs, ridge_errs = [], []
    detail = []
    for i in range(min_train, n_weeks):
        penalty = select_penalty(i)
        coeffs = fit(i, penalty)
        for j, c in enumerate(cities):
            s = series[c]
            actual, prior = s[i], s[i - 1]
            trail4 = statistics.mean(s[i - 4:i])
            pred = predict(coeffs, j, prior, trail4, i / max(1, n_weeks - 1))
            persistence_errs.append(abs(actual - prior))
            ridge_errs.append(abs(actual - pred))
            detail.append({"week": weeks[i], "city": c, "actual_rs_per_kg": actual,
                           "last_week_rs_per_kg": prior, "pooled_ridge_pred_rs_per_kg": round(pred, 2),
                           "ridge_penalty": penalty})
    summary = [
        {"task": "Weekly city price (Rs/kg)", "model": "Last week", "mae": round(statistics.mean(persistence_errs), 2), "n": len(persistence_errs)},
        {"task": "Weekly city price (Rs/kg)", "model": "Pooled ridge (city FE)", "mae": round(statistics.mean(ridge_errs), 2), "n": len(ridge_errs)},
    ]
    return summary, detail


def main():
    OUT.mkdir(exist_ok=True)
    yield_summary, yield_detail = district_yield_trial()
    price_summary, price_detail = weekly_price_trial()
    summary = yield_summary + price_summary
    with (OUT / "additional_task_trials_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "district_yield_trial_detail.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(yield_detail[0]))
        writer.writeheader()
        writer.writerows(yield_detail)
    with (OUT / "price_trial_detail.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(price_detail[0]))
        writer.writeheader()
        writer.writerows(price_detail)
    for row in summary:
        print(f"{row['task']} | {row['model']}: MAE={row['mae']} (n={row['n']})")
    print("Neither trial beat its simple baseline. No model from this script is used in the dashboard.")


if __name__ == "__main__":
    main()
