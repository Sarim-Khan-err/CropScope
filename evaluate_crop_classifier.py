"""Real satellite crop-type classifier: field-level Sentinel-2 time series vs
Asian Development Bank crop-type ground truth for Pakistan (Zenodo 22806700,
Ali et al. 2026, CC BY 4.0). 85,618 surveyed fields, 79 districts, two
seasons. See data/sources/crop_classification/SOURCE_README.md for full
provenance, class balance, and the authors' own leakage warning.

Evaluation follows the dataset authors' explicit instruction: split by
District, never by field, because neighbouring fields are spatially
correlated and a random split inflates accuracy. Whole districts are held
out for testing; no district appears in both splits.

Model is a plain multinomial (softmax) logistic regression over the
flattened 7x28 feature tensor, trained with class-balanced weights by
gradient descent in numpy -- no external ML framework, consistent with the
rest of this project. Compared against a majority-class baseline on the
same held-out districts, using macro-F1 (the metric the dataset's own
documentation recommends, since both seasons are steeply imbalanced).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "sources" / "crop_classification"
OUT = ROOT / "outputs"
SEASONS = ["rabi", "kharif"]
TEST_DISTRICT_FRACTION = 0.2
SPLIT_SEED = 42
L2 = 1.0
LEARNING_RATE = 0.3
MOMENTUM = 0.9
ITERATIONS = 1500


# Per the dataset's own SOURCE_README: Rabi contains ~118 off-season rarity
# fields (Cotton, Sorghum, Jantar, Rice -- normally Kharif crops) with 1-14
# samples each. The paper drops these before reporting; we do the same,
# documented rather than silently filtered.
RABI_OFF_SEASON_RARITIES = {"Cotton", "Sorghum", "Jantar", "Rice"}


def load_season(season):
    X = np.load(SRC / f"X_{season}.npy")
    y = np.load(SRC / f"y_{season}.npy")
    with (SRC / f"meta_{season}.csv").open(encoding="utf-8") as file:
        meta = list(csv.DictReader(file))
    if len(meta) != len(y):
        raise ValueError(f"{season}: metadata row count does not match label count")
    if season == "rabi":
        keep = np.array([row["crop_type"] not in RABI_OFF_SEASON_RARITIES for row in meta])
        X, y = X[keep], y[keep]
        meta = [row for row, k in zip(meta, keep) if k]
    districts = np.array([row["District"] for row in meta])
    return X.reshape(len(y), -1).astype(np.float64), y, districts, meta


def load_label_names():
    with (SRC / "label_map.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {int(row["label"]): row["crop_type"] for row in rows}


def district_split(districts, seed, test_fraction):
    unique = sorted(set(districts))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    test_districts = set(unique[:n_test])
    is_test = np.array([d in test_districts for d in districts])
    return ~is_test, is_test, sorted(test_districts)


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def train_softmax(X_train, y_train, n_classes, class_weight):
    n, d = X_train.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    vW, vb = np.zeros_like(W), np.zeros_like(b)
    Y = one_hot(y_train, n_classes)
    sample_weight = class_weight[y_train]
    sample_weight = sample_weight * n / sample_weight.sum()  # keep gradient scale stable
    for _ in range(ITERATIONS):
        logits = X_train @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        error = (probs - Y) * sample_weight[:, None]
        grad_W = X_train.T @ error / n + L2 * W / n
        grad_b = error.sum(axis=0) / n
        vW = MOMENTUM * vW - LEARNING_RATE * grad_W
        vb = MOMENTUM * vb - LEARNING_RATE * grad_b
        W += vW
        b += vb
    return W, b


def predict(X, W, b):
    logits = X @ W + b
    return logits.argmax(axis=1)


def macro_f1_and_report(y_true, y_pred, class_ids, label_names):
    report = []
    f1s = []
    for c in class_ids:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = int(np.sum(y_true == c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report.append({"crop": label_names[c], "support": support,
                       "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)})
        if support > 0:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    accuracy = float(np.mean(y_pred == y_true))
    return macro_f1, accuracy, report


def confusion_matrix(y_true, y_pred, class_ids, label_names):
    matrix = {label_names[c]: {label_names[k]: 0 for k in class_ids} for c in class_ids}
    for t, p in zip(y_true, y_pred):
        matrix[label_names[int(t)]][label_names[int(p)]] += 1
    return matrix


def run_season(season, label_names):
    X, y, districts, meta = load_season(season)
    train_mask, test_mask, test_districts = district_split(districts, SPLIT_SEED, TEST_DISTRICT_FRACTION)
    X_train, X_test = standardize(X[train_mask], X[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]
    class_ids = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    n_classes = max(class_ids) + 1

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_weight = np.zeros(n_classes)
    present = counts > 0
    class_weight[present] = counts[present].sum() / (present.sum() * counts[present])

    W, b = train_softmax(X_train, y_train, n_classes, class_weight)
    y_pred = predict(X_test, W, b)
    model_macro_f1, model_accuracy, model_report = macro_f1_and_report(y_test, y_pred, class_ids, label_names)

    majority_class = int(np.bincount(y_train, minlength=n_classes).argmax())
    y_baseline = np.full_like(y_test, majority_class)
    baseline_macro_f1, baseline_accuracy, baseline_report = macro_f1_and_report(y_test, y_baseline, class_ids, label_names)

    return {
        "season": season.capitalize(),
        "n_districts_train": len(set(districts[train_mask])),
        "n_districts_test": len(test_districts),
        "test_districts": test_districts,
        "n_fields_train": int(train_mask.sum()),
        "n_fields_test": int(test_mask.sum()),
        "crop_classes": [label_names[c] for c in class_ids],
        "model": {"macro_f1": round(model_macro_f1, 4), "accuracy": round(model_accuracy, 4), "per_crop": model_report},
        "majority_baseline": {"predicted_crop": label_names[majority_class],
                              "macro_f1": round(baseline_macro_f1, 4), "accuracy": round(baseline_accuracy, 4),
                              "per_crop": baseline_report},
        "macro_f1_improvement": round(model_macro_f1 - baseline_macro_f1, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, class_ids, label_names),
    }


def build():
    label_names = load_label_names()
    seasons = {season: run_season(season, label_names) for season in SEASONS}
    result = {
        "status": "loaded",
        "dataset": "Field-level Sentinel-2 time series for Pakistan crop classification (Ali et al. 2026)",
        "source_url": "https://zenodo.org/records/22806700",
        "code_url": "https://github.com/rms-ali247/conformal-crop-sits",
        "ground_truth_source": "Asian Development Bank crop-type survey of Pakistan",
        "license": "CC BY 4.0",
        "field_count": 85618, "district_count": 79,
        "split_method": f"Grouped by District (never by field); {int(TEST_DISTRICT_FRACTION*100)}% of districts held out entirely for test, seed {SPLIT_SEED}.",
        "rabi_filter_note": "Rabi excludes ~118 off-season rarity fields (Cotton/Sorghum/Jantar/Rice, 1-14 samples each) per the dataset authors' own documented filter, matching their published methodology. Kharif is unfiltered.",
        "model_method": "Multinomial (softmax) logistic regression on flattened 7-month x 28-channel Sentinel-2 feature tensor, class-balanced weights, trained by momentum gradient descent in numpy. No external ML framework.",
        "seasons": seasons,
        "note": "Real ground-truth labels (ADB survey), real Sentinel-2-derived features, a genuine multi-class classifier, and a spatial (district-held-out) evaluation against a majority-class baseline using macro-F1. This is Pakistani agriculture, but field crops (wheat, rice, cotton, sugarcane, etc.), not onions -- no onion labels exist in this dataset. Not used anywhere in the onion Monitor.",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "crop_classifier_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = build()
    for season, data in result["seasons"].items():
        print(f"{data['season']}: model macro-F1={data['model']['macro_f1']} vs baseline={data['majority_baseline']['macro_f1']} "
              f"(+{data['macro_f1_improvement']}), accuracy {data['model']['accuracy']} vs {data['majority_baseline']['accuracy']}")
        print(f"  train districts={data['n_districts_train']} fields={data['n_fields_train']}; "
              f"test districts={data['n_districts_test']} fields={data['n_fields_test']}")
