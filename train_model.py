"""
Phase 2: train and evaluate static hand gesture classifiers.

Run:
    python train_model.py

The script trains SVM and KNN classifiers, prints accuracy, confusion matrix,
precision, recall, and F1-score, then saves the trained models and scaler.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parent
DATASET_CSV = ROOT_DIR / "gesture_dataset.csv"

SCALER_PATH = ROOT_DIR / "scaler.pkl"
BEST_MODEL_PATH = ROOT_DIR / "best_model.pkl"
SVM_MODEL_PATH = ROOT_DIR / "svm_model.pkl"
KNN_MODEL_PATH = ROOT_DIR / "knn_model.pkl"
METRICS_COMPARISON_CSV = ROOT_DIR / "model_metrics_comparison.csv"
CONFUSION_MATRIX_CSV = ROOT_DIR / "confusion_matrix_comparison.csv"

RANDOM_STATE = 42
LABEL_COLUMN = "label"
EXPECTED_LABELS = [
    "clasp",
    "down",
    "home",
    "lb",
    "left",
    "leftrunning",
    "lt",
    "paper",
    "rb",
    "right",
    "rock",
    "rt",
    "scissors",
    "thumb",
    "up",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Read gesture_dataset.csv and split it into feature data and labels."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV was not found: {DATASET_CSV}\n"
            "Run Phase 1 first: python extract_features.py"
        )

    data = pd.read_csv(DATASET_CSV)
    if data.empty:
        raise ValueError("gesture_dataset.csv is empty and cannot be used for training.")

    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"Missing label column in dataset: {LABEL_COLUMN}")

    feature_columns = [column for column in data.columns if column != LABEL_COLUMN]
    if len(feature_columns) != 60:
        raise ValueError(
            f"Expected 60 feature columns, found {len(feature_columns)}. "
            "Check the Phase 1 feature extraction output."
        )

    X = data[feature_columns]
    y = data[LABEL_COLUMN]

    unexpected_labels = sorted(set(y.unique()) - set(EXPECTED_LABELS))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels in dataset: {unexpected_labels}")

    missing_labels = [label for label in EXPECTED_LABELS if label not in set(y.unique())]
    if missing_labels:
        raise ValueError(
            f"Dataset is missing required labels: {missing_labels}. "
            "Make sure gesture_data_sample contains all 15 gesture folders, "
            "then rerun: python extract_features.py"
        )

    class_counts = y.value_counts()
    too_small = [label for label in EXPECTED_LABELS if class_counts[label] < 2]
    if too_small:
        raise ValueError(f"Each class needs at least two samples. Too small: {too_small}")

    return X, y


def print_dataset_summary(y: pd.Series) -> None:
    """Print the class distribution for the current dataset."""
    print("=" * 60)
    print("Dataset Summary")
    print(f"Total samples: {len(y)}")
    print("Class distribution:")
    for label in EXPECTED_LABELS:
        print(f"  {label}: {int((y == label).sum())}")
    print("=" * 60)


def evaluate_model(
    model_name: str,
    model,
    X_test,
    y_test,
    labels: list[str],
) -> tuple[float, list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate a model with accuracy, confusion matrix, and class metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n[{model_name}]")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(matrix, index=labels, columns=labels))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    metric_rows: list[dict[str, object]] = []
    for label in labels:
        metric_rows.append(
            {
                "model": model_name,
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": report[label]["support"],
                "accuracy": accuracy,
            }
        )

    for aggregate_label in ["macro avg", "weighted avg"]:
        metric_rows.append(
            {
                "model": model_name,
                "label": aggregate_label,
                "precision": report[aggregate_label]["precision"],
                "recall": report[aggregate_label]["recall"],
                "f1_score": report[aggregate_label]["f1-score"],
                "support": report[aggregate_label]["support"],
                "accuracy": accuracy,
            }
        )

    confusion_rows: list[dict[str, object]] = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            confusion_rows.append(
                {
                    "model": model_name,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )

    return accuracy, metric_rows, confusion_rows


def write_evaluation_csvs(
    metric_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
) -> None:
    """Write visualizable CSV files for model metric and confusion-matrix comparison."""
    pd.DataFrame(metric_rows).to_csv(METRICS_COMPARISON_CSV, index=False)
    pd.DataFrame(confusion_rows).to_csv(CONFUSION_MATRIX_CSV, index=False)


def main() -> None:
    """Train, evaluate, and persist the gesture classifiers."""
    X, y = load_dataset()
    print_dataset_summary(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "KNN_5": KNeighborsClassifier(n_neighbors=5),
    }

    trained_models = {}
    scores = {}
    all_metric_rows: list[dict[str, object]] = []
    all_confusion_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        accuracy, metric_rows, confusion_rows = evaluate_model(
            model_name=model_name,
            model=model,
            X_test=X_test_scaled,
            y_test=y_test,
            labels=EXPECTED_LABELS,
        )
        scores[model_name] = accuracy
        all_metric_rows.extend(metric_rows)
        all_confusion_rows.extend(confusion_rows)

    best_model_name = max(scores, key=scores.get)
    best_model = trained_models[best_model_name]
    best_accuracy = scores[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(trained_models["SVM_RBF"], SVM_MODEL_PATH)
    joblib.dump(trained_models["KNN_5"], KNN_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    write_evaluation_csvs(all_metric_rows, all_confusion_rows)

    print("\n" + "=" * 60)
    print("Phase 2 model training completed")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best model saved: {BEST_MODEL_PATH}")
    print(f"SVM model saved: {SVM_MODEL_PATH}")
    print(f"KNN model saved: {KNN_MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Metrics comparison CSV saved: {METRICS_COMPARISON_CSV}")
    print(f"Confusion matrix CSV saved: {CONFUSION_MATRIX_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
