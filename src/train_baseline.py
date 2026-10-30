import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = os.path.join("data", "processed", "features_ref.csv")
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = "models"

POPULATION_FEATURES = [
    "AF_precise", "SAS_AF", "EUR_AF", "AFR_AF", "EAS_AF", "AMR_AF",
    "in_1000g",
]
BASE_FEATURES = [
    "substitution", "is_transition", "is_CT_or_GA", "gc_50", "gc_500",
    "is_cpg", "is_cpg_transition",
]


def build_pipeline(model, numeric_features, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "substitution",
                OneHotEncoder(handle_unknown="ignore"),
                ["substitution"],
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def make_model(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        )
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    missing_groups = pd.Series(
        [f"NA_{index}" for index in df.index], index=df.index
    )
    groups = df["GeneSymbol"].where(df["GeneSymbol"].notna(), missing_groups)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        split.split(df, df["label"], groups=groups)
    )
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    print(f"Total unique groups/genes: {groups.nunique()}")
    print(f"Training unique groups/genes: {len(train_groups)}")
    print(f"Test unique groups/genes: {len(test_groups)}")
    print(
        "Training label counts:",
        df.iloc[train_indices]["label"].value_counts().sort_index().to_dict(),
    )
    print(
        "Test label counts:",
        df.iloc[test_indices]["label"].value_counts().sort_index().to_dict(),
    )
    print(f"Gene groups overlap: {bool(train_groups & test_groups)}")

    pd.DataFrame({"row_index": test_indices}).to_csv(
        os.path.join(REPORTS_DIR, "test_indices.csv"), index=False
    )
    y_train = df.iloc[train_indices]["label"]
    y_test = df.iloc[test_indices]["label"]
    metrics_rows = []
    roc_curves = []

    feature_sets = {
        "A": POPULATION_FEATURES + BASE_FEATURES,
        "B": BASE_FEATURES,
    }

    for model_name in ("logistic_regression", "random_forest"):
        for set_name, feature_columns in feature_sets.items():
            numeric_features = [
                column for column in feature_columns
                if column != "substitution"
            ]
            scale = model_name == "logistic_regression"
            pipeline = build_pipeline(
                make_model(model_name), numeric_features, scale_numeric=scale
            )
            pipeline.fit(
                df.iloc[train_indices][feature_columns], y_train
            )
            X_test = df.iloc[test_indices][feature_columns]
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            metrics_rows.append({
                "model": model_name,
                "feature_set": set_name,
                "precision": precision_score(
                    y_test, predictions, pos_label=1, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, pos_label=1, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, pos_label=1, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
            })

            figure, axis = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Benign", "Pathogenic"],
            ).plot(ax=axis, colorbar=False, cmap="Blues")
            axis.set_title(f"{model_name} — Feature set {set_name}")
            figure.tight_layout()
            figure.savefig(
                os.path.join(
                    FIGURES_DIR,
                    f"confusion_{model_name}_{set_name}.png",
                ),
                dpi=150,
            )
            plt.close(figure)

            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test, probabilities
            )
            roc_curves.append((
                f"{model_name} (set {set_name})",
                false_positive_rate,
                true_positive_rate,
            ))
            if model_name == "random_forest" and set_name == "A":
                joblib.dump(
                    pipeline, os.path.join(MODELS_DIR, "rf_full.joblib")
                )

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(REPORTS_DIR, "metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    markdown_path = os.path.join(REPORTS_DIR, "metrics.md")
    with open(markdown_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("| " + " | ".join(metrics.columns) + " |\n")
        markdown_file.write(
            "| " + " | ".join("---" for _ in metrics.columns) + " |\n"
        )
        for _, row in metrics.iterrows():
            values = [
                f"{value:.4f}" if isinstance(value, (float, np.floating))
                else str(value)
                for value in row
            ]
            markdown_file.write("| " + " | ".join(values) + " |\n")

    figure, axis = plt.subplots(figsize=(7, 6))
    for label, false_positive_rate, true_positive_rate in roc_curves:
        axis.plot(false_positive_rate, true_positive_rate, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close(figure)

    print("\nMetrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nPopulation-frequency comparison (A vs B):")
    mean_metrics = metrics.groupby("feature_set")[[
        "roc_auc", "pr_auc", "f1"
    ]].mean()
    print(
        "Across both models, adding frequencies changed mean ROC-AUC by "
        f"{mean_metrics.loc['A', 'roc_auc'] - mean_metrics.loc['B', 'roc_auc']:+.4f}, "
        f"PR-AUC by {mean_metrics.loc['A', 'pr_auc'] - mean_metrics.loc['B', 'pr_auc']:+.4f}, "
        f"and F1 by {mean_metrics.loc['A', 'f1'] - mean_metrics.loc['B', 'f1']:+.4f}."
    )
    for model_name in ("logistic_regression", "random_forest"):
        model_metrics = metrics.set_index(
            ["model", "feature_set"]
        ).loc[model_name]
        set_a = model_metrics.loc["A"]
        set_b = model_metrics.loc["B"]
        print(
            f"{model_name}: set A minus B: "
            f"ROC-AUC {set_a['roc_auc'] - set_b['roc_auc']:+.4f}, "
            f"PR-AUC {set_a['pr_auc'] - set_b['pr_auc']:+.4f}, "
            f"F1 {set_a['f1'] - set_b['f1']:+.4f}."
        )


if __name__ == "__main__":
    main()
