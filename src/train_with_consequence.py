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
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


INPUT_PATH = os.path.join("data", "processed", "features_ref_mc.csv")
EXPECTED_TEST_INDICES_PATH = os.path.join("reports", "test_indices.csv")
METRICS_CSV_PATH = os.path.join("reports", "metrics_consequence.csv")
METRICS_MD_PATH = os.path.join("reports", "metrics_consequence.md")
FIGURES_DIR = os.path.join("reports", "figures")
MODEL_PATH = os.path.join("models", "rf_consequence.joblib")

POPULATION_FEATURES = [
    "AF_precise",
    "SAS_AF",
    "EUR_AF",
    "AFR_AF",
    "EAS_AF",
    "AMR_AF",
    "in_1000g",
]
BASE_FEATURES = [
    "substitution",
    "is_transition",
    "is_CT_or_GA",
    "gc_50",
    "gc_500",
    "is_cpg",
    "is_cpg_transition",
]
CONSEQUENCE_FEATURES = [
    "consequence_group",
    "is_lof",
    "is_missense",
]
FEATURE_SETS = {
    "A": POPULATION_FEATURES + BASE_FEATURES,
    "B": BASE_FEATURES,
    "C": POPULATION_FEATURES + BASE_FEATURES + CONSEQUENCE_FEATURES,
    "D": BASE_FEATURES + CONSEQUENCE_FEATURES,
}
METRIC_COLUMNS = [
    "model",
    "feature_set",
    "subset",
    "n_test_rows",
    "n_pathogenic",
    "random_pr_auc",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "best_f1_threshold",
    "best_f1",
]


def build_pipeline(model, feature_columns, scale_numeric=False):
    categorical_features = [
        column
        for column in feature_columns
        if column in {"substitution", "consequence_group"}
    ]
    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = [
        ("numeric", Pipeline(numeric_steps), numeric_features),
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def make_model(model_name, y_train):
    if model_name == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        return model, True

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        return model, False

    benign_count = int((y_train == 0).sum())
    pathogenic_count = int((y_train == 1).sum())
    if pathogenic_count == 0:
        raise ValueError("Training split has no pathogenic rows for XGBoost.")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=benign_count / pathogenic_count,
        eval_metric="logloss",
        random_state=42,
    )
    return model, False


def calculate_metrics(
    model_name,
    feature_set,
    subset_name,
    y_true,
    probabilities,
):
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    threshold_f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + np.finfo(float).eps)
    )
    best_index = int(np.argmax(threshold_f1))
    pathogenic_count = int((y_true == 1).sum())
    return {
        "model": model_name,
        "feature_set": feature_set,
        "subset": subset_name,
        "n_test_rows": len(y_true),
        "n_pathogenic": pathogenic_count,
        "random_pr_auc": pathogenic_count / len(y_true),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "best_f1_threshold": thresholds[best_index],
        "best_f1": threshold_f1[best_index],
    }


def markdown_table(frame):
    display = frame.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].map(
        lambda value: f"{value:.4f}"
    )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_PATH)
    if len(df) != 17_935:
        print(
            f"ERROR: Expected 17,935 input rows, but found {len(df):,}. "
            "Stopping."
        )
        raise SystemExit(1)

    missing_groups = pd.Series(
        [f"NA_{index}" for index in df.index],
        index=df.index,
    )
    groups = df["GeneSymbol"].where(df["GeneSymbol"].notna(), missing_groups)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(split.split(df, df["label"], groups=groups))

    expected_test_indices = pd.read_csv(EXPECTED_TEST_INDICES_PATH)["row_index"]
    if not np.array_equal(
        test_indices,
        expected_test_indices.to_numpy(dtype=int),
    ):
        print(
            "ERROR: Computed test rows do not match "
            f"{EXPECTED_TEST_INDICES_PATH}. Stopping."
        )
        raise SystemExit(1)

    y_train = df.iloc[train_indices]["label"].astype(int)
    y_test = df.iloc[test_indices]["label"].astype(int)
    all_test_rows = np.ones(len(test_indices), dtype=bool)
    missense_test_rows = (
        df.iloc[test_indices]["is_missense"].to_numpy(dtype=int) == 1
    )
    subsets = {
        "ALL": all_test_rows,
        "MISSENSE-ONLY": missense_test_rows,
    }

    metrics_rows = []
    roc_data = {"ALL": {}, "MISSENSE-ONLY": {}}
    best_rf_pr_auc = -np.inf
    best_rf_pipeline = None

    for model_name in (
        "logistic_regression",
        "random_forest",
        "xgboost",
    ):
        for set_name, feature_columns in FEATURE_SETS.items():
            model, scale_numeric = make_model(model_name, y_train)
            pipeline = build_pipeline(
                model,
                feature_columns,
                scale_numeric=scale_numeric,
            )
            pipeline.fit(
                df.iloc[train_indices][feature_columns],
                y_train,
            )
            probabilities = pipeline.predict_proba(
                df.iloc[test_indices][feature_columns]
            )[:, 1]

            for subset_name, row_mask in subsets.items():
                subset_y = y_test.iloc[np.flatnonzero(row_mask)]
                subset_probabilities = probabilities[row_mask]
                metric_row = calculate_metrics(
                    model_name,
                    set_name,
                    subset_name,
                    subset_y.to_numpy(),
                    subset_probabilities,
                )
                metrics_rows.append(metric_row)

                if set_name in {"A", "C"}:
                    false_positive_rate, true_positive_rate, _ = roc_curve(
                        subset_y,
                        subset_probabilities,
                    )
                    roc_data[subset_name][(model_name, set_name)] = (
                        false_positive_rate,
                        true_positive_rate,
                    )

                if (
                    model_name == "random_forest"
                    and set_name == "C"
                    and subset_name == "ALL"
                    and metric_row["pr_auc"] > best_rf_pr_auc
                ):
                    best_rf_pr_auc = metric_row["pr_auc"]
                    best_rf_pipeline = pipeline

    metrics = pd.DataFrame(metrics_rows, columns=METRIC_COLUMNS)
    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    metrics.to_csv(METRICS_CSV_PATH, index=False)
    if best_rf_pipeline is None:
        raise RuntimeError("The Random Forest feature-set C model was not trained.")
    joblib.dump(best_rf_pipeline, MODEL_PATH)

    markdown_sections = []
    tables = {}
    for subset_name in ("ALL", "MISSENSE-ONLY"):
        subset_metrics = metrics.loc[metrics["subset"] == subset_name].drop(
            columns="subset"
        )
        table = markdown_table(subset_metrics)
        tables[subset_name] = table
        markdown_sections.extend(
            [
                f"## {subset_name}",
                "",
                table,
                "",
            ]
        )
    with open(METRICS_MD_PATH, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(markdown_sections))

    for subset_name, filename in (
        ("ALL", "roc_curves_consequence_all.png"),
        ("MISSENSE-ONLY", "roc_curves_consequence_missense.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 6))
        for model_name in (
            "logistic_regression",
            "random_forest",
            "xgboost",
        ):
            for set_name, linestyle in (("C", "-"), ("A", "--")):
                false_positive_rate, true_positive_rate = roc_data[subset_name][
                    (model_name, set_name)
                ]
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    linestyle=linestyle,
                    label=f"{model_name} (set {set_name})",
                )
        axis.plot([0, 1], [0, 1], linestyle=":", color="gray")
        axis.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"ROC curves — {subset_name}",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
        plt.close(figure)

    missense_count = int(missense_test_rows.sum())
    missense_pathogenic_count = int(
        y_test.to_numpy()[missense_test_rows].sum()
    )
    random_missense_pr_auc = missense_pathogenic_count / missense_count
    print(
        "MISSENSE-ONLY test rows: "
        f"{missense_count}; pathogenic rows: {missense_pathogenic_count}; "
        f"random-guess PR-AUC: {random_missense_pr_auc:.6f}"
    )
    print("\nALL metrics:")
    print(tables["ALL"])
    print("\nMISSENSE-ONLY metrics:")
    print(tables["MISSENSE-ONLY"])


if __name__ == "__main__":
    main()
