import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


INPUT_PATH = os.path.join("data", "processed", "features_ref_mc.csv")
EXPECTED_TEST_INDICES_PATH = os.path.join("reports", "test_indices.csv")
METRICS_PATH = os.path.join("reports", "metrics_consequence.csv")
MODEL_PATH = os.path.join("models", "xgb_consequence.joblib")
MAPPING_PATH = os.path.join("data", "processed", "consequence_mapping.csv")
FEATURE_COLUMNS = [
    "AF_precise",
    "SAS_AF",
    "EUR_AF",
    "AFR_AF",
    "EAS_AF",
    "AMR_AF",
    "in_1000g",
    "substitution",
    "is_transition",
    "is_CT_or_GA",
    "gc_50",
    "gc_500",
    "is_cpg",
    "is_cpg_transition",
    "consequence_group",
    "is_lof",
    "is_missense",
]


def build_feature_c_pipeline(y_train):
    numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in {"substitution", "consequence_group"}
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["substitution", "consequence_group"],
            ),
        ]
    )
    benign_count = int((y_train == 0).sum())
    pathogenic_count = int((y_train == 1).sum())
    if pathogenic_count == 0:
        raise ValueError("Training split has no pathogenic rows for XGBoost.")
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=benign_count / pathogenic_count,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


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
    train_indices, test_indices = next(
        split.split(df, df["label"], groups=groups)
    )

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
    pipeline = build_feature_c_pipeline(y_train)
    pipeline.fit(df.iloc[train_indices][FEATURE_COLUMNS], y_train)
    probabilities = pipeline.predict_proba(
        df.iloc[test_indices][FEATURE_COLUMNS]
    )[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    expected_metrics = pd.read_csv(METRICS_PATH)
    expected_metrics = expected_metrics.loc[
        (expected_metrics["model"] == "xgboost")
        & (expected_metrics["feature_set"] == "C")
        & (expected_metrics["subset"] == "ALL")
    ]
    if len(expected_metrics) != 1:
        print(
            "ERROR: Expected exactly one xgboost / C / ALL row in "
            f"{METRICS_PATH}, found {len(expected_metrics)}. Stopping."
        )
        raise SystemExit(1)
    expected_row = expected_metrics.iloc[0]
    if (
        abs(roc_auc - expected_row["roc_auc"]) >= 0.001
        or abs(pr_auc - expected_row["pr_auc"]) >= 0.001
    ):
        print(
            "ERROR: Rebuilt model metrics differ from "
            f"{METRICS_PATH} by at least 0.001. "
            f"ROC-AUC={roc_auc:.6f} (expected {expected_row['roc_auc']:.6f}); "
            f"PR-AUC={pr_auc:.6f} (expected {expected_row['pr_auc']:.6f}). "
            "Stopping without saving."
        )
        raise SystemExit(1)

    consequence_mapping = (
        df[["consequence_primary", "consequence_group"]]
        .drop_duplicates()
        .sort_values(["consequence_primary", "consequence_group"])
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MAPPING_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    consequence_mapping.to_csv(MAPPING_PATH, index=False)

    print(f"Test rows match {EXPECTED_TEST_INDICES_PATH}.")
    print(f"ROC-AUC: {roc_auc:.6f} (metrics report: {expected_row['roc_auc']:.6f})")
    print(f"PR-AUC: {pr_auc:.6f} (metrics report: {expected_row['pr_auc']:.6f})")
    print("Both metrics match within 0.001.")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved consequence mapping to {MAPPING_PATH}")


if __name__ == "__main__":
    main()
