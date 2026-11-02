import os

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


INPUT_PATH = os.path.join("data", "processed", "features_ref_mc.csv")
OUTPUT_PATH = os.path.join("reports", "consequence_only_baseline.md")


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

    train = df.iloc[train_indices]
    test = df.iloc[test_indices]
    overall_train_rate = train["label"].mean()
    consequence_rates = train.groupby("consequence_group")["label"].mean()
    predicted_probabilities = (
        test["consequence_group"]
        .map(consequence_rates)
        .fillna(overall_train_rate)
        .to_numpy()
    )
    test_labels = test["label"].to_numpy()

    predictions = (predicted_probabilities >= 0.5).astype(int)
    all_metrics = {
        "ROC-AUC": roc_auc_score(test_labels, predicted_probabilities),
        "PR-AUC": average_precision_score(
            test_labels,
            predicted_probabilities,
        ),
        "F1 at threshold 0.5": f1_score(
            test_labels,
            predictions,
            zero_division=0,
        ),
    }

    missense_mask = test["is_missense"].to_numpy(dtype=int) == 1
    missense_roc_auc = roc_auc_score(
        test_labels[missense_mask],
        predicted_probabilities[missense_mask],
    )

    lines = [
        "# Consequence-only baseline",
        "",
        f"- Training rows: {len(train):,}",
        f"- Test rows: {len(test):,}",
        f"- Overall training pathogenic rate (fallback): "
        f"{overall_train_rate:.6f}",
        "",
        "## All test rows",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {metric} | {value:.6f} |"
        for metric, value in all_metrics.items()
    )
    lines.extend(
        [
            "",
            "## Missense-only test rows",
            "",
            f"- Test rows: {int(missense_mask.sum()):,}",
            f"- ROC-AUC: {missense_roc_auc:.6f}",
            "",
        ]
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(lines))

    print("All test rows:")
    for metric, value in all_metrics.items():
        print(f"{metric}: {value:.6f}")
    print(f"Missense-only test rows: {int(missense_mask.sum()):,}")
    print(f"Missense-only ROC-AUC: {missense_roc_auc:.6f}")
    print(f"Saved report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
