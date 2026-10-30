import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


FEATURES_PATH = os.path.join("data", "processed", "features_ref.csv")
MODEL_PATH = os.path.join("models", "rf_full.joblib")
TEST_INDICES_PATH = os.path.join("reports", "test_indices.csv")
FIGURES_DIR = os.path.join("reports", "figures")
FEATURE_COLUMNS = [
    "AF_precise", "SAS_AF", "EUR_AF", "AFR_AF", "EAS_AF", "AMR_AF",
    "in_1000g", "substitution", "is_transition", "is_CT_or_GA", "gc_50",
    "gc_500", "is_cpg", "is_cpg_transition",
]


def pathogenic_shap_values(shap_output, feature_count):
    """Select class-1 SHAP values from supported TreeExplainer output formats."""
    if isinstance(shap_output, list):
        if len(shap_output) != 2:
            raise ValueError("Expected SHAP output for two classifier classes.")
        values = np.asarray(shap_output[1])
    else:
        values = np.asarray(shap_output)
        if values.ndim == 3:
            if values.shape[1] == feature_count and values.shape[2] == 2:
                values = values[:, :, 1]
            elif values.shape[1] == 2 and values.shape[2] == feature_count:
                values = values[:, 1, :]
            else:
                raise ValueError(
                    f"Unexpected 3D SHAP output shape: {values.shape}"
                )
        elif values.ndim != 2:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    if values.shape[1] != feature_count:
        raise ValueError(
            f"SHAP feature count {values.shape[1]} does not match "
            f"transformed feature count {feature_count}."
        )
    return values


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    test_indices = pd.read_csv(TEST_INDICES_PATH)["row_index"].to_numpy()
    test_df = df.loc[test_indices]
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out()

    sample_df = test_df.sample(
        n=min(2000, len(test_df)), random_state=42
    )
    transformed = preprocessor.transform(sample_df[FEATURE_COLUMNS])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)

    explainer = shap.TreeExplainer(classifier)
    raw_shap_values = explainer.shap_values(transformed)
    values = pathogenic_shap_values(
        raw_shap_values, len(transformed_names)
    )
    base_values = np.asarray(explainer.expected_value)
    base_value = float(base_values[1] if base_values.ndim else base_values)

    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(sample_df), base_value),
        data=transformed,
        feature_names=transformed_names,
    )

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "shap_summary.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "shap_bar.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    predictions = pipeline.predict(sample_df[FEATURE_COLUMNS])
    labels = sample_df["label"].to_numpy()
    sample_positions = sample_df.index.to_numpy()
    for label, suffix in ((1, "pathogenic"), (0, "benign")):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) == 0:
            raise RuntimeError(
                f"No correctly predicted {suffix} variant in the SHAP sample."
            )
        sample_position = int(candidates[0])
        one_variant = shap.Explanation(
            values=values[sample_position],
            base_values=base_value,
            data=transformed[sample_position],
            feature_names=transformed_names,
        )
        plt.figure()
        shap.plots.waterfall(one_variant, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, f"shap_example_{suffix}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    mean_absolute_shap = np.abs(values).mean(axis=0)
    top_features = pd.Series(
        mean_absolute_shap, index=transformed_names
    ).sort_values(ascending=False).head(10)
    print("Top 10 features by mean absolute SHAP (pathogenic class):")
    print(top_features.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
