import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain_shap import FEATURE_COLUMNS, pathogenic_shap_values
from src.features_reference import gc_content, is_cpg
from src.load_reference import load_chr21_sequence


AF_PATH = os.path.join("data", "processed", "1000g_chr21_af.csv")
MODEL_PATH = os.path.join("models", "rf_full.joblib")
FREQUENCY_COLUMNS = [
    "AF_precise", "SAS_AF", "EUR_AF", "AFR_AF", "EAS_AF", "AMR_AF",
]


@st.cache_resource
def cached_sequence():
    return load_chr21_sequence()


@st.cache_resource
def cached_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_population_frequencies():
    columns = ["CHROM", "POS", "REF", "ALT", *FREQUENCY_COLUMNS]
    frequencies = pd.read_csv(AF_PATH, usecols=columns)
    frequencies = frequencies[frequencies["CHROM"].astype(str) == "21"]
    return frequencies.set_index(["POS", "REF", "ALT"])


def build_variant_features(seq, pos, ref, alt, frequencies):
    substitution = f"{ref}>{alt}"
    is_transition = int(
        substitution in {"A>G", "G>A", "C>T", "T>C"}
    )
    is_ct_or_ga = int(substitution in {"C>T", "G>A"})
    cpg = is_cpg(seq, pos, ref)

    feature_values = {
        "AF_precise": 0.0,
        "SAS_AF": 0.0,
        "EUR_AF": 0.0,
        "AFR_AF": 0.0,
        "EAS_AF": 0.0,
        "AMR_AF": 0.0,
        "in_1000g": 0,
        "substitution": substitution,
        "is_transition": is_transition,
        "is_CT_or_GA": is_ct_or_ga,
        "gc_50": gc_content(seq, pos, 50),
        "gc_500": gc_content(seq, pos, 500),
        "is_cpg": cpg,
        "is_cpg_transition": int(cpg and is_ct_or_ga),
    }

    key = (pos, ref, alt)
    if key in frequencies.index:
        frequency = frequencies.loc[key]
        if isinstance(frequency, pd.DataFrame):
            frequency = frequency.iloc[0]
        for column in FREQUENCY_COLUMNS:
            feature_values[column] = float(frequency[column])
        feature_values["in_1000g"] = 1
    return feature_values


def explain_variant(pipeline, feature_values):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_frame = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    transformed = preprocessor.transform(feature_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed)
    values = pathogenic_shap_values(raw_values, len(feature_names))[0]
    contributions = pd.Series(values, index=feature_names)
    top_contributions = contributions.loc[
        contributions.abs().sort_values(ascending=False).head(15).index
    ].sort_values()

    figure, axis = plt.subplots(figsize=(8, 5))
    colors = ["#c44e52" if value > 0 else "#4c72b0"
              for value in top_contributions]
    axis.barh(top_contributions.index, top_contributions.values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("SHAP contribution to pathogenic class")
    axis.set_title("Feature contributions for this variant")
    figure.tight_layout()
    return figure


def main():
    st.set_page_config(page_title="Chr21 Variant Predictor", layout="wide")
    st.title("Chromosome 21 Variant Pathogenicity Predictor")
    st.write("Chromosome: 21")
    st.caption("Research prototype for an FYP, not for clinical use.")

    seq = cached_sequence()
    frequencies = load_population_frequencies()
    with st.form("variant_form"):
        pos = st.number_input(
            "Position (1-based)", value=1, step=1, format="%d"
        )
        ref = st.selectbox("REF", ["A", "C", "G", "T"])
        alt = st.selectbox("ALT", ["A", "C", "G", "T"], index=1)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return
    pos = int(pos)
    if ref == alt:
        st.error("REF and ALT must differ.")
        return
    if pos < 1 or pos > len(seq):
        st.error(f"Position must be between 1 and {len(seq)}.")
        return
    actual_base = seq[pos - 1]
    if ref != actual_base:
        st.error(
            f"REF does not match the reference genome at position {pos}. "
            f"The reference base is {actual_base}."
        )
        return

    feature_values = build_variant_features(
        seq, pos, ref, alt, frequencies
    )
    pipeline = cached_model()
    feature_frame = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(feature_frame)[0, 1])
    prediction = "Pathogenic" if probability >= 0.5 else "Benign"

    st.metric("Predicted probability of pathogenicity", f"{probability:.1%}")
    st.subheader(prediction)
    figure = explain_variant(pipeline, feature_values)
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("Feature values used")
    st.dataframe(
        pd.DataFrame([feature_values], columns=FEATURE_COLUMNS),
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
