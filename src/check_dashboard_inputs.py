import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.dashboard import (
    FEATURE_COLUMNS,
    build_variant_features,
    cached_model,
    cached_sequence,
    load_consequence_annotations,
    load_consequence_mapping,
    load_population_frequencies,
    lookup_consequence,
)


def choose_variant(df, label, consequence, used_positions):
    candidates = df.loc[
        (df["label"] == label)
        & (df["consequence_primary"] == consequence)
    ].sort_values("POS")
    for _, row in candidates.iterrows():
        pos = int(row["POS"])
        if pos not in used_positions:
            used_positions.add(pos)
            return row
    raise ValueError(
        f"No distinct-position variant found for label={label}, "
        f"consequence={consequence}."
    )


def main():
    df = pd.read_csv("data/processed/features_ref_mc.csv")
    sequence = cached_sequence()
    frequencies = load_population_frequencies()
    annotations = load_consequence_annotations()
    consequence_mapping = load_consequence_mapping()
    pipeline = cached_model()

    cases = [
        ("pathogenic missense", 1, "missense_variant"),
        ("benign missense", 0, "missense_variant"),
        ("nonsense", None, "nonsense"),
        ("synonymous", None, "synonymous_variant"),
    ]
    used_positions = set()

    print(
        "Case                 POS       REF ALT Label Consequence group "
        "Probability"
    )
    print("-" * 91)
    for case_name, label, consequence in cases:
        if label is None:
            candidates = df.loc[
                df["consequence_primary"] == consequence
            ].sort_values("POS")
            row = None
            for _, candidate in candidates.iterrows():
                candidate_pos = int(candidate["POS"])
                if candidate_pos not in used_positions:
                    used_positions.add(candidate_pos)
                    row = candidate
                    break
            if row is None:
                raise ValueError(
                    f"No distinct-position variant found for "
                    f"consequence={consequence}."
                )
        else:
            row = choose_variant(df, label, consequence, used_positions)

        pos = int(row["POS"])
        ref = str(row["REF"])
        alt = str(row["ALT"])
        primary, group, found = lookup_consequence(
            pos,
            ref,
            alt,
            annotations,
            consequence_mapping,
        )
        if not found or primary != consequence:
            raise ValueError(
                f"ClinVar lookup disagrees with training row at "
                f"{pos} {ref}>{alt}: {primary!r}."
            )
        feature_values = build_variant_features(
            sequence,
            pos,
            ref,
            alt,
            frequencies,
            primary,
            group,
        )
        feature_frame = pd.DataFrame(
            [feature_values],
            columns=FEATURE_COLUMNS,
        )
        probability = float(pipeline.predict_proba(feature_frame)[0, 1])
        print(
            f"{case_name:<21}{pos:<10}{ref:<4}{alt:<4}"
            f"{int(row['label']):<6}{group:<21}{probability:.6f}"
        )
        print(f"  POS/REF/ALT: {pos} / {ref} / {alt}")


if __name__ == "__main__":
    main()
