import os

import pandas as pd

from .load_reference import load_chr21_sequence


def gc_content(seq, pos, w):
    """Return the GC fraction in the requested 1-based position window."""
    window = seq[max(0, pos - 1 - w):pos + w]
    denominator = sum(base in "ACGT" for base in window)
    if denominator == 0:
        return float("nan")
    return sum(base in "GC" for base in window) / denominator


def is_cpg(seq, pos, ref):
    """Return whether a C or G variant is part of a CpG dinucleotide."""
    if ref == "C" and 0 <= pos < len(seq):
        return int(seq[pos] == "G")
    if ref == "G" and 0 <= pos - 2 < len(seq):
        return int(seq[pos - 2] == "C")
    return 0


def main():
    input_path = os.path.join("data", "processed", "features_basic.csv")
    output_path = os.path.join("data", "processed", "features_ref.csv")
    df = pd.read_csv(input_path)
    seq = load_chr21_sequence()

    df["gc_50"] = df["POS"].apply(lambda pos: gc_content(seq, int(pos), 50))
    df["gc_500"] = df["POS"].apply(lambda pos: gc_content(seq, int(pos), 500))
    df["is_cpg"] = df.apply(
        lambda row: is_cpg(seq, int(row["POS"]), row["REF"]), axis=1
    )
    df["is_cpg_transition"] = (
        df["is_cpg"] & df["is_CT_or_GA"]
    ).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved reference features to '{output_path}'")
    print(f"Shape: {df.shape}")
    print("\ngc_50 describe():")
    print(df["gc_50"].describe())
    print("\ngc_500 describe():")
    print(df["gc_500"].describe())
    print("\nMean features by label:")
    print(
        df.groupby("label")[[
            "gc_50", "gc_500", "is_cpg", "is_cpg_transition"
        ]].mean()
    )
    print("\nNaN counts:")
    print(df.isna().sum())

    print("\nThree random variants with their 21-base sequence windows:")
    for _, row in df.sample(n=3, random_state=42).iterrows():
        pos = int(row["POS"])
        window = seq[max(0, pos - 11):pos + 10]
        print(
            f"POS={pos}, REF={row['REF']}, window={window}, "
            f"is_cpg={row['is_cpg']}"
        )


if __name__ == "__main__":
    main()
