from collections import Counter
import gzip
import os

import pandas as pd


def load_chr21_sequence():
    """Load the uppercase chr21 reference sequence from the compressed FASTA."""
    fasta_path = os.path.join("data", "raw", "chr21.fa.gz")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"Reference FASTA not found at '{fasta_path}'."
        )

    with gzip.open(fasta_path, "rt") as fasta_file:
        sequence = "".join(
            line.strip()
            for line in fasta_file
            if not line.startswith(">")
        )
    return sequence.upper()


if __name__ == "__main__":
    csv_path = os.path.join("data", "processed", "features_basic.csv")

    print("Loading reference genome from FASTA...")
    seq = load_chr21_sequence()
    print(f"Reference sequence length: {len(seq)}")
    print("\nBase composition:")
    print(Counter(seq))

    print("\nChecking coordinate matching with features_basic.csv...")
    df = pd.read_csv(csv_path)

    matches = 0
    mismatches = []
    for _, row in df.iterrows():
        pos = int(row["POS"])
        ref_vcf = row["REF"]
        if 1 <= pos <= len(seq):
            ref_fasta = seq[pos - 1]
            if ref_fasta == ref_vcf:
                matches += 1
            elif len(mismatches) < 5:
                mismatches.append({
                    "POS": pos,
                    "REF_vcf": ref_vcf,
                    "REF_fasta": ref_fasta,
                    "GeneSymbol": row.get("GeneSymbol", "N/A"),
                })

    total_rows = len(df)
    match_percentage = (matches / total_rows) * 100
    print(f"\nTotal rows checked: {total_rows}")
    print(f"Coordinate Match Percentage: {match_percentage:.4f}%")

    if mismatches:
        print("\nSample Mismatches (POS, REF_vcf, REF_fasta):")
        for mismatch in mismatches:
            print(mismatch)
    else:
        print("\nAll coordinates match perfectly with the reference genome!")
