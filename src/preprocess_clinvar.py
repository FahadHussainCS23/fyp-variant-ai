import os
import pandas as pd

# File load karo
input_file = os.path.join("data", "processed", "clinvar_chr21.csv")
output_file = os.path.join("data", "processed", "clinvar_chr21_labeled.csv")

df = pd.read_csv(input_file)
print(f"Initial rows: {len(df):,}")

# 1. Review Status Filter: Remove no assertion / no classification
excluded_reviews = [
    "no assertion criteria provided",
    "no classification provided",
]
df = df[~df["ReviewStatus"].isin(excluded_reviews)]
print(f"After ReviewStatus filter: {len(df):,}")

# 2. Label Mapping (Pathogenic = 1, Benign = 0)
pathogenic_vals = [
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
]
benign_vals = ["Benign", "Likely benign", "Benign/Likely benign"]


def assign_label(sig):
  if sig in pathogenic_vals:
    return 1
  elif sig in benign_vals:
    return 0
  else:
    return None  # VUS, Conflicting, Risk factor etc.


df["label"] = df["ClinicalSignificance"].apply(assign_label)

# Drop rows where label is NaN (removes VUS, conflicting, etc.)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
print(f"After keeping only clear Pathogenic/Benign labels: {len(df):,}")

# 3. Handle Duplicates & Conflicting Labels for same genomic position/alleles
# Check duplicates based on genomic coordinates and alleles
coord_cols = ["PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]

# Group by coordinates and check unique labels per variant
label_conflicts = (
    df.groupby(coord_cols)["label"].nunique() > 1
)  # True if both 0 and 1 exist for same variant
conflicting_variants = label_conflicts[label_conflicts].index

if len(conflicting_variants) > 0:
  print(
      f"Found {len(conflicting_variants):,} variants with conflicting labels across"
      " rows. Removing them..."
  )
  # Filter out rows matching conflicting variants
  # Multi-index or filtering out
  df = df[
      ~df.set_index(coord_cols)
      .index.isin(conflicting_variants)
  ]

# Now drop exact duplicate rows
df = df.drop_duplicates(subset=coord_cols)
print(f"Final rows after duplicate & conflict removal: {len(df):,}")

# Save processed labeled dataset
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
print(f"\nSaved labeled dataset to '{output_file}'\n")

print("--- Final Label Distribution (Class Imbalance Check) ---")
print(df["label"].value_counts())
print("\nPercentages:")
print(df["label"].value_counts(normalize=True) * 100)
print(f"\nFinal Shape: {df.shape}")