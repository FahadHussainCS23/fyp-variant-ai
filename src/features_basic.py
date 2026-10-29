import os
import pandas as pd

# 1. Load
input_path = os.path.join("data", "processed", "merged_chr21.csv")
output_path = os.path.join("data", "processed", "features_basic.csv")

df = pd.read_csv(input_path)
print(f"Loaded shape: {df.shape}")

# Ensure bases are uppercase
df["REF"] = df["REF"].str.upper()
df["ALT"] = df["ALT"].str.upper()

# Filter out rows where REF == ALT (e.g. G>G, A>A, T>T)
initial_len = len(df)
df = df[df["REF"] != df["ALT"]].reset_index(drop=True)
dropped = initial_len - len(df)
print(f"Dropped {dropped} invalid rows (REF == ALT). New shape: {df.shape}")

# 2. substitution column
df["substitution"] = df["REF"] + ">" + df["ALT"]

# 3. is_transition column
transition_set = {"A>G", "G>A", "C>T", "T>C"}
df["is_transition"] = df["substitution"].isin(transition_set).astype(int)

# 4. is_CT_or_GA column
ct_ga_set = {"C>T", "G>A"}
df["is_CT_or_GA"] = df["substitution"].isin(ct_ga_set).astype(int)

# 5. Sanity checks
print(f"Number of unique substitutions: {df['substitution'].nunique()}")
print("\n--- Substitution Value Counts ---")
print(df["substitution"].value_counts())

# 6. Save keeping specified columns
columns_to_keep = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "GeneSymbol",
    "substitution",
    "is_transition",
    "is_CT_or_GA",
    "AF_precise",
    "SAS_AF",
    "EUR_AF",
    "AFR_AF",
    "EAS_AF",
    "AMR_AF",
    "in_1000g",
    "label",
]

columns_to_keep = [c for c in columns_to_keep if c in df.columns]
df_filtered = df[columns_to_keep]

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_filtered.to_csv(output_path, index=False)
print(f"\nSaved clean features dataset to '{output_path}' with shape: {df_filtered.shape}")