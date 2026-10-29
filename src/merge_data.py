import os
import pandas as pd

# Files load karo
clinvar_path = os.path.join("data", "processed", "clinvar_chr21_labeled.csv")
g1000_path = os.path.join("data", "processed", "1000g_chr21_af.csv")
output_path = os.path.join("data", "processed", "merged_chr21.csv")

print("Loading labeled ClinVar and 1000G AF datasets...")
df_clinvar = pd.read_csv(clinvar_path)
df_1000g = pd.read_csv(g1000_path)

# Ensure coordinate columns are of integer type for precise matching
df_clinvar["PositionVCF"] = df_clinvar["PositionVCF"].astype(int)
df_1000g["POS"] = df_1000g["POS"].astype(int)

print(f"ClinVar initial shape: {df_clinvar.shape}")
print(f"1000G initial shape: {df_1000g.shape}")

# Left merge: ClinVar as left table, keeping all 17,941 rows
df_merged = pd.merge(
    df_clinvar,
    df_1000g,
    left_on=["PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"],
    right_on=["POS", "REF", "ALT"],
    how="left",
)

# Create in_1000g flag (1 if AF_precise is not null, else 0)
df_merged["in_1000g"] = df_merged["AF_precise"].notna().astype(int)

# Fill missing AF_precise for variants not found in 1000G with 0.0
df_merged["AF_precise"] = df_merged["AF_precise"].fillna(0.0)

# Also fill missing regional frequencies with 0.0 if needed
regional_cols = ["SAS_AF", "EUR_AF", "AFR_AF", "EAS_AF", "AMR_AF"]
for col in regional_cols:
  if col in df_merged.columns:
    df_merged[col] = df_merged[col].fillna(0.0)

# Drop redundant columns from 1000G side if any (like duplicate POS, REF, ALT)
drop_cols = ["POS", "REF", "ALT"]
df_merged = df_merged.drop(columns=[col for col in drop_cols if col in df_merged.columns])

# Save merged dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print(f"\nSaved merged dataset to '{output_path}'")
print(f"Final Merged Shape: {df_merged.shape}\n")

print("--- in_1000g Value Counts ---")
print(df_merged["in_1000g"].value_counts())

print("\n--- Proportion of variants found in 1000G by Label (0 = Benign, 1 = Pathogenic) ---")
print(df_merged.groupby("label")["in_1000g"].mean())

print("\n--- AF_precise Summary Describe by Label ---")
print(df_merged.groupby("label")["AF_precise"].describe())