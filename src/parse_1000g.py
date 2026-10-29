import os
import re
import pandas as pd
import gzip

vcf_path = os.path.join("data", "raw", "1000g_chr21.vcf.gz")
output_path = os.path.join("data", "processed", "1000g_chr21_af.csv")

print("Reading 1000 Genomes VCF file safely with precise AF calculation...")

data = []
sample_info_logged = False

with gzip.open(vcf_path, "rt", encoding="utf-8", errors="ignore") as f:
  for line in f:
    if line.startswith("#"):
      continue

    parts = line.strip().split("\t")
    if len(parts) < 8:
      continue

    chrom, pos, _, ref, alt, _, _, info = parts[:8]

    # Keep only SNVs (REF and ALT length == 1)
    if len(ref) == 1 and len(alt) == 1:
      if not sample_info_logged:
        print("\n--- Sample INFO Row ---")
        print(info)
        print("------------------------\n")
        sample_info_logged = True

      data.append(
          {"CHROM": chrom.replace("chr", ""), "POS": pos, "REF": ref, "ALT": alt, "INFO": info}
      )

print(f"Total SNVs extracted: {len(data):,}")
df = pd.DataFrame(data)


# Helper function to extract numerical values from INFO field
def extract_value(info_str, key, is_float=False):
  match = re.search(r"(?:^|;)" + key + r"=([^;]+)", info_str)
  if match:
    try:
      val = match.group(1)
      # In case of comma-separated multi-allelic AC/AF, take the first one
      if "," in val:
        val = val.split(",")[0]
      return float(val) if is_float else int(float(val))
    except ValueError:
      return None
  return None


# Extract AC, AN, and other population frequencies
print("Extracting AC, AN, and population allele frequencies...")
df["AC"] = df["INFO"].apply(lambda x: extract_value(x, "AC", is_float=False))
df["AN"] = df["INFO"].apply(lambda x: extract_value(x, "AN", is_float=False))

# Calculate precise AF: AC / AN
df["AF_precise"] = df["AC"] / df["AN"]

# Extract regional allele frequencies
af_keys = ["SAS_AF", "EUR_AF", "AFR_AF", "EAS_AF", "AMR_AF"]
for key in af_keys:
  df[key] = df["INFO"].apply(lambda x: extract_value(x, key, is_float=True))

# Clean and select essential columns
output_cols = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "AF_precise",
    "SAS_AF",
    "EUR_AF",
    "AFR_AF",
    "EAS_AF",
    "AMR_AF",
]
df_clean = df[output_cols].copy()

# Drop rows where AF_precise could not be calculated (NaN)
df_clean = df_clean.dropna(subset=["AF_precise"])

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_clean.to_csv(output_path, index=False)

print(f"\nSaved precise 1000G AF data to '{output_path}'")
print(f"Final shape: {df_clean.shape}\n")

print("--- First row AF_precise check ---")
print(f"AF_precise for first variant: {df_clean['AF_precise'].iloc[0]:.6f}")

print("\n--- df.head() ---")
print(df_clean.head())

print("\n--- AF_precise Summary (describe) ---")
print(df_clean["AF_precise"].describe())