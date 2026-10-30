from collections import Counter
import gzip
import os
import pandas as pd

# Paths
fasta_path = os.path.join("data", "raw", "chr21.fa.gz")
csv_path = os.path.join("data", "processed", "features_basic.csv")

# 1. Load and parse FASTA reference genome
print("Loading reference genome from FASTA...")
if not os.path.exists(fasta_path):
  raise FileNotFoundError(
      f"'{fasta_path}' nahi mili. Pehle file ko data/raw/ mein rakhein!"
  )

seq_lines = []
with gzip.open(fasta_path, "rt") as f:
  for line in f:
    if line.startswith(">"):
      continue
    seq_lines.append(line.strip())

# Join all lines and convert to uppercase (handling lowercase repeats)
seq = "".join(seq_lines).upper()
print(f"Reference sequence length: {len(seq)}")

# Base counts (A, C, G, T, N, etc.)
print("\nBase composition:")
print(Counter(seq))

# 2. Load features dataset and check coordinate matching
print("\nChecking coordinate matching with features_basic.csv...")
df = pd.read_csv(csv_path)

matches = 0
mismatches = []

for idx, row in df.iterrows():
  pos = int(row["POS"])
  ref_vcf = row["REF"]

  # VCF is 1-based, Python index is 0-based -> seq[pos - 1]
  if 1 <= pos <= len(seq):
    ref_fasta = seq[pos - 1]
    if ref_fasta == ref_vcf:
      matches += 1
    else:
      if len(mismatches) < 5:
        mismatches.append({
            "POS": pos,
            "REF_vcf": ref_vcf,
            "REF_fasta": ref_fasta,
            "GeneSymbol": row.get("GeneSymbol", "N/A"),
        })
  else:
    # Position out of bounds
    pass

total_rows = len(df)
match_percentage = (matches / total_rows) * 100

print(f"\nTotal rows checked: {total_rows}")
print(f"Coordinate Match Percentage: {match_percentage:.4f}%")

if mismatches:
  print("\nSample Mismatches (POS, REF_vcf, REF_fasta):")
  for m in mismatches:
    print(m)
else:
  print("\nAll coordinates match perfectly with the reference genome!")