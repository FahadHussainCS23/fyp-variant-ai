import os
import pandas as pd

# File paths
raw_file_path = os.path.join("data", "raw", "variant_summary.txt")
output_file_path = os.path.join("data", "processed", "clinvar_chr21.csv")

# Columns to load
cols_to_use = [
    "#AlleleID",
    "Type",
    "GeneSymbol",
    "ClinicalSignificance",
    "PhenotypeList",
    "Origin",
    "Assembly",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "ReviewStatus",
    "VariationID",
]

print("Processing ClinVar data in chunks...")

filtered_chunks = []
chunk_size = 500_000
total_rows_processed = 0

# Reading file in chunks
for chunk in pd.read_csv(
    raw_file_path, sep="\t", usecols=cols_to_use, chunksize=chunk_size, dtype=str
):
    total_rows_processed += len(chunk)

    # Filtering logic
    mask = (
        (chunk["Assembly"] == "GRCh38")
        & (chunk["Chromosome"] == "21")
        & (chunk["Type"] == "single nucleotide variant")
        & (
            chunk["Origin"]
            .str.lower()
            .str.contains("germline", na=False, regex=False)
        )
    )

    filtered_chunk = chunk[mask]
    filtered_chunks.append(filtered_chunk)
    print(
        f"Processed {total_rows_processed:,} rows... Matched so far: {sum(len(c) for c in filtered_chunks):,} rows"
    )

# Combine all chunks
df_chr21 = pd.concat(filtered_chunks, ignore_index=True)

# Rename #AlleleID column
df_chr21.rename(columns={"#AlleleID": "AlleleID"}, inplace=True)

# Save to processed folder
df_chr21.to_csv(output_file_path, index=False)

print(
    f"\nDone! Filtered dataset saved to '{output_file_path}' with {len(df_chr21):,} rows."
)