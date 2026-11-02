import os

import pandas as pd


input_file_path = os.path.join("data", "raw", "clinvar.vcf.gz")
output_file_path = os.path.join("data", "processed", "clinvar_mc_chr21.csv")
output_columns = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "VariationID",
    "consequence_primary",
    "consequence_all",
]

severity_order = [
    "nonsense",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "initiator_codon_variant",
    "stop_lost",
    "missense_variant",
    "splice_region_variant",
    "synonymous_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non-coding_transcript_variant",
    "intron_variant",
    "upstream_variant",
    "2KB_upstream_variant",
    "downstream_variant",
    "500B_downstream_variant",
]
severity_rank = {term: rank for rank, term in enumerate(severity_order)}


def extract_terms(mc_value):
    if not mc_value:
        return []

    terms = []
    for entry in mc_value.split(","):
        _, separator, term = entry.partition("|")
        if separator:
            terms.append(term)
    return terms


def primary_consequence(terms):
    if not terms:
        return "unknown"

    return min(
        terms,
        key=lambda term: (severity_rank.get(term, len(severity_order)), term),
    )


print("Processing ClinVar VCF in chunks...")
filtered_chunks = []

for chunk in pd.read_csv(
    input_file_path,
    sep="\t",
    comment="#",
    header=None,
    usecols=[0, 1, 2, 3, 4, 7],
    names=["CHROM", "POS", "VariationID", "REF", "ALT", "INFO"],
    dtype=str,
    chunksize=500_000,
    compression="gzip",
):
    is_chr21_snv = (
        chunk["CHROM"].eq("21")
        & chunk["REF"].str.len().eq(1)
        & chunk["ALT"].str.len().eq(1)
    )
    variants = chunk.loc[is_chr21_snv].copy()
    if variants.empty:
        continue

    mc_values = variants["INFO"].str.extract(r"(?:^|;)MC=([^;]+)", expand=False)
    consequence_terms = mc_values.fillna("").apply(extract_terms)
    variants["consequence_all"] = consequence_terms.apply(";".join)
    variants["consequence_primary"] = consequence_terms.apply(primary_consequence)
    variants["POS"] = variants["POS"].astype(int)
    filtered_chunks.append(variants[output_columns])

if filtered_chunks:
    result = pd.concat(filtered_chunks, ignore_index=True)
else:
    result = pd.DataFrame(columns=output_columns)
    result["POS"] = result["POS"].astype(int)

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
result.to_csv(output_file_path, index=False)

duplicate_count = result.duplicated(subset=["POS", "REF", "ALT"]).sum()
print(f"Output shape: {result.shape}")
print(f"Duplicate (POS, REF, ALT) rows: {duplicate_count}")
print("consequence_primary value counts:")
print(result["consequence_primary"].value_counts())
