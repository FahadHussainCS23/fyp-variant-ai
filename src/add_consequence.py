import os

import pandas as pd


features_path = os.path.join("data", "processed", "features_ref.csv")
consequences_path = os.path.join("data", "processed", "clinvar_mc_chr21.csv")
output_path = os.path.join("data", "processed", "features_ref_mc.csv")
expected_rows = 17_935
consequence_columns = ["consequence_primary", "consequence_all"]
lof_consequences = {
    "nonsense",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "initiator_codon_variant",
    "stop_lost",
}

features = pd.read_csv(features_path)
consequences = pd.read_csv(
    consequences_path,
    usecols=["POS", "REF", "ALT", *consequence_columns],
)

merged = features.merge(
    consequences,
    on=["POS", "REF", "ALT"],
    how="left",
)

if len(merged) != expected_rows:
    print(
        f"ERROR: Expected {expected_rows:,} rows after merging, "
        f"but got {len(merged):,}. Stopping without saving output."
    )
    raise SystemExit(1)

merged["consequence_primary"] = merged["consequence_primary"].fillna("unknown")

primary_counts = merged["consequence_primary"].value_counts()
rare_consequences = primary_counts[primary_counts < 30].index.tolist()
print(f"Categories mapped to 'other' (fewer than 30 rows): {rare_consequences}")

merged["consequence_group"] = merged["consequence_primary"].where(
    ~merged["consequence_primary"].isin(rare_consequences),
    "other",
)
merged["is_lof"] = merged["consequence_primary"].isin(lof_consequences).astype(int)
merged["is_missense"] = (
    merged["consequence_primary"].eq("missense_variant").astype(int)
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged.to_csv(output_path, index=False)

print(f"Final shape: {merged.shape}")
print(
    "Rows with a real consequence: "
    f"{merged['consequence_primary'].ne('unknown').sum():,}"
)
print("\nconsequence_group value counts:")
print(merged["consequence_group"].value_counts())

label_counts = (
    merged.groupby(["consequence_group", "label"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "benign_count", 1: "pathogenic_count"})
)
for column in ["benign_count", "pathogenic_count"]:
    if column not in label_counts:
        label_counts[column] = 0

group_totals = merged.groupby("consequence_group").size()
label_counts["pathogenic_percentage"] = (
    label_counts["pathogenic_count"] / group_totals * 100
)
label_counts = label_counts[
    ["benign_count", "pathogenic_count", "pathogenic_percentage"]
]
print("\nconsequence_group by label:")
print(label_counts.to_string())
