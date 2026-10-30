# Explainable Variant Pathogenicity Framework

This project is an FYP research prototype for predicting whether a human
single-nucleotide variant on chromosome 21 (GRCh38) is pathogenic (label `1`)
or benign (label `0`). It combines ClinVar classifications, 1000 Genomes
population allele frequencies, and local sequence features from the UCSC
hg38 chromosome 21 reference. Results are for research only and are not for
clinical use.

## Data sources and expected local files

- ClinVar variant summary, filtered to GRCh38 chromosome 21 germline SNVs.
- 1000 Genomes chromosome 21 VCF and its population allele frequencies.
- UCSC hg38 chromosome 21 FASTA reference.

The pipeline expects these input files under `data\raw\`:

- `variant_summary.txt`
- `1000g_chr21.vcf.gz`
- `chr21.fa.gz`

Generated datasets are stored under `data\processed\`. The large raw sources
and `data\processed\1000g_chr21_af.csv` are excluded from Git.

## Run order

Run commands from the repository root in an activated `.venv` (or use the
explicit Windows executable shown below). The four existing scripts
`preprocess_clinvar.py`, `parse_1000g.py`, `merge_data.py`, and
`features_basic.py` are the original data-preparation steps; this order is
documented for a fresh pipeline run.

```powershell
.\.venv\Scripts\python.exe -m src.load_clinvar
.\.venv\Scripts\python.exe -m src.preprocess_clinvar
.\.venv\Scripts\python.exe -m src.parse_1000g
.\.venv\Scripts\python.exe -m src.merge_data
.\.venv\Scripts\python.exe -m src.features_basic
.\.venv\Scripts\python.exe -m src.load_reference
.\.venv\Scripts\python.exe -m src.features_reference
.\.venv\Scripts\python.exe -m src.train_baseline
.\.venv\Scripts\python.exe -m src.explain_shap
```

The ClinVar load step reads `variant_summary.txt` in chunks. Do not fetch or
reload it when the prepared ClinVar CSV files already exist. The baseline
script creates a group-isolated train/test split by `GeneSymbol`, trains
logistic-regression and random-forest models with and without population
frequency features, and writes metrics and figures to `reports\`. SHAP
explanations and the fitted random-forest model are also saved under
`reports\` and `models\`.

## Start the dashboard

After running the feature, training, and SHAP steps, start the app from the
repository root:

```powershell
.\.venv\Scripts\streamlit.exe run app/dashboard.py
```

The dashboard accepts a 1-based chromosome 21 position and REF/ALT bases,
checks the reference allele, displays the predicted pathogenic probability,
and shows a per-variant SHAP explanation with the feature values used.
The app loads the local reference FASTA, frequency CSV, and
`models\rf_full.joblib`.
