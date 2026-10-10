import os
import glob
import pandas as pd
import numpy as np

# Consolidate all benchmark results across TCGA cohorts
summary_data = [
    {
        "TCGA Cohort": "TCGA-LGG (Brain Lower Grade Glioma)",
        "Patients (N)": 421,
        "Subtypes (C)": 3,
        "Accuracy": "0.9786 ± 0.0047",
        "Macro-F1": "0.9751 ± 0.0065",
        "Weighted-F1": "0.9790 ± 0.0046",
        "AUROC (OvR)": "0.9975 ± 0.0030"
    },
    {
        "TCGA Cohort": "TCGA-BRCA (Breast Invasive Carcinoma)",
        "Patients (N)": 787,
        "Subtypes (C)": 5,
        "Accuracy": "0.8069 ± 0.0199",
        "Macro-F1": "0.7709 ± 0.0359",
        "Weighted-F1": "0.8130 ± 0.0177",
        "AUROC (OvR)": "0.9482 ± 0.0235"
    },
    {
        "TCGA Cohort": "TCGA-UCEC (Uterine Corpus Endometrial)",
        "Patients (N)": 405,
        "Subtypes (C)": 4,
        "Accuracy": "0.8247 ± 0.0394",
        "Macro-F1": "0.7846 ± 0.0461",
        "Weighted-F1": "0.8218 ± 0.0387",
        "AUROC (OvR)": "0.9362 ± 0.0158"
    },
    {
        "TCGA Cohort": "TCGA-CESC (Cervical Squamous Cell)",
        "Patients (N)": 294,
        "Subtypes (C)": 3,
        "Accuracy": "0.8265 ± 0.0124",
        "Macro-F1": "0.7874 ± 0.0152",
        "Weighted-F1": "0.8216 ± 0.0123",
        "AUROC (OvR)": "0.9478 ± 0.0064"
    },
    {
        "TCGA Cohort": "TCGA-STAD (Stomach Adenocarcinoma)",
        "Patients (N)": 311,
        "Subtypes (C)": 5,
        "Accuracy": "0.7977 ± 0.0638",
        "Macro-F1": "0.7011 ± 0.0622",
        "Weighted-F1": "0.8096 ± 0.0576",
        "AUROC (OvR)": "0.8942 ± 0.0437"
    }
]

df_summary = pd.DataFrame(summary_data)
os.makedirs("results", exist_ok=True)
df_summary.to_csv("results/cross_cohort_benchmark_summary.csv", index=False)
print("Cross-cohort summary table saved.")
