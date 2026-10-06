# LeakageSafe-ClinicalML

**Reproducible Leakage-Safe Evaluation of Structured Clinical Machine Learning**

`LeakageSafe-ClinicalML` is a research-software repository supporting reproducible evaluation of machine-learning methods for structured clinical data. The repository emphasizes leakage-safe experimental design, repeated-holdout evaluation, comparison of training strategies, statistical characterization of performance stability, and subgroup fairness assessment.

The repository accompanies a research study developed around a central reproducibility principle: **all reported conclusions should be traceable to executable code and preserved experimental evidence**.

The public release is intended to provide the code, configurations, machine-readable results, figures, and documentation necessary to understand and reproduce the analyses reported in the associated manuscript.

---

## Overview

Machine-learning performance on relatively small structured clinical datasets can be strongly influenced by data leakage, preprocessing decisions, train/test partitioning, class imbalance, and variability across data splits.

This project provides an evaluation framework designed to address these issues explicitly.

The repository focuses on:

- leakage-safe preprocessing and evaluation;
- repeated stratified holdout experiments;
- comparison of training strategies;
- preservation of test-set independence;
- machine-readable experimental outputs;
- performance stability across repeated splits;
- subgroup fairness assessment;
- statistical characterization of repeated results;
- transparent provenance between code, experiments, results, and manuscript claims; and
- reproducible research-software practices.

The repository intentionally distinguishes between **predictive performance**, **stability across repeated holdouts**, **subgroup fairness diagnostics**, and **formal statistical inference**.

---

## Research Scope

The validated project concerns **structured/tabular clinical machine learning**.

The current reproducibility release is based on a clinical dataset containing **193 participants** and uses repeated leakage-safe evaluation to characterize model performance under different training strategies.

The principal validated comparison includes:

1. **Unbalanced training**, where the original class distribution is retained in the training data.

2. **Training-only oversampling**, where class balancing is applied only to the training partition after the train/test split.

The test partition remains untouched by the oversampling procedure.

This distinction is essential because performing oversampling or related data-dependent operations before partitioning can introduce information leakage and produce overly optimistic performance estimates.

---

## Key Design Principles

### 1. Leakage-Safe Evaluation

All data-dependent preprocessing and training operations should be learned or performed using the training partition only.

The held-out test data must not influence:

- oversampling;
- feature transformation parameters;
- feature selection;
- model fitting;
- hyperparameter-dependent training operations; or
- other procedures capable of transferring information from the test set into the training process.

The repository includes evidence and audit materials used to verify the evaluation chronology.

### 2. Repeated-Holdout Evaluation

A single train/test partition may provide an unstable estimate of performance, particularly for relatively small clinical datasets.

The validated analysis therefore uses repeated holdout evaluation across multiple random seeds.

The preserved evidence includes **10 repeated holdouts**.

These repetitions are used primarily to characterize the **stability and variability of predictive performance across different data partitions**.

They should not be interpreted as ten independent clinical cohorts because the same underlying participant pool is reused across the repeated splits.

### 3. Training-Only Class Balancing

Where oversampling is evaluated, it is applied **after splitting the data and only to the training subset**.

The test set retains its natural distribution and is not synthetically balanced.

This design prevents synthetic or duplicated training observations from contaminating the independent test partition.

### 4. Reproducible Evidence

Experimental conclusions are preserved as machine-readable outputs whenever possible.

The repository is structured so that a result reported in the manuscript can be traced through:

**manuscript claim → numerical result → experiment → configuration → implementation → input data/provenance**

---

# Validated Experimental Findings

The authoritative repeated-holdout evidence currently supports the following summary.

## Unbalanced Training

Across the validated repeated holdouts:

- **Accuracy:** 0.9821 ± 0.0211
- **F1-score:** 0.9816 ± 0.0217
- **ROC-AUC:** 0.9987 ± 0.0033

## Training-Only Oversampling

Across the same evaluation framework:

- **Accuracy:** 0.9769 ± 0.0225
- **F1-score:** 0.9761 ± 0.0235
- **ROC-AUC:** 0.9984 ± 0.0033

These results indicate strong predictive performance under both evaluated training strategies.

Importantly, the evidence does **not** establish that training-only oversampling provides a statistically significant improvement over unbalanced training.

For the preserved paired comparisons:

| Metric | Paired t-test p-value | Wilcoxon p-value |
|---|---:|---:|
| Accuracy | 0.4433 | 0.4142 |
| F1-score | 0.4476 | 0.4142 |
| ROC-AUC | 0.3434 | 0.3173 |

Accordingly, the repository does not claim superiority of oversampling based on these results.

---

# Interpretation of Repeated Evaluations

The repeated-holdout results provide useful information about performance stability under different train/test partitions.

However, the repeated runs reuse observations from the same cohort of 193 participants.

They therefore **do not constitute independent external replications**.

The repeated results should primarily be interpreted as:

- estimates of split-to-split variability;
- evidence of performance stability;
- sensitivity analysis with respect to random partitioning; and
- paired comparisons of training strategies under matched experimental conditions.

They should not be interpreted as evidence from ten independent patient cohorts.

---

# Subgroup Fairness Analysis

The project also includes subgroup fairness diagnostics.

The preserved aggregated evidence includes measures such as:

- Statistical Parity Difference (SPD);
- Equal Opportunity Difference (EOD); and
- Disparate Impact (DI).

Examples from the validated aggregated evidence include:

| Subgroup comparison | SPD | EOD | DI |
|---|---:|---:|---:|
| Gender \| M | −0.103 ± 0.252 | 0.027 ± 0.044 | 0.886 ± 0.568 |
| Nationality \| S | 0.043 ± 0.109 | 0.032 ± 0.052 | 1.106 ± 0.222 |

These values are provided as **subgroup diagnostic evidence**.

They should not be interpreted as proof that the model is universally fair, unbiased, or clinically equitable.

Fairness estimates obtained from relatively small subgroups may exhibit substantial uncertainty and should be interpreted in the context of subgroup size, outcome prevalence, sampling variability, and the clinical setting.

---

# Statistical Interpretation

Statistical analyses are included to characterize differences between matched experimental conditions.

Because repeated holdouts reuse participants from the same original dataset, the resulting observations are not equivalent to measurements obtained from independent clinical cohorts.

Consequently, statistical testing is interpreted conservatively.

The primary purpose of the repeated experiments is to evaluate:

- robustness to data partitioning;
- stability of predictive metrics;
- direction and magnitude of paired differences; and
- consistency of findings across repeated evaluation conditions.

The repository avoids interpreting non-independent repeated splits as independent external replications.

---

# Repository Structure

The intended public repository structure is:

```text
LeakageSafe-ClinicalML/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   ├── fairness/
│   └── utils/
│
├── experiments/
│   ├── leakage_safe_evaluation/
│   ├── repeated_holdout/
│   ├── baseline_comparison/
│   ├── fairness_analysis/
│   └── statistical_analysis/
│
├── configs/
│
├── data/
│   └── README.md
│
├── results/
│   ├── primary/
│   ├── repeated_holdout/
│   ├── baselines/
│   ├── fairness/
│   └── statistics/
│
├── figures/
│
├── models/
│
├── reproducibility/
│   ├── experiment_manifest.csv
│   ├── result_provenance.csv
│   ├── seeds.csv
│   └── environment_info.txt
│
└── docs/
    ├── DATA_DESCRIPTION.md
    ├── REPRODUCIBILITY.md
    └── RESULTS.md
```

The exact public release may contain only the files required to reproduce and verify the results reported in the associated manuscript.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<USERNAME>/LeakageSafe-ClinicalML.git
cd LeakageSafe-ClinicalML
```

Replace `<USERNAME>` with the repository owner's GitHub username or organization.

---

## 2. Create a Python Environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If the final release includes `environment.yml`, the environment may alternatively be created with Conda:

```bash
conda env create -f environment.yml
conda activate leakagesafe-clinicalml
```

Exact dependency versions should be taken from the files distributed with the corresponding tagged release.

---

# Data

## Clinical Data

The project was developed using structured clinical participant-level data.

Clinical data require special consideration because redistribution may be restricted by:

- participant consent;
- institutional policies;
- ethics approvals;
- privacy requirements;
- data-use agreements; and
- applicable laws and regulations.

The presence of source data in the original research environment does **not** automatically imply permission for public redistribution.

Therefore, only data that are explicitly approved for public release should be placed in the GitHub repository or DOI archive.

The `data/README.md` file should document:

- dataset origin;
- participant/sample count;
- variables used;
- target variable;
- preprocessing requirements;
- inclusion/exclusion criteria where applicable;
- access conditions;
- redistribution restrictions; and
- instructions for obtaining the data when public redistribution is not permitted.

No sensitive or personally identifying participant information should be committed to the repository.

---

# Reproducing the Analysis

The public release is intended to separate the reproducibility workflow into distinct stages.

## Stage 1 — Data Preparation

The preprocessing implementation should:

1. load the permitted structured dataset;
2. perform the defined data-cleaning procedures;
3. construct the target and predictor variables;
4. create the required train/test partition;
5. fit data-dependent transformations using training data only; and
6. preserve the held-out test data for independent evaluation.

---

## Stage 2 — Leakage-Safe Training

For each random seed:

1. construct the train/test split;
2. preserve the test set;
3. apply any class-balancing procedure only to the training data;
4. train the model using the training partition;
5. generate predictions for the untouched test partition; and
6. save the resulting performance metrics and predictions.

---

## Stage 3 — Repeated-Holdout Evaluation

The evaluation pipeline repeats the procedure across the predefined random seeds.

The resulting outputs are aggregated to characterize:

- mean predictive performance;
- standard deviation;
- split-to-split variation;
- paired differences between training strategies; and
- consistency of the conclusions.

---

## Stage 4 — Fairness Analysis

Fairness analysis uses preserved predictions and subgroup information to calculate the supported fairness metrics.

The fairness analysis should be performed on evaluation predictions rather than training performance.

Results should be interpreted as subgroup diagnostics rather than universal guarantees of fairness.

---

## Stage 5 — Statistical Analysis

Matched results from the repeated evaluation are used to characterize differences between training strategies.

Where inferential tests are reported, their interpretation must account for the dependence induced by repeatedly partitioning the same underlying participant cohort.

---

# Results and Provenance

Machine-readable result files are retained wherever possible.

The `results/` directory should contain the numerical evidence underlying the manuscript's tables and figures.

The `reproducibility/` directory provides additional provenance information linking experiments to their corresponding:

- scripts;
- configurations;
- random seeds;
- input data;
- result files;
- figures; and
- manuscript outputs.

This structure is intended to prevent results from becoming disconnected from the computational procedures that generated them.

---

# Models

Only model artifacts that are necessary for reproducing or verifying the published analyses should be included in the DOI release.

Saved models should be accompanied by sufficient information to identify:

- model type;
- training configuration;
- corresponding experiment;
- random seed where relevant;
- preprocessing assumptions; and
- compatible software environment.

Large or unnecessary intermediate checkpoints should not be included solely for archival completeness.

---

# Figures

Publication figures included in this repository should be traceable to machine-readable result files.

Whenever possible, figures should be regenerable from the corresponding results rather than being distributed only as static images.

---

# Reproducibility

The repository is designed around four reproducibility requirements:

### Computational reproducibility

The code and software environment should permit the reported analysis to be reconstructed.

### Experimental reproducibility

Random seeds, configurations, preprocessing order, training procedures, and evaluation protocols should be documented.

### Result traceability

Reported numerical findings should be traceable to preserved machine-readable experimental outputs.

### Claim traceability

Scientific claims should not exceed what is supported by the preserved evidence.

---

# Scope of Supported Claims

This repository supports claims concerning:

- structured clinical machine learning;
- leakage-safe evaluation;
- repeated-holdout performance;
- stability across repeated data partitions;
- comparison of the validated training strategies;
- subgroup fairness diagnostics; and
- statistical characterization of the preserved repeated results.

The current evidence package should **not** be interpreted as establishing:

- independent external clinical validation;
- cross-dataset generalization;
- multimodal clinical validation;
- synthetic-data fidelity;
- generative-model superiority;
- reproducible generative ablation results;
- empirical privacy guarantees;
- universal algorithmic fairness;
- Pareto-optimality; or
- clinical deployment readiness.

These distinctions are intentional and reflect the evidence available in the reproducibility package.

---

# External Validation

The current validated evidence does not include a dedicated independent external or cross-dataset validation experiment.

Therefore, performance reported in this repository should be interpreted as internal repeated-holdout performance on the available clinical cohort.

External clinical validation remains outside the scope of this release.

---

# Clinical Use Disclaimer

This repository is provided for **research and reproducibility purposes**.

The software and models are not medical devices and have not been established here as suitable for autonomous diagnosis, treatment selection, clinical decision-making, or deployment in routine patient care.

Clinical use would require additional validation, regulatory consideration, prospective evaluation, and assessment in the intended clinical population and setting.

---

# Limitations

Important limitations include:

1. The available cohort contains 193 participants.

2. Repeated holdouts reuse participants and therefore do not represent independent external cohorts.

3. Strong internal predictive performance does not establish external clinical generalizability.

4. Subgroup fairness estimates may be unstable when subgroup sample sizes are limited.

5. The comparison of training strategies does not establish a statistically supported superiority of oversampling based on the preserved repeated evaluations.

6. No dedicated external-validation experiment is included in the validated evidence package.

7. Results should therefore be interpreted within the population, variables, preprocessing procedures, and evaluation protocol represented by the available dataset.

---

# Citation

If you use this repository, its code, or its archived experimental materials, please cite the corresponding DOI release.

GitHub citation metadata are provided in:

```text
CITATION.cff
```

The permanent DOI citation will be added after archival of the corresponding release.

A Zenodo archive is intended to provide a persistent, version-specific research-software record.

---

# Versioning

The repository follows semantic versioning where practical.

The initial DOI-backed reproducibility release is intended to be:

```text
v1.0.0
```

Future changes should be released as new tagged versions rather than silently modifying the archived research record.

A DOI assigned to a specific release should always refer to the exact archived version associated with that publication.

---

## DOI

The software is permanently archived on Zenodo.

**Version:** v1.0.0  
**DOI:** 10.5281/zenodo.23195503

[![DOI](https://zenodo.org/badge/1407620848.svg)](https://doi.org/10.5281/zenodo.23195503)

# License

The software is distributed under the **MIT License**.

```text
Copyright (c) 2026 Mahmoud Rokaya
```

See the [`LICENSE`](LICENSE) file for the complete license text.

The MIT License applies to the software unless explicitly stated otherwise.

It does **not** automatically grant rights to redistribute third-party or participant-level clinical datasets. Dataset licensing and access conditions must be considered separately.

---

# Research Transparency

This repository follows a conservative evidence policy:

- corrected evidence takes precedence over historical results;
- leakage-safe evaluation takes precedence over earlier potentially contaminated estimates;
- repeated results are reported with their variability;
- unsupported claims are not reconstructed from indirect evidence;
- audit evidence is not treated as experimental evidence when an experiment itself could not be reproduced; and
- absence of external validation is reported explicitly rather than inferred from internal performance.

This policy is intended to make the public repository consistent with the evidence underlying the associated manuscript.

---

# Authors

**Mahmoud Rokaya**

Additional authors and contributors associated with the accompanying manuscript should be listed in `CITATION.cff` according to their verified contributions and the authorship of the corresponding research work.

---

# Funding

Funding information for the associated study should be reported exactly as approved by the funding institution and as stated in the accompanying manuscript.

Where applicable:

**Deanship of Graduate Studies and Scientific Research, Taif University**

The final Zenodo metadata and `.zenodo.json` file should contain the same verified funding information.

---

# Contact

Questions concerning the research software, reproducibility materials, or associated publication should be submitted through the repository's GitHub issue tracker or directed to the corresponding author using the contact information provided in the associated publication.

---

## Release Status

**Repository:** LeakageSafe-ClinicalML  
**Planned DOI release:** v1.0.0  
**License:** MIT  
**Research domain:** Structured clinical machine learning  
**Evaluation:** Leakage-safe repeated holdout  
**Reproducibility status:** Code and experimental evidence release  
**External validation:** Not included in the current validated evidence
