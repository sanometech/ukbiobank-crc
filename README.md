# Additive pre-diagnostic and diagnostic value of routine blood-based biomarkers in the detection of colorectal cancer in the UK Biobank cohort

by Gizem Tanriver, Ece Kocagoncu

This repository is the official implementation of the [paper](link), which has been submitted for preprint in medRxiv and publication in British Journal of Cancer.

## Abstract

## Implementation

All source code used to generate the results and figures in the paper can be found in the respective folders.

- ```preprocessing``` folder contains codes used to preprocess and featurise the raw dataset
- ```eda``` folder contains codes used to calculate descriptive statistics using baseline data
- ```cox``` folder contains the codes for cox regression model
- ```gpboost``` folder contains the codes for gpboost model with feature selection using RFE

The models were run inside Jupyter notebooks.

## Requirements

- Python 3.7
- Lifelines 0.27.1
- GPBoost 0.7.9
- PDPbox 0.2.1

## Data availability

Approval for the study and permission to access the data was granted by the UK Biobank Resource. UK Biobank is an open access resource and bona fide researchers can access the UK Biobank dataset by registering and applying at  <https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access>. This research has been conducted using the UK Biobank Resource under application number 87991 for the project titled ‘Validation of an AI-powered online search strategy for finding optimal biomarker combinations’.

## License

Distributed under GNU General Public License v3.0. See LICENSE for more information.

## Citation

If this work is helpful, please cite as:

```

```
