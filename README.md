# P1 Sepsis Phenotype ML Calculator

A Streamlit-based machine-learning research calculator for early identification of the hemodynamically–metabolically decoupled sepsis phenotype (P1).

## Model

The deployed model uses early 0–6 h VIS–lactate dynamic features and was trained in the MIMIC-IV derivation cohort. The model was externally evaluated in eICU.

## Features

- Mean VIS 0–6 h
- Maximum VIS 0–6 h
- Baseline lactate
- Mean lactate 0–6 h
- Lactate clearance 0–6 h

## Outputs

- ML-predicted P1 probability
- ML classification using the MIMIC-derived threshold
- Bedside rule status
- Rule component table

## Disclaimer

This calculator is intended for academic demonstration and research communication only. It is not a medical device and should not replace clinical judgment.
