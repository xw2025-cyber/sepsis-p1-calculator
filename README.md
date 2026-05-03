# P1 Sepsis Phenotype Calculator

A Streamlit-based research calculator for early identification of the hemodynamically–metabolically decoupled sepsis phenotype.

## Background

This calculator is designed to support academic communication of a sepsis phenotyping framework based on early vasoactive-inotropic score (VIS) and serum lactate dynamics.

The P1 phenotype represents a high-risk pattern characterized by discordance between macro-circulatory support and micro-circulatory/metabolic recovery.

## Bedside enrichment rule

A patient is considered rule-positive when all three criteria are met:

1. Mean VIS 0–6 h > 20
2. Baseline lactate > 4 mmol/L
3. Lactate clearance < 10%

## Inputs

- Mean VIS 0–6 h
- Maximum VIS 0–6 h
- Baseline lactate
- Lactate at 6 h
- Minimum pH 0–6 h
- Maximum anion gap 0–6 h

## Outputs

- Lactate clearance
- Bedside rule status
- Estimated P1 likelihood
- Rule component table

## Disclaimer

This calculator is intended for academic demonstration and research communication only. It is not a medical device and should not replace clinical judgment.
