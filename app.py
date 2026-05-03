import streamlit as st
import numpy as np
import pandas as pd
import math

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="P1 Sepsis Phenotype Calculator",
    page_icon="🧬",
    layout="centered"
)

# =========================================================
# Helper functions
# =========================================================
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def lactate_clearance_pct(baseline_lactate: float, lactate_6h: float) -> float:
    if baseline_lactate <= 0:
        return 0.0
    return (baseline_lactate - lactate_6h) / baseline_lactate * 100


def bedside_rule(mean_vis_0_6h: float, baseline_lactate: float, lactate_clearance: float) -> int:
    """
    Early bedside enrichment rule for the P1 decoupled phenotype.

    Rule-positive:
    - Mean VIS 0–6 h > 20
    - Baseline lactate > 4 mmol/L
    - Lactate clearance < 10%
    """
    return int(
        (mean_vis_0_6h > 20)
        and (baseline_lactate > 4)
        and (lactate_clearance < 10)
    )


def p1_probability_model(
    mean_vis_0_6h: float,
    max_vis_0_6h: float,
    baseline_lactate: float,
    lactate_6h: float,
    lactate_clearance: float,
    ph_min_0_6h: float,
    anion_gap_max_0_6h: float,
):
    """
    Demonstration risk score for early P1 phenotype enrichment.

    This is a research demonstration model designed to reflect the manuscript
    concept: high vasoactive support with persistent hyperlactatemia suggests
    hemodynamic-metabolic decoupling.

    The rule-based classification remains the primary interpretable tool.
    """
    lp = (
        -5.20
        + 0.035 * mean_vis_0_6h
        + 0.010 * max_vis_0_6h
        + 0.42 * baseline_lactate
        + 0.30 * lactate_6h
        - 0.035 * lactate_clearance
        - 1.20 * (ph_min_0_6h - 7.35)
        + 0.055 * anion_gap_max_0_6h
    )
    return sigmoid(lp)


def category_from_probability(prob: float) -> str:
    if prob < 0.05:
        return "Low likelihood"
    elif prob < 0.15:
        return "Intermediate likelihood"
    elif prob < 0.30:
        return "High likelihood"
    else:
        return "Very high likelihood"


# =========================================================
# Header
# =========================================================
st.title("P1 Sepsis Phenotype Calculator")

st.markdown(
    """
This research calculator supports early identification of the
**hemodynamically–metabolically decoupled sepsis phenotype (P1)** using early
VIS–lactate dynamics.

The core bedside enrichment rule is based on three clinically interpretable
features within the first 6 hours after ICU admission:

- Mean vasoactive-inotropic score (VIS) 0–6 h
- Baseline serum lactate
- Early lactate clearance
"""
)

st.warning(
    "Research use only. This calculator is not a medical device and should not be used as a standalone clinical decision-making tool."
)

st.divider()

# =========================================================
# Sidebar inputs
# =========================================================
st.sidebar.header("Early 0–6 h Inputs")

mean_vis_0_6h = st.sidebar.number_input(
    "Mean VIS 0–6 h",
    min_value=0.0,
    max_value=300.0,
    value=25.0,
    step=1.0,
    help="Mean vasoactive-inotropic score during the first 6 hours after ICU admission."
)

max_vis_0_6h = st.sidebar.number_input(
    "Maximum VIS 0–6 h",
    min_value=0.0,
    max_value=300.0,
    value=40.0,
    step=1.0,
    help="Maximum vasoactive-inotropic score during the first 6 hours."
)

baseline_lactate = st.sidebar.number_input(
    "Baseline lactate (mmol/L)",
    min_value=0.0,
    max_value=30.0,
    value=5.0,
    step=0.1,
    help="Initial serum lactate near ICU admission."
)

lactate_6h = st.sidebar.number_input(
    "Lactate at 6 h (mmol/L)",
    min_value=0.0,
    max_value=30.0,
    value=4.8,
    step=0.1,
    help="Serum lactate around 6 hours after ICU admission."
)

ph_min_0_6h = st.sidebar.number_input(
    "Minimum pH 0–6 h",
    min_value=6.80,
    max_value=7.80,
    value=7.25,
    step=0.01,
    help="Lowest arterial or venous pH during the first 6 hours."
)

anion_gap_max_0_6h = st.sidebar.number_input(
    "Maximum anion gap 0–6 h",
    min_value=0.0,
    max_value=60.0,
    value=18.0,
    step=0.5,
    help="Maximum anion gap during the first 6 hours."
)

# =========================================================
# Calculations
# =========================================================
clearance = lactate_clearance_pct(baseline_lactate, lactate_6h)

rule_positive = bedside_rule(
    mean_vis_0_6h=mean_vis_0_6h,
    baseline_lactate=baseline_lactate,
    lactate_clearance=clearance
)

p1_prob = p1_probability_model(
    mean_vis_0_6h=mean_vis_0_6h,
    max_vis_0_6h=max_vis_0_6h,
    baseline_lactate=baseline_lactate,
    lactate_6h=lactate_6h,
    lactate_clearance=clearance,
    ph_min_0_6h=ph_min_0_6h,
    anion_gap_max_0_6h=anion_gap_max_0_6h,
)

# =========================================================
# Results display
# =========================================================
st.subheader("Early P1 Enrichment Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Lactate clearance", f"{clearance:.1f}%")

with col2:
    st.metric("Bedside rule", "Positive" if rule_positive == 1 else "Negative")

with col3:
    st.metric("Estimated P1 likelihood", f"{p1_prob * 100:.1f}%")

if rule_positive == 1:
    st.error(
        "Rule-positive profile: high early VIS, elevated baseline lactate, and poor lactate clearance. This pattern is consistent with enrichment for the P1 decoupled phenotype."
    )
else:
    st.success(
        "Rule-negative profile: the patient does not meet the prespecified bedside enrichment rule for the P1 decoupled phenotype."
    )

st.markdown("### Model-based likelihood category")
st.info(category_from_probability(p1_prob))

# =========================================================
# Rule details
# =========================================================
st.markdown("### Bedside rule components")

rule_df = pd.DataFrame(
    {
        "Component": [
            "Mean VIS 0–6 h > 20",
            "Baseline lactate > 4 mmol/L",
            "Lactate clearance < 10%",
        ],
        "Patient value": [
            f"{mean_vis_0_6h:.1f}",
            f"{baseline_lactate:.1f} mmol/L",
            f"{clearance:.1f}%",
        ],
        "Criterion met": [
            "Yes" if mean_vis_0_6h > 20 else "No",
            "Yes" if baseline_lactate > 4 else "No",
            "Yes" if clearance < 10 else "No",
        ],
    }
)

st.dataframe(rule_df, use_container_width=True, hide_index=True)

# =========================================================
# Interpretation
# =========================================================
st.divider()
st.subheader("Interpretation")

st.markdown(
    """
The P1 phenotype represents a high-risk pattern characterized by **discordance
between macro-circulatory support and micro-circulatory/metabolic recovery**.
Clinically, this appears as substantial vasoactive requirement with persistent
hyperlactatemia and impaired lactate clearance.

The bedside rule is designed for **early enrichment**, not definitive diagnosis.
Its role is to identify patients who may resemble the P1 trajectory phenotype
before the full 72-hour trajectory is available.
"""
)

# =========================================================
# Model information and disclaimer
# =========================================================
with st.expander("Model information and disclaimer"):
    st.markdown(
        """
**Primary rule**

A patient is rule-positive when all three conditions are met:

1. Mean VIS 0–6 h > 20  
2. Baseline lactate > 4 mmol/L  
3. Lactate clearance < 10%  

**Intended use**

This calculator is intended for academic demonstration, research communication,
and reproducibility support. It is not intended to guide treatment decisions.

**Important limitation**

The probability score shown here is a demonstration score. For formal publication
or clinical research deployment, the coefficients should be replaced by the final
model coefficients derived from the locked analysis dataset.
"""
    )

st.caption(
    "P1 sepsis phenotype research calculator. For academic and research use only."
)
