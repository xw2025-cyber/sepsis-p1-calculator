import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="P1 Sepsis Phenotype ML Calculator",
    page_icon="🧬",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("p1_ml_model.pkl")
    with open("p1_model_features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    with open("p1_model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, features, metadata

def lactate_clearance_pct(baseline_lactate: float, lactate_6h: float) -> float:
    if baseline_lactate <= 0:
        return 0.0
    return (baseline_lactate - lactate_6h) / baseline_lactate * 100

def bedside_rule(mean_vis_0_6h: float, baseline_lactate: float, lactate_clearance: float) -> bool:
    return (
        mean_vis_0_6h > 20
        and baseline_lactate > 4
        and lactate_clearance < 10
    )

def probability_category(prob: float) -> str:
    if prob < 0.05:
        return "Low ML-predicted P1 probability"
    elif prob < 0.15:
        return "Intermediate ML-predicted P1 probability"
    elif prob < 0.30:
        return "High ML-predicted P1 probability"
    return "Very high ML-predicted P1 probability"

model, features, metadata = load_artifacts()
threshold = float(metadata.get("classification_threshold", 0.5))

st.title("P1 Sepsis Phenotype ML Calculator")

st.markdown(
    """
This web calculator estimates **machine-learning predicted probability** of the
hemodynamically–metabolically decoupled sepsis phenotype (**P1**) using early
0–6 h VIS–lactate dynamics.

The model was trained in the MIMIC-IV derivation cohort and externally evaluated
in eICU. The bedside rule is shown alongside the model output for interpretability.
"""
)

st.warning(
    "Research use only. This calculator is not a medical device and should not be used as a standalone clinical decision-making tool."
)

st.sidebar.header("Early 0–6 h Inputs")

mean_vis_0_6h = st.sidebar.number_input(
    "Mean VIS 0–6 h",
    min_value=0.0, max_value=300.0, value=25.0, step=1.0,
    help="Mean vasoactive-inotropic score during the first 6 hours after ICU admission."
)

max_vis_0_6h = st.sidebar.number_input(
    "Maximum VIS 0–6 h",
    min_value=0.0, max_value=300.0, value=40.0, step=1.0,
    help="Maximum vasoactive-inotropic score during the first 6 hours."
)

baseline_lactate = st.sidebar.number_input(
    "Baseline lactate (mmol/L)",
    min_value=0.0, max_value=30.0, value=5.0, step=0.1,
    help="Initial serum lactate near ICU admission."
)

lactate_6h = st.sidebar.number_input(
    "Lactate at 6 h (mmol/L)",
    min_value=0.0, max_value=30.0, value=4.8, step=0.1,
    help="Serum lactate around 6 hours after ICU admission."
)

mean_lactate_0_6h = st.sidebar.number_input(
    "Mean lactate 0–6 h (mmol/L)",
    min_value=0.0, max_value=30.0, value=4.9, step=0.1,
    help="Mean serum lactate during the first 6 hours."
)

clearance = lactate_clearance_pct(baseline_lactate, lactate_6h)

input_dict = {
    "mean_vis_0_6h": mean_vis_0_6h,
    "max_vis_0_6h": max_vis_0_6h,
    "baseline_lactate": baseline_lactate,
    "mean_lactate_0_6h": mean_lactate_0_6h,
    "lactate_clearance_6h_pct": clearance,
}

X_input = pd.DataFrame([[input_dict[f] for f in features]], columns=features)

ml_probability = float(model.predict_proba(X_input)[0, 1])
ml_positive = ml_probability >= threshold
rule_positive = bedside_rule(mean_vis_0_6h, baseline_lactate, clearance)

st.divider()
st.subheader("Early P1 Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ML-predicted P1 probability", f"{ml_probability * 100:.1f}%")

with col2:
    st.metric("ML classification", "Positive" if ml_positive else "Negative")
    st.caption(f"Threshold: {threshold:.3f}")

with col3:
    st.metric("Bedside rule", "Positive" if rule_positive else "Negative")

st.markdown("### Probability category")
if ml_positive:
    st.error(probability_category(ml_probability))
else:
    st.info(probability_category(ml_probability))

st.markdown("### Input-derived features")
feature_df = pd.DataFrame({
    "Feature": [
        "Mean VIS 0–6 h",
        "Maximum VIS 0–6 h",
        "Baseline lactate",
        "Lactate at 6 h",
        "Mean lactate 0–6 h",
        "Lactate clearance",
    ],
    "Value": [
        f"{mean_vis_0_6h:.1f}",
        f"{max_vis_0_6h:.1f}",
        f"{baseline_lactate:.1f} mmol/L",
        f"{lactate_6h:.1f} mmol/L",
        f"{mean_lactate_0_6h:.1f} mmol/L",
        f"{clearance:.1f}%",
    ],
})
st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.markdown("### Bedside rule components")
rule_df = pd.DataFrame({
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
})
st.dataframe(rule_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Interpretation")

st.markdown(
    """
The ML model estimates early similarity to the P1 trajectory phenotype before the
full 72-hour VIS–lactate trajectory is available. The bedside rule provides a
transparent enrichment signal, whereas the ML probability summarizes the combined
dynamic pattern using the locked model artifact.

A positive ML classification or rule-positive profile should be interpreted as
**research enrichment for the P1 phenotype**, not as a definitive bedside diagnosis.
"""
)

with st.expander("Model information and disclaimer"):
    st.markdown(
        f"""
**Model type:** {metadata.get("model_name", "ML model")} dynamic-only early prediction model

**Training dataset:** {metadata.get("training_dataset", "MIMIC-IV")}

**External validation dataset:** {metadata.get("external_validation_dataset", "eICU")}

**Features used by the deployed model:**

{chr(10).join([f"- `{f}`" for f in features])}

**Classification threshold:** {threshold:.3f}

**Disclaimer:** This calculator is intended for academic demonstration and research
communication only. It is not a medical device and should not replace clinical judgment.
"""
    )

st.caption("P1 sepsis phenotype ML calculator. For academic and research use only.")
