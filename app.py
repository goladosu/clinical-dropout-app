import os
import sys
# Standard library imports
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ML and Streamlit components
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import streamlit as st


# ==============================================================================
# Custom Pipeline Components (Names MUST match the pickled model!)
# ==============================================================================

class ClinicalConsistencyTransformer(BaseEstimator, TransformerMixin):
    # Rule enforcement step: check data logic before processing.

    def __init__(
        self,
        visit1_cols: Optional[List[str]] = None,
        visit2_cols: Optional[List[str]] = None,
        age_col: str = "age",
        min_age: int = 18,
        max_age: int = 90
    ):
        # Default columns for V1 and V2
        self.visit1_cols = visit1_cols or [
            "visit1_symptom_score",
            "visit1_adherence_rate",
            "visit1_AE_count",
        ]
        self.visit2_cols = visit2_cols or [
            "visit2_symptom_score",
            "visit2_adherence_rate",
            "visit2_AE_count",
        ]

        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Clamp age to a sensible 18-90 range, don't drop rows.
        if self.age_col in X.columns:
            X.loc[X[self.age_col] < self.min_age, self.age_col] = self.min_age
            X.loc[X[self.age_col] > self.max_age, self.age_col] = self.max_age

        # Visit dependency logic: V2 can't exist if V1 is completely blank.
        v1_cols = [c for c in self.visit1_cols if c in X.columns]
        v2_cols = [c for c in self.visit2_cols if c in X.columns]

        if v1_cols and v2_cols:
            # Mask where ALL V1 fields are NaN
            no_v1_mask = X[v1_cols].isna().all(axis=1)

            # Wipe V2 data if V1 is missing
            X.loc[no_v1_mask, v2_cols] = np.nan

        return X


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    # Adds the `_missing` flags needed for XGBoost to learn from missingness.

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # 1 if missing, 0 otherwise
                X[f"{col}_missing"] = X[col].isna().astype(int)
        return X


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fix unpickling errors (often needed for Streamlit/notebooks)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sys.modules["main"] = sys.modules[__name__]
sys.modules["__main__"] = sys.modules[__name__]


# ==============================================================================
# App config + model loader
# ==============================================================================

st.set_page_config(page_title="Abdul Oladosu", layout="wide")

MODEL_PATH = "xgb_dropout_pipeline.pkl"

# Deployment cutoff, set by evaluation notebook
CHOSEN_THRESHOLD = 0.30


@st.cache_resource
def load_pipeline():
    # Only load the model once, cache it.
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        pipe = joblib.load(MODEL_PATH)
        return pipe, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


pipeline, load_err = load_pipeline()


# ==============================================================================
# Navigation state
# ==============================================================================

PAGES = ["Home", "Resume", "Projects", "Dropout Model", "Grade Prediction Model"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    PAGES,
    index=PAGES.index(st.session_state.page),
)
st.session_state.page = page


# ==============================================================================
# Helpers
# ==============================================================================

def risk_bucket(p: float) -> str:
    # Assign risk level based on probability p.
    # High risk means >= CHOSEN_THRESHOLD
    if p < 0.15:
        return "Low"
    elif p < CHOSEN_THRESHOLD:  # 0.30
        return "Moderate"
    else:
        return "High"


def build_feature_names_from_preprocessor(preprocess, numeric_features, categorical_features):
    # Reconstruct feature names after OHE for SHAP display.
    num_names = list(numeric_features)

    try:
        # Get OHE names from the 'cat' step
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = []  # Handle if encoding step isn't available

    return np.array(num_names + cat_names, dtype=object)


def get_shap_for_single_row(pipe, input_df):
    # Runs the single input row through the pipeline steps to generate SHAP values.
    # Returns (prob, pred, shap_values_row, feature_names)

    # Run prediction first
    prob = float(pipe.predict_proba(input_df)[0, 1])
    pred = int(prob >= CHOSEN_THRESHOLD)

    # Extract components for manual transformation
    clinical = pipe.named_steps.get("clinical_logic")
    flags = pipe.named_steps.get("missing_flags")
    preprocess = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    # Transform data through custom steps
    X_logic = clinical.transform(input_df) if clinical else input_df
    X_flags = flags.transform(X_logic) if flags else X_logic

    # Final preprocessing (scaling, encoding, etc.)
    X_prep = preprocess.transform(X_flags)

    # Ensure it's a dense matrix for SHAP
    try:
        X_prep_dense = X_prep.toarray()
    except Exception:
        X_prep_dense = X_prep

    # Try to extract the feature names list
    try:
        num_cols = preprocess.transformers_[0][2]
        numeric_features = list(num_cols)
    except Exception:
        # Generic names if extraction fails
        numeric_features = [f"x{i}" for i in range(X_prep_dense.shape[1])]

    try:
        # Rebuild full list for plot labels
        feature_names = build_feature_names_from_preprocessor(
            preprocess=preprocess,
            numeric_features=numeric_features,
            categorical_features=["sex", "race"],
        )
        if feature_names.shape[0] != X_prep_dense.shape[1]:
            feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)
    except Exception:
        feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prep_dense)

    # Get the values for class 1 (Dropout)
    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[1][0]).ravel()
    else:
        shap_row = np.array(shap_values[0]).ravel()

    return prob, pred, shap_row, feature_names


# ==============================================================================
# Pages
# ==============================================================================

def page_home():
    st.title("Welcome — I'm Abdul Oladosu")

    st.markdown(
        """
**Clinical Laboratory Scientist → Aspiring Data Scientist**

This is a space where I build and explore **data-driven systems**—from analytics to machine learning.

I’m interested in problems where data is imperfect, decisions matter, and solutions need to be **clear, explainable, and useful**. My work focuses on turning raw data into insights and tools that support real-world decision-making across different domains.

---

### What You’ll Find Here

Hands-on projects that demonstrate:
- Applied analytics and machine learning  
- End-to-end workflows, from data to deployment  
- Thoughtful evaluation and interpretation of results  

---

### How I Approach Data Work

I focus on:
- Understanding the data before modeling  
- Making assumptions explicit  
- Explaining results clearly to non-technical audiences  
- Considering how outputs are actually used in practice  

Interpretability, transparency, and responsible use of data are themes that run across my projects.

---

### Explore

Use the navigation to explore projects, interact with models, and see how data science ideas translate into working applications.
        """
    )

    st.divider()
    st.subheader("Quick Links")
    a, b, c, d = st.columns(4)

    with a:
        if st.button("📄 View Resume", use_container_width=True):
            st.session_state.page = "Resume"
            st.rerun()

    with b:
        if st.button("📊 Projects", use_container_width=True):
            st.session_state.page = "Projects"
            st.rerun()

    with c:
        if st.button("🧪 Dropout Model", use_container_width=True):
            st.session_state.page = "Dropout Model"
            st.rerun()

    with d:
        if st.button("🧑🏻‍🏫 Grade Model", use_container_width=True):
            st.session_state.page = "Grade Prediction Model"
            st.rerun()


def page_resume():
    st.title("Resume")

    st.subheader("Education")
    st.markdown(
        """
- **M.S. Data Science** — Eastern University
- **M.S. Biomedical Science** — Roosevelt University
- **B.S. Biomedical Science** — Gulf Medical University
        """
    )

    st.subheader("Work Experience")
    st.markdown(
        """
**Clinical Laboratory Scientist — Saint Mary Hospital (Chicago)**  
- Perform high-complexity diagnostic testing and QC in a hospital lab environment  
- Collaborate with clinical teams to ensure accurate and timely results  
- Apply data-driven thinking to workflow, quality, and operational improvement  

**Business Associate — Insight Hospital** *(April 2023 – February 2025)*  
- Conducted in-depth market research and analysis, identifying **10+ actionable trends**  
- Produced reports and presentations enabling data-driven decision-making  
- Managed policy adherence and regulatory compliance with healthcare standards  
        """
    )

    st.subheader("Technical Skills")
    st.markdown(
        """
- **Programming:** Python, SQL, R  
- **Machine Learning:** scikit-learn, XGBoost, SHAP  
- **Data Analysis:** EDA, feature engineering, model training  
- **Evaluation:** ROC-AUC, PR-AUC, F1-score, Precision/Recall, Confusion Matrix  
- **Deployment:** Streamlit, model serialization (joblib / pickle)
        """
    )

    st.subheader("Links")
    st.markdown("**GitHub:** https://github.com/goladosu")

    if os.path.exists("Resume.pdf"):
        with open("Resume.pdf", "rb") as f:
            st.download_button(
                "⬇️ Download Resume (PDF)",
                f,
                file_name="Resume.pdf"
            )
    else:
        st.caption("(Optional) Add Resume.pdf to your repo root to enable a download button.")


def page_projects():
    st.title("Projects")
    st.markdown("This page highlights projects.")

    st.subheader("Clinical Trial Participant Dropout Prediction")
    st.markdown(
        """
- Goal: predict **dropout risk after Visit 2** using demographic, clinical, and engagement features  
- Models explored: Logistic Regression, Random Forest, **XGBoost (final)**  
- Interpretability: **SHAP** explanations + error analysis (false positives/negatives)  
- Deployment: this Streamlit web app
        """
    )

    st.subheader("Student Grade Prediction")
    st.markdown(
        """
- Built predictive models (Linear Regression, Lasso, SVR) to forecast students’ final academic performance using early-term grades, attendance, study habits, and background data.
- Achieved strong predictive accuracy (RMSE ≈ 2.2, R² = 0.76).
- Designed a two-stage early-warning system enabling timely interventions before mid-term assessments.
- Identified key drivers of performance, including attendance patterns, study time, and prior outcomes, to support data-driven improvement strategies.
        """
    )


def page_dropout_project():
    st.title("Clinical Trial Dropout Risk — Deployed Model")

    if load_err:
        st.error(load_err)
        st.write("Current working directory:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.stop()

    st.markdown(
        f"""
### Problem (non-technical)
Clinical trials often lose participants before the primary endpoint. Dropout can delay timelines, reduce statistical power,
and introduce bias. This model estimates the **probability of dropout** using information collected up to **Visit 2** so teams
can intervene early.

**Operational threshold:** `{CHOSEN_THRESHOLD:.2f}`
        """
    )

    st.divider()
    st.subheader("Try it: Enter participant data")

    with st.form("participant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=90, value=55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
            BMI = st.slider("BMI", 15.0, 45.0, 27.0)

        with c2:
            baseline_lab_score = st.slider("Baseline Lab Score", 0.0, 200.0, 110.0)
            disease_severity = st.slider("Disease Severity (1–10)", 1.0, 10.0, 5.0)
            prior_treatments = st.number_input("Prior Treatments", min_value=0, max_value=20, value=1)
            missed_appointments = st.number_input("Missed Appointments", min_value=0, max_value=20, value=0)

        with c3:
            communication_score = st.slider("Communication Score (1–5)", 1.0, 5.0, 3.0, step=0.1)
            st.caption("Visits")
            visit1_symptom_score = st.slider("Visit 1 Symptom Score", 0.0, 100.0, 50.0)
            visit1_adherence_rate = st.slider("Visit 1 Adherence (%)", 0.0, 100.0, 80.0)
            visit1_AE_count = st.number_input("Visit 1 AE Count", min_value=0, max_value=20, value=0)

            visit2_symptom_score = st.slider("Visit 2 Symptom Score", 0.0, 100.0, 45.0)
            visit2_adherence_rate = st.slider("Visit 2 Adherence (%)", 0.0, 100.0, 75.0)
            visit2_AE_count = st.number_input("Visit 2 AE Count", min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("Predict Dropout Risk")

    if not submitted:
        return

    # Create the DataFrame from inputs
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "race": race,
        "BMI": BMI,
        "baseline_lab_score": baseline_lab_score,
        "disease_severity": disease_severity,
        "prior_treatments": prior_treatments,
        "visit1_symptom_score": visit1_symptom_score,
        "visit1_adherence_rate": visit1_adherence_rate,
        "visit1_AE_count": visit1_AE_count,
        "visit2_symptom_score": visit2_symptom_score,
        "visit2_adherence_rate": visit2_adherence_rate,
        "visit2_AE_count": visit2_AE_count,
        "missed_appointments": missed_appointments,
        "communication_score": communication_score,
    }])

    # Scale percentage adherence to proportion (0.0 to 1.0)
    for col in ["visit1_adherence_rate", "visit2_adherence_rate"]:
        input_df[col] = input_df[col] / 100.0

    try:
        prob, pred, shap_row, feat_names = get_shap_for_single_row(pipeline, input_df)
    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")
        st.stop()

    st.divider()
    st.subheader("Results")

    bucket = risk_bucket(prob)
    if bucket == "Low":
        st.success(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")
    elif bucket == "Moderate":
        st.warning(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")
    else:
        st.error(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")

    st.write(f"Predicted class (thresholded): {'Dropout (1)' if pred == 1 else 'Completer (0)'}")
    st.caption("This is a risk estimate, not a guarantee.")

    st.subheader("Top drivers of this prediction (SHAP)")
    order = np.argsort(np.abs(shap_row))[::-1][:10]
    top_df = pd.DataFrame({
        "feature": feat_names[order],
        "shap_value": shap_row[order],
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top_df["feature"][::-1], top_df["shap_value"][::-1])
    ax.set_xlabel("SHAP contribution (positive increases dropout risk)")
    ax.set_ylabel("Feature")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show participant input"):
        st.dataframe(input_df, use_container_width=True)


def page_grade_project():
    st.title("Student Grade Prediction — Deployed Model")
    st.info("In progress 🚧")


# ==============================================================================
# Router
# ==============================================================================

if page == "Home":
    page_home()
elif page == "Resume":
    page_resume()
elif page == "Projects":
    page_projects()
elif page == "Dropout Model":
    page_dropout_project()
elif page == "Grade Prediction Model":
    page_grade_project()
