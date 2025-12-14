import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# 1) Custom Transformers (MUST exist for unpickling)
# ============================================================

class ClinicalConsistencyTransformer(BaseEstimator, TransformerMixin):
    """Enforces trial protocol consistency.

    - Age cleaning: clamp age < min_age to min_age (adult-only trial).
                 AND clamp age > max_age to max_age (clinically plausible upper bound).
    - Visit logic: if ALL Visit 1 fields are missing, clear ALL Visit 2 fields.
    """

    def __init__(self, visit1_cols=None, visit2_cols=None, age_col="age", min_age=18, max_age=90):
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

        # Age cleaning
        if self.age_col in X.columns:
            X.loc[X[self.age_col] < self.min_age, self.age_col] = self.min_age
            X.loc[X[self.age_col] > self.max_age, self.age_col] = self.max_age

        # Visit logic: only when ALL Visit 1 vars are missing
        v1_cols = [c for c in self.visit1_cols if c in X.columns]
        v2_cols = [c for c in self.visit2_cols if c in X.columns]
        if v1_cols and v2_cols:
            no_v1_mask = X[v1_cols].isna().all(axis=1)
            X.loc[no_v1_mask, v2_cols] = np.nan

        return X


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adds *_missing indicator columns (0/1) for selected columns.

    NOTE: Keep constructor clone-safe (do not modify params here).
    """

    def __init__(self, columns):
        self.columns = columns  # do NOT wrap with list() here

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(int)
        return X


# ------------------------------------------------------------
# Make unpickling robust across Colab / local / Streamlit Cloud
# ------------------------------------------------------------
sys.modules["main"] = sys.modules[__name__]
sys.modules["__main__"] = sys.modules[__name__]


# ============================================================
# 2) App config + model loader
# ============================================================

st.set_page_config(page_title="Gbolahan Oladosu | DTSC691 Project", layout="wide")

MODEL_PATH = "xgb_dropout_pipeline.pkl"

# Use your tuned threshold from notebook here:
CHOSEN_THRESHOLD = 0.45  # <-- replace with your printed chosen_threshold


@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        pipe = joblib.load(MODEL_PATH)
        return pipe, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


pipeline, load_err = load_pipeline()


# ============================================================
# 3) Navigation state
# ============================================================

PAGES = ["Home", "Resume", "Projects", "Dropout Project"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    PAGES,
    index=PAGES.index(st.session_state.page),
)
st.session_state.page = page


# ============================================================
# 4) Helpers
# ============================================================

def risk_bucket(p: float) -> str:
    # Bucket anchored around your operational threshold
    if p < CHOSEN_THRESHOLD * 0.7:
        return "Low"
    if p < CHOSEN_THRESHOLD:
        return "Moderate"
    return "High"


def build_feature_names_from_preprocessor(preprocess, numeric_features, categorical_features):
    """Rebuild transformed feature names for SHAP plots."""
    num_names = list(numeric_features)

    try:
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = []

    feat_names = num_names + cat_names
    return np.array(feat_names, dtype=object)


def get_shap_for_single_row(pipe, input_df):
    """Returns (prob, pred, shap_values_row, feature_names) for one participant."""
    # Probability (always from predict_proba)
    prob = float(pipe.predict_proba(input_df)[0, 1])
    # Class decision using tuned threshold
    pred = int(prob >= CHOSEN_THRESHOLD)

    # Prepare data for SHAP (must match what the model sees)
    clinical = pipe.named_steps.get("clinical_logic")
    flags = pipe.named_steps.get("missing_flags")
    preprocess = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    X_logic = clinical.transform(input_df) if clinical else input_df
    X_flags = flags.transform(X_logic) if flags else X_logic
    X_prep = preprocess.transform(X_flags)

    # Dense conversion (helps SHAP + plotting)
    try:
        X_prep_dense = X_prep.toarray()
    except Exception:
        X_prep_dense = X_prep

    # Infer numeric feature list from ColumnTransformer definition
    try:
        num_cols = preprocess.transformers_[0][2]
        numeric_features = list(num_cols)
    except Exception:
        numeric_features = [f"x{i}" for i in range(X_prep_dense.shape[1])]

    # Build feature names
    try:
        feature_names = build_feature_names_from_preprocessor(
            preprocess=preprocess,
            numeric_features=numeric_features,
            categorical_features=["sex", "race"],
        )
        if feature_names.shape[0] != X_prep_dense.shape[1]:
            feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)
    except Exception:
        feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prep_dense)

    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[0][0]).ravel()
    else:
        shap_row = np.array(shap_values[0]).ravel()

    return prob, pred, shap_row, feature_names


# ============================================================
# 5) Pages
# ============================================================

def page_home():
    st.title("Welcome — I'm Gbolahan (Abdul) Oladosu")

    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        if os.path.exists("assets/headshot.jpg"):
            st.image("assets/headshot.jpg", width=220)
        else:
            st.info("Add a headshot at assets/headshot.jpg")

    with c2:
        st.markdown(
            """
**Clinical Laboratory Scientist** → **Aspiring Clinical Data Scientist** (Healthcare & Clinical Trials)

I’m completing my **M.S. in Data Science** and building an end-to-end machine learning project (DTSC691) focused on
**clinical trial participant retention**.

**Professional interests**
- Clinical trial analytics (retention, adherence, operational quality)
- Healthcare machine learning + interpretable AI (SHAP)
- Responsible/fair modeling across demographic subgroups

**A bit personal**
I enjoy problem-solving, learning new tools in Python/R/SQL, and translating technical findings into decisions teams can act on.
"""
        )

    st.divider()
    st.subheader("Quick Links")
    a, b, c = st.columns(3)
    with a:
        if st.button("📄 View Resume", use_container_width=True):
            st.session_state.page = "Resume"
            st.rerun()
    with b:
        if st.button("📊 Projects", use_container_width=True):
            st.session_state.page = "Projects"
            st.rerun()
    with c:
        if st.button("🧪 Dropout Project", use_container_width=True):
            st.session_state.page = "Dropout Project"
            st.rerun()


def page_resume():
    st.title("Resume")

    st.subheader("Education")
    st.markdown(
        """
- **M.S. Data Science** — Eastern University *(in progress)*
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

    st.markdown("This page highlights selected projects.")

    st.subheader("Clinical Trial Participant Dropout Prediction (DTSC691)")
    st.markdown(
        """
- Goal: predict **dropout risk after Visit 2** using demographic, clinical, and engagement features  
- Models explored: Logistic Regression, Random Forest, **XGBoost (final)**  
- Interpretability: **SHAP** explanations + error analysis (false positives/negatives)  
- Deployment: this Streamlit web app
"""
    )

    st.subheader("Other Projects")
    st.markdown("- Student Grade Prediction (Machine Learning)")


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

**Operational threshold:** `{CHOSEN_THRESHOLD:.2f}` (tuned in model evaluation)

### What the model uses
- Demographics: age, sex, race
- Clinical status: BMI, baseline lab score, disease severity, prior treatments
- Engagement & safety signals: adherence rates, adverse event counts, missed appointments, communication score
- Clinical consistency rules (adult-only ages; no Visit 2 without Visit 1; ages capped at 90)
"""
    )

    st.divider()
    st.subheader("Try it: Enter participant data")

    with st.form("participant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            # UPDATED: max_value now 90 to match transformer upper bound
            age = st.number_input("Age", min_value=18, max_value=90, value=55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
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

    # Build input row
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

    # Convert adherence from percent (0–100) → proportion (0–1) if your model was trained on 0–1
    for col in ["visit1_adherence_rate", "visit2_adherence_rate"]:
        input_df[col] = input_df[col] / 100.0

    # Predict + SHAP
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
    st.caption("Interpretation: probability is a risk estimate, not a guarantee.")

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


# ============================================================
# 6) Router
# ============================================================

if page == "Home":
    page_home()
elif page == "Resume":
    page_resume()
elif page == "Projects":
    page_projects()
elif page == "Dropout Project":
    page_dropout_project()
