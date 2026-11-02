from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "loan_pipeline.joblib"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def label(name: str) -> str:
    return name.replace("_", " ").title()


def confidence_band(value: float) -> str:
    if value >= 0.9:
        return "Very high"
    if value >= 0.7:
        return "High"
    if value >= 0.55:
        return "Moderate"
    return "Low"


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Run the notebook first to generate `models/loan_pipeline.joblib`."
    )
    st.stop()

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
options = artifact["categorical_options"]
ranges = artifact["numeric_ranges"]

st.title("🏦 Loan Approval Predictor")
st.caption("Fill in the applicant and loan details, then click **Predict** to see the decision.")

with st.sidebar:
    st.header("About the model")
    st.write("K-Nearest Neighbors (k=3) inside a scikit-learn pipeline.")
    for name, value in artifact["metrics"].items():
        st.metric(name.title(), f"{value:.1%}")
    st.caption("Metrics measured on the held-out 30% test split.")

with st.form("loan_form"):
    st.subheader("👤 Applicant")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=int(ranges["person_age"][2]), step=1)
    gender = c2.selectbox("Gender", options["person_gender"], format_func=str.title)
    education = c1.selectbox("Education", options["person_education"])
    emp_exp = c2.number_input(
        "Employment experience (years)", min_value=0, max_value=80, value=int(ranges["person_emp_exp"][2]), step=1
    )
    income = c1.number_input(
        "Annual income ($)", min_value=1000.0, max_value=1_000_000.0,
        value=float(ranges["person_income"][2]), step=1000.0,
    )
    home = c2.selectbox("Home ownership", options["person_home_ownership"], format_func=str.title)

    st.subheader("💳 Credit profile")
    c3, c4 = st.columns(2)
    credit_score = c3.slider(
        "Credit score", int(ranges["credit_score"][0]), int(ranges["credit_score"][1]),
        int(ranges["credit_score"][2]),
    )
    hist_len = c4.number_input(
        "Credit history length (years)", min_value=0, max_value=50,
        value=int(ranges["cb_person_cred_hist_length"][2]), step=1,
    )
    prev_default = c3.selectbox(
        "Previous loan defaults on file", options["previous_loan_defaults_on_file"]
    )

    st.subheader("💰 Loan request")
    c5, c6 = st.columns(2)
    loan_amnt = c5.number_input(
        "Loan amount ($)", min_value=500.0, max_value=100_000.0,
        value=float(ranges["loan_amnt"][2]), step=500.0,
    )
    intent = c6.selectbox("Loan purpose", options["loan_intent"], format_func=lambda x: label(x.lower()))
    int_rate = c5.number_input(
        "Interest rate (%)", min_value=1.0, max_value=40.0,
        value=float(ranges["loan_int_rate"][2]), step=0.1,
    )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    percent_income = loan_amnt / income
    row = {
        "person_age": float(age),
        "person_gender": gender,
        "person_education": education,
        "person_income": float(income),
        "person_emp_exp": int(emp_exp),
        "person_home_ownership": home,
        "loan_amnt": float(loan_amnt),
        "loan_intent": intent,
        "loan_int_rate": float(int_rate),
        "loan_percent_income": float(percent_income),
        "cb_person_cred_hist_length": float(hist_len),
        "credit_score": int(credit_score),
        "previous_loan_defaults_on_file": prev_default,
    }
    X = pd.DataFrame([row])[feature_columns]

    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.classes_)
    prediction = int(pipeline.predict(X)[0])
    confidence = float(proba[classes.index(prediction)])
    approve_prob = float(proba[classes.index(1)])

    st.divider()
    band = confidence_band(confidence)
    if prediction == 1:
        st.success(
            f"✅ **Loan APPROVED** — the model is **{confidence:.0%} confident** ({band}) in this decision."
        )
    else:
        st.error(
            f"❌ **Loan REJECTED** — the model is **{confidence:.0%} confident** ({band}) in this decision."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{confidence:.0%}")
    m2.metric("Approval probability", f"{approve_prob:.0%}")
    m3.metric("Loan / income", f"{percent_income:.1%}")
    st.progress(approve_prob, text=f"Approval probability: {approve_prob:.0%}")

    if percent_income > ranges["loan_percent_income"][1]:
        st.warning(
            "The loan is larger relative to income than anything in the training data; "
            "treat this result with caution."
        )

    with st.expander("Show the data sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}).astype(str))
