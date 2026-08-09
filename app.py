import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "diabetes_random_forest_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(14, 165, 233, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(37, 99, 235, 0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07111f 0%,
            #0b1628 50%,
            #0f1d31 100%
        );

    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 30px 35px;
    border-radius: 24px;
    margin-bottom: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(14, 165, 233, 0.15),
            rgba(37, 99, 235, 0.07)
        );

    border: 1px solid rgba(56, 189, 248, 0.18);

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.25);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 17px;
    color: #94a3b8;
}

.model-badge {
    display: inline-block;

    margin-top: 15px;

    padding: 7px 14px;

    border-radius: 50px;

    background: rgba(34, 197, 94, 0.10);

    border: 1px solid rgba(34, 197, 94, 0.25);

    color: #86efac;

    font-size: 13px;

    font-weight: 600;
}

.card {
    background: rgba(15, 23, 42, 0.78);

    border: 1px solid rgba(148, 163, 184, 0.12);

    border-radius: 20px;

    padding: 24px;

    margin-bottom: 20px;

    box-shadow:
        0 12px 35px rgba(0, 0, 0, 0.20);
}

.card-title {
    font-size: 20px;
    font-weight: 750;
    color: #f8fafc;
}

.card-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

.section-title {
    font-size: 13px;

    font-weight: 700;

    color: #38bdf8;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-top: 18px;

    margin-bottom: 10px;
}

.positive-card {
    padding: 22px;

    border-radius: 18px;

    text-align: center;

    background: rgba(239, 68, 68, 0.08);

    border: 1px solid rgba(239, 68, 68, 0.25);
}

.negative-card {
    padding: 22px;

    border-radius: 18px;

    text-align: center;

    background: rgba(34, 197, 94, 0.08);

    border: 1px solid rgba(34, 197, 94, 0.25);
}

.prediction-label {
    color: #94a3b8;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.prediction-value {
    font-size: 34px;

    font-weight: 800;

    margin-top: 5px;
}

.metric-card {
    background: rgba(30, 41, 59, 0.55);

    border: 1px solid rgba(148, 163, 184, 0.10);

    border-radius: 15px;

    padding: 18px;

    text-align: center;
}

.metric-title {
    font-size: 11px;

    color: #64748b;

    text-transform: uppercase;
}

.metric-value {
    font-size: 20px;

    font-weight: 750;

    color: #f8fafc;

    margin-top: 5px;
}

.summary-card {
    background: rgba(30, 41, 59, 0.45);

    border: 1px solid rgba(148, 163, 184, 0.08);

    border-radius: 13px;

    padding: 14px;

    text-align: center;
}

.summary-label {
    font-size: 11px;

    color: #64748b;

    text-transform: uppercase;
}

.summary-value {
    font-size: 18px;

    font-weight: 700;

    color: #e2e8f0;

    margin-top: 4px;
}

section[data-testid="stSidebar"] {
    background: #07111f;

    border-right:
        1px solid rgba(148, 163, 184, 0.10);
}

div[data-baseweb="select"] > div {
    background-color: #111c2e;

    border-color:
        rgba(148, 163, 184, 0.15);

    border-radius: 10px;
}

input {
    background-color: #111c2e !important;

    color: #f8fafc !important;
}

.stButton > button {
    width: 100%;

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            135deg,
            #0ea5e9,
            #2563eb
        );

    color: white;

    font-weight: 700;

    padding: 12px 20px;

    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(14, 165, 233, 0.30);
}

.footer {
    text-align: center;

    color: #64748b;

    font-size: 12px;

    margin-top: 35px;

    padding: 20px;
}

</style>
""")


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        🩺 Diabetes Prediction
    </div>

    <div class="hero-subtitle">
        Diabetes Risk Assessment
    </div>

    <div class="model-badge">
        ● Random Forest Classification Model
    </div>

</div>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧑‍⚕️ Patient Information")

    st.html("""
    <div style="
        color:#64748b;
        font-size:13px;
        margin-bottom:20px;
    ">
        Enter patient demographic information
        and symptoms.
    </div>
    """)

    # =====================================================
    # PATIENT DETAILS
    # =====================================================

    st.html("""
    <div class="section-title">
        Patient Details
    </div>
    """)

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    # =====================================================
    # SYMPTOMS
    # =====================================================

    st.html("""
    <div class="section-title">
        Symptoms
    </div>
    """)

    polyuria = st.selectbox(
        "Polyuria",
        ["No", "Yes"]
    )

    polydipsia = st.selectbox(
        "Polydipsia",
        ["No", "Yes"]
    )

    sudden_weight_loss = st.selectbox(
        "Sudden Weight Loss",
        ["No", "Yes"]
    )

    weakness = st.selectbox(
        "Weakness",
        ["No", "Yes"]
    )

    polyphagia = st.selectbox(
        "Polyphagia",
        ["No", "Yes"]
    )

    genital_thrush = st.selectbox(
        "Genital Thrush",
        ["No", "Yes"]
    )

    visual_blurring = st.selectbox(
        "Visual Blurring",
        ["No", "Yes"]
    )

    itching = st.selectbox(
        "Itching",
        ["No", "Yes"]
    )

    irritability = st.selectbox(
        "Irritability",
        ["No", "Yes"]
    )

    delayed_healing = st.selectbox(
        "Delayed Healing",
        ["No", "Yes"]
    )

    partial_paresis = st.selectbox(
        "Partial Paresis",
        ["No", "Yes"]
    )

    muscle_stiffness = st.selectbox(
        "Muscle Stiffness",
        ["No", "Yes"]
    )

    alopecia = st.selectbox(
        "Alopecia",
        ["No", "Yes"]
    )

    obesity = st.selectbox(
        "Obesity",
        ["No", "Yes"]
    )

    st.html("<br>")

    predict_button = st.button(
        "🔍 Predict Diabetes Risk"
    )


# =========================================================
# MAIN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [1.05, 0.95],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left_col:

    st.html("""
    <div class="card">

        <div class="card-title">
            📋 Patient Assessment
        </div>

        <div class="card-subtitle">
            Current patient information and selected symptoms
        </div>

    </div>
    """)

    # =====================================================
    # SYMPTOMS DATA
    # =====================================================

    symptom_data = {

        "Polyuria": polyuria,

        "Polydipsia": polydipsia,

        "Sudden Weight Loss":
            sudden_weight_loss,

        "Weakness":
            weakness,

        "Polyphagia":
            polyphagia,

        "Genital Thrush":
            genital_thrush,

        "Visual Blurring":
            visual_blurring,

        "Itching":
            itching,

        "Irritability":
            irritability,

        "Delayed Healing":
            delayed_healing,

        "Partial Paresis":
            partial_paresis,

        "Muscle Stiffness":
            muscle_stiffness,

        "Alopecia":
            alopecia,

        "Obesity":
            obesity
    }

    yes_count = list(
        symptom_data.values()
    ).count("Yes")

    # =====================================================
    # SUMMARY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html(f"""
        <div class="summary-card">

            <div class="summary-label">
                Age
            </div>

            <div class="summary-value">
                {age}
            </div>

        </div>
        """)

    with col2:

        st.html(f"""
        <div class="summary-card">

            <div class="summary-label">
                Gender
            </div>

            <div class="summary-value">
                {gender}
            </div>

        </div>
        """)

    with col3:

        st.html(f"""
        <div class="summary-card">

            <div class="summary-label">
                Symptoms
            </div>

            <div class="summary-value">
                {yes_count} / 14
            </div>

        </div>
        """)

    st.html("<br>")

    # =====================================================
    # SYMPTOMS OVERVIEW
    # =====================================================

    st.html("""
    <div class="card">

        <div class="card-title">
            Symptoms Overview
        </div>

    </div>
    """)

    symptom_columns = st.columns(2)

    for i, (symptom, value) in enumerate(
        symptom_data.items()
    ):

        with symptom_columns[i % 2]:

            if value == "Yes":

                icon = "●"
                icon_color = "#38bdf8"

            else:

                icon = "○"
                icon_color = "#64748b"

            st.html(f"""
            <div style="
                padding:9px 12px;
                margin-bottom:7px;
                background:
                    rgba(30,41,59,0.45);
                border-radius:10px;
                border:
                    1px solid
                    rgba(148,163,184,0.07);
            ">

                <span style="
                    color:{icon_color};
                ">
                    {icon}
                </span>

                <span style="
                    color:#cbd5e1;
                    margin-left:7px;
                    font-size:13px;
                ">
                    {symptom}
                </span>

                <span style="
                    float:right;
                    color:#64748b;
                    font-size:12px;
                ">
                    {value}
                </span>

            </div>
            """)


# =========================================================
# RIGHT SIDE
# =========================================================

with right_col:

    st.html("""
    <div class="card">

        <div class="card-title">
            📊 Prediction Result
        </div>

        <div class="card-subtitle">
            Random Forest model assessment
        </div>

    </div>
    """)

    # =====================================================
    # BEFORE PREDICTION
    # =====================================================

    if not predict_button:

        st.html("""
        <div style="
            background:
                rgba(15,23,42,0.70);

            border:
                1px solid
                rgba(148,163,184,0.10);

            border-radius:20px;

            padding:60px 25px;

            text-align:center;
        ">

            <div style="
                font-size:55px;
                margin-bottom:15px;
            ">
                🩺
            </div>

            <div style="
                font-size:23px;
                font-weight:750;
                color:#f8fafc;
            ">
                Ready for Assessment
            </div>

            <div style="
                color:#64748b;
                margin-top:10px;
                font-size:14px;
            ">
                Enter the patient information
                from the sidebar and click
                Predict Diabetes Risk.
            </div>

        </div>
        """)

    # =====================================================
    # PREDICTION
    # =====================================================

    else:

        # -------------------------------------------------
        # Encoding
        # -------------------------------------------------

        binary_mapping = {
            "No": 0,
            "Yes": 1
        }

        gender_mapping = {
            "Female": 0,
            "Male": 1
        }

        # -------------------------------------------------
        # Patient DataFrame
        # -------------------------------------------------

        patient_data = pd.DataFrame(
            {

                "Age": [
                    age
                ],

                "Gender": [
                    gender_mapping[gender]
                ],

                "Polyuria": [
                    binary_mapping[polyuria]
                ],

                "Polydipsia": [
                    binary_mapping[polydipsia]
                ],

                "sudden weight loss": [
                    binary_mapping[
                        sudden_weight_loss
                    ]
                ],

                "weakness": [
                    binary_mapping[weakness]
                ],

                "Polyphagia": [
                    binary_mapping[polyphagia]
                ],

                "Genital thrush": [
                    binary_mapping[
                        genital_thrush
                    ]
                ],

                "visual blurring": [
                    binary_mapping[
                        visual_blurring
                    ]
                ],

                "Itching": [
                    binary_mapping[itching]
                ],

                "Irritability": [
                    binary_mapping[irritability]
                ],

                "delayed healing": [
                    binary_mapping[
                        delayed_healing
                    ]
                ],

                "partial paresis": [
                    binary_mapping[
                        partial_paresis
                    ]
                ],

                "muscle stiffness": [
                    binary_mapping[
                        muscle_stiffness
                    ]
                ],

                "Alopecia": [
                    binary_mapping[alopecia]
                ],

                "Obesity": [
                    binary_mapping[obesity]
                ]
            }
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        try:

            prediction = model.predict(
                patient_data
            )[0]

            probabilities = model.predict_proba(
                patient_data
            )[0]

            negative_probability = (
                probabilities[0] * 100
            )

            positive_probability = (
                probabilities[1] * 100
            )

            # =================================================
            # RESULT
            # =================================================

            if prediction == 1:

                st.html(f"""
                <div class="positive-card">

                    <div class="prediction-label">
                        Prediction
                    </div>

                    <div class="prediction-value"
                         style="color:#f87171;">
                        Positive
                    </div>

                    <div style="
                        color:#94a3b8;
                        margin-top:5px;
                    ">
                        Estimated diabetes probability
                    </div>

                </div>
                """)

            else:

                st.html(f"""
                <div class="negative-card">

                    <div class="prediction-label">
                        Prediction
                    </div>

                    <div class="prediction-value"
                         style="color:#4ade80;">
                        Negative
                    </div>

                    <div style="
                        color:#94a3b8;
                        margin-top:5px;
                    ">
                        Estimated diabetes probability
                    </div>

                </div>
                """)

            # =================================================
            # GAUGE
            # =================================================

            fig = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=positive_probability,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 34
                        }
                    },

                    title={
                        "text":
                            "Diabetes Probability",
                        "font": {
                            "size": 16
                        }
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "thickness": 0.25
                        },

                        "steps": [

                            {
                                "range": [0, 40]
                            },

                            {
                                "range": [40, 70]
                            },

                            {
                                "range": [70, 100]
                            }
                        ],

                        "threshold": {

                            "line": {
                                "width": 4
                            },

                            "value":
                                positive_probability
                        }
                    }
                )
            )

            fig.update_layout(

                height=280,

                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=10
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                font={
                    "color": "#f8fafc"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )

            # =================================================
            # PROBABILITY CARDS
            # =================================================

            metric1, metric2 = st.columns(2)

            with metric1:

                st.html(f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Positive Probability
                    </div>

                    <div class="metric-value">
                        {positive_probability:.2f}%
                    </div>

                </div>
                """)

            with metric2:

                st.html(f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Negative Probability
                    </div>

                    <div class="metric-value">
                        {negative_probability:.2f}%
                    </div>

                </div>
                """)

            # =================================================
            # MODEL INFORMATION
            # =================================================

            st.html("<br>")

            st.html("""
            <div class="card">

                <div class="card-title">
                    Model Information
                </div>

                <div class="card-subtitle">
                    Final selected model
                </div>

            </div>
            """)

            info1, info2 = st.columns(2)

            with info1:

                st.html("""
                <div class="metric-card">

                    <div class="metric-title">
                        Model
                    </div>

                    <div class="metric-value">
                        Random Forest
                    </div>

                </div>
                """)

            with info2:

                st.html("""
                <div class="metric-card">

                    <div class="metric-title">
                        ROC-AUC
                    </div>

                    <div class="metric-value">
                        97.77%
                    </div>

                </div>
                """)

            # =================================================
            # PATIENT DATA
            # =================================================

            st.html("<br>")

            with st.expander(
                "View Patient Data"
            ):

                st.dataframe(
                    patient_data,
                    use_container_width=True
                )

            # =================================================
            # MEDICAL DISCLAIMER
            # =================================================

            st.warning(
                "This prediction is for educational "
                "and demonstration purposes only. "
                "It is not a medical diagnosis."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.warning(
                "Make sure the model was trained using "
                "the same feature order and encoding "
                "used in this application."
            )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    Diabetes Prediction System

    <br>

    Machine Learning Classification Project

    <br><br>

    Developed by
    <b>Mohamed Gamal Zamel</b>

</div>
""")