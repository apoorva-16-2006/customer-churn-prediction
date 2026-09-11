import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #8b8b8b;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    svm_model = joblib.load(
        os.path.join(base_dir, "SVM_pipeline.pkl")
    )

    decision_tree_model = joblib.load(
        os.path.join(base_dir, "decision_Tree_pipeline.pkl")
    )

    random_forest_model = joblib.load(
        os.path.join(base_dir, "random_forest_pipeline.pkl")
    )

    return (
        svm_model,
        decision_tree_model,
        random_forest_model
    )


try:

    (
        svm_model,
        decision_tree_model,
        random_forest_model
    ) = load_models()

except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.code("""
SVM_pipeline.pkl
decision_Tree_pipeline.pkl
random_forest_pipeline.pkl
""")

    st.error(f"Error: {e}")

    st.stop()


st.markdown(
    '<div class="main-title">'
    '📊 Customer Churn Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Predict customer churn using SVM, Decision Tree '
    'and Random Forest'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">'
    '👤 Customer Information'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=120,
        value=24,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber Optic",
            "No",
            "unknown"
        ]
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0,
        step=10.0
    )

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Month-to-Month",
            "One-Year",
            "Two-Year"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )


input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "InternetService": [internet_service],
    "TotalCharges": [total_charges],
    "ContractType": [contract_type],
    "TechSupport": [tech_support]
})


def convert_prediction(prediction):

    if isinstance(prediction, str):

        prediction = prediction.strip().lower()

        if prediction in ["yes", "churn", "1"]:
            return "Churn"

        return "Non-Churn"

    if int(prediction) == 1:
        return "Churn"

    return "Non-Churn"


def predict_model(model, data):

    prediction = model.predict(data)[0]

    return convert_prediction(prediction)


st.write("")

predict_button = st.button(
    "🔮 Predict Customer Churn",
    width="stretch",
    type="primary"
)


if predict_button:

    try:

        svm_prediction = predict_model(
            svm_model,
            input_data
        )

        decision_tree_prediction = predict_model(
            decision_tree_model,
            input_data
        )

        random_forest_prediction = predict_model(
            random_forest_model,
            input_data
        )

        predictions = [
            svm_prediction,
            decision_tree_prediction,
            random_forest_prediction
        ]

        churn_votes = predictions.count("Churn")

        non_churn_votes = predictions.count("Non-Churn")


        if churn_votes >= 2:
            final_prediction = "Churn"
        else:
            final_prediction = "Non-Churn"


        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 Final Prediction'
            '</div>',
            unsafe_allow_html=True
        )


        if final_prediction == "Churn":

            st.error(
                f"⚠️ CUSTOMER IS LIKELY TO CHURN\n\n"
                f"Churn Votes: {churn_votes}/3"
            )

        else:

            st.success(
                f"✅ CUSTOMER IS UNLIKELY TO CHURN\n\n"
                f"Non-Churn Votes: {non_churn_votes}/3"
            )


        st.markdown(
            '<div class="section-title">'
            '🗳️ Majority Voting'
            '</div>',
            unsafe_allow_html=True
        )


        vote_col1, vote_col2 = st.columns(2)


        with vote_col1:

            st.metric(
                "🔴 Churn Votes",
                f"{churn_votes}/3"
            )


        with vote_col2:

            st.metric(
                "🟢 Non-Churn Votes",
                f"{non_churn_votes}/3"
            )


        st.write("Churn Voting Strength")

        st.progress(
            churn_votes / 3
        )


        st.markdown(
            '<div class="section-title">'
            '🤖 Individual Model Predictions'
            '</div>',
            unsafe_allow_html=True
        )


        model_col1, model_col2, model_col3 = st.columns(3)


        with model_col1:

            st.subheader("SVM")

            if svm_prediction == "Churn":
                st.error("🔴 CHURN")
            else:
                st.success("🟢 NON-CHURN")

            st.caption(
                "Support Vector Machine"
            )


        with model_col2:

            st.subheader("Decision Tree")

            if decision_tree_prediction == "Churn":
                st.error("🔴 CHURN")
            else:
                st.success("🟢 NON-CHURN")

            st.caption(
                "Decision Tree Classifier"
            )


        with model_col3:

            st.subheader("Random Forest")

            if random_forest_prediction == "Churn":
                st.error("🔴 CHURN")
            else:
                st.success("🟢 NON-CHURN")

            st.caption(
                "Random Forest Classifier"
            )


        st.info(
            "📌 **Decision Rule:** "
            "2 or more Churn votes → Final result: Churn. "
            "Otherwise → Final result: Non-Churn."
        )


        with st.expander(
            "📋 View Customer Information"
        ):

            st.dataframe(
                input_data,
                width="stretch",
                hide_index=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.error(
            f"Error: {e}"
        )


st.divider()

st.caption(
    "Customer Churn Prediction | "
    "SVM + Decision Tree + Random Forest | "
    "Majority Voting"
)