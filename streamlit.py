import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_PATH = r"C:\Users\ACER\Desktop\stacking_logistic_regression_model.pkl"
SCALER_PATH = r"C:\Users\ACER\Desktop\scaler.pkl"
DATA_PATH = r"C:\Users\ACER\Desktop\combine.xlsx"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_data():
    data = pd.read_excel(DATA_PATH)
    target = 'label'
    feature_cols = [col for col in data.columns if col != target]
    categorical_features = ['Tumour Burden', 'SVI']
    numeric_features = [col for col in feature_cols if col not in categorical_features]
    return data, feature_cols, categorical_features, numeric_features

model, scaler = load_model_and_scaler()
data, feature_cols, categorical_features, numeric_features = load_data()

st.title("CRPC risk prediction and SHAP interpretation")

st.markdown("Please fill in the feature values. Categorical features should be entered in coded form, and numerical features should be entered as numerical values.")

cols = st.columns(2)
input_dict = {}
for idx, feature in enumerate(feature_cols):
    col = cols[idx % 2]
    with col:
        if feature in categorical_features:
            if feature == 'SVI':
                val = st.selectbox(f"{feature}", options=[0,1,2], key=feature)
            else:
                val = st.selectbox(f"{feature}", options=[0,1], key=feature)
        else:
            val = st.number_input(f"{feature}", format="%.5f", key=feature)
        input_dict[feature] = val

label_map = {0: "very high-risk(< 1y)", 1: "high-risk(1 - 4y)", 2: "low-risk(> 4y)"}

if st.button("Predict and explain"):

    input_df = pd.DataFrame([input_dict])
    input_num = input_df[numeric_features].values
    input_num_scaled = scaler.transform(input_num)
    input_cat = input_df[categorical_features].values
    input_processed = np.hstack([input_num_scaled, input_cat])

    pred = model.predict(input_processed)
    pred_proba = model.predict_proba(input_processed)
    pred_label = label_map.get(pred[0], "未知类别")

    st.write("CRPC Risk：", pred_label)
    st.write("probability：")
    for i, p in enumerate(pred_proba[0]):
        st.write(f"{label_map.get(i, f'类别{i}')} : {p:.4f}")

    st.markdown(" SHAP Explain")
    background_df = data.sample(min(100, len(data)), random_state=42)
    background_num = background_df[numeric_features].values
    background_num_scaled = scaler.transform(background_num)
    background_cat = background_df[categorical_features].values
    background_processed = np.hstack([background_num_scaled, background_cat])

    feature_names = numeric_features + categorical_features

    background_pd = pd.DataFrame(background_processed, columns=feature_names)
    input_pd = pd.DataFrame(input_processed, columns=feature_names)

    with st.spinner("wait please..."):
        explainer = shap.KernelExplainer(model.predict_proba, background_pd)
        shap_values = explainer.shap_values(input_pd)


    class_idx = pred[0]
    shap_vals = shap_values[0,:,class_idx].reshape(1,-1)

    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value[class_idx],
        data=input_pd,
        feature_names=feature_names
    )

    st.write(f"SHAP waterfall plot")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    shap.plots.waterfall(shap_exp[0], show=False)
    st.pyplot(fig2)
    plt.close(fig2)

    st.write(f"features important")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    abs_vals = np.abs(shap_vals[0])
    sorted_idx = np.argsort(abs_vals)[::-1]
    sorted_vals = shap_vals[0][sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    ax3.barh(sorted_features, sorted_vals)
    ax3.invert_yaxis()
    ax3.set_xlabel("SHAP Value")
    st.pyplot(fig3)
    plt.close(fig3)
