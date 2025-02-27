import streamlit as st
import pandas as pd
import pickle

st.title("Deploy Your Machine Learning Model 🚀")

# Upload a trained model
uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

if uploaded_model:
    # Load the model using pickle
    with open("models'best_classification_model.pkl", "wb") as f:
        f.write(uploaded_model.getbuffer())

    with open("models'best_classification_model.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("Model uploaded and loaded successfully!")

    # Dynamically ask for feature inputs
    st.subheader("Enter Feature Values")

    # Get feature names from the model
    feature_names = model.feature_names_in_  # Works with scikit-learn models
    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert input to DataFrame
    if st.button("Predict"):
        try:
            # Convert user inputs into DataFrame
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df.astype(float)  # Convert input values to numeric

            # Make predictions
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
