import streamlit as st
import pandas as pd
from joblib import load 
import os


@st.cache_data
def  read_data():
    return pd.read_csv('https://raw.githubusercontent.com/NUELBUNDI/Machine-Learning-Data-Set/main/diabetes.csv')


@st.cache_resource
def model_load():
    loaded_models , loaded_model_results = load('models_metadata.pkl')
    return loaded_models , loaded_model_results



def predict_model(data):
    predictions = {}

    loaded_models, loaded_model_results = model_load()
    for model, result in zip(loaded_models, loaded_model_results):
        model_name = result['model_name']
        y_predict  = model.predict(data)
        y_predict_proba = model.predict_proba(data)[:, 1]  # Probability of positive class (class 1)

        predictions[model_name] = {'prediction': y_predict, 'probability': y_predict_proba[0]}

    return predictions

def model_category_using_prediction(predictions_dict,thershold):
    
    if predictions_dict['probability'] > float(thershold):
        return 'Diabetic'
    else:
        return 'Non-Diabetic'
    
def model_category_using_y_preds(y_preds):
    
    if y_preds == 0:
        return 'Non-Diabetic'
    else:
        return 'Diabetic'
    
    


# Create a button to download the model
def download_objects(model_path):
    
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Click to download",
        data=model_bytes,
        file_name=os.path.basename(model_path),
        mime="application/octet-stream"
    )
    
