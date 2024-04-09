import streamlit as st
import pandas as pd
from joblib import load 
import os
import numpy as np

from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from lime import lime_tabular


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
        y_predict_proba = model.predict_proba(data)[:, 1] 

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
    
    
    
def get_pyg_renderer():
    df           = pd.read_csv("https://raw.githubusercontent.com/NUELBUNDI/Machine-Learning-Data-Set/main/diabetes.csv")
    df['Outcome'] = np.where(df['Outcome']== 0, 'Non-Diabetic', 'Diabetic')
    return StreamlitRenderer(df,default_tab='data',theme_key='vega',dark='dark')


def train_test(df):
    x = df.drop(columns=['Outcome'])
    Y = df['Outcome']
    x_train , x_test , y_train , y_test = train_test_split(x,Y , test_size=0.3)
    return x_train, x_test, y_train, y_test ,x 
    
def predict_fn(x):
    loaded_models, loaded_model_results = model_load()
    for model, result in zip(loaded_models, loaded_model_results):
        if result['model_name']== 'LogisticRegression':
            model = model.predict_proba(x)
    return model

@st.cache_resource
def lime_explainer(df,instance_index):

    x_train, x_test, y_train, y_test ,x = train_test(df)
    
    explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(x_train),
                                                    feature_names= x.columns.tolist(),
                                                    class_names=['No Diabets','Diabets'],
                                                    mode='classification',
                                                    random_state=42)
    

    instance    = x_test.iloc[[int(instance_index)]]
    explanation = explainer.explain_instance(instance.values[0], predict_fn, num_features=len(x.columns))
    html        = explanation.as_html()
    
    return html
