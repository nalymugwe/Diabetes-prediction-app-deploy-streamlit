import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import streamlit.components.v1 as components



from data import *
from plot_fun import *


# Page configuration

st.set_page_config(page_title="Diabetes Prediction Tool", page_icon="🥗", layout="wide")


note_book_path = 'Notebook.html'
model_path     = 'models_metadata.pkl'


with st.sidebar:
    selected = option_menu(None, ["Home","EDA", "Models", "Prediction",'Interpretation'], 
    icons     =['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon ="cast", default_index=0, orientation="vertical",
    
    styles={
    "icon"              : {"color": "orange", "font-size": "16px"}, 
    "nav-link"          : {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected" : {"background-color": "green"},
    })
    
    
if selected == "EDA":
    st.header(":orange[Diabetes Prediction] Tool Exploratory Analysis",divider=True,)
    st.markdown("---")

    renderer = get_pyg_renderer()
    renderer.render_explore()

    
if selected == "Models":
    st.subheader('Trained Models Information',divider=True)
    
    with st.expander('Training Data'):
        st.dataframe(read_data())
    loaded_models , loaded_model_results = model_load()
    df = pd.DataFrame(loaded_model_results)    
    col9, col10 = st.columns(2)
    with col9:
        with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'accuracy_score'),use_container_width=True)
    with col10:
        with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'f1_score'),use_container_width=True)
            
    with st.expander("Learn how the model was trained?",expanded=False):
        
        with open(note_book_path, 'r',encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=1000, width=800, scrolling=True)
    
    # Download Model
    st.sidebar.markdown("### Download")
    download_choice=st.sidebar.selectbox(label='Select what to download 👇',options=["Serialized Model","Notebook"])
    
    if download_choice=='Serialized Model':
        download_objects(model_path)
    if download_choice=='Notebook':
        download_objects(note_book_path)
        
    
if selected == "Prediction":
    
    st.header(":blue[Prediction Page]",divider=True)
    st.markdown(" ")
    
    with st.form('my_form',border=True):
        col1, col2, col3 = st.columns(3)
    
        with col1:
            pregant        = st.selectbox("How many types have you been pregant?",options=range(0,20))
        with col2:
            glucose        = st.number_input("What is your glucose level?")
        with col3:
            blood_pressure = st.number_input("What is your blood pressure?")
        
        col4, col5,col6 = st.columns(3)
        with col4:
            skin_thickness  = st.number_input("What is your skin thickness?")
        with col5:
            insuline        = st.number_input("What is your insuline Level?")
        with col6:
            bmi             = st.number_input("What is your BMI?")
        
        col7 , col8 = st.columns(2)
        
        with col7:
            age              = st.slider("What is your age?" , 0,120,18)
        with col8:
            dpf              = st.selectbox("What is your Diabetes Pedigree Function?", ["1.0", "1.5", "2.0", "2.5", "3.0"])
            
        submited = st.form_submit_button(label='Predict')
        
    if submited:
        # check if all paramaters have been checked
        list_of_params  = [pregant, glucose, blood_pressure, skin_thickness, insuline, bmi, dpf, age]
        
        if list_of_params is not None:
            list_of_params = [float(i) for i in list_of_params]
            
            # create a dataframe
            data = pd.DataFrame(
                {
                    "Pregnancies": [pregant],
                    "Glucose": [glucose],
                    "BloodPressure": [blood_pressure],
                    "SkinThickness": [skin_thickness],
                    "Insulin": [insuline],
                    "BMI": [bmi],
                    "DiabetesPedigreeFunction": [dpf],
                    "Age": [age],
                    
                })
            # st.dataframe(data)
            
            # st.write(predict_model(data))
            st.markdown("#### Predictions Results")
            col11, col12, col13 = st.columns(3)
            
            predict_results = predict_model(data)
            
            col11.warning(list(predict_results.keys())[0])
            col11.success(model_category_using_y_preds(predict_results['LogisticRegression']['prediction'][0]))
            col11.markdown(':grey[Probability of Diabetes] ')
            col11.info(round((predict_results['LogisticRegression']['probability'])* 100,2))
            
            col12.warning(list(predict_results.keys())[1])
            col12.success(model_category_using_y_preds(predict_results['RandomForestClassifier']['prediction'][0]))
            col12.markdown(':grey[Probability of Diabetes] ')
            col12.info(round((predict_results['RandomForestClassifier']['probability'])* 100,2))
            
            col13.warning(list(predict_results.keys())[2])
            col13.success(model_category_using_y_preds(predict_results['DecisionTreeClassifier']['prediction'][0]))
            col13.markdown(':grey[Probability of Diabetes] ')
            col13.info(round((predict_results['DecisionTreeClassifier']['probability'])* 100,2))
        else:
            st.warning('Please fill all the fields')

    else:
        st.warning("Please fill the form to Predict the chance of Diabetes")
        
        
if selected == "Interpretation":
    data   = read_data()
    st.markdown("### :brown[Explainable AI]")
    st.markdown("---")
    
    data_instance = st.sidebar.selectbox("Select a Data Instance",options= data.index.to_list())
    st.data_editor(data,use_container_width=True,height=250)
    
    if data_instance:  
        data_picked= data.loc[[data_instance]]
        st.write('Data Instance Selected')
        st.data_editor(data_picked,use_container_width=True,)
        
        on = st.toggle("Show Interpretability")
        if on:
            with st.container(border=True):
                components.html(lime_explainer(read_data(),12), height=800, width=900, scrolling=True)




                