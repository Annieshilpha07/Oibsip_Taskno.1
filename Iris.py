import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set up page configuration for Streamlit
def main():
    st.set_page_config(
        page_title='Iris Flower Classification',
        page_icon='🌸',
        initial_sidebar_state='expanded',
        layout='wide',
        menu_items={"about": 'This Streamlit application is developed for Iris Flower Classification using Machine Learning.'}
    )

    # Display the page title at the top of your app
    st.title(':rainbow[Iris Flower Classification]')

    # Set up the sidebar with option menu
    selected = option_menu("Iris Flower Classification | Data Exploration and Model Prediction",
                            options=["Home", "Get Prediction"],
                            icons=["house", "lightbulb"],
                            default_index=1, menu_icon="globe",
                            orientation="horizontal")

    # Set up the information for 'Home' menu
    if selected == 'Home':
        title_text = '''<h1 style='font-size: 30px;text-align: center; color:grey;'>Iris Flower Species Classification</h1>'''
        st.markdown(title_text, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1.5], gap="large")
        with col1:
            st.markdown("### :red[Skills Takeaway]:")
            st.markdown('<h5> Data Preprocessing, Model Training, Model Deployment </h5>', unsafe_allow_html=True)

            st.markdown("### :red[Overview]:")
            st.markdown('''<h5>
                            <li> Used Iris dataset for classifying flower species into three classes (Setosa, Versicolor, Virginica) using ML models,<br>
                            <li> Processed and cleaned data for machine learning,<br>
                            <li> Developed a classification model (Logistic Regression),<br>
                            <li> Developed a web application for predictions and exploration.<br>
                        </h5>''', unsafe_allow_html=True)

            st.markdown("### :red[Problem Statement]:")
            st.markdown('<h5> ' \
            'The Iris Flower Dataset contains three species — Setosa, Versicolor, and Virginica — each identified by distinct measurements.' \
            ' We build machine learning models that learn from these measurements and predict the flower species automatically. ' \
            'This project tackles a multi-class classification task by training, comparing, selecting, and deploying ML models.</h5>', unsafe_allow_html=True)

        with col2:
            st.image("https://miro.medium.com/v2/resize:fit:700/1*uo6VfVH87jRjMZWVdwq3Vw.png",use_container_width=True)

    # User input values for model prediction
    if selected == "Get Prediction":
        st.write('')
        title_text = '''<h2 style='font-size: 32px;text-align: center;color:grey;'>Iris Flower Prediction</h2>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.markdown("<h5 style=color:orange>To Predict the Iris Flower Species, Please Provide the Following Information:", unsafe_allow_html=True)
        st.write('')

        # Form to get the user input
        with st.form('prediction'):
            col1, col2 = st.columns(2)
            with col1:
                sepal_length = st.number_input(label='Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
                sepal_width = st.number_input(label='Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)

            with col2:
                petal_length = st.number_input(label='Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
                petal_width = st.number_input(label='Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.3)

            submit_button = st.form_submit_button('PREDICT SPECIES', use_container_width=True)

        if submit_button:
            with st.spinner("Predicting..."):
                # Load the pre-trained logistic regression model from the pickle file
                with open('logistic_reg.pkl', 'rb') as file:
                    model = pickle.load(file)
                
                # Make prediction using the model
                prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
                
                # Load your iris.csv dataset
                iris_data = pd.read_csv('Iris.csv')  # Update with the correct path
                
                
                # Directly use the predicted species as the label
                species = prediction[0]  # prediction is a string, e.g. 'Iris-setosa'
                
                # Display the prediction
                st.subheader(f"Predicted Iris Species: :green[{species}]")

if __name__ == "__main__":
    main()
