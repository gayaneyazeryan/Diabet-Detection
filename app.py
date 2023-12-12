import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title='Diabetic Prediction Web App', page_icon=':moneybag', layout='centered')


loaded_model = pickle.load(open('./trained_model1.sav', 'rb'))


def diabetes_prediction(input_data):
    # creating a function for prediction
    input_data_numpy = np.asarray(input_data).reshape(1, -1)

    # Make predictions
    predictions = loaded_model.predict(input_data_numpy)

    if predictions == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"





def main():

    html_temp = """
    <div style ="background-color:#2a3990;padding:15px">
    <h2 style ="color:#edf7f7;text-aling:center;">Diabet Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # giving a title
    # st.title("Diabetic Prediction Web App")

    mapping_1 = {0 :'Female', 1: 'Male'}
    mapping_2 = { 1: 'Yes', 0: 'No'}
    mapping_3 = {0: 'No info', 2: 'Ever',3: 'Former', 4: 'Never'}
    
    def format_func1(option):
        return mapping_1[option]
    
    # getting the input data from the user 
    gender_options = list(mapping_1.keys())
    gender = st.selectbox('Gender of the Person', gender_options, format_func=format_func1)
    
    age = st.text_input("Age of the Person")

    def format_func2(option):
        return mapping_2[option]
    
    hypertension_options = list(mapping_2.keys())
    hypertension = st.selectbox("Does the Person have hypertension", hypertension_options, format_func=format_func2)
    
    heart_disease_options = list(mapping_2.keys())
    heart_disease = st.selectbox("Does the Person have heart disease", heart_disease_options, format_func=format_func2)
    # Create a list of options for smoking history

    def format_func3(option):
        return mapping_3[option]
    smoking_history_options = list(mapping_3.keys())
    smoking_history = st.selectbox("Smoking history of the Person", smoking_history_options, format_func=format_func3)
    
    bmi = st.text_input("BMI value")
    HbA1c_level = st.text_input("HbA1c level")
    blood_glucose_level = st.text_input("Glucose level in blood")
    # code for prediction 
     
    result = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        result = diabetes_prediction([gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level])
    
        if result == 'The person is diabetic':
            st.error(f'{result}\n ## Sorry for bad news :sob:')
            st.image('111.jpg')

        else:
            st.success(f'{result}\n ## Congratulations :doughnut:')
            st.image('333.jpg')
            st.balloons()
            

if __name__ == '__main__':
    main()