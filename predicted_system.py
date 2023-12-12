import numpy as np
import streamlit as st
import pickle 

st.set_page_config(page_title='Diabetic Prediction Web App', page_icon=':moneybag', layout='centered')

loaded_model = pickle.load(open('./trained_model1.sav', 'rb'))
#creating a function for prediction

input_data = (1, 10.0, 0, 0, 0, 22.68, 6.6,100)
input_data_numpy = np.asarray(input_data).reshape(1,-1)

# Make predictions
predictions = loaded_model.predict(input_data_numpy)

if predictions == 0:
  print("The person is not diabetic")

else:
  print("The person is diabetic")

