import numpy as np
import pickle
import streamlit as st

st.title('DIABETES PREDICTION')

# Load the model
loaded_model = pickle.load(open('Diabetesmodel.pkl', 'rb'))

def Disease(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_reshape)

    if prediction[0] == 0: 
        return st.success('The person does not have Diabetes')
    else:
        return st.error('The person has Diabetes')

def main():
    st.write("Prediction Model")

    BMI = st.number_input("Enter Body Mass Index")
    Insulin = st.number_input("Enter Insulin", step=2)
    Glucose = st.number_input("Enter Glucose", step=2)
    Age = st.number_input("Enter your Age", step=2)
    SkinThickness=st.number_input("Enter thickness",step=2)
    DiabetesPedigreeFunction=st.number_input("Enter Pedigree Function", step=2)
    Pregnancies=st.number_input("Enter number",step=2)
    Outcome=st.number_input("Enter your number",step=2)

    diagnosis = ""

    if st.button('PREDICT'):
        diagnosis = Disease([Glucose, Insulin, BMI, Age,SkinThickness,DiabetesPedigreeFunction,Pregnancies,Outcome])

if __name__ == '__main__':
    main()
