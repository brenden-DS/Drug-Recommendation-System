import pandas as pd
import streamlit as st
import pickle
import numpy as np



model = pickle.load(open('drug_rf.sav','rb'))

st.image('meds.jpg')

st.title('Drug Recommendation System')
st.write('This is a drug recommendation app. To use it, simply enter the user input & click on the "RECOMMEND", button to get a recommendation')

def drug_recommendation(features):
    features = np.array(features).reshape(1,-1)
    drug_recommendation= model.predict(features)

    return drug_recommendation[0]

st.sidebar.header('Enter your details')
age = st.sidebar.number_input('Age',min_value=1,max_value=80,value=12)
sex = st.sidebar.selectbox('Sex',['F','M'])
bp = st.sidebar.selectbox('BP',['LOW','NORMAL','HIGH'])
cholesterol = st.sidebar.selectbox('Cholesterol',['NORMAL','HIGH'])
na_to_k = st.sidebar.number_input('Sodium to Potassium Ratio',min_value=5.0,max_value=40.0,value=15.0)

sex_map = {'F': 0, 'M': 1}
sex = sex_map[sex]

bp_map = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}
bp = bp_map[bp]

cholesterol_map = {'NORMAL': 0, 'HIGH': 1}
cholesterol = cholesterol_map[cholesterol]

features= pd.DataFrame({'age':[age],
                        'sex':[sex],
                        'bp':[bp],
                        'cholesterol':[cholesterol],
                        'na_to_k':[na_to_k]})

st.write(features)  # Add this line to check the user inputs

if st.button('RECOMMEND'):
    #st.write(f'User inputs: Age={age}, Sex={sex}, BP={bp}, Cholesterol={cholesterol}, Na_to_K={na_to_k}')  # Add this line to check the user inputs

    drug_recommendation = model.predict(features)[0]
    st.write(f'RECOMMENDED DRUG: {drug_recommendation}')


