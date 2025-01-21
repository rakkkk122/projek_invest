import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = pickle.load(open('model_uas.pkl', 'rb'))

# Set page title and description
st.title('Insurance Premium Prediction')
st.markdown("<h4 style='text-align:center;color:#002966'>By rakha sidi hatta - 2021230013</h4>", unsafe_allow_html=True)
st.write("<br>",unsafe_allow_html=True)
st.write('masukan detail dibawah untuk menghitung biaya asuransi')
st.markdown("<style>h1 {color: red;text-align:center}</style>", unsafe_allow_html=True)

# Create input fields
age = st.number_input('Age', min_value=18, max_value=64, value=25)
sex = st.selectbox('Sex', ['Female', 'Male'])
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
smoker = st.selectbox('Smoker', ['No', 'Yes'])

# Convert categorical inputs to numeric
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0

# Create a button for prediction
if st.button('Predict Premium'):
    # Create input array
    X = np.array([age, sex, bmi, children, smoker])
    X = X.reshape(1, -1)
    
    # Standardize the input
    scaler = StandardScaler()
    # Fit scaler on the same range as training data
    scaler.fit([[18, 0, 15.96, 0, 0],
                [64, 1, 49.06, 5, 1]])
    X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)
    
    # Display result
    st.success(f'Predicted Insurance Premium: ${prediction[0]:,.2f} per month')

# Add some information about the inputs
st.markdown("""
### Input Information:
- **Age**: Usia harus antara 18 dan 64 tahun
- **Sex**: Laki-laki atau Perempuan
- **BMI**: Indeks Massa Tubuh (berat badan dalam kg / tinggi badan dalam meter kuadrat)
- **Children**: Jumlah anak yang tergantung
- **Smoker**: Apakah orang tersebut merokok atau tidak
""")