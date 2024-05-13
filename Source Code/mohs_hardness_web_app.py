import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = pickle.load(open("C:/Users/Hehe/Desktop/BSc_SE_CIS6005_CIS_st20284593/DT_model.pk", 'rb'))

# Load the scaler 
scaler = pickle.load(open("C:/Users/Hehe/Desktop/BSc_SE_CIS6005_CIS_st20284593/scaler.pk", 'rb'))

def hardness_prediction(input_data):
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform([input_data])
    
    # Get the prediction from the model
    prediction = model.predict(input_data_scaled)

    return f"Mohs Hardness is: {prediction[0]}"

def main():
    st.title('Mohs Hardness Prediction')

    # Collecting user inputs using st.text_input and converting them to floats
    density_Total = st.text_input('Please enter Total elemental density')
    val_e_Average = st.text_input('Please enter Atomic average number of valence electrons')
    atomicweight_Average = st.text_input('Please enter Atomic average number of electrons')
    ionenergy_Average = st.text_input('Please enter Average Ionization Energy')
    el_neg_chi_Average = st.text_input('Please enter Average electronegativity')
    R_vdw_element_Average = st.text_input('Please enter Atomic average van der Waals atomic radius')
    R_cov_element_Average = st.text_input('Please enter Atomic average covalent atomic radius')
    zaratio_Average = st.text_input('Please enter Atomic average atomic number to mass number ratio')
    density_Average = st.text_input('Please enter Atomic average elemental density')

    # Prediction button
    if st.button("Mohs Hardness Prediction"):
        try:
            # Convert inputs to float and aggregate them
            inputs = [float(density_Total), float(val_e_Average), float(atomicweight_Average),
                      float(ionenergy_Average), float(el_neg_chi_Average), float(R_vdw_element_Average),
                      float(R_cov_element_Average), float(zaratio_Average), float(density_Average)]

            # Make prediction
            prediction = hardness_prediction(inputs)
            st.success(prediction)
        except ValueError:
            st.error("Please ensure all inputs are numeric.")

if __name__ == '__main__':
    main()