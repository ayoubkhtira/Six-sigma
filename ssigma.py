import pandas as pd
import numpy as np
import streamlit as st

# Table des valeurs de d2 en fonction du nombre de pièces et d'opérateurs
d2_table = {
    2: {2: 1.41, 3: 1.91, 4: 2.24, 5: 2.48, 6: 2.67, 7: 2.83, 8: 2.96, 9: 3.08, 10: 3.18},
    3: {2: 1.28, 3: 1.81, 4: 2.15, 5: 2.40, 6: 2.60, 7: 2.77, 8: 2.91, 9: 3.02, 10: 3.13},
    4: {2: 1.23, 3: 1.77, 4: 2.12, 5: 2.38, 6: 2.58, 7: 2.75, 8: 2.89, 9: 3.01, 10: 3.11},
    5: {2: 1.21, 3: 1.75, 4: 2.11, 5: 2.37, 6: 2.57, 7: 2.74, 8: 2.88, 9: 3.00, 10: 3.10},
    6: {2: 1.19, 3: 1.74, 4: 2.10, 5: 2.36, 6: 2.56, 7: 2.78, 8: 2.87, 9: 2.99, 10: 3.10},
    # Ajout pour plus de lignes si nécessaire
}

# Function to calculate the R&R
def calculate_rr(df, num_operators, num_measurements, confidence_coefficient):
    # Retrieve d2 value from the table based on number of pieces and operators
    d2 = d2_table.get(num_operators, {}).get(num_measurements, 1.41)  # Default to a value if not found
    
    # Validate dimensions
    if df.shape[1] < num_operators * num_measurements + 1:
        st.error(f"Les données du fichier ne contiennent pas assez de colonnes pour {num_operators} opérateurs et {num_measurements} mesures.")
        return None, None, None, None
    
    # Extract the measurements (excluding the first column)
    measurements = df.iloc[:, 1:num_operators * num_measurements + 1]
    
    # Calculate repeatability (within operator variance)
    repeatability = measurements.apply(lambda x: np.std(x), axis=1).mean()
    
    # Calculate reproducibility (between operator variance)
    reproducibility = measurements.apply(lambda x: np.mean(x), axis=0).std()
    
    # Total variance
    total_variance = measurements.stack().std()
    
    # Calculate %R&R, %EV, %AV
    rr = repeatability / total_variance * 100
    ev = reproducibility / total_variance * 100
    av = (repeatability + reproducibility) / total_variance * 100
    vp = 100 - rr - ev - av
    
    # Adjust the results based on the confidence coefficient and d2 value
    rr_adjusted = rr * d2 * confidence_coefficient
    ev_adjusted = ev * d2 * confidence_coefficient
    av_adjusted = av * d2 * confidence_coefficient
    vp_adjusted = vp * d2 * confidence_coefficient
    
    return rr_adjusted, ev_adjusted, av_adjusted, vp_adjusted

# Streamlit interface
st.title('Calcul Professionnel de Gage R&R')

# Upload file
uploaded_file = st.file_uploader("Téléchargez un fichier Excel avec les mesures", type=["xlsx"])

if uploaded_file is not None:
    # Load the data from the uploaded file
    df = pd.read_excel(uploaded_file, sheet_name='Feuil1')
    
    # Show data
    st.write("Données des mesures des opérateurs :")
    st.dataframe(df)
    
    # User Inputs for customization
    st.sidebar.header("Paramètres de l'étude R&R")
    
    num_operators = st.sidebar.slider("Nombre d'opérateurs", min_value=2, max_value=6, value=3)
    num_measurements = st.sidebar.slider("Nombre de mesures par opérateur", min_value=2, max_value=10, value=3)
    num_pieces = st.sidebar.slider("Nombre de pièces", min_value=5, max_value=20, value=10)
    confidence_level = st.sidebar.slider("Coefficient de niveau de confiance", min_value=1.0, max_value=10.0, value=5.15, step=0.01)
    
    st.sidebar.write(f"Vous avez sélectionné : {num_operators} opérateurs, {num_measurements} mesures, et {num_pieces} pièces.")
    st.sidebar.write(f"Coefficient de niveau de confiance : {confidence_level:.2f}")
    
    # Perform calculation if data is valid
    if df.shape[0] >= num_pieces and df.shape[1] >= num_operators * num_measurements + 1:
        rr, ev, av, vp = calculate_rr(df, num_operators, num_measurements, confidence_level)
        
        if rr is not None:
            # Display the results
            st.subheader("Résultats de l'analyse R&R")
            st.write(f"**%R&R :** {rr:.2f}")
            st.write(f"**%EV :** {ev:.2f}")
            st.write(f"**%AV :** {av:.2f}")
            st.write(f"**%Vp :** {vp:.2f}")
    else:
        st.error("Les données dans le fichier sont insuffisantes pour correspondre aux paramètres sélectionnés.")
