import pandas as pd
import numpy as np
import streamlit as st

# Function to calculate the R&R
def calculate_rr(df, num_operators, num_measurements):
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
    
    return rr, ev, av, vp

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
    
    num_operators = st.sidebar.slider("Nombre d'opérateurs", min_value=2, max_value=5, value=3)
    num_measurements = st.sidebar.slider("Nombre de mesures par opérateur", min_value=2, max_value=5, value=3)
    num_pieces = st.sidebar.slider("Nombre de pièces", min_value=5, max_value=20, value=10)
    
    st.sidebar.write(f"Vous avez sélectionné : {num_operators} opérateurs, {num_measurements} mesures, et {num_pieces} pièces.")
    
    # Perform calculation if data is valid
    if df.shape[0] >= num_pieces and df.shape[1] >= num_operators * num_measurements + 1:
        rr, ev, av, vp = calculate_rr(df, num_operators, num_measurements)
        
        if rr is not None:
            # Display the results
            st.subheader("Résultats de l'analyse R&R")
            st.write(f"**%R&R :** {rr:.2f}")
            st.write(f"**%EV :** {ev:.2f}")
            st.write(f"**%AV :** {av:.2f}")
            st.write(f"**%Vp :** {vp:.2f}")
    else:
        st.error("Les données dans le fichier sont insuffisantes pour correspondre aux paramètres sélectionnés.")
