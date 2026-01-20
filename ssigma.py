import pandas as pd
import numpy as np
import streamlit as st

# Function to calculate the R&R
def calculate_rr(df):
    # Extract the measurements (excluding the first column)
    measurements = df.drop(columns=['N° de la pièce'])
    
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

# Load the data
xls = pd.ExcelFile('TEMPLATE CAGE RR.xlsx')
df = pd.read_excel(xls, sheet_name='Feuil1')

# Streamlit interface
st.title('Calcul de Gage R&R')

# Display the dataframe
st.write("Données des mesures des opérateurs :")
st.dataframe(df)

# Calculate the R&R metrics
rr, ev, av, vp = calculate_rr(df)

# Display the results
st.write(f"**%R&R :** {rr:.2f}")
st.write(f"**%EV :** {ev:.2f}")
st.write(f"**%AV :** {av:.2f}")
st.write(f"**%Vp :** {vp:.2f}")
