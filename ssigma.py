"""
App Streamlit Gage R&R - FIX EXCEL DOWNLOAD
Version avec sauvegarde EXCEL garantie (pas CSV)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Config
st.set_page_config(page_title="Gage R&R", page_icon="🧰", layout="wide")

st.title("🧰 Gage R&R Calculator - Excel Ready")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
k_factor = st.sidebar.slider("K-factor (99%)", 4.5, 6.0, 5.15, 0.01)

# Tabs
tab1, tab2 = st.tabs(["📁 Import Excel/CSV", "⌨️ Saisie + Excel Test"])

with tab1:
    st.header("Import Fichier")
    uploaded_file = st.file_uploader("Choisir Excel/CSV", type=['xlsx', 'csv'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ {len(df)} lignes chargées")
            st.dataframe(df.head(10))
            
            if st.button("🚀 Calculer", type="primary"):
                results = compute_gage_rr(df, k_factor)
                if results:
                    st.session_state.results = results
                    st.rerun()
        except:
            st.error("❌ Format invalide")

with tab2:
    st.header("🎲 Générer Template Excel")
    
    # Paramètres saisie
    col1, col2 = st.columns(2)
    with col1:
        n_ops = st.number_input("Opérateurs", 2, 5, 3, key="n_ops")
        n_parts = st.number_input("Pièces", 5, 15, 10, key="n_parts")
    with col2:
        n_trials = st.number_input("Essais", 2, 5, 3, key="n_trials")
    
    if st.button("✅ CRÉER FICHIER EXCEL TEST", type="primary", use_container_width=True):
        # Génération données réalistes ~45mm
        np.random.seed(42)
        ops = [f"Op{i+1}" for i in range(n_ops)]
        parts = [f"P{i+1}" for i in range(n_parts)]
        
        data = []
        for op in ops:
            for part in parts:
                base = 45.5 + (int(part[1:])-1)*0.2  # Progression pièces
                for trial in range(n_trials):
                    # Variation réaliste: répétabilité ±0.1, reproductibilité ±0.3
                    meas = base + np.random.normal(0, 0.15 + 0.1*np.random.random())
                    data.append([op, part, trial+1, round(meas, 2)])
        
        df_test = pd.DataFrame(data, columns=['Operator', 'Part', 'Trial', 'Measurement'])
        
        # ✅ SAUVEGARDE EXCEL DIRECT
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_test.to_excel(writer, index=False, sheet_name='GageRR_Data')
            
            # Ajout onglet instructions
            instructions = pd.DataFrame({
                'INSTRUCTIONS': [
                    '1. REMPLACEZ les valeurs dans colonne D (Measurement)',
                    f'2. Gardez Opérateurs: {n_ops}, Pièces: {n_parts}, Essais: {n_trials}',
                    '3. Sauvegardez et importez dans onglet gauche',
                    f'Total lignes: {len(df_test)}',
                    '',
                    'Exemple ligne: Op1 | P1 | 1 | 45.23'
                ]
            })
            instructions.to_excel(writer, index=False, sheet_name='Instructions')
        
        st.session_state.df_test = df_test
        st.session_state.excel_buffer = output.getvalue()
        
        st.success(f"✅ **EXCEL CRÉÉ** : {n_ops}x{n_parts}x{n_trials} = {len(df_test)} lignes")
        st.dataframe(df_test.head(10))
        
        # BOUTON DOWNLOAD EXCEL UNIQUEMENT
        st.download_button(
            label="📥 TÉLÉCHARGER EXCEL TEST.xlsx",
            data=st.session_state.excel_buffer,
            file_name=f"gage_rr_test_{n_ops}_{n_parts}_{n_trials}.xlsx",
            mime="application/vnd.openpyxl.xlsx",
            type="primary",
            use_container_width=True
        )

# Fonction calcul (identique)
@st.cache_data
def compute_gage_rr(df, k):
    df = df.dropna()
    n_ops = df['Operator'].nunique()
    n_parts = df['Part'].nunique()
    n_trials = df['Trial'].nunique()
    
    df['Xdouble'] = df.groupby(['Operator', 'Part'])['Measurement'].transform('mean')
    range_data = df.groupby(['Operator', 'Part'])['Measurement'].agg(['min','max']).reset_index()
    range_data['R'] = range_data['max'] - range_data['min']
    Rbar = range_data['R'].mean()
    
    part_means = df.groupby('Part')['Xdouble'].mean().reset_index()
    part_means.columns = ['Part', 'Xbar_part']
    range_parts = part_means['Xbar_part'].max() - part_means['Xbar_part'].min()
    
    d2 = {2:1.128, 3:1.693, 4:2.059, 5:2.326}
    d2_gage = d2.get(n_trials, 1.693)
    
    EV = Rbar * (k / d2_gage)
    AV = range_data.groupby('Operator')['R'].mean().mean() * (k / d2_gage)
    GRR = np.sqrt(EV**2 + AV**2)
    PV = range_parts * (k / d2_gage)
    TV = np.sqrt(GRR**2 + PV**2)
    
    pct_GRR = 100 * GRR / TV if TV > 0 else 0
    pct_EV = 100 * EV / TV if TV > 0 else 0
    
    status = "✅ Excellent" if pct_GRR < 10 else "⚠️ Acceptable" if pct_GRR < 30 else "❌ Non acceptable"
    
    return {
        'n_ops': n_ops, 'n_parts': n_parts, 'n_trials': n_trials,
        'EV': EV, 'AV': AV, 'GRR': GRR, 'PV': PV, 'TV': TV,
        'pct_GRR': pct_GRR, 'pct_EV': pct_EV, 'status': status,
        'df': df, 'part_means': part_means, 'range_data': range_data
    }

# Affichage résultats
if 'results' in st.session_state:
    results = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("%GRR", f"{results['pct_GRR']:.1f}%")
    with col2: st.metric("%EV", f"{results['pct_EV']:.1f}%")
    with col3: st.metric("TV", f"{results['TV']:.3f}")
    with col4: st.metric("Statut", results['status'])
    
    # Graphiques
    fig = make_subplots(2, 2, subplot_titles=('Xbar/Pièces', 'R/Opérateurs', 'Histogramme', 'Variance'))
    fig.add_trace(px.box(results['part_means'], y='Xbar_part', x='Part').data[0], 1, 1)
    fig.add_trace(px.box(results['range_data'], y='R', x='Operator').data[0], 1, 2)
    fig.add_trace(go.Histogram(x=results['df']['Measurement'], nbinsx=15), 2, 1)
    fig.add_trace(go.Bar(x=['EV²','AV²','PV²'], y=[results['EV']**2,results['AV']**2,results['PV']**2]), 2, 2)
    st.plotly_chart(fig, use_container_width=True)

st.info("🎯 **WORKFLOW:** 1° Cliquez 'CRÉER FICHIER EXCEL TEST' → 2° Remplissez mesures → 3° Importez → 4° Résultats!")
