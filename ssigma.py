import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Cage R&R App", layout="wide")

st.title("🧰 App de Calcul Gage R&R (Crossed)")

# Sidebar pour paramètres
st.sidebar.header("Paramètres de l'étude")
n_operators = st.sidebar.number_input("Nombre d'opérateurs", min_value=2, max_value=10, value=3)
n_parts = st.sidebar.number_input("Nombre de pièces", min_value=5, max_value=30, value=10)
n_trials = st.sidebar.number_input("Nombre d'essais", min_value=2, max_value=5, value=3)
confidence_k = st.sidebar.number_input("Niveau de confiance (K-factor)", min_value=4.5, max_value=6.0, value=5.15, step=0.01, 
                                       help="5.15 pour 99% de confiance")

# Choix du mode d'entrée
input_mode = st.radio("Mode de saisie des données:", ["Manuel", "Importer Excel/CSV"])

def compute_gage_rr(df, n_op, n_p, n_tr, k):
    expected_rows = n_op * n_p * n_tr
    if len(df) != expected_rows:
        st.error(f"Données doivent avoir {expected_rows} lignes ({n_op} op x {n_p} pièces x {n_tr} essais)")
        return
    
    df['Xdouble'] = df.groupby(['Operator', 'Part'])['Measurement'].transform('mean')
    
    df_range = df.groupby(['Operator', 'Part'])['Measurement'].agg(['min', 'max']).reset_index()
    df_range['R'] = df_range['max'] - df_range['min']
    Rbar = df_range['R'].mean()
    
    part_means = df.groupby('Part')['Xdouble'].mean().reset_index()
    part_means.columns = ['Part', 'Xbar_part']
    Xbar_parts_mean = part_means['Xbar_part'].mean()
    Rbar_parts = part_means['Xbar_part'].max() - part_means['Xbar_part'].min()
    
    d2_dict = {2:1.128, 3:1.693, 4:2.059, 5:2.326}
    d2_val = d2_dict.get(n_tr, 1.693)
    EV = Rbar * (k / d2_val)
    
    op_ranges = df_range.groupby('Operator')['R'].mean()
    Rbar_op = op_ranges.mean()
    AV = Rbar_op * (k / 1.693) * np.sqrt(0.523)  # Constants standard
    
    DV = 0.0
    GRR = np.sqrt(EV**2 + AV**2 + DV**2)
    
    d2_parts = 1.693  # for n_op ≈3
    PV = Rbar_parts * (k / d2_parts)
    
    TV = np.sqrt(GRR**2 + PV**2)
    
    pct_GRR = 100 * GRR / TV
    pct_EV = 100 * EV / TV
    
    if pct_GRR < 10:
        accept = "✅ Excellent"
    elif pct_GRR < 30:
        accept = "⚠️ Acceptable"
    else:
        accept = "❌ Non acceptable"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("%GRR", f"{pct_GRR:.2f}%")
    with col2:
        st.metric("%EV", f"{pct_EV:.2f}%")
    with col3:
        st.metric("TV", f"{TV:.4f}")
    with col4:
        st.metric("Statut", accept)
    
    results_df = pd.DataFrame({
        'Source': ['Equipment (EV)', 'Appraiser (AV)', 'Drift (DV)', 'GRR Total', 'Part (PV)', 'Total (TV)'],
        'Std Dev': [EV, AV, DV, GRR, PV, TV],
        '% Contribution': [100*(EV/TV)**2, 100*(AV/TV)**2, 100*(DV/TV)**2, 100*(GRR/TV)**2, 100*(PV/TV)**2, 100.0]
    }).round(4)
    st.subheader("Tableau des Résultats")
    st.dataframe(results_df)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Xbar par Pièce', 'Range par Opérateur', 'Histogramme Mesures', 'Variance Components'))
    fig.add_trace(px.box(part_means, y='Xbar_part', x='Part', color='Part').data[0], row=1, col=1)
    fig.add_trace(px.box(df_range, y='R', x='Operator').data[0], row=1, col=2)
    fig.add_trace(go.Histogram(x=df['Measurement'], nbinsx=20, name='Mesures'), row=2, col=1)
    sources = ['EV²', 'AV²', 'PV²']
    vars_comp = [EV**2, AV**2, PV**2]
    fig.add_trace(go.Bar(x=sources, y=vars_comp, name='Variance'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Télécharger résultats CSV", csv_buffer.getvalue(), "gage_rr_results.csv", use_container_width=True)
    
    st.success(f"Interprétation: {accept} (%GRR = {pct_GRR:.1f}%)")

if input_mode == "Manuel":
    st.header("📝 Saisie Manuelle (pour petits datasets)")
    operators = [f"Op{i+1}" for i in range(n_operators)]
    parts = [f"P{i+1}" for i in range(n_parts)]
    
    data_dict = {}
    for op_idx, op in enumerate(operators):
        st.subheader(f"{op}")
        cols = st.columns(min(5, n_parts))
        for i, part in enumerate(parts):
            with cols[i % len(cols)]:
                data_dict[(op, part)] = st.number_input(f"{part}", key=f"{op}_{part}", value=0.0, format="%.3f")
    
    if st.button("🚀 Calculer (Manuel)"):
        rows = []
        for op in operators:
            for part in parts:
                val = data_dict.get((op, part), np.nan)
                for trial in range(n_trials):
                    rows.append({"Operator": op, "Part": part, "Trial": trial+1, "Measurement": val})
        df = pd.DataFrame(rows).dropna()
        compute_gage_rr(df, n_operators, n_parts, n_trials, confidence_k)

else:
    st.header("📁 Import Excel/CSV")
    uploaded_file = st.file_uploader("Choisissez fichier (template: Operator, Part, Trial, Measurement)", type=['xlsx','csv'])
    if uploaded_file:
        try:
            if 'csv' in uploaded_file.name:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(10))
            if st.button("🚀 Calculer (Import)"):
                compute_gage_rr(df, n_operators, n_parts, n_trials, confidence_k)
        except Exception as e:
            st.error(f"Erreur: {e}")
            st.info("**Template requis:**\n| Operator | Part | Trial | Measurement |\n| Op1 | P1 | 1 | 10.5 |")
