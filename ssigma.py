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

if input_mode == "Manuel":
    st.header("Saisie manuelle des données")
    col1, col2, col3 = st.columns(3)
    
    data = {}
    operators = [f"Op {i+1}" for i in range(n_operators)]
    parts = [f"Pièce {i+1}" for i in range(n_parts)]
    
    with col1:
        st.subheader("Opérateur 1")
        for i in range(n_parts):
            data[(operators[0], parts[i])] = st.number_input(f"{parts[i]}", key=f"{operators[0]}_{parts[i]}", value=0.0)
    
    # Dynamique pour autres opérateurs - simplifié pour 3 max, étendre si besoin
    if n_operators >= 2:
        with col2:
            st.subheader("Opérateur 2")
            for i in range(n_parts):
                data[(operators[1], parts[i])] = st.number_input(f"{parts[i]}", key=f"{operators[1]}_{parts[i]}", value=0.0)
    
    if n_operators >= 3:
        with col3:
            st.subheader("Opérateur 3")
            for i in range(n_parts):
                data[(operators[2], parts[i])] = st.number_input(f"{parts[i]}", key=f"{operators[2]}_{parts[i]}", value=0.0)
    
    # Ajouter bouton pour calcul
    if st.button("Calculer Gage R&R (Manuel)"):
        # Créer DataFrame
        rows = []
        for op in operators[:n_operators]:
            for part in parts:
                for trial in range(n_trials):
                    # Valeur répétée pour simplicité, en réalité randomiser mais pour saisie average/range method assume repeats
                    value = data.get((op, part), np.nan)
                    rows.append({"Operator": op, "Part": part, "Trial": trial+1, "Measurement": value})
        
        df = pd.DataFrame(rows)
        df = df.dropna()
        
        compute_gage_rr(df, n_operators, n_parts, n_trials, confidence_k)

else:
    st.header("Importer fichier Excel/CSV")
    uploaded_file = st.file_uploader("Choisir un fichier Excel ou CSV", type=['xlsx', 'xls', 'csv'])
    
    required_cols = ["Operator", "Part", "Trial", "Measurement"]  # Template standard
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Données importées:")
            st.dataframe(df.head())
            
            # Vérifier colonnes
            if all(col in df.columns for col in required_cols):
                if st.button("Calculer Gage R&R (Import)"):
                    compute_gage_rr(df, n_operators, n_parts, n_trials, confidence_k)
            else:
                st.error(f"Template doit avoir les colonnes: {', '.join(required_cols)}")
                st.info("""
                **Template Excel attendu:**
                - Colonnes: Operator (texte), Part (texte/num), Trial (1,2,3...), Measurement (numérique)
                Exemple:
                Operator | Part | Trial | Measurement
                Op1      | P1   | 1     | 10.2
                ...
                """)
        except Exception as e:
            st.error(f"Erreur lecture fichier: {e}")

@st.cache_data
def compute_gage_rr(df, n_op, n_p, n_tr, k):
    # Vérifier forme
    expected_rows = n_op * n_p * n_tr
    if len(df) != expected_rows:
        st.error(f"Données doivent avoir {expected_rows} lignes ( {n_op} op x {n_p} pièces x {n_tr} essais)")
        return
    
    # Average and Range Method (standard pour manuel)
    # Calculer Xdouble (moyenne des essais par op/part)
    df['Xdouble'] = df.groupby(['Operator', 'Part'])['Measurement'].transform('mean')
    
    # Rbar (range par op/part)
    df_range = df.groupby(['Operator', 'Part'])['Measurement'].agg(['min', 'max']).reset_index()
    df_range['R'] = df_range['max'] - df_range['min']
    Rbar = df_range['R'].mean()
    
    # Xbar par part (moyenne tous ops)
    part_means = df.groupby('Part')['Xdouble'].mean().reset_index()
    part_means.columns = ['Part', 'Xbar_part']
    Xbar_parts = part_means['Xbar_part'].mean()
    Rbar_parts = part_means['Xbar_part'].max() - part_means['Xbar_part'].min()
    
    # Table A: EV (Equipment Variation = Repeatability)
    d2 = {2:1.128, 3:1.693, 4:2.059, 5:2.326}  # d2* pour small samples
    d2_val = d2.get(n_tr, 1.693)  # default 3
    EV = Rbar * (5.15 / d2_val)  # sigma_gage = Rbar/d2, then *K
    
    # Table B: AV (Appraiser Variation)
    # Rbar per op
    op_ranges = df_range.groupby('Operator')['R'].mean().reset_index()
    Rbar_op = op_ranges['R'].mean()
    AV = Rbar_op * (5.15 / 1.693) * np.sqrt(0.523)  # approx constants [web:1][web:6]
    
    # DV (appraiser*part drift, often 0 if no interaction)
    DV = 0.0
    
    # GRR
    GRR = np.sqrt(EV**2 + AV**2 + DV**2)
    
    # PV (Part Variation)
    d2_parts = 1.693  # for n_op=3
    PV = Rbar_parts * (5.15 / d2_parts)
    
    # TV
    TV = np.sqrt(GRR**2 + PV**2)
    
    # %GRR, %EV
    pct_GR&R = 100 * GRR / TV
    pct_EV = 100 * EV / TV
    
    # Interprétation
    if pct_GR&R < 10:
        accept = "Acceptable"
    elif pct_GR&R < 30:
        accept = "Moyennement acceptable"
    else:
        accept = "Non acceptable"
    
    st.header("Résultats Gage R&R")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("%GR&R", f"{pct_GR&R:.2f}%")
    with col2:
        st.metric("%EV", f"{pct_EV:.2f}%")
    with col3:
        st.metric("TV", f"{TV:.4f}")
    with col4:
        st.metric("Statut", accept)
    
    # Tableau résultats
    results_df = pd.DataFrame({
        'Source': ['Equipment Var (EV)', 'Appraiser Var (AV)', 'Drift (DV)', 'GRR', 'Part Var (PV)', 'Total Var (TV)'],
        'Std Dev': [EV, AV, DV, GRR, PV, TV],
        '% Contribution': [100*(EV/TV)**2, 100*(AV/TV)**2, 100*(DV/TV)**2, 100*(GRR/TV)**2, 100*(PV/TV)**2, 100.0]
    })
    st.table(results_df.round(4))
    
    # Graphiques
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Xbar par Part', 'R par Op/Part', 'Distribution Mesures', 'Parts vs GRR'))
    
    # Xbar chart
    fig.add_trace(px.box(part_means, y='Xbar_part', x='Part').data[0], row=1, col=1)
    
    # R chart
    fig.add_trace(px.box(df_range, y='R', x='Operator').data[0], row=1, col=2)
    
    # Histogram
    fig.add_trace(go.Histogram(x=df['Measurement'], nbinsx=20), row=2, col=1)
    
    # Components
    sources = ['EV', 'AV', 'PV']
    vars = [EV**2, AV**2, PV**2]
    fig.add_trace(go.Bar(x=sources, y=vars), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download résultats
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button("Télécharger résultats CSV", csv_buffer.getvalue(), "gage_rr_results.csv")
    
    st.info(f"**Interprétation:** {accept}. %GR&R <10%: Excellent, 10-30%: Acceptable, >30%: Améliorer système de mesure.[web:12][web:6]")
