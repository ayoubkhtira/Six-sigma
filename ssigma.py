"""
App Streamlit Gage R&R (Crossed) - Version Production
Compatible Excel/CSV + Saisie Manuelle + Flexible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Config page
st.set_page_config(
    page_title="Gage R&R Calculator", 
    page_icon="🧰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .metric-card {background: linear-gradient(90deg, #f0f2f6 0%, #e6f3ff 100%);}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧰 Gage R&R Calculator</h1>', unsafe_allow_html=True)

# ============================================================================
# FONCTION PRINCIPALE DE CALCUL (Flexible)
# ============================================================================
def compute_gage_rr(df, k=5.15):
    """Calcule Gage R&R avec détection automatique structure données"""
    
    # Nettoyage et validation
    df = df.dropna().reset_index(drop=True)
    required_cols = ['Operator', 'Part', 'Trial', 'Measurement']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Colonnes requises: {', '.join(required_cols)}")
        return None
    
    df['Operator'] = df['Operator'].astype(str)
    df['Part'] = df['Part'].astype(str)
    df['Trial'] = df['Trial'].astype(int)
    
    # Détection automatique structure
    n_ops = df['Operator'].nunique()
    n_parts = df['Part'].nunique()
    n_trials = df['Trial'].nunique()
    total_rows = len(df)
    expected_rows = n_ops * n_parts * n_trials
    
    st.info(f"📊 Structure détectée: **{n_ops} op** × **{n_parts} pièces** × **{n_trials} essais** = {expected_rows} lignes")
    
    if total_rows != expected_rows:
        st.warning(f"⚠️ {total_rows}/{expected_rows} lignes. Calcul avec données disponibles.")
    
    # CALCULS Gage R&R (Méthode Average & Range)
    
    # 1. X-double bar (moyenne par op/part)
    df['Xdouble'] = df.groupby(['Operator', 'Part'])['Measurement'].transform('mean')
    
    # 2. R-bar (range par op/part)
    range_data = df.groupby(['Operator', 'Part'])['Measurement'].agg(['min', 'max']).reset_index()
    range_data['R'] = range_data['max'] - range_data['min']
    Rbar = range_data['R'].mean()
    
    # 3. Moyennes par pièce (tous opérateurs)
    part_means = df.groupby('Part')['Xdouble'].mean().reset_index()
    part_means.columns = ['Part', 'Xbar_part']
    range_parts = part_means['Xbar_part'].max() - part_means['Xbar_part'].min()
    
    # 4. Facteurs d2* (AIAG standards)
    d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326}
    d2_gage = d2_values.get(n_trials, 1.693)
    d2_parts = d2_values.get(n_ops, 1.693)
    
    # 5. Composantes de variance
    EV = Rbar * (k / d2_gage)  # Equipment Variation
    AV = range_data.groupby('Operator')['R'].mean().mean() * (k / d2_gage)  # Appraiser Variation
    GRR = np.sqrt(EV**2 + AV**2)
    PV = range_parts * (k / d2_parts)  # Part Variation
    TV = np.sqrt(GRR**2 + PV**2)
    
    # 6. Pourcentages
    pct_GRR = 100 * (GRR / TV) if TV > 0 else 0
    pct_EV = 100 * (EV / TV) if TV > 0 else 0
    
    # 7. Interprétation
    if pct_GRR < 10:
        status = "✅ Excellent"
        color = "green"
    elif pct_GRR < 30:
        status = "⚠️ Acceptable"
        color = "orange"
    else:
        status = "❌ Non acceptable"
        color = "red"
    
    return {
        'n_ops': n_ops, 'n_parts': n_parts, 'n_trials': n_trials,
        'EV': EV, 'AV': AV, 'GRR': GRR, 'PV': PV, 'TV': TV,
        'pct_GRR': pct_GRR, 'pct_EV': pct_EV, 'status': status,
        'status_color': color, 'Rbar': Rbar, 'df': df, 'part_means': part_means,
        'range_data': range_data
    }

# ============================================================================
# INTERFACE
# ============================================================================

# Sidebar paramètres
st.sidebar.header("⚙️ Paramètres")
k_factor = st.sidebar.slider("Niveau confiance (K)", 4.5, 6.0, 5.15, 0.01, 
                            help="5.15 = 99% confiance (AIAG standard)")

# Choix mode
tab1, tab2 = st.tabs(["📁 Import Excel/CSV", "⌨️ Saisie Manuelle"])

with tab1:
    st.header("Import Fichier")
    uploaded_file = st.file_uploader("Choisir Excel/CSV", type=['xlsx', 'csv'], help="Template: Operator|Part|Trial|Measurement")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier chargé: {len(df)} lignes")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Calculer Gage R&R", type="primary", use_container_width=True):
                results = compute_gage_rr(df, k_factor)
                if results:
                    st.session_state.results = results
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Erreur lecture: {e}")
            st.info("""
            **Template Excel requis:**
            ```
            Operator | Part | Trial | Measurement
            Op1      | P1   | 1     | 45.2
            Op1      | P1   | 2     | 45.3
            Op2      | P1   | 1     | 45.1
            ...
            ```
            """)

with tab2:
    st.header("Saisie Manuelle")
    
    col1, col2 = st.columns(2)
    with col1:
        n_ops = st.number_input("Opérateurs", 2, 5, 3)
        n_parts = st.number_input("Pièces", 5, 15, 10)
    
    with col2:
        n_trials = st.number_input("Essais", 2, 5, 3)
    
    if st.button("🎲 Générer Données Test", use_container_width=True):
        ops = [f"Op{i+1}" for i in range(n_ops)]
        parts = [f"P{i+1}" for i in range(n_parts)]
        
        np.random.seed(42)
        data = []
        for op in ops:
            for part in parts:
                base = np.random.normal(45.5, 2.0)  # ~45mm
                for trial in range(n_trials):
                    meas = base + np.random.normal(0, 0.3)
                    data.append([op, part, trial+1, meas])
        
        df_test = pd.DataFrame(data, columns=['Operator', 'Part', 'Trial', 'Measurement'])
        st.dataframe(df_test, use_container_width=True)
        
        if st.button("💾 Sauvegarder Test Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_test.to_excel(writer, index=False, sheet_name='GageRR')
            st.download_button(
                "📥 Télécharger test.xlsx",
                buffer.getvalue(),
                "donnees_test_gage_rr.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.info("💡 Utilisez 'Générer Données Test' pour commencer rapidement")

# ============================================================================
# AFFICHAGE RÉSULTATS
# ============================================================================
if 'results' in st.session_state:
    results = st.session_state.results
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>%GR&R</h3>
            <h2 style='color: var(--theme-color);'>{results['pct_GRR']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("%EV", f"{results['pct_EV']:.1f}%")
    
    with col3:
        st.metric("TV", f"{results['TV']:.4f}")
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Statut</h3>
            <h2 style='color: {results['status_color']}'>{results['status']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tableau complet
    st.subheader("📋 Tableau Complet")
    results_df = pd.DataFrame({
        'Source': ['Equipment (EV)', 'Appraiser (AV)', 'GRR Total', 'Parts (PV)', 'Total Variation (TV)'],
        'σ (Std Dev)': [results['EV'], results['AV'], results['GRR'], results['PV'], results['TV']],
        '%GRR': [f"{100*(results['EV']/results['TV']):.1f}%", 
                f"{100*(results['AV']/results['TV']):.1f}%", 
                f"{results['pct_GRR']:.1f}%", 
                f"{100*(results['PV']/results['TV']):.1f}%", 100.0]
    })
    st.dataframe(results_df, use_container_width=True)
    
    # Graphiques
    st.subheader("📈 Visualisations")
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Xbar par Pièce', 'Range par Op/Part', 
                                     'Distribution Mesures', 'Composantes Variance'))
    
    # 1. Xbar par pièce
    fig.add_trace(px.box(results['part_means'], y='Xbar_part', x='Part').data[0], row=1, col=1)
    
    # 2. Range par op/part
    fig.add_trace(px.box(results['range_data'], y='R', x='Operator').data[0], row=1, col=2)
    
    # 3. Histogramme
    fig.add_trace(go.Histogram(x=results['df']['Measurement'], nbinsx=20, name='Mesures'), row=2, col=1)
    
    # 4. Variance components
    vars_comp = [results['EV']**2, results['AV']**2, results['PV']**2]
    fig.add_trace(go.Bar(x=['EV²', 'AV²', 'PV²'], y=vars_comp), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, font_size=10)
    st.plotly_chart(fig, use_container_width=True)
    
    # Données brutes
    with st.expander("📋 Données utilisées"):
        st.dataframe(results['df'])
    
    # Export
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Exporter Résultats CSV", csv_buffer.getvalue(), "gage_rr_results.csv")
    
    st.success(f"""
    🎯 **Interprétation:**
    - <10%: Excellent système mesure
    - 10-30%: Acceptable (améliorer si possible)  
    - >30%: Système mesure inadéquat
    """, unsafe_allow_html=True)

# Instructions
with st.expander("📖 Guide d'utilisation"):
    st.markdown("""
    ## 🚀 Démarrage rapide
    
    **Option 1 - Import:**
    1. Préparez Excel/CSV avec colonnes: `Operator | Part | Trial | Measurement`
    2. Importez → Cliquez "Calculer"
    
    **Option 2 - Test:**
    1. Onglet "Saisie Manuelle"
    2. Cliquez "Générer Données Test" 
    3. Téléchargez Excel template
    
    **Paramètres optimaux (AIAG):**
    - 3 opérateurs × 10 pièces × 3 essais = 90 mesures
    """)

st.markdown("---")
st.caption("✅ App Gage R&R - Compatible logistique/qualité - Casablanca 2026")
