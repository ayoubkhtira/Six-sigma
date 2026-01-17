import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION GLOBALE ---
st.set_page_config(
    page_title="📏 Gage R&R Pro Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📏"
)

# --- CSS MODERNE ET HOMOGÈNE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { 
        color: #1e293b; 
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .header-main {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Cards modernes */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.12);
    }
    
    .status-card {
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    
    .status-good { border-left-color: #10b981; background: rgba(16,185,129,0.05); }
    .status-warning { border-left-color: #f59e0b; background: rgba(245,158,11,0.05); }
    .status-bad { border-left-color: #ef4444; background: rgba(239,68,68,0.05); }
    
    /* Boutons modernes */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 20px rgba(59,130,246,0.3);
        transition: all 0.3s ease;
        height: auto;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59,130,246,0.4);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Sidebar améliorée */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    
    .step-indicator {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        justify-content: center;
    }
    
    .step {
        padding: 1rem;
        border-radius: 50px;
        font-weight: 600;
        min-width: 120px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .step-active { 
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 8px 25px rgba(59,130,246,0.3);
    }
    
    .step-completed { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- ÉTAT DE L'APPLICATION ---
if 'df_gage' not in st.session_state:
    st.session_state.df_gage = None
if 'gage_config' not in st.session_state:
    st.session_state.gage_config = {
        'n_operateurs': 3, 'n_pieces': 10, 'n_essais': 3,
        'tol_lower': 45, 'tol_upper': 55, 'target': 50
    }
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'gage_results' not in st.session_state:
    st.session_state.gage_results = None

# --- FONCTIONS (gardées identiques mais optimisées) ---
def generate_gage_rr_data(n_operateurs=3, n_pieces=10, n_essais=3, target=50, process_variation=1):
    """Génère des données simulées Gage R&R"""
    data = []
    operateurs = [f'Op {chr(65+i)}' for i in range(n_operateurs)]
    pieces = [f'P{i+1:02d}' for i in range(n_pieces)]
    
    piece_effects = np.random.normal(0, 0.5 * process_variation, n_pieces)
    operator_effects = np.random.normal(0, 0.3 * process_variation, n_operateurs)
    
    for op_idx, operateur in enumerate(operateurs):
        for piece_idx, piece in enumerate(pieces):
            for essai in range(1, n_essais + 1):
                base_value = target + piece_effects[piece_idx] + operator_effects[op_idx]
                measurement = round(base_value + np.random.normal(0, 0.2 * process_variation), 3)
                data.append({'Opérateur': operateur, 'Pièce': piece, 'Essai': essai, 'Mesure': measurement})
    
    return pd.DataFrame(data)

def calculate_gage_rr(df, tol_lower, tol_upper):
    """Calcule les statistiques Gage R&R avec ANOVA (code identique, simplifié)"""
    try:
        df['Opérateur'] = df['Opérateur'].astype('category')
        df['Pièce'] = df['Pièce'].astype('category')
        df['Essai'] = df['Essai'].astype('category')
        
        model = ols('Mesure ~ C(Opérateur) + C(Pièce) + C(Opérateur):C(Pièce)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extraire et calculer les composantes (logique identique)
        ss_operator = anova_table.loc['C(Opérateur)', 'sum_sq']
        ss_piece = anova_table.loc['C(Pièce)', 'sum_sq']
        ss_interaction = anova_table.loc['C(Opérateur):C(Pièce)', 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']
        
        df_operator = anova_table.loc['C(Opérateur)', 'df']
        df_piece = anova_table.loc['C(Pièce)', 'df']
        df_interaction = anova_table.loc['C(Opérateur):C(Pièce)', 'df']
        df_error = anova_table.loc['Residual', 'df']
        
        ms_operator = ss_operator / df_operator
        ms_piece = ss_piece / df_piece
        ms_interaction = ss_interaction / df_interaction
        ms_error = ss_error / df_error
        
        f_critical = stats.f.ppf(0.95, df_interaction, df_error)
        f_interaction = ms_interaction / ms_error
        
        if f_interaction > f_critical:
            sigma_repeatability = ms_error
            sigma_reproducibility = max(0, (ms_operator - ms_interaction) / (df['Pièce'].nunique() * df['Essai'].nunique()))
            sigma_interaction = max(0, (ms_interaction - ms_error) / df['Essai'].nunique())
            sigma_rr = np.sqrt(sigma_repeatability + sigma_reproducibility + sigma_interaction)
        else:
            ms_combined = (ss_interaction + ss_error) / (df_interaction + df_error)
            sigma_repeatability = ms_combined
            sigma_reproducibility = max(0, (ms_operator - ms_combined) / (df['Pièce'].nunique() * df['Essai'].nunique()))
            sigma_rr = np.sqrt(sigma_repeatability + sigma_reproducibility)
        
        sigma_piece = max(0, (ms_piece - ms_interaction) / (df['Opérateur'].nunique() * df['Essai'].nunique()))
        total_variation = np.sqrt(sigma_rr**2 + sigma_piece**2)
        tol_width = tol_upper - tol_lower
        
        # Calculs finaux
        results = {
            'ANOVA Table': anova_table,
            'Repeatability (EV)': 6 * np.sqrt(sigma_repeatability),
            'Reproducibility (AV)': 6 * np.sqrt(sigma_reproducibility),
            'R&R (GRR)': 6 * sigma_rr,
            'Part Variation (PV)': 6 * np.sqrt(sigma_piece),
            'Total Variation (TV)': 6 * total_variation,
            '%EV': (sigma_repeatability / total_variation) * 100,
            '%AV': (sigma_reproducibility / total_variation) * 100,
            '%R&R': (sigma_rr / total_variation) * 100,
            '%PV': (sigma_piece / total_variation) * 100,
            '%Tol EV': (6 * np.sqrt(sigma_repeatability) / tol_width) * 100,
            '%Tol AV': (6 * np.sqrt(sigma_reproducibility) / tol_width) * 100,
            '%Tol GRR': (6 * sigma_rr / tol_width) * 100,
            'ndc': int(1.41 * (sigma_piece / sigma_rr)),
            'Classification': "Acceptable" if (6 * sigma_rr / tol_width) <= 10 else "Marginal" if (6 * sigma_rr / tol_width) <= 30 else "Inacceptable",
            'Sigma Repeatability': np.sqrt(sigma_repeatability),
            'Sigma Reproducibility': np.sqrt(sigma_reproducibility),
            'Sigma R&R': sigma_rr,
            'Sigma Piece': np.sqrt(sigma_piece)
        }
        return results
    except Exception as e:
        st.error(f"Erreur calcul: {str(e)}")
        return None

# --- SIDEBAR MODERNE ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 16px; color: white;'>
        <h2 style='margin: 0; font-size: 1.3rem;'>📏 Gage R&R Pro</h2>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Analyse système de mesure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Indicateur de progression
    st.markdown('<div class="step-indicator">', unsafe_allow_html=True)
    steps = [
        ("Configuration", st.session_state.current_step >= 1),
        ("Données", st.session_state.current_step >= 2),
        ("Analyse", st.session_state.current_step >= 3)
    ]
    
    for i, (step_name, completed) in enumerate(steps):
        step_class = "step-active" if i+1 == st.session_state.current_step else "step-completed" if completed else "step"
        st.markdown(f'<div class="{step_class}">{step_name}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration rapide
    st.markdown("### ⚙️ Configuration rapide")
    
    col1, col2 = st.columns(2)
    with col1:
        n_operateurs = st.number_input("👥 Opérateurs", min_value=2, max_value=10, value=3, key="n_op")
        n_essais = st.number_input("🔄 Essais", min_value=2, max_value=10, value=3, key="n_ess")
    with col2:
        n_pieces = st.number_input("📦 Pièces", min_value=5, max_value=50, value=10, key="n_pieces")
        target = st.number_input("🎯 Cible", value=50.0, format="%.2f", key="target")
    
    st.markdown("### 🎯 Tolérances")
    col1, col2 = st.columns(2)
    with col1:
        tol_lower = st.number_input("LSL", value=45.0, format="%.2f", key="lsl")
    with col2:
        tol_upper = st.number_input("USL", value=55.0, format="%.2f", key="usl")
    
    st.session_state.gage_config = {
        'n_operateurs': n_operateurs, 'n_pieces': n_pieces, 
        'n_essais': n_essais, 'tol_lower': tol_lower, 
        'tol_upper': tol_upper, 'target': target
    }
    
    st.markdown("---")
    
    # Actions principales
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🎲 Générer données", key="generate", use_container_width=True):
            st.session_state.df_gage = generate_gage_rr_data(**st.session_state.gage_config)
            st.session_state.current_step = 2
            st.rerun()
    
    with col2:
        uploaded_file = st.file_uploader("📤 Charger CSV", type=['csv'], key="upload")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if all(col in df.columns for col in ['Opérateur', 'Pièce', 'Essai', 'Mesure']):
                    st.session_state.df_gage = df
                    st.session_state.current_step = 2
                    st.success("✅ Données chargées!")
                    st.rerun()
                else:
                    st.error("❌ Colonnes manquantes")
            except:
                st.error("❌ Erreur fichier")
    
    if st.session_state.df_gage is not None:
        with col3:
            csv = st.session_state.df_gage.to_csv(index=False)
            st.download_button("💾 Export CSV", csv, "gage_rr_data.csv", use_container_width=True)

# --- INTERFACE PRINCIPALE MODERNE ---
def render_header():
    st.markdown("""
    <div class='header-main'>
        <h1 style='margin: 0 0 0.5rem 0; font-size: 2.5rem;'>📏 Gage R&R Analysis</h1>
        <p style='margin: 0; font-size: 1.2rem; opacity: 0.95;'>Évaluez la fiabilité de votre système de mesure en 3 étapes</p>
    </div>
    """, unsafe_allow_html=True)

def render_step1():
    st.markdown('<h2 style="text-align: center; margin-bottom: 2rem;">👋 Étape 1: Préparation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🚀 Démarrer rapidement</h3>
            <p>Utilisez notre générateur de données simulées pour tester l'outil immédiatement.</p>
            <ul style='color: #64748b;'>
                <li>Configuration personnalisable</li>
                <li>Données réalistes</li>
                <li>1 clic</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Données réelles</h3>
            <p>Chargez vos données d'étude Gage R&R au format CSV standard.</p>
            <ul style='color: #64748b;'>
                <li>Format AIAG compatible</li>
                <li>Édition intégrée</li>
                <li>Validation automatique</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_step2():
    if st.session_state.df_gage is None:
        st.error("⚠️ Aucune donnée chargée. Utilisez la sidebar.")
        return
    
    st.markdown('<h2 style="text-align: center;">📊 Étape 2: Données & Vérification</h2>', unsafe_allow_html=True)
    
    # Métriques rapides
    col1, col2, col3, col4 = st.columns(4)
    df = st.session_state.df_gage
    with col1: st.metric("📈 Moyenne", f"{df['Mesure'].mean():.3f}")
    with col2: st.metric("📏 Écart-type", f"{df['Mesure'].std():.3f}")
    with col3: st.metric("⬇️ Min", f"{df['Mesure'].min():.3f}")
    with col4: st.metric("⬆️ Max", f"{df['Mesure'].max():.3f}")
    
    # Onglets organisés
    tab1, tab2 = st.tabs(["✏️ Éditeur", "📈 Aperçu"])
    
    with tab1:
        edited_df = st.data_editor(
            df,
            column_config={
                "Opérateur": st.column_config.TextColumn("👤 Opérateur"),
                "Pièce": st.column_config.TextColumn("📦 Pièce"),
                "Essai": st.column_config.NumberColumn("🔄 Essai", min_value=1),
                "Mesure": st.column_config.NumberColumn("📏 Mesure", format="%.3f")
            },
            use_container_width=True,
            height=400
        )
        if not edited_df.equals(df):
            st.session_state.df_gage = edited_df
            st.rerun()
    
    with tab2:
        st.dataframe(df.groupby(['Opérateur', 'Pièce']).size().reset_index(name='Nb Essais'), use_container_width=True)

def render_step3():
    if st.session_state.df_gage is None:
        return
    
    st.markdown('<h2 style="text-align: center;">🎯 Étape 3: Analyse complète</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Lancer l'analyse Gage R&R", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            results = calculate_gage_rr(st.session_state.df_gage, 
                                      st.session_state.gage_config['tol_lower'],
                                      st.session_state.gage_config['tol_upper'])
            if results:
                st.session_state.gage_results = results
                st.session_state.current_step = 3
                st.rerun()
    
    if st.session_state.gage_results:
        results = st.session_state.gage_results
        
        # Statut principal
        col1, col2 = st.columns([3,1])
        with col1:
            status_class = {
                "Acceptable": "status-good", "Marginal": "status-warning", "Inacceptable": "status-bad"
            }[results['Classification']]
            
            st.markdown(f"""
            <div class='status-card {status_class}'>
                <h3>{results['Classification']} 📊</h3>
                <h1 style='margin: 0; font-size: 3rem;'>
                    {results['%R&R']:.1f}% <span style='font-size: 1.2rem;'>R&R</span>
                </h1>
                <p style='margin: 1rem 0 0 0; font-size: 1.1rem;'>
                    {results['ndc']} catégories distinctes
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Métriques clés
            col_a, col_b = st.columns(2)
            with col_a: st.metric("Répétabilité", f"{results['%EV']:.1f}%")
            with col_b: st.metric("Reproductibilité", f"{results['%AV']:.1f}%")
        
        # Résultats détaillés
        st.markdown("### 📋 Tableau de résultats")
        results_df = pd.DataFrame({
            'Métrique': ['Répétabilité (EV)', 'Reproductibilité (AV)', 'R&R Total', 'Variation Pièces', 'Total'],
            '% Variation': [f"{results['%EV']:.1f}%", f"{results['%AV']:.1f}%", f"{results['%R&R']:.1f}%", f"{results['%PV']:.1f}%", "100%"],
            '% Tolérance': [f"{results['%Tol EV']:.1f}%", f"{results['%Tol AV']:.1f}%", f"{results['%Tol GRR']:.1f}%", '-', '-'],
            '6σ': [f"{results['Repeatability (EV)']:.3f}", f"{results['Reproducibility (AV)']:.3f}", f"{results['R&R (GRR)']:.3f}", f"{results['Part Variation (PV)']:.3f}", f"{results['Total Variation (TV)']:.3f}"]
        })
        st.dataframe(results_df, use_container_width=True)
        
        # Rapport automatique
        st.markdown("---")
        report = f"""📄 RAPPORT GAGE R&R AUTOMATIQUE

Étude: {len(st.session_state.df_gage)} mesures
Résultat: {results['Classification']} ({results['%R&R']:.1f}% R&R)
ndc: {results['ndc']} | Tolérance: {results['%Tol GRR']:.1f}%

Recommandations:
- {results['Classification']} système détecté
- Actions à planifier selon criticité
"""
        st.download_button("📥 Télécharger rapport", report, "rapport_gage_rr.txt", use_container_width=True)

# --- RENDU PRINCIPAL ---
render_header()

if st.session_state.df_gage is None:
    st.session_state.current_step = 1
    render_step1()
elif st.session_state.gage_results is None:
    st.session_state.current_step = 2
    render_step2()
else:
    st.session_state.current_step = 3
    render_step3()

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #64748b;'>
    <p><strong>Gage R&R Pro Suite v3.0</strong> | AIAG MSA 4th Edition | 2026</p>
    <p style='font-size: 0.9em;'>Outil professionnel pour l'analyse des systèmes de mesure</p>
</div>
""", unsafe_allow_html=True)
