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

# --- CONFIGURATION ET STYLE ---
st.set_page_config(
    page_title="Six Sigma Pro Suite - Gage R&R",
    layout="wide",
    page_icon="📏"
)

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    .card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALISATION DES DONNÉES ---
if 'df_gage' not in st.session_state:
    st.session_state.df_gage = None
if 'gage_config' not in st.session_state:
    st.session_state.gage_config = {
        'n_operateurs': 3,
        'n_pieces': 10,
        'n_essais': 3,
        'tol_lower': 45,
        'tol_upper': 55,
        'target': 50
    }

# --- FONCTIONS GAGE R&R ---
def generate_gage_rr_data(n_operateurs=3, n_pieces=10, n_essais=3, target=50, process_variation=1):
    """
    Génère des données simulées pour une étude Gage R&R
    """
    data = []
    
    # Créer les noms des opérateurs
    operateurs = [f'Opérateur {chr(65+i)}' for i in range(n_operateurs)]
    
    # Créer les noms des pièces
    pieces = [f'Pièce {i+1:02d}' for i in range(n_pieces)]
    
    # Générer les variations
    piece_effects = np.random.normal(0, 0.5 * process_variation, n_pieces)
    operator_effects = np.random.normal(0, 0.3 * process_variation, n_operateurs)
    
    for op_idx, operateur in enumerate(operateurs):
        for piece_idx, piece in enumerate(pieces):
            for essai in range(1, n_essais + 1):
                # Valeur de base avec effets
                base_value = target + piece_effects[piece_idx] + operator_effects[op_idx]
                
                # Ajouter de la variation d'essai (repeatability)
                measurement = base_value + np.random.normal(0, 0.2 * process_variation)
                
                # Arrondir à 3 décimales
                measurement = round(measurement, 3)
                
                data.append({
                    'Opérateur': operateur,
                    'Pièce': piece,
                    'Essai': essai,
                    'Mesure': measurement
                })
    
    return pd.DataFrame(data)

def calculate_gage_rr(df, tol_lower, tol_upper, tol_width=None):
    """
    Calcule les statistiques Gage R&R avec ANOVA
    """
    try:
        # Préparer les données pour ANOVA
        df['Opérateur'] = df['Opérateur'].astype('category')
        df['Pièce'] = df['Pièce'].astype('category')
        df['Essai'] = df['Essai'].astype('category')
        
        # Modèle ANOVA à deux facteurs avec interaction
        model = ols('Mesure ~ C(Opérateur) + C(Pièce) + C(Opérateur):C(Pièce)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extraire les sommes des carrés
        ss_operator = anova_table.loc['C(Opérateur)', 'sum_sq']
        ss_piece = anova_table.loc['C(Pièce)', 'sum_sq']
        ss_interaction = anova_table.loc['C(Opérateur):C(Pièce)', 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq'] if 'Residual' in anova_table.index else 0
        
        # Degrés de liberté
        df_operator = anova_table.loc['C(Opérateur)', 'df']
        df_piece = anova_table.loc['C(Pièce)', 'df']
        df_interaction = anova_table.loc['C(Opérateur):C(Pièce)', 'df']
        df_error = anova_table.loc['Residual', 'df'] if 'Residual' in anova_table.index else 0
        
        # Calculer les carrés moyens
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0
        ms_piece = ss_piece / df_piece if df_piece > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0
        
        # Calculer les composantes de variance
        # Si l'interaction n'est pas significative, on la combine avec l'erreur
        f_critical = stats.f.ppf(0.95, df_interaction, df_error)
        f_interaction = ms_interaction / ms_error if ms_error > 0 else 0
        
        if f_interaction > f_critical:
            # Interaction significative
            sigma_repeatability = ms_error
            sigma_reproducibility = max(0, (ms_operator - ms_interaction) / (df['Pièce'].nunique() * df['Essai'].nunique()))
            sigma_interaction = max(0, (ms_interaction - ms_error) / df['Essai'].nunique())
            sigma_rr = np.sqrt(sigma_repeatability + sigma_reproducibility + sigma_interaction)
        else:
            # Interaction non significative - combiner avec l'erreur
            ms_combined = (ss_interaction + ss_error) / (df_interaction + df_error)
            sigma_repeatability = ms_combined
            sigma_reproducibility = max(0, (ms_operator - ms_combined) / (df['Pièce'].nunique() * df['Essai'].nunique()))
            sigma_interaction = 0
            sigma_rr = np.sqrt(sigma_repeatability + sigma_reproducibility)
        
        # Variation des pièces
        sigma_piece = max(0, (ms_piece - ms_interaction) / (df['Opérateur'].nunique() * df['Essai'].nunique()))
        
        # Calculer les différents pourcentages
        total_variation = np.sqrt(sigma_rr**2 + sigma_piece**2)
        
        # Pourcentage de variation
        pct_ev = (sigma_repeatability / total_variation) * 100 if total_variation > 0 else 0
        pct_av = (sigma_reproducibility / total_variation) * 100 if total_variation > 0 else 0
        pct_rr = (sigma_rr / total_variation) * 100 if total_variation > 0 else 0
        pct_pv = (sigma_piece / total_variation) * 100 if total_variation > 0 else 0
        
        # Calculer par rapport à la tolérance
        if tol_width is None:
            tol_width = tol_upper - tol_lower
        
        tol_pct_ev = (6 * np.sqrt(sigma_repeatability) / tol_width) * 100 if tol_width > 0 else 0
        tol_pct_av = (6 * np.sqrt(sigma_reproducibility) / tol_width) * 100 if tol_width > 0 else 0
        tol_pct_rr = (6 * sigma_rr / tol_width) * 100 if tol_width > 0 else 0
        
        # Nombre de catégories distinctes
        ndc = int(1.41 * (sigma_piece / sigma_rr)) if sigma_rr > 0 else 0
        
        # Classification
        if tol_pct_rr <= 10:
            classification = "Acceptable"
            color = "green"
        elif tol_pct_rr <= 30:
            classification = "Marginal"
            color = "orange"
        else:
            classification = "Inacceptable"
            color = "red"
        
        results = {
            'ANOVA Table': anova_table,
            'Repeatability (EV)': 6 * np.sqrt(sigma_repeatability),
            'Reproducibility (AV)': 6 * np.sqrt(sigma_reproducibility),
            'R&R (GRR)': 6 * sigma_rr,
            'Part Variation (PV)': 6 * np.sqrt(sigma_piece),
            'Total Variation (TV)': 6 * total_variation,
            '%EV': pct_ev,
            '%AV': pct_av,
            '%R&R': pct_rr,
            '%PV': pct_pv,
            '%Tol EV': tol_pct_ev,
            '%Tol AV': tol_pct_av,
            '%Tol GRR': tol_pct_rr,
            'ndc': ndc,
            'Classification': classification,
            'Color': color,
            'Sigma Repeatability': np.sqrt(sigma_repeatability),
            'Sigma Reproducibility': np.sqrt(sigma_reproducibility),
            'Sigma R&R': sigma_rr,
            'Sigma Piece': np.sqrt(sigma_piece)
        }
        
        return results
        
    except Exception as e:
        st.error(f"Erreur dans le calcul Gage R&R: {str(e)}")
        return None

def create_gage_rr_plot(df):
    """
    Crée des visualisations pour l'étude Gage R&R
    """
    plots = {}
    
    # 1. Graphique par opérateur
    fig_op = go.Figure()
    
    for operateur in df['Opérateur'].unique():
        df_op = df[df['Opérateur'] == operateur]
        fig_op.add_trace(go.Box(
            y=df_op['Mesure'],
            name=operateur,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig_op.update_layout(
        title='Distribution des Mesures par Opérateur',
        yaxis_title='Mesure',
        xaxis_title='Opérateur',
        height=400
    )
    plots['par_operateur'] = fig_op
    
    # 2. Graphique par pièce
    fig_piece = go.Figure()
    
    for piece in df['Pièce'].unique():
        df_piece = df[df['Pièce'] == piece]
        fig_piece.add_trace(go.Box(
            y=df_piece['Mesure'],
            name=piece,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False
        ))
    
    fig_piece.update_layout(
        title='Distribution des Mesures par Pièce',
        yaxis_title='Mesure',
        xaxis_title='Pièce',
        height=400
    )
    plots['par_piece'] = fig_piece
    
    # 3. Graphique Interaction Opérateur x Pièce
    fig_interaction = px.line(df, x='Pièce', y='Mesure', 
                             color='Opérateur', 
                             title='Interaction Opérateur x Pièce',
                             markers=True)
    
    # Calculer les moyennes par opérateur et pièce
    df_mean = df.groupby(['Opérateur', 'Pièce']).agg({'Mesure': 'mean'}).reset_index()
    
    for operateur in df['Opérateur'].unique():
        df_mean_op = df_mean[df_mean['Opérateur'] == operateur]
        fig_interaction.add_trace(go.Scatter(
            x=df_mean_op['Pièce'],
            y=df_mean_op['Mesure'],
            mode='lines',
            name=f'{operateur} (moyenne)',
            line=dict(dash='dash'),
            showlegend=True
        ))
    
    fig_interaction.update_layout(height=400)
    plots['interaction'] = fig_interaction
    
    # 4. Graphique R&R Components
    if 'gage_results' in st.session_state and st.session_state.gage_results:
        results = st.session_state.gage_results
        
        components = ['Repeatability', 'Reproducibility', 'R&R', 'Part Variation']
        values = [
            results['%EV'],
            results['%AV'],
            results['%R&R'],
            results['%PV']
        ]
        
        fig_components = go.Figure(data=[
            go.Bar(x=components, y=values, 
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ])
        
        fig_components.update_layout(
            title='Composantes de Variation (%)',
            yaxis_title='Pourcentage de Variation Totale',
            height=400
        )
        plots['components'] = fig_components
    
    return plots

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h2>📏 Gage R&R</h2>
            <p style='color: #666;'>Analyse de la fiabilité du système de mesure</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Configuration de l'étude
    with st.expander("⚙️ Configuration de l'étude", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            n_operateurs = st.number_input(
                "Nombre d'opérateurs",
                min_value=2,
                max_value=10,
                value=st.session_state.gage_config['n_operateurs'],
                help="Minimum 2 opérateurs requis"
            )
            
            n_essais = st.number_input(
                "Nombre d'essais",
                min_value=2,
                max_value=10,
                value=st.session_state.gage_config['n_essais'],
                help="Nombre de répétitions par opérateur/pièce"
            )
            
        with col2:
            n_pieces = st.number_input(
                "Nombre de pièces",
                min_value=5,
                max_value=50,
                value=st.session_state.gage_config['n_pieces'],
                help="Nombre de pièces différentes à mesurer"
            )
            
            target = st.number_input(
                "Valeur cible",
                value=st.session_state.gage_config['target'],
                format="%.2f",
                help="Valeur nominale du processus"
            )
    
    # Spécifications de tolérance
    with st.expander("🎯 Spécifications de tolérance"):
        tol_lower = st.number_input(
            "Limite inférieure (LSL)",
            value=st.session_state.gage_config['tol_lower'],
            format="%.2f"
        )
        
        tol_upper = st.number_input(
            "Limite supérieure (USL)",
            value=st.session_state.gage_config['tol_upper'],
            format="%.2f"
        )
        
        if tol_upper <= tol_lower:
            st.warning("⚠️ La limite supérieure doit être supérieure à la limite inférieure")
    
    # Boutons d'action
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Générer données", use_container_width=True):
            st.session_state.gage_config.update({
                'n_operateurs': n_operateurs,
                'n_pieces': n_pieces,
                'n_essais': n_essais,
                'tol_lower': tol_lower,
                'tol_upper': tol_upper,
                'target': target
            })
            
            # Générer les données
            st.session_state.df_gage = generate_gage_rr_data(
                n_operateurs=n_operateurs,
                n_pieces=n_pieces,
                n_essais=n_essais,
                target=target
            )
            
            st.success("✅ Données générées avec succès!")
            st.rerun()
    
    with col2:
        if st.session_state.df_gage is not None:
            csv_data = st.session_state.df_gage.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name="gage_rr_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Upload de données
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📤 Ou charger un fichier CSV",
        type=['csv'],
        help="Format requis: Colonnes 'Opérateur', 'Pièce', 'Essai', 'Mesure'"
    )
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            required_columns = ['Opérateur', 'Pièce', 'Essai', 'Mesure']
            
            if all(col in df_uploaded.columns for col in required_columns):
                st.session_state.df_gage = df_uploaded
                st.success("✅ Fichier chargé avec succès!")
                st.rerun()
            else:
                st.error("❌ Format incorrect. Colonnes requises: Opérateur, Pièce, Essai, Mesure")
        except Exception as e:
            st.error(f"Erreur lors du chargement: {str(e)}")
    
    # Information sur l'étude
    if st.session_state.df_gage is not None:
        st.markdown("---")
        st.markdown("### 📊 Résumé de l'étude")
        st.markdown(f"**Opérateurs:** {st.session_state.df_gage['Opérateur'].nunique()}")
        st.markdown(f"**Pièces:** {st.session_state.df_gage['Pièce'].nunique()}")
        st.markdown(f"**Essais:** {st.session_state.df_gage['Essai'].nunique()}")
        st.markdown(f"**Mesures totales:** {len(st.session_state.df_gage)}")

# --- CONTENU PRINCIPAL ---
st.title("📏 Étude Gage R&R - Analyse du Système de Mesure")
st.markdown("Évaluez la fiabilité et la reproductibilité de votre système de mesure")

if st.session_state.df_gage is None:
    st.info("""
    ### 👋 Commencez par configurer votre étude Gage R&R
    
    1. Définissez les paramètres de l'étude dans la barre latérale
    2. Cliquez sur "Générer données" pour créer un jeu de données simulé
    3. Ou téléchargez un template CSV et chargez vos propres données
    
    **Critères d'acceptation:**
    - %R&R < 10% : Système acceptable
    - 10% < %R&R < 30% : Système marginal (à améliorer)
    - %R&R > 30% : Système inacceptable
    """)
    
    # Template de données
    st.markdown("### 📋 Template de données requis")
    template_data = {
        'Opérateur': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Pièce': ['P1', 'P1', 'P2', 'P1', 'P1', 'P2'],
        'Essai': [1, 2, 1, 1, 2, 1],
        'Mesure': [50.1, 50.2, 49.9, 50.3, 50.1, 50.0]
    }
    template_df = pd.DataFrame(template_data)
    st.dataframe(template_df, use_container_width=True)
    
    # Bouton pour télécharger le template
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger le template CSV",
        data=template_csv,
        file_name="template_gage_rr.csv",
        mime="text/csv"
    )
    
else:
    # Afficher les données
    st.markdown("### 📋 Données de l'étude")
    
    tab_data, tab_analysis, tab_results = st.tabs(["📊 Données", "📈 Visualisations", "📋 Résultats"])
    
    with tab_data:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Données brutes de mesure")
        
        # Afficher un résumé statistique
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Moyenne", f"{st.session_state.df_gage['Mesure'].mean():.3f}")
        with col2:
            st.metric("Écart-type", f"{st.session_state.df_gage['Mesure'].std():.3f}")
        with col3:
            st.metric("Min", f"{st.session_state.df_gage['Mesure'].min():.3f}")
        with col4:
            st.metric("Max", f"{st.session_state.df_gage['Mesure'].max():.3f}")
        
        # Éditeur de données
        edited_df = st.data_editor(
            st.session_state.df_gage,
            num_rows="dynamic",
            column_config={
                "Opérateur": st.column_config.TextColumn(
                    "Opérateur",
                    help="Nom ou code de l'opérateur"
                ),
                "Pièce": st.column_config.TextColumn(
                    "Pièce",
                    help="Identifiant de la pièce"
                ),
                "Essai": st.column_config.NumberColumn(
                    "Essai",
                    min_value=1,
                    help="Numéro de l'essai (répétition)"
                ),
                "Mesure": st.column_config.NumberColumn(
                    "Mesure",
                    min_value=0.0,
                    format="%.3f",
                    help="Valeur mesurée"
                )
            },
            use_container_width=True,
            height=400
        )
        
        # Sauvegarder les modifications
        if not edited_df.equals(st.session_state.df_gage):
            st.session_state.df_gage = edited_df
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_analysis:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Visualisations de l'étude")
        
        # Générer les visualisations
        plots = create_gage_rr_plot(st.session_state.df_gage)
        
        # Graphique par opérateur
        st.plotly_chart(plots['par_operateur'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plots['par_piece'], use_container_width=True)
        
        with col2:
            st.plotly_chart(plots['interaction'], use_container_width=True)
        
        # Matrice de corrélation entre opérateurs
        st.markdown("##### Corrélation entre opérateurs")
        
        # Pivoter les données pour avoir une ligne par pièce/essai
        df_pivot = st.session_state.df_gage.pivot_table(
            index=['Pièce', 'Essai'],
            columns='Opérateur',
            values='Mesure'
        ).reset_index()
        
        # Calculer la matrice de corrélation
        corr_matrix = df_pivot[df_pivot.columns[2:]].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            title='Matrice de Corrélation entre Opérateurs'
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_results:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Analyse Gage R&R")
        
        # Bouton pour calculer
        if st.button("🎯 Calculer l'analyse Gage R&R", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                # Calculer les résultats
                tol_width = st.session_state.gage_config['tol_upper'] - st.session_state.gage_config['tol_lower']
                results = calculate_gage_rr(
                    st.session_state.df_gage,
                    st.session_state.gage_config['tol_lower'],
                    st.session_state.gage_config['tol_upper'],
                    tol_width
                )
                
                if results:
                    st.session_state.gage_results = results
                    st.rerun()
        
        # Afficher les résultats si disponibles
        if 'gage_results' in st.session_state and st.session_state.gage_results:
            results = st.session_state.gage_results
            
            # Tableau ANOVA
            st.markdown("##### Tableau ANOVA")
            st.dataframe(results['ANOVA Table'], use_container_width=True)
            
            # Métriques principales
            st.markdown("##### Métriques Gage R&R")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("% R&R", f"{results['%R&R']:.1f}%")
                st.markdown(f"**{results['Classification']}**")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("% Répétabilité", f"{results['%EV']:.1f}%")
                st.markdown(f"σ = {results['Sigma Repeatability']:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("% Reproductibilité", f"{results['%AV']:.1f}%")
                st.markdown(f"σ = {results['Sigma Reproducibility']:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("ndc", f"{results['ndc']}")
                st.markdown("Catégories distinctes")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Tableau détaillé des résultats
            st.markdown("##### Résultats détaillés")
            
            results_table = pd.DataFrame({
                'Composante': [
                    'Répétabilité (EV)',
                    'Reproductibilité (AV)',
                    'R&R (GRR)',
                    'Variation Pièce (PV)',
                    'Variation Totale (TV)'
                ],
                '6σ': [
                    f"{results['Repeatability (EV)']:.4f}",
                    f"{results['Reproducibility (AV)']:.4f}",
                    f"{results['R&R (GRR)']:.4f}",
                    f"{results['Part Variation (PV)']:.4f}",
                    f"{results['Total Variation (TV)']:.4f}"
                ],
                '% Variation': [
                    f"{results['%EV']:.1f}%",
                    f"{results['%AV']:.1f}%",
                    f"{results['%R&R']:.1f}%",
                    f"{results['%PV']:.1f}%",
                    "100%"
                ],
                '% Tolérance': [
                    f"{results['%Tol EV']:.1f}%",
                    f"{results['%Tol AV']:.1f}%",
                    f"{results['%Tol GRR']:.1f}%",
                    "-",
                    "-"
                ]
            })
            
            st.dataframe(results_table, use_container_width=True)
            
            # Graphique des composantes
            if 'components' in plots:
                st.plotly_chart(plots['components'], use_container_width=True)
            
            # Recommandations
            st.markdown("##### 🎯 Recommandations")
            
            if results['Classification'] == "Acceptable":
                st.success("""
                ✅ **Système de mesure EXCELLENT**
                
                Le système de mesure est statistiquement capable et peut être utilisé pour:
                - Le contrôle de la production
                - L'analyse de la capabilité des processus
                - La prise de décisions basées sur les données
                """)
            elif results['Classification'] == "Marginal":
                st.warning("""
                ⚠️ **Système de mesure MARGINAL**
                
                Améliorations recommandées:
                1. **Formation des opérateurs**: Standardiser les méthodes de mesure
                2. **Étalonnage**: Vérifier et ajuster l'équipement
                3. **Procédures**: Documenter clairement les procédures de mesure
                4. **Environnement**: Contrôler les conditions environnementales
                """)
            else:
                st.error("""
                ❌ **Système de mesure INACCEPTABLE**
                
                Actions prioritaires requises:
                1. **Équipement**: Investir dans un équipement de mesure plus précis
                2. **Audit**: Réaliser un audit complet du système de mesure
                3. **Formation**: Former intensivement tous les opérateurs
                4. **Processus**: Revoir complètement le processus de mesure
                
                **Ne pas utiliser ce système pour des décisions critiques!**
                """)
            
            # Générer un rapport
            st.markdown("---")
            st.markdown("##### 📄 Rapport d'analyse")
            
            # Créer un rapport texte
            report_text = f"""
            RAPPORT D'ANALYSE GAGE R&R
            ===========================
            
            Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            1. PARAMÈTRES DE L'ÉTUDE
            ------------------------
            - Nombre d'opérateurs: {st.session_state.df_gage['Opérateur'].nunique()}
            - Nombre de pièces: {st.session_state.df_gage['Pièce'].nunique()}
            - Nombre d'essais: {st.session_state.df_gage['Essai'].nunique()}
            - Total de mesures: {len(st.session_state.df_gage)}
            - Tolérance: {st.session_state.gage_config['tol_lower']} à {st.session_state.gage_config['tol_upper']}
            
            2. RÉSULTATS PRINCIPAUX
            -----------------------
            - %R&R Total: {results['%R&R']:.1f}%
            - Classification: {results['Classification']}
            - Nombre de catégories distinctes (ndc): {results['ndc']}
            
            3. COMPOSANTES DE VARIATION
            ---------------------------
            - Répétabilité (EV): {results['%EV']:.1f}%
            - Reproductibilité (AV): {results['%AV']:.1f}%
            - Variation Pièce (PV): {results['%PV']:.1f}%
            
            4. PAR RAPPORT À LA TOLÉRANCE
            -----------------------------
            - %R&R/Tolérance: {results['%Tol GRR']:.1f}%
            - %EV/Tolérance: {results['%Tol EV']:.1f}%
            - %AV/Tolérance: {results['%Tol AV']:.1f}%
            
            5. CONCLUSION
            -------------
            """
            
            if results['Classification'] == "Acceptable":
                report_text += "Le système de mesure est acceptable pour une utilisation en production."
            elif results['Classification'] == "Marginal":
                report_text += "Le système de mesure est marginal et nécessite des améliorations."
            else:
                report_text += "Le système de mesure est inacceptable et ne doit pas être utilisé pour des décisions critiques."
            
            # Bouton pour télécharger le rapport
            st.download_button(
                label="📥 Télécharger le rapport complet",
                data=report_text,
                file_name="rapport_gage_rr.txt",
                mime="text/plain"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Gage R&R Pro v2.0 | AIAG MSA 4th Edition | © 2024 Excellence Métrologique</p>
        <p style='font-size: 0.9em;'>Référence: AIAG Measurement Systems Analysis (MSA) - 4th Edition</p>
    </div>
    """,
    unsafe_allow_html=True
)
