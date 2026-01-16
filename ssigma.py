import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import io
import base64
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Six Sigma Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des données dans session_state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None
if 'defects_data' not in st.session_state:
    st.session_state.defects_data = None
if 'projects' not in st.session_state:
    st.session_state.projects = []

# Fonctions utilitaires
def calculate_cp_cpk(data, lsl, usl):
    """Calcul des indices de capacité"""
    if len(data) == 0:
        return 0, 0, 0, 0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1) if len(data) > 1 else 0
    
    cp = (usl - lsl) / (6 * std) if std != 0 else 0
    cpu = (usl - mean) / (3 * std) if std != 0 else 0
    cpl = (mean - lsl) / (3 * std) if std != 0 else 0
    cpk = min(cpu, cpl) if std != 0 else 0
    
    return cp, cpk, mean, std

def create_pareto_chart(defect_data):
    """Création du diagramme de Pareto"""
    df = pd.DataFrame(defect_data)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Aucune donnée disponible")
        return fig
    
    df = df.sort_values('count', ascending=False)
    df['cumulative_percentage'] = df['count'].cumsum() / df['count'].sum() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=df['defect_type'], y=df['count'], name="Nombre de défauts"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df['defect_type'], y=df['cumulative_percentage'], 
                  mode='lines+markers', name="Pourcentage cumulé"),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Diagramme de Pareto des Défauts",
        xaxis_title="Type de Défaut",
        yaxis_title="Nombre de Défauts",
        template="plotly_white",
        height=500
    )
    
    fig.update_yaxes(title_text="<b>Nombre</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>% Cumulé</b>", secondary_y=True, range=[0, 100])
    
    return fig

def create_control_chart(data, subgroup_size=5):
    """Création de carte de contrôle X-bar R"""
    if len(data) < subgroup_size * 2:  # Au moins 2 sous-groupes
        fig = go.Figure()
        fig.update_layout(title="Données insuffisantes pour créer la carte de contrôle")
        return fig
    
    subgroups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
    subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
    subgroup_ranges = [np.ptp(subgroup) for subgroup in subgroups]
    
    overall_mean = np.mean(subgroup_means)
    r_bar = np.mean(subgroup_ranges)
    
    # Coefficients pour X-bar R chart
    a2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
    d4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004}
    
    ucl_x = overall_mean + a2.get(subgroup_size, 0.577) * r_bar
    lcl_x = overall_mean - a2.get(subgroup_size, 0.577) * r_bar
    cl_x = overall_mean
    
    ucl_r = d4.get(subgroup_size, 2.114) * r_bar
    cl_r = r_bar
    
    # Création des graphiques
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Carte de contrôle X-bar", "Carte de contrôle R"))
    
    # X-bar chart
    fig.add_trace(
        go.Scatter(y=subgroup_means, mode='lines+markers', name='Moyennes'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=[ucl_x]*len(subgroup_means), mode='lines', 
                  name=f'UCL: {ucl_x:.2f}', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=[lcl_x]*len(subgroup_means), mode='lines',
                  name=f'LCL: {lcl_x:.2f}', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=[cl_x]*len(subgroup_means), mode='lines',
                  name=f'CL: {cl_x:.2f}', line=dict(color='green', dash='dot')),
        row=1, col=1
    )
    
    # R chart
    fig.add_trace(
        go.Scatter(y=subgroup_ranges, mode='lines+markers', name='Étendues'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=[ucl_r]*len(subgroup_ranges), mode='lines',
                  name=f'UCL: {ucl_r:.2f}', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=[cl_r]*len(subgroup_ranges), mode='lines',
                  name=f'CL: {cl_r:.2f}', line=dict(color='green', dash='dot')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    return fig

def create_ishikawa_diagram(causes, main_effect="Problème"):
    """Création d'un diagramme d'Ishikawa (arête de poisson)"""
    fig = go.Figure()
    
    # Arête centrale
    fig.add_trace(go.Scatter(x=[0, 6], y=[0, 0], mode='lines', 
                           line=dict(color='black', width=3)))
    
    # Tête du poisson
    fig.add_trace(go.Scatter(x=[6, 7, 6], y=[0, 0.5, -0.5], 
                           fill='toself', mode='lines', line=dict(color='black')))
    
    # Effet principal
    fig.add_annotation(x=6.5, y=0, text=main_effect, showarrow=False,
                      font=dict(size=12, color='red'))
    
    # Catégories principales
    categories = list(causes.keys())
    angles = [45, 30, 0, -30, -45]
    
    for i, category in enumerate(categories):
        if i >= len(angles):
            angle = 0
        else:
            angle = angles[i]
        length = 2
        
        # Branche principale
        end_x = length * np.cos(np.radians(angle))
        end_y = length * np.sin(np.radians(angle))
        
        fig.add_trace(go.Scatter(x=[0, end_x], y=[0, end_y], 
                               mode='lines', line=dict(color='blue', width=2)))
        
        # Nom de la catégorie
        fig.add_annotation(x=end_x*1.1, y=end_y*1.1, text=category,
                          showarrow=False, font=dict(size=10, color='darkblue'))
        
        # Sous-causes
        sub_causes = causes[category]
        for j, cause in enumerate(sub_causes):
            sub_x = end_x + 0.5 * np.cos(np.radians(angle))
            sub_y = end_y + 0.5 * np.sin(np.radians(angle)) + (j - len(sub_causes)/2) * 0.2
            
            fig.add_trace(go.Scatter(x=[end_x, sub_x], y=[end_y, sub_y],
                                   mode='lines', line=dict(color='gray', width=1)))
            
            fig.add_annotation(x=sub_x*1.05, y=sub_y, text=cause,
                              showarrow=False, font=dict(size=8, color='gray'))
    
    fig.update_layout(
        title=f"Diagramme de Causes et Effets (Ishikawa) - {main_effect}",
        xaxis=dict(visible=False, range=[-3, 8]),
        yaxis=dict(visible=False, range=[-3, 3]),
        showlegend=False,
        height=600,
        width=800
    )
    
    return fig

def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Utilisez CSV ou Excel.")
            return None
        
        st.session_state.uploaded_data = data
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return None

# Interface principale
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        app_mode = st.selectbox(
            "Sélectionnez une section",
            ["📊 Accueil", "📈 Analyse des Données", "📉 Diagramme de Pareto", 
             "🐟 Diagramme Ishikawa", "⚙️ Analyse de Capacité", "🔄 Cartes de Contrôle"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Données")
        
        # Upload de fichier
        uploaded_file = st.file_uploader("Téléchargez vos données", 
                                        type=['csv', 'xlsx'],
                                        help="Formats acceptés: CSV, Excel")
        
        if uploaded_file is not None:
            if st.button("📂 Charger les données"):
                data = load_data(uploaded_file)
                if data is not None:
                    st.success(f"Données chargées avec succès! {data.shape[0]} lignes, {data.shape[1]} colonnes")
        
        st.markdown("---")
        st.markdown("#### 📊 À propos")
        st.markdown("""
        **Six Sigma Analytics Suite**
        Version 1.0
        
        Fonctionnalités :
        - Analyse statistique des données
        - Diagramme de Pareto
        - Diagramme Ishikawa
        - Analyse de capacité (Cp, Cpk)
        - Cartes de contrôle
        """)
    
    # Contenu principal
    if app_mode == "📊 Accueil":
        st.markdown('<h1 class="main-header">📊 Six Sigma Analytics Suite</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📈</h3>
                <p>Analyse Statistique</p>
                <p>Tests d'hypothèses, normalité, intervalles de confiance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📉</h3>
                <p>Diagramme de Pareto</p>
                <p>Analyse 80/20 des causes de défauts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>⚙️</h3>
                <p>Analyse de Capacité</p>
                <p>Calcul des indices Cp, Cpk, niveau Sigma</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📝 Instructions d'utilisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Téléchargez vos données**
            - Cliquez sur "Téléchargez vos données" dans la barre latérale
            - Sélectionnez un fichier CSV ou Excel
            - Cliquez sur "Charger les données"
            
            **2. Naviguez entre les sections**
            - Utilisez le menu déroulant dans la barre latérale
            - Chaque section offre des analyses spécifiques
            """)
        
        with col2:
            st.markdown("""
            **3. Format des données recommandé**
            - **Pour l'analyse statistique**: Colonnes numériques
            - **Pour Pareto**: Colonnes "defect_type" et "count"
            - **Pour Ishikawa**: Liste des causes par catégorie
            
            **4. Exports disponibles**
            - Graphiques en PNG
            - Données en CSV/Excel
            - Rapports PDF
            """)
        
        # Aperçu des données si disponibles
        if st.session_state.uploaded_data is not None:
            st.markdown("### 📋 Aperçu des données chargées")
            st.dataframe(st.session_state.uploaded_data.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre de lignes", st.session_state.uploaded_data.shape[0])
            with col2:
                st.metric("Nombre de colonnes", st.session_state.uploaded_data.shape[1])
    
    elif app_mode == "📈 Analyse des Données":
        st.markdown('<h1 class="main-header">📈 Analyse Statistique des Données</h1>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Veuillez d'abord charger des données dans la section Accueil")
            return
        
        data = st.session_state.uploaded_data
        
        # Sélection de colonne
        st.markdown('<h3 class="sub-header">Sélection des données à analyser</h3>', unsafe_allow_html=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ Aucune colonne numérique trouvée dans les données")
            return
        
        selected_col = st.selectbox("Sélectionnez une colonne numérique à analyser", numeric_cols)
        st.session_state.selected_column = selected_col
        
        # Affichage des statistiques descriptives
        analysis_data = data[selected_col].dropna().values
        
        if len(analysis_data) == 0:
            st.error("❌ La colonne sélectionnée ne contient pas de données valides")
            return
        
        st.session_state.analysis_data = analysis_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taille échantillon", len(analysis_data))
        with col2:
            st.metric("Moyenne", f"{np.mean(analysis_data):.4f}")
        with col3:
            st.metric("Écart-type", f"{np.std(analysis_data, ddof=1):.4f}")
        with col4:
            st.metric("Médiane", f"{np.median(analysis_data):.4f}")
        
        # Histogramme
        st.markdown('<h3 class="sub-header">Distribution des données</h3>', unsafe_allow_html=True)
        
        fig = px.histogram(data, x=selected_col, nbins=30, 
                          title=f"Distribution de {selected_col}",
                          marginal="box")
        
        # Ajouter courbe normale si assez de données
        if len(analysis_data) > 1:
            x_range = np.linspace(min(analysis_data), max(analysis_data), 100)
            pdf = stats.norm.pdf(x_range, np.mean(analysis_data), np.std(analysis_data, ddof=1))
            fig.add_trace(go.Scatter(x=x_range, y=pdf * len(analysis_data) * (max(analysis_data)-min(analysis_data))/30,
                                   mode='lines', name='Distribution normale', line=dict(color='red', width=2)))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tests statistiques
        st.markdown('<h3 class="sub-header">Tests Statistiques</h3>', unsafe_allow_html=True)
        
        test_type = st.radio("Choisissez un test", 
                            ["Test de normalité (Shapiro-Wilk)", 
                             "Intervalle de confiance",
                             "Test t à un échantillon"])
        
        if test_type == "Test de normalité (Shapiro-Wilk)":
            if len(analysis_data) >= 3 and len(analysis_data) <= 5000:
                stat, p_value = stats.shapiro(analysis_data)
                st.write(f"**Statistique de test:** {stat:.4f}")
                st.write(f"**p-value:** {p_value:.4f}")
                if p_value > 0.05:
                    st.success("✅ Les données suivent une distribution normale (p > 0.05)")
                else:
                    st.warning("⚠️ Les données ne suivent pas une distribution normale (p ≤ 0.05)")
            else:
                st.warning("Le test de Shapiro-Wilk nécessite entre 3 et 5000 observations")
        
        elif test_type == "Intervalle de confiance":
            confidence = st.slider("Niveau de confiance", 0.80, 0.99, 0.95, 0.01)
            
            if len(analysis_data) > 1:
                ci_low, ci_high = stats.t.interval(confidence, len(analysis_data)-1, 
                                                  loc=np.mean(analysis_data), 
                                                  scale=stats.sem(analysis_data))
                st.write(f"**Intervalle de confiance à {confidence*100}%:**")
                st.write(f"**[{ci_low:.4f}, {ci_high:.4f}]**")
                st.write(f"**Largeur de l'intervalle:** {ci_high - ci_low:.4f}")
        
        elif test_type == "Test t à un échantillon":
            pop_mean = st.number_input("Moyenne populationnelle hypothétique", 
                                      value=float(np.mean(analysis_data)))
            
            if len(analysis_data) > 1:
                t_stat, p_value = stats.ttest_1samp(analysis_data, pop_mean)
                st.write(f"**Statistique t:** {t_stat:.4f}")
                st.write(f"**p-value (bilatéral):** {p_value:.4f}")
                
                # Test unilatéral
                if np.mean(analysis_data) > pop_mean:
                    p_value_unilateral = p_value / 2
                    st.write(f"**p-value (unilatéral >):** {p_value_unilateral:.4f}")
                else:
                    p_value_unilateral = p_value / 2
                    st.write(f"**p-value (unilatéral <):** {p_value_unilateral:.4f}")
    
    elif app_mode == "📉 Diagramme de Pareto":
        st.markdown('<h1 class="main-header">📉 Diagramme de Pareto</h1>', unsafe_allow_html=True)
        
        # Options pour les données
        st.markdown('<h3 class="sub-header">Source des données</h3>', unsafe_allow_html=True)
        
        data_option = st.radio("Choisissez la source des données",
                              ["Données uploadées", "Saisie manuelle", "Exemple"])
        
        defect_data = None
        
        if data_option == "Données uploadées":
            if st.session_state.uploaded_data is not None:
                data = st.session_state.uploaded_data
                
                # Essayer de détecter automatiquement les colonnes
                if 'defect_type' in data.columns and 'count' in data.columns:
                    defect_data = data[['defect_type', 'count']]
                    if 'category' in data.columns:
                        defect_data['category'] = data['category']
                else:
                    st.warning("Les colonnes 'defect_type' et 'count' sont requises")
                    
                    # Permettre la sélection manuelle des colonnes
                    col1, col2 = st.columns(2)
                    with col1:
                        defect_col = st.selectbox("Colonne des types de défauts", data.columns)
                    with col2:
                        count_col = st.selectbox("Colonne des comptes", 
                                                data.select_dtypes(include=[np.number]).columns.tolist())
                    
                    if defect_col and count_col:
                        defect_data = pd.DataFrame({
                            'defect_type': data[defect_col].astype(str),
                            'count': pd.to_numeric(data[count_col], errors='coerce')
                        }).dropna()
        
        elif data_option == "Saisie manuelle":
            num_defects = st.number_input("Nombre de types de défauts", min_value=1, max_value=50, value=5)
            
            defect_data = []
            for i in range(int(num_defects)):
                with st.expander(f"Défaut {i+1}", expanded=(i < 3)):
                    col1, col2 = st.columns(2)
                    with col1:
                        defect_type = st.text_input(f"Type de défaut {i+1}", 
                                                   value=f"Défaut {i+1}", 
                                                   key=f"defect_{i}")
                    with col2:
                        count = st.number_input(f"Nombre {i+1}", 
                                               min_value=1, 
                                               value=100-(i*10),
                                               key=f"count_{i}")
                    defect_data.append({'defect_type': defect_type, 'count': count})
            
            defect_data = pd.DataFrame(defect_data)
        
        else:  # Exemple
            defect_types = ['Rayures', 'Inclusions', 'Déformations', 
                          'Fissures', 'Porosités', 'Décalage', 'Couleur incorrecte']
            counts = [150, 120, 95, 80, 65, 45, 30]
            defect_data = pd.DataFrame({
                'defect_type': defect_types,
                'count': counts,
                'category': ['Surface', 'Surface', 'Structure', 'Structure', 
                           'Structure', 'Dimension', 'Apparence']
            })
        
        if defect_data is not None and not defect_data.empty:
            # Affichage des données
            st.markdown('<h3 class="sub-header">Données des défauts</h3>', unsafe_allow_html=True)
            st.dataframe(defect_data.sort_values('count', ascending=False), 
                        use_container_width=True)
            
            # Diagramme de Pareto
            st.markdown('<h3 class="sub-header">Diagramme de Pareto</h3>', unsafe_allow_html=True)
            fig = create_pareto_chart(defect_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse 80/20
            total_defects = defect_data['count'].sum()
            sorted_defects = defect_data.sort_values('count', ascending=False)
            sorted_defects['cumulative'] = sorted_defects['count'].cumsum()
            sorted_defects['cumulative_percentage'] = sorted_defects['cumulative'] / total_defects * 100
            
            st.markdown("### 📊 Analyse 80/20")
            
            # Trouver le point 80%
            if not sorted_defects[sorted_defects['cumulative_percentage'] >= 80].empty:
                pareto_point = sorted_defects[sorted_defects['cumulative_percentage'] >= 80].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total défauts", total_defects)
                with col2:
                    top_defects = len(sorted_defects[sorted_defects['cumulative_percentage'] <= 80])
                    st.metric("Défauts principaux (80%)", f"{top_defects} types")
                with col3:
                    percentage = (sorted_defects.iloc[0]['count'] / total_defects * 100)
                    st.metric("Défaut principal", f"{percentage:.1f}%")
                
                st.info(f"**Règle 80/20:** {pareto_point['defect_type']} représente le seuil des 80% des défauts totaux")
            
            # Export des données
            st.markdown("### 💾 Export")
            csv = defect_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Télécharger les données (CSV)",
                data=csv,
                file_name="donnees_pareto.csv",
                mime="text/csv"
            )
    
    elif app_mode == "🐟 Diagramme Ishikawa":
        st.markdown('<h1 class="main-header">🐟 Diagramme de Causes et Effets (Ishikawa)</h1>', unsafe_allow_html=True)
        
        # Catégories standard 6M
        default_categories = {
            'Méthode': ['Procédures non standardisées', 'Formation insuffisante', 
                       'Mauvaise planification', 'Documentation incomplète'],
            'Matériel': ['Équipement défectueux', 'Usure normale', 
                        'Mauvaise maintenance', 'Calibration incorrecte'],
            'Main d\'œuvre': ['Manque de compétences', 'Fatigue', 
                             'Erreur humaine', 'Manque de motivation'],
            'Environnement': ['Température élevée', 'Humidité', 
                            'Éclairage insuffisant', 'Vibrations'],
            'Mesure': ['Instrument non étalonné', 'Méthode de mesure incorrecte', 
                      'Erreur de lecture', 'Précision insuffisante'],
            'Matériaux': ['Qualité variable', 'Fournisseur non fiable', 
                         'Stockage inadéquat', 'Spécifications incorrectes']
        }
        
        # Interface de saisie
        st.markdown("### 🎯 Effet/Problème Principal")
        main_effect = st.text_input("Décrivez l'effet ou le problème à analyser", 
                                   value="Taux de défauts élevé sur la ligne de production")
        
        st.markdown("### 📋 Causes par Catégorie (Méthode 6M)")
        
        causes = {}
        
        for category in default_categories.keys():
            with st.expander(f"**{category}**", expanded=True):
                # Pré-remplir avec les causes par défaut
                default_causes = default_categories[category]
                causes_text = "\n".join(default_causes)
                
                # Zone de texte pour les causes
                causes_input = st.text_area(
                    f"Causes pour {category} (une par ligne)",
                    value=causes_text,
                    height=120,
                    key=f"causes_{category}"
                )
                
                # Traiter les causes
                causes_list = [c.strip() for c in causes_input.split('\n') if c.strip()]
                causes[category] = causes_list
        
        # Bouton de génération
        if st.button("🎨 Générer le Diagramme Ishikawa", type="primary"):
            if main_effect.strip():
                st.markdown(f"### 📊 Diagramme d'Ishikawa : {main_effect}")
                
                # Créer le diagramme
                fig = create_ishikawa_diagram(causes, main_effect)
                st.plotly_chart(fig, use_container_width=True)
                
                # Option de téléchargement
                img_bytes = fig.to_image(format="png", width=1000, height=600)
                st.download_button(
                    label="📥 Télécharger le diagramme (PNG)",
                    data=img_bytes,
                    file_name=f"diagramme_ishikawa_{main_effect[:50].replace(' ', '_')}.png",
                    mime="image/png"
                )
                
                # Liste des causes
                st.markdown("### 📝 Liste complète des causes")
                causes_list = []
                for category, cause_items in causes.items():
                    for cause in cause_items:
                        causes_list.append({'Catégorie': category, 'Cause': cause})
                
                causes_df = pd.DataFrame(causes_list)
                st.dataframe(causes_df, use_container_width=True)
                
                # Export CSV
                csv = causes_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Exporter les causes (CSV)",
                    data=csv,
                    file_name="causes_ishikawa.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Veuillez saisir un effet principal")
    
    elif app_mode == "⚙️ Analyse de Capacité":
        st.markdown('<h1 class="main-header">⚙️ Analyse de Capacité du Processus</h1>', unsafe_allow_html=True)
        
        # Sélection des données
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Veuillez d'abord charger des données dans la section Accueil")
            
            # Option pour utiliser des données d'exemple
            if st.button("📊 Utiliser des données d'exemple"):
                np.random.seed(42)
                example_data = np.random.normal(loc=100, scale=5, size=200)
                st.session_state.analysis_data = example_data
                st.session_state.selected_column = "Exemple (normale, μ=100, σ=5)"
                st.rerun()
            return
        
        data = st.session_state.uploaded_data
        
        # Sélection de colonne
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ Aucune colonne numérique trouvée dans les données")
            return
        
        selected_col = st.selectbox("Sélectionnez une colonne numérique", numeric_cols)
        analysis_data = data[selected_col].dropna().values
        
        if len(analysis_data) < 2:
            st.error("❌ Données insuffisantes pour l'analyse de capacité")
            return
        
        # Paramètres de spécification
        st.markdown('<h3 class="sub-header">Spécifications du processus</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lsl = st.number_input("Limite Inférieure de Spécification (LSL)", 
                                 value=float(np.percentile(analysis_data, 5)))
        with col2:
            target = st.number_input("Valeur Cible", 
                                    value=float(np.mean(analysis_data)))
        with col3:
            usl = st.number_input("Limite Supérieure de Spécification (USL)", 
                                 value=float(np.percentile(analysis_data, 95)))
        
        # Calculs de capacité
        cp, cpk, mean, std = calculate_cp_cpk(analysis_data, lsl, usl)
        
        # Calcul PPM et niveau Sigma
        if std > 0:
            ppm_below = stats.norm.cdf(lsl, mean, std) * 1e6
            ppm_above = (1 - stats.norm.cdf(usl, mean, std)) * 1e6
            ppm_total = ppm_below + ppm_above
            sigma_level = abs(stats.norm.ppf(ppm_total/2e6))
        else:
            ppm_below = ppm_above = ppm_total = 0
            sigma_level = 0
        
        # Affichage des métriques
        st.markdown('<h3 class="sub-header">Indices de Capacité</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "green" if cp >= 1.33 else "orange" if cp >= 1.0 else "red"
            st.metric("Cp", f"{cp:.3f}", 
                     delta="✅ Capable" if cp >= 1.33 else "⚠️ Limite" if cp >= 1.0 else "❌ Incapable",
                     delta_color="normal")
        
        with col2:
            color = "green" if cpk >= 1.33 else "orange" if cpk >= 1.0 else "red"
            st.metric("Cpk", f"{cpk:.3f}", 
                     delta="✅ Centré" if cpk >= 1.33 else "⚠️ Limite" if cpk >= 1.0 else "❌ Non centré",
                     delta_color="normal")
        
        with col3:
            st.metric("PPM Total", f"{ppm_total:.0f}")
        
        with col4:
            st.metric("Niveau Sigma", f"{sigma_level:.2f}σ")
        
        # Interprétation
        st.markdown("### 📊 Interprétation")
        
        if cp >= 1.33 and cpk >= 1.33:
            st.success("""
            **✅ Processus capable et centré**
            - Le processus est statistiquement stable
            - La dispersion est inférieure aux limites de spécification
            - Le processus est centré sur la valeur cible
            """)
        elif cp >= 1.33 and cpk < 1.33:
            st.warning("""
            **⚠️ Processus capable mais non centré**
            - La dispersion est acceptable
            - Mais le processus n'est pas centré
            - Ajustez le centrage pour améliorer Cpk
            """)
        else:
            st.error("""
            **❌ Processus non capable**
            - La dispersion est trop grande
            - Action nécessaire pour réduire la variabilité
            - Revoyez le processus
            """)
        
        # Visualisation
        st.markdown("### 📈 Visualisation de la capacité")
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Distribution du processus", 
                                        "Indice Cpk"))
        
        # Histogramme avec spécifications
        fig.add_trace(
            go.Histogram(x=analysis_data, nbinsx=30, name='Distribution',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=1
        )
        
        # Lignes de spécification
        for spec, name, color in [(lsl, 'LSL', 'red'), (target, 'Cible', 'green'), 
                                 (usl, 'USL', 'red'), (mean, 'Moyenne', 'blue')]:
            fig.add_vline(x=spec, line_dash="dash", line_color=color,
                         annotation_text=name, row=1, col=1)
        
        # Courbe normale
        x_range = np.linspace(min(analysis_data.min(), lsl*0.9), 
                             max(analysis_data.max(), usl*1.1), 1000)
        if std > 0:
            pdf = stats.norm.pdf(x_range, mean, std)
            fig.add_trace(
                go.Scatter(x=x_range, y=pdf, mode='lines', 
                          name='Distribution normale',
                          line=dict(color='darkblue', width=2)),
                row=1, col=1
            )
        
        # Jauge Cpk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cpk,
                title={'text': "Cpk"},
                domain={'row': 0, 'column': 1},
                gauge={
                    'axis': {'range': [0, max(2.0, cpk*1.5)]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "red"},
                        {'range': [1, 1.33], 'color': "yellow"},
                        {'range': [1.33, 2], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.33
                    }
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "🔄 Cartes de Contrôle":
        st.markdown('<h1 class="main-header">🔄 Cartes de Contrôle Statistiques</h1>', unsafe_allow_html=True)
        
        # Options pour les données
        st.markdown('<h3 class="sub-header">Source des données</h3>', unsafe_allow_html=True)
        
        data_option = st.radio("Source des données", 
                              ["Données uploadées", "Données d'exemple"])
        
        if data_option == "Données uploadées":
            if st.session_state.uploaded_data is None:
                st.warning("⚠️ Veuillez d'abord charger des données")
                return
            
            data = st.session_state.uploaded_data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.error("❌ Aucune colonne numérique trouvée")
                return
            
            selected_col = st.selectbox("Colonne à analyser", numeric_cols)
            analysis_data = data[selected_col].dropna().values
            
        else:  # Données d'exemple
            np.random.seed(42)
            # Créer des données avec une dérive
            base_data = np.random.normal(loc=100, scale=5, size=50)
            drift_data = np.random.normal(loc=105, scale=5, size=50)
            analysis_data = np.concatenate([base_data, drift_data])
            selected_col = "Exemple (avec dérive)"
        
        if len(analysis_data) < 10:
            st.error("❌ Données insuffisantes pour les cartes de contrôle (minimum 10 points)")
            return
        
        # Paramètres
        st.markdown('<h3 class="sub-header">Paramètres de la carte de contrôle</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            subgroup_size = st.slider("Taille du sous-groupe", 2, 10, 5,
                                     help="Nombre de points par sous-groupe")
        
        with col2:
            show_r_chart = st.checkbox("Afficher la carte R", value=True)
        
        # Création des cartes de contrôle
        fig = create_control_chart(analysis_data, subgroup_size)
        
        if not show_r_chart:
            # Ne montrer que la carte X-bar
            fig = go.Figure()
            
            subgroups = [analysis_data[i:i+subgroup_size] for i in range(0, len(analysis_data), subgroup_size)]
            subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
            overall_mean = np.mean(subgroup_means)
            r_bar = np.mean([np.ptp(subgroup) for subgroup in subgroups])
            
            # Coefficients pour X-bar R chart
            a2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
            ucl_x = overall_mean + a2.get(subgroup_size, 0.577) * r_bar
            lcl_x = overall_mean - a2.get(subgroup_size, 0.577) * r_bar
            
            fig.add_trace(
                go.Scatter(y=subgroup_means, mode='lines+markers', 
                          name='Moyennes des sous-groupes',
                          line=dict(color='blue', width=2))
            )
            fig.add_trace(
                go.Scatter(y=[ucl_x]*len(subgroup_means), mode='lines', 
                          name=f'UCL: {ucl_x:.2f}', 
                          line=dict(color='red', dash='dash'))
            )
            fig.add_trace(
                go.Scatter(y=[lcl_x]*len(subgroup_means), mode='lines',
                          name=f'LCL: {lcl_x:.2f}', 
                          line=dict(color='red', dash='dash'))
            )
            fig.add_trace(
                go.Scatter(y=[overall_mean]*len(subgroup_means), mode='lines',
                          name=f'CL: {overall_mean:.2f}', 
                          line=dict(color='green', dash='dot'))
            )
            
            fig.update_layout(
                title=f"Carte de contrôle X-bar - {selected_col}",
                xaxis_title="Numéro du sous-groupe",
                yaxis_title="Moyenne",
                height=500,
                template="plotly_white"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des points hors contrôle
        st.markdown("### 📊 Analyse des points hors contrôle")
        
        if data_option == "Données uploadées" and len(analysis_data) >= subgroup_size * 5:
            # Détection des points hors contrôle (règles Western Electric)
            subgroups = [analysis_data[i:i+subgroup_size] for i in range(0, len(analysis_data), subgroup_size)]
            subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
            overall_mean = np.mean(subgroup_means)
            r_bar = np.mean([np.ptp(subgroup) for subgroup in subgroups])
            
            a2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
            ucl_x = overall_mean + a2.get(subgroup_size, 0.577) * r_bar
            lcl_x = overall_mean - a2.get(subgroup_size, 0.577) * r_bar
            
            out_of_control = []
            for i, mean_val in enumerate(subgroup_means):
                if mean_val > ucl_x or mean_val < lcl_x:
                    out_of_control.append({
                        'Sous-groupe': i+1,
                        'Moyenne': mean_val,
                        'Statut': 'Hors limite'
                    })
            
            if out_of_control:
                st.warning(f"**{len(out_of_control)} points hors contrôle détectés**")
                out_df = pd.DataFrame(out_of_control)
                st.dataframe(out_df, use_container_width=True)
            else:
                st.success("✅ Aucun point hors contrôle détecté")
        
        # Téléchargement
        st.markdown("### 💾 Export")
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Télécharger la carte (PNG)",
            data=img_bytes,
            file_name="carte_controle.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
