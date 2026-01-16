import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import io
import warnings
import base64
warnings.filterwarnings('ignore')

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

# Initialisation des données en mémoire (pour Streamlit Cloud)
def init_session_state():
    """Initialise les données en mémoire dans st.session_state"""
    if 'projects_df' not in st.session_state:
        st.session_state.projects_df = pd.DataFrame(columns=[
            'id', 'project_name', 'description', 'status', 
            'start_date', 'end_date', 'sigma_level', 'created_at'
        ])
    
    if 'measurements_df' not in st.session_state:
        st.session_state.measurements_df = pd.DataFrame(columns=[
            'id', 'project_id', 'measurement_date', 'value', 
            'category', 'subgroup_size'
        ])
    
    if 'defects_df' not in st.session_state:
        st.session_state.defects_df = pd.DataFrame(columns=[
            'id', 'project_id', 'defect_type', 'count', 'category', 'date'
        ])
    
    # Générer des données d'exemple pour la démonstration
    if st.session_state.projects_df.empty:
        example_projects = pd.DataFrame([
            {
                'id': 1,
                'project_name': 'Amélioration du Processus de Fabrication',
                'description': 'Réduction des défauts dans la ligne de production',
                'status': 'Actif',
                'start_date': '2024-01-15',
                'end_date': '2024-06-30',
                'sigma_level': 3.5,
                'created_at': '2024-01-15 10:30:00'
            },
            {
                'id': 2,
                'project_name': 'Optimisation du Temps de Cycle',
                'description': 'Réduction du temps de cycle de production de 20%',
                'status': 'Terminé',
                'start_date': '2023-10-01',
                'end_date': '2024-01-31',
                'sigma_level': 4.2,
                'created_at': '2023-10-01 09:15:00'
            }
        ])
        st.session_state.projects_df = example_projects
    
    if st.session_state.measurements_df.empty:
        np.random.seed(42)
        example_measurements = pd.DataFrame({
            'id': range(1, 101),
            'project_id': np.random.choice([1, 2], 100),
            'measurement_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'value': np.random.normal(100, 10, 100),
            'category': np.random.choice(['Dimension', 'Poids', 'Température', 'Pression'], 100),
            'subgroup_size': 5
        })
        st.session_state.measurements_df = example_measurements
    
    if st.session_state.defects_df.empty:
        example_defects = pd.DataFrame({
            'id': range(1, 31),
            'project_id': 1,
            'defect_type': np.random.choice([
                'Rayures', 'Inclusions', 'Déformations', 'Fissures', 
                'Porosités', 'Décalage', 'Couleur incorrecte'
            ], 30),
            'count': np.random.randint(10, 100, 30),
            'category': np.random.choice(['Surface', 'Structure', 'Dimension', 'Apparence'], 30),
            'date': pd.date_range(start='2024-01-01', periods=30, freq='D')
        })
        st.session_state.defects_df = example_defects

# Fonctions utilitaires
def calculate_cp_cpk(data, lsl, usl):
    """Calcul des indices de capacité"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    cp = (usl - lsl) / (6 * std) if std != 0 else 0
    cpu = (usl - mean) / (3 * std) if std != 0 else 0
    cpl = (mean - lsl) / (3 * std) if std != 0 else 0
    cpk = min(cpu, cpl) if std != 0 else 0
    
    return cp, cpk, mean, std

def create_pareto_chart(defect_data):
    """Création du diagramme de Pareto"""
    df = pd.DataFrame(defect_data)
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
    
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    return fig

def create_ishikawa_diagram(causes):
    """Création d'un diagramme d'Ishikawa (arête de poisson)"""
    fig = go.Figure()
    
    # Arête centrale
    fig.add_trace(go.Scatter(x=[0, 6], y=[0, 0], mode='lines', 
                           line=dict(color='black', width=3)))
    
    # Tête du poisson
    fig.add_trace(go.Scatter(x=[6, 7, 6], y=[0, 0.5, -0.5], 
                           fill='toself', mode='lines', line=dict(color='black')))
    
    # Catégories principales
    categories = list(causes.keys())
    angles = [45, 30, 0, -30, -45]
    
    for i, category in enumerate(categories):
        angle = angles[i % len(angles)]
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
        title="Diagramme de Causes et Effets (Ishikawa)",
        xaxis=dict(visible=False, range=[-3, 8]),
        yaxis=dict(visible=False, range=[-3, 3]),
        showlegend=False,
        height=600,
        width=1000
    )
    
    return fig

# Fonction pour générer un ID unique
def generate_id(existing_ids):
    """Génère un nouvel ID unique"""
    if len(existing_ids) == 0:
        return 1
    return max(existing_ids) + 1

# Interface principale
def main():
    # Initialiser les données en mémoire
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        app_mode = st.selectbox(
            "Sélectionnez une section",
            ["🏠 Tableau de bord", "📁 Gestion des Projets", "📊 Analyse Statistique",
             "📈 Cartes de Contrôle", "📉 Diagramme de Pareto", "🐟 Diagramme Ishikawa",
             "⚙️ Indices de Capacité", "📤 Import/Export"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Projet Actif")
        
        # Sélection du projet
        projects_df = st.session_state.projects_df
        if not projects_df.empty:
            project_names = projects_df['project_name'].tolist()
            selected_project = st.selectbox("Choisir un projet", project_names)
            project_id = projects_df[projects_df['project_name'] == selected_project]['id'].values[0]
        else:
            selected_project = "Aucun projet"
            project_id = None
        
        st.markdown("---")
        st.markdown("### ⚙️ Paramètres")
        confidence_level = st.slider("Niveau de confiance", 0.90, 0.99, 0.95)
        
        st.markdown("---")
        st.markdown("#### 📊 À propos")
        st.markdown("""
        **Six Sigma Analytics Suite**
        Version 2.0
        
        Fonctionnalités :
        - Gestion des projets DMAIC
        - Analyse statistique avancée
        - Cartes de contrôle interactives
        - Diagrammes d'analyse des défauts
        """)
    
    # Contenu principal
    if app_mode == "🏠 Tableau de bord":
        st.markdown('<h1 class="main-header">📊 Six Sigma Analytics Suite</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_projects = len(projects_df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_projects}</h3>
                <p>Projets Totaux</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            active_projects = len(projects_df[projects_df['status'] == 'Actif'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{active_projects}</h3>
                <p>Projets Actifs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_sigma = projects_df['sigma_level'].mean() if not projects_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_sigma:.2f}σ</h3>
                <p>Niveau Sigma Moyen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_measurements = len(st.session_state.measurements_df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_measurements}</h3>
                <p>Mesures</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphique des projets par statut
        if not projects_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="sub-header">Projets par Statut</h3>', unsafe_allow_html=True)
                status_counts = projects_df['status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h3 class="sub-header">Niveau Sigma des Projets</h3>', unsafe_allow_html=True)
                if 'sigma_level' in projects_df.columns:
                    fig = px.histogram(projects_df, x='sigma_level', nbins=10,
                                     color_discrete_sequence=['#636EFA'])
                    fig.update_layout(height=300, xaxis_title="Niveau Sigma",
                                    yaxis_title="Nombre de Projets")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Dernières mesures
        st.markdown('<h3 class="sub-header">Dernières Mesures</h3>', unsafe_allow_html=True)
        if project_id:
            recent_measurements = st.session_state.measurements_df[
                st.session_state.measurements_df['project_id'] == project_id
            ].sort_values('measurement_date', ascending=False).head(10)
            if not recent_measurements.empty:
                st.dataframe(recent_measurements, use_container_width=True)
    
    elif app_mode == "📁 Gestion des Projets":
        st.markdown('<h1 class="main-header">📁 Gestion des Projets Six Sigma</h1>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📝 Nouveau Projet", "📋 Liste des Projets", "✏️ Modifier Projet"])
        
        with tab1:
            with st.form("new_project_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    project_name = st.text_input("Nom du Projet")
                    description = st.text_area("Description")
                    sigma_level = st.number_input("Objectif Sigma", min_value=1.0, max_value=6.0, value=3.0)
                
                with col2:
                    status = st.selectbox("Statut", ["Planifié", "Actif", "En attente", "Terminé", "Annulé"])
                    start_date = st.date_input("Date de début")
                    end_date = st.date_input("Date de fin prévue")
                
                if st.form_submit_button("💾 Créer le Projet"):
                    # Générer un nouvel ID
                    existing_ids = st.session_state.projects_df['id'].tolist() if not st.session_state.projects_df.empty else []
                    new_id = generate_id(existing_ids)
                    
                    # Ajouter le nouveau projet
                    new_project = pd.DataFrame([{
                        'id': new_id,
                        'project_name': project_name,
                        'description': description,
                        'status': status,
                        'start_date': start_date,
                        'end_date': end_date,
                        'sigma_level': sigma_level,
                        'created_at': pd.Timestamp.now()
                    }])
                    
                    st.session_state.projects_df = pd.concat(
                        [st.session_state.projects_df, new_project], 
                        ignore_index=True
                    )
                    
                    st.success(f"Projet '{project_name}' créé avec succès!")
                    st.rerun()
        
        with tab2:
            projects_df = st.session_state.projects_df.sort_values('created_at', ascending=False)
            if not projects_df.empty:
                st.dataframe(projects_df, use_container_width=True)
                
                # Options d'export
                csv = projects_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter en CSV",
                    data=csv,
                    file_name="projets_six_sigma.csv",
                    mime="text/csv"
                )
            else:
                st.info("Aucun projet créé pour le moment.")
        
        with tab3:
            if not projects_df.empty:
                project_to_edit = st.selectbox("Sélectionnez un projet à modifier", projects_df['project_name'])
                
                if project_to_edit:
                    project_data = projects_df[projects_df['project_name'] == project_to_edit].iloc[0]
                    
                    with st.form("edit_project_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_name = st.text_input("Nom du Projet", value=project_data['project_name'])
                            new_desc = st.text_area("Description", value=project_data['description'])
                            new_sigma = st.number_input("Objectif Sigma", value=float(project_data['sigma_level']))
                        
                        with col2:
                            new_status = st.selectbox("Statut", 
                                                     ["Planifié", "Actif", "En attente", "Terminé", "Annulé"],
                                                     index=["Planifié", "Actif", "En attente", "Terminé", "Annulé"].index(project_data['status']))
                            new_start = st.date_input("Date de début", 
                                                     value=pd.to_datetime(project_data['start_date']).date() 
                                                     if pd.notnull(project_data['start_date']) else pd.Timestamp.now().date())
                        
                        if st.form_submit_button("💾 Mettre à jour"):
                            # Mettre à jour le projet dans le DataFrame
                            idx = st.session_state.projects_df[
                                st.session_state.projects_df['project_name'] == project_to_edit
                            ].index
                            
                            if len(idx) > 0:
                                st.session_state.projects_df.at[idx[0], 'project_name'] = new_name
                                st.session_state.projects_df.at[idx[0], 'description'] = new_desc
                                st.session_state.projects_df.at[idx[0], 'status'] = new_status
                                st.session_state.projects_df.at[idx[0], 'start_date'] = new_start
                                st.session_state.projects_df.at[idx[0], 'sigma_level'] = new_sigma
                                
                                st.success("Projet mis à jour avec succès!")
                                st.rerun()
    
    elif app_mode == "📊 Analyse Statistique":
        st.markdown('<h1 class="main-header">📊 Analyse Statistique</h1>', unsafe_allow_html=True)
        
        if project_id:
            # Import des données
            uploaded_file = st.file_uploader("📁 Importer des données (CSV ou Excel)", type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.dataframe(data.head(), use_container_width=True)
                
                # Sélection des colonnes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Sélectionner la colonne à analyser", numeric_cols)
                        analysis_data = data[selected_col].dropna().values
                    else:
                        st.error("Aucune colonne numérique trouvée!")
                        analysis_data = np.array([])
                
                with col2:
                    if len(analysis_data) > 0:
                        st.metric("Taille de l'échantillon", len(analysis_data))
                        st.metric("Moyenne", f"{np.mean(analysis_data):.4f}")
                
                with col3:
                    if len(analysis_data) > 0:
                        st.metric("Écart-type", f"{np.std(analysis_data, ddof=1):.4f}")
                        st.metric("Variance", f"{np.var(analysis_data, ddof=1):.4f}")
                
                # Tests statistiques
                if len(analysis_data) > 0:
                    st.markdown('<h3 class="sub-header">Tests d\'Hypothèses</h3>', unsafe_allow_html=True)
                    
                    test_type = st.radio("Type de test", 
                                        ["Test de normalité (Shapiro-Wilk)", "Test t à un échantillon", "Intervalle de confiance"])
                    
                    if test_type == "Test de normalité (Shapiro-Wilk)":
                        stat, p_value = stats.shapiro(analysis_data)
                        st.write(f"**Statistique de test:** {stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")
                        if p_value > 0.05:
                            st.success("Les données suivent une distribution normale (p > 0.05)")
                        else:
                            st.warning("Les données ne suivent pas une distribution normale (p ≤ 0.05)")
                    
                    elif test_type == "Test t à un échantillon":
                        pop_mean = st.number_input("Moyenne populationnelle hypothétique", value=0.0)
                        t_stat, p_value = stats.ttest_1samp(analysis_data, pop_mean)
                        st.write(f"**Statistique t:** {t_stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")
                    
                    elif test_type == "Intervalle de confiance":
                        confidence = st.slider("Niveau de confiance", 0.80, 0.99, 0.95)
                        ci_low, ci_high = stats.t.interval(confidence, len(analysis_data)-1, 
                                                          loc=np.mean(analysis_data), 
                                                          scale=stats.sem(analysis_data))
                        st.write(f"**Intervalle de confiance à {confidence*100}%:** [{ci_low:.4f}, {ci_high:.4f}]")
                    
                    # Histogramme avec courbe normale
                    fig = px.histogram(data, x=selected_col, nbins=30, 
                                     marginal="box", opacity=0.7)
                    
                    # Ajouter courbe normale
                    if len(analysis_data) > 1:
                        x_range = np.linspace(min(analysis_data), max(analysis_data), 100)
                        pdf = stats.norm.pdf(x_range, np.mean(analysis_data), np.std(analysis_data, ddof=1))
                        fig.add_trace(go.Scatter(x=x_range, y=pdf * len(analysis_data) * (max(analysis_data)-min(analysis_data))/30,
                                               mode='lines', name='Distribution normale', line=dict(color='red', width=2)))
                    
                    fig.update_layout(title=f"Distribution de {selected_col}",
                                    xaxis_title=selected_col,
                                    yaxis_title="Fréquence",
                                    height=500)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📈 Cartes de Contrôle":
        st.markdown('<h1 class="main-header">📈 Cartes de Contrôle</h1>', unsafe_allow_html=True)
        
        if project_id:
            # Options pour les données
            data_option = st.radio("Source des données", 
                                  ["Exemple simulé", "Données existantes", "Upload fichier"])
            
            if data_option == "Exemple simulé":
                np.random.seed(42)
                sample_data = np.random.normal(loc=100, scale=10, size=100)
                data = pd.DataFrame({'value': sample_data})
                
            elif data_option == "Données existantes":
                measurements_df = st.session_state.measurements_df
                if not measurements_df.empty:
                    data = measurements_df[['value']]
                else:
                    st.warning("Aucune donnée de mesure pour ce projet")
                    data = pd.DataFrame()
            
            else:  # Upload fichier
                uploaded_file = st.file_uploader("Choisir un fichier", type=['csv', 'xlsx'])
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                else:
                    data = pd.DataFrame()
            
            if not data.empty:
                # Sélection de la colonne numérique
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Colonne à analyser", numeric_cols)
                    analysis_data = data[selected_col].dropna().values
                    
                    # Paramètres de la carte de contrôle
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        subgroup_size = st.slider("Taille du sous-groupe", 2, 10, 5)
                    with col2:
                        lsl = st.number_input("Limite inférieure de spécification", value=float(np.min(analysis_data)*0.9))
                    with col3:
                        usl = st.number_input("Limite supérieure de spécification", value=float(np.max(analysis_data)*1.1))
                    
                    if len(analysis_data) >= subgroup_size * 5:  # Au moins 5 sous-groupes
                        fig = create_control_chart(analysis_data, subgroup_size)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistiques
                        cp, cpk, mean, std = calculate_cp_cpk(analysis_data, lsl, usl)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Cp", f"{cp:.3f}")
                        with col2:
                            st.metric("Cpk", f"{cpk:.3f}")
                        with col3:
                            st.metric("Moyenne", f"{mean:.3f}")
                        with col4:
                            st.metric("Écart-type", f"{std:.3f}")
                        
                        # Interprétation
                        st.markdown('<h3 class="sub-header">Interprétation</h3>', unsafe_allow_html=True)
                        if cp >= 1.33 and cpk >= 1.33:
                            st.success("✅ Processus capable et centré")
                        elif cp >= 1.33 and cpk < 1.33:
                            st.warning("⚠️ Processus capable mais non centré")
                        else:
                            st.error("❌ Processus non capable")
                    else:
                        st.warning(f"Données insuffisantes. Minimum requis: {subgroup_size * 5} points")
    
    elif app_mode == "📉 Diagramme de Pareto":
        st.markdown('<h1 class="main-header">📉 Analyse de Pareto</h1>', unsafe_allow_html=True)
        
        if project_id:
            # Options pour les données
            data_option = st.radio("Source des données sur les défauts",
                                  ["Exemple simulé", "Données existantes", "Saisie manuelle"])
            
            if data_option == "Exemple simulé":
                defect_types = ['Rayures', 'Inclusions', 'Déformations', 'Fissures', 'Porosités']
                counts = np.random.randint(50, 200, size=len(defect_types))
                defect_data = pd.DataFrame({
                    'defect_type': defect_types,
                    'count': counts,
                    'category': ['Surface']*2 + ['Structure']*3
                })
                
            elif data_option == "Données existantes":
                defect_data = st.session_state.defects_df
                if defect_data.empty:
                    st.info("Aucune donnée de défauts pour ce projet")
                    defect_data = pd.DataFrame()
            
            else:  # Saisie manuelle
                num_defects = st.number_input("Nombre de types de défauts", min_value=2, max_value=20, value=5)
                defect_data = []
                
                for i in range(int(num_defects)):
                    with st.expander(f"Défaut {i+1}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            defect_type = st.text_input(f"Type de défaut {i+1}", value=f"Défaut {i+1}")
                        with col2:
                            count = st.number_input(f"Nombre {i+1}", min_value=1, value=100)
                        with col3:
                            category = st.selectbox(f"Catégorie {i+1}", 
                                                  ['Méthode', 'Matériel', 'Main d\'œuvre', 'Environnement', 'Mesure'])
                        defect_data.append({'defect_type': defect_type, 'count': count, 'category': category})
                
                defect_data = pd.DataFrame(defect_data)
            
            if not defect_data.empty:
                # Affichage des données
                st.dataframe(defect_data.sort_values('count', ascending=False), use_container_width=True)
                
                # Diagramme de Pareto
                fig = create_pareto_chart(defect_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Loi de Pareto (80/20)
                total_defects = defect_data['count'].sum()
                sorted_defects = defect_data.sort_values('count', ascending=False)
                sorted_defects['cumulative'] = sorted_defects['count'].cumsum()
                sorted_defects['cumulative_percentage'] = sorted_defects['cumulative'] / total_defects * 100
                
                # Trouver le point 80%
                pareto_point = sorted_defects[sorted_defects['cumulative_percentage'] >= 80].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total des défauts", total_defects)
                with col2:
                    st.metric("Principaux défauts (80%)", 
                             f"{len(sorted_defects[sorted_defects['cumulative_percentage'] <= 80])} types")
                
                st.info(f"**Règle 80/20:** {pareto_point['defect_type']} représente le seuil des 80% des défauts totaux")
    
    elif app_mode == "🐟 Diagramme Ishikawa":
        st.markdown('<h1 class="main-header">🐟 Diagramme de Causes et Effets</h1>', unsafe_allow_html=True)
        
        # Catégories standard 6M
        default_categories = {
            'Méthode': ['Procédures non standardisées', 'Formation insuffisante', 'Mauvaise planification'],
            'Matériel': ['Équipement défectueux', 'Usure normale', 'Mauvaise maintenance'],
            'Main d\'œuvre': ['Manque de compétences', 'Fatigue', 'Erreur humaine'],
            'Environnement': ['Température élevée', 'Humidité', 'Éclairage insuffisant'],
            'Mesure': ['Instrument non étalonné', 'Méthode de mesure incorrecte', 'Erreur de lecture'],
            'Matériaux': ['Qualité variable', 'Fournisseur non fiable', 'Stockage inadéquat']
        }
        
        # Interface de saisie
        st.markdown("### 🎯 Effet/Problème Principal")
        main_effect = st.text_input("Décrivez l'effet ou le problème à analyser", 
                                   value="Taux de défauts élevé")
        
        st.markdown("### 📋 Causes par Catégorie")
        
        causes = {}
        categories = list(default_categories.keys())
        
        for category in categories:
            with st.expander(f"**{category}**"):
                default_causes = default_categories[category]
                causes_list = st.text_area(
                    f"Causes pour {category} (une par ligne)",
                    value="\n".join(default_causes),
                    height=100
                )
                causes[category] = [c.strip() for c in causes_list.split('\n') if c.strip()]
        
        # Bouton de génération
        if st.button("🎨 Générer le Diagramme Ishikawa"):
            if main_effect:
                st.markdown(f"### 📊 Diagramme d'Ishikawa : {main_effect}")
                fig = create_ishikawa_diagram(causes)
                st.plotly_chart(fig, use_container_width=True)
                
                # Téléchargement
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                st.download_button(
                    label="📥 Télécharger le diagramme (PNG)",
                    data=img_bytes,
                    file_name=f"diagramme_ishikawa_{main_effect.replace(' ', '_')}.png",
                    mime="image/png"
                )
            else:
                st.warning("Veuillez saisir un effet principal")
        
        # Section d'export des causes
        st.markdown("---")
        st.markdown("### 💾 Export des Causes")
        
        causes_df = pd.DataFrame([
            {'Catégorie': cat, 'Cause': cause}
            for cat, cause_list in causes.items()
            for cause in cause_list
        ])
        
        if not causes_df.empty:
            st.dataframe(causes_df, use_container_width=True)
            
            csv = causes_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Exporter les causes (CSV)",
                data=csv,
                file_name="causes_ishikawa.csv",
                mime="text/csv"
            )
    
    elif app_mode == "⚙️ Indices de Capacité":
        st.markdown('<h1 class="main-header">⚙️ Analyse de Capacité</h1>', unsafe_allow_html=True)
        
        if project_id:
            # Paramètres
            col1, col2, col3 = st.columns(3)
            with col1:
                lsl = st.number_input("Limite Inférieure de Spécification (LSL)", value=90.0)
            with col2:
                target = st.number_input("Valeur Cible", value=100.0)
            with col3:
                usl = st.number_input("Limite Supérieure de Spécification (USL)", value=110.0)
            
            # Données
            data_option = st.radio("Source des données", ["Exemple", "Mesures existantes", "Upload"])
            
            if data_option == "Exemple":
                np.random.seed(42)
                data = np.random.normal(loc=target, scale=(usl-lsl)/6, size=200)
            elif data_option == "Mesures existantes":
                measurements_df = st.session_state.measurements_df
                data = measurements_df['value'].values if not measurements_df.empty else np.array([])
            else:
                uploaded_file = st.file_uploader("Uploader des données", type=['csv', 'xlsx'])
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_col = st.selectbox("Colonne numérique", numeric_cols)
                        data = df[selected_col].dropna().values
                    else:
                        st.error("Aucune colonne numérique trouvée")
                        data = np.array([])
                else:
                    data = np.array([])
            
            if len(data) > 0:
                # Calculs
                cp, cpk, mean, std = calculate_cp_cpk(data, lsl, usl)
                ppm = stats.norm.cdf(lsl, mean, std) * 1e6 + (1 - stats.norm.cdf(usl, mean, std)) * 1e6
                sigma_level = abs(stats.norm.ppf(ppm/1e6/2))
                
                # Affichage des métriques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cp", f"{cp:.3f}", 
                             delta="✅ Capable" if cp >= 1.33 else "❌ Non capable")
                with col2:
                    st.metric("Cpk", f"{cpk:.3f}", 
                             delta="✅ Centré" if cpk >= 1.33 else "⚠️ Non centré")
                with col3:
                    st.metric("PPM", f"{ppm:.0f}")
                with col4:
                    st.metric("Niveau Sigma", f"{sigma_level:.2f}σ")
                
                # Visualisation
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=("Distribution du Processus", 
                                                "Capabilité du Processus"))
                
                # Histogramme
                fig.add_trace(
                    go.Histogram(x=data, nbinsx=30, name='Distribution',
                                marker_color='lightblue'),
                    row=1, col=1
                )
                
                # Lignes de spécification
                for spec, name, color in [(lsl, 'LSL', 'red'), (target, 'Target', 'green'), 
                                         (usl, 'USL', 'red'), (mean, 'Moyenne', 'blue')]:
                    fig.add_vline(x=spec, line_dash="dash", line_color=color,
                                annotation_text=name, row=1, col=1)
                
                # Courbe normale théorique
                x_range = np.linspace(min(data.min(), lsl*0.9), max(data.max(), usl*1.1), 1000)
                pdf = stats.norm.pdf(x_range, mean, std)
                fig.add_trace(
                    go.Scatter(x=x_range, y=pdf, mode='lines', name='Distribution normale',
                              line=dict(color='darkblue', width=2)),
                    row=1, col=1
                )
                
                # Diagramme de capabilité
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=cpk,
                        title={'text': "Cpk"},
                        domain={'row': 0, 'column': 0},
                        gauge={
                            'axis': {'range': [0, max(2, cp*1.5)]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 1], 'color': "red"},
                                {'range': [1, 1.33], 'color': "yellow"},
                                {'range': [1.33, 2], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 1.33
                            }
                        }
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📤 Import/Export":
        st.markdown('<h1 class="main-header">📤 Import/Export de Données</h1>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])
        
        with tab1:
            st.markdown("### Import de Données")
            
            import_type = st.selectbox("Type de données à importer",
                                      ["Projets", "Mesures", "Défauts"])
            
            uploaded_file = st.file_uploader(f"Choisir un fichier CSV pour {import_type}",
                                            type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    st.success(f"Fichier chargé avec succès! ({len(df)} lignes)")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Aperçu des colonnes
                    st.markdown("**Structure du fichier:**")
                    col_info = pd.DataFrame({
                        'Colonne': df.columns,
                        'Type': df.dtypes.values,
                        'Valeurs non-nulles': df.notnull().sum().values,
                        'Exemple': df.iloc[0].values if len(df) > 0 else ['N/A']*len(df.columns)
                    })
                    st.table(col_info)
                    
                    if st.button(f"Importer les {import_type}"):
                        if import_type == "Projets":
                            st.session_state.projects_df = df
                        elif import_type == "Mesures":
                            st.session_state.measurements_df = df
                        elif import_type == "Défauts":
                            st.session_state.defects_df = df
                        
                        st.success(f"{len(df)} {import_type} importés avec succès!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Erreur lors du chargement: {str(e)}")
        
        with tab2:
            st.markdown("### Export de Données")
            
            export_type = st.selectbox("Type de données à exporter",
                                      ["Tous les projets", "Toutes les mesures", 
                                       "Tous les défauts", "Rapport complet"])
            
            if export_type == "Tous les projets":
                data = st.session_state.projects_df
            elif export_type == "Toutes les mesures":
                data = st.session_state.measurements_df
            elif export_type == "Tous les défauts":
                data = st.session_state.defects_df
            elif export_type == "Rapport complet":
                # Créer un rapport consolidé
                report_data = {
                    'Projets': st.session_state.projects_df,
                    'Mesures': st.session_state.measurements_df,
                    'Défauts': st.session_state.defects_df
                }
            
            if 'data' in locals() and not data.empty:
                st.dataframe(data, use_container_width=True)
                
                # Options d'export
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"{export_type.replace(' ', '_').lower()}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        data.to_excel(writer, index=False, sheet_name='Data')
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=excel_data,
                        file_name=f"{export_type.replace(' ', '_').lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            elif export_type == "Rapport complet":
                # Créer un fichier Excel avec plusieurs onglets
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    for sheet_name, df_data in report_data.items():
                        if not df_data.empty:
                            df_data.to_excel(writer, index=False, sheet_name=sheet_name[:31])
                
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📥 Télécharger Rapport Complet (Excel)",
                    data=excel_data,
                    file_name="rapport_complet_six_sigma.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
