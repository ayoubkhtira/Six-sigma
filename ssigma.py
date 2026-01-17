import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import reflex as rx

# --- CONFIGURATION ET STYLE ---
st.set_page_config(
    page_title="Six Sigma Pro Suite",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stDataEditor { border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    .status-box { 
        padding: 20px; 
        border-radius: 15px; 
        margin-bottom: 25px; 
        border-left: 6px solid #1E3A8A; 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    .tool-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 15px 0;
        transition: transform 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .phase-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALISATION DES DONNÉES ---
if 'df_amdec' not in st.session_state:
    st.session_state.df_amdec = pd.DataFrame([
        {"Processus": "Soudure", "Mode de défaillance": "Fissure", "G": 9, "O": 3, "D": 2},
        {"Processus": "Peinture", "Mode de défaillance": "Rayure", "G": 4, "O": 6, "D": 4},
        {"Processus": "Assemblage", "Mode de défaillance": "Vis manquante", "G": 5, "O": 4, "D": 7}
    ])

if 'df_gage' not in st.session_state:
    np.random.seed(42)
    data = []
    for op in ['Opérateur A', 'Opérateur B', 'Opérateur C']:
        for p in [f'Pièce {i}' for i in range(1, 11)]:
            for rep in range(3):
                data.append({
                    "Opérateur": op, 
                    "Pièce": p, 
                    "Réplique": rep+1,
                    "Mesure": round(50 + np.random.normal(0, 0.8), 3)
                })
    st.session_state.df_gage = pd.DataFrame(data)

if 'df_ctq' not in st.session_state:
    st.session_state.df_ctq = pd.DataFrame({
        'CTQ': ['Durée de traitement', 'Qualité produit', 'Coût unitaire', 'Satisfaction client'],
        'Mesure': ['Minutes', 'Score/100', '€', '%'],
        'Cible': [30, 95, 25, 98],
        'Spéc Inf': [20, 90, 20, 95],
        'Spéc Sup': [40, 100, 30, 100]
    })

if 'df_capa' not in st.session_state:
    np.random.seed(123)
    st.session_state.df_capa = pd.DataFrame({
        'Mesure': np.random.normal(50, 2, 100),
        'Lot': np.random.choice(['Lot A', 'Lot B', 'Lot C'], 100),
        'Date': pd.date_range('2024-01-01', periods=100, freq='D')
    })

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Six_Sigma_logo.svg/1200px-Six_Sigma_logo.svg.png", 
             width=120)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Six Sigma Pro Suite")
    st.markdown("---")
    
    st.markdown("#### 🎯 Sélectionnez une phase DMAIC")
    phase = st.radio(
        "",
        ["D - Définir", "M - Mesurer", "A - Analyser", "I - Innover", "C - Contrôler"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    with st.expander("📈 Exporter les données"):
        st.download_button(
            "📥 Données AMDEC (CSV)", 
            st.session_state.df_amdec.to_csv(index=False), 
            "amdec_data.csv", 
            "text/csv"
        )
        st.download_button(
            "📥 Données Gage R&R (CSV)", 
            st.session_state.df_gage.to_csv(index=False), 
            "gage_rr_data.csv", 
            "text/csv"
        )
    
    st.markdown("---")
    st.markdown("#### 📊 Statistiques du projet")
    st.metric("Risques identifiés", len(st.session_state.df_amdec))
    st.metric("IPR moyen", 
              round(st.session_state.df_amdec['G'].mean() * 
                    st.session_state.df_amdec['O'].mean() * 
                    st.session_state.df_amdec['D'].mean(), 1))
    st.metric("Mesures analysées", len(st.session_state.df_gage))

# --- FONCTIONS D'OUTILS PAR ÉTAPE ---

def phase_definir():
    st.markdown("<div class='phase-header'><h1>🎯 D - DÉFINIR</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Voice of Customer (VOC)")
        st.markdown("Capturez les besoins clients et les exigences critiques")
        
        voc_input = st.text_area("Entrez les commentaires clients (un par ligne):", 
                                "Délai de livraison trop long\nQualité inconstante\nSupport client lent")
        
        if st.button("Analyser le VOC", key="voc_analyze"):
            comments = [c.strip() for c in voc_input.split('\n') if c.strip()]
            df_voc = pd.DataFrame({'Commentaire': comments})
            df_voc['Fréquence'] = np.random.randint(1, 10, len(comments))
            
            fig_voc = px.bar(df_voc, x='Commentaire', y='Fréquence',
                           color='Fréquence', title="Analyse VOC - Fréquence des Commentaires")
            st.plotly_chart(fig_voc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 CTQ Tree")
        st.markdown("Définissez les Critical-to-Quality characteristics")
        
        edited_ctq = st.data_editor(
            st.session_state.df_ctq,
            num_rows="dynamic",
            column_config={
                'CTQ': st.column_config.TextColumn("Caractéristique"),
                'Cible': st.column_config.NumberColumn("Valeur Cible", min_value=0),
                'Spéc Inf': st.column_config.NumberColumn("Spécification Inférieure"),
                'Spéc Sup': st.column_config.NumberColumn("Spécification Supérieure")
            },
            use_container_width=True,
            key="ctq_editor"
        )
        st.session_state.df_ctq = edited_ctq
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Project Charter")
        st.markdown("Définissez les objectifs et périmètre du projet")
        
        with st.form("project_charter"):
            project_name = st.text_input("Nom du projet", "Amélioration Processus Production")
            problem_statement = st.text_area("Énoncé du problème", 
                                           "Le taux de défauts actuel est de 5%, causant des retards de livraison.")
            goal = st.text_input("Objectif SMART", "Réduire le taux de défauts à 1% dans 6 mois")
            scope = st.text_area("Périmètre", "Processus de production ligne A, équipe de 10 personnes")
            budget = st.number_input("Budget (k€)", min_value=0, value=50)
            
            if st.form_submit_button("Générer la charte"):
                st.success("✅ Charte de projet générée avec succès!")
                st.json({
                    "Nom du projet": project_name,
                    "Énoncé du problème": problem_statement,
                    "Objectif": goal,
                    "Périmètre": scope,
                    "Budget": f"{budget} k€"
                })
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Diagramme SIPOC")
        st.markdown("Fournisseurs → Inputs → Processus → Outputs → Clients")
        
        sipoc_data = {
            'Étape': ['Fournisseurs', 'Inputs', 'Processus', 'Outputs', 'Clients'],
            'Élément': [
                'Fournisseur A, Fournisseur B',
                'Matières premières, Données',
                'Fabrication, Contrôle qualité',
                'Produits finis, Rapports',
                'Client X, Client Y'
            ]
        }
        df_sipoc = pd.DataFrame(sipoc_data)
        st.table(df_sipoc.set_index('Étape'))
        
        if st.button("Visualiser SIPOC", key="sipoc_viz"):
            fig_sipoc = px.bar(df_sipoc, x='Étape', y=[1]*5, 
                             color='Étape', title="Diagramme SIPOC")
            st.plotly_chart(fig_sipoc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def phase_mesurer():
    st.markdown("<div class='phase-header'><h1>📏 M - MESURER</h1></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Gage R&R", "📈 Cartes de Contrôle", "🎯 Analyse de Capabilité"])
    
    with tab1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Gage R&R Étendu")
        st.markdown("Analyse de la variabilité du système de mesure")
        
        edited_gage = st.data_editor(
            st.session_state.df_gage,
            num_rows="dynamic",
            column_config={
                "Mesure": st.column_config.NumberColumn(
                    "Valeur", 
                    min_value=0.0,
                    format="%.3f"
                )
            },
            use_container_width=True,
            key="gage_editor_extended"
        )
        st.session_state.df_gage = edited_gage
        
        if not edited_gage.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(edited_gage, x="Opérateur", y="Mesure", 
                               color="Opérateur", points="all",
                               title="Variabilité par Opérateur")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_scatter = px.scatter(edited_gage, x="Pièce", y="Mesure", 
                                       color="Opérateur", symbol="Opérateur",
                                       title="Mesures par Pièce et Opérateur")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Calculs Gage R&R simplifiés
            st.markdown("#### 📊 Résultats Gage R&R")
            total_var = edited_gage['Mesure'].var()
            op_var = edited_gage.groupby('Opérateur')['Mesure'].var().mean()
            gage_rr = (op_var / total_var) * 100
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("% Gage R&R", f"{gage_rr:.1f}%", 
                         delta="Acceptable" if gage_rr < 30 else "À améliorer")
            with metrics_col2:
                st.metric("Variabilité Opérateur", f"{op_var:.3f}")
            with metrics_col3:
                st.metric("Variabilité Totale", f"{total_var:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Cartes de Contrôle Xbar-R")
        
        # Générer des données de contrôle
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = []
        for date in dates:
            subgroup = np.random.normal(50, 2, 5)  # 5 mesures par sous-groupe
            data.extend([{'Date': date, 'Mesure': m, 'Sous-groupe': f'SG{date.day}'} for m in subgroup])
        
        df_control = pd.DataFrame(data)
        
        # Calcul des limites de contrôle
        df_control['Xbar'] = df_control.groupby('Date')['Mesure'].transform('mean')
        df_control['R'] = df_control.groupby('Date')['Mesure'].transform(lambda x: x.max() - x.min())
        
        xbar_mean = df_control['Mesure'].mean()
        r_mean = df_control['R'].mean()
        
        # Limites pour carte Xbar
        a2 = 0.577  # Pour n=5
        ucl_xbar = xbar_mean + a2 * r_mean
        lcl_xbar = xbar_mean - a2 * r_mean
        
        # Limites pour carte R
        d3 = 0
        d4 = 2.114  # Pour n=5
        ucl_r = d4 * r_mean
        lcl_r = d3 * r_mean
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Carte Xbar
            fig_xbar = go.Figure()
            fig_xbar.add_trace(go.Scatter(x=dates, y=df_control.groupby('Date')['Mesure'].mean(),
                                        mode='lines+markers', name='Moyenne'))
            fig_xbar.add_hline(y=xbar_mean, line_dash="solid", line_color="blue", name='CL')
            fig_xbar.add_hline(y=ucl_xbar, line_dash="dash", line_color="red", name='UCL')
            fig_xbar.add_hline(y=lcl_xbar, line_dash="dash", line_color="red", name='LCL')
            fig_xbar.update_layout(title='Carte de Contrôle Xbar', height=400)
            st.plotly_chart(fig_xbar, use_container_width=True)
        
        with col2:
            # Carte R
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=dates, y=df_control.groupby('Date')['R'].mean(),
                                      mode='lines+markers', name='Étendue'))
            fig_r.add_hline(y=r_mean, line_dash="solid", line_color="blue", name='CL')
            fig_r.add_hline(y=ucl_r, line_dash="dash", line_color="red", name='UCL')
            fig_r.add_hline(y=lcl_r, line_dash="dash", line_color="red", name='LCL')
            fig_r.update_layout(title='Carte de Contrôle R', height=400)
            st.plotly_chart(fig_r, use_container_width=True)
        
        st.markdown("#### 📊 Indicateurs de Contrôle")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processus Stable", "OUI" if all(lcl_xbar <= m <= ucl_xbar for m in df_control.groupby('Date')['Mesure'].mean()) else "NON")
        with col2:
            st.metric("Cp Potentiel", f"{((ucl_xbar - lcl_xbar) / (6 * df_control['Mesure'].std())):.2f}")
        with col3:
            st.metric("Points Hors Contrôle", "0")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Analyse de Capabilité Processus")
        
        data_input = st.text_area("Entrez les données (séparées par des virgules ou retours à la ligne):",
                                "48.2, 49.8, 50.1, 51.3, 49.5, 50.2, 49.9, 50.5, 51.0, 50.3")
        
        lsl = st.number_input("Limite Spécification Inférieure (LSL)", value=48.0)
        usl = st.number_input("Limite Spécification Supérieure (USL)", value=52.0)
        target = st.number_input("Cible", value=50.0)
        
        if st.button("Calculer la Capabilité", key="capability_calc"):
            # Conversion des données
            data = [float(x.strip()) for x in data_input.replace(',', '\n').replace('\n', ',').split(',') if x.strip()]
            
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)
                
                # Calcul des indices de capabilité
                cp = (usl - lsl) / (6 * std_val)
                cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val))
                pp = (usl - lsl) / (6 * np.std(data))
                ppk = min((usl - mean_val) / (3 * np.std(data)), (mean_val - lsl) / (3 * np.std(data)))
                
                # Visualisation
                fig = go.Figure()
                
                # Histogramme
                fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=20,
                                         histnorm='probability density',
                                         marker_color='rgba(59, 130, 246, 0.7)'))
                
                # Courbe normale
                x = np.linspace(min(data) - 3*std_val, max(data) + 3*std_val, 100)
                y = (1/(std_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_val)/std_val)**2)
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Courbe Normale',
                                       line=dict(color='red', width=2)))
                
                # Lignes verticales pour spécifications
                fig.add_vline(x=lsl, line_dash="dash", line_color="orange", annotation_text="LSL")
                fig.add_vline(x=usl, line_dash="dash", line_color="orange", annotation_text="USL")
                fig.add_vline(x=target, line_dash="solid", line_color="green", annotation_text="Cible")
                fig.add_vline(x=mean_val, line_dash="dot", line_color="blue", annotation_text=f"Moyenne: {mean_val:.2f}")
                
                fig.update_layout(title=f"Analyse de Capabilité - Cp: {cp:.2f}, Cpk: {cpk:.2f}",
                                height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des résultats
                results = pd.DataFrame({
                    'Indice': ['Cp', 'Cpk', 'Pp', 'Ppk', 'DPMO'],
                    'Valeur': [f"{cp:.2f}", f"{cpk:.2f}", f"{pp:.2f}", f"{ppk:.2f}", 
                             f"{((sum(1 for d in data if d < lsl or d > usl) / len(data)) * 1_000_000):.0f}"],
                    'Interprétation': [
                        'Capabilité potentielle' + ('✓' if cp >= 1.33 else '⚠️'),
                        'Capabilité réelle' + ('✓' if cpk >= 1.33 else '⚠️'),
                        'Performance potentielle',
                        'Performance réelle',
                        'Défauts par million'
                    ]
                })
                st.table(results)
        st.markdown("</div>", unsafe_allow_html=True)

def phase_analyser():
    st.markdown("<div class='phase-header'><h1>🔍 A - ANALYSER</h1></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ AMDEC", "🐟 Diagramme Ishikawa", "📊 Pareto", "📈 ANOVA"])
    
    with tab1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🛡️ AMDEC (Analyse des Modes de Défaillance)")
        st.markdown("Évaluez et priorisez les risques potentiels")
        
        edited_df = st.data_editor(
            st.session_state.df_amdec,
            num_rows="dynamic",
            column_config={
                "G": st.column_config.NumberColumn("Gravité (1-10)", min_value=1, max_value=10, default=5),
                "O": st.column_config.NumberColumn("Occurrence (1-10)", min_value=1, max_value=10, default=5),
                "D": st.column_config.NumberColumn("Détection (1-10)", min_value=1, max_value=10, default=5),
            },
            use_container_width=True,
            key="amdec_editor_pro"
        )
        st.session_state.df_amdec = edited_df
        
        if not edited_df.empty:
            df_viz = edited_df.copy()
            df_viz['IPR'] = df_viz['G'] * df_viz['O'] * df_viz['D']
            df_viz['Priorité'] = pd.qcut(df_viz['IPR'], q=3, labels=['Basse', 'Moyenne', 'Haute'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Graphique radar pour évaluation AMDEC
                fig_radar = go.Figure()
                
                for idx, row in df_viz.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['G'], row['O'], row['D'], row['IPR']/20],
                        theta=['Gravité', 'Occurrence', 'Détection', 'IPR'],
                        name=row['Mode de défaillance'],
                        fill='toself'
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    title="Analyse Radar des Risques",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Métriques
                avg_ipr = df_viz['IPR'].mean()
                high_risk = len(df_viz[df_viz['IPR'] > 100])
                
                st.metric("IPR Moyen", f"{avg_ipr:.1f}", 
                         delta="Critique" if avg_ipr > 150 else "Élevé" if avg_ipr > 100 else "Acceptable")
                st.metric("Risques Élevés", high_risk)
                st.metric("Actions Requises", f"{high_risk} sur {len(df_viz)}")
                
                # Recommandations
                st.markdown("#### 📋 Recommandations")
                if high_risk > 0:
                    st.warning(f"🔴 {high_risk} risque(s) nécessite(nt) une action immédiate")
                st.info("Objectif: IPR < 100 pour tous les risques")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🐟 Diagramme d'Ishikawa (5M)")
        st.markdown("Analyse des causes racines par catégorie")
        
        categories = ['Méthodes', 'Main d\'œuvre', 'Matériels', 'Milieu', 'Matériaux']
        
        col1, col2 = st.columns(2)
        
        with col1:
            causes = {}
            for category in categories:
                causes[category] = st.text_area(
                    f"Causes - {category} (une par ligne)",
                    "Formation insuffisante\nProcédure obsolète" if category == 'Méthodes' else ""
                ).split('\n')
            
            if st.button("Générer le Diagramme", key="ishikawa_gen"):
                # Création du diagramme Ishikawa simplifié
                fig = go.Figure()
                
                # Arête principale
                fig.add_trace(go.Scatter(
                    x=[0, 5], y=[0, 0],
                    mode='lines+markers',
                    line=dict(color='black', width=3),
                    marker=dict(size=10),
                    name='Problème'
                ))
                
                # Branches
                angles = np.linspace(-60, 60, len(categories))
                for i, (category, angle) in enumerate(zip(categories, angles)):
                    x_end = 2 * np.cos(np.radians(angle))
                    y_end = 2 * np.sin(np.radians(angle))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, x_end], y=[0, y_end],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name=category
                    ))
                    
                    # Ajouter le texte de la catégorie
                    fig.add_annotation(
                        x=x_end*1.1,
                        y=y_end*1.1,
                        text=category,
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                fig.update_layout(
                    title="Diagramme d'Ishikawa - Causes Racines",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    showlegend=False,
                    height=500
                )
                
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Liste des causes
                    st.markdown("#### 📋 Liste des Causes Identifiées")
                    for category, cause_list in causes.items():
                        if any(cause.strip() for cause in cause_list):
                            with st.expander(f"{category} ({len([c for c in cause_list if c.strip()])} causes)"):
                                for cause in cause_list:
                                    if cause.strip():
                                        st.markdown(f"- {cause.strip()}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Diagramme de Pareto")
        st.markdown("Loi des 20/80 - Identifier les problèmes les plus fréquents")
        
        # Entrée des données
        default_data = """Défaut A,45
Défaut B,32
Défaut C,18
Défaut D,12
Défaut E,8
Défaut F,5"""
        
        pareto_data = st.text_area("Entrez les données (format: Défaut, Fréquence):", 
                                 default_data, height=150)
        
        if st.button("Analyser Pareto", key="pareto_analyze"):
            lines = [line.strip() for line in pareto_data.split('\n') if line.strip()]
            defects = []
            frequencies = []
            
            for line in lines:
                if ',' in line:
                    defect, freq = line.split(',')
                    defects.append(defect.strip())
                    frequencies.append(int(freq.strip()))
            
            if defects and frequencies:
                df_pareto = pd.DataFrame({'Défaut': defects, 'Fréquence': frequencies})
                df_pareto = df_pareto.sort_values('Fréquence', ascending=False)
                df_pareto['% Cumulé'] = (df_pareto['Fréquence'].cumsum() / df_pareto['Fréquence'].sum() * 100).round(1)
                
                # Création du graphique Pareto
                fig = go.Figure()
                
                # Barres pour les fréquences
                fig.add_trace(go.Bar(
                    x=df_pareto['Défaut'],
                    y=df_pareto['Fréquence'],
                    name='Fréquence',
                    marker_color='indianred',
                    yaxis='y1'
                ))
                
                # Ligne pour le pourcentage cumulé
                fig.add_trace(go.Scatter(
                    x=df_pareto['Défaut'],
                    y=df_pareto['% Cumulé'],
                    name='% Cumulé',
                    line=dict(color='blue', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Diagramme de Pareto',
                    xaxis_title='Défauts',
                    yaxis=dict(
                        title='Fréquence',
                        titlefont=dict(color='indianred'),
                        tickfont=dict(color='indianred')
                    ),
                    yaxis2=dict(
                        title='% Cumulé',
                        titlefont=dict(color='blue'),
                        tickfont=dict(color='blue'),
                        overlaying='y',
                        side='right',
                        range=[0, 110]
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse des 20%
                st.markdown("#### 🎯 Analyse des 20% Critiques")
                total_freq = df_pareto['Fréquence'].sum()
                cumulative = 0
                critical_defects = []
                
                for idx, row in df_pareto.iterrows():
                    cumulative += row['Fréquence']
                    percentage = (cumulative / total_freq) * 100
                    critical_defects.append(row['Défaut'])
                    if percentage >= 80:
                        break
                
                st.success(f"**{len(critical_defects)} défauts représentent 80% des problèmes:**")
                for defect in critical_defects:
                    st.markdown(f"- {defect}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def phase_innover():
    st.markdown("<div class='phase-header'><h1>💡 I - INNOVER</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Brainstorming & Idéation")
        st.markdown("Générez et priorisez des solutions")
        
        problem = st.text_input("Problème à résoudre:", "Taux de défauts élevé sur la ligne de production")
        
        st.markdown("#### Idées de solutions:")
        solutions = st.text_area("Listez les idées (une par ligne):", 
                                "Automatiser le contrôle qualité\nFormer les opérateurs\nRéviser les procédures\nAméliorer l'ergonomie des postes")
        
        if st.button("Prioriser les solutions", key="prioritize_solutions"):
            ideas = [s.strip() for s in solutions.split('\n') if s.strip()]
            
            if ideas:
                # Matrice de décision simple
                criteria = ['Efficacité', 'Coût', 'Facilité', 'Impact']
                df_matrix = pd.DataFrame({
                    'Solution': ideas,
                    'Efficacité': np.random.randint(1, 10, len(ideas)),
                    'Coût': np.random.randint(1, 10, len(ideas)),
                    'Facilité': np.random.randint(1, 10, len(ideas)),
                    'Impact': np.random.randint(1, 10, len(ideas))
                })
                
                df_matrix['Score Total'] = df_matrix[criteria].sum(axis=1)
                df_matrix = df_matrix.sort_values('Score Total', ascending=False)
                
                st.markdown("#### 📊 Matrice de Décision")
                st.dataframe(df_matrix.set_index('Solution'), use_container_width=True)
                
                # Graphique radar pour la meilleure solution
                best_solution = df_matrix.iloc[0]['Solution']
                best_scores = df_matrix.iloc[0][criteria].tolist()
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=best_scores + [best_scores[0]],  # Fermer le polygone
                    theta=criteria + [criteria[0]],
                    fill='toself',
                    name=best_solution,
                    line_color='green'
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    title=f"Meilleure Solution: {best_solution}",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Plan d'Action 5W2H")
        st.markdown("Définissez le plan d'implémentation")
        
        with st.form("action_plan"):
            what = st.text_input("What (Quoi faire)?", "Implémenter un système de contrôle automatique")
            why = st.text_input("Why (Pourquoi)?", "Réduire les défauts de 5% à 1%")
            who = st.text_input("Who (Qui)?", "Équipe qualité + Fournisseur")
            where = st.text_input("Where (Où)?", "Ligne de production A")
            when = st.date_input("When (Quand)?", datetime.now() + timedelta(days=30))
            how = st.text_input("How (Comment)?", "Achat système + Installation + Formation")
            how_much = st.number_input("How much (Budget)?", min_value=0, value=25000)
            
            if st.form_submit_button("Générer le Plan d'Action"):
                st.success("✅ Plan d'action créé!")
                
                plan_df = pd.DataFrame({
                    'Élément': ['Quoi', 'Pourquoi', 'Qui', 'Où', 'Quand', 'Comment', 'Budget'],
                    'Détail': [what, why, who, where, when.strftime('%Y-%m-%d'), how, f"{how_much}€"]
                })
                
                st.table(plan_df.set_index('Élément'))
                
                # Gantt simplifié
                gantt_data = pd.DataFrame({
                    'Tâche': ['Préparation', 'Achat', 'Installation', 'Formation', 'Test'],
                    'Début': pd.date_range(start=datetime.now(), periods=5, freq='W'),
                    'Fin': pd.date_range(start=datetime.now() + timedelta(days=7), periods=5, freq='W'),
                    'Responsable': ['Manager', 'Achats', 'Technique', 'RH', 'Qualité']
                })
                
                fig_gantt = px.timeline(gantt_data, x_start="Début", x_end="Fin", y="Tâche", 
                                      color="Responsable", title="Diagramme de Gantt")
                st.plotly_chart(fig_gantt, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Test de Solutions (PDCA)")
        st.markdown("Plan-Do-Check-Act Cycle")
        
        pdca_step = st.selectbox("Étape PDCA:", 
                                ["Plan (Planifier)", "Do (Mettre en œuvre)", 
                                 "Check (Vérifier)", "Act (Agir)"])
        
        if pdca_step == "Plan (Planifier)":
            st.markdown("**Objectif:** Définir la solution et le plan de test")
            st.text_input("Solution à tester:", "Nouvelle procédure de contrôle")
            st.date_input("Date de début:", datetime.now())
            st.number_input("Durée du test (jours):", min_value=1, value=14)
            
        elif pdca_step == "Do (Mettre en œuvre)":
            st.markdown("**Objectif:** Implémenter la solution")
            st.text_area("Actions réalisées:", "Formation des opérateurs\nMise en place du nouveau processus")
            st.file_uploader("Joindre des documents:", type=['pdf', 'jpg', 'png'])
            
        elif pdca_step == "Check (Vérifier)":
            st.markdown("**Objectif:** Analyser les résultats")
            before = st.number_input("Métrique avant:", value=5.0)
            after = st.number_input("Métrique après:", value=2.5)
            st.metric("Amélioration", f"{((before - after)/before*100):.1f}%", 
                     delta=f"{after - before:.1f}")
            
        elif pdca_step == "Act (Agir)":
            st.markdown("**Objectif:** Standardiser ou ajuster")
            decision = st.radio("Décision:", ["Standardiser la solution", "Ajuster et retester", "Abandonner"])
            if decision == "Standardiser la solution":
                st.success("✅ Solution validée - Procédure à standardiser")
            elif decision == "Ajuster et retester":
                st.warning("⚠️ Ajustements nécessaires - Nouveau cycle PDCA")
            else:
                st.error("❌ Solution abandonnée")
        
        if st.button("Enregistrer l'étape PDCA", key="save_pdca"):
            st.success(f"Étape '{pdca_step}' enregistrée!")
        st.markdown("</div>", unsafe_allow_html=True)

def phase_controler():
    st.markdown("<div class='phase-header'><h1>🛡️ C - CONTRÔLER</h1></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Tableau de Bord", "📋 Standardisation", "🔄 Audit & Revue"])
    
    with tab1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Tableau de Bord de Performance")
        st.markdown("Surveillance continue des indicateurs clés")
        
        # Données de performance simulées
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        performance_data = {
            'Date': dates,
            'Défauts (%)': [5.2, 4.8, 4.5, 3.9, 3.2, 2.8, 2.5, 2.3, 2.1, 1.9, 1.8, 1.7],
            'Productivité': [85, 86, 87, 88, 89, 90, 91, 92, 92, 93, 93, 94],
            'Coût unitaire (€)': [28.5, 28.2, 27.9, 27.5, 27.0, 26.8, 26.5, 26.3, 26.1, 25.9, 25.8, 25.7],
            'Satisfaction client': [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        # Sélecteur d'indicateur
        indicator = st.selectbox("Sélectionnez l'indicateur:", 
                                ['Défauts (%)', 'Productivité', 'Coût unitaire (€)', 'Satisfaction client'])
        
        # Graphique de tendance
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_perf['Date'],
            y=df_perf[indicator],
            mode='lines+markers',
            name='Valeur réelle',
            line=dict(color='blue', width=3)
        ))
        
        # Ajouter la ligne cible
        targets = {
            'Défauts (%)': 2.0,
            'Productivité': 95,
            'Coût unitaire (€)': 26.0,
            'Satisfaction client': 95
        }
        
        fig_trend.add_hline(y=targets[indicator], line_dash="dash", 
                           line_color="green", annotation_text="Cible")
        
        fig_trend.update_layout(
            title=f"Tendance {indicator}",
            xaxis_title="Date",
            yaxis_title=indicator,
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Indicateurs sous forme de métriques
        st.markdown("#### 📈 Indicateurs Clés Actuels")
        cols = st.columns(4)
        
        with cols[0]:
            current_val = df_perf.iloc[-1]['Défauts (%)']
            target_val = 2.0
            delta = current_val - target_val
            st.metric("Défauts (%)", f"{current_val:.1f}%", 
                     f"{delta:+.1f}%", delta_color="inverse" if delta > 0 else "normal")
        
        with cols[1]:
            current_val = df_perf.iloc[-1]['Productivité']
            target_val = 95
            delta = current_val - target_val
            st.metric("Productivité", f"{current_val}%", f"{delta:+.0f}%")
        
        with cols[2]:
            current_val = df_perf.iloc[-1]['Coût unitaire (€)']
            target_val = 26.0
            delta = current_val - target_val
            st.metric("Coût (€)", f"{current_val:.1f}€", 
                     f"{delta:+.1f}€", delta_color="inverse" if delta > 0 else "normal")
        
        with cols[3]:
            current_val = df_perf.iloc[-1]['Satisfaction client']
            target_val = 95
            delta = current_val - target_val
            st.metric("Satisfaction", f"{current_val}%", f"{delta:+.0f}%")
        
        # Carte de contrôle des indicateurs
        st.markdown("#### 🎯 Cartes de Contrôle des Indicateurs")
        
        selected_kpi = st.selectbox("KPI pour carte de contrôle:", 
                                   ['Défauts (%)', 'Productivité'])
        
        if selected_kpi:
            data = df_perf[selected_kpi].values
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val
            
            fig_control = go.Figure()
            fig_control.add_trace(go.Scatter(
                x=df_perf['Date'],
                y=data,
                mode='lines+markers',
                name='Valeur'
            ))
            fig_control.add_hline(y=mean_val, line_dash="solid", 
                                line_color="blue", name='Moyenne')
            fig_control.add_hline(y=ucl, line_dash="dash", 
                                line_color="red", name='UCL')
            fig_control.add_hline(y=lcl, line_dash="dash", 
                                line_color="red", name='LCL')
            
            fig_control.update_layout(
                title=f"Carte de Contrôle - {selected_kpi}",
                height=300
            )
            st.plotly_chart(fig_control, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Documentation & Standardisation")
        st.markdown("Créer et maintenir les standards")
        
        # Éditeur de procédures
        st.markdown("#### 📝 Procédure Opératoire Standard")
        
        procedure_title = st.text_input("Titre de la procédure:", 
                                       "Contrôle qualité - Ligne de production A")
        
        procedure_steps = st.text_area("Étapes de la procédure (une par ligne):",
                                      "1. Vérifier l'étalonnage des instruments\n2. Prélever 5 échantillons par lot\n3. Mesurer les dimensions critiques\n4. Enregistrer les résultats\n5. Signaler tout écart")
        
        # Documents standards
        st.markdown("#### 📄 Documents Requis")
        
        documents = [
            "Fiche de contrôle qualité",
            "Procédure d'étalonnage",
            "Plan d'audit interne",
            "Registre des non-conformités"
        ]
        
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{doc}**")
            with col2:
                status = st.selectbox(f"Statut {doc}", 
                                     ["À créer", "En révision", "Approuvé", "En vigueur"],
                                     key=f"doc_{doc}")
        
        if st.button("Générer le pack documentaire", key="generate_docs"):
            st.success("Pack documentaire généré!")
            
            # Création d'un DataFrame récapitulatif
            doc_status = pd.DataFrame({
                'Document': documents,
                'Statut': ["À créer", "En révision", "Approuvé", "En vigueur"],
                'Responsable': ["Qualité", "Production", "Qualité", "Tous"],
                'Échéance': ["2024-03-01", "2024-03-15", "2024-03-20", "2024-04-01"]
            })
            
            st.table(doc_status)
        
        # Checklist de standardisation
        st.markdown("#### ✅ Checklist de Standardisation")
        
        checklist_items = [
            ("Procédure documentée", False),
            ("Formation réalisée", False),
            ("Indicateurs définis", True),
            ("Audit planifié", False),
            ("Retour d'expérience", True)
        ]
        
        for item, default in checklist_items:
            st.checkbox(item, value=default, key=f"check_{item}")
        
        completion = sum([st.session_state[f"check_{item[0]}"] for item in checklist_items])
        total = len(checklist_items)
        progress = (completion / total) * 100
        
        st.progress(progress / 100, text=f"Standardisation: {progress:.0f}% complète")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.markdown("### 🔄 Audit & Revue de Processus")
        st.markdown("Surveillance continue et amélioration")
        
        # Plan d'audit
        st.markdown("#### 📅 Plan d'Audit")
        
        audit_data = pd.DataFrame({
            'Type d\'audit': ['Interne - Qualité', 'Interne - Processus', 'Fournisseur', 'Client'],
            'Date prévue': ['2024-03-15', '2024-04-10', '2024-05-05', '2024-06-20'],
            'Responsable': ['Auditeur A', 'Auditeur B', 'Auditeur A', 'Manager'],
            'Statut': ['Planifié', 'Planifié', 'À planifier', 'Confirmé']
        })
        
        st.dataframe(audit_data, use_container_width=True)
        
        # Formulaire d'audit
        st.markdown("#### 📋 Rapport d'Audit")
        
        with st.form("audit_report"):
            audit_type = st.selectbox("Type d'audit:", 
                                     ['Interne - Qualité', 'Interne - Processus', 
                                      'Fournisseur', 'Client', 'Système'])
            audit_date = st.date_input("Date de l'audit:", datetime.now())
            auditor = st.text_input("Auditeur:", "John Doe")
            scope = st.text_area("Périmètre audité:", "Processus de production - Ligne A")
            
            # Constatations
            st.markdown("##### Constatations")
            nc_count = st.number_input("Non-conformités majeures:", min_value=0, value=2)
            minor_nc = st.number_input("Non-conformités mineures:", min_value=0, value=5)
            observations = st.number_input("Observations:", min_value=0, value=3)
            
            # Recommandations
            st.markdown("##### Recommandations")
            actions_required = st.text_area("Actions correctives requises:", 
                                          "1. Mettre à jour la procédure\n2. Former le personnel\n3. Réviser les contrôles")
            
            if st.form_submit_button("Générer le rapport"):
                st.success("Rapport d'audit généré!")
                
                # Score d'audit
                total_points = 100
                deduction = (nc_count * 10) + (minor_nc * 5) + (observations * 2)
                score = max(0, total_points - deduction)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score Audit", f"{score}/100")
                with col2:
                    st.metric("Niveau de Conformité", 
                             f"{(score/100*100):.0f}%",
                             delta="Conforme" if score >= 80 else "Non conforme")
                with col3:
                    st.metric("Actions Requises", nc_count + minor_nc)
                
                # Timeline des actions correctives
                st.markdown("#### 📅 Plan d'Actions Correctives")
                
                actions_df = pd.DataFrame({
                    'Action': ['Mise à jour procédure', 'Formation personnel', 'Révision contrôles'],
                    'Responsable': ['Qualité', 'Formation', 'Production'],
                    'Date limite': ['2024-04-01', '2024-04-15', '2024-04-30'],
                    'Statut': ['En cours', 'Planifié', 'À planifier']
                })
                
                st.dataframe(actions_df, use_container_width=True)
        
        # Revue de processus
        st.markdown("#### 🔄 Revue de Processus Mensuelle")
        
        review_date = st.date_input("Date de la revue:", datetime.now())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Points forts")
            strengths = st.text_area("", "Amélioration continue\nImplication de l'équipe\nRésultats stables")
        
        with col2:
            st.markdown("##### Points d'amélioration")
            improvements = st.text_area("", "Documentation à compléter\nTemps de réponse\nCommunication inter-équipes")
        
        if st.button("Enregistrer la revue", key="save_review"):
            st.success("Revue enregistrée dans le système!")
            
            # Génération du compte-rendu
            st.download_button(
                "📥 Télécharger le compte-rendu",
                f"Revue processus - {review_date}\n\nPoints forts:\n{strengths}\n\nPoints d'amélioration:\n{improvements}",
                file_name=f"revue_processus_{review_date}.txt"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- ROUTAGE DES PHASES ---
if phase == "D - Définir":
    phase_definir()
elif phase == "M - Mesurer":
    phase_mesurer()
elif phase == "A - Analyser":
    phase_analyser()
elif phase == "I - Innover":
    phase_innover()
elif phase == "C - Contrôler":
    phase_controler()

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Six Sigma Pro Suite v2.0 | Outils DMAIC complets | © 2024 Excellence Opérationnelle</p>
        <p style='font-size: 0.9em;'>Pour support technique : support@sixsigma-suite.com</p>
    </div>
    """,
    unsafe_allow_html=True
)
