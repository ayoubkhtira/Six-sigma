import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="Six Sigma Pro Suite", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stDataEditor { border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    h1, h2 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1E3A8A; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION DES DONNÉES (SESSION STATE) ---
if 'df_amdec' not in st.session_state:
    # Données initiales pour l'exemple
    st.session_state.df_amdec = pd.DataFrame([
        {"Processus": "Soudure", "Mode de défaillance": "Fissure", "G": 9, "O": 3, "D": 2},
        {"Processus": "Peinture", "Mode de défaillance": "Rayure", "G": 4, "O": 6, "D": 4}
    ])

if 'df_gage' not in st.session_state:
    # Simulation de données Gage R&R
    data = []
    for op in ['OpA', 'OpB']:
        for p in [f'Pièce {i}' for i in range(1, 6)]:
            data.append({"Opérateur": op, "Pièce": p, "Mesure": round(10 + np.random.normal(0, 0.05), 3)})
    st.session_state.df_gage = pd.DataFrame(data)

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Six_Sigma_logo.svg/1200px-Six_Sigma_logo.svg.png", width=100)
    st.title("Navigation")
    step = st.radio("Phase DMAIC", ["D - Définir", "M - Mesurer", "A - Analyser", "I - Innover", "C - Contrôler"])
    st.divider()
    st.download_button("📥 Exporter les données (CSV)", 
                       st.session_state.df_amdec.to_csv(index=False), 
                       "data_six_sigma.csv", "text/csv")

# --- LOGIQUE DES ÉTAPES ---

# --- ANALYSER : AMDEC PRO ---
if "A - Analyser" in step:
    st.title("🛡️ AMDEC : Gestion des Risques")
    st.markdown("<div class='status-box'><b>Note Pro :</b> Modifiez directement les cellules du tableau ci-dessous. Les IPR sont calculés en temps réel.</div>", unsafe_allow_html=True)

    # Éditeur de données interactif
    edited_df = st.data_editor(
        st.session_state.df_amdec,
        num_rows="dynamic", # Permet d'ajouter/supprimer des lignes
        column_config={
            "G": st.column_config.NumberColumn("Gravité", min_value=1, max_value=10, format="%d"),
            "O": st.column_config.NumberColumn("Occurrence", min_value=1, max_value=10, format="%d"),
            "D": st.column_config.NumberColumn("Détection", min_value=1, max_value=10, format="%d"),
        },
        key="amdec_editor",
        use_container_width=True
    )

    # Sauvegarder les modifications
    st.session_state.df_amdec = edited_df

    # Calcul de l'IPR et Visualisation
    if not edited_df.empty:
        df_viz = edited_df.copy()
        df_viz['IPR'] = df_viz['G'] * df_viz['O'] * df_viz['D']
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(df_viz, x="Mode de défaillance", y="IPR", color="IPR",
                         color_continuous_scale="RdYlGn_r", title="Pareto des Risques")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            avg_ipr = df_viz['IPR'].mean()
            st.metric("IPR Moyen", round(avg_ipr, 1), delta="-5%" if avg_ipr < 100 else "+2%")
            st.warning("Action requise si IPR > 100")

# --- MESURER : GAGE R&R PRO ---
elif "M - Mesurer" in step:
    st.title("📏 Gage R&R : Fiabilité de la Mesure")
    st.markdown("<div class='status-box'>Saisissez vos relevés de mesures par opérateur et par pièce.</div>", unsafe_allow_html=True)

    edited_gage = st.data_editor(
        st.session_state.df_gage,
        num_rows="dynamic",
        use_container_width=True,
        key="gage_editor"
    )
    st.session_state.df_gage = edited_gage

    if not edited_gage.empty:
        fig_gage = px.box(edited_gage, x="Opérateur", y="Mesure", color="Opérateur", 
                         points="all", title="Analyse de Reproductibilité (Variabilité Inter-Opérateur)")
        st.plotly_chart(fig_gage, use_container_width=True)

# --- AUTRES ÉTAPES ---
else:
    st.title(f"🚀 Phase {step}")
    st.info("Le module d'édition pour cette phase est en cours de configuration.")
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=200)
