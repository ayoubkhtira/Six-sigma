import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Personnalisation du CSS pour améliorer le style
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .header {
            font-size: 32px;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
        }
        .subheader {
            font-size: 20px;
            color: #388E3C;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            border: none;
            font-size: 16px;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Fonction pour calculer les éléments liés à Cage R&R
def calculate_piece_variance(df):
    piece_means = df.groupby('Pièces')['Mesures'].mean()
    return piece_means.var(ddof=1)

def calculate_operator_variance(df):
    operator_means = df.groupby('Opérateurs')['Mesures'].mean()
    return operator_means.var(ddof=1)

def calculate_total_variance(df):
    return df['Mesures'].var(ddof=1)

def calculate_error_variation(df):
    return df['Mesures'].std(ddof=1)

def calculate_interaction_variance(df, Vp, Vop, Vt):
    return Vt - (Vp + Vop)

def calculate_cage_r_and_r(df):
    Vp = calculate_piece_variance(df)
    Vop = calculate_operator_variance(df)
    Vt = calculate_total_variance(df)
    EV = calculate_error_variation(df)
    interaction_variance = calculate_interaction_variance(df, Vp, Vop, Vt)
    return Vp, Vop, Vt, EV, interaction_variance

# Fonction pour afficher un graphique
def plot_graph(df):
    plt.figure(figsize=(10,6))
    plt.boxplot(df['Mesures'], vert=False)
    plt.title('Graphique des Mesures', fontsize=16)
    plt.xlabel('Valeurs des Mesures', fontsize=14)
    st.pyplot(plt)

# Fonction pour générer un rapport
def generate_report(df, Vp, Vop, Vt, EV, interaction_variance):
    report = f"""
    Rapport Cage R&R:
    -------------------
    - Nombre de pièces: {len(df['Pièces'].unique())}
    - Nombre d'opérateurs: {len(df['Opérateurs'].unique())}
    - Nombre de mesures par opérateur: {len(df) // len(df['Opérateurs'].unique())}

    Calculs:
    ---------
    - Variance des pièces (Vp): {Vp}
    - Variance des opérateurs (Vop): {Vop}
    - Variance totale (Vt): {Vt}
    - Erreur de variation (EV): {EV}
    - Interaction opérateur-pièce (Vin): {interaction_variance}

    Interprétation:
    ---------------
    - Si Vp est significativement plus grand que Vop et EV, cela signifie que la variation est principalement causée par les pièces.
    - Si Vop est important, cela indique que l'opérateur a un impact sur la variation des mesures.
    - Une interaction (Vin) importante suggère que les opérateurs peuvent traiter différemment certaines pièces.
    """
    
    with open("rapport_cage_rr.txt", "w") as file:
        file.write(report)
    
    st.download_button('Télécharger le rapport', 'rapport_cage_rr.txt')

# Titre de l'application avec un style moderne
st.markdown('<div class="header">Calcul de la Cage R&R</div>', unsafe_allow_html=True)

# Utilisation de la barre latérale pour l'organisation des entrées
with st.sidebar:
    st.header("Paramètres d'Entrée")
    num_pieces = st.number_input("Entrez le nombre de pièces", min_value=1, step=1)
    num_operators = st.number_input("Entrez le nombre d'opérateurs", min_value=1, step=1)
    num_measurements = st.number_input("Entrez le nombre de mesures par opérateur", min_value=1, step=1)

    data_choice = st.radio("Choisir la méthode d'entrée des données", ('Manuel', 'Importer'))

# Ajout d'un conteneur central pour le calcul
st.markdown('<div class="subheader">Résultats</div>', unsafe_allow_html=True)
container = st.container()

if data_choice == 'Manuel':
    with container:
        data = []
        for i in range(num_operators):
            for j in range(num_pieces):
                piece = st.number_input(f"Entrez la mesure pour l'opérateur {i+1}, pièce {j+1}", key=f"measure_{i}_{j}")
                data.append([f"Opérateur {i+1}", f"Pièce {j+1}", piece])

        df = pd.DataFrame(data, columns=['Opérateurs', 'Pièces', 'Mesures'])

elif data_choice == 'Importer':
    with container:
        uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

if 'df' in locals():
    Vp, Vop, Vt, EV, interaction_variance = calculate_cage_r_and_r(df)
    
    # Affichage des résultats avec un style moderne
    st.markdown(f"**Variance des pièces (Vp):** {Vp}")
    st.markdown(f"**Variance des opérateurs (Vop):** {Vop}")
    st.markdown(f"**Variance totale (Vt):** {Vt}")
    st.markdown(f"**Erreur de variation (EV):** {EV}")
    st.markdown(f"**Interaction opérateur-pièce (Vin):** {interaction_variance}")
    
    # Affichage du graphique
    plot_graph(df)
    
    # Génération du rapport
    generate_report(df, Vp, Vop, Vt, EV, interaction_variance)
