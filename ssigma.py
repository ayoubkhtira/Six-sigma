import streamlit as st
import sys
import subprocess
import importlib.util

# Fonction pour vérifier et installer les packages
def check_and_install_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    # Vérifier si le package est déjà installé
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        st.warning(f"📦 Package '{package_name}' non trouvé. Installation en cours...")
        progress_bar = st.progress(0)
        
        try:
            # Essayer d'installer le package
            progress_bar.progress(50)
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            progress_bar.progress(100)
            st.success(f"✅ Package '{package_name}' installé avec succès!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erreur lors de l'installation de {package_name}: {e}")
            st.info("💡 Installation manuelle: pip install " + package_name)
            return False
    return True

# Vérifier et installer streamlit-authenticator
if not check_and_install_package("streamlit-authenticator", "streamlit_authenticator"):
    st.stop()

# Maintenant on peut importer
import streamlit_authenticator as stauth
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import io

# -----------------------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------------------
st.set_page_config(page_title="GMAO & Inventaire", page_icon="📦", layout="wide")

# -----------------------------------------
# CONFIGURATION ET INITIALISATION
# -----------------------------------------
DOSSIER_IMAGES = "images_sauvegardees"
FICHIER_CSV = "interventions.csv"

if not os.path.exists(DOSSIER_IMAGES):
    os.makedirs(DOSSIER_IMAGES)

# -----------------------------------------
# FONCTIONS DE GESTION CSV
# -----------------------------------------
def lire_csv():
    """Lit le fichier CSV et retourne un DataFrame"""
    if os.path.exists(FICHIER_CSV):
        return pd.read_csv(FICHIER_CSV)
    else:
        return pd.DataFrame(columns=['id', 'date_heure', 'nom', 'fonction', 'type', 'description', 'chemins_images'])

def ecrire_csv(df):
    """Écrit le DataFrame dans le fichier CSV"""
    df.to_csv(FICHIER_CSV, index=False)

def ajouter_intervention(nom, fonction, type_interv, description, chemins_images):
    """Ajoute une nouvelle intervention dans le CSV"""
    df = lire_csv()
    
    if df.empty:
        new_id = 1
    else:
        new_id = df['id'].max() + 1
    
    nouvelle_ligne = pd.DataFrame({
        'id': [new_id],
        'date_heure': [datetime.now().strftime("%d/%m/%Y %H:%M")],
        'nom': [nom],
        'fonction': [fonction],
        'type': [type_interv],
        'description': [description],
        'chemins_images': [",".join(chemins_images)]
    })
    
    df = pd.concat([df, nouvelle_ligne], ignore_index=True)
    ecrire_csv(df)
    return new_id

def supprimer_intervention(id_interv, chemins_images):
    """Supprime une intervention du CSV et ses images"""
    df = lire_csv()
    df = df[df['id'] != id_interv]
    ecrire_csv(df)
    
    if chemins_images and isinstance(chemins_images, str):
        for chemin in chemins_images.split(','):
            if os.path.exists(chemin):
                os.remove(chemin)
    return True

# --- CONFIGURATION DES UTILISATEURS ---
names = ["Administrateur"]
usernames = ["admin"]
passwords = ["1234"]

# Hachage des mots de passe
hashed_passwords = stauth.Hasher(passwords).generate()

# Création de l'objet d'authentification
authenticator = stauth.Authenticate(
    {'usernames': {
        usernames[0]: {'name': names[0], 'password': hashed_passwords[0]}
    }},
    "cookie_intervention",
    "abcdef",
    cookie_expiry_days=30
)

# --- AFFICHAGE DE L'ÉCRAN DE CONNEXION ---
name, authentication_status, username = authenticator.login('Connexion', 'main')

if authentication_status == False:
    st.error('L\'identifiant ou le mot de passe est incorrect')

elif authentication_status == None:
    st.warning('Veuillez entrer votre identifiant et votre mot de passe')

elif authentication_status:
    with st.sidebar:
        st.write(f"Bienvenue **{name}**")
        authenticator.logout('Déconnexion', 'sidebar')

    st.title("📦 Gestion des Interventions & Inventaires")

    onglet_saisie, onglet_historique = st.tabs(["📝 Saisie", "📊 Historique & Export"])

    with onglet_saisie:
        with st.form("form_interv", clear_on_submit=True):
            st.subheader("Informations Générales")
            c1, c2, c3 = st.columns(3)
            nom = c1.text_input("Nom de l'intervenant / Responsable")
            fonction = c2.text_input("Fonction")
            
            type_interv = c3.selectbox("Nature de l'opération", [
                "Maintenance Préventive", 
                "Maintenance Curative", 
                "Installation", 
                "Audit", 
                "Inventaire", 
                "Autre"
            ])
            
            description = st.text_area("Rapport détaillé (Travaux, Liste de matériel, observations...)")
            
            st.subheader("Photos / Justificatifs")
            fichiers = st.file_uploader("Preuves photos", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            
            submit = st.form_submit_button("💾 Enregistrer dans la base", use_container_width=True)

        if submit:
            if nom and description:
                chemins = []
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, f in enumerate(fichiers):
                    path = os.path.join(DOSSIER_IMAGES, f"{ts}_{i}_{f.name}")
                    with open(path, "wb") as buffer:
                        buffer.write(f.getbuffer())
                    chemins.append(path)
                
                ajouter_intervention(nom, fonction, type_interv, description, chemins)
                st.success(f"✅ Opération de type '{type_interv}' enregistrée avec succès.")
            else:
                st.warning("⚠️ Veuillez remplir au moins le nom et la description.")

    with onglet_historique:
        df = lire_csv()

        if not df.empty:
            st.subheader("Indicateurs clés")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Opérations", len(df))
            m2.metric("Dernière saisie", df['date_heure'].iloc[0])
            m3.metric("Intervenants", len(df['nom'].unique()))
            nb_inventaires = len(df[df['type'] == "Inventaire"])
            m4.metric("Nb Inventaires", nb_inventaires)

            st.divider()

            col_f1, col_f2, col_export = st.columns([2, 2, 1])
            with col_f1:
                search = st.text_input("🔍 Rechercher (Nom, Type ou Description)")
            with col_export:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Interventions_Inventaires')
                processed_data = output.getvalue()
                
                st.download_button(
                    label="📥 Export complet vers Excel",
                    data=processed_data,
                    file_name=f"Rapport_Global_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            df_display = df.copy()
            if search:
                mask = df_display.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
                df_display = df_display[mask]

            st.dataframe(df_display.drop(columns=['chemins_images']), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🔍 Détails et Médias")
            
            if not df_display.empty:
                selected_id = st.selectbox("Sélectionner un ID pour consulter", options=df_display['id'].tolist())
                
                if selected_id:
                    row = df_display[df_display['id'] == selected_id].iloc[0]
                    
                    c_det1, c_det2 = st.columns([2, 1])
                    with c_det1:
                        st.write(f"**Type :** {row['type']}")
                        st.write(f"**Description :** {row['description']}")
                        imgs = row['chemins_images']
                        if imgs and isinstance(imgs, str) and imgs.strip():
                            list_imgs = imgs.split(',')
                            cols = st.columns(3)
                            for idx, p in enumerate(list_imgs):
                                if os.path.exists(p):
                                    try:
                                        cols[idx % 3].image(Image.open(p), use_container_width=True)
                                    except:
                                        cols[idx % 3].warning("Image corrompue")
                        else:
                            st.info("Aucun média joint.")
                    
                    with c_det2:
                        st.error("Administration")
                        if st.button("🗑️ Supprimer l'entrée", use_container_width=True):
                            if supprimer_intervention(selected_id, row['chemins_images']):
                                st.success("Entrée supprimée.")
                                st.rerun()
            else:
                st.info("Aucun résultat trouvé pour votre recherche.")
        else:
            st.info("La base de données est vide. Commencez par ajouter une intervention dans l'onglet 'Saisie'.")
