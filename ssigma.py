import streamlit as st
import sqlite3
import pandas as pd
import os
import shutil
from datetime import datetime
from PIL import Image
import io
from fpdf import FPDF  # <-- AJOUT : Import de FPDF

# -----------------------------------------
# CONFIGURATION ET INITIALISATION
# -----------------------------------------
st.set_page_config(page_title="GMAO & Inventaire", page_icon="📦", layout="wide")

DOSSIER_IMAGES = "images_sauvegardees"
if not os.path.exists(DOSSIER_IMAGES):
    os.makedirs(DOSSIER_IMAGES)

def init_db():
    conn = sqlite3.connect('interventions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_heure TEXT,
            nom TEXT,
            fonction TEXT,
            type TEXT,
            description TEXT,
            chemins_images TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------------------
# FONCTIONS UTILITAIRES
# -----------------------------------------
def supprimer_intervention(id_interv, chemins_images):
    conn = sqlite3.connect('interventions.db')
    c = conn.cursor()
    c.execute("DELETE FROM interventions WHERE id=?", (id_interv,))
    conn.commit()
    conn.close()
    if chemins_images:
        for chemin in chemins_images.split(','):
            if os.path.exists(chemin):
                os.remove(chemin)
    return True

# ==========================================
# AJOUT : FONCTION DE GÉNÉRATION PDF
# ==========================================
def generer_pdf(donnees, chemins_images):
    """
    Génère un PDF à partir des données d'intervention et des chemins d'images
    """
    pdf = FPDF()
    pdf.add_page()
    
    # --- En-tête ---
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "FICHE D'INTERVENTION", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="R")
    pdf.ln(5)
    
    # --- Informations Client/Intervenant ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, " Détails de l'intervention", ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    
    pdf.cell(50, 10, "ID Intervention :", border="B")
    pdf.cell(0, 10, str(donnees['id']), border="B", ln=True)
    
    pdf.cell(50, 10, "Intervenant :", border="B")
    pdf.cell(0, 10, f"{donnees['nom']} ({donnees['fonction']})", border="B", ln=True)
    
    pdf.cell(50, 10, "Date & Heure :", border="B")
    pdf.cell(0, 10, str(donnees['date_heure']), border="B", ln=True)
    
    pdf.cell(50, 10, "Nature :", border="B")
    pdf.cell(0, 10, str(donnees['type']), border="B", ln=True)
    
    pdf.ln(5)
    
    # --- Description ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, " Rapport de mission", ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 10, str(donnees['description']), border=1)
    
    pdf.ln(10)
    
    # --- Photos ---
    if chemins_images:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, " Photos jointes", ln=True, fill=True)
        pdf.ln(5)
        
        y_start = pdf.get_y()
        x_start = 10
        img_width = 60
        
        for i, chemin in enumerate(chemins_images.split(',')):
            chemin = chemin.strip()
            if os.path.exists(chemin):
                try:
                    # Aligner 3 images par ligne
                    col = i % 3
                    row = i // 3
                    pdf.image(chemin, x=x_start + (col * 65), y=y_start + (row * 65), w=img_width)
                except Exception as e:
                    print(f"Erreur lors de l'insertion de l'image {chemin}: {e}")
    
    # Retourner le PDF sous forme de bytes
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------
# INTERFACE PRINCIPALE
# -----------------------------------------
st.title("📦 Gestion des Interventions & Inventaires")

onglet_saisie, onglet_historique = st.tabs(["📝 Saisie", "📊 Historique & Export"])

# ==========================================
# ONGLET 1 : SAISIE
# ==========================================
with onglet_saisie:
    with st.form("form_interv", clear_on_submit=True):
        st.subheader("Informations Générales")
        c1, c2, c3 = st.columns(3)
        nom = c1.text_input("Nom de l'intervenant / Responsable")
        fonction = c2.text_input("Fonction")
        
        # AJOUT DE L'OPTION "INVENTAIRE" ICI
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
            
            conn = sqlite3.connect('interventions.db')
            c = conn.cursor()
            c.execute("INSERT INTO interventions (date_heure, nom, fonction, type, description, chemins_images) VALUES (?,?,?,?,?,?)",
                      (datetime.now().strftime("%d/%m/%Y %H:%M"), nom, fonction, type_interv, description, ",".join(chemins)))
            conn.commit()
            conn.close()
            st.success(f"✅ Opération de type '{type_interv}' enregistrée avec succès.")
        else:
            st.warning("⚠️ Veuillez remplir au moins le nom et la description.")

# ==========================================
# ONGLET 2 : HISTORIQUE & EXPORT
# ==========================================
with onglet_historique:
    conn = sqlite3.connect('interventions.db')
    df = pd.read_sql_query("SELECT * FROM interventions ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        st.subheader("Indicateurs clés")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Opérations", len(df))
        m2.metric("Dernière saisie", df['date_heure'].iloc[0])
        m3.metric("Intervenants", len(df['nom'].unique()))
        # Statistique spécifique pour l'inventaire
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

        # Filtrage dynamique
        df_display = df.copy()
        if search:
            mask = df_display.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
            df_display = df_display[mask]

        st.dataframe(df_display.drop(columns=['chemins_images']), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔍 Détails et Médias")
        selected_id = st.selectbox("Sélectionner un ID pour consulter", options=df_display['id'].tolist())
        
        if selected_id:
            row = df_display[df_display['id'] == selected_id].iloc[0]
            
            # ==========================================
            # AJOUT : Bouton de téléchargement PDF
            # ==========================================
            col_pdf, col_details = st.columns([1, 3])
            
            with col_pdf:
                st.write("")  # Espacement
                st.write("")  # Espacement
                # Génération du PDF
                pdf_bytes = generer_pdf(row, row['chemins_images'])
                
                st.download_button(
                    label="📄 Télécharger la fiche PDF",
                    data=pdf_bytes,
                    file_name=f"Fiche_Intervention_{selected_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col_details:
                c_det1, c_det2 = st.columns([2, 1])
                with c_det1:
                    st.write(f"**Type :** {row['type']}")
                    st.write(f"**Description :** {row['description']}")
                    imgs = row['chemins_images']
                    if imgs:
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
        st.info("La base de données est vide.")
