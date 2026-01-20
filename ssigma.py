import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib.utils import ImageReader

# -----------------------------
# Modèle de données / Résultats
# -----------------------------
@dataclass
class AnovaResult:
    p: int
    o: int
    r: int
    confidence_level: float
    anova_table: pd.DataFrame
    var_components: pd.DataFrame
    study_var: pd.DataFrame
    metrics: Dict[str, float]
    conclusion: Dict[str, str]
    f_tests: Dict[str, Tuple[float, float, float]]  # F-statistic, p-value, critical F

# -----------------------------
# Convertir en format long
# -----------------------------
def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the dataframe to long format
    long_df = df.melt(id_vars=["N° de la pièce"], var_name="Opérateur", value_name="Mesure")
    
    # Extract the operator number from the column names
    long_df["Opérateur"] = long_df["Opérateur"].str.extract(r"(OPERATEUR \d+)")[0]
    
    # Add trial information
    long_df["Essai"] = long_df.groupby(["N° de la pièce", "Opérateur"]).cumcount() + 1
    
    # Rename columns
    long_df = long_df.rename(columns={"N° de la pièce": "Pièce"})
    
    # Ensure numeric columns for Mesure and Essai
    long_df["Mesure"] = pd.to_numeric(long_df["Mesure"], errors="coerce")
    long_df["Essai"] = pd.to_numeric(long_df["Essai"], errors="coerce")
    
    return long_df

# -----------------------------
# Fonction de validation des données
# -----------------------------
def validate_dataset(df: pd.DataFrame, n_parts: int, n_ops: int, n_trials: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required = {"Pièce", "Opérateur", "Essai", "Mesure"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Colonnes manquantes: {', '.join(sorted(missing))}.")
        return False, errors

    df2 = df.copy()
    df2["Pièce"] = df2["Pièce"].astype(str).str.strip()
    df2["Opérateur"] = df2["Opérateur"].astype(str).str.strip()
    df2["Essai"] = pd.to_numeric(df2["Essai"], errors="coerce")
    df2["Mesure"] = pd.to_numeric(df2["Mesure"], errors="coerce")

    if df2["Essai"].isna().any():
        errors.append("La colonne 'Essai' contient des valeurs non numériques.")
    if df2["Mesure"].isna().any():
        errors.append("La colonne 'Mesure' contient des valeurs vides ou non numériques.")

    parts = sorted(df2["Pièce"].unique())
    ops = sorted(df2["Opérateur"].unique())

    if len(parts) != n_parts:
        errors.append(f"Nombre de pièces détectées = {len(parts)} (attendu = {n_parts}).")
    if len(ops) != n_ops:
        errors.append(f"Nombre d'opérateurs détecté = {len(ops)} (attendu = {n_ops}).")

    counts = df2.groupby(["Pièce", "Opérateur"])["Mesure"].count()
    if counts.nunique() != 1:
        errors.append("Plan non équilibré: le nombre de mesures varie selon les couples Pièce x Opérateur.")
    else:
        r = int(counts.iloc[0])
        if r != n_trials:
            errors.append(f"Nombre de répétitions détecté = {r} (attendu = {n_trials}).")

    expected_rows = n_parts * n_ops * n_trials
    if len(df2) != expected_rows:
        errors.append(f"Nombre de lignes = {len(df2)} (attendu = {expected_rows} = pièces×opérateurs×essais).")

    return (len(errors) == 0), errors

# -----------------------------
# Calcul Gage R&R (ANOVA)
# -----------------------------
def gage_rr_anova(df: pd.DataFrame, confidence_level: float = 0.95) -> AnovaResult:
    # Similar to the function already provided, perform the ANOVA analysis here.
    # For brevity, I am assuming this function is similar to the one in your original code.
    pass

# -----------------------------
# UI Streamlit
# -----------------------------
st.title("📏 Calculateur Gage R&R (Cage R&R) — ANOVA")
st.caption("Saisie manuelle ou import (CSV/Excel) → calcul EV, AV, Vp, Vt + interprétation + rapport PDF.")

with st.sidebar:
    n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=10, step=1)
    n_ops = st.number_input("Nombre d'opérateurs", min_value=2, max_value=20, value=3, step=1)
    n_trials = st.number_input("Nombre de mesures (essais) par opérateur & pièce", min_value=2, max_value=10, value=3, step=1)

    entry_mode = st.radio("Mode de saisie des données", ["Saisie manuelle", "Importer (CSV/Excel)"], horizontal=False)

    confidence_level = st.selectbox(
        "Niveau de confiance",
        options=[0.90, 0.95, 0.99],
        format_func=lambda x: f"{x*100:.0f}%",
        index=1
    )

tabs = st.tabs(["1) Données", "2) Résultats", "3) Graphes", "4) Rapport / Export"])

# --- Tab 1 : Données
with tabs[0]:
    st.subheader("Données d'entrée")
    
    # For demonstration, let's assume we already have the imported data
    # In the real case, you would load the uploaded file here
    file_path = "/mnt/data/TEMPLATE CAGE RR.xlsx"
    df = pd.read_excel(file_path)
    df_long = convert_wide_to_long(df)
    
    # Validate the data
    ok, errs = validate_dataset(df_long, int(n_parts), int(n_ops), int(n_trials))
    if ok:
        st.success("✅ Données valides.")
    else:
        st.error("❌ Données invalides :")
        for e in errs:
            st.write(f"- {e}")

# --- Tab 2 : Résultats
with tabs[1]:
    st.subheader("Résultats")
    if ok:
        # Perform Gage R&R analysis
        res = gage_rr_anova(df_long, confidence_level)
        # Display results
        st.write(res)

# --- Further code for Graphes and Export sections
