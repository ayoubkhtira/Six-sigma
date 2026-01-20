from __future__ import annotations
import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
# Config Streamlit (UX)
# -----------------------------
st.set_page_config(
    page_title="Gage R&R (Cage R&R) — Calculateur",
    page_icon="📏",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.kpi-card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.60);
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.kpi-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
.kpi-sub { font-size: 0.85rem; opacity: 0.75; margin-top: 0.35rem; }

.rr-card { border-radius: 16px; padding: 16px 16px; color: white; }
.rr-green { background: linear-gradient(135deg, rgba(21, 128, 61, 0.95), rgba(34, 197, 94, 0.85)); }
.rr-orange { background: linear-gradient(135deg, rgba(180, 83, 9, 0.95), rgba(245, 158, 11, 0.85)); }
.rr-red { background: linear-gradient(135deg, rgba(153, 27, 27, 0.95), rgba(239, 68, 68, 0.85)); }
.rr-card h3 { margin: 0 0 0.4rem 0; }
.rr-card p { margin: 0.2rem 0; opacity: 0.95; }

.small-note { font-size: 0.88rem; opacity: 0.75; }
hr { margin: 1.0rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Modèle de données / Résultats
# -----------------------------
@dataclass
class AnovaResult:
    p: int
    o: int
    r: int
    anova_table: pd.DataFrame
    var_components: pd.DataFrame
    study_var: pd.DataFrame
    metrics: Dict[str, float]
    conclusion: Dict[str, str]


# -----------------------------
# Utilitaires
# -----------------------------
def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def build_template(n_parts: int, n_ops: int, n_trials: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n_parts + 1):
        for j in range(1, n_ops + 1):
            for t in range(1, n_trials + 1):
                rows.append({"Pièce": f"P{i}", "Opérateur": f"O{j}", "Essai": t, "Mesure": np.nan})
    return pd.DataFrame(rows)


def normalize_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepte plusieurs variantes de noms de colonnes et les ramène à:
    Pièce, Opérateur, Essai, Mesure
    """
    colmap = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in {"pièce", "piece", "part", "n° de la pièce", "no de la piece", "numéro pièce"}:
            colmap[c] = "Pièce"
        elif key in {"opérateur", "operateur", "operator"}:
            colmap[c] = "Opérateur"
        elif key in {"essai", "trial", "rep", "répétition", "repetition", "repeat"}:
            colmap[c] = "Essai"
        elif key in {"mesure", "measurement", "valeur", "value", "y"}:
            colmap[c] = "Mesure"
    return df.rename(columns=colmap).copy()


def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un format large (comme le template) en format long.
    
    Format attendu:
    - Première ligne: noms d'opérateurs avec répétition (ex: OPERATEUR 1, OPERATEUR 1, OPERATEUR 1)
    - Deuxième ligne: numéros d'essai (ESSAI 1, ESSAI 2, ESSAI 3)
    - Colonne A: numéros de pièces
    
    Retourne un DataFrame avec colonnes: Pièce, Opérateur, Essai, Mesure
    """
    df = df.copy()
    
    # Supprimer les lignes vides
    df = df.dropna(how='all')
    
    # Si le DataFrame a un MultiIndex (lignes d'en-tête fusionnées), le gérer
    if isinstance(df.columns, pd.MultiIndex):
        # Extraire les noms d'opérateurs et essais du MultiIndex
        operators = []
        trials = []
        
        for col in df.columns:
            if pd.isna(col[0]) or str(col[0]).strip() == '':
                operators.append(operators[-1] if operators else 'Opérateur 1')
            else:
                operators.append(str(col[0]).strip())
            trials.append(str(col[1]).strip() if len(col) > 1 and not pd.isna(col[1]) else '1')
    else:
        # Sinon, lire depuis les deux premières lignes
        if len(df) < 2:
            raise ValueError("Format de template invalide: besoin d'au moins 2 lignes d'en-tête")
        
        # Les deux premières lignes contiennent les en-têtes
        operators_row = df.iloc[0].fillna(method='ffill')
        trials_row = df.iloc[1]
        
        # Extraire les données à partir de la 3ème ligne
        data_df = df.iloc[2:].reset_index(drop=True)
        
        # Préparer les listes d'opérateurs et essais
        operators = []
        trials = []
        
        for i in range(len(operators_row)):
            if i == 0:  # Première colonne: numéro de pièce
                operators.append('Opérateur')
                trials.append('Essai')
            else:
                operators.append(str(operators_row.iloc[i]).strip())
                trials.append(str(trials_row.iloc[i]).strip())
        
        # Renommer les colonnes
        data_df.columns = pd.MultiIndex.from_tuples(zip(operators, trials))
        df = data_df
    
    # Convertir en format long
    long_rows = []
    
    for idx, row in df.iterrows():
        piece_num = str(row.iloc[0]).strip()
        if not piece_num or piece_num.lower() in ['pièce', 'piece', 'part']:
            continue
            
        # Pour chaque colonne de mesure
        for i in range(1, len(row)):
            if pd.isna(row.iloc[i]):
                continue
                
            op_name = operators[i]
            trial_num = trials[i]
            
            # Extraire le numéro d'essai (ex: "ESSAI 1" -> 1)
            trial_val = 1
            if isinstance(trial_num, str):
                for part in trial_num.split():
                    if part.isdigit():
                        trial_val = int(part)
                        break
            
            # Extraire le nom d'opérateur (ex: "OPERATEUR 1" -> "O1")
            op_val = op_name
            if isinstance(op_name, str):
                for part in op_name.split():
                    if part.isdigit():
                        op_val = f"O{part}"
                        break
            
            long_rows.append({
                'Pièce': piece_num,
                'Opérateur': op_val,
                'Essai': trial_val,
                'Mesure': float(row.iloc[i])
            })
    
    return pd.DataFrame(long_rows)


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
def gage_rr_anova(df: pd.DataFrame) -> AnovaResult:
    """
    Calcule l'étude Gage R&R par ANOVA (plan équilibré).

    DF attendu: colonnes Pièce, Opérateur, Essai, Mesure
    """
    d = df.copy()
    d["Pièce"] = d["Pièce"].astype(str)
    d["Opérateur"] = d["Opérateur"].astype(str)
    d["Essai"] = d["Essai"].astype(int)
    d["Mesure"] = pd.to_numeric(d["Mesure"], errors="coerce")

    if d["Mesure"].isna().any():
        raise ValueError("Valeurs de mesure manquantes / non numériques.")

    parts = sorted(d["Pièce"].unique())
    ops = sorted(d["Opérateur"].unique())
    p = len(parts)
    o = len(ops)

    counts = d.groupby(["Pièce", "Opérateur"])["Mesure"].count()
    if counts.nunique() != 1:
        raise ValueError("Plan non équilibré: répétitions différentes selon Pièce x Opérateur.")
    r = int(counts.iloc[0])

    grand_mean = d["Mesure"].mean()
    mean_part = d.groupby("Pièce")["Mesure"].mean()
    mean_op = d.groupby("Opérateur")["Mesure"].mean()
    mean_po = d.groupby(["Pièce", "Opérateur"])["Mesure"].mean()

    # Sommes des carrés (balanced ANOVA)
    SS_part = o * r * ((mean_part - grand_mean) ** 2).sum()
    SS_op = p * r * ((mean_op - grand_mean) ** 2).sum()

    tmp = mean_po.reset_index().rename(columns={"Mesure": "mean_po"})
    tmp["mean_part"] = tmp["Pièce"].map(mean_part)
    tmp["mean_op"] = tmp["Opérateur"].map(mean_op)
    tmp["res"] = tmp["mean_po"] - tmp["mean_part"] - tmp["mean_op"] + grand_mean
    SS_po = r * (tmp["res"] ** 2).sum()

    d2 = d.merge(mean_po.rename("mean_po").reset_index(), on=["Pièce", "Opérateur"], how="left")
    SS_e = ((d2["Mesure"] - d2["mean_po"]) ** 2).sum()
    SS_total = ((d["Mesure"] - grand_mean) ** 2).sum()

    # ddl
    df_part = p - 1
    df_op = o - 1
    df_po = (p - 1) * (o - 1)
    df_e = p * o * (r - 1)
    df_total = p * o * r - 1

    # Carrés moyens
    MS_part = SS_part / df_part if df_part > 0 else np.nan
    MS_op = SS_op / df_op if df_op > 0 else np.nan
    MS_po = SS_po / df_po if df_po > 0 else np.nan
    MS_e = SS_e / df_e if df_e > 0 else np.nan

    # Composantes de variance (méthode ANOVA)
    sigma_e2 = MS_e  # répétabilité (Equipment)

    sigma_po2 = max((MS_po - MS_e) / r, 0.0) if df_po > 0 else 0.0  # interaction

    sigma_o2 = (
        max((MS_op - MS_po) / (p * r), 0.0) if df_po > 0 else max((MS_op - MS_e) / (p * r), 0.0)
    )  # opérateur

    sigma_p2 = (
        max((MS_part - MS_po) / (o * r), 0.0) if df_po > 0 else max((MS_part - MS_e) / (o * r), 0.0)
    )  # pièce-à-pièce

    sigma_grr2 = sigma_e2 + sigma_o2 + sigma_po2
    sigma_total2 = sigma_grr2 + sigma_p2

    # Sigmas
    sigma_e = math.sqrt(max(sigma_e2, 0.0))
    sigma_av = math.sqrt(max(sigma_o2 + sigma_po2, 0.0))
    sigma_p = math.sqrt(max(sigma_p2, 0.0))
    sigma_grr = math.sqrt(max(sigma_grr2, 0.0))
    sigma_total = math.sqrt(max(sigma_total2, 0.0))

    # Study Variation (6σ)
    EV = 6.0 * sigma_e
    AV = 6.0 * sigma_av
    GRR = 6.0 * sigma_grr
    PV = 6.0 * sigma_p
    TV = 6.0 * sigma_total

    pct_EV = (EV / TV) * 100 if TV > 0 else np.nan
    pct_AV = (AV / TV) * 100 if TV > 0 else np.nan
    pct_GRR = (GRR / TV) * 100 if TV > 0 else np.nan
    pct_PV = (PV / TV) * 100 if TV > 0 else np.nan

    # %Contribution (variance)
    contrib = {
        "Équipement (EV)": sigma_e2,
        "Opérateur": sigma_o2,
        "Pièce*Opérateur": sigma_po2,
        "Pièce-à-pièce (PV)": sigma_p2,
        "Total": sigma_total2,
    }
    pct_contrib = {k: (v / sigma_total2) * 100 if sigma_total2 > 0 else np.nan for k, v in contrib.items()}

    ndc = int(math.floor(1.41 * (PV / GRR))) if GRR > 0 else 0

    anova_table = pd.DataFrame(
        {
            "Source": ["Pièce", "Opérateur", "Pièce*Opérateur", "Erreur (Répétabilité)", "Total"],
            "SS": [SS_part, SS_op, SS_po, SS_e, SS_total],
            "ddl": [df_part, df_op, df_po, df_e, df_total],
            "MS": [MS_part, MS_op, MS_po, MS_e, np.nan],
        }
    )

    var_components = pd.DataFrame(
        {
            "Composante": [
                "EV (Répétabilité)",
                "Opérateur",
                "Pièce*Opérateur",
                "AV (Reproductibilité)",
                "PV (Vp)",
                "GRR",
                "TV (Vt)",
            ],
            "Variance": [sigma_e2, sigma_o2, sigma_po2, sigma_o2 + sigma_po2, sigma_p2, sigma_grr2, sigma_total2],
            "Sigma": [
                sigma_e,
                math.sqrt(max(sigma_o2, 0.0)),
                math.sqrt(max(sigma_po2, 0.0)),
                sigma_av,
                sigma_p,
                sigma_grr,
                sigma_total,
            ],
        }
    )

    var_components["%Contribution"] = np.nan
    base_map = {
        "EV (Répétabilité)": "Équipement (EV)",
        "Opérateur": "Opérateur",
        "Pièce*Opérateur": "Pièce*Opérateur",
        "PV (Vp)": "Pièce-à-pièce (PV)",
        "TV (Vt)": "Total",
    }
    for i, row in var_components.iterrows():
        label = row["Composante"]
        if label in base_map:
            var_components.loc[i, "%Contribution"] = pct_contrib[base_map[label]]

    study_var = pd.DataFrame(
        {
            "Indicateur": ["EV", "AV", "GRR", "PV (Vp)", "TV (Vt)"],
            "Study Variation (6σ)": [EV, AV, GRR, PV, TV],
            "%StudyVar": [pct_EV, pct_AV, pct_GRR, pct_PV, 100.0],
        }
    )

    # Interprétation (selon %R&R = %StudyVar de GRR)
    if pct_GRR < 10:
        niveau = "Satisfaisant"
        regle = "< 10 % : le processus est satisfaisant"
        css = "rr-green"
    elif pct_GRR <= 30:
        niveau = "Acceptable"
        regle = "Entre 10 % et 30 % : acceptable mais améliorable"
        css = "rr-orange"
    else:
        niveau = "Inacceptable"
        regle = "> 30 % : le processus est inacceptable"
        css = "rr-red"

    conclusion = {"niveau": niveau, "regle": regle, "css": css, "pct_grr": f"{pct_GRR:.2f}%", "ndc": str(ndc)}

    metrics = {
        "EV": EV,
        "AV": AV,
        "PV": PV,
        "TV": TV,
        "GRR": GRR,
        "%GRR": pct_GRR,
        "%EV": pct_EV,
        "%AV": pct_AV,
        "%PV": pct_PV,
        "ndc": ndc,
    }

    return AnovaResult(
        p=p,
        o=o,
        r=r,
        anova_table=anova_table,
        var_components=var_components,
        study_var=study_var,
        metrics=metrics,
        conclusion=conclusion,
    )


# -----------------------------
# Graphiques (matplotlib)
# -----------------------------
def fig_boxplot(df: pd.DataFrame, by: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    df.boxplot(column="Mesure", by=by, ax=ax, grid=False)
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel("Mesure")
    plt.suptitle("")
    fig.tight_layout()
    return fig


def fig_interaction(df: pd.DataFrame) -> plt.Figure:
    means = df.groupby(["Pièce", "Opérateur"])["Mesure"].mean().reset_index()
    pivot = means.pivot(index="Pièce", columns="Opérateur", values="Mesure")

    fig, ax = plt.subplots()
    x = np.arange(len(pivot.index))
    for op in pivot.columns:
        ax.plot(x, pivot[op].values, marker="o", label=str(op))

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
    ax.set_title("Graphique d'interaction (moyennes)")
    ax.set_xlabel("Pièce")
    ax.set_ylabel("Mesure moyenne")
    ax.legend(title="Opérateur", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def fig_contribution(var_components: pd.DataFrame) -> plt.Figure:
    base = var_components.dropna(subset=["%Contribution"]).copy()
    order = ["EV (Répétabilité)", "Opérateur", "Pièce*Opérateur", "PV (Vp)", "TV (Vt)"]
    base = base[base["Composante"].isin(order)]
    base["Composante"] = pd.Categorical(base["Composante"], categories=order, ordered=True)
    base = base.sort_values("Composante")

    fig, ax = plt.subplots()
    ax.bar(base["Composante"].astype(str), base["%Contribution"].values)
    ax.set_title("%Contribution (variance)")
    ax.set_ylabel("%")
    ax.set_xlabel("Composante")
    ax.set_xticks(np.arange(len(base)))
    ax.set_xticklabels(base["Composante"].astype(str).tolist(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def fig_studyvar(study_var: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.bar(study_var["Indicateur"], study_var["%StudyVar"].values)
    ax.set_title("%Study Variation (6σ)")
    ax.set_ylabel("%")
    ax.set_xlabel("Indicateur")
    fig.tight_layout()
    return fig


# -----------------------------
# Rapport PDF (ReportLab)
# -----------------------------
def fig_to_rl_image(fig: plt.Figure, width_cm: float = 16.5) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img_reader = ImageReader(buf)
    iw, ih = img_reader.getSize()
    w = width_cm * cm
    h = w * (ih / iw) if iw else 10 * cm
    return RLImage(buf, width=w, height=h)


def df_to_rl_table(df: pd.DataFrame, col_widths: Optional[List[float]] = None, max_rows: int = 30) -> Table:
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    data = [df2.columns.tolist()] + df2.values.tolist()
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def generate_pdf_report(df: pd.DataFrame, res: AnovaResult, tol: Optional[float] = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Rapport Gage R&R",
        author="Streamlit App",
    )
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Rapport d'étude Gage R&amp;R (Cage R&amp;R)", styles["Title"]))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1) Paramètres", styles["Heading2"]))
    story.append(
        Paragraph(
            f"- Pièces: <b>{res.p}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Opérateurs: <b>{res.o}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Répétitions (essais): <b>{res.r}</b>",
            styles["Normal"],
        )
    )
    if tol and tol > 0:
        pct_tol = (res.metrics["GRR"] / tol) * 100
        story.append(
            Paragraph(
                f"- Tolérance (optionnelle): <b>{tol}</b> → %Tolérance(GRR) ≈ <b>{pct_tol:.2f}%</b>",
                styles["Normal"],
            )
        )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("2) Conclusion", styles["Heading2"]))
    story.append(
        Paragraph(
            f"<b>%R&amp;R (Study Variation) = {res.metrics['%GRR']:.2f}%</b> — "
            f"Niveau: <b>{res.conclusion['niveau']}</b> — "
            f"ndc: <b>{res.metrics['ndc']}</b><br/>"
            f"Règle: {res.conclusion['regle']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("3) Résultats principaux", styles["Heading2"]))
    synth = pd.DataFrame(
        {
            "Indicateur": ["EV", "AV", "GRR", "PV (Vp)", "TV (Vt)", "%GRR", "ndc"],
            "Valeur": [
                _fmt(res.metrics["EV"]),
                _fmt(res.metrics["AV"]),
                _fmt(res.metrics["GRR"]),
                _fmt(res.metrics["PV"]),
                _fmt(res.metrics["TV"]),
                f"{res.metrics['%GRR']:.2f}%",
                str(res.metrics["ndc"]),
            ],
        }
    )
    story.append(df_to_rl_table(synth, col_widths=[5.0 * cm, 10.0 * cm], max_rows=20))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("4) Table ANOVA", styles["Heading2"]))
    anova_df = res.anova_table.copy()
    story.append(df_to_rl_table(anova_df.round({"SS": 6, "MS": 6}), max_rows=20))
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph("5) Composantes de variance", styles["Heading2"]))
    vc = res.var_components.copy()
    story.append(df_to_rl_table(vc.round({"Variance": 6, "Sigma": 6, "%Contribution": 2}), max_rows=30))
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph("6) %Study Variation (6σ)", styles["Heading2"]))
    sv = res.study_var.copy()
    story.append(df_to_rl_table(sv.round({"Study Variation (6σ)": 6, "%StudyVar": 2}), max_rows=20))

    story.append(PageBreak())

    story.append(Paragraph("7) Graphes", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))

    story.append(fig_to_rl_image(fig_boxplot(df, by="Opérateur", title="Boxplot des mesures par opérateur"), width_cm=17.2))
    story.append(Spacer(1, 0.35 * cm))
    story.append(fig_to_rl_image(fig_boxplot(df, by="Pièce", title="Boxplot des mesures par pièce"), width_cm=17.2))
    story.append(Spacer(1, 0.35 * cm))
    story.append(fig_to_rl_image(fig_interaction(df), width_cm=17.2))
    story.append(Spacer(1, 0.35 * cm))
    story.append(fig_to_rl_image(fig_studyvar(res.study_var), width_cm=17.2))
    story.append(Spacer(1, 0.35 * cm))
    story.append(fig_to_rl_image(fig_contribution(res.var_components), width_cm=17.2))

    story.append(Spacer(1, 0.35 * cm))
    story.append(
        Paragraph(
            "<span fontSize=8 color='#6b7280'>Note: Calculs basés sur une étude ANOVA à plan équilibré. Study Variation = 6σ.</span>",
            styles["Normal"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def export_excel(df: pd.DataFrame, res: AnovaResult) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Donnees", index=False)
        res.study_var.to_excel(writer, sheet_name="StudyVar", index=False)
        res.var_components.to_excel(writer, sheet_name="Variance", index=False)
        res.anova_table.to_excel(writer, sheet_name="ANOVA", index=False)
        pd.DataFrame([res.metrics]).to_excel(writer, sheet_name="KPIs", index=False)
    return out.getvalue()


# -----------------------------
# UI Streamlit
# -----------------------------
st.title("📏 Calculateur Gage R&R (Cage R&R) — ANOVA")
st.caption("Saisie manuelle ou import (CSV/Excel) → calcul EV, AV, Vp, Vt + interprétation + rapport PDF.")

with st.sidebar:
    st.header("Paramètres")
    n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=10, step=1)
    n_ops = st.number_input("Nombre d'opérateurs", min_value=2, max_value=20, value=3, step=1)
    n_trials = st.number_input("Nombre de mesures (essais) par opérateur & pièce", min_value=2, max_value=10, value=3, step=1)

    st.divider()
    entry_mode = st.radio("Mode de saisie des données", ["Saisie manuelle", "Importer (CSV/Excel)"], horizontal=False)

    st.divider()
    tol = st.number_input("Tolérance (optionnel, pour %Tolérance)", min_value=0.0, value=0.0, step=0.1)

    st.divider()
    st.markdown(
        "<div class='small-note'>💡 <b>Formats acceptés</b>:<br>"
        "- <b>Format long</b>: colonnes Pièce, Opérateur, Essai, Mesure<br>"
        "- <b>Format large</b>: template avec opérateurs en colonnes (comme le fichier TEMPLATE CAGE RR.xlsx)</div>",
        unsafe_allow_html=True,
    )

tabs = st.tabs(["1) Données", "2) Résultats", "3) Graphes", "4) Rapport / Export"])

# --- Tab 1 : Données
with tabs[0]:
    st.subheader("Données d'entrée")

    shape = (int(n_parts), int(n_ops), int(n_trials))
    if "template_shape" not in st.session_state or st.session_state["template_shape"] != shape:
        st.session_state["template_shape"] = shape
        st.session_state["df_data"] = build_template(*shape)

    if entry_mode == "Saisie manuelle":
        st.markdown("Remplissez la colonne **Mesure** (valeurs numériques).")
        edited = st.data_editor(
            st.session_state["df_data"],
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key="editor_manual",
        )
        st.session_state["df_data"] = edited

        template_csv = build_template(*shape).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger un template CSV", data=template_csv, file_name="template_gage_rr.csv", mime="text/csv")

    else:
        st.markdown("Importez un fichier **CSV** ou **Excel**.")
        st.markdown("**Formats acceptés:**")
        st.markdown("- **Format long**: colonnes Pièce, Opérateur, Essai, Mesure")
        st.markdown("- **Format large (template)**: opérateurs en colonnes avec essais en sous-colonnes")
        
        upl = st.file_uploader("Importer", type=["csv", "xlsx"])

        if upl is not None:
            try:
                if upl.name.lower().endswith(".csv"):
                    df_imp = pd.read_csv(upl, header=None)
                else:
                    # Essayer de lire avec header=[0,1] pour les templates avec MultiIndex
                    try:
                        df_imp = pd.read_excel(upl, header=[0, 1])
                    except:
                        df_imp = pd.read_excel(upl, header=None)
                
                # Essayer de détecter le format et convertir
                try:
                    # Si le fichier a plus de 4 colonnes, c'est probablement un format large
                    if df_imp.shape[1] > 4:
                        df_imp = convert_wide_to_long(df_imp)
                        st.success("✅ Format large détecté et converti en format long")
                    else:
                        # Essayer de lire comme format long
                        df_imp = pd.read_excel(upl) if upl.name.lower().endswith(".xlsx") else pd.read_csv(upl)
                        df_imp = normalize_imported_columns(df_imp)
                        st.success("✅ Format long détecté")
                except Exception as conv_e:
                    st.error(f"Erreur de conversion du format: {conv_e}")
                    # Revenir à la lecture normale
                    if upl.name.lower().endswith(".csv"):
                        df_imp = pd.read_csv(upl)
                    else:
                        df_imp = pd.read_excel(upl)
                    df_imp = normalize_imported_columns(df_imp)
                
                st.session_state["df_data"] = df_imp
                st.rerun()
                
            except Exception as e:
                st.error(f"Impossible de lire le fichier: {e}")

        st.markdown("Aperçu (vous pouvez corriger/compléter si besoin) :")
        edited = st.data_editor(
            st.session_state["df_data"],
            use_container_width=True,
            hide_index=True,
            key="editor_import",
        )
        st.session_state["df_data"] = edited

        template_csv = build_template(*shape).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger un template CSV", data=template_csv, file_name="template_gage_rr.csv", mime="text/csv")

    st.divider()
    df_current = st.session_state["df_data"].copy()
    ok, errs = validate_dataset(df_current, int(n_parts), int(n_ops), int(n_trials))
    if ok:
        st.success("✅ Données valides (plan équilibré). Vous pouvez consulter les résultats.")
    else:
        st.error("❌ Données invalides :")
        for e in errs:
            st.write(f"- {e}")

# --- Calcul (si données valides)
res: Optional[AnovaResult] = None
df_for_calc: Optional[pd.DataFrame] = None
ok, errs = validate_dataset(st.session_state["df_data"], int(n_parts), int(n_ops), int(n_trials))
if ok:
    df_for_calc = st.session_state["df_data"].copy()
    try:
        res = gage_rr_anova(df_for_calc)
    except Exception as e:
        res = None
        with tabs[1]:
            st.error(f"Erreur de calcul: {e}")

# --- Tab 2 : Résultats
with tabs[1]:
    st.subheader("Résultats (EV, AV, Vp, Vt, %R&R)")
    if res is None:
        st.info("Renseignez/importe des données valides pour voir les résultats.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("EV", _fmt(res.metrics["EV"]), help="Repeatability (6σ)")
        c2.metric("AV", _fmt(res.metrics["AV"]), help="Reproducibility (6σ) incluant interaction")
        c3.metric("Vp (PV)", _fmt(res.metrics["PV"]), help="Part-to-Part (6σ)")
        c4.metric("Vt (TV)", _fmt(res.metrics["TV"]), help="Total Variation (6σ)")
        c5.metric("%R&R", f"{res.metrics['%GRR']:.2f}%", help="%Study Variation = GRR / TV")

        st.markdown(
            f"""
<div class="rr-card {res.conclusion['css']}">
  <h3>Interprétation (comme votre règle)</h3>
  <p><b>%R&amp;R = {res.metrics['%GRR']:.2f}%</b></p>
  <p>{res.conclusion['regle']}</p>
  <p><b>Niveau :</b> {res.conclusion['niveau']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>ndc :</b> {res.metrics['ndc']}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if tol and tol > 0:
            pct_tol = (res.metrics["GRR"] / tol) * 100
            st.write(f"**%Tolérance (GRR/Tol)** ≈ **{pct_tol:.2f}%**")

        st.divider()

        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("### %Study Variation (6σ)")
            st.dataframe(res.study_var, use_container_width=True, hide_index=True)

        with colB:
            st.markdown("### Table ANOVA")
            st.dataframe(res.anova_table, use_container_width=True, hide_index=True)

        st.markdown("### Composantes de variance")
        st.dataframe(res.var_components, use_container_width=True, hide_index=True)

# --- Tab 3 : Graphes
with tabs[2]:
    st.subheader("Graphes")
    if res is None or df_for_calc is None:
        st.info("Renseignez/importe des données valides pour voir les graphes.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(fig_boxplot(df_for_calc, by="Opérateur", title="Boxplot par opérateur"), clear_figure=True)
        with g2:
            st.pyplot(fig_boxplot(df_for_calc, by="Pièce", title="Boxplot par pièce"), clear_figure=True)

        g3, g4 = st.columns(2)
        with g3:
            st.pyplot(fig_interaction(df_for_calc), clear_figure=True)
        with g4:
            st.pyplot(fig_studyvar(res.study_var), clear_figure=True)

        st.pyplot(fig_contribution(res.var_components), clear_figure=True)

# --- Tab 4 : Rapport / Export
with tabs[3]:
    st.subheader("Rapport & exports")
    if res is None or df_for_calc is None:
        st.info("Renseignez/importe des données valides pour générer le rapport.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📄 Rapport PDF (avec graphes)")
            pdf_bytes = generate_pdf_report(df_for_calc, res, tol=(tol if tol > 0 else None))
            st.download_button(
                "⬇️ Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name="rapport_gage_rr.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown("<div class='small-note'>Le PDF inclut: paramètres, conclusion, tableaux ANOVA/variance, et graphes.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 📊 Export Excel (données + résultats)")
            xlsx_bytes = export_excel(df_for_calc, res)
            st.download_button(
                "⬇️ Télécharger Excel",
                data=xlsx_bytes,
                file_name="resultats_gage_rr.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### Export des données")
        st.download_button(
            "⬇️ Télécharger les données (CSV)",
            data=df_for_calc.to_csv(index=False).encode("utf-8"),
            file_name="donnees_gage_rr.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-note'>⚠️ Cette app suppose un plan <b>équilibré</b> (même nombre d'essais pour chaque couple Pièce×Opérateur). "
    "Pour des plans non équilibrés, il faut une approche ANOVA plus générale (modèle mixte).</div>",
    unsafe_allow_html=True,
)
