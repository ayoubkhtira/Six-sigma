import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import f
import plotly.express as px

# ---------- CONFIG & STYLE ----------
st.set_page_config(
    page_title="Gage R&R Pro",
    page_icon="📏",
    layout="wide",
)

CUSTOM_CSS = """
<style>
body {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #e5e7eb;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont;
}
header, .css-18ni7ap, .css-1avcm0n, .css-1d391kg {
    background: transparent !important;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background: rgba(15,23,42,0.9);
    border-radius: 1rem;
    padding: 1.2rem 1.5rem;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 24px 60px rgba(15,23,42,0.9);
    backdrop-filter: blur(12px);
}
.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: .04em;
    text-transform: uppercase;
}
.pill-ok    { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
.pill-mid   { background: rgba(234,179,8,0.15);  color: #fde68a; border: 1px solid rgba(234,179,8,0.4); }
.pill-bad   { background: rgba(248,113,113,0.15);color: #fecaca; border: 1px solid rgba(248,113,113,0.4); }
.gradient-title {
    background: linear-gradient(90deg,#38bdf8,#a855f7,#f97316);
    -webkit-background-clip: text;
    color: transparent;
}
.metric-badge {
    font-size: 0.8rem;
    opacity: 0.9;
}
.glow {
    animation: pulseGlow 2.4s ease-in-out infinite;
}
@keyframes pulseGlow {
  0% { box-shadow: 0 0 0 rgba(56,189,248,0.0); }
  50% { box-shadow: 0 0 24px rgba(56,189,248,0.4); }
  100% { box-shadow: 0 0 0 rgba(56,189,248,0.0); }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- FUNCTIONS GAGE R&R (ANOVA) ----------

def build_long_from_template(df, n_operators, n_parts, n_reps):
    """
    Construit un DataFrame long:
    Colonnes attendues (exemple template):
    [N° de la pièce,
     OPERATEUR 1 1, OPERATEUR 1 2, OPERATEUR 1 3,
     OPERATEUR 2 1, OPERATEUR 2 2, OPERATEUR 2 3, ...]
    Première ligne: labels opérateur / répète.
    Deuxième ligne: labels Mesure 1..3 (à ignorer pour les valeurs).
    """
    # Les 2 premières lignes sont des entêtes "complexes" dans votre fichier
    header1 = df.iloc[0].tolist()
    header2 = df.iloc[1].tolist()

    data = df.iloc[2:, :]

    long_rows = []
    part_col_name = df.columns[0]

    for _, row in data.iterrows():
        part = int(row[part_col_name])
        for op_idx in range(n_operators):
            for rep_idx in range(n_reps):
                col_index = 1 + op_idx * n_reps + rep_idx
                value = float(row.iloc[col_index])
                long_rows.append({
                    "Part": part,
                    "Operator": op_idx + 1,
                    "Rep": rep_idx + 1,
                    "Value": value
                })
    return pd.DataFrame(long_rows)


def gage_rr_anova(df_long, alpha=0.05):
    """
    df_long: colonnes Part, Operator, Rep, Value
    Méthode ANOVA à 2 facteurs pour Gage R&R.
    """
    # Moyennes globales
    grand_mean = df_long["Value"].mean()

    # Sets
    parts = df_long["Part"].unique()
    ops = df_long["Operator"].unique()
    p = len(parts)
    o = len(ops)
    r = df_long["Rep"].nunique()

    # Moyennes par part, opérateur, combinaison part-op, et globale
    mean_p = df_long.groupby("Part")["Value"].mean()
    mean_o = df_long.groupby("Operator")["Value"].mean()
    mean_po = df_long.groupby(["Part", "Operator"])["Value"].mean()

    # Sommes de carrés
    # Part
    ss_p = r * o * ((mean_p - grand_mean) ** 2).sum()
    # Operator
    ss_o = r * p * ((mean_o - grand_mean) ** 2).sum()
    # Part*Operator
    ss_po = r * ((mean_po - mean_p.reindex(mean_po.index.get_level_values(0)).values
                  - mean_o.reindex(mean_po.index.get_level_values(1)).values
                  + grand_mean) ** 2).sum()
    # Total
    ss_total = ((df_long["Value"] - grand_mean) ** 2).sum()
    # Repeatability (erreur)
    ss_e = ss_total - ss_p - ss_o - ss_po

    # Degrés de liberté
    df_p = p - 1
    df_o = o - 1
    df_po = (p - 1) * (o - 1)
    df_e = p * o * (r - 1)

    # Carrés moyens
    ms_p = ss_p / df_p if df_p > 0 else np.nan
    ms_o = ss_o / df_o if df_o > 0 else np.nan
    ms_po = ss_po / df_po if df_po > 0 else np.nan
    ms_e = ss_e / df_e if df_e > 0 else np.nan

    # Variances composantes (classique Gage R&R)
    var_repeat = ms_e
    var_op = max((ms_o - ms_po) / (p * r), 0)
    var_part = max((ms_p - ms_po) / (o * r), 0)
    var_interaction = max((ms_po - ms_e) / r, 0)

    # Total Gage R&R
    var_grr = var_repeat + var_op + var_interaction
    var_total = var_grr + var_part

    # Écart-types
    sd_repeat = np.sqrt(var_repeat)
    sd_op = np.sqrt(var_op)
    sd_interaction = np.sqrt(var_interaction)
    sd_grr = np.sqrt(var_grr)
    sd_part = np.sqrt(var_part)
    sd_total = np.sqrt(var_total)

    # % contribution
    pct_grr = 100 * sd_grr / sd_total if sd_total > 0 else np.nan
    pct_repeat = 100 * sd_repeat / sd_total if sd_total > 0 else np.nan
    pct_op = 100 * sd_op / sd_total if sd_total > 0 else np.nan
    pct_part = 100 * sd_part / sd_total if sd_total > 0 else np.nan

    return {
        "grand_mean": grand_mean,
        "var_repeat": var_repeat,
        "var_op": var_op,
        "var_part": var_part,
        "var_interaction": var_interaction,
        "var_grr": var_grr,
        "var_total": var_total,
        "sd_grr": sd_grr,
        "sd_part": sd_part,
        "sd_total": sd_total,
        "pct_grr": pct_grr,
        "pct_repeat": pct_repeat,
        "pct_op": pct_op,
        "pct_part": pct_part,
        "df": {
            "Part": df_p,
            "Operator": df_o,
            "Part*Operator": df_po,
            "Repeatability": df_e
        },
        "ms": {
            "Part": ms_p,
            "Operator": ms_o,
            "Part*Operator": ms_po,
            "Repeatability": ms_e
        }
    }


def interpret_grr(pct_grr):
    if pct_grr <= 10:
        return "Système de mesure acceptable (≤ 10 %).", "ok"
    elif pct_grr <= 30:
        return "Système marginal (10–30 %), amélioration recommandée.", "mid"
    else:
        return "Système de mesure non acceptable (> 30 %).", "bad"


# ---------- UI ----------

left, right = st.columns([1.1, 1])

with left:
    st.markdown(
        """
        <div class="card glow">
            <div class="pill pill-mid">📏 Gage R&amp;R • MSA</div>
            <h1 class="gradient-title" style="margin-top:0.7rem;margin-bottom:0.3rem;">
                Plateforme Gage R&amp;R Pro
            </h1>
            <p style="color:#9ca3af;font-size:0.9rem;margin-bottom:0.4rem;">
                Analyse ANOVA, %R&amp;R et interprétation automatique de votre système de mesure.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        """
        <div class="card">
            <span class="metric-badge">Astuce</span>
            <p style="color:#9ca3af;font-size:0.85rem;">
                Utilisez votre template Excel CAGE R&amp;R : 10 pièces, 3 opérateurs, 3 mesures par pièce et opérateur.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    n_operators = st.number_input("Nombre d'opérateurs", min_value=2, max_value=10, value=3, step=1)
    n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=10, step=1)
    n_reps = st.number_input("Nombre de mesures (répétitions)", min_value=2, max_value=10, value=3, step=1)
    alpha = st.slider("Niveau de confiance (1 - α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    st.caption("Format attendu du fichier :")
    st.caption("- Ligne 1 : opérateurs / mesures (comme votre template).")
    st.caption("- Ligne 2 : étiquettes MESURE 1, MESURE 2, MESURE 3…")
    st.caption("- Lignes suivantes : valeurs numériques.")

uploaded_file = st.file_uploader("📂 Importer le fichier Excel Gage R&R", type=["xlsx"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file, header=None)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    try:
        df_long = build_long_from_template(raw_df, n_operators, n_parts, n_reps)
    except Exception as e:
        st.error(f"Erreur lors de la conversion du template : {e}")
        st.stop()

    results = gage_rr_anova(df_long, alpha=1 - alpha)
    pct_grr = results["pct_grr"]
    interp_text, interp_level = interpret_grr(pct_grr)

    pill_class = {
        "ok": "pill-ok",
        "mid": "pill-mid",
        "bad": "pill-bad"
    }[interp_level]

    st.markdown("### Résultats Gage R&R")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("%R&R total", f"{pct_grr:0.2f} %")
    with c2:
        st.metric("%Pièce", f"{results['pct_part']:0.2f} %")
    with c3:
        st.metric("%Répétabilité", f"{results['pct_repeat']:0.2f} %")
    with c4:
        st.metric("%Reproductibilité", f"{results['pct_op']:0.2f} %")

    st.markdown(
        f"""
        <div class="card">
            <span class="pill {pill_class}">{interp_text}</span>
            <p style="color:#9ca3af;font-size:0.9rem;margin-top:0.6rem;">
                %R&amp;R = {pct_grr:0.2f} %. Un système &gt; 30 % est généralement considéré comme non acceptable pour la plupart des applications.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Petite visualisation moderne
    st.markdown("### Contributions aux variations")
    contrib_df = pd.DataFrame({
        "Source": ["Gage R&R", "Pièce"],
        "Pourcentage": [results["pct_grr"], results["pct_part"]]
    })
    fig = px.pie(
        contrib_df,
        values="Pourcentage",
        names="Source",
        color="Source",
        color_discrete_map={"Gage R&R": "#38bdf8", "Pièce": "#a855f7"},
        hole=0.4
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table des variances
    st.markdown("### Variances et écart-types")
    var_table = pd.DataFrame({
        "Source": ["Répétabilité", "Reproductibilité opérateur", "Interaction P×O", "Total Gage R&R", "Pièce", "Total"],
        "Variance": [
            results["var_repeat"],
            results["var_op"],
            results["var_interaction"],
            results["var_grr"],
            results["var_part"],
            results["var_total"],
        ],
        "Écart-type": [
            np.sqrt(results["var_repeat"]),
            np.sqrt(results["var_op"]),
            np.sqrt(results["var_interaction"]),
            results["sd_grr"],
            results["sd_part"],
            results["sd_total"],
        ]
    })
    st.dataframe(var_table.style.format({"Variance": "{:0.6f}", "Écart-type": "{:0.6f}"}), use_container_width=True)

else:
    st.info("Chargez votre fichier TEMPLATE-CAGE-RR.xlsx pour lancer le calcul.")
