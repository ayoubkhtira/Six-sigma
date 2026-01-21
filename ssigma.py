import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import f
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    Construit un DataFrame long format à partir du template simple.
    Structure: première colonne = pièces, puis 3 colonnes par opérateur (3 mesures).
    """
    long_rows = []
    
    for idx, row in df.iterrows():
        part = int(row.iloc[0])
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
    Calcul Gage R&R par méthode ANOVA.
    Retourne: %R&R basé sur %Tolérance (5.15 * sigma / tolérance * 100)
    """
    grand_mean = df_long["Value"].mean()
    
    parts = df_long["Part"].unique()
    ops = df_long["Operator"].unique()
    p = len(parts)
    o = len(ops)
    r = df_long["Rep"].nunique()
    
    mean_p = df_long.groupby("Part")["Value"].mean()
    mean_o = df_long.groupby("Operator")["Value"].mean()
    mean_po = df_long.groupby(["Part", "Operator"])["Value"].mean()
    
    # Sommes de carrés
    ss_p = r * o * ((mean_p - grand_mean) ** 2).sum()
    ss_o = r * p * ((mean_o - grand_mean) ** 2).sum()
    ss_po = r * ((mean_po - mean_p.reindex(mean_po.index.get_level_values(0)).values
                  - mean_o.reindex(mean_po.index.get_level_values(1)).values
                  + grand_mean) ** 2).sum()
    ss_total = ((df_long["Value"] - grand_mean) ** 2).sum()
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
    
    # Composantes de variance
    var_repeat = ms_e
    var_op = max((ms_o - ms_po) / (p * r), 0)
    var_part = max((ms_p - ms_po) / (o * r), 0)
    var_interaction = max((ms_po - ms_e) / r, 0)
    
    var_grr = var_repeat + var_op + var_interaction
    var_total = var_grr + var_part
    
    # Écart-types
    sd_repeat = np.sqrt(var_repeat)
    sd_op = np.sqrt(var_op)
    sd_interaction = np.sqrt(var_interaction)
    sd_grr = np.sqrt(var_grr)
    sd_part = np.sqrt(var_part)
    sd_total = np.sqrt(var_total)
    
    # Calcul du %R&R basé sur la tolérance (méthode standard)
    # Tolérance = range des pièces ou 6 * sigma_part
    tolerance = 6 * sd_part
    study_var_grr = 5.15 * sd_grr
    pct_grr_tolerance = 100 * study_var_grr / tolerance if tolerance > 0 else np.nan
    
    # % par rapport à la variation totale
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
        "sd_repeat": sd_repeat,
        "sd_op": sd_op,
        "sd_grr": sd_grr,
        "sd_part": sd_part,
        "sd_total": sd_total,
        "pct_grr": pct_grr,
        "pct_grr_tolerance": pct_grr_tolerance,
        "pct_repeat": pct_repeat,
        "pct_op": pct_op,
        "pct_part": pct_part,
        "tolerance": tolerance,
        "study_var_grr": study_var_grr,
        "df_long": df_long,
        "df": {
            "Part": df_p,
            "Operator": df_o,
            "Part*Operator": df_po,
            "Repeatability": df_e
        },
        "ss": {
            "Part": ss_p,
            "Operator": ss_o,
            "Part*Operator": ss_po,
            "Repeatability": ss_e,
            "Total": ss_total
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


def generate_report(results):
    """Génère un rapport détaillé en format texte."""
    report = f"""
═══════════════════════════════════════════════════════════
               RAPPORT GAGE R&R - ANALYSE MSA
═══════════════════════════════════════════════════════════

1. STATISTIQUES DESCRIPTIVES
   • Moyenne générale : {results['grand_mean']:.4f}
   • Tolérance estimée : {results['tolerance']:.4f}
   
2. ANALYSE DE VARIANCE (ANOVA)
   Source               SS           DF        MS
   ──────────────────────────────────────────────────
   Pièce            {results['ss']['Part']:10.6f}    {results['df']['Part']:3d}   {results['ms']['Part']:10.6f}
   Opérateur        {results['ss']['Operator']:10.6f}    {results['df']['Operator']:3d}   {results['ms']['Operator']:10.6f}
   Part*Operator    {results['ss']['Part*Operator']:10.6f}    {results['df']['Part*Operator']:3d}   {results['ms']['Part*Operator']:10.6f}
   Répétabilité     {results['ss']['Repeatability']:10.6f}    {results['df']['Repeatability']:3d}   {results['ms']['Repeatability']:10.6f}
   Total            {results['ss']['Total']:10.6f}
   
3. COMPOSANTES DE VARIANCE
   Source                    Variance      Écart-type    %Total
   ─────────────────────────────────────────────────────────────
   Répétabilité          {results['var_repeat']:10.6f}   {results['sd_repeat']:10.6f}   {results['pct_repeat']:6.2f}%
   Reproductibilité      {results['var_op']:10.6f}   {results['sd_op']:10.6f}   {results['pct_op']:6.2f}%
   Interaction           {results['var_interaction']:10.6f}   {np.sqrt(results['var_interaction']):10.6f}   
   ─────────────────────────────────────────────────────────────
   Total Gage R&R        {results['var_grr']:10.6f}   {results['sd_grr']:10.6f}   {results['pct_grr']:6.2f}%
   Pièce                 {results['var_part']:10.6f}   {results['sd_part']:10.6f}   {results['pct_part']:6.2f}%
   ─────────────────────────────────────────────────────────────
   Variation totale      {results['var_total']:10.6f}   {results['sd_total']:10.6f}  100.00%

4. ÉVALUATION DU SYSTÈME DE MESURE
   • %R&R (variation totale) : {results['pct_grr']:.2f}%
   • %R&R (tolérance)        : {results['pct_grr_tolerance']:.2f}%
   • Variation d'étude       : {results['study_var_grr']:.4f}
   
5. INTERPRÉTATION
   {interpret_grr(results['pct_grr_tolerance'])[0]}
   
   Recommandations :
   {'✓ Système acceptable pour la production' if results['pct_grr_tolerance'] <= 10 else '⚠ Amélioration nécessaire' if results['pct_grr_tolerance'] <= 30 else '✗ Système non acceptable - action immédiate requise'}

═══════════════════════════════════════════════════════════
    """
    return report


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
            <span class="metric-badge">📊 Rapports inclus</span>
            <p style="color:#9ca3af;font-size:0.85rem;">
                Graphiques interactifs, tableaux ANOVA, et rapport détaillé téléchargeable.
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
    
    st.markdown("---")
    st.caption("📂 **Format du fichier Excel :**")
    st.caption("• Colonne 1 : N° pièce (1-10)")
    st.caption("• Colonnes 2-4 : Opérateur 1 (mesures 1-3)")
    st.caption("• Colonnes 5-7 : Opérateur 2 (mesures 1-3)")
    st.caption("• Colonnes 8-10 : Opérateur 3 (mesures 1-3)")

uploaded_file = st.file_uploader("📂 Importer le fichier Excel Gage R&R", type=["xlsx"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    try:
        df_long = build_long_from_template(raw_df, n_operators, n_parts, n_reps)
    except Exception as e:
        st.error(f"Erreur lors de la conversion du template : {e}")
        st.stop()

    results = gage_rr_anova(df_long, alpha=1 - alpha)
    pct_grr = results["pct_grr_tolerance"]  # Utiliser le % par rapport à la tolérance
    interp_text, interp_level = interpret_grr(pct_grr)

    pill_class = {
        "ok": "pill-ok",
        "mid": "pill-mid",
        "bad": "pill-bad"
    }[interp_level]

    st.markdown("### 📊 Résultats Gage R&R")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("%R&R (Tolérance)", f"{pct_grr:0.2f} %")
    with c2:
        st.metric("%R&R (Total)", f"{results['pct_grr']:0.2f} %")
    with c3:
        st.metric("%Pièce", f"{results['pct_part']:0.2f} %")
    with c4:
        st.metric("%Répétabilité", f"{results['pct_repeat']:0.2f} %")
    with c5:
        st.metric("%Reproductibilité", f"{results['pct_op']:0.2f} %")

    st.markdown(
        f"""
        <div class="card">
            <span class="pill {pill_class}">{interp_text}</span>
            <p style="color:#9ca3af;font-size:0.9rem;margin-top:0.6rem;">
                %R&amp;R = {pct_grr:0.2f} % (par rapport à la tolérance). 
                Un système &gt; 30 % est généralement considéré comme non acceptable.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========== GRAPHIQUES ==========
    
    st.markdown("---")
    st.markdown("### 📈 Graphiques d'analyse")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🥧 Contributions", 
        "📊 Par opérateur", 
        "🔄 Par pièce", 
        "📉 Contrôle",
        "🎯 Interaction"
    ])
    
    with tab1:
        # Diagramme en camembert des contributions
        col1, col2 = st.columns(2)
        
        with col1:
            contrib_df = pd.DataFrame({
                "Source": ["Gage R&R", "Pièce"],
                "Pourcentage": [results["pct_grr"], results["pct_part"]]
            })
            fig1 = px.pie(
                contrib_df,
                values="Pourcentage",
                names="Source",
                title="Contribution à la variation totale",
                color="Source",
                color_discrete_map={"Gage R&R": "#f97316", "Pièce": "#38bdf8"},
                hole=0.4
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Décomposition du Gage R&R
            grr_detail_df = pd.DataFrame({
                "Composante": ["Répétabilité", "Reproductibilité"],
                "Pourcentage": [results["pct_repeat"], results["pct_op"]]
            })
            fig2 = px.bar(
                grr_detail_df,
                x="Composante",
                y="Pourcentage",
                title="Décomposition du Gage R&R",
                color="Composante",
                color_discrete_map={
                    "Répétabilité": "#a855f7",
                    "Reproductibilité": "#fbbf24"
                }
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Graphique par opérateur
        op_stats = df_long.groupby("Operator")["Value"].agg(['mean', 'std']).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.box(
                df_long,
                x="Operator",
                y="Value",
                title="Distribution des mesures par opérateur",
                color="Operator",
                color_discrete_sequence=["#38bdf8", "#a855f7", "#f97316"]
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Opérateur",
                yaxis_title="Valeur"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=op_stats['Operator'],
                y=op_stats['mean'],
                name='Moyenne',
                marker_color='#38bdf8',
                error_y=dict(type='data', array=op_stats['std'])
            ))
            fig4.update_layout(
                title="Moyennes par opérateur (± écart-type)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Opérateur",
                yaxis_title="Valeur moyenne"
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        # Graphique par pièce
        part_stats = df_long.groupby("Part")["Value"].agg(['mean', 'std']).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig5 = px.line(
                part_stats,
                x="Part",
                y="mean",
                title="Moyenne des mesures par pièce",
                markers=True,
                color_discrete_sequence=["#a855f7"]
            )
            fig5.add_scatter(
                x=part_stats['Part'],
                y=part_stats['mean'] + part_stats['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
            fig5.add_scatter(
                x=part_stats['Part'],
                y=part_stats['mean'] - part_stats['std'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(168, 85, 247, 0.2)',
                fill='tonexty',
                showlegend=False
            )
            fig5.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Pièce",
                yaxis_title="Valeur"
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            fig6 = px.box(
                df_long,
                x="Part",
                y="Value",
                title="Distribution par pièce",
                color_discrete_sequence=["#38bdf8"]
            )
            fig6.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Pièce",
                yaxis_title="Valeur"
            )
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        # Cartes de contrôle
        mean_by_part_op = df_long.groupby(['Part', 'Operator'])['Value'].mean().reset_index()
        
        fig7 = px.line(
            mean_by_part_op,
            x='Part',
            y='Value',
            color='Operator',
            title="Carte de contrôle - Moyennes par pièce et opérateur",
            markers=True,
            color_discrete_sequence=["#38bdf8", "#a855f7", "#f97316"]
        )
        fig7.add_hline(
            y=results['grand_mean'],
            line_dash="dash",
            line_color="#6ee7b7",
            annotation_text="Moyenne générale"
        )
        fig7.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Pièce",
            yaxis_title="Valeur moyenne"
        )
        st.plotly_chart(fig7, use_container_width=True)
        
        # Graphique des étendues
        range_by_part_op = df_long.groupby(['Part', 'Operator'])['Value'].apply(lambda x: x.max() - x.min()).reset_index()
        range_by_part_op.columns = ['Part', 'Operator', 'Range']
        
        fig8 = px.line(
            range_by_part_op,
            x='Part',
            y='Range',
            color='Operator',
            title="Étendue (Range) par pièce et opérateur",
            markers=True,
            color_discrete_sequence=["#38bdf8", "#a855f7", "#f97316"]
        )
        fig8.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Pièce",
            yaxis_title="Étendue"
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    with tab5:
        # Graphique d'interaction Pièce × Opérateur
        interaction_data = df_long.groupby(['Part', 'Operator'])['Value'].mean().reset_index()
        
        fig9 = px.line(
            interaction_data,
            x='Part',
            y='Value',
            color='Operator',
            title="Graphique d'interaction Pièce × Opérateur",
            markers=True,
            color_discrete_sequence=["#38bdf8", "#a855f7", "#f97316"]
        )
        fig9.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Pièce",
            yaxis_title="Valeur moyenne"
        )
        st.plotly_chart(fig9, use_container_width=True)
        
        st.info("💡 Des lignes parallèles indiquent une absence d'interaction. Des lignes qui se croisent suggèrent une interaction Pièce × Opérateur.")

    # ========== TABLEAUX DÉTAILLÉS ==========
    
    st.markdown("---")
    st.markdown("### 📋 Tableaux détaillés")
    
    tab_t1, tab_t2, tab_t3 = st.tabs(["ANOVA", "Composantes de variance", "Données brutes"])
    
    with tab_t1:
        anova_table = pd.DataFrame({
            "Source": ["Pièce", "Opérateur", "Pièce × Opérateur", "Répétabilité", "Total"],
            "SS": [
                results["ss"]["Part"],
                results["ss"]["Operator"],
                results["ss"]["Part*Operator"],
                results["ss"]["Repeatability"],
                results["ss"]["Total"]
            ],
            "DF": [
                results["df"]["Part"],
                results["df"]["Operator"],
                results["df"]["Part*Operator"],
                results["df"]["Repeatability"],
                "-"
            ],
            "MS": [
                results["ms"]["Part"],
                results["ms"]["Operator"],
                results["ms"]["Part*Operator"],
                results["ms"]["Repeatability"],
                "-"
            ]
        })
        st.dataframe(
            anova_table.style.format({
                "SS": "{:0.6f}",
                "MS": "{:0.6f}"
            }),
            use_container_width=True
        )
    
    with tab_t2:
        var_table = pd.DataFrame({
            "Source": [
                "Répétabilité (équipement)",
                "Reproductibilité (opérateur)",
                "Interaction Pièce × Opérateur",
                "Total Gage R&R",
                "Variation pièce à pièce",
                "Variation totale"
            ],
            "Variance": [
                results["var_repeat"],
                results["var_op"],
                results["var_interaction"],
                results["var_grr"],
                results["var_part"],
                results["var_total"],
            ],
            "Écart-type": [
                results["sd_repeat"],
                results["sd_op"],
                np.sqrt(results["var_interaction"]),
                results["sd_grr"],
                results["sd_part"],
                results["sd_total"],
            ],
            "% Total": [
                results["pct_repeat"],
                results["pct_op"],
                "-",
                results["pct_grr"],
                results["pct_part"],
                100.0
            ]
        })
        st.dataframe(
            var_table.style.format({
                "Variance": "{:0.6f}",
                "Écart-type": "{:0.6f}",
                "% Total": "{:0.2f}%"
            }),
            use_container_width=True
        )
    
    with tab_t3:
        st.dataframe(df_long, use_container_width=True)

    # ========== RAPPORT TÉLÉCHARGEABLE ==========
    
    st.markdown("---")
    st.markdown("### 📄 Rapport détaillé")
    
    report_text = generate_report(results)
    st.text_area("Rapport Gage R&R", report_text, height=400)
    
    # Bouton de téléchargement
    st.download_button(
        label="⬇️ Télécharger le rapport (TXT)",
        data=report_text,
        file_name="rapport_gage_rr.txt",
        mime="text/plain"
    )
    
    # Export Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_long.to_excel(writer, sheet_name='Données', index=False)
        anova_table.to_excel(writer, sheet_name='ANOVA', index=False)
        var_table.to_excel(writer, sheet_name='Composantes', index=False)
    
    st.download_button(
        label="⬇️ Télécharger les tableaux (Excel)",
        data=output.getvalue(),
        file_name="analyse_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("📂 Chargez votre fichier TEMPLATE-CAGE-RR.xlsx pour lancer l'analyse complète.")
    st.markdown("""
    ### 📖 Guide d'utilisation
    
    1. **Préparez votre fichier Excel** avec la structure suivante :
       - Colonne 1 : Numéro de pièce (1 à 10)
       - Colonnes 2-4 : 3 mesures de l'opérateur 1
       - Colonnes 5-7 : 3 mesures de l'opérateur 2
       - Colonnes 8-10 : 3 mesures de l'opérateur 3
    
    2. **Importez le fichier** via le bouton ci-dessus
    
    3. **Consultez les résultats** :
       - Métriques principales (%R&R, %Pièce, etc.)
       - 5 types de graphiques interactifs
       - Tableaux ANOVA détaillés
       - Rapport téléchargeable
    
    4. **Interprétation** :
       - **%R&R ≤ 10%** : Système acceptable ✅
       - **10% < %R&R ≤ 30%** : Système marginal ⚠️
       - **%R&R > 30%** : Système non acceptable ❌
    """)
