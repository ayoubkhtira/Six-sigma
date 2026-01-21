import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
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
    Construit un DataFrame long format à partir du template.
    Structure: Colonne A = pièces
              Colonnes B,C,D = Opérateur 1 (essais 1,2,3)
              Colonnes E,F,G = Opérateur 2 (essais 1,2,3)
              Colonnes H,I,J = Opérateur 3 (essais 1,2,3)
    """
    long_rows = []
    
    for idx, row in df.iterrows():
        part = int(row.iloc[0])
        for op_idx in range(n_operators):
            for rep_idx in range(n_reps):
                col_index = 1 + op_idx * n_reps + rep_idx
                # Vérifier si l'index de colonne existe
                if col_index < len(row):
                    value = float(row.iloc[col_index])
                    long_rows.append({
                        "Part": part,
                        "Operator": op_idx + 1,
                        "Rep": rep_idx + 1,
                        "Value": value
                    })
                else:
                    st.error(f"Colonne manquante dans le fichier. Vérifiez le nombre d'opérateurs et de répétitions.")
                    return None
    return pd.DataFrame(long_rows)


def gage_rr_anova(df_long, alpha=0.05, confidence_coefficient=5.15):
    """
    Calcul Gage R&R par méthode ANOVA.
    Utilise le facteur de coefficient de confiance pour la variation d'étude.
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
    ms_p = ss_p / df_p if df_p > 0 else 0
    ms_o = ss_o / df_o if df_o > 0 else 0
    ms_po = ss_po / df_po if df_po > 0 else 0
    ms_e = ss_e / df_e if df_e > 0 else 0
    
    # Composantes de variance (éviter les valeurs négatives)
    var_repeat = max(ms_e, 0)
    var_op = max((ms_o - ms_po) / (p * r), 0) if ms_o > ms_po else 0
    var_part = max((ms_p - ms_po) / (o * r), 0) if ms_p > ms_po else 0
    var_interaction = max((ms_po - ms_e) / r, 0) if ms_po > ms_e else 0
    
    var_grr = var_repeat + var_op + var_interaction
    var_total = var_grr + var_part
    
    # Écart-types (sigma)
    sd_repeat = np.sqrt(var_repeat) if var_repeat > 0 else 0
    sd_op = np.sqrt(var_op) if var_op > 0 else 0
    sd_interaction = np.sqrt(var_interaction) if var_interaction > 0 else 0
    sd_grr = np.sqrt(var_grr) if var_grr > 0 else 0
    sd_part = np.sqrt(var_part) if var_part > 0 else 0
    sd_total = np.sqrt(var_total) if var_total > 0 else 0
    
    # FACTEUR de coefficient de confiance pour la variation d'étude
    FACTOR = confidence_coefficient
    
    # Variation d'étude (Study Variation = coefficient × sigma)
    EV = sd_repeat * FACTOR  # Répétabilité
    AV = sd_op * FACTOR  # Reproductibilité
    RR = sd_grr * FACTOR  # Total Gage R&R
    Vp = sd_part * FACTOR  # Variation pièce
    VT = sd_total * FACTOR  # Variation totale
    
    # Pourcentages par rapport à VT
    pct_grr = 100 * RR / VT if VT > 0 else 0
    pct_repeat = 100 * EV / VT if VT > 0 else 0
    pct_op = 100 * AV / VT if VT > 0 else 0
    pct_part = 100 * Vp / VT if VT > 0 else 0
    
    # Pourcentages par rapport aux variances (pour compatibilité)
    pct_grr_var = 100 * sd_grr / sd_total if sd_total > 0 else 0
    pct_repeat_var = 100 * sd_repeat / sd_total if sd_total > 0 else 0
    pct_op_var = 100 * sd_op / sd_total if sd_total > 0 else 0
    pct_part_var = 100 * sd_part / sd_total if sd_total > 0 else 0
    
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
        # Valeurs avec coefficient de confiance
        "EV": EV,
        "AV": AV,
        "RR": RR,
        "Vp": Vp,
        "VT": VT,
        "pct_grr": pct_grr,
        "pct_repeat": pct_repeat,
        "pct_op": pct_op,
        "pct_part": pct_part,
        # Pourcentages par variance (sigma)
        "pct_grr_var": pct_grr_var,
        "pct_repeat_var": pct_repeat_var,
        "pct_op_var": pct_op_var,
        "pct_part_var": pct_part_var,
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
        },
        "confidence_coefficient": FACTOR
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
    confidence_coefficient = results.get('confidence_coefficient', 5.15)
    report = f"""
═══════════════════════════════════════════════════════════════════
                 RAPPORT GAGE R&R - ANALYSE MSA
═══════════════════════════════════════════════════════════════════

1. STATISTIQUES DESCRIPTIVES
   • Moyenne générale : {results['grand_mean']:.4f}
   
2. ANALYSE DE VARIANCE (ANOVA)
   
   Source               SS           DF        MS
   ──────────────────────────────────────────────────────────
   Pièce            {results['ss']['Part']:10.6f}    {results['df']['Part']:3d}   {results['ms']['Part']:10.6f}
   Opérateur        {results['ss']['Operator']:10.6f}    {results['df']['Operator']:3d}   {results['ms']['Operator']:10.6f}
   Part*Operator    {results['ss']['Part*Operator']:10.6f}    {results['df']['Part*Operator']:3d}   {results['ms']['Part*Operator']:10.6f}
   Répétabilité     {results['ss']['Repeatability']:10.6f}    {results['df']['Repeatability']:3d}   {results['ms']['Repeatability']:10.6f}
   Total            {results['ss']['Total']:10.6f}
   
3. COMPOSANTES DE VARIANCE (Sigma)
   
   Source                    Variance      Écart-type    %Contribution
   ────────────────────────────────────────────────────────────────────
   Répétabilité          {results['var_repeat']:10.6f}   {results['sd_repeat']:10.6f}      {results['pct_repeat_var']:6.2f}%
   Reproductibilité      {results['var_op']:10.6f}   {results['sd_op']:10.6f}      {results['pct_op_var']:6.2f}%
   Interaction           {results['var_interaction']:10.6f}   {np.sqrt(results['var_interaction']):10.6f}
   ────────────────────────────────────────────────────────────────────
   Total Gage R&R        {results['var_grr']:10.6f}   {results['sd_grr']:10.6f}      {results['pct_grr_var']:6.2f}%
   Pièce                 {results['var_part']:10.6f}   {results['sd_part']:10.6f}      {results['pct_part_var']:6.2f}%
   ────────────────────────────────────────────────────────────────────
   Variation totale      {results['var_total']:10.6f}   {results['sd_total']:10.6f}     100.00%

4. VARIATION D'ÉTUDE (Study Variation = {confidence_coefficient} × Sigma)
   
   Composante                    Valeur       %SV
   ──────────────────────────────────────────────────
   EV (Répétabilité)            {results['EV']:8.3f}     {results['pct_repeat']:6.2f}%
   AV (Reproductibilité)        {results['AV']:8.3f}     {results['pct_op']:6.2f}%
   R&R (Total Gage R&R)         {results['RR']:8.3f}     {results['pct_grr']:6.2f}%
   Vp (Variation pièce)         {results['Vp']:8.3f}     {results['pct_part']:6.2f}%
   VT (Variation totale)        {results['VT']:8.3f}    100.00%

5. ÉVALUATION DU SYSTÈME DE MESURE
   
   • %R&R (Study Variation) : {results['pct_grr']:.2f}%
   • %R&R (Sigma)           : {results['pct_grr_var']:.2f}%
   
6. INTERPRÉTATION
   
   {interpret_grr(results['pct_grr'])[0]}
   
   Recommandations :
   {'✓ Système acceptable pour la production' if results['pct_grr'] <= 10 else '⚠ Amélioration nécessaire' if results['pct_grr'] <= 30 else '✗ Système non acceptable - action immédiate requise'}

═══════════════════════════════════════════════════════════════════
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
            <span class="metric-badge">📊 Calcul exact</span>
            <p style="color:#9ca3af;font-size:0.85rem;">
                Méthode ANOVA avec variation d'étude (5.15 × σ par défaut). Résultats conformes aux normes MSA.
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
    st.header("📐 Coefficient de confiance")
    confidence_coefficient = st.number_input(
        "Coefficient pour la variation d'étude",
        min_value=1.0,
        max_value=10.0,
        value=5.15,
        step=0.01,
        help="Facteur multiplicateur pour convertir sigma en variation d'étude. Par défaut: 5.15 (99% de la population)"
    )
    
    st.markdown("---")
    st.caption("📂 **Format du fichier Excel :**")
    st.caption("• Colonne A : N° pièce (1-10)")
    st.caption("• Colonnes B-D : Opérateur 1 (essais 1-3)")
    st.caption("• Colonnes E-G : Opérateur 2 (essais 1-3)")
    st.caption("• Colonnes H-J : Opérateur 3 (essais 1-3)")
    st.caption("• Cellule B2 = Essai 1, Op 1, Pièce 1")

uploaded_file = st.file_uploader("📂 Importer le fichier Excel Gage R&R", type=["xlsx"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    try:
        df_long = build_long_from_template(raw_df, n_operators, n_parts, n_reps)
        if df_long is None:
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la conversion du template : {e}")
        st.stop()

    results = gage_rr_anova(df_long, alpha=1 - alpha, confidence_coefficient=confidence_coefficient)
    pct_grr = results["pct_grr"]
    interp_text, interp_level = interpret_grr(pct_grr)

    pill_class = {
        "ok": "pill-ok",
        "mid": "pill-mid",
        "bad": "pill-bad"
    }[interp_level]

    st.markdown(f"### 📊 Résultats Gage R&R - Variation d'étude ({confidence_coefficient} × σ)")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("R&R", f"{results['RR']:.3f}")
    with c2:
        st.metric("EV (Répétabilité)", f"{results['EV']:.3f}")
    with c3:
        st.metric("AV (Reproductibilité)", f"{results['AV']:.3f}")
    with c4:
        st.metric("Vp (Variation pièce)", f"{results['Vp']:.3f}")
    with c5:
        st.metric("VT (Variation totale)", f"{results['VT']:.3f}")

    st.markdown("### 📈 Pourcentages (%SV)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("%R&R", f"{pct_grr:.2f} %")
    with c2:
        st.metric("%EV", f"{results['pct_repeat']:.2f} %")
    with c3:
        st.metric("%AV", f"{results['pct_op']:.2f} %")
    with c4:
        st.metric("%Vp", f"{results['pct_part']:.2f} %")

    st.markdown(
        f"""
        <div class="card">
            <span class="pill {pill_class}">{interp_text}</span>
            <p style="color:#9ca3af;font-size:0.9rem;margin-top:0.6rem;">
                %R&amp;R = {pct_grr:.2f} % (variation d'étude). 
                Un système &gt; 30 % est généralement considéré comme non acceptable.
                Coefficient de confiance utilisé : {confidence_coefficient}
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
            grr_detail_df = pd.DataFrame({
                "Composante": ["Répétabilité (EV)", "Reproductibilité (AV)"],
                "Valeur": [results["EV"], results["AV"]]
            })
            fig2 = px.bar(
                grr_detail_df,
                x="Composante",
                y="Valeur",
                title="Décomposition du Gage R&R",
                color="Composante",
                color_discrete_map={
                    "Répétabilité (EV)": "#a855f7",
                    "Reproductibilité (AV)": "#fbbf24"
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
    
    tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs(["ANOVA", "Variance (Sigma)", f"Variation d'étude ({confidence_coefficient}σ)", "Données brutes"])
    
    with tab_t1:
        anova_table = pd.DataFrame({
            "Source": ["Pièce", "Opérateur", "Pièce × Opérateur", "Répétabilité", "Total"],
            "SS": [
                f"{results['ss']['Part']:.6f}",
                f"{results['ss']['Operator']:.6f}",
                f"{results['ss']['Part*Operator']:.6f}",
                f"{results['ss']['Repeatability']:.6f}",
                f"{results['ss']['Total']:.6f}"
            ],
            "DF": [
                str(results["df"]["Part"]),
                str(results["df"]["Operator"]),
                str(results["df"]["Part*Operator"]),
                str(results["df"]["Repeatability"]),
                "-"
            ],
            "MS": [
                f"{results['ms']['Part']:.6f}",
                f"{results['ms']['Operator']:.6f}",
                f"{results['ms']['Part*Operator']:.6f}",
                f"{results['ms']['Repeatability']:.6f}",
                "-"
            ]
        })
        st.dataframe(anova_table, use_container_width=True)
    
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
                f"{results['var_repeat']:.6f}",
                f"{results['var_op']:.6f}",
                f"{results['var_interaction']:.6f}",
                f"{results['var_grr']:.6f}",
                f"{results['var_part']:.6f}",
                f"{results['var_total']:.6f}",
            ],
            "Écart-type (σ)": [
                f"{results['sd_repeat']:.6f}",
                f"{results['sd_op']:.6f}",
                f"{np.sqrt(results['var_interaction']):.6f}",
                f"{results['sd_grr']:.6f}",
                f"{results['sd_part']:.6f}",
                f"{results['sd_total']:.6f}",
            ],
            "% Contribution": [
                f"{results['pct_repeat_var']:.2f}%",
                f"{results['pct_op_var']:.2f}%",
                "-",
                f"{results['pct_grr_var']:.2f}%",
                f"{results['pct_part_var']:.2f}%",
                "100.00%"
            ]
        })
        st.dataframe(var_table, use_container_width=True)
    
    with tab_t3:
        sv_table = pd.DataFrame({
            "Composante": [
                "EV - Répétabilité",
                "AV - Reproductibilité",
                "R&R - Total Gage R&R",
                "Vp - Variation pièce",
                "VT - Variation totale"
            ],
            f"Valeur ({confidence_coefficient} × σ)": [
                f"{results['EV']:.3f}",
                f"{results['AV']:.3f}",
                f"{results['RR']:.3f}",
                f"{results['Vp']:.3f}",
                f"{results['VT']:.3f}"
            ],
            "%SV": [
                f"{results['pct_repeat']:.2f}%",
                f"{results['pct_op']:.2f}%",
                f"{results['pct_grr']:.2f}%",
                f"{results['pct_part']:.2f}%",
                "100.00%"
            ]
        })
        st.dataframe(sv_table, use_container_width=True)
        
        st.info(f"📌 **Variation d'étude (Study Variation)** = {confidence_coefficient} × écart-type (σ). Représente environ 99% de la variation du processus (pour 5.15).")
    
    with tab_t4:
        st.dataframe(df_long, use_container_width=True)

    # ========== RAPPORT TÉLÉCHARGEABLE ==========
    
    st.markdown("---")
    st.markdown("### 📄 Rapport détaillé")
    
    report_text = generate_report(results)
    st.text_area("Rapport Gage R&R", report_text, height=500)
    
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
        var_table.to_excel(writer, sheet_name='Variance', index=False)
        sv_table.to_excel(writer, sheet_name='Variation étude', index=False)
    
    st.download_button(
        label="⬇️ Télécharger les tableaux (Excel)",
        data=output.getvalue(),
        file_name="analyse_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("📂 Chargez votre fichier Excel pour lancer l'analyse complète.")
    st.markdown("""
    ### 📖 Structure attendue du fichier
    
    | Colonne | Contenu |
    |---------|---------|
    | A | Numéro de pièce (1 à 10) |
    | B | Opérateur 1 - Essai 1 |
    | C | Opérateur 1 - Essai 2 |
    | D | Opérateur 1 - Essai 3 |
    | E | Opérateur 2 - Essai 1 |
    | F | Opérateur 2 - Essai 2 |
    | G | Opérateur 2 - Essai 3 |
    | H | Opérateur 3 - Essai 1 |
    | I | Opérateur 3 - Essai 2 |
    | J | Opérateur 3 - Essai 3 |
    
    **Exemple :** Cellule B2 = Mesure de l'essai 1, opérateur 1, pièce 1
    
    ### 🎯 Résultats attendus avec votre template
    
    - **R&R** = 0.193
    - **EV (Répétabilité)** = 0.175
    - **AV (Reproductibilité)** = 0.080
    - **Vp** = 0.530
    - **VT** = 0.561
    
    ### 📊 Interprétation
    
    - **%R&R ≤ 10%** : Système acceptable ✅
    - **10% < %R&R ≤ 30%** : Système marginal ⚠️
    - **%R&R > 30%** : Système non acceptable ❌
    
    ### ⚙️ Coefficient de confiance
    Par défaut : **5.15** (correspond à 99% de la population normale)
    - 4.00 : 95.45% de la population
    - 5.15 : 99.00% de la population (recommandé pour MSA)
    - 6.00 : 99.73% de la population
    """)
