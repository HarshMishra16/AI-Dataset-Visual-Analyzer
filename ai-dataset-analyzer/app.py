import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Dataset Visual Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D0F14;
    color: #E8EAF0;
}

.main { background-color: #0D0F14; }

.stApp { background-color: #0D0F14; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #00FFB3;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #7A8098;
    margin-top: 8px;
}
.stat-card {
    background: #161A24;
    border: 1px solid #252B3B;
    border-left: 3px solid #00FFB3;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7A8098;
    font-family: 'Space Mono', monospace;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00FFB3;
    font-family: 'Space Mono', monospace;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00FFB3;
    border-bottom: 1px solid #252B3B;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 30px;
}
.quality-bar-wrap {
    background: #161A24;
    border: 1px solid #252B3B;
    border-radius: 10px;
    padding: 24px;
}
.quality-score-num {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}
.badge-green { background: #00FFB322; color: #00FFB3; border: 1px solid #00FFB355; }
.badge-yellow { background: #FFD60022; color: #FFD600; border: 1px solid #FFD60055; }
.badge-red { background: #FF4B4B22; color: #FF4B4B; border: 1px solid #FF4B4B55; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#161A24',
    'axes.facecolor': '#161A24',
    'axes.edgecolor': '#252B3B',
    'axes.labelcolor': '#7A8098',
    'xtick.color': '#7A8098',
    'ytick.color': '#7A8098',
    'grid.color': '#252B3B',
    'text.color': '#E8EAF0',
    'font.family': 'monospace',
})

ACCENT = '#00FFB3'
ACCENT2 = '#5B6BF8'
WARN = '#FFD600'
DANGER = '#FF4B4B'
PALETTE = [ACCENT, ACCENT2, WARN, DANGER, '#B388FF', '#FF7043', '#26C6DA', '#FFCA28']

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_quality_score(df):
    score = 100
    missing_pct = df.isnull().mean().mean() * 100
    score -= min(40, missing_pct * 2)
    dup_pct = df.duplicated().mean() * 100
    score -= min(20, dup_pct * 2)
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        outlier_pct = 0
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).mean()
            outlier_pct += outliers
        outlier_pct = (outlier_pct / len(num_cols)) * 100
        score -= min(20, outlier_pct * 1.5)
    return max(0, round(score, 1))

def quality_label(score):
    if score >= 80: return "PRODUCTION READY", "badge-green"
    elif score >= 55: return "NEEDS CLEANING", "badge-yellow"
    else: return "LOW QUALITY", "badge-red"

def score_color(score):
    if score >= 80: return ACCENT
    elif score >= 55: return WARN
    else: return DANGER

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 AI Dataset<br>Visual Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a CSV dataset → get infographic-quality visual analysis for AI training pipelines</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="border: 1.5px dashed #252B3B; border-radius:12px; padding: 40px; text-align:center; color:#7A8098; margin-top:20px;">
        <div style="font-family:'Space Mono',monospace; font-size:2rem; color:#252B3B;">▲</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; letter-spacing:2px; margin-top:8px;">DROP A CSV DATASET TO BEGIN</div>
        <div style="font-size:0.85rem; margin-top:6px;">Works with any structured CSV — Iris, Titanic, custom datasets, AI training data</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded, encoding='latin-1')
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
missing_pct = round(df.isnull().mean().mean() * 100, 1)
dup_count = df.duplicated().sum()
quality = compute_quality_score(df)
ql, qbadge = quality_label(quality)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">// Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val in zip(
    [c1, c2, c3, c4, c5],
    ["ROWS", "COLUMNS", "NUMERIC FEATURES", "CATEGORICAL", "MISSING DATA"],
    [f"{len(df):,}", str(len(df.columns)), str(len(num_cols)), str(len(cat_cols)), f"{missing_pct}%"]
):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{val}</div>
    </div>""", unsafe_allow_html=True)

# ── Quality Score ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">// Data Quality Score</div>', unsafe_allow_html=True)
qc1, qc2 = st.columns([1, 2])

with qc1:
    sc = score_color(quality)
    st.markdown(f"""
    <div class="quality-bar-wrap" style="text-align:center;">
        <div class="stat-label">OVERALL QUALITY INDEX</div>
        <div class="quality-score-num" style="color:{sc};">{quality}</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:{sc}; margin-bottom:12px;">/100</div>
        <span class="badge {qbadge}">{ql}</span>
    </div>
    """, unsafe_allow_html=True)

with qc2:
    fig, ax = plt.subplots(figsize=(6, 2.2))
    categories = ['Completeness', 'Uniqueness', 'Outlier Health']
    miss_score = max(0, 100 - min(100, missing_pct * 2))
    dup_score = max(0, 100 - min(100, (dup_count / max(len(df),1)) * 100 * 2))
    if len(num_cols) > 0:
        outlier_rates = []
        for c in num_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            outlier_rates.append(((df[c] < q1-1.5*iqr)|(df[c] > q3+1.5*iqr)).mean())
        out_score = max(0, 100 - min(100, np.mean(outlier_rates)*100*1.5))
    else:
        out_score = 100
    scores = [miss_score, dup_score, out_score]
    colors = [ACCENT if s >= 80 else WARN if s >= 55 else DANGER for s in scores]
    bars = ax.barh(categories, scores, color=colors, height=0.5, zorder=2)
    ax.barh(categories, [100]*3, color='#252B3B', height=0.5, zorder=1)
    ax.set_xlim(0, 110)
    for bar, score in zip(bars, scores):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                f'{score:.0f}', va='center', color='#E8EAF0', fontsize=9)
    ax.set_xlabel('')
    ax.grid(False)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Column 1: Missing Values + Distribution ───────────────────────────────────
st.markdown('<div class="section-title">// Feature Analysis</div>', unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.markdown("**Missing Values by Column**")
    miss = df.isnull().mean().sort_values(ascending=True) * 100
    fig, ax = plt.subplots(figsize=(5, max(3, len(miss)*0.35)))
    colors_m = [DANGER if v > 20 else WARN if v > 5 else ACCENT for v in miss.values]
    ax.barh(miss.index, miss.values, color=colors_m, height=0.6)
    ax.set_xlabel('Missing %', color='#7A8098')
    ax.set_xlim(0, max(miss.max()+5, 10))
    ax.grid(axis='x', alpha=0.3)
    ax.spines[:].set_visible(False)
    for i, (idx, val) in enumerate(miss.items()):
        ax.text(val+0.3, i, f'{val:.1f}%', va='center', fontsize=8, color='#E8EAF0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with right:
    if num_cols:
        st.markdown("**Numeric Feature Distributions**")
        n = min(6, len(num_cols))
        cols_to_plot = num_cols[:n]
        fig, axes = plt.subplots(1, n, figsize=(max(5, n*1.7), 3.2))
        if n == 1: axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            data = df[col].dropna()
            ax.hist(data, bins=20, color=ACCENT2, alpha=0.85, edgecolor='none')
            ax.set_title(col[:10], fontsize=8, color='#E8EAF0')
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif cat_cols:
        st.markdown("**Categorical Column — Value Counts**")
        col = cat_cols[0]
        vc = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(range(len(vc)), vc.values, color=PALETTE[:len(vc)])
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(x)[:12] for x in vc.index], rotation=30, ha='right', fontsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Correlation + Label Distribution ─────────────────────────────────────────
st.markdown('<div class="section-title">// Correlation & Label Distribution</div>', unsafe_allow_html=True)
l2, r2 = st.columns(2)

with l2:
    if len(num_cols) >= 2:
        st.markdown("**Feature Correlation Matrix**")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=0.3, linecolor='#0D0F14',
                    annot=len(num_cols)<=8, fmt='.1f', annot_kws={"size": 7},
                    ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Need 2+ numeric columns for correlation matrix.")

with r2:
    if cat_cols:
        target = cat_cols[-1]
        st.markdown(f"**Label Distribution — `{target}`**")
        vc = df[target].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=None, autopct='%1.1f%%',
            colors=PALETTE[:len(vc)], startangle=140,
            wedgeprops=dict(edgecolor='#0D0F14', linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color('#0D0F14')
            at.set_fontweight('bold')
        ax.legend(wedges, [str(x)[:15] for x in vc.index],
                  loc="center left", bbox_to_anchor=(1, 0.5),
                  fontsize=8, frameon=False, labelcolor='#E8EAF0')
        circle = plt.Circle((0,0), 0.55, color='#161A24')
        ax.add_patch(circle)
        ax.text(0, 0, f'{len(vc)}\nCLASSES', ha='center', va='center',
                fontsize=10, fontweight='bold', color=ACCENT,
                fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif len(num_cols) > 0:
        target = num_cols[-1]
        st.markdown(f"**Target Distribution — `{target}`**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df[target].dropna(), bins=25, color=ACCENT, edgecolor='#0D0F14', linewidth=0.5)
        ax.set_xlabel(target, color='#7A8098')
        ax.spines[:].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Data Sample ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">// Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(
    df.head(10).style.set_properties(**{
        'background-color': '#161A24',
        'color': '#E8EAF0',
        'border-color': '#252B3B'
    }),
    use_container_width=True
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:40px; padding:20px; border-top: 1px solid #252B3B; color:#7A8098; font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:1px;">
    BUILT BY HARSH KUMAR MISHRA · AI DATASET VISUAL ANALYZER 
</div>
""", unsafe_allow_html=True)
