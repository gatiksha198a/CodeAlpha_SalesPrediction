
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Sales Prediction | CodeAlpha",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── BACKGROUND CSS — Dark luxury finance theme ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;700&display=swap');

.stApp {
    background-color: #020c07;
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(0,200,100,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 0%, rgba(0,255,150,0.07) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 100%, rgba(0,180,80,0.08) 0%, transparent 55%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cdefs%3E%3Cpattern id='grid' width='40' height='40' patternUnits='userSpaceOnUse'%3E%3Cpath d='M 40 0 L 0 0 0 40' fill='none' stroke='rgba(0,200,80,0.04)' stroke-width='0.5'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='400' height='400' fill='url(%23grid)'/%3E%3Cpolyline points='0,320 40,280 80,300 120,240 160,200 200,220 240,160 280,130 320,110 360,80 400,60' fill='none' stroke='rgba(0,200,80,0.08)' stroke-width='2'/%3E%3Ccircle cx='120' cy='240' r='3' fill='rgba(0,200,80,0.15)'/%3E%3Ccircle cx='200' cy='220' r='3' fill='rgba(0,200,80,0.15)'/%3E%3Ccircle cx='280' cy='130' r='3' fill='rgba(0,200,80,0.15)'/%3E%3Ccircle cx='360' cy='80' r='3' fill='rgba(0,200,80,0.15)'/%3E%3C/svg%3E");
    background-size: cover, cover, cover, 400px 400px;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(2, 12, 7, 0.97) !important;
    border-right: 1px solid rgba(0,200,80,0.15);
}
[data-testid="stSidebar"] * { color: #b8f0d0 !important; }

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #00c850;
    text-align: center;
    letter-spacing: 2px;
    text-shadow: 0 0 50px rgba(0,200,80,0.3);
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: rgba(184,240,208,0.45);
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.kpi-card {
    background: rgba(0,200,80,0.06);
    border: 1px solid rgba(0,200,80,0.2);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}
.kpi-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00c850;
}
.kpi-label { font-size: 0.7rem; color: rgba(184,240,208,0.5); letter-spacing: 1.5px; text-transform: uppercase; }

.pred-result {
    background: linear-gradient(135deg, rgba(0,200,80,0.12), rgba(0,150,60,0.08));
    border: 2px solid rgba(0,200,80,0.4);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-amount {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #00c850;
}

h1,h2,h3,h4,h5,h6,p,li,label,span,div { color: #b8f0d0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #00c850, #008040) !important;
    color: #020c07 !important; border: none !important;
    border-radius: 8px !important; padding: 11px 34px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 1.5px !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 20px rgba(0,200,80,0.3) !important;
}
.stButton > button:hover { box-shadow: 0 8px 30px rgba(0,200,80,0.5) !important; }
.stTabs [data-baseweb="tab"] { color: #00c850 !important; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid #00c850 !important; }
[data-testid="stMetricValue"] { color: #00c850 !important; font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }
.stSlider > label { color: #00c850 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA & TRAIN ───────────────────────────────────────
@st.cache_data
def load_and_train(file=None):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        # Map common variations
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == 'tv': col_map[c] = 'TV'
            elif cl == 'radio': col_map[c] = 'Radio'
            elif cl in ['newspaper','news']: col_map[c] = 'Newspaper'
            elif cl == 'sales': col_map[c] = 'Sales'
        df = df.rename(columns=col_map)
    else:
        np.random.seed(42); n = 200
        TV = np.random.uniform(0.7, 296.4, n)
        Radio = np.random.uniform(0.0, 49.6, n)
        News  = np.random.uniform(0.3, 114.0, n)
        Sales = (0.045*TV + 0.19*Radio + 0.002*News
                 + np.random.normal(0, 1.2, n) + 2.5).clip(1, 30)
        df = pd.DataFrame({'TV': TV, 'Radio': Radio,
                           'Newspaper': News, 'Sales': np.round(Sales, 2)})

    features = [c for c in ['TV','Radio','Newspaper'] if c in df.columns]
    X = df[features]; y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train); Xte = scaler.transform(X_test)

    models = {
        'Linear Regression':  LinearRegression(),
        'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        results[name] = {
            'model': m,
            'predictions': yp,
            'R2':  r2_score(y_test, yp),
            'MAE': mean_absolute_error(y_test, yp),
            'RMSE': np.sqrt(mean_squared_error(y_test, yp))
        }
    return df, scaler, results, y_test, features

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Upload Dataset")
    uploaded = st.file_uploader("Upload 'Advertising.csv'", type=['csv'])
    st.markdown("---")
    st.markdown("**Kaggle Dataset:**")
    st.markdown("kaggle.com/datasets/\nbumba5341/advertisingcsv")
    st.markdown("---")
    st.markdown("## 🔮 Predict Sales")

df, scaler, results, y_test, features = load_and_train(uploaded)

if uploaded is None:
    st.info("💡 Using demo data. Upload Advertising.csv for real predictions!")

# ── HERO ────────────────────────────────────────────────────
st.markdown('<div class="hero-title">💰 Sales Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Advertising Spend → Revenue Forecast · CodeAlpha · Gatiksha</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── METRICS ─────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("📊 Dataset Size", f"{len(df)} rows")
with col2: st.metric("🏆 Best R² Score", f"{best['R2']:.3f}")
with col3: st.metric("📉 Best MAE", f"{best['MAE']:.3f}")
with col4: st.metric("🥇 Best Model", best_name.split()[0])

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR SLIDERS ─────────────────────────────────────────
with st.sidebar:
    tv_val    = st.slider("📺 TV Budget ($K)",      0.0, 300.0, 150.0, 1.0) if 'TV' in features else 0
    radio_val = st.slider("📻 Radio Budget ($K)",   0.0, 50.0,  25.0,  0.5) if 'Radio' in features else 0
    news_val  = st.slider("📰 Newspaper Budget ($K)", 0.0, 115.0, 30.0, 0.5) if 'Newspaper' in features else 0
    model_sel = st.selectbox("Model", list(results.keys()))
    pred_btn  = st.button("📈 Predict Sales")

# ── TABS ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 EDA", "🏆 Model Comparison", "📋 Data"])

# ── TAB 1: PREDICT ──────────────────────────────────────────
with tab1:
    if pred_btn:
        inp_vals = [v for v, f in zip([tv_val, radio_val, news_val], ['TV','Radio','Newspaper']) if f in features]
        inp = scaler.transform([inp_vals])
        pred = results[model_sel]['model'].predict(inp)[0]

        st.markdown(f"""
        <div class="pred-result">
            <div style="color:rgba(184,240,208,0.5);font-size:0.85rem;letter-spacing:2px;text-transform:uppercase;">Predicted Sales Revenue</div>
            <div class="pred-amount">${pred:.2f}K</div>
            <div style="color:rgba(184,240,208,0.4);font-size:0.85rem;margin-top:6px;">Using {model_sel}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val">📺 ${tv_val:.0f}K</div>
                <div class="kpi-label">TV Budget</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val">📻 ${radio_val:.0f}K</div>
                <div class="kpi-label">Radio Budget</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val">📰 ${news_val:.0f}K</div>
                <div class="kpi-label">Newspaper Budget</div>
            </div>""", unsafe_allow_html=True)

        roi_tv = (pred / tv_val * 100) if tv_val > 0 else 0
        st.markdown(f"""
        <br><div style="background:rgba(0,200,80,0.06);border:1px solid rgba(0,200,80,0.2);border-radius:10px;padding:16px 20px;margin-top:8px;">
            <b>💡 Business Insight:</b> For every $1K spent on TV, you're generating approximately 
            <span style="color:#00c850;font-weight:700;">${roi_tv:.2f}K</span> in predicted sales.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:4rem;">💰</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;color:#00c850;margin-top:16px;">
                Set your ad budget in the sidebar
            </div>
            <div style="color:rgba(184,240,208,0.4);margin-top:8px;">
                Then click "Predict Sales" to forecast revenue
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: EDA ──────────────────────────────────────────────
with tab2:
    BG = '#020c07'
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Ad Budget vs Sales (Scatter)")
        fig = go.Figure()
        colors_map = {'TV': '#00c850', 'Radio': '#00a0ff', 'Newspaper': '#ffa500'}
        for feat in features:
            fig.add_trace(go.Scatter(
                x=df[feat], y=df['Sales'],
                mode='markers', name=feat,
                marker=dict(color=colors_map.get(feat,'#00c850'), size=5, opacity=0.7)
            ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8f0d0'), height=350,
            xaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Budget ($K)'),
            yaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Sales ($K)'),
            legend=dict(bgcolor='rgba(0,0,0,0.3)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Sales Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df['Sales'], nbinsx=25,
            marker_color='#00c850', opacity=0.8,
            marker_line=dict(color='#020c07', width=0.5)
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8f0d0'), height=350,
            xaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Sales ($K)'),
            yaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Count')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Correlation Heatmap")
    fig3, ax = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor(BG); ax.set_facecolor(BG)
    corr = df[features + ['Sales']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens', ax=ax,
                linewidths=0.5, linecolor=BG, annot_kws={'color':'white','size':11})
    ax.tick_params(colors='#b8f0d0')
    ax.set_title('Feature Correlations', color='#00c850', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig3)

# ── TAB 3: MODEL COMPARISON ─────────────────────────────────
with tab3:
    BG = '#020c07'
    # Model scores
    names = list(results.keys()); r2s = [results[n]['R2'] for n in names]
    maes  = [results[n]['MAE'] for n in names]; rmses = [results[n]['RMSE'] for n in names]

    col_a, col_b, col_c = st.columns(3)
    for col, (name, r2, mae, rmse) in zip([col_a,col_b,col_c],
            zip(names, r2s, maes, rmses)):
        is_best = name == best_name
        border_style = "border: 2px solid #00c850;" if is_best else ""
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="{border_style}">
                {'<div style="color:#00c850;font-size:0.7rem;letter-spacing:1px;margin-bottom:4px;">🏆 BEST MODEL</div>' if is_best else ''}
                <div style="font-family:Cormorant Garamond,serif;font-size:1.1rem;color:#00c850;margin-bottom:10px;">{name}</div>
                <div class="kpi-val">{r2:.4f}</div>
                <div class="kpi-label">R² Score</div>
                <div style="margin-top:8px;color:rgba(184,240,208,0.6);font-size:0.8rem;">MAE: {mae:.3f} | RMSE: {rmse:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>")
    st.markdown("##### Actual vs Predicted — Best Model")
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=y_test, y=best['predictions'],
        mode='markers', name='Predictions',
        marker=dict(color='#00c850', size=6, opacity=0.7)
    ))
    mn = min(y_test.min(), best['predictions'].min())
    mx = max(y_test.max(), best['predictions'].max())
    fig_ap.add_trace(go.Scatter(
        x=[mn,mx], y=[mn,mx], mode='lines',
        line=dict(color='rgba(184,240,208,0.4)', dash='dash'), name='Perfect'
    ))
    fig_ap.update_layout(
        title=f'{best_name} — R²={best["R2"]:.4f}',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8f0d0'), height=420,
        xaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Actual Sales ($K)'),
        yaxis=dict(gridcolor='rgba(0,200,80,0.08)', title='Predicted Sales ($K)'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)')
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # Feature Importance
    if hasattr(results['Gradient Boosting']['model'], 'feature_importances_'):
        st.markdown("##### Feature Importance (Gradient Boosting)")
        imp = pd.Series(
            results['Gradient Boosting']['model'].feature_importances_,
            index=features).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker_color=['#00c850','#00a0ff','#ffa500'][:len(features)],
            text=[f'{v:.3f}' for v in imp.values], textposition='outside',
            textfont=dict(color='#b8f0d0')
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8f0d0'), height=300,
            xaxis=dict(gridcolor='rgba(0,200,80,0.08)')
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 4: DATA ──────────────────────────────────────────────
with tab4:
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown("##### Stats")
    st.dataframe(df.describe(), use_container_width=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid rgba(0,200,80,0.1);">
    <span style="color:rgba(184,240,208,0.3);font-size:0.82rem;">
        💰 CodeAlpha Data Science Internship · Task 4 · Gatiksha · CA/DF1/101914
    </span>
</div>
""", unsafe_allow_html=True)
