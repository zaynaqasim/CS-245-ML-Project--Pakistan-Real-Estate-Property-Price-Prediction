"""
CS-245 Machine Learning Project — Pakistan Property Price Predictor
Sukaina Nasir · Zayna Qasim · Hamna Shah
Streamlit 1.31+ compatible. No external font imports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, silhouette_score)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS & MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#0f1117"
CARD   = "#1a1f2e"
ACCENT = "#FFBE32"
PAL    = ["#2E86AB", "#FFBE32", "#A23B72", "#44BBA4", "#E94F37"]

def _mpl():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   CARD,
        "axes.edgecolor":   "#2a2a2a",
        "axes.grid":        True,
        "grid.color":       "#252c3a",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "text.color":       "#cccccc",
        "axes.labelcolor":  "#aaaaaa",
        "xtick.color":      "#888888",
        "ytick.color":      "#888888",
        "axes.titlecolor":  "#dddddd",
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "font.family":      "DejaVu Sans",
        "legend.facecolor": CARD,
        "legend.edgecolor": "#2e3a4a",
        "legend.labelcolor":"#cccccc",
        "legend.fontsize":  8,
    })

_mpl()

# ─────────────────────────────────────────────────────────────────────────────
# CSS — injected via st.html() so it is never rendered as visible text.
# NO @import — that caused "Failed to fetch dynamically imported module" errors.
# ─────────────────────────────────────────────────────────────────────────────
st.html("""<style>
html, body, [class*="css"] { font-family: Georgia, serif; }
.stApp { background: #0f1117; }
.block-container { padding-top: 1.5rem !important; }
#MainMenu, header, footer { visibility: hidden; }

[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] label { color: #cccccc !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] p { color: #aaaaaa !important; }

[data-baseweb="tab-list"] { background: #161c2a !important; border-radius: 10px !important; padding: 4px !important; gap: 3px !important; }
[data-baseweb="tab"] { border-radius: 7px !important; color: #888888 !important; font-size: 0.85rem !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #FFBE32 !important; color: #0f1117 !important; font-weight: 700 !important; }

[data-testid="metric-container"] { background: #1a1f2e; border: 1px solid #2a2a3a; border-radius: 12px; padding: 14px !important; }
[data-testid="metric-container"] label { color: #888888 !important; font-size: 0.75rem !important; letter-spacing: 0.8px; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.5rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #888888 !important; }

.stButton > button { background: linear-gradient(135deg, #FFBE32, #f0a500) !important; color: #0f1117 !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 12px 24px !important; width: 100% !important; font-size: 0.95rem !important; }
.stButton > button:hover { opacity: 0.88; }

.stSelectbox > div > div, .stNumberInput > div > div { background: #1a1f2e !important; border: 1px solid #2e3a4a !important; border-radius: 8px !important; color: #ffffff !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
hr { border-color: #1e2535 !important; }
</style>""")


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("property_data.csv")

    def parse_price(p):
        try:
            p = str(p).replace("PKR","").replace("\n","").strip().replace(",","")
            if "Crore" in p: return float(p.replace("Crore","").strip()) * 100
            elif "Lakh" in p: return float(p.replace("Lakh","").strip())
            elif "Arab"  in p: return float(p.replace("Arab","").strip()) * 10000
            else: return float(p)
        except: return np.nan

    def parse_area(a):
        try:
            a = str(a).strip()
            if "Kanal"    in a: return float(a.replace("Kanal","").strip()) * 20
            elif "Marla"   in a: return float(a.replace("Marla","").strip())
            elif "Sq. Yd." in a: return float(a.replace("Sq. Yd.","").strip()) / 25.2929
            elif "Sq. Ft." in a: return float(a.replace("Sq. Ft.","").strip()) / 272.25
            else: return np.nan
        except: return np.nan

    def clean_int(v):
        try: return float(str(v).strip().replace("+","").replace("Studio","0"))
        except: return np.nan

    def clean_purpose(p):
        p = str(p).lower()
        if "rent" in p: return "For Rent"
        elif any(x in p for x in ["sale","buy","luxury"]): return "For Sale"
        else: return "Other"

    def drop_iqr(df, col, f=3.0):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[col] >= Q1 - f*IQR) & (df[col] <= Q3 + f*IQR)]

    df["price_lakh"]    = df["price"].apply(parse_price)
    df["area_marla"]    = df["area"].apply(parse_area)
    df["purpose_clean"] = df["purpose"].apply(clean_purpose)
    df["bedroom"]       = df["bedroom"].apply(clean_int)
    df["bath"]          = df["bath"].apply(clean_int)
    df["city"]          = df["location_city"].str.strip()

    dc = df.dropna(subset=["price_lakh","area_marla"]).copy()
    dc = drop_iqr(dc,"price_lakh"); dc = drop_iqr(dc,"area_marla")
    dc = dc[dc["purpose_clean"].isin(["For Sale","For Rent"])]
    dc = dc[dc["type"].isin(dc["type"].value_counts().head(7).index)]
    cc = dc["city"].value_counts()
    dc = dc[dc["city"].isin(cc[cc >= 200].index)]

    le_t = LabelEncoder(); dc["type_enc"]    = le_t.fit_transform(dc["type"])
    le_p = LabelEncoder(); dc["purpose_enc"] = le_p.fit_transform(dc["purpose_clean"])
    le_c = LabelEncoder(); dc["city_enc"]    = le_c.fit_transform(dc["city"])

    FEAT = ["area_marla","bedroom","bath","type_enc","purpose_enc","city_enc"]
    dm   = dc[FEAT + ["price_lakh","city","type","purpose_clean"]].dropna().copy()
    return dm, le_t, le_p, le_c


@st.cache_resource(show_spinner=False)
def train_all():
    dm, le_t, le_p, le_c = load_data()
    FEAT = ["area_marla","bedroom","bath","type_enc","purpose_enc","city_enc"]
    X = dm[FEAT]; y = dm["price_lakh"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

    lr = LinearRegression();      lr.fit(Xs_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=4, random_state=42)
    gb.fit(X_tr, y_tr)

    preds = {
        "Linear Regression": lr.predict(Xs_te),
        "Random Forest":     rf.predict(X_te),
        "Gradient Boosting": gb.predict(X_te),
    }
    metrics = {
        n: {"RMSE": round(np.sqrt(mean_squared_error(y_te,p)),2),
            "MAE":  round(mean_absolute_error(y_te,p),2),
            "R²":   round(r2_score(y_te,p),4)}
        for n,p in preds.items()
    }

    # Clustering
    CF  = ["area_marla","bedroom","bath","price_lakh","city_enc","type_enc"]
    csc = StandardScaler(); Xc = csc.fit_transform(dm[CF])
    km  = KMeans(n_clusters=3, random_state=42, n_init=10)
    cl  = km.fit_predict(Xc)
    sil = silhouette_score(Xc, cl)
    cp  = {i: dm["price_lakh"].values[cl==i].mean() for i in range(3)}
    sc3 = sorted(cp, key=cp.get)
    seg_map = {sc3[0]:"Budget", sc3[1]:"Mid-Range", sc3[2]:"Luxury"}

    return dict(lr=lr, rf=rf, gb=gb, sc=sc, csc=csc, km=km,
                X_te=X_te, Xs_te=Xs_te, y_te=y_te,
                preds=preds, metrics=metrics,
                Xc=Xc, cl=cl, sil=sil, seg_map=seg_map)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data & training models — please wait (~45s)…"):
    dm, le_t, le_p, le_c = load_data()
    B = train_all()

FEAT    = ["area_marla","bedroom","bath","type_enc","purpose_enc","city_enc"]
cities  = sorted(le_c.classes_.tolist())
types   = sorted(le_t.classes_.tolist())
purposes= sorted(le_p.classes_.tolist())

def fmt(v):
    return f"PKR {v/100:.2f} Crore" if v >= 100 else f"PKR {v:.1f} Lakh"


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.html("""
<div style="background:linear-gradient(135deg,#1a1f2e,#0d1b2a);
            border:1px solid #2a3a4a;border-radius:16px;
            padding:36px 40px;margin-bottom:24px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
              background:radial-gradient(circle,rgba(255,190,50,0.12),transparent 70%);
              border-radius:50%;"></div>
  <div style="display:inline-block;background:rgba(255,190,50,0.15);
              border:1px solid rgba(255,190,50,0.4);color:#FFBE32;
              font-size:0.68rem;font-weight:700;letter-spacing:1.6px;
              text-transform:uppercase;padding:4px 12px;border-radius:20px;
              margin-bottom:14px;">CS-245 · Machine Learning Project</div>
  <h1 style="font-family:Georgia,serif;font-size:2.4rem;font-weight:900;
             color:#fff;margin:0 0 6px;line-height:1.1;">
    Pakistan <span style="color:#FFBE32;">Property</span> Price Predictor
  </h1>
  <p style="color:#888;font-size:0.95rem;margin:0;">
    Sukaina Nasir &nbsp;·&nbsp; Zayna Qasim &nbsp;·&nbsp; Hamna Shah
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Linear Regression &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp;
    Gradient Boosting &nbsp;·&nbsp; K-Means
  </p>
</div>
""")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.html('<p style="color:#FFBE32;font-size:1.1rem;font-weight:700;'
            'font-family:Georgia,serif;margin-bottom:4px;">🏡 Property Details</p>')
    st.divider()

    st.markdown("**📍 Location & Purpose**")
    city_sel    = st.selectbox("City", cities, index=cities.index("Lahore"))
    purpose_sel = st.radio("Purpose", ["For Sale","For Rent"], horizontal=True)

    st.divider()
    st.markdown("**🏗️ Type & Size**")
    type_sel = st.selectbox("Property Type", types, index=types.index("House"))
    unit     = st.radio("Area Unit", ["Marla","Kanal","Sq. Ft."], horizontal=True)
    defaults = {"Marla":5.0, "Kanal":0.5, "Sq. Ft.":500.0}
    area_raw = st.number_input(f"Area ({unit})",
                               min_value=0.1, max_value=2000.0,
                               value=defaults[unit], step=0.5)
    if unit == "Kanal":     area_m = area_raw * 20
    elif unit == "Sq. Ft.": area_m = area_raw / 272.25
    else:                   area_m = area_raw
    st.caption(f"= {area_m:.2f} Marla")

    st.divider()
    st.markdown("**🛏️ Rooms**")
    beds  = st.slider("Bedrooms",  0, 10, 3)
    baths = st.slider("Bathrooms", 1, 10, 2)

    st.divider()
    st.markdown("**🤖 Model**")
    model_choice = st.radio("", ["Random Forest ⭐","Gradient Boosting","Linear Regression"])
    st.button("🔮 Predict Price", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INPUT VECTOR
# ─────────────────────────────────────────────────────────────────────────────
te = le_t.transform([type_sel])[0]
pe = le_p.transform([purpose_sel])[0]
ce = le_c.transform([city_sel])[0]
inp   = np.array([[area_m, beds, baths, te, pe, ce]])
inp_s = B["sc"].transform(inp)

mkey   = {"Random Forest ⭐":"rf","Gradient Boosting":"gb","Linear Regression":"lr"}[model_choice]
mlabel = model_choice.replace(" ⭐","")

if mkey == "lr":  pred_price = float(B["lr"].predict(inp_s)[0])
elif mkey == "rf": pred_price = float(B["rf"].predict(inp)[0])
else:              pred_price = float(B["gb"].predict(inp)[0])

pred_price = max(1.0, pred_price)
mae_v = B["metrics"][mlabel]["MAE"]

# Cluster prediction
cf_inp = np.array([[area_m, beds, baths, pred_price, ce, te]])
cf_s   = B["csc"].transform(cf_inp)
pred_seg = B["seg_map"].get(int(B["km"].predict(cf_s)[0]), "Mid-Range")
seg_clr  = {"Budget":"#44BBA4","Mid-Range":"#FFBE32","Luxury":"#E94F37"}[pred_seg]


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", "📊 Model Performance", "🗂️ Cluster Analysis", "📈 EDA"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([3, 2], gap="large")

    with col_main:
        st.html('<p style="color:#FFBE32;font-size:1.25rem;font-weight:700;'
                'border-left:4px solid #FFBE32;padding-left:10px;margin-bottom:12px;">'
                'Price Prediction</p>')

        # Big prediction card
        st.html(f"""
        <div style="background:linear-gradient(135deg,#1a1f2e,#0d1b2a);
                    border:2px solid #FFBE32;border-radius:18px;
                    padding:32px;text-align:center;margin:0 0 16px;">
          <p style="color:#FFBE32;font-size:0.65rem;font-weight:700;
                    letter-spacing:1.8px;text-transform:uppercase;margin:0 0 10px;">
            {city_sel} &nbsp;·&nbsp; {type_sel} &nbsp;·&nbsp; {purpose_sel}
          </p>
          <p style="font-family:Georgia,serif;font-size:2.8rem;font-weight:900;
                    color:#fff;margin:0;line-height:1;">
            {fmt(pred_price)}
          </p>
          <p style="color:#888;font-size:0.9rem;margin:8px 0 0;">
            Predicted by {mlabel}
          </p>
          <div style="display:inline-block;margin-top:14px;
                      background:rgba(255,190,50,0.1);
                      border:1px solid rgba(255,190,50,0.3);
                      border-radius:8px;padding:6px 18px;
                      font-size:0.8rem;color:#aaa;">
            Range: {fmt(max(0, pred_price - mae_v))} &ndash; {fmt(pred_price + mae_v)}
          </div>
          <br>
          <span style="display:inline-block;margin-top:14px;
                       background:{seg_clr}28;color:{seg_clr};
                       border:1.5px solid {seg_clr}60;
                       border-radius:8px;padding:6px 20px;
                       font-size:0.85rem;font-weight:700;">
            Market Segment: {pred_seg}
          </span>
        </div>
        """)

        # All 3 models bar chart
        st.markdown("**All Models Comparison**")
        lr_p = float(B["lr"].predict(inp_s)[0])
        rf_p = float(B["rf"].predict(inp)[0])
        gb_p = float(B["gb"].predict(inp)[0])

        _mpl()
        fig, ax = plt.subplots(figsize=(7, 2.6))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        names3 = ["Linear Regression","Random Forest","Gradient Boosting"]
        vals3  = [max(0,lr_p), max(0,rf_p), max(0,gb_p)]
        bars   = ax.barh(names3, vals3, color=PAL[:3], edgecolor="none", height=0.45)
        for bar, val in zip(bars, vals3):
            ax.text(val + max(vals3)*0.01, bar.get_y()+bar.get_height()/2,
                    fmt(val), va="center", fontsize=8.5, color="#ddd", fontweight="600")
        ax.set_xlabel("Predicted Price (Lakhs)")
        ax.spines[:].set_visible(False)
        ax.set_xlim(0, max(vals3)*1.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_side:
        st.html('<p style="color:#FFBE32;font-size:1.25rem;font-weight:700;'
                'border-left:4px solid #FFBE32;padding-left:10px;margin-bottom:12px;">'
                'Property Summary</p>')

        m1, m2, m3 = st.columns(3)
        m1.metric("Area",  f"{area_m:.1f}", "Marla")
        m2.metric("Beds",  str(beds),       "rooms")
        m3.metric("Baths", str(baths),      "baths")

        st.html(f"""
        <div style="background:#1a1f2e;border:1px solid #2a3a4a;border-radius:12px;
                    padding:16px;margin:12px 0 8px;">
          <p style="color:#666;font-size:0.62rem;letter-spacing:1px;
                    text-transform:uppercase;margin:0 0 5px;">Area Breakdown</p>
          <p style="color:#ccc;font-size:0.88rem;margin:0;">
            {area_m:.2f} Marla &nbsp;=&nbsp; {area_m/20:.3f} Kanal
            &nbsp;=&nbsp; {area_m*272.25:.0f} Sq.Ft
          </p>
        </div>
        <div style="background:#1a1f2e;border:1px solid #2a3a4a;border-radius:12px;
                    padding:16px;margin:8px 0;">
          <p style="color:#666;font-size:0.62rem;letter-spacing:1px;
                    text-transform:uppercase;margin:0 0 5px;">Model Accuracy</p>
          <p style="color:#ccc;font-size:0.88rem;margin:0;">
            <b style="color:#FFBE32;">{mlabel}</b> &mdash;
            R² = <b style="color:#FFBE32;">{B['metrics'][mlabel]['R²']}</b>
            ({B['metrics'][mlabel]['R²']*100:.1f}% variance explained)
          </p>
        </div>
        <div style="background:#1a1f2e;border:1px solid #2a3a4a;border-radius:12px;
                    padding:16px;margin:8px 0;">
          <p style="color:#666;font-size:0.62rem;letter-spacing:1px;
                    text-transform:uppercase;margin:0 0 5px;">Avg Error (MAE)</p>
          <p style="color:#ccc;font-size:0.88rem;margin:0;">
            &plusmn; <b style="color:#FFBE32;">{fmt(mae_v)}</b> on test set
          </p>
        </div>
        <div style="background:#1a1f2e;border:1px solid #2a3a4a;border-radius:12px;
                    padding:16px;margin:8px 0;">
          <p style="color:#666;font-size:0.62rem;letter-spacing:1px;
                    text-transform:uppercase;margin:0 0 5px;">Market Segment</p>
          <p style="color:{seg_clr};font-size:0.95rem;font-weight:700;margin:0;">
            {pred_seg}
          </p>
        </div>
        """)

        # Feature importance mini chart
        st.markdown("**Feature Importance (RF)**")
        _mpl()
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(CARD)
        fi = pd.Series(B["rf"].feature_importances_, index=FEAT).sort_values()
        colors_fi = [ACCENT if f in ["area_marla","city_enc"] else PAL[0] for f in fi.index]
        ax2.barh(fi.index, fi.values, color=colors_fi, edgecolor="none", height=0.55)
        ax2.set_xlabel("Importance"); ax2.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.html('<p style="color:#FFBE32;font-size:1.25rem;font-weight:700;'
            'border-left:4px solid #FFBE32;padding-left:10px;margin-bottom:12px;">'
            'Supervised Model Evaluation</p>')

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1,c2,c3],
                         ["Linear Regression","Random Forest","Gradient Boosting"]):
        m = B["metrics"][name]
        col.metric(name, f"R²={m['R²']}", f"RMSE={m['RMSE']}L  MAE={m['MAE']}L")

    df_m = pd.DataFrame(B["metrics"]).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(
        df_m.style
            .highlight_max(subset=["R²"], color="#1a3a1a")
            .highlight_min(subset=["RMSE","MAE"], color="#1a3a1a")
            .format({"RMSE":"{:.2f}","MAE":"{:.2f}","R²":"{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown("##### Actual vs Predicted — Random Forest")
        _mpl()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        ax.scatter(B["y_te"], B["preds"]["Random Forest"],
                   alpha=0.2, s=6, color=ACCENT, edgecolors="none")
        lo, hi = B["y_te"].min(), B["y_te"].max()
        ax.plot([lo,hi],[lo,hi], color="#E94F37", linewidth=1.5,
                linestyle="--", label="Perfect prediction")
        ax.set_title(f"Random Forest  ·  R² = {B['metrics']['Random Forest']['R²']}",
                     color="#ddd")
        ax.set_xlabel("Actual (Lakhs)"); ax.set_ylabel("Predicted (Lakhs)")
        ax.spines[:].set_visible(False); ax.legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with cb:
        st.markdown("##### Residual Distributions")
        _mpl()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        for i,(n,p) in enumerate(B["preds"].items()):
            res = B["y_te"].values - p
            ax.hist(res, bins=60, alpha=0.55, color=PAL[i],
                    edgecolor="none", label=f"{n} (std={res.std():.0f}L)")
        ax.axvline(0, color="#E94F37", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_ylabel("Count")
        ax.set_title("Residuals — All Models", color="#ddd")
        ax.spines[:].set_visible(False); ax.legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    cc2, cd = st.columns(2, gap="large")

    with cc2:
        st.markdown("##### Metrics Comparison")
        _mpl()
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
        fig.patch.set_facecolor(BG)
        snames      = ["LR","RF","GB"]
        mnames_full = ["Linear Regression","Random Forest","Gradient Boosting"]
        for ax, (key, lbl, lo_b) in zip(axes, [
            ("RMSE","RMSE (Lakhs)",True),
            ("MAE","MAE (Lakhs)",True),
            ("R²","R² Score",False)
        ]):
            ax.set_facecolor(CARD)
            vs   = [B["metrics"][m][key] for m in mnames_full]
            best = min(vs) if lo_b else max(vs)
            bcs  = [ACCENT if v==best else PAL[0] for v in vs]
            bars2= ax.bar(snames, vs, color=bcs, edgecolor="none", width=0.5)
            if not lo_b: ax.set_ylim(0, 1.1)
            ax.set_title(lbl, color="#ddd", fontsize=9)
            ax.spines[:].set_visible(False)
            for bar,val in zip(bars2,vs):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f"{val:.2f}", ha="center", fontsize=7, color="#ddd", fontweight="600")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with cd:
        st.markdown("##### Absolute Error Distribution")
        _mpl()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        for i,(n,p) in enumerate(B["preds"].items()):
            errs = np.abs(B["y_te"].values - p)
            ax.hist(errs, bins=70, alpha=0.55, color=PAL[i],
                    edgecolor="none", label=f"{n} (med={np.median(errs):.0f}L)")
        ax.set_xlabel("Absolute Error (Lakhs)"); ax.set_ylabel("Count")
        ax.set_title("Error Distribution", color="#ddd")
        ax.spines[:].set_visible(False); ax.legend(); ax.set_xlim(0)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown("##### Feature Importance — RF vs Gradient Boosting")
    _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor(BG)
    for ax,(mo,nm,c) in zip(axes,[(B["rf"],"Random Forest",ACCENT),
                                   (B["gb"],"Gradient Boosting",PAL[2])]):
        ax.set_facecolor(CARD)
        fi = pd.Series(mo.feature_importances_, index=FEAT).sort_values()
        ax.barh(fi.index, fi.values, color=c, edgecolor="none", alpha=0.9, height=0.6)
        ax.set_title(nm, color="#ddd", fontsize=10); ax.set_xlabel("Importance Score")
        ax.spines[:].set_visible(False)
        for i,(f,v) in enumerate(fi.items()):
            ax.text(v+0.003, i, f"{v:.3f}", va="center", fontsize=8, color="#bbb")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.html('<p style="color:#FFBE32;font-size:1.25rem;font-weight:700;'
            'border-left:4px solid #FFBE32;padding-left:10px;margin-bottom:12px;">'
            'K-Means Clustering (K=3)</p>')

    k1, k2, k3 = st.columns(3)
    k1.metric("Silhouette Score", f"{B['sil']:.4f}", "K=3")
    k2.metric("Clusters Found", "3", "Budget · Mid · Luxury")
    k3.metric("Properties Clustered", f"{len(dm):,}", "listings")

    st.divider()
    seg_clrs = {"Budget":"#44BBA4","Mid-Range":ACCENT,"Luxury":"#E94F37"}
    seg_rev  = {v:k for k,v in B["seg_map"].items()}

    ke, kf = st.columns(2, gap="large")
    with ke:
        st.markdown("##### PCA Cluster Visualisation")
        _mpl()
        pca2  = PCA(n_components=2, random_state=42)
        Xpca  = pca2.fit_transform(B["Xc"])
        exp   = pca2.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        samp  = np.random.choice(len(B["cl"]), size=min(5000,len(B["cl"])), replace=False)
        for seg,color in seg_clrs.items():
            cid  = seg_rev.get(seg, 0)
            mask = B["cl"][samp] == cid
            ax.scatter(Xpca[samp][mask,0], Xpca[samp][mask,1],
                       alpha=0.35, s=10, color=color, edgecolors="none", label=seg)
        ctrs = pca2.transform(B["km"].cluster_centers_)
        ax.scatter(ctrs[:,0], ctrs[:,1], c="white", marker="X",
                   s=150, zorder=10, edgecolors="#333", linewidths=1)
        ax.set_xlabel(f"PC1 ({exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({exp[1]*100:.1f}%)")
        ax.set_title(f"Clusters in 2D  |  Silhouette={B['sil']:.4f}", color="#ddd")
        ax.spines[:].set_visible(False); ax.legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with kf:
        st.markdown("##### Price Distribution per Segment")
        _mpl()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        for seg,color in seg_clrs.items():
            cid   = seg_rev.get(seg, 0)
            prices= dm["price_lakh"].values[B["cl"]==cid]
            ax.hist(prices, bins=50, alpha=0.65, color=color,
                    edgecolor="none", label=f"{seg} (n={len(prices):,})")
            ax.axvline(np.median(prices), color=color, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Price (Lakhs)"); ax.set_ylabel("Count")
        ax.set_title("Price per Cluster", color="#ddd")
        ax.spines[:].set_visible(False); ax.legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown("##### Cluster Profiles")
    cdf     = dm.copy()
    cdf["Segment"] = [B["seg_map"].get(c,"?") for c in B["cl"]]
    profile = cdf.groupby("Segment")[["area_marla","bedroom","bath","price_lakh"]].mean().round(2)
    profile.columns = ["Avg Area (Marla)","Avg Bedrooms","Avg Bathrooms","Avg Price (Lakhs)"]
    try: profile = profile.loc[["Budget","Mid-Range","Luxury"]]
    except: pass
    st.dataframe(profile, use_container_width=True)

    st.markdown("##### Area vs Price — by Segment")
    _mpl()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    samp2 = np.random.choice(len(B["cl"]), size=min(6000,len(B["cl"])), replace=False)
    for seg,color in seg_clrs.items():
        cid  = seg_rev.get(seg, 0)
        mask = B["cl"][samp2] == cid
        ax.scatter(dm["area_marla"].values[samp2][mask],
                   dm["price_lakh"].values[samp2][mask],
                   alpha=0.35, s=8, color=color, edgecolors="none", label=seg)
    ax.set_xlabel("Area (Marla)"); ax.set_ylabel("Price (Lakhs)")
    ax.set_title("Area vs Price — Coloured by Segment", color="#ddd")
    ax.spines[:].set_visible(False); ax.legend()
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.html('<p style="color:#FFBE32;font-size:1.25rem;font-weight:700;'
            'border-left:4px solid #FFBE32;padding-left:10px;margin-bottom:12px;">'
            'Exploratory Data Analysis</p>')

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Listings",  f"{len(dm):,}",                             "after cleaning")
    e2.metric("Cities",          str(len(cities)),                           "across Pakistan")
    e3.metric("Price Range",
              f"{dm['price_lakh'].min():.0f}–{dm['price_lakh'].max():.0f}", "Lakhs PKR")
    e4.metric("Median Price",    f"{dm['price_lakh'].median():.0f}L",        "PKR")

    st.divider()
    ea, eb = st.columns(2, gap="large")

    with ea:
        st.markdown("##### Price Distribution")
        _mpl()
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        fig.patch.set_facecolor(BG)
        for ax in axes: ax.set_facecolor(CARD)
        axes[0].hist(dm["price_lakh"], bins=60, color=PAL[0],
                     edgecolor="none", alpha=0.85)
        axes[0].set_title("Raw Price (Lakhs)", color="#ddd", fontsize=9)
        axes[0].spines[:].set_visible(False)
        axes[1].hist(np.log1p(dm["price_lakh"]), bins=60, color=ACCENT,
                     edgecolor="none", alpha=0.85)
        axes[1].set_title("Log-Transformed", color="#ddd", fontsize=9)
        axes[1].spines[:].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("##### Median Price by City")
        _mpl()
        cp2 = dm.groupby("city")["price_lakh"].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(4, len(cp2)*0.34)))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        bar_c2 = [ACCENT if c==city_sel else PAL[0] for c in cp2.index]
        ax.barh(cp2.index[::-1], cp2.values[::-1],
                color=bar_c2[::-1], edgecolor="none", height=0.65)
        ax.set_xlabel("Median Price (Lakhs)")
        ax.set_title("Median Price by City", color="#ddd")
        ax.spines[:].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with eb:
        st.markdown("##### Feature Correlation Heatmap")
        _mpl()
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        corr = dm[FEAT + ["price_lakh"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.4, linecolor=BG,
                    annot_kws={"size":8})
        ax.set_title("Correlation Heatmap", color="#ddd", fontsize=10)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("##### Bedrooms vs Median Price")
        _mpl()
        bp2 = dm[dm["bedroom"] <= 8].groupby("bedroom")["price_lakh"].median()
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        bc3 = [ACCENT if b==beds else PAL[0] for b in bp2.index]
        ax.bar(bp2.index, bp2.values, color=bc3, edgecolor="none", width=0.65)
        ax.set_xlabel("Bedrooms"); ax.set_ylabel("Median Price (Lakhs)")
        ax.set_title("Median Price by # Bedrooms", color="#ddd")
        ax.spines[:].set_visible(False)
        for x,y in zip(bp2.index, bp2.values):
            ax.text(x, y+1.5, f"{y:.0f}", ha="center", fontsize=8, color="#ddd")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown("##### Price Distribution by Property Type")
    _mpl()
    type_ord = dm.groupby("type")["price_lakh"].median().sort_values(ascending=False).index
    data_t   = [dm[dm["type"]==t]["price_lakh"].values for t in type_ord]
    fig, ax  = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    bp3 = ax.boxplot(data_t, patch_artist=True,
                     medianprops={"color":"#0f1117","linewidth":2},
                     whiskerprops={"color":"#aaa"},
                     capprops={"color":"#aaa"},
                     flierprops={"marker":"o","markersize":2,"alpha":0.3,"color":"#888"})
    for patch,color in zip(bp3["boxes"], PAL*3):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(type_ord, rotation=20)
    ax.set_ylabel("Price (Lakhs)")
    ax.set_title("Price by Property Type", color="#ddd")
    ax.spines[:].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.html("""
<p style="text-align:center;color:#444;font-size:0.75rem;padding:4px 0 12px;">
  CS-245 Machine Learning Project &nbsp;·&nbsp;
  Sukaina Nasir &middot; Zayna Qasim &middot; Hamna Shah &nbsp;·&nbsp;
  Random Forest &middot; Gradient Boosting &middot; Linear Regression &middot; K-Means
</p>
""")
