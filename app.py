from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import sys
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import add_features, get_raw_columns

API_URL = "http://127.0.0.1:8000"   # địa chỉ API (api.py)

# ============================================================
# CẤU HÌNH CHUNG (sửa thông tin đồ án ở đây)
# ============================================================
APP_TITLE = "Dự đoán giá nhà tại TP. Hồ Chí Minh"
PROJECT_INFO = {
    "Môn học": "Machine Learning / Khoa học dữ liệu",
    "Nhóm thực hiện": "01",
    "GVHD": "Ths. Nguyễn Thanh Phước",
}

st.set_page_config(
    page_title="HCMC Housing | Dự đoán giá nhà",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GIAO DIỆN (DARK MODE MODERN CSS)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root { color-scheme: dark; }
html, body, [class*="css"], .stApp {
    font-family: 'Be Vietnam Pro', sans-serif;
    color: #f1f5f9 !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0b1120 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(11, 17, 32, 0.85) !important;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1380px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.03);
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(59, 130, 246, 0.18);
    border-color: rgba(96, 165, 250, 0.35);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

/* Text & Headings */
h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, label {
    color: #f8fafc !important;
}
small, .stCaption, [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
.page-sub {
    color: #94a3b8 !important;
    margin: -8px 0 20px;
    font-size: 0.98rem;
}
.section-title {
    color: #f8fafc !important;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 26px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 20px;
    background: #3b82f6;
    border-radius: 4px;
}

/* Hero Banner */
.hero {
    padding: 32px 38px;
    border-radius: 22px;
    color: #ffffff;
    margin-bottom: 24px;
    background: linear-gradient(125deg, #020617 0%, #1e3a8a 55%, #0369a1 100%);
    border: 1px solid rgba(59, 130, 246, 0.35);
    box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.5);
}
.hero h1 { color: #ffffff !important; font-size: 2.15rem; margin: 0 0 8px 0; font-weight: 800; }
.hero p { color: #bae6fd !important; margin: 0 0 18px 0; font-size: 1.02rem; }
.chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: #ffffff !important;
    border-radius: 999px;
    padding: 6px 15px;
    font-size: 0.83rem;
    margin: 0 8px 6px 0;
    font-weight: 600;
}

/* Info Cards */
.info-card {
    background: #172235 !important;
    border: 1px solid #2a3950 !important;
    border-radius: 16px;
    padding: 18px 22px;
    height: 100%;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.22);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.info-card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6 !important;
}
.info-card .label {
    color: #94a3b8 !important;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.info-card .value {
    color: #ffffff !important;
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.2;
}
.info-card .hint {
    color: #64748b !important;
    font-size: 0.8rem;
    margin-top: 6px;
}
.info-card.accent {
    background: linear-gradient(145deg, #172b49 0%, #172235 100%) !important;
    border-color: #3b82f6 !important;
}
.info-card.accent .value { color: #60a5fa !important; }

/* Pipeline Flow */
.flow { display: flex; align-items: stretch; gap: 10px; }
.flow-card {
    flex: 1;
    background: #172235 !important;
    border: 1px solid #2a3950 !important;
    border-top: 3px solid #3b82f6 !important;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.flow-card.soon {
    background: #111927 !important;
    border: 1.5px dashed #334155 !important;
    border-top: 3px dashed #64748b !important;
}
.flow-card .tag { color: #60a5fa !important; font-weight: 800; font-size: 0.72rem; letter-spacing: 0.05em; }
.flow-card .ttl { color: #f8fafc !important; font-weight: 700; font-size: 0.95rem; margin: 4px 0; }
.flow-card .cnt { color: #ffffff !important; font-size: 1.35rem; font-weight: 800; }
.flow-card .dsc { color: #94a3b8 !important; font-size: 0.78rem; margin-top: 5px; }
.flow .arrow { align-self: center; color: #475569; font-size: 1.6rem; font-weight: bold; }

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 26px 28px;
    color: white;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    border: 1px solid #60a5fa;
    box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.4);
}
.result-card .label { color: #bfdbfe !important; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; }
.result-card .value { color: #ffffff !important; font-size: 2.7rem; font-weight: 800; line-height: 1.15; margin: 4px 0; }
.result-card .hint { color: #dbeafe !important; font-size: 0.84rem; margin-top: 6px; }

.empty-box {
    background: #172235 !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 16px;
    padding: 28px;
    color: #94a3b8 !important;
    text-align: center;
    margin: 10px 0 16px;
}

/* Container border, Tabs, Inputs & Metrics */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #172235 !important;
    border-color: #2a3950 !important;
    border-radius: 16px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #172235 !important;
    padding: 5px;
    border-radius: 12px;
    border: 1px solid #2a3950;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 600;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}
div[data-testid="stMetric"] {
    background: #0f172a !important;
    padding: 14px 18px;
    border: 1px solid #2a3950 !important;
    border-radius: 14px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f8fafc !important; }
input, textarea, [data-baseweb="select"] > div {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border-color: #334155 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] { background: #172235 !important; }
[data-baseweb="menu"] * { color: #f1f5f9 !important; }
button[kind="secondary"] {
    background-color: #172235 !important;
    color: #f1f5f9 !important;
    border-color: #334155 !important;
}
button[kind="secondary"]:hover {
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

PRESENT_CSS = """
<style>
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1rem; max-width: 1600px; }
html, body, [class*="css"], .stMarkdown p, label { font-size: 1.08rem; }
.info-card .value { font-size: 2.05rem; }
.hero h1 { font-size: 2.5rem; }
.hero p { font-size: 1.2rem; }
.section-title { font-size: 1.5rem; }
</style>
"""

PRESENT = False


def H(base=400):
    return base + 100 if PRESENT else base


# ============================================================
# TIỆN ÍCH HIỂN THỊ
# ============================================================
COLORWAY = ["#3b82f6", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

FEATURE_VI = {
    "area": "Diện tích (m²)", "frontage": "Mặt tiền (m)", "access_road": "Đường vào (m)",
    "floors": "Số tầng", "bedrooms": "Phòng ngủ", "bathrooms": "Phòng tắm", "price": "Giá (tỷ đồng)",
    "house_direction": "Hướng nhà", "balcony_direction": "Hướng ban công", "legal_status": "Pháp lý",
    "furniture_state": "Nội thất", "district": "Quận / huyện", "ward": "Phường / xã", "address": "Địa chỉ",
    "price_per_m2": "Đơn giá (triệu/m²)",
}
NUM_COLS = ["area", "frontage", "access_road", "floors", "bedrooms", "bathrooms", "price"]
UNK = "Không rõ"


def show(fn, *args, **kwargs):
    try:
        return fn(*args, use_container_width=True, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def plot(fig, height=None):
    """Render biểu đồ Plotly đồng bộ Dark Mode."""
    fig.update_layout(
        template="plotly_dark",
        height=height or H(),
        margin=dict(l=12, r=16, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", size=13 if PRESENT else 12, color="#e2e8f0"),
        title_font=dict(size=15, color="#f8fafc"),
        colorway=COLORWAY,
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#2a3950", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#2a3950", zerolinecolor="#334155")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def card(label, value, hint="", accent=False):
    cls = "info-card accent" if accent else "info-card"
    st.markdown(
        f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True
    )


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def page_header(title, sub):
    st.markdown(f"<h1 style='margin-bottom:4px; color:#f8fafc; font-weight:800;'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def money(v):
    return "—" if v is None or pd.isna(v) else f"{v:,.2f} tỷ"


# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================
def _candidates(*names):
    bases = (ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data", ROOT)
    return [b / n for n in names for b in bases]


DATASETS = {
    "raw": {"label": "Dữ liệu gốc (raw)",
            "files": _candidates("_HCM__vietnam_housing_dataset.csv", "[HCM] vietnam_housing_dataset.csv")},
    "clean": {"label": "Đã làm sạch (clean)", "files": _candidates("hcm_housing_clean.csv")},
    "filtered": {"label": "Đã lọc ngoại lai (filtered)", "files": _candidates("hcm_housing_filtered.csv")},
    "train": {"label": "Tập Train", "files": _candidates("train.csv")},
    "dev": {"label": "Tập Dev (Validation)", "files": _candidates("dev.csv")},
    "test": {"label": "Tập Test", "files": _candidates("test.csv")},
}


def find_path(key):
    for p in DATASETS[key]["files"]:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_original(key):
    p = find_path(key)
    return pd.read_csv(p) if p else None


@st.cache_data(show_spinner=False)
def get_df(key):
    df = load_original(key)
    if df is None:
        return None
    df = df.copy()
    df.columns = [c.strip().lower().replace(" / ", "_").replace(" ", "_") for c in df.columns]
    df = df.dropna(axis=1, how="all")
    if "address" in df.columns:
        parts = df["address"].astype(str).str.split(",")
        if "district" not in df.columns:
            df["district"] = parts.map(lambda p: p[-2].strip() if len(p) >= 2 else np.nan)
        if "ward" not in df.columns:
            df["ward"] = parts.map(lambda p: p[-3].strip() if len(p) >= 3 else np.nan)
    return df


def first_available(*keys):
    for k in keys:
        d = get_df(k)
        if d is not None:
            return d
    return None


def n_rows(key):
    d = get_df(key)
    return f"{len(d):,}" if d is not None else "—"


def missing_pct(df, cols=None):
    cols = [c for c in (cols or df.columns) if c in df.columns and c != "address"]
    s = (df[cols].isna().mean() * 100).sort_values(ascending=False)
    return s[s > 0]


def need_data(df):
    if df is None:
        st.warning("⚠️ Chưa tìm thấy file CSV. Hãy đặt các file cùng thư mục với `app.py` hoặc trong `data/raw`, `data/processed`.")
        st.stop()
    return df


# ============================================================
# TRANG 1: TỔNG QUAN
# ============================================================
def page_overview():
    df = need_data(first_available("filtered", "clean", "raw"))
    n_dist = df["district"].nunique()
    st.markdown(f"""
    <div class="hero">
        <h1>🏙️ {APP_TITLE}</h1>
        <p>Hệ thống phân tích thị trường bất động sản và mô hình hóa dự đoán giá nhà phố bằng Machine Learning.</p>
        <span class="chip">📊 {len(df):,} bất động sản</span>
        <span class="chip">📍 {n_dist} quận / huyện</span>
        <span class="chip">🧩 {len(df.columns) - 1} đặc trưng đầu vào</span>
        <span class="chip">💰 Đơn vị: Tỷ VNĐ</span>
    </div>""", unsafe_allow_html=True)

    c = st.columns(4)
    with c[0]: card("Giá trung vị", money(df["price"].median()), "50% căn nhà có giá thấp hơn mức này", accent=True)
    with c[1]: card("Biên độ giá", f'{df["price"].min():.1f} – {df["price"].max():.1f} tỷ', "Thấp nhất đến cao nhất")
    with c[2]: card("Diện tích trung vị", f'{df["area"].median():.0f} m²', f'Từ {df["area"].min():.0f} m² đến {df["area"].max():.0f} m²')
    with c[3]: card("Đơn giá trung vị", f'{(df["price"] * 1000 / df["area"]).median():.0f} tr/m²', "Tính theo giá ÷ diện tích")

    section("Pipeline xử lý & mô hình hóa")
    raw_o, clean_o = load_original("raw"), load_original("clean")
    raw_desc = f"{raw_o.shape[1]} cột · {int(raw_o.isna().all().sum())} cột rỗng" if raw_o is not None else "Chưa có file"
    clean_desc = f"{clean_o.shape[1]} cột · tách quận/phường" if clean_o is not None else "Chưa có file"
    removed = (len(get_df("clean")) - len(get_df("filtered"))) if get_df("clean") is not None and get_df("filtered") is not None else None
    filt_desc = f"Loại {removed:,} mẫu ngoại lai" if removed is not None else "Chưa có file"
    items = [
        ("BƯỚC 1", "Dữ liệu gốc", n_rows("raw"), raw_desc, ""),
        ("BƯỚC 2", "Làm sạch", n_rows("clean"), clean_desc, ""),
        ("BƯỚC 3", "Lọc ngoại lai", n_rows("filtered"), filt_desc, ""),
        ("BƯỚC 4", "Chia tập dữ liệu", f'{n_rows("train")} / {n_rows("dev")} / {n_rows("test")}', "Train · Dev · Test", ""),
        ("BƯỚC 5", "Mô hình dự đoán", "Chờ tích hợp", "Đang huấn luyện trên Notebook", "soon"),
    ]
    cells = [
        f'<div class="flow-card {s}"><div class="tag">{t}</div><div class="ttl">{ti}</div>'
        f'<div class="cnt">{cnt}</div><div class="dsc">{d}</div></div>' for t, ti, cnt, d, s in items
    ]
    st.markdown('<div class="flow">' + '<div class="arrow">›</div>'.join(cells) + "</div>", unsafe_allow_html=True)

    section("Bức tranh phân hóa giá theo khu vực")
    g = df.groupby("district")["price"].agg(median="median", n="size")
    g = g[g["n"] >= 30].sort_values("median")
    left, right = st.columns([1.75, 1])
    with left:
        with st.container(border=True):
            fig = px.bar(g.reset_index(), x="median", y="district", orientation="h", color="median",
                         color_continuous_scale="Blues", title="Giá trung vị theo quận / huyện (tỷ đồng, ≥ 30 mẫu)",
                         labels={"median": "Giá trung vị (tỷ đồng)", "district": ""})
            fig.update_traces(texttemplate="%{x:.1f} tỷ", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False)
            plot(fig, H(500))
    with right:
        corr = df[NUM_COLS].corr()["price"].drop("price").sort_values(ascending=False)
        top_feat = corr.index[0]
        card("🏆 Khu vực giá cao nhất", g.index[-1], f"Trung vị đạt {g['median'].iloc[-1]:.2f} tỷ đồng")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        card("🌱 Khu vực giá mềm nhất", g.index[0], f"Trung vị ở mức {g['median'].iloc[0]:.2f} tỷ đồng")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        card("📈 Yếu tố tác động mạnh nhất", FEATURE_VI[top_feat].split(" (")[0],
             f"Tương quan tuyến tính với giá: r = {corr.iloc[0]:.2f}", accent=True)


# ============================================================
# TRANG 2: DỮ LIỆU & TIỀN XỬ LÝ
# ============================================================
def page_data():
    page_header("🧹 Dữ liệu & tiền xử lý", "Theo dõi sự thay đổi của dữ liệu qua từng bước làm sạch và phân chia.")
    t1, t2, t3 = st.tabs(["1 · Raw → Clean", "2 · Lọc giá trị bất thường", "3 · Chia Train / Dev / Test"])

    with t1:
        ro, co = need_data(load_original("raw")), need_data(load_original("clean"))

        def stats(d):
            empty = d.isna().all()
            return {
                "Số dòng": f"{len(d):,}", "Số cột": d.shape[1],
                "Cột rỗng hoàn toàn": int(empty.sum()),
                "Ô dữ liệu thiếu": f"{int(d.loc[:, ~empty].isna().sum().sum()):,}",
                "Dòng trùng lặp": int(d.duplicated().sum()),
            }

        comp = pd.DataFrame({"Dữ liệu gốc (Raw)": stats(ro), "Sau làm sạch (Clean)": stats(co)})
        c = st.columns(3)
        with c[0]: card("Số lượng mẫu", f"{len(ro):,} → {len(co):,}", f"Loại bỏ {len(ro) - len(co):,} dòng lỗi/trùng", accent=True)
        with c[1]: card("Số lượng thuộc tính", f"{ro.shape[1]} → {co.shape[1]}", f"Xóa {int(ro.isna().all().sum())} cột rỗng hoàn toàn")
        with c[2]: card("Dòng trùng lặp", f"{int(ro.duplicated().sum())} → {int(co.duplicated().sum())}", "Đã khử trùng lặp hoàn toàn")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1.3])
        with left:
            with st.container(border=True):
                st.markdown("#### Bảng đối chiếu tổng quát")
                show(st.dataframe, comp.astype(str), height=230)
                empty_cols = ro.columns[ro.isna().all()].tolist()
                if empty_cols:
                    with st.expander(f"Xem {len(empty_cols)} cột rỗng đã loại bỏ"):
                        st.write(", ".join(empty_cols))
        with right:
            with st.container(border=True):
                mp = missing_pct(get_df("clean"), [c for c in NUM_COLS + ["house_direction", "legal_status", "furniture_state"]])
                fig = px.bar(mp.rename(index=FEATURE_VI).reset_index(), x=0, y="index", orientation="h",
                             title="Tỷ lệ khuyết thiếu dữ liệu sau làm sạch (%)", labels={"0": "% thiếu", "index": ""})
                fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(range=[0, max(100, mp.max() + 12)])
                plot(fig, H(290))

        with st.expander("🔍 So sánh mẫu dữ liệu thực tế (8 dòng đầu)"):
            a, b = st.columns(2)
            with a:
                st.caption("Dữ liệu gốc (đã ẩn cột rỗng)")
                show(st.dataframe, ro.dropna(axis=1, how="all").head(8), height=250)
            with b:
                st.caption("Dữ liệu sau làm sạch")
                show(st.dataframe, co.head(8), height=250)

    with t2:
        cl, fi = need_data(get_df("clean")), need_data(get_df("filtered"))
        c = st.columns(3)
        with c[0]: card("Trước khi lọc", f"{len(cl):,}", "Dòng dữ liệu sạch")
        with c[1]: card("Sau khi lọc", f"{len(fi):,}", "Dòng dữ liệu chuẩn", accent=True)
        with c[2]: card("Tỷ lệ ngoại lai đã loại", f"{len(cl) - len(fi):,}", f"Chiếm {(len(cl) - len(fi)) / len(cl):.1%} tổng mẫu")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1.25])
        with left:
            with st.container(border=True):
                st.markdown("#### Biên độ giá trị trước & sau lọc")
                rng = pd.DataFrame({
                    "Thuộc tính": [FEATURE_VI[c] for c in NUM_COLS],
                    "Min (Clean)": [cl[c].min() for c in NUM_COLS], "Max (Clean)": [cl[c].max() for c in NUM_COLS],
                    "Min (Filtered)": [fi[c].min() for c in NUM_COLS], "Max (Filtered)": [fi[c].max() for c in NUM_COLS],
                })
                show(st.dataframe, rng.round(2), hide_index=True, height=295)
        with right:
            with st.container(border=True):
                both = pd.concat([cl.assign(Tập="Clean"), fi.assign(Tập="Filtered")])
                fig = px.histogram(both, x="area", color="Tập", barmode="overlay", opacity=0.65, nbins=70,
                                   title="Phân phối diện tích trước và sau khi lọc ngoại lai",
                                   labels={"area": "Diện tích (m²)"})
                plot(fig, H(360))

    with t3:
        sets = {n: get_df(k) for n, k in [("Train", "train"), ("Dev", "dev"), ("Test", "test")]}
        if any(v is None for v in sets.values()):
            need_data(None)
        total = sum(len(v) for v in sets.values())
        c = st.columns(3)
        for col, (name, d) in zip(c, sets.items()):
            with col: card(f"Tập {name}", f"{len(d):,}", f"{len(d) / total:.1%} tổng số mẫu", accent=(name == "Train"))
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1.6])
        with left:
            with st.container(border=True):
                fig = px.pie(names=list(sets), values=[len(v) for v in sets.values()], hole=0.58,
                             title="Tỷ lệ phân chia dữ liệu")
                fig.update_traces(textinfo="percent+label", textposition="inside")
                fig.update_layout(showlegend=False)
                plot(fig, H(340))
        with right:
            with st.container(border=True):
                both = pd.concat([d.assign(Tập=n) for n, d in sets.items()])
                fig = px.histogram(both, x="price", color="Tập", barmode="overlay", opacity=0.5, nbins=40,
                                   histnorm="probability density", title="Mật độ phân phối giá trên 3 tập",
                                   labels={"price": "Giá (tỷ đồng)"})
                plot(fig, H(340))

        summary = pd.DataFrame({
            n: {"Số mẫu": len(d), "Giá TB (tỷ)": d["price"].mean(), "Giá trung vị (tỷ)": d["price"].median(),
                "Độ lệch chuẩn": d["price"].std(), "Diện tích Max (m²)": d["area"].max()}
            for n, d in sets.items()}).T.round(2)
        show(st.dataframe, summary)

        filt = get_df("filtered")
        if filt is not None and total != len(filt):
            big = int(sum((d["area"] > filt["area"].max()).sum() for d in sets.values()))
            st.warning(
                f"⚠️ Tổng Train + Dev + Test = **{total:,}** dòng, lệch so với file filtered (**{len(filt):,}** dòng). "
                f"Có {big} mẫu diện tích > {filt['area'].max():.0f} m² chưa được lọc. Nên chia lại từ tập `filtered` trước khi train."
            )


# ============================================================
# TRANG 3: KHÁM PHÁ DỮ LIỆU
# ============================================================
def label_series(s):
    return s.round().astype(int).astype(str) if pd.api.types.is_numeric_dtype(s) else s.astype(str)


def page_explore():
    page_header("📊 Khám phá dữ liệu (EDA)", "Tương tác trực tiếp với bộ dữ liệu qua bộ lọc đa chiều bên dưới.")
    keys = [k for k in DATASETS if get_df(k) is not None]
    if not keys:
        need_data(None)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.2, 1.2])
        src = c1.selectbox("Nguồn dữ liệu", keys, format_func=lambda k: DATASETS[k]["label"],
                           index=keys.index("filtered") if "filtered" in keys else 0)
        base = get_df(src)
        sel = c2.multiselect("Lọc Quận / huyện", sorted(base["district"].dropna().unique()), placeholder="Tất cả khu vực")
        pmin, pmax = float(base["price"].min()), float(base["price"].max())
        amin, amax = float(base["area"].min()), float(base["area"].max())
        prange = c3.slider("Khoảng giá (tỷ)", pmin, pmax, (pmin, pmax), step=0.1)
        arange = c4.slider("Diện tích (m²)", amin, amax, (amin, amax), step=1.0)

    data = base[base["price"].between(*prange) & base["area"].between(*arange)].copy()
    if sel:
        data = data[data["district"].isin(sel)]
    if data.empty:
        st.warning("Không tìm thấy căn nhà nào khớp với bộ lọc hiện tại.")
        return
    data["price_per_m2"] = data["price"] * 1000 / data["area"]

    c = st.columns(4)
    with c[0]: card("Mẫu đang hiển thị", f"{len(data):,}", f"Chiếm {len(data) / len(base):.1%} tập dữ liệu", accent=True)
    with c[1]: card("Giá trung vị", money(data["price"].median()), f"Trung bình: {money(data['price'].mean())}")
    with c[2]: card("Diện tích trung vị", f'{data["area"].median():.0f} m²', f"Trung bình: {data['area'].mean():.0f} m²")
    with c[3]: card("Đơn giá trung vị", f'{data["price_per_m2"].median():.0f} tr/m²', "Mức giá trên mỗi m²")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["Phân phối biến", "Theo khu vực", "Tương quan với giá", "Ma trận & Khuyết thiếu", "Dữ liệu chi tiết"])

    with t1:
        num_opts = [c for c in NUM_COLS + ["price_per_m2"] if c in data.columns]
        feat = st.selectbox("Chọn thuộc tính cần khảo sát", num_opts, format_func=lambda c: FEATURE_VI[c], index=num_opts.index("price"))
        left, right = st.columns([2, 1])
        with left:
            with st.container(border=True):
                fig = px.histogram(data, x=feat, nbins=40, marginal="box", title=f"Biểu đồ phân phối: {FEATURE_VI[feat]}",
                                   labels=FEATURE_VI)
                plot(fig, H(410))
        with right:
            with st.container(border=True):
                st.markdown("#### Chỉ số thống kê")
                desc = data[feat].describe().rename({"count": "Số lượng", "mean": "Trung bình", "std": "Độ lệch chuẩn",
                                                     "min": "Nhỏ nhất", "25%": "Phân vị 25% (Q1)", "50%": "Trung vị (Q2)",
                                                     "75%": "Phân vị 75% (Q3)", "max": "Lớn nhất"}).round(2)
                show(st.dataframe, desc.to_frame("Giá trị"), height=335)

    with t2:
        agg = data.groupby("district").agg(median_price=("price", "median"), count=("price", "size"),
                                           ppm2=("price_per_m2", "median")).reset_index()
        c1, c2 = st.columns([2, 1])
        metric_label = c1.radio("Tiêu chí xếp hạng", ["Giá trung vị (tỷ đồng)", "Số lượng tin đăng", "Đơn giá trung vị (triệu/m²)"],
                                horizontal=True)
        min_n = c2.slider("Số mẫu tối thiểu/quận", 1, 100, 30 if len(agg) > 5 else 1)
        col = {"Giá trung vị (tỷ đồng)": "median_price", "Số lượng tin đăng": "count",
               "Đơn giá trung vị (triệu/m²)": "ppm2"}[metric_label]
        agg = agg[agg["count"] >= min_n].sort_values(col)
        if agg.empty:
            st.info("Không có quận/huyện nào đạt ngưỡng số mẫu tối thiểu.")
        else:
            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    fig = px.bar(agg, x=col, y="district", orientation="h", color=col, color_continuous_scale="Blues",
                                 title=metric_label, labels={col: metric_label, "district": ""})
                    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False)
                    fig.update_layout(coloraxis_showscale=False)
                    plot(fig, H(520))
            with right:
                with st.container(border=True):
                    top = agg.sort_values("count", ascending=False).head(12)["district"].tolist()
                    order = data[data["district"].isin(top)].groupby("district")["price"].median().sort_values().index.tolist()
                    fig = px.box(data[data["district"].isin(top)], x="price", y="district", orientation="h",
                                 category_orders={"district": order}, title="Biên độ giá — Top 12 quận nhiều tin nhất",
                                 labels={"price": "Giá (tỷ đồng)", "district": ""})
                    plot(fig, H(520))

    with t3:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                color_opts = {"Mặc định": None, "Tình trạng nội thất": "furniture_state", "Giấy tờ pháp lý": "legal_status",
                              "Hướng nhà": "house_direction"}
                cc = st.selectbox("Phân nhóm màu sắc theo", list(color_opts))
                sample = data.sample(min(len(data), 3000), random_state=42).copy()
                ccol = color_opts[cc]
                if ccol and ccol in sample.columns:
                    sample[ccol] = sample[ccol].fillna(UNK)
                else:
                    ccol = None
                fig = px.scatter(sample, x="area", y="price", color=ccol, opacity=0.65,
                                 title="Tương quan Diện tích & Giá (tối đa 3.000 điểm)", labels=FEATURE_VI)
                plot(fig, H(430))
        with right:
            with st.container(border=True):
                cat_opts = [c for c in ["floors", "bedrooms", "bathrooms", "house_direction", "legal_status", "furniture_state"]
                            if c in data.columns]
                cat = st.selectbox("Phân tích giá theo đặc điểm", cat_opts, format_func=lambda c: FEATURE_VI[c])
                d = data.dropna(subset=[cat]).copy()
                d["_x"] = label_series(d[cat])
                if pd.api.types.is_numeric_dtype(d[cat]):
                    order = [str(v) for v in sorted(d[cat].round().astype(int).unique())]
                else:
                    order = d.groupby("_x")["price"].median().sort_values().index.tolist()
                fig = px.box(d, x="_x", y="price", category_orders={"_x": order},
                             title=f"Phân phối giá theo {FEATURE_VI[cat].lower()}", labels={"_x": FEATURE_VI[cat], "price": "Giá (tỷ đồng)"})
                plot(fig, H(430))

    with t4:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                corr = data[NUM_COLS].corr().rename(index=FEATURE_VI, columns=FEATURE_VI)
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                title="Ma trận hệ số tương quan Pearson")
                plot(fig, H(440))
        with right:
            with st.container(border=True):
                mp = missing_pct(data, NUM_COLS + ["house_direction", "legal_status", "furniture_state"])
                if mp.empty:
                    st.success("✅ Tập dữ liệu đang xem không có giá trị khuyết thiếu.")
                else:
                    fig = px.bar(mp.rename(index=FEATURE_VI).reset_index(), x=0, y="index", orientation="h",
                                 title="Tỷ lệ dữ liệu thiếu (%)", labels={"0": "% thiếu", "index": ""})
                    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(range=[0, max(100, mp.max() + 12)])
                    plot(fig, H(440))

    with t5:
        show(st.dataframe, data.drop(columns=["price_per_m2"]).head(500), height=420)
        col1, col2 = st.columns([3, 1])
        col1.caption(f"Hiển thị 500 dòng đầu tiên trên tổng số {len(data):,} bản ghi thỏa điều kiện lọc.")
        col2.download_button("⬇️ Xuất file CSV", data.drop(columns=["price_per_m2"]).to_csv(index=False).encode("utf-8-sig"),
                             file_name="hcmc_housing_filtered_export.csv", mime="text/csv", use_container_width=True)


# ============================================================
# TRANG 4: HUẤN LUYỆN MÔ HÌNH (khung giao diện sẵn sàng kết nối)
# ============================================================
MODEL_FILES = {   # giữ giống api/api.py
    "XGBoost": "xgboost.pkl",
    "XGBoost Optimized": "xgboost_optimized_model.pkl",
    "Random Forest": "random_forest.pkl",
    "Random Forest (model 2)": "random_forest_model.pkl",
    "Linear Regression": "linear.pkl",
    "Linear Regression (Liner_Model)": "Liner_Model.pkl",
    "Ridge Regression": "ridge.pkl",
    "Lasso Regression": "lasso.pkl",
}
# Các file dự đoán thẳng giá (tỷ đồng); file còn lại dự đoán log1p(price)
RAW_PRICE_FILES = {"Liner_Model.pkl"}


def predict_price(model, file, df):
    """Chuẩn bị đúng cột mà từng model cần, trả về giá dự đoán (tỷ đồng)."""
    num_cols = ["area", "floors", "bedrooms", "bathrooms", "frontage", "access_road"]
    raw_cols = list(get_raw_columns())
    needed = list(getattr(model, "feature_names_in_", raw_cols))
    cols = list(dict.fromkeys(raw_cols + [c for c in needed if c in df.columns]))
    X = df.reindex(columns=cols)
    nums = [c for c in num_cols if c in X.columns]
    cats = [c for c in raw_cols if c not in num_cols]
    X[nums] = X[nums].astype(float)
    X[cats] = X[cats].astype(object).where(X[cats].notna(), np.nan)
    if any(c not in X.columns for c in needed):
        X = add_features(X.copy())   # tạo thêm các cột đặc trưng
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(missing)}")
    pred = np.asarray(model.predict(X[needed]), dtype=float)
    return pred if file in RAW_PRICE_FILES else np.expm1(pred)


def evaluate_models(names):
    dev = get_df("dev")
    if dev is None:
        st.warning("Chưa tìm thấy file dev.csv.")
        return None

    y = dev["price"].to_numpy()

    rows = []
    for name in names:
        file = MODEL_FILES.get(name)
        path = ROOT / "models" / file if file else None
        if path is None or not path.exists():
            st.warning(f"{name} chưa có model đã lưu để đánh giá.")
            continue
        try:
            pred = predict_price(joblib.load(path), file, dev)
            err = y - pred
            rows.append({
                "Mô hình": name,
                "MAE": np.mean(np.abs(err)),
                "RMSE": np.sqrt(np.mean(err ** 2)),
                "R²": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
            })
        except Exception as e:
            st.warning(f"Không đánh giá được {name}: {e}")

    return pd.DataFrame(rows).sort_values("RMSE") if rows else None

def page_train():
    page_header("Đánh giá mô hình", "So sánh hiệu năng các model đã lưu trên tập dữ liệu bất động sản.")

    left, right = st.columns([1, 1.65])
    with left:
        with st.container(border=True):
            st.markdown("#### ⚙️ Thiết lập đánh giá")
            st.info(f"📊 **Train:** {n_rows('train')} | **Dev:** {n_rows('dev')} | **Test:** {n_rows('test')} mẫu")
            available = [n for n, f in MODEL_FILES.items() if (ROOT / "models" / f).exists()]
            models = st.multiselect("Chọn các model so sánh", available, default=available)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("📊 Đánh giá trên tập Dev", disabled=not models, type="primary", use_container_width=True):
                with st.spinner("Đang đánh giá..."):
                    st.session_state["model_results"] = evaluate_models(models)
            st.caption("💡 Chỉ dùng các model đã lưu trong `models/`, không huấn luyện lại.")

    results = st.session_state.get("model_results")

    with right:
        st.markdown("#### 📈 Bảng tổng hợp kết quả")
        if results is None:
            c = st.columns(3)
            for col, name, unit in zip(c, ["MAE", "RMSE", "R² Score"], ["Sai số tuyệt đối", "Sai số toàn phương", "Độ phù hợp"]):
                with col: card(name, "—", unit)
            st.markdown("""
            <div class="empty-box">
                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:4px;">⏳ Chưa có kết quả đánh giá</div>
                Bấm <b>Đánh giá trên tập Dev</b> để xem bảng xếp hạng và biểu đồ so sánh RMSE.
            </div>
            """, unsafe_allow_html=True)
            skeleton = pd.DataFrame({
                "Mô hình": models or ["Chưa chọn mô hình"],
                "MAE (tỷ)": "—", "RMSE (tỷ)": "—", "R²": "—", "Trạng thái": "Chưa đánh giá"
            })
            show(st.dataframe, skeleton, hide_index=True)
        else:
            best = results.sort_values("RMSE").iloc[0]
            c = st.columns(3)
            with c[0]: card("Mô hình tối ưu", str(best["Mô hình"]), "RMSE thấp nhất trên tập Dev", accent=True)
            with c[1]: card("RMSE tốt nhất", f'{best["RMSE"]:.3f} tỷ', "Căn bậc hai sai số bình phương TB")
            with c[2]: card("Hệ số R²", f'{best["R²"]:.3f}', "Mức độ giải thích phương sai")
            show(st.dataframe, results, hide_index=True)
            fig = px.bar(results, x="Mô hình", y="RMSE", color="RMSE", color_continuous_scale="Blues_r",
                         title="So sánh sai số RMSE giữa các model")
            plot(fig, H(300))

    section("Danh sách đặc trưng đầu vào (Features)")
    tr = get_df("train")
    if tr is None:
        st.info("Chưa tìm thấy file `train.csv`.")
        return
    feats = [c for c in tr.columns if c not in ("price", "address")]
    table = pd.DataFrame({
        "Tên đặc trưng": [FEATURE_VI.get(c, c) for c in feats],
        "Biến gốc": feats,
        "Kiểu dữ liệu": ["Số học (Numeric)" if pd.api.types.is_numeric_dtype(tr[c]) else "Phân loại (Categorical)" for c in feats],
        "Tỷ lệ thiếu (%)": [round(tr[c].isna().mean() * 100, 1) for c in feats],
        "Số giá trị duy nhất": [tr[c].nunique() for c in feats],
    })
    show(st.dataframe, table, hide_index=True)


# ============================================================
# TRANG 5: DỰ ĐOÁN GIÁ NHÀ
# ============================================================
LEGAL_VI = {"Have certificate": "Đã có sổ hồng / sổ đỏ", "Sale contract": "Hợp đồng mua bán"}
FURN_VI = {"Full": "Nội thất đầy đủ", "Basic": "Nội thất cơ bản"}
DIR_ORDER = ["Đông", "Tây", "Nam", "Bắc", "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"]

PRESETS = {
    "🏘️ Nhà phố ngoại thành": dict(p_district="Hóc Môn", p_area=50.0, p_frontage=4.0, p_access=4.0, p_floors=2, p_beds=2, p_baths=2),
    "🏠 Nhà ở gia đình": dict(p_district="Gò Vấp", p_area=75.0, p_frontage=4.5, p_access=6.0, p_floors=3, p_beds=4, p_baths=3),
    "🏙️ Nhà trung tâm": dict(p_district="Quận 3", p_area=60.0, p_frontage=4.5, p_access=8.0, p_floors=4, p_beds=4, p_baths=4),
}


def apply_preset(values):
    for k, v in values.items():
        st.session_state[k] = v


def call_api(method, endpoint, **kwargs):
    """Gọi API và trả về JSON. Báo lỗi bằng RuntimeError với nội dung dễ hiểu."""
    try:
        with requests.Session() as session:
            session.trust_env = False   # gọi thẳng API trên máy, không qua proxy
            res = session.request(method, API_URL + endpoint, timeout=60, **kwargs)
    except requests.RequestException as e:
        raise RuntimeError(f"Không kết nối được API tại {API_URL}. Hãy kiểm tra api.py đã chạy chưa. ({e})")
    if not res.ok:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        raise RuntimeError(f"API báo lỗi {res.status_code}: {detail}")
    return res.json()


def page_predict():
    page_header("🏡 Định giá bất động sản", "Nhập thông số căn nhà để mô hình ước tính giá trị thị trường.")
    df = need_data(first_available("filtered", "clean", "raw"))
    districts = sorted(df["district"].dropna().unique())
    legal_opts = [UNK] + [v for v in LEGAL_VI if v in set(df["legal_status"].dropna())]
    furn_opts = [UNK] + [v for v in FURN_VI if v in set(df["furniture_state"].dropna())]
    dir_opts = [UNK] + [d for d in DIR_ORDER if d in set(df["house_direction"].dropna())]

    st.markdown("**⚡ Điền nhanh kịch bản mẫu:**")
    pc = st.columns([1, 1, 1, 1.5])
    for col, (name, vals) in zip(pc[:3], PRESETS.items()):
        with col:
            st.button(name, on_click=apply_preset, args=(vals,), key=f"preset_{name}", use_container_width=True)

    models, model_name = [], None
    try:
        models = call_api("GET", "/models")["models"]
    except RuntimeError as e:
        st.error(str(e))
    else:
        if models:
            model_name = st.selectbox("🤖 Chọn mô hình dự đoán", models, key="p_model")
            st.caption(f"🟢 Đã kết nối API: `{API_URL}`")
        else:
            st.warning("API chưa tìm thấy file model nào trong thư mục models/.")

    form_col, result_col = st.columns([1.45, 1])

    with form_col:
        with st.container(border=True):
            st.markdown("#### 📍 1. Vị trí & Hướng")
            c1, c2, c3 = st.columns(3)
            district = c1.selectbox("Quận / huyện *", districts, key="p_district",
                                    index=districts.index("Gò Vấp") if "Gò Vấp" in districts else 0)
            wards = sorted(df.loc[df["district"] == district, "ward"].dropna().unique())
            ward = c2.selectbox("Phường / xã", [UNK] + wards, key=f"p_ward_{district}")
            direction = c3.selectbox("Hướng nhà", dir_opts, key="p_dir")

            st.markdown("#### 📐 2. Kích thước & Kết cấu")
            c1, c2, c3 = st.columns(3)
            area = c1.number_input("Diện tích đất (m²) *", min_value=10.0, max_value=300.0, value=60.0, step=1.0, key="p_area")
            frontage = c2.number_input("Mặt tiền (m)", min_value=1.0, max_value=30.0, value=None, step=0.5,
                                       placeholder="Để trống nếu không rõ", key="p_frontage")
            access = c3.number_input("Đường trước nhà (m)", min_value=1.0, max_value=80.0, value=None, step=0.5,
                                     placeholder="Để trống nếu không rõ", key="p_access")

            c1, c2, c3 = st.columns(3)
            floors = c1.number_input("Số tầng", min_value=1, max_value=10, value=3, step=1, key="p_floors")
            beds = c2.number_input("Số phòng ngủ", min_value=1, max_value=12, value=3, step=1, key="p_beds")
            baths = c3.number_input("Số phòng tắm", min_value=1, max_value=12, value=3, step=1, key="p_baths")

            st.markdown("#### 📜 3. Pháp lý & Nội thất")
            c1, c2 = st.columns(2)
            legal = c1.selectbox("Giấy tờ pháp lý", legal_opts, format_func=lambda x: LEGAL_VI.get(x, x), key="p_legal")
            furn = c2.selectbox("Tình trạng nội thất", furn_opts, format_func=lambda x: FURN_VI.get(x, x), key="p_furn")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            clicked = st.button("🔍 Tính toán giá ước tính", type="primary", key="p_go", use_container_width=True)

    unk = lambda v: None if v == UNK else v
    payload = {
        "area": area, "frontage": frontage, "access_road": access, "house_direction": unk(direction),
        "floors": floors, "bedrooms": beds, "bathrooms": baths, "legal_status": unk(legal),
        "furniture_state": unk(furn), "district": district, "ward": unk(ward),
    }

    with result_col:
        if clicked and model_name:
            try:
                result = call_api("POST", "/predict", json={**payload, "model": model_name})
                pred = result["predicted_price"]
                big, hint = f"{pred:,.2f} tỷ", f"Đơn giá ước tính: ~{pred * 1000 / area:,.0f} triệu/m²"
            except RuntimeError as e:
                big, hint = "Lỗi dự đoán", str(e)
        elif clicked:
            big, hint = "Chưa dự đoán được", "Hãy kiểm tra API và thư mục models/"
        else:
            big, hint = "— tỷ", "Nhấn nút bên trái để thực hiện định giá"

        st.markdown(f"""
        <div class="result-card">
            <div class="label">KẾT QUẢ ĐỊNH GIÁ · {model_name or "CHƯA CÓ MÔ HÌNH"}</div>
            <div class="value">{big}</div>
            <div class="hint">{hint}</div>
        </div>
        """, unsafe_allow_html=True)

        # Khối tham chiếu thị trường
        sim = df[(df["district"] == district) & df["area"].between(area * 0.75, area * 1.25)]
        scope = f"{district} (diện tích {area * 0.75:.0f}–{area * 1.25:.0f} m²)"
        if len(sim) < 20:
            sim, scope = df[df["district"] == district], f"toàn khu vực {district}"

        with st.container(border=True):
            st.markdown("#### 📊 Đối chiếu thị trường thực tế")
            st.caption(f"Dữ liệu giao dịch tương đồng tại **{scope}**:")
            if sim.empty:
                st.info("Chưa có dữ liệu tham chiếu cho khu vực này.")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Số căn tương đồng", f"{len(sim):,} căn")
                m2.metric("Giá trung vị khu vực", money(sim["price"].median()))
                st.caption(f"Khoảng giá phổ biến (Q1–Q3): **{sim['price'].quantile(.25):.2f} – {sim['price'].quantile(.75):.2f} tỷ**")
                fig = px.histogram(sim, x="price", nbins=22, labels={"price": "Giá giao dịch (tỷ đồng)"})
                fig.add_vline(x=float(sim["price"].median()), line_dash="dash", line_color="#f59e0b",
                              annotation_text="Trung vị", annotation_position="top right")
                fig.update_layout(showlegend=False, yaxis_title="Số căn", margin=dict(t=25, b=10, l=10, r=10))
                plot(fig, H(220))


# ============================================================
# SIDEBAR + ĐIỀU HƯỚNG
# ============================================================
PAGES = {
    "🏠  Tổng quan dự án": page_overview,
    "🧹  Dữ liệu & Tiền xử lý": page_data,
    "📊  Khám phá dữ liệu": page_explore,
    "🧠  Đánh giá mô hình": page_train,
    "🏡  Dự đoán giá nhà": page_predict,
}

with st.sidebar:
    st.markdown("## 🏙️ HCMC HOUSING")
    st.markdown("---")
    page = st.radio("ĐIỀU HƯỚNG", list(PAGES), label_visibility="collapsed")
    st.markdown("---")


    with st.expander("📁 Trạng thái dữ liệu", expanded=False):
        for k, v in DATASETS.items():
            st.markdown(f"{'🟢' if find_path(k) else '⚪'} {v['label']}")

    with st.expander("🎓 Thông tin đồ án", expanded=True):
        for k, v in PROJECT_INFO.items():
            st.markdown(f"**{k}:** {v}")


PAGES[page]()