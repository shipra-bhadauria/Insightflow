import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from state                import new_state
from tools.quality        import run_quality_report
from tools.detect_columns import detect_columns
from graph.workflow       import workflow

load_dotenv()

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="InsightFlow",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global styles ─────────────────────────────────────────────
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #0a0e0a; color: #e8f0e8; }

    /* hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* sidebar */
    [data-testid="stSidebar"] { background-color: #0c110c; }

    /* inputs */
    .stTextArea textarea {
        background-color: #0f150f !important;
        color: #c8e0c8 !important;
        border: 1px solid #2a3e2a !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* buttons */
    .stButton button {
        background-color: #c5f432 !important;
        color: #0a0e0a !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }

    /* metric cards */
    [data-testid="stMetric"] {
        background-color: #0c110c;
        border: 1px solid #1e2a1e;
        border-radius: 8px;
        padding: 12px;
    }

    /* divider */
    hr { border-color: #1e2a1e !important; }

    /* file uploader */
    [data-testid="stFileUploader"] {
        background-color: #0c110c;
        border: 1px dashed #2a3e2a;
        border-radius: 8px;
    }
    /* file uploader button */
[data-testid="stFileUploaderDropzone"] {
    background-color: #0f150f !important;
    border: 1px dashed #2a4a2a !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #0f1f0f !important;
    color: #c5f432 !important;
    border: 1px solid #2a4a2a !important;
}

[data-testid="stFileUploaderDropzone"] p {
    color: #4a6a4a !important;
}
            
/* metric cards */
[data-testid="stMetric"] {
    background-color: #0f1f0f !important;
    border: 1px solid #2a4a2a !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #c5f432 !important;
    font-size: 20px !important;
}

[data-testid="stMetricLabel"] {
    color: #4a6a4a !important;
    font-size: 10px !important;
}

[data-testid="stMetricDelta"] {
    color: #6ab46a !important;
    font-size: 9px !important;
}
</style>
""", unsafe_allow_html=True)


# ── session state initialisation ──────────────────────────────
if "df"               not in st.session_state: st.session_state.df               = None
if "dataset_path"     not in st.session_state: st.session_state.dataset_path     = None
if "source_type" not in st.session_state:
    st.session_state.source_type = "csv"
if "context" not in st.session_state:
    st.session_state.context = ""
if "quality_report"   not in st.session_state: st.session_state.quality_report   = None
if "detected_columns" not in st.session_state: st.session_state.detected_columns = None
if "final_state"      not in st.session_state: st.session_state.final_state      = None
if "running"          not in st.session_state: st.session_state.running          = False
if "past_analyses"    not in st.session_state: st.session_state.past_analyses    = []
if "pdf_path" not in st.session_state: st.session_state.pdf_path = None

# ── top navigation bar ────────────────────────────────────────
col_logo, col_title, col_file = st.columns([1, 4, 2])

with col_logo:
    st.markdown("""
    <div style="background:#c5f432;width:36px;height:36px;border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-weight:800;font-size:16px;color:#0a0e0a;margin-top:4px">I</div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <div style="margin-top:4px">
        <span style="color:#c5f432;font-size:20px;font-weight:700;
        font-family:JetBrains Mono,monospace">InsightFlow</span>
        <span style="color:#3a5a3a;font-size:11px;margin-left:12px">
        Plan · Execute · Critique — an analyst agent that checks its own work</span>
    </div>
    """, unsafe_allow_html=True)

with col_file:
    if st.session_state.dataset_path:
        fname = os.path.basename(st.session_state.dataset_path)
        nrows = len(st.session_state.df) if st.session_state.df is not None else 0
        st.markdown(f"""
        <div style="background:#111a11;border:1px solid #1e2a1e;border-radius:6px;
        padding:6px 12px;font-size:11px;color:#6a8a6a;margin-top:6px;text-align:right">
        🟢 {fname} · {nrows} rows</div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── three panel layout ────────────────────────────────────────
left, center, right = st.columns([0.8, 3.5, 1.2])


# ── LEFT PANEL ────────────────────────────────────────────────
with left:
    from ui.left_panel import render_left_panel
    render_left_panel()


# ── CENTER PANEL ──────────────────────────────────────────────
with center:
    from ui.center_panel import render_center_panel
    render_center_panel()


# ── RIGHT PANEL ───────────────────────────────────────────────
with right:
    from ui.right_panel import render_right_panel
    render_right_panel()