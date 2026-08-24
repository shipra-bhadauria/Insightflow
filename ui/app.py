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
    page_title="INSIGHTFLOW",
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

    /* chat sidebar past chat buttons — grey */
    [data-testid="column"] [data-testid="stButton"] button {
        background-color: #0f150f !important;
        color: #4a6a4a !important;
        border: 1px solid #1e2a1e !important;
        font-weight: 400 !important;
        font-size: 10px !important;
    }
    [data-testid="column"] [data-testid="stButton"] button:hover {
        background-color: #1a2a1a !important;
        color: #c5f432 !important;
        border-color: #2a4a2a !important;
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

    /* left right panel font */
    section[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stMarkdownContainer"] p {
        font-size: 14px !important;
        line-height: 1.8 !important;
    }
    div[data-testid="stMarkdownContainer"] div {
        font-size: 13px !important;
        line-height: 1.7 !important;
    }
    div[data-testid="stMarkdownContainer"] span {
        font-size: 13px !important;
    }

           /* tab colors */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c5f432 !important;
        border-bottom: 3px solid #c5f432 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: .08em !important;
        padding: 12px 24px !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"] {
        color: #3a5a3a !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: .08em !important;
        padding: 12px 20px !important;
        border-bottom: 3px solid transparent !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #6a8a6a !important;
        background: transparent !important;
    }
    div[role="tablist"] {
        border-bottom: 1px solid #1e2a1e !important;
    }

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
if "user_id"          not in st.session_state: st.session_state.user_id          = None
if "user_requests"    not in st.session_state: st.session_state.user_requests    = []


# ── simple username entry (no password) ──────────────────────
if not st.session_state.user_id:
    st.markdown("""
    <div style="text-align:center;margin-top:80px">
        <div style="background:#c5f432;width:56px;height:56px;border-radius:12px;
        display:flex;align-items:center;justify-content:center;margin:0 auto 20px;
        font-weight:800;font-size:26px;color:#0a0e0a">I</div>
        <span style="color:#c5f432;font-size:28px;font-weight:800;
        letter-spacing:.15em;font-family:JetBrains Mono,monospace">INSIGHTFLOW</span>
        <div style="color:#3a5a3a;font-size:12px;margin-top:8px">
        Enter a name to start your session — your data and chats stay private to you</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        name_input = st.text_input(
            "Your name",
            placeholder="e.g. shipra",
            label_visibility="collapsed",
            key="username_entry",
        )
        if st.button("Start →", use_container_width=True):
            clean = "".join(c for c in name_input.strip().lower() if c.isalnum() or c in ("-", "_"))
            if clean:
                st.session_state.user_id = clean
                st.rerun()
            else:
                st.warning("Please enter a valid name (letters/numbers only).")
    st.stop()


# ── top navigation bar ────────────────────────────────────────
col_logo, col_title, col_file = st.columns([1, 4, 2])

with col_logo:
    st.markdown("""
    <div style="background:#c5f432;width:36px;height:36px;border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-weight:800;font-size:16px;color:#0a0e0a;margin-top:4px">I</div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown(f"""
    <div style="margin-top:4px">
        <span style="color:#c5f432;font-size:26px;font-weight:800;
        letter-spacing:.15em;font-family:JetBrains Mono,monospace">INSIGHTFLOW</span>
        <span style="color:#3a5a3a;font-size:11px;margin-left:14px">
        Plan · Execute · Critique — an analyst agent that checks its own work</span>
        <span style="color:#4a6a4a;font-size:10px;margin-left:14px">
        👤 {st.session_state.user_id}</span>
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


# ── tab layout ────────────────────────────────────────
tab_analysis, tab_chat = st.tabs(["📊 ANALYSIS", "💬 CHAT"])

with tab_analysis:
    left, center, right = st.columns([1, 3.2, 1])

    with left:
        from ui.left_panel import render_left_panel
        render_left_panel()

    with center:
        from ui.center_panel import render_center_panel
        render_center_panel()

    with right:
        from ui.right_panel import render_right_panel
        render_right_panel()

with tab_chat:
    from ui.chat_panel import render_chat_panel
    render_chat_panel()
