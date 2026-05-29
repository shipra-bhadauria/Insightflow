import streamlit as st
import os


def render_left_panel():

    st.markdown("""
    <p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:8px">Workspace</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        import pandas as pd
        from tools.quality        import run_quality_report
        from tools.detect_columns import detect_columns

        save_path = os.path.join("data", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(save_path)
            st.session_state.df               = df
            st.session_state.dataset_path     = save_path
            st.session_state.quality_report   = run_quality_report(df)
            st.session_state.detected_columns = detect_columns(df)
            st.session_state.source_type      = "csv"
            st.session_state.context          = ""
            

        elif uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(save_path)
            st.session_state.df               = df
            st.session_state.dataset_path     = save_path
            st.session_state.quality_report   = run_quality_report(df)
            st.session_state.detected_columns = detect_columns(df)
            st.session_state.source_type      = "excel"
            st.session_state.context          = ""
            

        elif uploaded.name.endswith(".pdf"):
            from tools.pdf_loader import load_pdf
            result = load_pdf(save_path)
            st.session_state.df               = None
            st.session_state.dataset_path     = save_path
            st.session_state.source_type      = "pdf"
            st.session_state.context          = result["full_text"]
            st.session_state.quality_report   = {
                "total_rows":         result["pages_loaded"],
                "total_columns":      0,
                "health_label":       "PDF",
                "completeness_score": 1.0,
                "null_counts":        {},
                "column_types":       {},
            }
            st.session_state.detected_columns = {
                "date_col":     None,
                "value_col":    None,
                "category_col": None,
            }
           

        elif uploaded.name.endswith((".png", ".jpg", ".jpeg")):
            from tools.image_loader import load_image
            result = load_image(save_path)
            st.session_state.df               = None
            st.session_state.dataset_path     = save_path
            st.session_state.source_type      = "image"
            st.session_state.context          = result["raw_text"]
            st.session_state.quality_report   = {
                "total_rows":         1,
                "total_columns":      0,
                "health_label":       "IMAGE",
                "completeness_score": 1.0,
                "null_counts":        {},
                "column_types":       {},
            }
            st.session_state.detected_columns = {
                "date_col":     None,
                "value_col":    None,
                "category_col": None,
            }
          

    # show active file
    if st.session_state.dataset_path:
        fname   = os.path.basename(st.session_state.dataset_path)
        quality = st.session_state.quality_report
        source  = st.session_state.get("source_type", "csv").upper()

        st.markdown(f"""
        <div style="background:#0a0e0a;border:1px solid #1a2a1a;
        border-radius:8px;padding:10px;margin-bottom:8px">
            <div style="font-size:11px;color:#8aaa8a;font-weight:500">{fname}</div>
            <div style="font-size:9px;color:#3a5a3a;margin-top:2px">
            {quality['total_rows']} rows · {quality['total_columns']} cols</div>
            <span style="font-size:8px;background:#0f1f0f;border:1px solid #1e3a1e;
            border-radius:3px;padding:1px 5px;color:#4a7a4a">{source}</span>
        </div>
        """, unsafe_allow_html=True)

        score = quality["completeness_score"] * 100
        color = "#6ab46a" if score >= 90 else "#e2a84a" if score >= 75 else "#d45a5a"
        st.markdown(f"""
        <div style="background:#0a0e0a;border:1px solid #1a2a1a;
        border-radius:8px;padding:10px;margin-bottom:8px">
            <div style="font-size:9px;color:#3a5a3a;margin-bottom:4px">DATA HEALTH</div>
            <div style="font-size:14px;font-weight:600;color:{color}">
            {quality['health_label'].upper()}</div>
            <div style="height:3px;background:#1a2a1a;border-radius:2px;margin-top:6px">
                <div style="width:{score}%;height:100%;background:{color};border-radius:2px"></div>
            </div>
            <div style="font-size:9px;color:#3a5a3a;margin-top:3px">{score:.0f}% complete</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.detected_columns:
            detected = st.session_state.detected_columns
            has_cols = any(detected.get(k) for k in ["date_col", "value_col", "category_col"])
            if has_cols:
                st.markdown("""
                <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                color:#3a5a3a;margin-bottom:6px;margin-top:8px">Detected columns</div>
                """, unsafe_allow_html=True)
                for key in ["date_col", "value_col", "category_col"]:
                    val = detected.get(key)
                    label = key.replace("_col", "")
                    if val:
                        st.markdown(f"""
                        <div style="font-size:9px;color:#4a6a4a;margin-bottom:3px">
                        {label} → <span style="color:#8acc8a">{val}</span></div>
                        """, unsafe_allow_html=True)

        # show PDF/image context info
        if st.session_state.get("source_type") in ["pdf", "image"]:
            context = st.session_state.get("context", "")
            st.markdown(f"""
            <div style="font-size:9px;color:#3a5a3a;margin-top:6px">
            {len(context)} characters extracted</div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:8px">Past Analyses</p>
    """, unsafe_allow_html=True)

    if st.session_state.past_analyses:
        for i, analysis in enumerate(reversed(st.session_state.past_analyses[-5:])):
            st.markdown(f"""
            <div style="background:#0a0e0a;border:1px solid #1a2a1a;
            border-radius:6px;padding:8px;margin-bottom:6px;cursor:pointer">
                <div style="font-size:10px;color:#8aaa8a">{analysis['question'][:40]}...</div>
                <div style="font-size:9px;color:#3a5a3a;margin-top:2px">
                conf: {analysis['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:10px;color:#2a4a2a;font-style:italic">
        No analyses yet</div>
        """, unsafe_allow_html=True)