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

        user_dir  = os.path.join("data", "users", st.session_state.get("user_id", "default"))
        os.makedirs(user_dir, exist_ok=True)
        save_path = os.path.join(user_dir, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # clear old cache when new file uploaded
        st.session_state.pop("file_overview", None)
        
        if uploaded.name.endswith(".csv"):
            try:
                df = pd.read_csv(save_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(save_path, encoding="latin-1")
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
                "date_col": None, "value_col": None, "category_col": None,
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
                "date_col": None, "value_col": None, "category_col": None,
            }

    # ── active file info ───────────────────────────────────────────────────────
    if st.session_state.get("dataset_path"):
        fname   = os.path.basename(st.session_state.dataset_path)
        quality = st.session_state.quality_report
        source  = st.session_state.get("source_type", "csv").upper()

        # file name + rows + cols
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

        # data health bar
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

        # ── detected columns — primary ────────────────────────────────────────
        if st.session_state.get("detected_columns"):
            detected = st.session_state.detected_columns
            has_cols = any(detected.get(k) for k in ["date_col", "value_col", "category_col"])

            if has_cols:
                st.markdown("""
                <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                color:#3a5a3a;margin-bottom:6px;margin-top:8px">Detected columns</div>
                """, unsafe_allow_html=True)

                for key, label in [("date_col", "date"), ("value_col", "value"),
                                    ("category_col", "primary category")]:
                    val = detected.get(key)
                    if val:
                        st.markdown(f"""
                        <div style="font-size:9px;color:#4a6a4a;margin-bottom:3px">
                        {label} → <span style="color:#8acc8a">{val}</span></div>
                        """, unsafe_allow_html=True)

            # ── all category columns ───────────────────────────────────────────
            all_cats = detected.get("all_category_cols", [])
            if len(all_cats) > 1:
                st.markdown("""
                <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                color:#3a5a3a;margin-bottom:6px;margin-top:10px">All categories</div>
                """, unsafe_allow_html=True)

                cats_html = "".join(
                    f'<span style="display:inline-block;background:#0f1f0f;'
                    f'border:1px solid #1e3a1e;border-radius:3px;padding:2px 6px;'
                    f'margin:2px;font-size:9px;color:#6ab46a">{cat}</span>'
                    for cat in all_cats
                )
                st.markdown(
                    f'<div style="margin-bottom:8px">{cats_html}</div>',
                    unsafe_allow_html=True
                )

            # ── all value columns ──────────────────────────────────────────────
            all_vals = detected.get("all_value_cols", [])
            if len(all_vals) > 1:
                st.markdown("""
                <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                color:#3a5a3a;margin-bottom:6px;margin-top:6px">Numeric columns</div>
                """, unsafe_allow_html=True)

                vals_html = "".join(
                    f'<span style="display:inline-block;background:#0f1f0f;'
                    f'border:1px solid #1a3a2a;border-radius:3px;padding:2px 6px;'
                    f'margin:2px;font-size:9px;color:#4a9a6a">{val}</span>'
                    for val in all_vals
                )
                st.markdown(
                    f'<div style="margin-bottom:8px">{vals_html}</div>',
                    unsafe_allow_html=True
                )

                            # ── high cardinality columns ───────────────────────────────────────
            high_card = detected.get("high_cardinality_cols", [])
            if high_card:
                st.markdown("""
                <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                color:#3a5a3a;margin-bottom:6px;margin-top:6px">Lookup columns</div>
                """, unsafe_allow_html=True)

                hc_html = "".join(
                    f'<span style="display:inline-block;background:#0f1f0f;'
                    f'border:1px solid #3a2a1e;border-radius:3px;padding:2px 6px;'
                    f'margin:2px;font-size:9px;color:#ca8a4a">{col}</span>'
                    for col in high_card
                )
                st.markdown(
                    f'<div style="margin-bottom:8px">{hc_html}</div>',
                    unsafe_allow_html=True
                )

            # ── domain badge ───────────────────────────────────────────────────
            domain = detected.get("domain")
            if domain and domain != "other":
                st.markdown(f"""
                <div style="margin-top:4px">
                <span style="font-size:8px;background:#1a2a0a;border:1px solid #2a4a1a;
                border-radius:3px;padding:2px 6px;color:#6a9a4a">
                domain: {domain}</span></div>
                """, unsafe_allow_html=True)

        # ── file overview — about + suggested questions ────────────────────────
        if st.session_state.get("source_type") in ["csv", "excel"] and \
        not st.session_state.get("final_state"):
            _render_file_overview()

        # ── pdf/image context info ─────────────────────────────────────────────
        if st.session_state.get("source_type") in ["pdf", "image"]:
            context = st.session_state.get("context", "")
            st.markdown(f"""
            <div style="font-size:9px;color:#3a5a3a;margin-top:6px">
            {len(context)} characters extracted</div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── past analyses ──────────────────────────────────────────────────────────
    st.markdown("""
    <p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:8px">Past Analyses</p>
    """, unsafe_allow_html=True)

    if st.session_state.get("past_analyses"):
        for analysis in reversed(st.session_state.past_analyses[-5:]):
            st.markdown(f"""
            <div style="background:#0a0e0a;border:1px solid #1a2a1a;
            border-radius:6px;padding:8px;margin-bottom:6px">
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


def _render_file_overview():
    """
    File upload ke baad — about this file + suggested questions.
    LLM sirf ek baar call hoga — session cache mein store.
    """
    from tools.file_overview import get_file_overview

    detected_columns = st.session_state.get("detected_columns", {})

    if "file_overview" not in st.session_state:
        with st.spinner("Analyzing file..."):
            df       = st.session_state.df
            overview = get_file_overview(df, detected_columns)
            st.session_state.file_overview = overview

    overview = st.session_state.file_overview

    # ── about section ──────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:6px">About this file</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0a0e0a;border:1px solid #1a2a1a;
    border-radius:8px;padding:10px;margin-bottom:8px">
        <div style="font-size:10px;color:#8aaa8a;line-height:1.6">
            {(overview or {}).get('about', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── suggested questions ────────────────────────────────────────────────────
    suggestions = (overview or {}).get("suggestions", [])
    if suggestions:
        st.markdown("""
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:6px">Try asking</div>
        """, unsafe_allow_html=True)

        for suggestion in suggestions:
            st.markdown(f"""
            <div style="background:#0a0e0a;border:1px solid #1a2a1a;
            border-radius:6px;padding:7px 10px;margin-bottom:4px;
            font-size:9px;color:#6ab46a;line-height:1.5">
            → {suggestion}
            </div>
            """, unsafe_allow_html=True)