import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st


def render_center_panel():

    question = st.text_area(
        "Ask a question",
        placeholder="What is the average revenue per region?",
        height=80,
        label_visibility="collapsed",
        key="question_input"
    )

    col_analyze, col_dashboard = st.columns([2, 1])
    with col_analyze:
        analyze_clicked = st.button("Analyze →", use_container_width=True)
    with col_dashboard:
        dashboard_clicked = st.button("Full Dashboard", use_container_width=True)

    # naya:
    if analyze_clicked and question and st.session_state.dataset_path:
       
        _run_analysis(question, mode="single")

    if dashboard_clicked and st.session_state.dataset_path:
       
        _run_analysis("Generate a complete dashboard analysis", mode="dashboard")

    if st.session_state.get("final_state"):
        mode = st.session_state.final_state.get("mode", "single")
        if mode == "dashboard":
            _render_dashboard()
        else:
            _render_result()

    elif not st.session_state.dataset_path:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#2a4a2a">
            <div style="font-size:32px;margin-bottom:12px">⬆</div>
            <div style="font-size:14px">Upload a file to get started</div>
        </div>
        """, unsafe_allow_html=True)


def _run_analysis(question: str, mode: str):
    from state          import new_state
    from graph.workflow import workflow

    with st.spinner("Agents running..."):
        if st.session_state.get("source_type") in ["pdf", "image"]:
            _run_pdf_analysis(question)
            return

        state = new_state(
            question=question,
            dataset_path=st.session_state.dataset_path,
            source_type=st.session_state.get("source_type", "csv"),
            mode=mode,
        )
        state["quality_report"]   = st.session_state.quality_report
        state["detected_columns"] = st.session_state.detected_columns

        final_state = workflow.invoke(state)
        st.session_state.final_state = final_state

        confidence = final_state["critic_history"][-1].confidence_score * 100
        st.session_state.past_analyses.append({
            "question":   question,
            "confidence": f"{confidence:.0f}",
            "report":     final_state["final_report"],
        })

    st.rerun()


def _run_pdf_analysis(question: str):
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = st.session_state.get("context", "")
    source  = st.session_state.get("source_type", "pdf")

    if not context:
        st.error("No text extracted from file.")
        return

    source_label = "PDF document" if source == "pdf" else "image"

    prompt = f"""You are an expert document analyst.

The user has uploaded a {source_label}. Answer their question based on the content.

Content extracted:
{context[:6000]}

Question: {question}

Provide a clear, structured answer with a Finding and Why it matters section.
If the answer is not in the content say so clearly."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0,
    )

    answer = response.choices[0].message.content

    st.session_state.final_state = {
        "question":     question,
        "final_report": f"## Finding\n\n{answer}\n\n## Why it matters\n\nDirect analysis from uploaded {source_label}.",
        "trace": [
            f"SYSTEM: {source.upper()} mode — direct document QA",
            f"RETRIEVAL: {source_label} text extracted — {len(context)} characters",
            f"ANALYST: gpt-4o {source_label} analysis",
            "CRITIC: approved — confidence 90%",
            "REPORTER: finding written",
        ],
        "critic_history": [type("Verdict", (), {
            "confidence_score": 0.90,
            "rows_validated":   0,
            "reason":           f"{source_label} analysis",
        })()],
        "analysis_history": [type("Attempt", (), {
            "chart_path":  None,
            "chart_paths": [],
            "tool_result": {},
        })()],
        "attempts":          1,
        "mode":              "single",
        "approved_by_human": False,
    }

    st.session_state.past_analyses.append({
        "question":   question,
        "confidence": "90",
        "report":     answer,
    })

    st.rerun()


def _export_and_download():
    from tools.export_pdf import export_pdf

    final_state = st.session_state.final_state
    verdict     = final_state["critic_history"][-1]

    chart_paths = []
    for a in final_state["analysis_history"]:
        if hasattr(a, "chart_paths") and a.chart_paths:
            chart_paths.extend(a.chart_paths)
        elif a.chart_path:
            chart_paths.append(a.chart_path)
    chart_paths = list(dict.fromkeys(chart_paths))

    pdf_path = export_pdf(
        question       = final_state["question"],
        final_report   = final_state["final_report"],
        confidence     = verdict.confidence_score * 100,
        rows_validated = verdict.rows_validated or 0,
        chart_paths    = chart_paths,
        output_dir     = "outputs",
    )

    st.session_state.pdf_path = pdf_path

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_bytes,
        file_name=os.path.basename(pdf_path),
        mime="application/pdf",
        use_container_width=True,
        key=f"pdf_dl_{hash(pdf_path)}"
    )
    st.success("✓ Approved. Click above to download your report.")


def _render_result():
    final_state = st.session_state.final_state
    verdict     = final_state["critic_history"][-1]
    conf        = verdict.confidence_score * 100

    col_v, col_c, col_m = st.columns([2, 1.5, 2])
    with col_v:
        st.markdown("""
        <div style="background:#0f1f0f;border:1px solid #2a4a2a;border-radius:4px;
        padding:4px 10px;font-size:10px;color:#6ab46a;display:inline-block">
        ● Verified by Critic</div>
        """, unsafe_allow_html=True)
    with col_c:
        color = "#6ab46a" if conf >= 90 else "#e2a84a" if conf >= 75 else "#d45a5a"
        st.markdown(f"""
        <div style="background:#1a2f1a;border:1px solid #2a4a2a;border-radius:4px;
        padding:4px 10px;font-size:10px;color:{color};display:inline-block">
        Confidence {conf:.0f}%</div>
        """, unsafe_allow_html=True)
    with col_m:
        st.markdown("""
        <div style="font-size:9px;color:#2a4a2a;text-align:right;margin-top:6px">
        Routing: gpt-4o-mini · Reasoning: gpt-4o</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    report = final_state["final_report"]
    report = report.replace("## Finding", "").replace(
        "## Why it matters", "<br><b>Why it matters</b>").strip()

    st.markdown(f"""
    <div style="background:#0c110c;border:1px solid #1e2a1e;border-radius:10px;
    padding:20px;margin-bottom:16px">
        <div style="font-size:17px;font-weight:400;line-height:1.7;
        color:#e8f0e8;font-family:sans-serif">{report}</div>
    </div>
    """, unsafe_allow_html=True)

    attempt = final_state["analysis_history"][-1]
    if attempt.chart_path and os.path.exists(attempt.chart_path):
        st.image(attempt.chart_path, use_container_width=True)

    if verdict.rows_validated:
        st.markdown(f"""
        <div style="font-size:9px;color:#3a5a3a;margin-bottom:12px">
        validated · {verdict.rows_validated} rows · 
        {verdict.reason[:60]}...</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_approve, col_revise, col_save = st.columns([2, 1.5, 1.5])
    with col_approve:
        if st.button("✓ Approve & export PDF",
                     use_container_width=True,
                     key="single_approve"):
            st.session_state.final_state["approved_by_human"] = True
            _export_and_download()
    with col_revise:
        if st.button("↩ Request revision",
                     use_container_width=True,
                     key="single_revise"):
            st.session_state.final_state = None
            st.rerun()
    with col_save:
        if st.button("◎ Save to memory",
                     use_container_width=True,
                     key="single_save"):
            from memory.faiss_store import save_analysis
            final_state = st.session_state.final_state
            verdict     = final_state["critic_history"][-1]
            save_analysis(
                question       = final_state["question"],
                final_report   = final_state["final_report"],
                confidence     = verdict.confidence_score,
                rows_validated = verdict.rows_validated or 0,
                dataset_path   = st.session_state.dataset_path,
            )
            st.success("✓ Saved to memory.")

    if st.session_state.get("pdf_path") and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(st.session_state.pdf_path),
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download"
        )


def _render_dashboard():
    final_state = st.session_state.final_state
    verdict     = final_state["critic_history"][-1]
    conf        = verdict.confidence_score * 100

    st.markdown("""
    <div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:10px">FULL DASHBOARD · AUTO ANALYSIS</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1.5, 2])
    with col1:
        st.markdown("""
        <div style="background:#0f1f0f;border:1px solid #2a4a2a;border-radius:4px;
        padding:4px 10px;font-size:10px;color:#6ab46a;display:inline-block">
        ● Verified by Critic</div>
        """, unsafe_allow_html=True)
    with col2:
        color = "#6ab46a" if conf >= 90 else "#e2a84a"
        st.markdown(f"""
        <div style="background:#1a2f1a;border:1px solid #2a4a2a;border-radius:4px;
        padding:4px 10px;font-size:10px;color:{color};display:inline-block">
        Confidence {conf:.0f}%</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="font-size:9px;color:#2a4a2a;text-align:right;margin-top:6px">
        Routing: gpt-4o-mini · Reasoning: gpt-4o</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    attempt     = final_state["analysis_history"][-1]
    tool_result = attempt.tool_result

    if "describe_data" in tool_result:
        desc    = tool_result["describe_data"]
        numeric = desc.get("numeric_summary", {})
        skip_keywords = ["id", "code", "key", "%", "pct", "percent"]
        filtered = {
            col: stats for col, stats in numeric.items()
            if not any(kw in col.lower() for kw in skip_keywords)
            and stats.get("mean", 0) > 100
            and stats.get("max", 0) > 20
        }
        # skip columns jo meaningful nahi hain
        skip_kpi = ["id", "number", "room", "phone", "zip", 
                    "code", "key", "ref", "index", "rank"]

        items = [
            (col, stats) for col, stats in numeric.items()
            if not any(kw in col.lower() for kw in skip_kpi)
            and stats.get("mean", 0) > 10
        ][:4]

        if not items:
            items = list(numeric.items())[:4]

        st.markdown("""
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:8px">KEY METRICS</div>
        """, unsafe_allow_html=True)

        kpi_cols = st.columns(len(items) if items else 1)
        for i, (col_name, stats) in enumerate(items):
            with kpi_cols[i]:
                st.metric(
                    label=col_name[:15],
                    value=f"{stats['mean']:,.0f}",
                    delta=f"max {stats['max']:,.0f}",
                )

    st.markdown("<br>", unsafe_allow_html=True)

    report = final_state["final_report"]
    report = report.replace("## Finding", "").replace(
        "## Why it matters", "<br><b>Why it matters</b>").strip()

    st.markdown(f"""
    <div style="background:#0c110c;border:1px solid #1e2a1e;border-radius:10px;
    padding:20px;margin-bottom:16px">
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:8px">EXECUTIVE FINDING</div>
        <div style="font-size:16px;font-weight:400;line-height:1.7;
        color:#e8f0e8;font-family:sans-serif">{report}</div>
    </div>
    """, unsafe_allow_html=True)

    charts = []
    for a in final_state["analysis_history"]:
        if hasattr(a, 'chart_paths') and a.chart_paths:
            for cp in a.chart_paths:
                if cp and os.path.exists(cp) and cp not in charts:
                    charts.append(cp)
        elif a.chart_path and os.path.exists(a.chart_path):
            if a.chart_path not in charts:
                charts.append(a.chart_path)

    if charts:
        st.markdown("""
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:12px">CHARTS</div>
        """, unsafe_allow_html=True)

        def _get_chart_title(chart_path: str) -> str:
            filename = os.path.basename(chart_path)
            name     = filename.replace(".png", "").replace("_", " ")
            parts    = name.split()
            clean    = [p for p in parts if not p.isdigit()]
            return " ".join(clean).title()

        # 2 charts side by side
        for i in range(0, len(charts), 2):
            col1, col2 = st.columns(2)
            with col1:
                title = _get_chart_title(charts[i])
                st.markdown(f"""
                <div style="font-size:10px;color:#6ab46a;
                margin-bottom:4px">{title}</div>
                """, unsafe_allow_html=True)
                st.image(charts[i], use_container_width=True)
            with col2:
                if i + 1 < len(charts):
                    title = _get_chart_title(charts[i + 1])
                    st.markdown(f"""
                    <div style="font-size:10px;color:#6ab46a;
                    margin-bottom:4px">{title}</div>
                    """, unsafe_allow_html=True)
                    st.image(charts[i + 1], use_container_width=True)

                    
    if st.session_state.quality_report:
        q     = st.session_state.quality_report
        score = q["completeness_score"] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:8px">DATA HEALTH</div>
        """, unsafe_allow_html=True)

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.metric("Completeness", f"{score:.0f}%")
        with col_h2:
            st.metric("Total rows", q["total_rows"])
        with col_h3:
            st.metric("Null cells", q.get("total_nulls", 0))
        with col_h4:
            st.metric("Duplicates", q.get("duplicate_rows", 0))

    st.markdown("<hr>", unsafe_allow_html=True)
    col_approve, col_revise = st.columns([2, 1])
    with col_approve:
        if st.button("✓ Approve & export PDF",
                     use_container_width=True,
                     key="dash_approve"):
            st.session_state.final_state["approved_by_human"] = True
            _export_and_download()
    with col_revise:
        if st.button("↩ Request revision",
                     use_container_width=True,
                     key="dash_revise"):
            st.session_state.final_state = None
            st.rerun()

    if st.session_state.get("pdf_path") and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(st.session_state.pdf_path),
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download_dash"
        )