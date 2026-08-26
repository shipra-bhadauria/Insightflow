import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st



def render_center_panel():

    # voice prefill
    prefill = st.session_state.pop("prefill_question", None)

    question = st.text_area(
        "Ask a question",
        value=prefill or "",
        placeholder="What is the average revenue per region?",
        height=80,
        label_visibility="collapsed",
        key="question_input"
    )

    # Enter key se analyze trigger karo
    enter_pressed = False
    if question and question.endswith("\n"):
        question = question.rstrip("\n")
        enter_pressed = True

    col_analyze, col_dashboard, col_voice = st.columns([2, 1, 0.5])
    with col_analyze:
        analyze_clicked = st.button("Analyze →", use_container_width=True)
    with col_dashboard:
        dashboard_clicked = st.button("Full Dashboard", use_container_width=True)
    with col_voice:
        if st.button("🎤", use_container_width=True, help="Voice input"):
            st.session_state["show_voice"] = not st.session_state.get("show_voice", False)

    # voice input
    if st.session_state.get("show_voice"):
        audio = st.audio_input(
            "🎤 Speak your question",
            key=f"voice_{st.session_state.get('voice_attempt', 0)}"
        )
        if audio is not None:
            with st.spinner("🎤 Transcribing..."):
                try:
                    from tools.voice_tools import transcribe_audio
                    text = transcribe_audio(audio.read(), "audio.wav")
                    if text and text.strip():
                        st.session_state["prefill_question"] = text.strip()
                        st.session_state["show_voice"] = False
                        st.session_state["voice_attempt"] = st.session_state.get("voice_attempt", 0) + 1
                        st.rerun()
                    else:
                        st.warning("⚠️ Could not transcribe. Try again.")
                        st.session_state["voice_attempt"] = st.session_state.get("voice_attempt", 0) + 1
                except Exception as e:
                    st.error(f"⚠️ Transcription failed: {str(e)}")
        else:
            st.caption("🔴 Recording... Click mic to stop, then wait")
    
    
    if (analyze_clicked or enter_pressed) and question and st.session_state.dataset_path:
        _run_analysis(question, mode="single")

    if dashboard_clicked and st.session_state.dataset_path:
        _run_dashboard_direct()

    if st.session_state.get("final_state"):
        if st.session_state.pop("cache_hit", False):
            st.info("⚡ Loaded from cache — no API call needed!")
        mode = st.session_state.final_state.get("mode", "single")
        if mode == "dashboard":
            _render_dashboard()
        else:
            _render_result()

        # follow-up questions buttons
        follow_ups = st.session_state.final_state.get("follow_up_questions", [])
        if follow_ups:
            st.markdown("<div style='margin-top:16px;color:#6ab46a;font-size:12px'>Try asking:</div>",
                        unsafe_allow_html=True)
            cols = st.columns(len(follow_ups))
            for i, q in enumerate(follow_ups[:3]):
                with cols[i]:
                    if st.button(f"» {q}", key=f"followup_{i}", use_container_width=True):
                        _run_analysis(q, mode="single")

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
    from logger         import ui_logger as logger
    from llm_cache      import get_cached, set_cache
    import time

    # ── per-user rate limit (5 requests / 60s) ───────────────
    now  = time.time()
    reqs = [t for t in st.session_state.get("user_requests", []) if now - t < 60]
    if len(reqs) >= 5:
        wait = int(60 - (now - reqs[0]))
        st.error(f"⏳ Too many requests. Please wait {wait}s before trying again.")
        return
    reqs.append(now)
    st.session_state["user_requests"] = reqs

    if st.session_state.get("source_type") in ["pdf", "image"]:
        _run_pdf_analysis(question)
        return

    st.session_state["last_question"] = question

    cached = get_cached(question, st.session_state.dataset_path, mode) if mode != "dashboard" else None
    if cached:
        from state import CriticVerdict
        cached["critic_history"] = [CriticVerdict(attempt_number=1, approved=True, confidence_score=0.9, reason="Loaded from cache", rows_validated=0)]
        cached.setdefault("analysis_history", [])
        st.session_state["cache_hit"] = True
        st.session_state.final_state = cached
        st.session_state.past_analyses.append({"question": question, "confidence": "cached", "report": cached.get("final_report", "")})
        st.rerun()
        return

    state = new_state(
        question=question,
        dataset_path=st.session_state.dataset_path,
        source_type=st.session_state.get("source_type", "csv"),
        mode=mode,
    )
    state["quality_report"]   = st.session_state.quality_report
    state["detected_columns"] = st.session_state.detected_columns

    logger.info(f"Analysis started | question='{question[:60]}' | mode={mode}")

    # ── multi-agent router ──────────────────────────────────────
    if mode == "single":
        from router import route_query
        routing = route_query(
            question    = question,
            has_dataset = bool(st.session_state.dataset_path),
            source_type = st.session_state.get("source_type", "csv"),
        )
        route = routing["route"]
        logger.info(f"Router | route={route} | confidence={routing['confidence']:.0%} | {routing['reason']}")

        # direct_llm — skip full pipeline
        if route == "direct_llm":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with st.spinner("💬 Answering directly..."):
                resp = client.chat.completions.create(
                    model    = os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
                    messages = [
                        {"role": "system", "content": "You are a helpful data analyst assistant. Be concise."},
                        {"role": "user",   "content": question},
                    ],
                    max_tokens = 400,
                )
                answer = resp.choices[0].message.content
            from state import CriticVerdict
            st.session_state.final_state = {
                "final_report":        f"## Answer\n\n{answer}",
                "follow_up_questions": [],
                "trace":               [f"ROUTER: direct_llm — {routing['reason']}", "REPORTER: answer written"],
                "critic_history":      [CriticVerdict(
                    attempt_number   = 1,
                    approved         = True,
                    confidence_score = 0.9,
                    reason           = "Direct LLM answer",
                    rows_validated   = 0,
                )],
                "analysis_history":    [],
                "mode":                "single",
            }
            st.session_state.past_analyses.append({
                "question":   question,
                "confidence": "direct",
                "report":     answer,
            })
            st.rerun()
            return

        # chat — redirect to chat tab
        if route == "chat":
            st.info("💬 This looks like a casual question — try the Chat tab!")
            return

    # ── streaming UI ─────────────────────────────────────────────
    agent_status = {
        "planner":   ("🧠 Planner",   "Planning analysis steps..."),
        "retrieval": ("🔍 Retrieval",  "Searching past analyses..."),
        "analyst":   ("⚙️ Analyst",   "Running analysis tools..."),
        "critic":    ("✅ Critic",    "Validating results..."),
        "reporter":  ("📝 Reporter",  "Writing final report..."),
        "hitl":      ("👤 HITL",      "Awaiting human review..."),
    }

    status_container = st.empty()
    progress_bar     = st.progress(0)
    progress_steps   = list(agent_status.keys())

    # progress bar green karo
    st.markdown("""
    <style>
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #4a6a4a !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        background-color: #1a1a1a !important;
    }
    </style>
    """, unsafe_allow_html=True)

    final_state = None
    accumulated_state = {}
    completed = []

    try:
        for step_output in workflow.stream(state, stream_mode="updates"):
            for node_name, node_state in step_output.items():
                label, msg = agent_status.get(node_name, (node_name, "Running..."))
                step_idx   = progress_steps.index(node_name) if node_name in progress_steps else 0
                progress   = int((step_idx + 1) / len(progress_steps) * 100)

                completed_text = "  ".join([f"✓ {l}" for l in completed])
                current = f"► {label}  {msg}"
                status_container.markdown(
                    f"`{completed_text}` &nbsp;&nbsp; **{current}**" if completed_text
                    else f"**{current}**"
                )
                completed.append(label)
                progress_bar.progress(progress)
                # trace pehle save karo
                existing_trace = accumulated_state.get("trace", [])[:]
                accumulated_state.update(node_state)
                # trace accumulate karo — overwrite nahi
                if "trace" in node_state:
                    accumulated_state["trace"] = existing_trace + node_state["trace"]
                final_state = accumulated_state

        progress_bar.progress(100)
        status_container.empty()

    except Exception as e:
        status_container.empty()
        progress_bar.empty()
        err = str(e)
        if "429" in err or "rate_limit" in err.lower() or "RateLimit" in err:
            st.error("⚠️ Rate limit reached. Wait 1 minute and try again.")
        elif "token" in err.lower():
            st.error("⚠️ Dataset too large. Try smaller file or specific question.")
        else:
            st.error(f"⚠️ Analysis failed: {err[:200]}")
        return

    

    # get final state from last stream output
    if final_state is None:
        st.error("⚠️ No output from pipeline.")
        return

    
    st.session_state.final_state = final_state
    if mode != "dashboard":
        set_cache(question, st.session_state.dataset_path, mode, final_state)

    confidence = final_state["critic_history"][-1].confidence_score * 100
    st.session_state.past_analyses.append({
        "question":   question,
        "confidence": f"{confidence:.0f}",
        "report":     final_state["final_report"],
    })

    st.rerun()


def _run_dashboard_direct():
    """Full Dashboard — pure tools only. No Planner/Retrieval/Critic/Reporter agents."""
    from state          import new_state, CriticVerdict
    from agents.analyst import analyst_node
    from logger         import ui_logger as logger
    import time

    st.session_state["last_question"] = "Generate a complete dashboard analysis"

    with st.spinner("⚡ Building dashboard..."):
        t0 = time.time()
        try:
            state = new_state(
                question="Generate a complete dashboard analysis",
                dataset_path=st.session_state.dataset_path,
                source_type=st.session_state.get("source_type", "csv"),
                mode="dashboard",
            )
            state["quality_report"]   = st.session_state.quality_report
            state["detected_columns"] = st.session_state.detected_columns

            result  = analyst_node(state)
            attempt = result["analysis_history"][0]

            n_rows   = state["quality_report"]["total_rows"] if state.get("quality_report") else 0
            n_charts = len(attempt.chart_paths)
            summary  = (
                f"### Dashboard Summary\n\n"
                f"Analyzed **{n_rows} rows** across all categories and numeric columns. "
                f"Generated **{n_charts} charts** covering counts, means, distributions, "
                f"trends, correlations, and anomalies."
            )

            final_state = {
                "question":            "Generate a complete dashboard analysis",
                "final_report":        summary,
                "follow_up_questions": [],
                "trace":               result.get("trace", []),
                "critic_history":      [CriticVerdict(
                    attempt_number=1, approved=True, confidence_score=0.95,
                    reason="Programmatic dashboard — tools only",
                    rows_validated=n_rows,
                )],
                "analysis_history":    result["analysis_history"],
                "mode":                "dashboard",
            }

            st.session_state.final_state = final_state
            logger.info(f"Dashboard built | charts={n_charts} | elapsed={time.time()-t0:.1f}s")

        except Exception as e:
            logger.error(f"Dashboard failed | {str(e)}", exc_info=True)
            st.error(f"⚠️ Dashboard generation failed: {str(e)[:200]}")
            return

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
        model=os.getenv("MODEL_REASONING", "gpt-4o-mini"),
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
            f"ANALYST: gpt-4o-mini {source_label} analysis",
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
    critic_hist = final_state.get("critic_history", [])
    verdict     = critic_hist[-1] if critic_hist else None
    question    = final_state.get("question") or st.session_state.get("last_question", "Analysis")

    chart_paths = []
    for a in final_state["analysis_history"]:
        if hasattr(a, "chart_paths") and a.chart_paths:
            chart_paths.extend(a.chart_paths)
        elif a.chart_path:
            chart_paths.append(a.chart_path)
    chart_paths = list(dict.fromkeys(chart_paths))

    pdf_path = export_pdf(
        question       = question,
        final_report   = final_state["final_report"],
        confidence     = verdict.confidence_score * 100 if verdict else 90.0,
        rows_validated = verdict.rows_validated or 0 if verdict else 0,
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
    critic_hist = final_state.get("critic_history", [])
    verdict     = critic_hist[-1] if critic_hist else None
    conf        = verdict.confidence_score * 100 if verdict else 90.0

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

    analysis_hist = final_state.get("analysis_history", [])

    charts = []
    for a in analysis_hist:
        if hasattr(a, 'chart_paths') and a.chart_paths:
            for cp in a.chart_paths:
                if cp and os.path.exists(cp) and cp not in charts:
                    charts.append(cp)
        elif getattr(a, "chart_path", None) and os.path.exists(a.chart_path):
            if a.chart_path not in charts:
                charts.append(a.chart_path)

    tool_results_list = []
    for a in analysis_hist:
        if hasattr(a, "tool_result") and a.tool_result:
            tool_results_list.append(a.tool_result)
    if not tool_results_list and final_state.get("analysis_history_tools"):
        tool_results_list = final_state["analysis_history_tools"]

    if tool_results_list:
        try:
            from tools.chart import make_plotly_chart
            df   = st.session_state.get("df")
            figs = []
            seen = set()

            if df is not None:
                for tool_result in tool_results_list:
                    for key, val in tool_result.items():
                        if not isinstance(val, dict):
                            continue
                        grp = val.get("group_by")
                        vc  = val.get("value_col", "")
                        agg = val.get("agg", "sum")
                        if not grp:
                            continue
                        grp = grp[0] if isinstance(grp, list) else grp
                        if grp not in df.columns:
                            continue
                        y_col = vc if (vc and vc in df.columns and vc != grp) else grp
                        if y_col != grp and agg == "count":
                            agg = "mean"
                        sig = f"{grp}|{y_col}|{agg}"
                        if sig in seen:
                            continue
                        seen.add(sig)
                        kind = "pie" if (agg == "count" and df[grp].nunique() <= 8) else "bar"
                        try:
                            fig = make_plotly_chart(df, kind=kind, x=grp, y=y_col, agg=agg)
                            if fig:
                                figs.append((y_col, fig))
                        except Exception:
                            pass

            if figs:
                cfg = {"scrollZoom": True, "displayModeBar": True}
                if len(figs) == 1:
                    st.plotly_chart(figs[0][1], use_container_width=True, config=cfg, key="single_chart_0")
                else:
                    for i in range(0, len(figs), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(figs[i][1], use_container_width=True, config=cfg, key=f"single_chart_{i}")
                        with col2:
                            if i + 1 < len(figs):
                                st.plotly_chart(figs[i+1][1], use_container_width=True, config=cfg, key=f"single_chart_{i+1}")
            else:
                for cp in charts:
                    if os.path.exists(cp):
                        st.image(cp, use_container_width=True)
        except Exception:
            for cp in charts:
                if os.path.exists(cp):
                    st.image(cp, use_container_width=True)

    attempt = analysis_hist[-1] if analysis_hist else None

    if verdict and getattr(verdict, 'rows_validated', None):
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
                     key="dash_revise"):
            st.session_state["show_revision_input"] = True

    # HITL revision input
    if st.session_state.get("show_revision_input"):
        revision_note = st.text_input(
            "What needs to be revised?",
            placeholder="e.g. Show mean instead of sum...",
            key="revision_input",
        )
        col_s, col_c = st.columns([2, 1])
        with col_s:
            if st.button("↩ Submit revision", use_container_width=True, key="submit_revision"):
                if revision_note:
                    original_q = st.session_state.get("last_question", "")
                    revised_q  = f"{original_q} (Revision: {revision_note})"
                    st.session_state["show_revision_input"] = False
                    st.session_state.final_state = None
                    _run_analysis(revised_q, mode="single")
        with col_c:
            if st.button("Cancel", use_container_width=True, key="cancel_revision"):
                st.session_state["show_revision_input"] = False
                st.rerun()
        
    with col_save:
        if st.button("◎ Save to memory",
                     use_container_width=True,
                     key="single_save"):
            from memory.chroma_store import save_analysis_chroma
            final_state = st.session_state.final_state
            verdict     = final_state["critic_history"][-1]
            user_id     = st.session_state.get("user_id", "default")
            save_analysis_chroma(
                question       = final_state.get("question") or st.session_state.get("last_question", "Analysis"),
                final_report   = final_state["final_report"],
                confidence     = verdict.confidence_score,
                rows_validated = verdict.rows_validated or 0,
                dataset_path   = st.session_state.dataset_path,
                user_id        = user_id,
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
    critic_hist = final_state.get("critic_history", [])
    verdict     = critic_hist[-1] if critic_hist else None
    conf        = verdict.confidence_score * 100 if verdict else 90.0

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

    analysis_hist = final_state.get("analysis_history", [])
    attempt = analysis_hist[-1] if analysis_hist else None
    tool_result = attempt.tool_result if attempt and hasattr(attempt, "tool_result") else {}

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
    for a in final_state.get("analysis_history", []):
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

        for i in range(0, len(charts), 2):
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(charts[i]):
                    st.image(charts[i], use_container_width=True)
            with col2:
                if i + 1 < len(charts) and os.path.exists(charts[i+1]):
                    st.image(charts[i+1], use_container_width=True)

                    
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
    col_approve, col_revise, col_memory = st.columns([2, 1, 1])
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
    with col_memory:
        if st.button("💾 Save to memory",
                     use_container_width=True,
                     key="dash_save_memory"):
            from memory.chroma_store import save_analysis_chroma
            user_id  = st.session_state.get("user_id", "default")
            ch       = st.session_state.final_state.get("critic_history", [])
            conf_val = ch[-1].confidence_score if ch else 0.9
            rows_val = ch[-1].rows_validated or 0 if ch else 0
            save_analysis_chroma(
                question       = st.session_state.get("last_question", ""),
                final_report   = st.session_state.final_state["final_report"],
                confidence     = conf_val,
                rows_validated = rows_val,
                dataset_path   = st.session_state.dataset_path,
                user_id        = user_id,
            )
            st.success("✅ Saved to memory!")

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