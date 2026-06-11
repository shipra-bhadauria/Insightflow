import streamlit as st


def render_right_panel():

    st.markdown("""
    <p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:10px">Live Agent Workflow</p>
    """, unsafe_allow_html=True)

    if not st.session_state.get("final_state"):
        # waiting state — show empty agent cards
        agents = [
            ("Planner",  "gpt-4o-mini", "waiting"),
            ("Retrieval","gpt-4o-mini", "waiting"),
            ("Analyst",  "gpt-4o",      "waiting"),
            ("Critic",   "gpt-4o-mini", "waiting"),
            ("Reporter", "gpt-4o",      "waiting"),
        ]
        for name, model, status in agents:
            _render_agent_card(name, model, status, "")
        return

    # show real trace from final state
    final_state = st.session_state.final_state
    trace       = final_state["trace"]

    # determine status of each agent from trace
    agent_data = {
        "Planner":  {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Retrieval":{"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Analyst":  {"model": "gpt-4o",      "status": "waiting", "body": ""},
        "Critic":   {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Reporter": {"model": "gpt-4o",      "status": "waiting", "body": ""},
    }

    for entry in trace:
        entry_upper = entry.upper()
        if "PLANNER"  in entry_upper:
            agent_data["Planner"]["status"] = "done"
            agent_data["Planner"]["body"]   = entry
        elif "RETRIEVAL" in entry_upper:
            agent_data["Retrieval"]["status"] = "done"
            agent_data["Retrieval"]["body"]   = entry
        elif "ANALYST" in entry_upper:
            agent_data["Analyst"]["status"] = "done"
            agent_data["Analyst"]["body"]   = entry
        elif "CRITIC" in entry_upper:
            status = "rejected" if "rejected" in entry.lower() else "done"
            agent_data["Critic"]["status"] = status
            agent_data["Critic"]["body"]   = entry
        elif "REPORTER" in entry_upper:
            agent_data["Reporter"]["status"] = "done"
            agent_data["Reporter"]["body"]   = entry

    # render each agent card
    for name, data in agent_data.items():
        _render_agent_card(name, data["model"], data["status"], data["body"])
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # attempts + confidence summary
    attempts    = final_state["attempts"]
    verdict     = final_state["critic_history"][-1]
    confidence  = verdict.confidence_score * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:9px;color:#3a5a3a;line-height:2">
        Attempts: <span style="color:#6a8a6a">{attempts}</span><br>
        Confidence: <span style="color:#8acc8a">{confidence:.0f}%</span><br>
        Rows validated: <span style="color:#6a8a6a">{verdict.rows_validated}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:8px;color:#2a4a2a;line-height:1.8">
    ROUTING · <span style="color:#4a6a4a">gpt-4o-mini</span><br>
    Planner · Retrieval · Critic<br><br>
    REASONING · <span style="color:#4a6a4a">gpt-4o</span><br>
    Analyst · Reporter
    </div>
    """, unsafe_allow_html=True)


def _render_agent_card(name: str, model: str, status: str, body: str):
    if status == "done":
        icon = "✅"
    elif status == "rejected":
        icon = "❌"
    else:
        icon = "⏳"

    clean_body = ""
    if body:
        clean_body = body.replace("PLANNER: ", "").replace(
            "ANALYST ", "").replace(
            "CRITIC: ", "").replace(
            "REPORTER: ", "")[:120]

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**{icon} {name}**",
                help=clean_body if clean_body else None
            )
        with col2:
            st.caption(model)
        if clean_body:
            st.caption(clean_body)
        st.divider()