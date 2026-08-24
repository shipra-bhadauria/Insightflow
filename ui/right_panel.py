import streamlit as st


def render_right_panel():

    st.markdown("""
    <p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    color:#3a5a3a;margin-bottom:10px">Live Agent Workflow</p>
    """, unsafe_allow_html=True)

    if not st.session_state.get("final_state"):
        agents = [
            ("Planner",   "gpt-4o-mini", "waiting", ""),
            ("Retrieval", "gpt-4o-mini", "waiting", ""),
            ("Analyst",   "gpt-4o-mini", "waiting", ""),
            ("Critic",    "gpt-4o-mini", "waiting", ""),
            ("Reporter",  "gpt-4o-mini", "waiting", ""),
        ]
        for name, model, status, body in agents:
            _render_agent_card(name, model, status, body)
        return

    final_state = st.session_state.final_state
    trace = final_state.get("trace", [])

    agent_data = {
        "Planner":   {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Retrieval": {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Analyst":   {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Critic":    {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
        "Reporter":  {"model": "gpt-4o-mini", "status": "waiting", "body": ""},
    }

    for entry in trace:
        eu = entry.upper()
        if "PLANNER" in eu:
            agent_data["Planner"]["status"] = "done"
            agent_data["Planner"]["body"] = entry.replace("PLANNER: ", "").strip()
        elif "RETRIEVAL" in eu:
            agent_data["Retrieval"]["status"] = "done"
            agent_data["Retrieval"]["body"] = entry.replace("RETRIEVAL: ", "").strip()
        elif "ANALYST" in eu:
            agent_data["Analyst"]["status"] = "done"
            agent_data["Analyst"]["body"] = entry.replace("ANALYST: ", "").replace("ANALYST ", "").strip()
        elif "CRITIC" in eu:
            agent_data["Critic"]["status"] = "rejected" if "rejected" in entry.lower() else "done"
            agent_data["Critic"]["body"] = entry.replace("CRITIC: ", "").strip()
        elif "REPORTER" in eu:
            agent_data["Reporter"]["status"] = "done"
            agent_data["Reporter"]["body"] = entry.replace("REPORTER: ", "").strip()

    for name, data in agent_data.items():
        _render_agent_card(name, data["model"], data["status"], data["body"])

    verdict    = final_state["critic_history"][-1]
    confidence = verdict.confidence_score * 100
    attempts   = final_state.get("attempts", 1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:9px;color:#3a5a3a;line-height:2">
        Attempts: <span style="color:#6a8a6a">{attempts}</span><br>
        Confidence: <span style="color:#8acc8a">{confidence:.0f}%</span><br>
        Rows validated: <span style="color:#6a8a6a">{verdict.rows_validated}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_agent_card(name: str, model: str, status: str, body: str):
    if status == "done":
        icon = "✅"
    elif status == "rejected":
        icon = "❌"
    else:
        icon = "⏳"

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{icon} {name}**")
        with col2:
            st.caption(model)
        if body:
            st.markdown(
                f"<div style='font-size:10px;color:#6a8a6a;"
                f"margin-top:-10px;padding-bottom:6px;line-height:1.4'>"
                f"{body[:150]}</div>",
                unsafe_allow_html=True,
            )
        st.divider()