import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

CHAT_HISTORY_FILE = "data/chat_sessions.json"


def _load_sessions() -> list:
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            return json.loads(open(CHAT_HISTORY_FILE).read())
    except Exception:
        pass
    return []


def _save_sessions(sessions: list):
    try:
        os.makedirs("data", exist_ok=True)
        open(CHAT_HISTORY_FILE, "w").write(json.dumps(sessions, ensure_ascii=False))
    except Exception:
        pass


def render_chat_panel():

    # past chat buttons grey style
    st.markdown("""
    <style>
    /* new chat button — bright lime green */
    div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stButton"] button {
        background-color: #c5f432 !important;
        color: #0a0e0a !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 13px !important;
        padding: 10px 16px !important;
        letter-spacing: .05em !important;
    }
    div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stButton"] button:hover {
        background-color: #d4ff40 !important;
    }
    /* past chat buttons — muted grey */
    div[data-testid="stVerticalBlock"] > div:not(:first-child) div[data-testid="stButton"] button {
        background-color: #111811 !important;
        color: #5a7a5a !important;
        border: 1px solid #1e2e1e !important;
        font-weight: 400 !important;
        font-size: 11px !important;
        text-align: left !important;
        padding: 6px 10px !important;
    }
    div[data-testid="stVerticalBlock"] > div:not(:first-child) div[data-testid="stButton"] button:hover {
        background-color: #1a2a1a !important;
        color: #c5f432 !important;
        border-color: #3a5a3a !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if "chat_messages"      not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id"    not in st.session_state:
        st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "chat_sessions"      not in st.session_state:
        st.session_state.chat_sessions = _load_sessions()
    if "chat_counter"       not in st.session_state:
        st.session_state.chat_counter = 0

    # two column layout: sidebar + main chat
    col_sidebar, col_main = st.columns([1, 3])

    # ── Sidebar ───────────────────────────────────────────────────
    with col_sidebar:
        st.markdown("""
        <div style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
        color:#3a5a3a;margin-bottom:8px">Chats</div>
        """, unsafe_allow_html=True)

        if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True):
            # save current chat if not empty
            if st.session_state.chat_messages:
                sessions = st.session_state.chat_sessions
                sessions.append({
                    "id":       st.session_state.chat_session_id,
                    "title":    st.session_state.chat_messages[0]["content"][:40] + "...",
                    "messages": st.session_state.chat_messages,
                    "time":     datetime.now().strftime("%d %b %H:%M"),
                })
                st.session_state.chat_sessions = sessions
                _save_sessions(sessions)
            # reset
            st.session_state.chat_messages   = []
            st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.chat_counter   += 1
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # list past sessions
        sessions = st.session_state.chat_sessions
        if not sessions:
            st.markdown("""
            <div style="font-size:10px;color:#2a4a2a;font-style:italic">
            No past chats</div>
            """, unsafe_allow_html=True)
        else:
            for i, sess in enumerate(reversed(sessions[-10:])):
                label = sess['title'][:28]
                time  = sess.get('time', '')
                clicked = st.button(
                    f"💬 {label}",
                    key=f"sess_{i}",
                    use_container_width=True,
                    help=time,
                )
                st.markdown(
                    f"<div style='font-size:8px;color:#2a3a2a;margin-top:-10px;"
                    f"margin-bottom:6px;padding-left:4px'>{time}</div>",
                    unsafe_allow_html=True,
                )
                if clicked:
                    st.session_state.chat_messages   = sess["messages"]
                    st.session_state.chat_session_id = sess["id"]
                    st.session_state.chat_counter   += 1
                    st.rerun()

    # ── Main Chat ─────────────────────────────────────────────────
    with col_main:
        st.markdown("""
        <div style="margin-bottom:16px">
            <span style="color:#c5f432;font-size:16px;font-weight:700">💬 Chat</span>
            <span style="color:#3a5a3a;font-size:11px;margin-left:10px">
            Ask anything — with or without a dataset</span>
        </div>
        """, unsafe_allow_html=True)

        # dataset context banner
        has_dataset = st.session_state.get("dataset_path") is not None
        if has_dataset:
            fname = os.path.basename(st.session_state.dataset_path)
            st.markdown(f"""
            <div style="background:#0f1f0f;border:1px solid #2a4a2a;border-radius:6px;
            padding:6px 12px;font-size:10px;color:#6a8a6a;margin-bottom:12px">
            📊 Dataset: <b style="color:#c5f432">{fname}</b> loaded
            </div>
            """, unsafe_allow_html=True)

        # chat history display
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:#0f1f0f;border:1px solid #2a4a2a;border-radius:8px;
                padding:10px 14px;margin:6px 0;text-align:right;font-size:13px;color:#c8e0c8">
                {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0a0e0a;border:1px solid #1a2a1a;border-radius:8px;
                padding:10px 14px;margin:6px 0;font-size:13px;color:#8acc8a;line-height:1.6">
                {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # input row
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Ask anything...",
                label_visibility="collapsed",
                key=f"chat_input_{st.session_state.get('chat_counter', 0)}",
            )
        with col_send:
            send = st.button("Send", use_container_width=True, key="chat_send")

        if send and user_input:
            st.session_state.chat_messages.append({
                "role": "user", "content": user_input,
            })

            # system prompt
            system_parts = [
                "You are InsightFlow's AI assistant — helpful data analyst and general assistant.",
                "Be concise, accurate, and friendly.",
            ]

            if has_dataset:
                df       = st.session_state.get("df")
                detected = st.session_state.get("detected_columns", {})
                quality  = st.session_state.get("quality_report", {})
                if df is not None:
                    system_parts.append(
                        f"\nDataset:\n"
                        f"- File: {os.path.basename(st.session_state.dataset_path)}\n"
                        f"- Rows: {len(df)}, Columns: {list(df.columns)}\n"
                        f"- Categories: {detected.get('all_category_cols', [])}\n"
                        f"- Numerics: {detected.get('all_value_cols', [])}\n"
                        f"- Health: {quality.get('health_label', 'unknown')}\n"
                                                f"\nCategory unique values: { {col: df[col].dropna().unique().tolist()[:20] for col in detected.get('all_category_cols', [])[:5] if col in df.columns} }\n"
                        f"\nSample (5 rows):\n{df.sample(min(5, len(df))).to_string()}"
                    )

            with st.spinner("Thinking..."):
                try:
                    client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    messages = [{"role": "system", "content": "\n".join(system_parts)}]
                    for m in st.session_state.chat_messages[-10:]:
                        messages.append({"role": m["role"], "content": m["content"]})

                    response = client.chat.completions.create(
                        model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7,
                    )
                    answer = response.choices[0].message.content
                    st.session_state.chat_messages.append({
                        "role": "assistant", "content": answer,
                    })
                    st.session_state["chat_counter"] = st.session_state.get("chat_counter", 0) + 1
                    st.rerun()

                except Exception as e:
                    err = str(e)
                    if "429" in err:
                        st.error("⚠️ Rate limit. Wait 1 minute.")
                    else:
                        st.error(f"⚠️ Error: {err[:200]}")
