import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import safe_json
from state import InsightFlowState

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_REASONING", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
)

SYSTEM_PROMPT = """You are the Reporter agent in InsightFlow — an AI data analyst system.

Your job: take validated analysis results and write a clear, concise executive finding.

Every finding must have two parts:
1. THE FINDING — the key insight in one or two sentences. Lead with the most important number.
2. WHY IT MATTERS — one sentence explaining the business implication.

Rules:
- Lead with the most important number or comparison
- Use plain English — no jargon, no technical terms
- Be specific — name the exact values, not just "higher" or "lower"
- The why-it-matters must connect to a business decision or action
- Keep the total response under 100 words
- Format as markdown with ## Finding and ## Why it matters sections"""


def reporter_node(state: InsightFlowState) -> dict:

    question       = state["question"]
    latest_attempt = state["analysis_history"][-1]
    latest_verdict = state["critic_history"][-1]

    # build context
    context = f"""Original question: {question}

            Validated result:
            Tool: {latest_attempt.tool_called}
            Result: {safe_json(latest_attempt.tool_result)}

            Critic confidence: {latest_verdict.confidence_score * 100:.0f}%
            Rows validated: {latest_verdict.rows_validated}"""

    # add conversation history for follow-up context
    if state.get("conversation_history"):
        history = state["conversation_history"]
        context += f"\n\nPrevious analyses in this session: {len(history)}"
        if history:
            context += f"\nLast question was: {history[-1].question}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context)
    ])

    final_report = response.content.strip()

    trace_entry = (
        f"REPORTER: executive finding written — "
        f"confidence {latest_verdict.confidence_score * 100:.0f}%"
    )

    return {
        "final_report": final_report,
        "next_agent":   "hitl",
        "trace":        [trace_entry],
    }


if __name__ == "__main__":
    import pandas as pd
    from tools.quality        import run_quality_report
    from tools.detect_columns import detect_columns
    from tools.aggregate      import aggregate
    from state                import AnalysisAttempt, CriticVerdict, new_state

    df       = pd.read_csv("data/sales.csv")
    quality  = run_quality_report(df)
    detected = detect_columns(df)

    # simulate approved Analyst result
    result = aggregate(
        df,
        group_by="Region",
        value_col="Total Revenue",
        agg="mean"
    )

    attempt = AnalysisAttempt(
        attempt_number=1,
        tool_called="aggregate",
        tool_args={"group_by": "Region", "value_col": "Total Revenue", "agg": "mean"},
        tool_result={"aggregate": result},
        chart_path=None,
    )

    verdict = CriticVerdict(
        attempt_number=1,
        approved=True,
        confidence_score=0.92,
        reason="Result is correct, nulls handled, answers the question.",
        rows_validated=100,
    )

    test_state = new_state(
        question="What is the average revenue per region?",
        dataset_path="data/sales.csv",
    )
    test_state["quality_report"]   = quality
    test_state["detected_columns"] = detected
    test_state["analysis_history"] = [attempt]
    test_state["critic_history"]   = [verdict]

    result = reporter_node(test_state)

    print("=== Reporter output ===\n")
    print(result["final_report"])
    print(f"\nTrace: {result['trace'][0]}")