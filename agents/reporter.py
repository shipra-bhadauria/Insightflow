import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import safe_json
from state import InsightFlowState
from logger import agent_logger as logger

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_REASONING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
)

SYSTEM_PROMPT = """You are the Reporter agent in InsightFlow — an AI data analyst system.

Your job: take validated analysis results and write a clear, accurate finding.

IF RAW RESULT (agg = "raw"):
- The full data table is already shown in the UI — do NOT reproduce it
- Write a 2-3 sentence SUMMARY:
  * Total records shown
  * Overall range (min to max) for each numeric column across ALL records
  * One interesting observation (highest, lowest, or notable pattern)
- Skip "Why it matters" section

IF AGGREGATED RESULT:
- THE FINDING — key insight in 1-2 sentences with most important number
- WHY IT MATTERS — one sentence business implication
- Keep under 100 words

Rules:
- Plain English, no jargon
- NEVER say analysis failed if data is present
- NEVER reproduce a table already shown in UI
- Format as markdown
"""


FOLLOWUP_PROMPT = """You are a data analyst assistant.
Given a question and its finding, suggest 2-3 natural follow-up questions.

Rules:
- Questions must be about the SAME dataset
- Each question should explore a different angle
- Keep each question under 10 words
- NO explanations, just the questions

Respond ONLY with a JSON array:
["question 1", "question 2", "question 3"]"""


def reporter_node(state: InsightFlowState) -> dict:

    question       = state["question"]
    latest_attempt = state["analysis_history"][-1]
    latest_verdict = state["critic_history"][-1]
    logger.info(f"Reporter | confidence={latest_verdict.confidence_score:.0%}")

    try:
        tool_result = latest_attempt.tool_result
        truncated_result = {}
        for k, v in tool_result.items():
            if isinstance(v, dict) and v.get("raw") is True:
                records = v.get("result", [])
                truncated_result[k] = {
                    **{kk: vv for kk, vv in v.items() if kk != "result"},
                    "result": records[:10],
                    "result_note": f"{len(records)} total records — showing first 10",
                }
            else:
                truncated_result[k] = v

        context = f"""Original question: {question}

            Validated result:
            Tool: {latest_attempt.tool_called}
            Result: {safe_json(truncated_result)}

            Critic confidence: {latest_verdict.confidence_score * 100:.0f}%
            Rows validated: {latest_verdict.rows_validated}"""

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

        follow_up_questions = []
        try:
            followup_response = llm.invoke([
                SystemMessage(content=FOLLOWUP_PROMPT),
                HumanMessage(content=f"Question: {question}\nFinding: {final_report[:300]}"),
            ])
            raw = followup_response.content.strip()
            follow_up_questions = json.loads(raw)
            if not isinstance(follow_up_questions, list):
                follow_up_questions = []
        except Exception:
            follow_up_questions = []

        trace_entry = (
            f"REPORTER: executive finding written — "
            f"confidence {latest_verdict.confidence_score * 100:.0f}%"
        )
        logger.info(f"Reporter done | report_len={len(final_report)}")

        return {
            "final_report":        final_report,
            "follow_up_questions": follow_up_questions,
            "next_agent":          "hitl",
            "trace":               [trace_entry],
        }

    except Exception as e:
        logger.error(f"Reporter failed | error={str(e)}", exc_info=True)
        raise


# if __name__ == "__main__":
#     import pandas as pd
#     from tools.quality        import run_quality_report
#     from tools.detect_columns import detect_columns
#     from tools.aggregate      import aggregate
#     from state                import AnalysisAttempt, CriticVerdict, new_state

#     df       = pd.read_csv("data/sales.csv")
#     quality  = run_quality_report(df)
#     detected = detect_columns(df)

#     # simulate approved Analyst result
#     result = aggregate(
#         df,
#         group_by="Region",
#         value_col="Total Revenue",
#         agg="mean"
#     )

#     attempt = AnalysisAttempt(
#         attempt_number=1,
#         tool_called="aggregate",
#         tool_args={"group_by": "Region", "value_col": "Total Revenue", "agg": "mean"},
#         tool_result={"aggregate": result},
#         chart_path=None,
#     )

#     verdict = CriticVerdict(
#         attempt_number=1,
#         approved=True,
#         confidence_score=0.92,
#         reason="Result is correct, nulls handled, answers the question.",
#         rows_validated=100,
#     )

#     test_state = new_state(
#         question="What is the average revenue per region?",
#         dataset_path="data/sales.csv",
#     )
#     test_state["quality_report"]   = quality
#     test_state["detected_columns"] = detected
#     test_state["analysis_history"] = [attempt]
#     test_state["critic_history"]   = [verdict]

#     result = reporter_node(test_state)

#     print("=== Reporter output ===\n")
#     print(result["final_report"])
#     print(f"\nTrace: {result['trace'][0]}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_path:
        print("Usage: python reporter.py <file_path>")
        sys.exit(1)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("Columns:", list(df.columns))