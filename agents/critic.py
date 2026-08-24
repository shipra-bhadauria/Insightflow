import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import safe_json
from state import InsightFlowState, CriticVerdict
from logger import agent_logger as logger

import pandas as pd



load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

SYSTEM_PROMPT = """You are the Critic agent in InsightFlow — an AI data analyst system.

Your job: validate the Analyst's output and decide if it should be approved or rejected.

Check these three things in order:
1. NULL HANDLING — were nulls dropped before calculation? Check rows_used vs total rows.
2. STATISTICAL SOUNDNESS — does the result make mathematical sense? No infinite values, no negative counts.
3. ANSWER QUALITY — does the result actually answer the user's original question?

SPECIAL RULE FOR RAW RESULTS:
- If tool result contains agg="raw" — this means exact per-row lookup was requested
- Approve if result is a non-empty list of records
- Do NOT reject because it shows all rows — that is the correct behavior
- Do NOT reject because no aggregation was done — raw mode is intentional

SPECIAL RULE FOR COUNT RESULTS:
- If question asks about categories with no numeric metric — count is the correct answer
- Approve if result contains counts per category
- Do NOT reject because no numeric metric (revenue, price etc) was calculated

Rules:
- Be strict but fair — only reject if there is a real problem
- confidence_score: 0.90-1.0 = excellent, 0.75-0.89 = good, 0.60-0.74 = acceptable
- If rejecting, be specific in the reason so the Analyst knows exactly what to fix
- rows_validated should be the rows_used value from the tool result

Respond ONLY in this JSON format, no extra text:
{
  "approved": true or false,
  "confidence_score": 0.0 to 1.0,
  "reason": "plain English explanation",
  "rows_validated": number or null
}"""


def critic_node(state: InsightFlowState) -> dict:

    question     = state["question"]
    attempts     = state["attempts"]
    max_attempts = state["max_attempts"]
    logger.info(f"Critic | attempt={attempts} | tool={state['analysis_history'][-1].tool_called}")

    try:
        latest_attempt = state["analysis_history"][-1]
        quality_report = state.get("quality_report")

        tool_result = latest_attempt.tool_result
        truncated_result = {}
        for k, v in tool_result.items():
            if isinstance(v, dict) and v.get("raw") is True:
                records = v.get("result", [])
                truncated_result[k] = {
                    **{kk: vv for kk, vv in v.items() if kk != "result"},
                    "result": records[:5],
                    "result_truncated": f"showing 5 of {len(records)} records",
                }
            else:
                truncated_result[k] = v

        context = f"""Original question: {question}

    Tool called: {latest_attempt.tool_called}
    Tool args: {json.dumps(latest_attempt.tool_args)}
    Tool result: {safe_json(truncated_result)}
    Attempt number: {latest_attempt.attempt_number}"""

        if quality_report:
            context += f"\n\nData quality context:"
            context += f"\nTotal rows in file: {quality_report['total_rows']}"
            context += f"\nNull counts: {json.dumps(quality_report['null_counts'])}"

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context)
        ])

        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        parsed = json.loads(content.strip())

        verdict = CriticVerdict(
            attempt_number   = latest_attempt.attempt_number,
            approved         = parsed["approved"],
            confidence_score = parsed["confidence_score"],
            reason           = parsed["reason"],
            rows_validated   = parsed.get("rows_validated"),
        )

        if not verdict.approved and attempts >= max_attempts:
            verdict = CriticVerdict(
                attempt_number   = latest_attempt.attempt_number,
                approved         = True,
                confidence_score = 0.60,
                reason           = f"Max attempts ({max_attempts}) reached — approving best available result.",
                rows_validated   = verdict.rows_validated,
            )

        if verdict.approved:
            next_agent  = "reporter"
            trace_entry = f"CRITIC: approved — confidence {verdict.confidence_score*100:.0f}% — {verdict.reason[:60]}"
            logger.info(f"Critic done | approved=True | confidence={verdict.confidence_score:.0%}")
        else:
            next_agent  = "analyst"
            trace_entry = f"CRITIC: rejected — {verdict.reason[:80]}"
            logger.warning(f"Critic done | approved=False | reason={verdict.reason[:80]}")

        return {
            "critic_history": [verdict],
            "next_agent":     next_agent,
            "trace":          [trace_entry],
        }

    except Exception as e:
        logger.error(f"Critic failed | error={str(e)}", exc_info=True)
        raise


# if __name__ == "__main__":
#     import pandas as pd
#     from tools.quality        import run_quality_report
#     from tools.detect_columns import detect_columns
#     from tools.aggregate      import aggregate
#     from state                import PlanStep, AnalysisAttempt, new_state

#     df       = pd.read_csv("data/sales.csv")
#     quality  = run_quality_report(df)
#     detected = detect_columns(df)

#     # simulate what the Analyst would have produced
#     result = aggregate(df, group_by="Region", value_col="Total Revenue", agg="mean")

#     attempt = AnalysisAttempt(
#         attempt_number=1,
#         tool_called="aggregate",
#         tool_args={"group_by": "Region", "value_col": "Total Revenue", "agg": "mean"},
#         tool_result={"aggregate": result},
#         chart_path=None,
#     )

#     test_state = new_state(
#         question="What is the average revenue per region?",
#         dataset_path="data/sales.csv",
#     )
#     test_state["quality_report"]   = quality
#     test_state["detected_columns"] = detected
#     test_state["analysis_history"] = [attempt]
#     test_state["attempts"]         = 1

#     result = critic_node(test_state)

#     print("=== Critic output ===\n")
#     verdict = result["critic_history"][0]
#     print(f"Approved:        {verdict.approved}")
#     print(f"Confidence:      {verdict.confidence_score * 100:.0f}%")
#     print(f"Reason:          {verdict.reason}")
#     print(f"Rows validated:  {verdict.rows_validated}")
#     print(f"Next agent:      {result['next_agent']}")
#     print(f"Trace:           {result['trace'][0]}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_path:
        print("Usage: python critic.py <file_path>")
        sys.exit(1)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("Columns:", list(df.columns))