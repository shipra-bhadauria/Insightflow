"""
InsightFlow — Multi-Agent Router
Routes questions to the right agent/pipeline based on intent.
"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

ROUTER_PROMPT = """You are a query router for InsightFlow — an AI data analyst system.

Given a user question and context, decide which agent should handle it.

Routes:
1. "analysis"   — Data analysis questions that need the full agent pipeline
                  (aggregation, charts, comparisons, distributions, correlations)
                  Examples: "unit sales per manufacturer", "average price by category"

2. "forecast"   — Future prediction questions (needs date column in dataset)
                  Examples: "forecast next month sales", "predict revenue trend"

3. "anomaly"    — Outlier/anomaly detection questions
                  Examples: "any anomalies in sales?", "find outliers in price"

4. "what_if"    — Scenario/simulation questions
                  Examples: "what if price increases by 10%?", "impact of 20% sales drop"

5. "direct_llm" — General knowledge questions that don't need dataset analysis
                  Examples: "what is pandas?", "explain correlation", "what does median mean?"

6. "document"   — Questions about uploaded PDF/image content
                  Examples: "summarize this document", "what does this chart show?"

7. "chat"       — Casual conversation, greetings, clarifications
                  Examples: "hello", "what can you do?", "help me understand"

Respond ONLY with JSON:
{
  "route": "analysis|forecast|anomaly|what_if|direct_llm|document|chat",
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}"""


def route_query(
    question: str,
    has_dataset: bool = True,
    source_type: str = "csv",
) -> dict:
    """
    Route a question to the appropriate agent.
    Returns: {route, confidence, reason}
    """
    context = f"Question: {question}\nDataset loaded: {has_dataset}\nSource type: {source_type}"

    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=context),
        ])
        result = json.loads(response.content.strip())
        return result
    except Exception:
        # fallback — if dataset loaded use analysis, else direct_llm
        return {
            "route":      "analysis" if has_dataset else "direct_llm",
            "confidence": 0.5,
            "reason":     "Router failed — using fallback",
        }


def get_route_label(route: str) -> str:
    """Human readable label for route."""
    labels = {
        "analysis":   "📊 Full Analysis Pipeline",
        "forecast":   "📈 Forecast Agent",
        "anomaly":    "🔍 Anomaly Detector",
        "what_if":    "🎯 What-If Simulator",
        "direct_llm": "💬 Direct LLM",
        "document":   "📄 Document Agent",
        "chat":       "💬 Chat",
    }
    return labels.get(route, route)
