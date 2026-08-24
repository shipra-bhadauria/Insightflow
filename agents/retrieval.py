import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from logger import agent_logger as logger

from state import InsightFlowState
from memory.chroma_store import save_analysis_chroma, search_similar_chroma

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

ROUTER_PROMPT = """You are a query classifier for a data analysis system.

Given a user question, classify it:

1. needs_memory: true if past analyses would help answer this question
2. complexity: "simple" or "complex"
3. route: "rag" or "direct"
   - rag: question benefits from past context
   - direct: answer directly from current dataset only

Respond ONLY with JSON:
{
  "needs_memory": true or false,
  "complexity": "simple" or "complex",
  "route": "rag" or "direct",
  "reason": "one sentence"
}"""

SYSTEM_PROMPT = """You are the Retrieval agent in InsightFlow.
Given a new question and past analyses, extract useful context.
Be generous — include anything with similarity above 0.3.
Respond ONLY in JSON:
{
  "relevant_analyses": [0],
  "context": "extracted context here",
  "has_relevant_context": true
}"""

def _classify_query(question: str) -> dict:
    """Smart Router — classify query before retrieval."""
    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=question),
        ])
        return json.loads(response.content)
    except Exception:
        return {
            "needs_memory": True,
            "complexity":   "simple",
            "route":        "rag",
            "reason":       "classification failed — defaulting to RAG",
        }

def retrieval_node(state: InsightFlowState) -> dict:

    question = state["question"]
    user_id  = state.get("user_id", "default")
    logger.info(f"Retrieval | user={user_id} | question='{question[:50]}'")

    classification = _classify_query(question)
    route        = classification.get("route", "rag")
    needs_memory = classification.get("needs_memory", True)
    complexity   = classification.get("complexity", "simple")

    trace_entry = "RETRIEVAL: searched ChromaDB —"

    # direct route — skip memory search
    if route == "direct" and not needs_memory:
        return {
            "context":    "",
            "trace":      [f"RETRIEVAL: Smart Router → direct. {classification.get('reason', '')}"],
            "next_agent": "analyst",
        }
    similar = search_similar_chroma(query=question, k=3, user_id=user_id)

    trace_entry += f" {len(similar)} past analyses found"

    if not similar:
        return {
            "context":    "",
            "trace":      [trace_entry + " — no memory yet"],
            "next_agent": "analyst",
        }

    # always use fallback based on similarity score directly
    high_similarity = [r for r in similar if r["similarity_score"] > 0.3]

    if high_similarity:
        top = high_similarity[0]
        context = (
            f"Similar past analysis:\n"
            f"Question: {top['question']}\n"
            f"Finding: {top['final_report'][:300]}\n"
            f"Confidence: {top['confidence']}"
        )
        trace_entry += f" — context from score {top['similarity_score']:.3f}"
    else:
        context = ""
        trace_entry += " — no relevant context found"

    return {
        "context":    context,
        "trace":      [trace_entry],
        "next_agent": "analyst",
    }


# if __name__ == "__main__":
#     from tools.quality        import run_quality_report
#     from tools.detect_columns import detect_columns
#     from state                import new_state
#     import pandas as pd

#     df       = pd.read_csv("data/sales.csv")
#     quality  = run_quality_report(df)
#     detected = detect_columns(df)

#     test_state = new_state(
#         question="What is the average revenue by region?",
#         dataset_path="data/sales.csv",
#     )
#     test_state["quality_report"]   = quality
#     test_state["detected_columns"] = detected

#     result = retrieval_node(test_state)

#     print("\n=== Retrieval output ===")
#     print(f"Context found: {bool(result['context'])}")
#     print(f"Context:\n{result['context'][:300] if result['context'] else 'none'}")
#     print(f"Trace: {result['trace'][0]}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_path:
        print("Usage: python retrieval.py <file_path>")
        sys.exit(1)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("Columns:", list(df.columns))