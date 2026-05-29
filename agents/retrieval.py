import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from state import InsightFlowState
from memory.faiss_store import search_similar

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_ROUTING", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

SYSTEM_PROMPT = """You are the Retrieval agent in InsightFlow.
Given a new question and past analyses, extract useful context.
Be generous — include anything with similarity above 0.3.
Respond ONLY in JSON:
{
  "relevant_analyses": [0],
  "context": "extracted context here",
  "has_relevant_context": true
}"""


def retrieval_node(state: InsightFlowState) -> dict:

    question = state["question"]
    similar  = search_similar(query=question, k=3)

    

    trace_entry = f"RETRIEVAL: searched FAISS — {len(similar)} past analyses found"

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


if __name__ == "__main__":
    from tools.quality        import run_quality_report
    from tools.detect_columns import detect_columns
    from state                import new_state
    import pandas as pd

    df       = pd.read_csv("data/sales.csv")
    quality  = run_quality_report(df)
    detected = detect_columns(df)

    test_state = new_state(
        question="What is the average revenue by region?",
        dataset_path="data/sales.csv",
    )
    test_state["quality_report"]   = quality
    test_state["detected_columns"] = detected

    result = retrieval_node(test_state)

    print("\n=== Retrieval output ===")
    print(f"Context found: {bool(result['context'])}")
    print(f"Context:\n{result['context'][:300] if result['context'] else 'none'}")
    print(f"Trace: {result['trace'][0]}")
