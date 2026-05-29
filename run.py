import os
import pandas as pd
from dotenv import load_dotenv

from state                import new_state
from tools.quality        import run_quality_report
from tools.detect_columns import detect_columns
from graph.workflow       import workflow

load_dotenv()


def run_analysis(question: str, dataset_path: str) -> dict:

    # load file and run pre-analysis
    df       = pd.read_csv(dataset_path)
    quality  = run_quality_report(df)
    detected = detect_columns(df)

    # create fresh state
    state = new_state(
        question=question,
        dataset_path=dataset_path,
        source_type="csv",
        mode="single",
    )
    state["quality_report"]   = quality
    state["detected_columns"] = detected

    # run the graph
    final_state = workflow.invoke(state)

    return final_state


def print_trace(final_state: dict):
    print("\n" + "="*50)
    print("INSIGHTFLOW — AGENT TRACE")
    print("="*50)
    for entry in final_state["trace"]:
        print(f"\n  {entry}")

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(final_state["final_report"])

    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"  Attempts:    {final_state['attempts']}")
    print(f"  Confidence:  {final_state['critic_history'][-1].confidence_score * 100:.0f}%")
    print(f"  Rows used:   {final_state['critic_history'][-1].rows_validated}")


if __name__ == "__main__":
    question     = input("\nAsk a question about your data: ").strip()
    dataset_path = "data/sales.csv"

    print(f"\nRunning analysis...")
    final_state = run_analysis(question, dataset_path)
    print_trace(final_state)