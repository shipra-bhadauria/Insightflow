"""
InsightFlow — Eval Harness
Run: python tests/eval_runner.py
"""
import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from state                import new_state
from tools.quality        import run_quality_report
from tools.detect_columns import detect_columns
from graph.workflow       import workflow


@dataclass
class EvalResult:
    case_id:          str
    question:         str
    passed:           bool
    confidence:       float
    expected_tool:    str
    actual_tools:     list
    tool_match:       bool
    keyword_matches:  list
    keyword_score:    float
    final_report:     str
    latency_s:        float
    error:            Optional[str] = None


@dataclass
class EvalReport:
    total:            int = 0
    passed:           int = 0
    failed:           int = 0
    avg_confidence:   float = 0.0
    avg_latency_s:    float = 0.0
    tool_accuracy:    float = 0.0
    keyword_accuracy: float = 0.0
    results:          list = field(default_factory=list)


def _load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load dataset from data/ folder."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def run_eval_case(case: dict) -> EvalResult:
    """Run a single eval case."""
    start = time.time()
    try:
        df       = _load_dataset(case["dataset"])
        quality  = run_quality_report(df)
        detected = detect_columns(df)

        state = new_state(
            question=case["question"],
            dataset_path=os.path.join("data", case["dataset"]),
            source_type="csv" if case["dataset"].endswith(".csv") else "excel",
            mode="single",
        )
        state["quality_report"]   = quality
        state["detected_columns"] = detected

        result = workflow.invoke(state)
        latency = time.time() - start

        # extract metrics
        verdict    = result["critic_history"][-1]
        confidence = verdict.confidence_score
        report     = result.get("final_report", "")
        attempts   = result["analysis_history"]

        # tools used
        actual_tools = []
        for attempt in attempts:
            for key in attempt.tool_result.keys():
                if key not in actual_tools:
                    actual_tools.append(key)

        # tool match
        expected_tool = case.get("expected_tool", "")
        tool_match = any(expected_tool in t for t in actual_tools)

        # keyword check
        expected_keywords = case.get("expected_keywords", [])
        report_lower = report.lower()
        matched = [kw for kw in expected_keywords if kw.lower() in report_lower]
        keyword_score = len(matched) / len(expected_keywords) if expected_keywords else 1.0

        # pass/fail
        min_confidence = case.get("expected_min_confidence", 0.70)
        passed = (
            confidence >= min_confidence and
            tool_match and
            keyword_score >= 0.5
        )

        return EvalResult(
            case_id         = case["id"],
            question        = case["question"],
            passed          = passed,
            confidence      = confidence,
            expected_tool   = expected_tool,
            actual_tools    = actual_tools,
            tool_match      = tool_match,
            keyword_matches = matched,
            keyword_score   = keyword_score,
            final_report    = report[:200],
            latency_s       = round(latency, 2),
        )

    except Exception as e:
        return EvalResult(
            case_id         = case["id"],
            question        = case["question"],
            passed          = False,
            confidence      = 0.0,
            expected_tool   = case.get("expected_tool", ""),
            actual_tools    = [],
            tool_match      = False,
            keyword_matches = [],
            keyword_score   = 0.0,
            final_report    = "",
            latency_s       = round(time.time() - start, 2),
            error           = str(e)[:200],
        )


def run_eval(cases_path: str = None) -> EvalReport:
    """Run all eval cases and return report."""
    if cases_path is None:
        cases_path = os.path.join(os.path.dirname(__file__), "eval_cases.json")

    cases = json.loads(open(cases_path).read())
    report = EvalReport(total=len(cases))
    results = []

    print(f"\n{'='*60}")
    print(f"InsightFlow Eval Harness — {len(cases)} cases")
    print(f"{'='*60}\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:50]}...")
        result = run_eval_case(case)
        results.append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} | conf={result.confidence:.0%} | "
              f"tools={result.actual_tools} | "
              f"keywords={result.keyword_score:.0%} | "
              f"latency={result.latency_s}s")
        if result.error:
            print(f"  ERROR: {result.error}")
        print()

    # aggregate
    report.results         = [r.__dict__ for r in results]
    report.passed          = sum(1 for r in results if r.passed)
    report.failed          = report.total - report.passed
    report.avg_confidence  = round(sum(r.confidence for r in results) / len(results), 3)
    report.avg_latency_s   = round(sum(r.latency_s for r in results) / len(results), 2)
    report.tool_accuracy   = round(sum(1 for r in results if r.tool_match) / len(results), 3)
    report.keyword_accuracy = round(sum(r.keyword_score for r in results) / len(results), 3)

    print(f"{'='*60}")
    print(f"RESULTS: {report.passed}/{report.total} passed "
          f"({report.passed/report.total:.0%})")
    print(f"Avg confidence:  {report.avg_confidence:.0%}")
    print(f"Tool accuracy:   {report.tool_accuracy:.0%}")
    print(f"Keyword accuracy:{report.keyword_accuracy:.0%}")
    print(f"Avg latency:     {report.avg_latency_s}s")
    print(f"{'='*60}\n")

    # save report
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    open(report_path, "w").write(json.dumps(report.__dict__, indent=2))
    print(f"Report saved: {report_path}")

    return report


if __name__ == "__main__":
    cases_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(cases_path)
