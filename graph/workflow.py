import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from state import InsightFlowState
from agents.planner  import planner_node
from agents.analyst  import analyst_node
from agents.critic   import critic_node
from agents.reporter import reporter_node
from agents.retrieval import retrieval_node


def route_after_critic(state: InsightFlowState) -> str:
    return state["next_agent"]


def route_from_start(state: InsightFlowState) -> str:
    """Dashboard mode — skip LLM Planner, go straight to retrieval/analyst."""
    if state.get("mode") == "dashboard":
        return "retrieval"
    return "planner"


def hitl_node(state: InsightFlowState) -> dict:
    """HITL checkpoint — UI handles approve/reject via buttons."""
    return {
        "approved_by_human": False,
    }


def build_graph() -> StateGraph:

    graph = StateGraph(InsightFlowState)

    graph.add_node("planner",   planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analyst",   analyst_node)
    graph.add_node("critic",    critic_node)
    graph.add_node("reporter",  reporter_node)
    graph.add_node("hitl",      hitl_node)

    graph.add_edge("planner",   "retrieval")
    graph.add_edge("retrieval", "analyst")
    graph.add_edge("analyst",   "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "analyst":  "analyst",
            "reporter": "reporter",
        }
    )

    graph.add_edge("reporter", "hitl")
    graph.add_edge("hitl",     END)

    graph.set_conditional_entry_point(
        route_from_start,
        {"planner": "planner", "retrieval": "retrieval"}
    )

    return graph.compile()


# compiled graph — import this in run.py
workflow = build_graph()