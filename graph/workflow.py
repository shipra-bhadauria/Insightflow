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
    # reads next_agent from state — set by the Critic
    return state["next_agent"]


def build_graph() -> StateGraph:

    graph = StateGraph(InsightFlowState)

    graph.add_node("planner",   planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analyst",   analyst_node)
    graph.add_node("critic",    critic_node)
    graph.add_node("reporter",  reporter_node)

    graph.add_edge("planner",   "retrieval")
    graph.add_edge("retrieval", "analyst")
    graph.add_edge("analyst",   "critic")

    # conditional edge after Critic — this is the retry loop
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "analyst":  "analyst",   # rejected — retry
            "reporter": "reporter",  # approved — move forward
        }
    )

    # start at Planner
    graph.set_entry_point("planner")

    return graph.compile()


# compiled graph — import this in run.py
workflow = build_graph()