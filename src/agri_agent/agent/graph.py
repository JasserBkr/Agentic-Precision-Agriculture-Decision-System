"""
Week 5-6 deliverable: the actual LangGraph StateGraph wiring — agent node,
tools node, conditional edge between them, matching the corrected
architecture diagram (pipeline-then-agent, agent/tools loop only for
follow-up evidence).

Not yet implemented. Rough skeleton below for when you get here.
"""

from langgraph.graph import END, StateGraph

from agri_agent.agent.state import AgentState


def agent_node(state: AgentState) -> AgentState:
    """
    TODO (Week 5-6): the LLM reasoning step. Given state (forecast bundle
    + raw signals + reasoning_trace so far), decide whether to call a
    follow-up tool or whether there's enough evidence to synthesize a
    recommendation. Append to reasoning_trace either way.
    """
    raise NotImplementedError


def tools_node(state: AgentState) -> AgentState:
    """TODO: execute whichever tool the agent node called, update state."""
    raise NotImplementedError


def should_continue(state: AgentState) -> str:
    """
    TODO: the conditional edge function — return "tools" if the agent's
    last message requested a tool call, else "synthesize".
    """
    raise NotImplementedError


def synthesize_node(state: AgentState) -> AgentState:
    """
    TODO: produce the final recommendation dict (see the recommendation
    card mockup discussed with Claude) from the accumulated
    reasoning_trace and forecast_bundle.
    """
    raise NotImplementedError


def build_graph():
    """TODO: wire the nodes above into a compiled StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "synthesize": "synthesize"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesize", END)
    return graph.compile()
