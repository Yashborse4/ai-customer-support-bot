from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.schemas.state import SupportState
from src.agents.support_agent import support_agent_node
from src.tools.retrieval_tool import retrieve_company_info

def create_workflow() -> CompiledStateGraph:
    """Creates and compiles the LangGraph workflow for the support bot.

    This workflow orchestrates the support agent node and handles tool execution
    seamlessly with checkpointer-based conversation state persistence.

    Returns:
        The compiled stateful LangGraph workflow ready for execution.
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(SupportState)

    # Define the nodes
    workflow.add_node("agent", support_agent_node)
    
    # Prebuilt ToolNode to handle tool execution
    tool_node = ToolNode([retrieve_company_info])
    workflow.add_node("tools", tool_node)

    # Define the edges
    workflow.add_edge(START, "agent")
    
    # Conditional edge: if agent calls a tool, go to tools node, else end
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # After tools are executed, go back to the agent to summarize/answer
    workflow.add_edge("tools", "agent")

    # Initialize MemorySaver checkpointer for state preservation
    checkpointer = MemorySaver()

    # Compile the graph with the checkpointer
    return workflow.compile(checkpointer=checkpointer)

# Singleton instance of the compiled graph
support_bot_graph: CompiledStateGraph = create_workflow()
