from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_flow.state_schema import State
from langgraph.checkpoint.memory import MemorySaver

from langgraph_flow.graph_nodes import (
    query_metadata_extraction, 
    db_filter_assembly, 
    vector_search,
    chat_response,
    error_response,
    error_check
)

def assemble_graph(memory: MemorySaver):
    builder = StateGraph(state_schema=State)

    builder.add_node("metadata extraction", query_metadata_extraction)
    builder.add_node("filter creation", db_filter_assembly)
    builder.add_node("vector search", vector_search)
    builder.add_node("chat response", chat_response)
    builder.add_node("error response", error_response)

    builder.add_edge(START, "metadata extraction")
    builder.add_conditional_edges("metadata extraction", error_check, {True: "error response", False: "filter creation"})
    builder.add_conditional_edges("filter creation", error_check, {True: "error response", False: "vector search"})
    builder.add_conditional_edges("vector search", error_check, {True: "error response", False: "chat response"})
    builder.add_conditional_edges("chat response", error_check, {True: "error response", False: END})

    app = builder.compile(checkpointer=memory)
    return app