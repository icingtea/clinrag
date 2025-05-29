import streamlit as st 
from langgraph_flow.graph_pipeline import assemble_graph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

def run_app():    

    st.set_page_config(
        page_title="clinRAG",
        page_icon="🧪",
        layout="wide"
    )

    memory: MemorySaver = MemorySaver()

    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {
            "question": None,
            "memory": [],
            "metadata": {},
            "filter": {},
            "context": [],
            "response": "",
            "error": "",
            "recent_context": ""
        }

    if "graph_config" not in st.session_state:
        st.session_state.graph_config = {
            "configurable": {"thread_id": "demo"}
        }

    graph = assemble_graph(memory=memory)

    st.title("clinRAG")

    for message in st.session_state.graph_state["memory"]:
        if isinstance(message, HumanMessage):
            role = "🌻"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "🤖"
            content = message.content
        else:
            role = "🤖"
            content = getattr(message, "content", str(message))
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Ask a clinical trial related question"):
        with st.chat_message("🌻"):
            st.markdown(prompt)

        st.session_state.graph_state["question"] = prompt

        new_state = graph.invoke(
            st.session_state.graph_state, 
            st.session_state.graph_config
        )
        st.session_state.graph_state = new_state

        with st.chat_message("assistant", avatar="🤖"):
            reply = new_state.get("response", "[ERROR] Could not get response.")
            st.markdown(reply)

if __name__ == "__main__":
    run_app()