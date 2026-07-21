import os
import uuid
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# Import tools from database manager
from database_manager import search_knowledge_base, query_tabular_database


def call_model(state: MessagesState, config: RunnableConfig = None):
    """Reasoning node: Dynamically selects tools (SQL or Doc Search) and synthesizes responses."""
    messages = state["messages"]
    configurable = config.get("configurable", {}) if config else {}
    
    ollama_url = configurable.get("ollama_base_url", "http://localhost:11434")
    temperature = configurable.get("temperature", 0.0)
    top_k = configurable.get("top_k", 40)
    
    system_instruction = SystemMessage(
        content=(
            "You are an advanced local RAG & Data Analytics Agent.\n\n"
            "You have two tools available:\n"
            "1. 'query_tabular_database': Use this tool to run SQL queries against uploaded CSV/Excel tables. "
            "Use it for any mathematical aggregations, sums, counts, quarterly/monthly summaries, or column filtering.\n"
            "2. 'search_knowledge_base': Use this tool to search unstructured text documents (PDFs, Word files, text docs) "
            "for text context and guidelines.\n\n"
            "Guidelines:\n"
            "- When writing SQL queries, use SQLite standard syntax and inspect dataset column names provided in the schema.\n"
            "- Never make up or hallucinate numbers. Rely strictly on query tool outputs.\n"
            "- Keep answers concise, grounded, and present tabular findings in clean Markdown tables."
        )
    )
    
    # Safely resolve messages array types
    processed_messages = []
    for m in messages:
        if isinstance(m, tuple):
            role, content_text = m
            processed_messages.append(HumanMessage(content=content_text) if role == "user" else AIMessage(content=content_text))
        else:
            processed_messages.append(m)

    # 🔥 SLIDING WINDOW HISTORY TRIMMER:
    # Keep system prompt + the last 6 active messages to prevent context window overflow (18k token crashes)
    if len(processed_messages) > 6:
        recent_history = processed_messages[-6:]
    else:
        recent_history = processed_messages

    routing_messages = [system_instruction] + recent_history
    
    llm = ChatOllama(
        base_url=ollama_url,
        model="deepseek-r1:1.5b",
        temperature=temperature,
        num_ctx=32768,  
        num_predict=1024,
        additional_kwargs={
            "top_k": top_k,
            "num_thread": 8  
        }
    )
    
    # Bind BOTH tools to the model
    tools = [search_knowledge_base, query_tabular_database]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(routing_messages)
    return {"messages": [response]}


def route_tools(state: MessagesState) -> Literal["tools", END]:
    """Conditional Edge router checking if LLM requested tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# =====================================================================
# LANGGRAPH AGENT WORKFLOW STATE MACHINE
# =====================================================================
workflow = StateGraph(MessagesState)

# List of tools provided to ToolNode
all_tools = [search_knowledge_base, query_tabular_database]

# 1. Define processing block nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(all_tools))

# 2. Build execution routing map
workflow.add_edge(START, "agent")

# Agent checks if it called a tool or is ready to return final answer
workflow.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "__end__": END  
    }
)

# Loop back to agent after tool execution so it can read tool results and form response
workflow.add_edge("tools", "agent")

agent_app = workflow.compile()