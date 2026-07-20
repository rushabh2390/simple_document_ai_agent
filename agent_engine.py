import os
import uuid
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# Import our search tool from the database manager
from database_manager import search_knowledge_base

def first_tool_call(state: MessagesState):
    """
    Seeding Node: Hardcodes a tool call using the user's latest message 
    without spending processing cycles on an LLM inference pass.
    """
    messages = state["messages"]
    if not messages:
        return {"messages": []}
        
    # Safely extract user query string from history types
    last_raw_message = messages[-1]
    if isinstance(last_raw_message, tuple):
        _, user_query = last_raw_message
    elif hasattr(last_raw_message, "content"):
        user_query = last_raw_message.content
    else:
        user_query = str(last_raw_message)

    # We manually build the tool call payload structure.
    # This acts exactly like an LLM requesting a tool execution block.
    forced_tool_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": user_query},
                "id": f"call_{uuid.uuid4().hex[:12]}" # Unique token for tracking trace integrity
            }
        ]
    )
    
    return {"messages": [forced_tool_message]}


def call_model(state: MessagesState, config: RunnableConfig = None):
    """Reasoning node: Reviews tool context data and writes the final response."""
    messages = state["messages"]
    configurable = config.get("configurable", {}) if config else {}
    
    ollama_url = configurable.get("ollama_base_url", "http://localhost:11434")
    temperature = configurable.get("temperature", 0.0)
    top_k = configurable.get("top_k", 40)
    
    system_instruction = SystemMessage(
        content=(
            "You are a local RAG Agent. Review the context provided by your retrieval tools. "
            "If the context contains data grids, metrics, or logs, construct a markdown table. "
            "Keep answers clear, grounded, and concise."
        )
    )
    
    # Safely clean and resolve messages array types
    processed_messages = []
    for m in messages:
        if isinstance(m, tuple):
            role, content_text = m
            processed_messages.append(HumanMessage(content=content_text) if role == "user" else AIMessage(content=content_text))
        else:
            processed_messages.append(m)

    routing_messages = [system_instruction] + processed_messages
    
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
    
    # We maintain binding syntax so history serialization remains consistent
    tools = [search_knowledge_base]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(routing_messages)
    return {"messages": [response]}


def route_tools(state: MessagesState) -> Literal["tools", END]:
    """Conditional Edge router tracking loop iterations."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# =====================================================================
# SEEDED LANGGRAPH AGENT WORKFLOW STATE MACHINE
# =====================================================================
workflow = StateGraph(MessagesState)

# 1. Define processing block nodes
workflow.add_node("startup_tool_seeder", first_tool_call)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_knowledge_base]))

# 2. Build the execution layout routing map
# 🔥 FORCE execution to start at our seeding node
workflow.add_edge(START, "startup_tool_seeder")

# This forces the generated call straight into your database tool node
workflow.add_edge("startup_tool_seeder", "tools")

# After the tool fetches metrics, execution loops into your agent node to answer
workflow.add_edge("tools", "agent")

# The agent checks if it needs any further follow-up or exits cleanly
workflow.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "__end__": END  
    }
)

agent_app = workflow.compile()