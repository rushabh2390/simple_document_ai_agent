import os
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# Import our search tool from the database manager
from database_manager import search_knowledge_base

def call_model(state: MessagesState, config: RunnableConfig = None):
    """Reasoning node: Analyzes current chat history and decides if a tool call is needed."""
    messages = state["messages"]
    
    # Safely handle config reading if config is passed as None
    configurable = config.get("configurable", {}) if config else {}
    
    # 1. Fetch dynamic hyperparameters passed down from the Streamlit UI sliders
    ollama_url = configurable.get("ollama_base_url", "http://localhost:11434")
    temperature = configurable.get("temperature", 0.0)
    top_k = configurable.get("top_k", 40)
    
    # 2. Inject a strict, concise system instruction to guide the agent
    system_instruction = SystemMessage(
        content=(
            "You are a local RAG Agent. If the user query requires tabular data, "
            "metrics, or documentation lookups, call 'search_knowledge_base' immediately. "
            "Be direct; do not elaborate until tools provide context."
        )
    )
    
    # 3. STRATEGY: Build strict LangChain tracking instances out of the state payload
    if not messages:
        return {"messages": [system_instruction]}
        
    last_raw_message = messages[-1]
    
    # If LangGraph received a tuple ("user", "text") from app_ui, unpack it safely
    if isinstance(last_raw_message, tuple):
        role, content_text = last_raw_message
        if role == "user":
            active_user_msg = HumanMessage(content=content_text)
        else:
            active_user_msg = AIMessage(content=content_text)
    else:
        # If it is already a message class instance, pass it straight through
        active_user_msg = last_raw_message

    # Assemble the lean routing payload (System rules + the exact user question)
    routing_messages = [system_instruction, active_user_msg]
    
    # 4. Initialize the model with performance settings
    llm = ChatOllama(
        base_url=ollama_url,
        model="llama3.2",
        temperature=temperature,
        num_ctx=4096,  # Kept small to make the prompt prefill phase instant
        num_predict=1024,
        additional_kwargs={
            "top_k": top_k,
            "num_thread": 8  # Instructs Ollama to use parallel core processing threads
        }
    )
    
    # Expose tool definitions to the model's environment
    tools = [search_knowledge_base]
    llm_with_tools = llm.bind_tools(tools)
    
    # 5. Invoke the model with the clear, structured object array
    response = llm_with_tools.invoke(routing_messages)
    return {"messages": [response]}

def route_tools(state: MessagesState) -> Literal["tools", END]:
    """Conditional Edge router checking if the model requested tool execution."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# =====================================================================
# BUILD THE LANGGRAPH AGENT WORKFLOW STATE MACHINE
# =====================================================================
workflow = StateGraph(MessagesState)

# Define our processing block nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_knowledge_base]))

# Build the execution layout routing map
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "__end__": END  # Directly mapping the string value returned by END
    }
)
workflow.add_edge("tools", "agent")

# Compile our graph architecture into an active runnable app engine
agent_app = workflow.compile()