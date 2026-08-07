import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

from config.config import settings
from database.database_manager import query_tabular_database, search_knowledge_base


def retrieve_from_db(state: MessagesState, config: RunnableConfig = {}):
    """
    FIRST NODE IN GRAPH:
    Automatically queries the database (CSV SQL or PDF FTS5 chunks)
    BEFORE sending context to the LLM agent.
    """
    messages = state["messages"]

    # 1. Extract latest human user query
    user_query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
        elif isinstance(m, tuple) and m[0] == "user":
            user_query = m[1]
            break

    if not user_query:
        return {"messages": []}

    # 2. Check if CSV / Tabular database exists on disk
    tabular_db_path = Path(settings.STATIC_ASSET_DIR) / "dynamic_tabular_data.db"
    has_tabular_data = tabular_db_path.exists()

    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

    # 3. Execute appropriate tool directly
    if has_tabular_data:
        # For CSVs, run tabular query (or SELECT default)
        sql_query = "SELECT * FROM tbl_data LIMIT 10"
        retrieved_result = query_tabular_database.invoke(
            {"sql_query": sql_query}, config=config
        )
        tool_name = "query_tabular_database"
    else:
        # For PDFs, execute FTS5 keyword search across document_chunks
        retrieved_result = search_knowledge_base.invoke(
            {"query": user_query}, config=config
        )
        tool_name = "search_knowledge_base"

    # 4. Attach retrieved DB context as a ToolMessage for the LLM
    tool_message = ToolMessage(
        content=retrieved_result,
        name=tool_name,
        tool_call_id=tool_call_id,
    )

    return {"messages": [tool_message]}


def call_model(state: MessagesState, config: RunnableConfig = None):
    """
    SECOND NODE IN GRAPH:
    Reads pre-retrieved DB context and synthesizes the final response.
    """
    messages = state["messages"]
    configurable = config.get("configurable", {}) if config else {}

    ollama_url = configurable.get("ollama_base_url", "http://localhost:11434")
    temperature = configurable.get("temperature", 0.0)
    top_k = configurable.get("top_k", 40)

    system_instruction = SystemMessage(
        content=(
            "You are an advanced local RAG & Data Analytics Agent.\n"
            "Database search results have ALREADY been retrieved for you in the message history.\n\n"
            "STRICT ANSWERING RULES:\n"
            "1. Answer the user's question directly using ONLY the retrieved database context provided.\n"
            "2. Never make up facts, numbers, or package designs.\n"
            "3. If tabular findings are present, format them clearly in Markdown tables.\n"
            "4. Keep your answer grounded, concise, and direct."
        )
    )

    # Convert tuple message formats if present
    processed_messages = []
    for m in messages:
        if isinstance(m, tuple):
            role, content_text = m
            processed_messages.append(
                HumanMessage(content=content_text)
                if role == "user"
                else AIMessage(content=content_text)
            )
        else:
            processed_messages.append(m)

    # Keep last 6 active messages for sliding history window
    recent_history = (
        processed_messages[-6:] if len(processed_messages) > 6 else processed_messages
    )
    routing_messages = [system_instruction] + recent_history

    llm = ChatOllama(
        base_url=ollama_url,
        model="deepseek-r1:1.5b",
        temperature=temperature,
        num_ctx=32768,
        num_predict=1024,
        additional_kwargs={"top_k": top_k, "num_thread": 8},
    )

    response = llm.invoke(routing_messages)
    return {"messages": [response]}


# =====================================================================
# LANGGRAPH WORKFLOW SETUP (GUARANTEED DB RETRIEVAL FIRST)
# =====================================================================
workflow = StateGraph(MessagesState)

# Define nodes
workflow.add_node("retrieve_from_db", retrieve_from_db)
workflow.add_node("agent", call_model)

# Sequence: START -> DB Retrieval -> LLM Answer Synthesis -> END
workflow.add_edge(START, "retrieve_from_db")
workflow.add_edge("retrieve_from_db", "agent")
workflow.add_edge("agent", END)

agent_app = workflow.compile()
