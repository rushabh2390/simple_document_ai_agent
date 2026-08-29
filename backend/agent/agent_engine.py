import uuid
from pathlib import Path

from config.config import settings
from db.database_manager import query_tabular_database, search_knowledge_base
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph


def retrieve_from_db(state: MessagesState, config: RunnableConfig = None):
    """
    FIRST NODE IN GRAPH:
    - If CSV/Tabular data exists: Calls Ollama to inspect schema and generate a precise SQL query, then executes query_tabular_database.
    - If PDF/Text document exists: Calls search_knowledge_base with BM25.
    """
    config = config or {}
    configurable = config.get("configurable", {})
    ollama_url = configurable.get("ollama_base_url", "http://localhost:11434")

    messages = state["messages"]

    # 1. Extract latest user query
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

    # 2. Check if CSV/Tabular database exists
    tabular_db_path = Path(settings.STATIC_ASSET_DIR) / "dynamic_tabular_data.db"
    has_tabular_data = tabular_db_path.exists()

    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

    # =====================================================================
    # CASE A: TABULAR DATA (CSV / EXCEL)
    # =====================================================================
    if has_tabular_data:
        tool_name = "query_tabular_database"

        # Extract schema dynamically from SQLite
        schema_info = ""
        try:
            conn = sqlite3.connect(tabular_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = cursor.fetchall()

            for tbl in tables:
                t_name = tbl[0]
                cursor.execute(f"PRAGMA table_info({t_name});")
                cols = cursor.fetchall()
                col_defs = [f"{c[1]} ({c[2]})" for c in cols]
                schema_info += f"Table '{t_name}' columns: {', '.join(col_defs)}\n"

            conn.close()
        except Exception:
            schema_info = "Table schema unreadable."

        # Prompt Ollama to generate ONLY the raw SQL query
        sql_generator_llm = ChatOllama(
            base_url=ollama_url,
            model="qwen2.5-coder:3b",
            temperature=0.0,
        )

        sql_prompt = [
            SystemMessage(
                content=(
                    "You are a SQL Expert.\n"
                    "Given the SQLite database schema below and a user's prompt, generate ONLY a valid executable SQL query.\n\n"
                    f"DATABASE SCHEMA:\n{schema_info}\n"
                    "CRITICAL RULES:\n"
                    "1. Respond ONLY with the raw SQL code inside a clean string. Do NOT use markdown tags (no ```sql), no explanation, no preamble.\n"
                    "2. Make sure WHERE filters strictly match user query criteria (e.g., YEAR_ID = 2004 or YEAR_ID = '2004').\n"
                    "3. Perform aggregations (SUM, AVG, COUNT, GROUP BY) when asked for totals, quarterlies, or summaries."
                )
            ),
            HumanMessage(content=f"User request: {user_query}"),
        ]

        # Get generated SQL from Ollama
        generated_sql_res = sql_generator_llm.invoke(sql_prompt, config=config)
        generated_sql = (
            generated_sql_res.content.strip()
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )

        # Fallback safeguard if model output is empty
        if not generated_sql.upper().startswith("SELECT"):
            generated_sql = "SELECT * FROM tbl_data LIMIT 10"

        # Execute generated SQL query via tool
        retrieved_result = query_tabular_database.invoke(
            {"sql_query": generated_sql}, config=config
        )

    # =====================================================================
    # CASE B: UNSTRUCTURED DATA (PDF / DOCX / MD)
    # =====================================================================
    else:
        tool_name = "search_knowledge_base"
        retrieved_result = search_knowledge_base.invoke(
            {"query": user_query}, config=config
        )

    # Attach retrieved DB context as a ToolMessage for synthesis node
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
        model="qwen2.5-coder:3b",
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
