import os
from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

load_dotenv()


# --- 1. State Definition ---
class AgentState(TypedDict):
    # Using 'add' ensures new messages are appended to the list rather than overwriting it
    messages: Annotated[list[BaseMessage], add]


# --- 2. Tool Definition ---
@tool
def execute_mssql_query(query: str) -> str:
    """
    Executes a T-SQL query against the configured Microsoft SQL Server database.
    Input should be a raw SQL string.
    """
    # Note: Import inside the tool to keep the global scope clean
    import mssql_python

    connection_config = (
        f"Server={os.getenv('SQL_SERVER_HOST')},{os.getenv('SQL_SERVER_PORT')};"
        f"Database={os.getenv('SQL_SERVER_DATABASE')};"
        f"UID={os.getenv('SQL_SERVER_USER')};"
        f"PWD={os.getenv('SQL_SERVER_PASSWORD')};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )

    try:
        print(f"\n[Tool] Executing SQL: {query}")
        with mssql_python.connect(connection_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return str(cursor.fetchall())
    except Exception as e:
        return f"Database Error: {str(e)}"


# --- 3. Agent Node Logic ---
def sql_generation_node(state: AgentState):
    """
    Node that processes the user request and generates a SQL query.
    """
    # Initialize the LLM (Adjust model name as per your Ollama setup)
    llm = ChatOllama(model="qwen3.5:4b")

    system_instructions = SystemMessage(
        content=(
            "You are a Microsoft SQL Server expert. "
            "Translate the user's natural language request into a valid T-SQL query. "
            "Output ONLY the SQL string. Do not use markdown blocks or explanations."
        )
    )

    # Prepend instructions to the conversation history
    full_prompt = [system_instructions] + state["messages"]

    response = llm.invoke(full_prompt)

    # We return a list so the 'add' operator appends it to the state
    return {"messages": [response]}


# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)

# Add our processing node
workflow.add_node("sql_writer", sql_generation_node)

# Define the flow: Start -> Writer -> End
workflow.add_edge(START, "sql_writer")
workflow.add_edge("sql_writer", END)

# Compile the graph
sql_assistant = workflow.compile()

# --- 5. Execution Example ---
if __name__ == "__main__":
    user_input = "Show me the top 5 customers based on their total spending."

    initial_state = {"messages": [HumanMessage(content=user_input)]}

    print("--- Starting SQL Assistant ---")
    for event in sql_assistant.stream(initial_state):
        for node, update in event.items():
            print(f"\n[Node: {node}]")
            # Extract the content from the last message in the update
            print(f"Response: {update['messages'][-1].content}")
