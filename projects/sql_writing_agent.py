import os
from typing import TypedDict

import mssql_python
from dotenv import load_dotenv
from IPython.display import Image
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class AgentState(TypedDict):
    messages: list[BaseMessage]
    prompt: list[BaseMessage]


def genrate_prompt(state: AgentState) -> AgentState:
    sql_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that writes SQL queries based on natural language prompts."
                "Your output is only the SQL query in plain string without any explanation."
                "The SQL query should be compatible with Microsoft SQL Server and later runs the query and returns the result."
                "If the tables does not exist, you can create temporary tables and insert values into them to run the query.",
            ),
            ("human", "{question}"),
        ]
    )
    prompt_messages = sql_template.invoke({"question": state["messages"]})
    return {"messages": prompt_messages}


@tool
def run_sql_query(query: str) -> str:
    """A simple function to run a SQL query against a SQL Server database.This accepts a SQL query as input and returns the results as a string."""

    connection_string = (
        f"Server={os.getenv('SQL_SERVER_HOST')},{os.getenv('SQL_SERVER_PORT')};"
        f"Database={os.getenv('SQL_SERVER_DATABASE')};"
        f"UID={os.getenv('SQL_SERVER_USER')};"
        f"PWD={os.getenv('SQL_SERVER_PASSWORD')};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    try:
        print(f"Running SQL query: {query}")
        with mssql_python.connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return str(rows)

    except Exception as e:
        return f"Error running SQL query: {e}"


tools = [run_sql_query]


def sql_writing_agent(state: AgentState) -> AgentState:
    """A simple agent that writes SQL queries based on natural language prompts."""

    llm = ChatOllama(model="qwen3.5:4b")
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


sql_agent_graph = StateGraph(AgentState)

sql_agent_graph.add_node("generate_sql", sql_writing_agent)
sql_agent_graph.add_node("tools", ToolNode(tools))
sql_agent_graph.add_node("prompt_gen", genrate_prompt)

sql_agent_graph.add_edge(START, "prompt_gen")
sql_agent_graph.add_edge("prompt_gen", "generate_sql")
sql_agent_graph.add_conditional_edges("generate_sql", tools_condition)
sql_agent_graph.add_edge("tools", "generate_sql")
sql_agent = sql_agent_graph.compile()
for step in sql_agent.stream(
    {
        "messages": "Hi, I want to find the top 5 customers by total spend. Can you help me with that?"
    }
):
    for node_name, state_update in step.items():
        # Get the latest message from the node that just ran
        print(node_name, state_update, f"\n{'-'* 100}")
Image(sql_agent.get_graph().draw_mermaid_png())
