"""Module containing the database query tool for the customer support bot.

Exposes a LangChain tool that leverages an SQL Agent to query relational support data.
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from src.core.config import settings
from src.database.sql_db import get_db_for_query

@tool
async def query_customer_database(query: str, department: str = "general") -> str:
    """Queries the Acme Corp customer database using natural language.

    This tool connects to relational tables such as customers, orders, products,
    returns, shipping, and support_tickets. It routes the question dynamically
    to fetch precise, structured answers about transactions, order statuses,
    inventory, and customer profiles.

    Args:
        query: The natural language question (e.g., 'What is Alice Johnson's loyalty tier?').
        department: Optional department scope (e.g., 'sales', 'support', 'general') of the request.

    Returns:
        The text response containing the answers retrieved from the database.
    """
    db = get_db_for_query(query, department=department)
    
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        base_url=settings.LOCAL_LLM_BASE_URL,
        api_key="local-placeholder",
        temperature=0
    )
    
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=False
    )
    
    try:
        response = await agent_executor.ainvoke({"input": query})
        return response.get("output", "Could not fetch data from database.")
    except Exception as e:
        return f"Error executing database query: {str(e)}"
