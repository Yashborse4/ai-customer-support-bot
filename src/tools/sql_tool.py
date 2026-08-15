"""Module containing the database query tool for the customer support bot.

Exposes a LangChain tool that leverages an SQL Agent to query relational support data.
"""

import logging
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import InjectedState
from src.core.config import settings
from src.database.sql_db import get_db_for_query
from src.core.security import PIISecurityGuard

# Configure logging
logger = logging.getLogger(__name__)

# Custom dialect-aware prompt guiding table queries and row limits
custom_sql_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert SQL translation agent designed to write valid queries.\n"
        "You are querying a {dialect} database.\n\n"
        "Rules for writing queries:\n"
        "1. Only query the tables and columns that are explicitly listed in the schema.\n"
        "2. DO NOT query columns that are not in the schema.\n"
        "3. LIMIT REGULATION:\n"
        "   - If the dialect is 'sqlite': Use standard `LIMIT N` to restrict rows.\n"
        "   - If the dialect is 'oracle': NEVER use the `LIMIT` clause! To restrict rows to N results in Oracle, use the `ROWNUM` filter (e.g. `WHERE ROWNUM <= N`).\n"
        "     Example: SELECT name FROM customers WHERE ROWNUM <= 5;\n"
        "4. Date Handling: Oracle uses VARCHAR2 for dates in this database. Match string formats exactly (e.g., 'YYYY-MM-DD').\n"
        "5. Respond ONLY with the executable SQL statement, wrapped in ```sql and ``` blocks.\n\n"
        "Only use the following tables:\n"
        "{table_info}"
    )),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

@tool
async def query_customer_database(
    query: str, 
    department: str = "general", 
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Queries the Acme Corp customer database using natural language.

    This tool connects to relational tables such as customers, orders, products,
    returns, shipping, and support_tickets. It routes the question dynamically
    to fetch precise, structured answers about transactions, order statuses,
    inventory, and customer profiles.

    Args:
        query: The natural language question (e.g., 'What is Alice Johnson's loyalty tier?').
        department: Optional department scope (e.g., 'sales', 'support', 'general') of the request.
        state: The LangGraph state injected automatically at runtime.

    Returns:
        The text response containing the answers retrieved from the database.
    """
    guard = PIISecurityGuard()
    masking_map = state.get("masking_map", {}) if state else {}
    
    # Unmask the input query so the database queries can match real customer PII
    unmasked_query = guard.unmask(query, masking_map)

    db = get_db_for_query(unmasked_query, department=department)
    
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        base_url=settings.LOCAL_LLM_BASE_URL,
        api_key="local-placeholder",
        temperature=0
    )
    
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        prompt=custom_sql_prompt,
        agent_type="openai-tools",
        verbose=False
    )
    
    try:
        response = await agent_executor.ainvoke({"input": unmasked_query})
        raw_output = response.get("output", "Could not fetch data from database.")
        
        # Mask the output to ensure no raw PII leaks back to the main LLM or chat history
        masked_output, updated_map = guard.mask(raw_output, masking_map)
        if state is not None:
            if "masking_map" not in state or not state["masking_map"]:
                state["masking_map"] = {}
            state["masking_map"].update(updated_map)
            
        return masked_output
    except Exception as e:
        logger.warning("SQL agent execution failed: %s. Retrying with self-correction...", e)
        try:
            # Query again, providing the failed query and execution error details to trigger self-correction
            retry_query = f"""The previous database query execution failed.
User Question: {unmasked_query}
Error details: {str(e)}

Please analyze the execution error and write a corrected query to fetch the correct data."""
            response = await agent_executor.ainvoke({"input": retry_query})
            raw_output = response.get("output", f"Error executing database query: {str(e)}")
            
            # Mask the output for the retry query response
            masked_output, updated_map = guard.mask(raw_output, masking_map)
            if state is not None:
                if "masking_map" not in state or not state["masking_map"]:
                    state["masking_map"] = {}
                state["masking_map"].update(updated_map)
                
            return masked_output
        except Exception as retry_err:
            error_output = f"Error executing database query: {str(retry_err)}"
            masked_output, updated_map = guard.mask(error_output, masking_map)
            if state is not None:
                if "masking_map" not in state or not state["masking_map"]:
                    state["masking_map"] = {}
                state["masking_map"].update(updated_map)
            return masked_output
