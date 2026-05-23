from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from src.core.config import settings
from src.tools.retrieval_tool import retrieve_company_info
from src.tools.sql_tool import query_customer_database
from src.schemas.state import SupportState

def get_support_model() -> Runnable[Any, Any]:
    """Returns the ChatOpenAI model bound with retrieval and SQL database tools.

    Returns:
        A LangChain Runnable model instance bound with retrieve_company_info and query_customer_database.
    """
    model = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True
    )
    return model.bind_tools([retrieve_company_info, query_customer_database])

async def support_agent_node(state: SupportState) -> Dict[str, Any]:
    """Processes the current conversation state and generates an assistant response or tool call.

    Supports multi-modal content (text and image-based screenshots).

    Args:
        state: The current SupportState representing the conversation history.

    Returns:
        A dictionary containing the generated AIMessage to be appended to the state.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional customer support assistant for Acme Corp. "
            "You can analyze images (screenshots) if provided. "
            "You have access to two specialized tools to retrieve accurate information: "
            "1. 'query_customer_database': Use this tool to look up structured customer, order, "
            "product stock inventory, shipping, refund, and support ticket records from our SQLite database. "
            "Use it for specific questions like 'What is Diana's loyalty tier?' or 'What is the tracking number for order ID 2?'. "
            "2. 'retrieve_company_info': Use this tool to search the text-based knowledge base for policies, "
            "shipping fees, return terms, device reset guides, and general descriptions. "
            "Use it for general queries like 'What is your refund policy?'. "
            "ALWAYS query the appropriate tool before answering questions about orders, products, or policies. "
            "If you see an error screenshot, explain what is happening and how to fix it based on your knowledge."
        )),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Ensure GPT-4o is used for vision capabilities
    model = ChatOpenAI(
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True
    ).bind_tools([retrieve_company_info, query_customer_database])
    
    chain = prompt | model
    response = await chain.ainvoke(state)
    
    return {"messages": [response]}
