from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from src.core.config import settings
from src.tools.retrieval_tool import retrieve_company_info
from src.tools.sql_tool import query_customer_database
from src.schemas.state import SupportState

def get_support_model() -> Runnable[Any, Any]:
    """Returns the ChatOpenAI model bound with retrieval and database tools.

    Returns:
        A LangChain Runnable model instance bound with retrieve_company_info and query_customer_database.
    """
    if settings.LLM_PROVIDER.lower() == "local":
        model = ChatOpenAI(
            model=settings.LOCAL_MODEL_NAME,
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key="local-placeholder",
            temperature=0,
            streaming=True
        )
    else:
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
    department = state.get("department", "general")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"You are a professional customer support assistant for Acme Corp in the {department.upper()} department. "
            "You can analyze images (screenshots) if provided. "
            "ALWAYS search the knowledge base before answering questions about products, shipping, or policies. "
            "Use the `query_customer_database` tool to fetch relational records regarding customers, orders, stock inventory, returns, or support tickets. "
            f"Your active department scope is '{department}'. You MUST supply department='{department}' when invoking search or query tools. "
            "If you see an error screenshot, explain what is happening and how to fix it based on your knowledge."
        )),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Select LLM based on provider settings
    if settings.LLM_PROVIDER.lower() == "local":
        model = ChatOpenAI(
            model=settings.LOCAL_MODEL_NAME,
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key="local-placeholder",
            temperature=0,
            streaming=True
        )
    else:
        model = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True
        )
    
    model = model.bind_tools([retrieve_company_info, query_customer_database])
    
    chain = prompt | model
    response = await chain.ainvoke(state)
    
    return {"messages": [response]}
