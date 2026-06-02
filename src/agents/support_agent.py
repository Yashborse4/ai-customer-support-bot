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
    model = ChatOpenAI(
        model=settings.MODEL_NAME,
        base_url=settings.LOCAL_LLM_BASE_URL,
        api_key="local-placeholder",
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
    system_instruction = f"""You are a professional customer support assistant for Acme Corp.
Active Department Scope: <dept>{department.upper()}</dept>

Follow these strict rules to resolve customer inquiries:

<rules>
1. ALWAYS search the knowledge base using the `retrieve_company_info` tool before answering any questions about product specifications, shipping limits, returns eligibility, or corporate policies.
2. Query customer relational database records (orders, tickets, billing, customers, returns, shipping) using the `query_customer_database` tool.
3. You MUST pass department='{department}' as a parameter when invoking search or query tools to satisfy enterprise data segregation boundaries.
4. If you see an error screenshot, analyze it carefully and explain what is happening and how to fix it based on your knowledge base.
5. If the context from tools does not contain the information needed to answer, state clearly that you do not have enough information. DO NOT make up or hallucinate details.
</rules>

<formatting>
- Wrap code terms, tracking numbers, or error details in backticks (e.g. `SuperWidget 3000`).
- Structure your response using clear bullet points or numbered lists where appropriate for a premium client experience.
</formatting>

<examples>
User: What is Alice Johnson's loyalty tier?
Assistant: Call tool `query_customer_database` with query="What is Alice Johnson's loyalty tier?" and department="{department}"

User: What is the return policy for opened items?
Assistant: Call tool `retrieve_company_info` with query="return policy opened items" and department="{department}"
</examples>
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Initialize the local LLM
    model = ChatOpenAI(
        model=settings.MODEL_NAME,
        base_url=settings.LOCAL_LLM_BASE_URL,
        api_key="local-placeholder",
        temperature=0,
        streaming=True
    )
    
    model = model.bind_tools([retrieve_company_info, query_customer_database])
    
    chain = prompt | model
    response = await chain.ainvoke(state)
    
    return {"messages": [response]}
