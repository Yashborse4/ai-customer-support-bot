from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from src.core.config import settings
from src.tools.retrieval_tool import retrieve_company_info
from src.schemas.state import SupportState

def get_support_model() -> Runnable[Any, Any]:
    """Returns the ChatOpenAI model bound with retrieval tools.

    Returns:
        A LangChain Runnable model instance bound with the `retrieve_company_info` tool.
    """
    model = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True
    )
    return model.bind_tools([retrieve_company_info])

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
            "ALWAYS search the knowledge base before answering questions about products, shipping, or policies. "
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
    ).bind_tools([retrieve_company_info])
    
    chain = prompt | model
    response = await chain.ainvoke(state)
    
    return {"messages": [response]}
