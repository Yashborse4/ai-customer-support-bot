from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.config import settings
from src.tools.retrieval_tool import retrieve_company_info
from src.schemas.state import SupportState

def get_support_model():
    """
    Returns the ChatOpenAI model bound with tools.
    """
    model = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=0
    )
    return model.bind_tools([retrieve_company_info])

async def support_agent_node(state: SupportState):
    """
    Node that processes the current state and generates a response or tool call.
    Supports multi-modal content (text and images).
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
        temperature=0
    ).bind_tools([retrieve_company_info])
    
    chain = prompt | model
    response = await chain.ainvoke(state)
    
    return {"messages": [response]}
