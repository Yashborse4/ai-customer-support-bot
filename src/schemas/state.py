from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportState(TypedDict):
    """
    Represents the state of the support conversation.
    """
    # The list of messages in the conversation.
    # Annotated with add_messages so that new messages are appended to the list.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Optional fields for tracking user intent or context
    customer_id: str
    query_category: str
