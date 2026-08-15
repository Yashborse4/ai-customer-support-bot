from typing import Annotated, Sequence, TypedDict, NotRequired, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportState(TypedDict):
    """Represents the state of the support conversation.

    Attributes:
        messages: The list of messages in the conversation (statefully appended).
        customer_id: Optional tracker for identifying the customer.
        query_category: Optional label for classification of the conversation query.
        department: Optional label for the department scope of the conversation.
        masking_map: Optional mapping of PII placeholders to raw values.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    customer_id: NotRequired[str]
    query_category: NotRequired[str]
    department: NotRequired[str]
    masking_map: NotRequired[Dict[str, str]]
