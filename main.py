import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.database.vector_store import vector_store_manager
from src.graph.workflow import support_bot_graph

async def main() -> None:
    """Main CLI entry point for the interactive Acme Corp AI Customer Support Bot."""
    # Load environment variables
    load_dotenv()

    print("--- Acme Corp AI Customer Support Bot ---")
    
    # Initialize RAG (Index documents in 'data/' directory)
    print("Indexing documents...")
    vector_store_manager.load_and_index_documents("data")
    print("Indexing complete.\n")

    # Conversation loop
    print("Bot: Hello! I'm the Acme Corp support assistant. How can I help you today? (Type 'exit' to quit)")
    
    messages = []
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Thank you for contacting Acme Corp support. Have a great day!")
            break

        messages.append(HumanMessage(content=user_input))
        
        # Invoke the LangGraph workflow
        # Note: We pass the entire state including message history
        state = {"messages": messages}
        
        try:
            # Running as async but calling synchronously for CLI simplicity here
            result = await support_bot_graph.ainvoke(state)
            
            # The result contains the updated messages list
            # The last message should be the bot's response
            bot_response = result["messages"][-1].content
            print(f"Bot: {bot_response}")
            
            # Update local history for next turn
            messages = result["messages"]
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
