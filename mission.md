# Mission: AI Customer Support Bot

## Goal
Build a professional, enterprise-grade AI customer support bot that uses RAG (Retrieval-Augmented Generation) to answer queries accurately based on company documentation.

## Core Objectives
1. **RAG Implementation**: Set up a robust retrieval system using LangChain and a vector database (ChromaDB for local, Pinecone for cloud).
2. **LangGraph Integration**: Orchestrate the conversation flow using LangGraph for state management and multi-agent logic.
3. **OpenAI Integration**: Use OpenAI's LLMs (GPT-4o/GPT-3.5-turbo) and Embeddings.
4. **Structured Architecture**: Follow a clean, modular folder structure as per Antigravity standards.
5. **Quality Responses**: Ensure the bot provides grounded, helpful, and professional responses.

## Success Criteria
- Bot can retrieve relevant information from a knowledge base.
- Bot can handle user queries using the retrieved context.
- Conversation state is maintained across multiple turns (if needed via LangGraph).
- Code is modular, typed, and well-documented.
