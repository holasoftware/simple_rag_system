import os


from rag.litellm_rag import LiteLlmRAGSystem
from rag.vector_store.pgvector_vectorstore import PgVectorVectorDB

API_KEY = os.getenv("LLM_API_KEY")

rag = LiteLlmRAGSystem(embedding_model="text-embedding-ada-002", llm_model="gpt-4o-mini", api_key=API_KEY, vector_store=PgVectorVectorDB.initialize_from_env_variables(vector_dimension=1536))

# Example long document
long_document = """
Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of both retrieval-based and generation-based approaches in natural language processing. 

The RAG architecture typically consists of two main components: a retriever and a generator. The retriever is responsible for finding relevant documents or passages from a large corpus given a query, while the generator produces a coherent response based on both the query and the retrieved documents.

Key advantages of RAG include:
1. It can incorporate up-to-date information without retraining the entire model
2. The generated responses can be grounded in actual documents
3. It provides traceability as you can see which documents influenced the answer

Popular implementations include Facebook's original RAG paper and various open-source implementations that use different retrieval backends like FAISS, Annoy, or pgvector.
"""

# Add with automatic chunking
rag.add_document(long_document, {"source": "technical_notes", "doc_type": "explanation"})

# Query the system
question = "What are the main components of RAG?"
answer = rag.query(question)
print(f"Question: {question}")
print(f"Answer: {answer}")

rag.close()
