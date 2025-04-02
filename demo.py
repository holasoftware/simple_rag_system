import os


from rag.litellm_rag import LiteLlmRAGSystem
from rag.vector_store.pgvector_vectorstore import PgVectorVectorDB


API_KEY = os.getenv("LLM_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_MODEL_VECTOR_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"


rag = LiteLlmRAGSystem(
    embedding_model=EMBEDDING_MODEL,
    llm_model=LLM_MODEL,
    api_key=API_KEY,
    vector_store=PgVectorVectorDB.initialize_from_env_variables(
        vector_dimension=EMBEDDING_MODEL_VECTOR_DIMENSION, table_name="documents_demo1"
    ),
)

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
rag.add_document(
    long_document, {"source": "technical_notes", "doc_type": "explanation"}
)

# Query the system
question = "What are the main components of RAG?"
answer = rag.query(question)
print(f"Question: {question}")
print(f"Answer: {answer}")

rag.close()


rag = LiteLlmRAGSystem(
    embedding_model=EMBEDDING_MODEL,
    llm_model=LLM_MODEL,
    api_key=API_KEY,
    vector_store=PgVectorVectorDB.initialize_from_env_variables(
        vector_dimension=EMBEDDING_MODEL_VECTOR_DIMENSION, table_name="documents_demo2"
    ),
)

rag.add_document(
    "Cybersecurity is the practice of protecting systems and networks from attacks. It includes measures like firewalls, intrusion detection, and encryption.",
    {"document_id": 1},
)
rag.add_document(
    "A zero-day vulnerability is an undisclosed flaw in software that attackers can exploit before the vendor issues a fix.",
    {"document_id": 2},
)
rag.add_document(
    "Two-factor authentication (2FA) enhances security by requiring users to provide two forms of verification before gaining access.",
    {"document_id": 3},
)
rag.add_document(
    "Machine learning models require large datasets and are often fine-tuned to improve accuracy for specific tasks.",
    {"document_id": 4},
)
rag.add_document(
    "The CIA triad—Confidentiality, Integrity, and Availability—is a foundational concept in cybersecurity.",
    {"document_id": 5},
)

question = "What is 2 factor authentication?"
answer_data = rag.query(question, return_full_data=True)

print(f"Question: {question}")
print(f"Answer: {answer_data['answer']}")


print("Extracted documents extracted for the context of the chat completion ordered by semantic similarity:")
for relevant_document in answer_data["relevant_docs"]:
    print(f"- {relevant_document.metadata['document_id']}")

rag.close()
