from fastapi import FastAPI
from pydantic import BaseModel


from rag.litellm_rag import LiteLlmRAGSystem
from rag.vector_store.pgvector_vectorstore import PgVectorVectorDB


API_KEY = os.getenv("LLM_API_KEY")

rag = LiteLlmRAGSystem(embedding_model="text-embedding-004", llm_model="gemini-1.5-flash", api_key=API_KEY, vector_store=PgVectorVectorDB.initialize_from_env_variables(vector_dimension=768))


app = FastAPI()

class DocumentRequest(BaseModel):
    text: str
    metadata: dict = None

class QueryRequest(BaseModel):
    question: str
    k: int = 3


@app.post("/add-document")
async def add_document(doc: DocumentRequest):
    rag.add_document(doc.text, doc.metadata)
    return {"status": "success"}


@app.post("/query")
async def query(req: QueryRequest):
    answer = rag.query(req.question, req.k)
    return {"answer": answer}


@app.on_event("shutdown")
async def shutdown():
    rag.close()