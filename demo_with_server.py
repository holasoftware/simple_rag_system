import os
from typing import Generator


from fastapi import FastAPI, Depends
from pydantic import BaseModel


from rag.litellm_rag import LiteLlmRAGSystem
from rag.vector_store.pgvector_vectorstore import PgVectorVectorDB


API_KEY = os.getenv("LLM_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_MODEL_VECTOR_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"


def get_rag() -> Generator[LiteLlmRAGSystem, None, None]:
    rag = LiteLlmRAGSystem(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        api_key=API_KEY,
        vector_store=PgVectorVectorDB.initialize_from_env_variables(
            vector_dimension=EMBEDDING_MODEL_VECTOR_DIMENSION
        ),
    )
    yield rag
    rag.close()


class DocumentRequest(BaseModel):
    text: str
    metadata: dict = None


class QueryRequest(BaseModel):
    question: str
    k: int = 3


class StatusResponse(BaseModel):
    ok: bool


class AddDocumentResponse(StatusResponse):
    pass


class QueryResponse(StatusResponse):
    answer: str


app = FastAPI()


@app.post("/add-document")
async def add_document(
    doc: DocumentRequest, rag: PgVectorVectorDB = Depends(get_rag)
) -> StatusResponse:
    rag.add_document(text=doc.text, metadata=doc.metadata)
    return AddDocumentResponse(status=True)


@app.post("/query")
async def query(
    req: QueryRequest, rag: PgVectorVectorDB = Depends(get_rag)
) -> QueryResponse:
    answer = rag.query(question=req.question, k=req.k)
    return QueryResponse(status=True, answer=answer)
