import os


from sentence_transformers import SentenceTransformer


SENTENCE_TRANSFORMER_LOCAL_EMBEDDING_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


model = SentenceTransformer(SENTENCE_TRANSFORMER_LOCAL_EMBEDDING_MODEL)


class LocalEmbeddingsMixin:
    def get_batch_embeddings(self, texts):
        return model.encode(texts)
