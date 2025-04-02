import os
import logging
import uuid


from .text_splitters import chunk_text


logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        vector_store=None,
        get_batch_embedding_vectors=None,
        generate_response=None,
        text_splitter=chunk_text,
        max_document_length_text=1500,
        system_message="Answer based on the context. If unsure, say you don't know. Cite sources when possible.",
    ):
        self.vector_store = vector_store
        self.max_document_length_text = max_document_length_text
        self.system_message = system_message

        self.text_splitter = text_splitter

        if get_batch_embedding_vectors is not None:
            self.get_batch_embedding_vectors = get_batch_embedding_vectors

        if generate_response is not None:
            self.generate_response = generate_response

    def get_system_message(self):
        return self.system_message

    def get_batch_embedding_vectors(self, texts):
        raise NotImplementedError

    def get_embedding_vector(self, text):
        return self.get_batch_embedding_vectors([text])[0]

    def generate_response(self, messages):
        raise NotImplementedError

    def chunk_text(self, text):
        """Split long text into chunks"""
        if self.text_splitter is not None:
            yield from self.text_splitter(text)
        else:
            yield text

    def add_document(self, text, metadata=None, document_uuid=None, batch_size=None):
        """Add single document, automatically chunking if needed"""

        if document_uuid is None:
            document_uuid = uuid.uuid4()

        if len(text) > self.max_document_length_text:
            chunks = self.chunk_text(text)

            if batch_size:
                # Batch process embeddings
                metadatas = [metadata] * batch_size
                for i in range(0, len(chunks), batch_size):
                    chunk_batch = chunks[i : i + batch_size]
                    embedding_vectors = self.get_batch_embedding_vectors(chunk_batch)

                    self.vector_store.store_document_chunks_in_batch(
                        document_uuid, chunk_batch, embedding_vectors, metadatas
                    )
            else:
                for chunk in chunks:
                    embedding_vector = self.get_embedding_vector(chunk)
                    self.vector_store.store_document_chunk(
                        collection_uuid=document_uuid,
                        content=chunk,
                        embedding_vector=embedding_vector,
                        metadata=metadata,
                    )
        else:
            embedding_vector = self.get_embedding_vector(text)
            self.vector_store.store_document_chunk(
                collection_uuid=document_uuid,
                content=text,
                embedding_vector=embedding_vector,
                metadata=metadata,
            )

    def query(self, question, k=3, metadata_filter=None, return_full_data=False):
        logger.info(
            "Making semantic query: %s (k=%s, metadata_filter=%s)",
            question,
            k,
            metadata_filter,
        )

        question_embedding = self.get_embedding_vector(question)
        relevant_docs = self.vector_store.similarity_search(
            question_embedding, k, metadata_filter
        )
        context = "\n\n".join([doc.content for doc in relevant_docs])

        system_message = self.get_system_message()
        chat_completion_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        answer = self.generate_response(chat_completion_messages)

        if return_full_data:
            return {
                "question": question,
                "system_message": system_message,
                "chat_completion_messages": chat_completion_messages,
                "relevant_docs": relevant_docs,
                "answer": answer,
            }
        else:
            return answer

    def close(self):
        self.vector_store.close()
