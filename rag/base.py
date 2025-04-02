import os
import logging

from .text_splitters import chunk_text


logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        vector_store=None,
        get_batch_embeddings=None,
        generate_response=None,
        text_splitter=chunk_text,
        max_document_length_text=1500,
        system_message="Answer based on the context. If unsure, say you don't know. Cite sources when possible.",
    ):
        self.vector_store = vector_store
        self.max_document_length_text = max_document_length_text
        self.system_message = system_message

        self.text_splitter = text_splitter

        if get_batch_embeddings is not None:
            self.get_batch_embeddings = get_batch_embeddings

        if generate_response is not None:
            self.generate_response = generate_response

    def get_system_message(self):
        return self.system_message

    def get_batch_embeddings(self, texts):
        raise NotImplementedError

    def get_embedding(self, text):
        return self.get_batch_embeddings([text])[0]

    def generate_response(self, messages):
        raise NotImplementedError

    def chunk_text(self, text):
        """Split long text into chunks"""
        if self.text_splitter is not None:
            yield from self.text_splitter(text)
        else:
            yield text

    def add_document(self, text, metadata=None):
        """Add single document, automatically chunking if needed"""
        if len(text) > self.max_document_length_text:
            chunks = self.chunk_text(text)
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                self.vector_store.store_document(
                    content=chunk, embedding=embedding, metadata=metadata
                )
        else:
            embedding = self.get_embedding(text)
            self.vector_store.store_document(
                content=text, embedding=embedding, metadata=metadata
            )

    def add_documents_batch(self, documents, metadata_list=None, batch_size=32):
        """Process multiple documents with efficient batch embedding"""
        if metadata_list is None:
            metadata_list = [None] * len(documents)

        all_chunks = []
        all_metadata = []

        for doc, meta in zip(documents, metadata_list):
            if len(doc) > self.MAX_DOCUMENT_LENGTH_TEXT:
                chunks = self.chunk_text(doc)
                all_chunks.extend(chunks)
                all_metadata.extend([meta] * len(chunks))
            else:
                all_chunks.append(doc)
                all_metadata.append(meta)

        # Batch process embeddings
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            chunk_batch = all_chunks[i : i + batch_size]
            embeddings_in_batch = self.get_batch_embeddings(chunk_batch)
            all_embeddings.extend(embeddings_in_batch)

        # Store all chunks
        self.vector_store.store_documents_in_batch(
            all_chunks, all_embeddings, all_metadata
        )

    def query(self, question, k=3, metadata_filter=None, return_full_data=False):
        question_embedding = self.get_embedding(question)
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
