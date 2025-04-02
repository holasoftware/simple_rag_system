class VectorDB:
    def store_document_chunk(self, collection_uuid, content, embedding, metadata=None):
        raise NotImplementedError

    def store_document_chunks_in_batch(
        self, collection_uuid, content_list, embedding_list, metadata_list
    ):
        for content, embedding, metadata in zip(
            content_list, embedding_list, metadata_list
        ):
            self.store_document(content, embedding, metadata, collection_uuid)

    def similarity_search(self, embedding_vector, k=3, metadata_filter=None):
        raise NotImplementedError

    def delete_document_chunk_by_id(self, document_chunk_id: int):
        raise NotImplementedError

    def delete_all_chunks_of_document(self, document_id: int):
        raise NotImplementedError

    def delete_document_chunks(self, **metadata_filter):
        raise NotImplementedError

    def close(self):
        pass
