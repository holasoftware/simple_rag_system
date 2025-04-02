class VectorDB:
    def store_document(self, content, embedding, metadata=None):
        raise NotImplementedError

    def store_documents_in_batch(self, content_list, embedding_list, metadata_list):
        for content, embedding, metadata in zip(
            content_list, embedding_list, metadata_list
        ):
            self.store_document(content, embedding, metadata)

    def similarity_search(self, embedding, k=3, metadata_filter=None):
        raise NotImplementedError

    def close(self):
        pass
