import litellm

from .base import RAGSystem


class LiteLlmRAGSystem(RAGSystem):
    def __init__(
        self, embedding_model, llm_model, api_key=None, temperature=0.7, **kwargs
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.api_key = api_key
        self.temperature = temperature

        super().__init__(**kwargs)

    def get_batch_embeddings(self, texts):
        response = litellm.embedding(
            model=self.embedding_model, api_key=self.api_key, input=texts
        )

        return [item["embedding"] for item in response.data]

    def generate_response(self, messages):
        response = litellm.completion(
            model=self.llm_model,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
