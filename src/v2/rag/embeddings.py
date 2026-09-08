from ollama import Client


class OllamaEmbeddingModel:
    """Generate text embeddings using a local Ollama embedding model."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""

        response = self.client.embed(
            model=self.model,
            input=text,
        )

        return response.embeddings[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings."""

        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        return response.embeddings