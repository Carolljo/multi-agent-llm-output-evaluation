import os

from ollama import Client


class LLMClient:
    def __init__(
        self,
        model: str,
        host: str | None = None,
    ):
        self.model = model
        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )
        self.client = Client(host=self.host)

    def generate(self, prompt: str, response_format: dict) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=response_format,
        )

        return response.message.content