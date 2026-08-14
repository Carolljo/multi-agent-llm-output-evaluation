from ollama import Client


class LLMClient:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.client = Client(host=host)

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