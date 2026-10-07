import os
import threading
import time
from copy import deepcopy

from dotenv import load_dotenv
from groq import APITimeoutError, Groq, RateLimitError
from ollama import Client


load_dotenv()


GROQ_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("GROQ_MAX_CONCURRENT_REQUESTS", "2")
)

GROQ_CONCURRENCY_LIMITER = threading.BoundedSemaphore(
    GROQ_MAX_CONCURRENT_REQUESTS
)

GROQ_MAX_RETRIES = int(
    os.getenv("GROQ_MAX_RETRIES", "3")
)


def make_groq_strict_schema(schema):
    """Convert a JSON schema to Groq strict structured-output format."""

    schema = deepcopy(schema)

    def process(node):
        if isinstance(node, dict):

            if node.get("type") == "object":
                node["additionalProperties"] = False

                properties = node.get("properties", {})

                if properties:
                    node["required"] = list(properties.keys())

                for value in properties.values():
                    process(value)

            if node.get("type") == "array":
                items = node.get("items")

                if items:
                    process(items)

            for key in ("anyOf", "oneOf", "allOf"):
                if key in node:
                    for item in node[key]:
                        process(item)

        elif isinstance(node, list):
            for item in node:
                process(item)

    process(schema)

    return schema


class LLMClient:
    """
    V2 LLM abstraction supporting both Ollama and Groq.

    Ollama remains the default for backward compatibility.

    Groq must be selected explicitly with:

        LLMClient(
            model="openai/gpt-oss-120b",
            provider="groq",
        )
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        provider: str | None = None,
    ):
        # Keep Ollama as the default so existing V2 unit tests and
        # legacy callers remain deterministic and offline.
        self.provider = (
            provider or "ollama"
        ).lower()

        self.model = model

        # Preserve the legacy host attribute expected by tests/callers.
        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )

        if self.provider == "ollama":
            self.client = Client(host=self.host)

        elif self.provider == "groq":
            self.client = Groq(
                api_key=os.getenv("GROQ_API_KEY")
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}"
            )

    def generate(
        self,
        prompt: str,
        response_format: dict,
    ) -> str:
        """Generate a structured response using the selected provider."""

        if self.provider == "ollama":
            return self._generate_ollama(
                prompt,
                response_format,
            )

        if self.provider == "groq":
            return self._generate_groq(
                prompt,
                response_format,
            )

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    def _generate_ollama(
        self,
        prompt: str,
        response_format: dict,
    ) -> str:

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

    def _generate_groq(
        self,
        prompt: str,
        response_format: dict,
    ) -> str:

        strict_schema = make_groq_strict_schema(
            response_format
        )

        last_error = None

        for attempt in range(GROQ_MAX_RETRIES + 1):

            try:
                with GROQ_CONCURRENCY_LIMITER:

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_response",
                                "strict": True,
                                "schema": strict_schema,
                            },
                        },
                    )

                return response.choices[0].message.content

            except RateLimitError as exc:
                last_error = exc

                if attempt >= GROQ_MAX_RETRIES:
                    raise

                retry_after = exc.response.headers.get(
                    "retry-after"
                )

                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = 5.0

                time.sleep(delay)

            except APITimeoutError as exc:
                last_error = exc

                if attempt >= GROQ_MAX_RETRIES:
                    raise

                # Transient network/API timeout.
                # Use increasing backoff rather than retrying immediately.
                delay = 2.0 * (attempt + 1)

                time.sleep(delay)

        raise last_error