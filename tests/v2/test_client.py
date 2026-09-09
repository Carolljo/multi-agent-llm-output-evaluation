from unittest.mock import Mock, patch

from src.v2.llm.client import LLMClient


def test_llm_client_uses_default_ollama_host():
    with patch("src.v2.llm.client.Client") as mock_client:
        client = LLMClient(model="llama3.2")

        assert client.model == "llama3.2"
        assert client.host == "http://localhost:11434"
        mock_client.assert_called_once_with(
            host="http://localhost:11434"
        )


def test_llm_client_uses_custom_host():
    with patch("src.v2.llm.client.Client") as mock_client:
        client = LLMClient(
            model="llama3.2",
            host="http://custom-host:11434",
        )

        assert client.host == "http://custom-host:11434"
        mock_client.assert_called_once_with(
            host="http://custom-host:11434"
        )


def test_generate_returns_message_content():
    with patch("src.v2.llm.client.Client") as mock_client:
        mock_response = Mock()
        mock_response.message.content = '{"answer": "test"}'

        mock_client.return_value.chat.return_value = mock_response

        client = LLMClient(model="llama3.2")

        result = client.generate(
            prompt="Generate an answer.",
            response_format={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
            },
        )

        assert result == '{"answer": "test"}'

        mock_client.return_value.chat.assert_called_once_with(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": "Generate an answer.",
                }
            ],
            format={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
            },
        )