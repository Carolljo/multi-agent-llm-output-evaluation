from ollama import Client

client= Client(host="http://localhost:11434")


response=client.chat(
    model='qwen3:1.7b',
    messages=[
        {
            'role':'user',
            'content':'What is the capital of France?'
            
        }
    ],
)

print(response.message.content)