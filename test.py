import ollama

response = ollama.chat(
    model="phi",
    messages=[{"role": "user", "content": "What is the capital of India?"}]
)

print(response)