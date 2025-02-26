from together import Together

client = Together()

stream = client.chat.completions.create(
  model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
  messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
  stream=True,
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="", flush=True)