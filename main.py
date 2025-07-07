
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    # Ez az alapértelmezett és elhagyható
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Te egy hasznos asszisztens vagy, aki magyarul válaszol."
        }
    ],
    model="llama-4-scout-17b-16e-instruct",
    stream=True,
    max_completion_tokens=2048,
    temperature=0.2,
    top_p=1
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
