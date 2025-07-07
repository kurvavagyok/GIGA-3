
# Cerebras AI Chat Alkalmazás Dokumentáció

## Fájlok és kódjuk

### main.py
```python
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
```

### pyproject.toml
```toml
[project]
name = "python-template"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
requires-python = ">=3.11"
dependencies = [
    "cerebras-cloud-sdk>=1.35.0",
]
```

## Alkalmazás leírása

Ez egy AI chat alkalmazás, amely a Cerebras Cloud SDK-t használja a Llama-4 Scout modellel való kommunikációhoz. Az alkalmazás streamelt válaszokat ad, magyarul válaszol, és környezeti változóból olvassa be az API kulcsot.

## Beállítás

1. Állítsd be a `CEREBRAS_API_KEY` környezeti változót a Secrets eszközzel
2. Futtasd az alkalmazást a Run gombbal
