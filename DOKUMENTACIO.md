
# Jade AI Chat Alkalmazás Dokumentáció

## Fájlok és kódjuk

### main.py
```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from cerebras.cloud.sdk import Cerebras
import threading
import time

app = Flask(__name__)
CORS(app)

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Nincs üzenet megadva'}), 400
        
        # Beszélgetési történet létrehozása
        messages = [
            {
                "role": "system",
                "content": "Te egy hasznos AI asszisztens vagy, aki magyarul válaszol. A neved Jade. Segítőkész vagy és részletes válaszokat adsz."
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        # Beszélgetési előzmények hozzáadása ha vannak
        if 'history' in data:
            messages = [messages[0]] + data['history'] + [messages[1]]
        
        # Streamelt válasz generálása
        stream = client.chat.completions.create(
            messages=messages,
            model="llama-4-scout-17b-16e-instruct",
            stream=True,
            max_completion_tokens=2048,
            temperature=0.2,
            top_p=1
        )
        
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        return jsonify({
            'response': response_text,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### templates/index.html
```html
<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jade - AI Asszisztens</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Modern chat alkalmazás teljes CSS és JavaScript kóddal -->
</head>
<body class="gradient-bg">
    <!-- Teljes chat interface sidebar-ral, üzenetekkel és modern design-nal -->
</body>
</html>
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
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
]
```

## Alkalmazás leírása

Ez egy modern webes AI chat alkalmazás, amely a Cerebras Cloud SDK-t használja a Meta Llama 4 Scout modellel való kommunikációhoz. Az alkalmazás Flask backend-et használ API végpontokkal, és egy elegáns, minimalista frontend-et biztosít.

## Főbb funkciók

- 🎨 **Modern UI**: Sötét téma, glassmorphism effektusok és animációk
- 💬 **Valós idejű chat**: Streamelt válaszok a Llama 4 Scout modeltől
- 📱 **Reszponzív design**: Tökéletesen működik mobil és asztali eszközökön
- 💾 **Chat előzmények**: Automatikus beszélgetés mentés
- ⚡ **Gyors válaszok**: Optimalizált API kommunikáció

## Technológiai stack

- **Backend**: Flask + Cerebras Cloud SDK
- **Frontend**: Vanilla JavaScript + Tailwind CSS
- **AI Model**: Meta Llama 4 Scout 17B Instruct
- **Streaming**: Valós idejű válaszok

## Beállítás

1. Állítsd be a `CEREBRAS_API_KEY` környezeti változót a Secrets eszközzel
2. Futtasd az alkalmazást a Run gombbal
3. Nyisd meg a böngészőben a megjelenő URL-t

Az alkalmazás a 5000-es porton fut és automatikusan megnyílik a Webview-ban.
```
