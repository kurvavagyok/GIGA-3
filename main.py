
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
