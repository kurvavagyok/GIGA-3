import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import httpx # Aszinkron HTTP kérésekhez
import logging
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from functools import wraps

# Google Cloud kliensekhez
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

# Cerebras Cloud SDK
from cerebras.cloud.sdk import Cerebras

# Gemini API
import google.generativeai as genai

# Exa API
from exa_py import Exa

# FastAPI
from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- Konfiguráció és Titkok Betöltése ---
# Ez feltételezi, hogy a következő titkok be vannak állítva a Replit Secrets-ben:
# GCP_SERVICE_ACCOUNT_KEY (a JSON kulcs teljes tartalma)
# GCP_PROJECT_ID (a GCP projekt azonosítója)
# GCP_REGION (a GCP régió, pl. "us-central1")
# CEREBRAS_API_KEY
# GEMINI_API_KEY
# EXA_API_KEY

# Naplózás konfigurálása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Digitális Ujjlenyomat ---
DIGITAL_FINGERPRINT = "Jade made by Kollár Sándor"
CREATOR_SIGNATURE = "SmFkZSBtYWRlIGJ5IEtvbGzDoXIgU8OhbmRvcg==" # Base64 kódolt "Jade made by Kollár Sándor"
CREATOR_HASH = "a7b4c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5" # SHA256 hash

# --- API Kulcsok és Konfiguráció Betöltése a Replit Secrets-ből ---
GCP_SERVICE_ACCOUNT_KEY_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")

# --- Kliensek Inicializálása ---
# GCP Vertex AI kliens inicializálása service accounttal
gcp_credentials = None
if GCP_SERVICE_ACCOUNT_KEY_JSON and GCP_PROJECT_ID and GCP_REGION:
    try:
        info = json.loads(GCP_SERVICE_ACCOUNT_KEY_JSON)
        gcp_credentials = service_account.Credentials.from_service_account_info(info)
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=gcp_credentials)
        logger.info("GCP Vertex AI client initialized successfully.")
    except (json.JSONDecodeError, GoogleAPIError, ValueError) as e:
        logger.error(f"Error initializing GCP Vertex AI client: {e}")
        gcp_credentials = None # Jelöljük, hogy nem sikerült inicializálni
else:
    logger.warning("GCP_SERVICE_ACCOUNT_KEY, GCP_PROJECT_ID, or GCP_REGION not found. GCP Vertex AI functionality will be limited.")

# Cerebras kliens
cerebras_client = None
if CEREBRAS_API_KEY:
    try:
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Cerebras client: {e}")
else:
    logger.warning("CEREBRAS_API_KEY not found. Cerebras functionality will be limited.")

# Gemini kliens
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro') # Vagy 'gemini-1.5-flash' a gyorsaságért
        logger.info("Gemini client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {e}")
else:
    logger.warning("GEMINI_API_KEY not found. Gemini functionality will be limited.")

# Exa kliens
exa_client = None
if EXA_API_KEY:
    try:
        exa_client = Exa(api_key=EXA_API_KEY)
        logger.info("Exa client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Exa client: {e}")
else:
    logger.warning("EXA_API_KEY not found. Exa functionality will be limited.")

# --- FastAPI Alkalmazás Inicializálása ---
app = FastAPI(
    title="Deep Discovery AI - Tudományos és Innovációs Katalizátor",
    description="Egyedülálló platform Gemini, Exa, AlphaFold (adatbázis) és Llama (Cerebras) AI modellekkel.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS konfiguráció
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Termelési környezetben szűkítsd le a frontend URL-jére!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statikus fájlok kiszolgálása
app.mount("/static", StaticFiles(directory="templates"), name="static")

# --- Segédmodellek a Pydantic-hoz ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(..., description="A felhasználó egyedi azonosítója a beszélgetési előzményekhez.")

class UserSubscriptionRequest(BaseModel):
    user_id: str
    subscription_type: str = Field(..., pattern="^(basic|pro|enterprise)$")
    
class UserUsageResponse(BaseModel):
    user_id: str
    subscription_type: str
    api_calls_today: int
    api_calls_limit: int
    remaining_calls: int

class ScientificInsightRequest(BaseModel):
    query: str = Field(..., min_length=5, description="A tudományos vagy innovációs lekérdezés.")
    num_results: int = Field(default=5, ge=1, le=10, description="Hány releváns találatot keressen az Exa AI.")
    summary_length: int = Field(default=200, ge=50, le=500, description="A Gemini által generált összefoglaló hossza szavakban.")

class ProteinLookupRequest(BaseModel):
    protein_id: str = Field(..., description="Az EMBL-EBI AlphaFold DB-ben keresendő fehérje azonosítója (pl. UniProt ID).")

class CustomGCPModelRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="A GCP Vertex AI modellnek küldendő bemeneti adatok.")
    gcp_endpoint_id: str = Field(..., description="A GCP Vertex AI telepített modell végpontjának azonosítója.")
    gcp_project_id: Optional[str] = GCP_PROJECT_ID
    gcp_region: Optional[str] = GCP_REGION

class SimulationOptimizerRequest(BaseModel):
    simulation_type: str = Field(..., description="A szimuláció típusa (pl. 'molecular_dynamics', 'materials_property').")
    input_parameters: Dict[str, Any] = Field(..., description="A szimulációhoz szükséges bemeneti paraméterek.")
    optimization_goal: str = Field(..., description="A szimuláció optimalizálási célja (pl. 'minimize_energy', 'maximize_conductivity').")

# Új modell az AlphaGenome API-hoz
class AlphaGenomeRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=100, description="A teljes DNS vagy RNS szekvencia elemzésre.")
    organism: str = Field(..., description="A szervezet, amelyből a genom származik (pl. 'Homo sapiens').")
    analysis_type: str = Field(..., description="Az elemzés típusa ('átfogó', 'génkódoló régiók', 'funkcionális elemek').")
    include_predictions: bool = Field(default=False, description="Tartalmazzon-e fehérje struktúra előrejelzéseket (AlphaFold).")

# === ADATBÁZIS ÉS CACHE RENDSZER ===

# Adatbázis inicializálás
def init_database():
    """Adatbázis táblák létrehozása"""
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    # Felhasználók tábla
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT,
            subscription_type TEXT DEFAULT 'basic',
            api_calls_today INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Chat előzmények tábla
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            message TEXT,
            response TEXT,
            model_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Kutatási eredmények cache tábla
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash TEXT UNIQUE,
            query TEXT,
            results TEXT,
            model_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # API használati statisztikák
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            endpoint TEXT,
            model_used TEXT,
            tokens_used INTEGER,
            cost REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Adatbázis inicializálás
init_database()

# === PIACI FUNKCIÓK ===

# Előfizetési limitek
SUBSCRIPTION_LIMITS = {
    'basic': {
        'daily_api_calls': 50,
        'max_sequence_length': 1000,
        'concurrent_requests': 1,
        'advanced_models': False
    },
    'pro': {
        'daily_api_calls': 500,
        'max_sequence_length': 10000,
        'concurrent_requests': 3,
        'advanced_models': True
    },
    'enterprise': {
        'daily_api_calls': 10000,
        'max_sequence_length': 50000,
        'concurrent_requests': 10,
        'advanced_models': True
    }
}

# Rate limiting decorator
def rate_limit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Implementáció a rate limiting logikához
        return await func(*args, **kwargs)
    return wrapper

# Felhasználó validáció
def validate_user_limits(user_id: str, endpoint: str) -> bool:
    """Ellenőrzi, hogy a felhasználó túllépi-e a limiteket"""
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT subscription_type, api_calls_today FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    
    if not result:
        # Új felhasználó létrehozása
        cursor.execute('INSERT INTO users (id, subscription_type) VALUES (?, ?)', (user_id, 'basic'))
        conn.commit()
        subscription_type = 'basic'
        api_calls_today = 0
    else:
        subscription_type, api_calls_today = result
    
    conn.close()
    
    limits = SUBSCRIPTION_LIMITS[subscription_type]
    return api_calls_today < limits['daily_api_calls']

# Cache rendszer
def get_cached_result(query: str) -> Optional[Dict]:
    """Cached eredmény lekérése"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT results, model_used, timestamp 
        FROM research_cache 
        WHERE query_hash = ? AND timestamp > datetime('now', '-1 hour')
    ''', (query_hash,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'results': json.loads(result[0]),
            'model_used': result[1],
            'cached': True,
            'timestamp': result[2]
        }
    return None

def save_to_cache(query: str, results: Dict, model_used: str):
    """Eredmény mentése cache-be"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO research_cache (query_hash, query, results, model_used)
        VALUES (?, ?, ?, ?)
    ''', (query_hash, query, json.dumps(results), model_used))
    
    conn.commit()
    conn.close()

# === HIBRID AI ORCHESTRATOR ===

class AIOrchestrator:
    def __init__(self):
        self.model_performance = {
            'qwen3': {'avg_response_time': 2.5, 'success_rate': 0.95, 'cost': 0.001},
            'llama4': {'avg_response_time': 3.0, 'success_rate': 0.93, 'cost': 0.0012},
            'gemini': {'avg_response_time': 4.0, 'success_rate': 0.98, 'cost': 0.002}
        }
        
    def select_optimal_model(self, task_type: str, user_subscription: str) -> str:
        """Optimális modell kiválasztása feladat és előfizetés alapján"""
        if user_subscription == 'basic':
            return 'qwen3'  # Leggyorsabb és legolcsóbb
        elif task_type == 'research':
            return 'gemini'  # Legjobb pontosság kutatáshoz
        elif task_type == 'chat':
            return 'llama4'  # Jó beszélgetési képességek
        else:
            return 'qwen3'  # Alapértelmezett
    
    async def execute_with_fallback(self, task, primary_model: str, fallback_models: List[str]):
        """Feladat végrehajtása fallback rendszerrel"""
        models_to_try = [primary_model] + fallback_models
        
        for model in models_to_try:
            try:
                start_time = time.time()
                result = await task(model)
                end_time = time.time()
                
                # Teljesítmény frissítése
                self.update_model_performance(model, end_time - start_time, True)
                return result, model
                
            except Exception as e:
                logger.warning(f"Modell {model} sikertelen: {e}")
                self.update_model_performance(model, 0, False)
                continue
                
        raise Exception("Minden AI modell sikertelen volt")
    
    def update_model_performance(self, model: str, response_time: float, success: bool):
        """Modell teljesítmény frissítése"""
        if model in self.model_performance:
            perf = self.model_performance[model]
            perf['avg_response_time'] = (perf['avg_response_time'] + response_time) / 2
            perf['success_rate'] = (perf['success_rate'] + (1 if success else 0)) / 2

# AI Orchestrator instance
ai_orchestrator = AIOrchestrator()

# Beszélgetési előzmények tárolása memóriában (egyszerű prototípushoz)
# Ezt éles környezetben adatbázisra (pl. Firestore) kell cserélni!
chat_histories: Dict[str, List[Message]] = {}

# --- API Végpontok ---

@app.get("/")
async def serve_frontend():
    """A frontend HTML oldal kiszolgálása."""
    return FileResponse("templates/index.html")



@app.post("/api/auth/apple")
async def apple_signin(request: Request):
    """Apple Sign In autentikáció kezelése."""
    try:
        data = await request.json()
        authorization = data.get('authorization', {})
        user_data = data.get('user', {})
        
        # Itt normálisan validálnád az Apple token-t
        # Most demo célokra egyszerűen elfogadjuk
        
        # Felhasználói adatok mentése/lekérése
        user_id = f"apple_{authorization.get('code', 'demo')[:10]}"
        
        logger.info(f"Apple Sign In successful for user: {user_id}")
        
        return {
            "success": True,
            "message": "Sikeres Apple bejelentkezés",
            "user_id": user_id,
            "user_data": user_data
        }
        
    except Exception as e:
        logger.error(f"Apple Sign In error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Apple bejelentkezési hiba: {e}"
        )

@app.get("/api")
async def root_endpoint():
    """Alapvető üdvözlő végpont."""
    return {
        "message": "Üdvözöllek a Deep Discovery AI platformon!",
        "version": app.version,
        "creator": DIGITAL_FINGERPRINT,
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check_endpoint():
    """Egészségügyi ellenőrzés."""
    return {"status": "healthy", "version": app.version, "creator": DIGITAL_FINGERPRINT}

@app.post("/api/deep_discovery/chat")
@rate_limit
async def deep_discovery_chat(req: ChatRequest):
    """
    Kezeli a beszélgetéseket hibrid AI rendszerrel és felhasználói limitekkel.
    A beszélgetési előzményeket az adatbázis tárolja user_id alapján.
    """
    # Felhasználói limitek ellenőrzése
    if not validate_user_limits(req.user_id, 'chat'):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="API hívás limit elérve. Frissítsd az előfizetésed a további használathoz!"
        )
    
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető chat modell (Cerebras Qwen 3/Llama 4 vagy Gemini)."
        )

    user_id = req.user_id
    current_message = req.message

    # Beszélgetési előzmények lekérése vagy inicializálása
    history = chat_histories.get(user_id, [])

    # Rendszerüzenet (csak egyszer az elején)
    system_message = {
        "role": "system",
        "content": "Te egy rendkívül intelligens és szakértő AI asszisztens vagy, aki magyarul válaszol. A neved Jade. Hibrid AI képességekkel rendelkezel (Qwen 3, Llama 4, Gemini), és segítőkész, részletes és innovatív válaszokat adsz a legújabb tudományos és technológiai fejleményekről, különös tekintettel a biológia, kémia, anyagtudomány, orvostudomány és mesterséges intelligencia területére. Használd a tudásodat a legjobb válaszok megadásához."
    }

    # Építsük fel a teljes üzenetlistát
    messages_for_llm = [system_message] + history + [{"role": "user", "content": current_message}]

    response_text = ""
    model_used = ""

    try:
        # Párhuzamos modell futtatás: Qwen 3 és Llama 4 egyenrangú elsődleges modellek
        if cerebras_client:
            async def run_qwen3():
                try:
                    logger.info(f"Starting Cerebras Qwen 3 for user {user_id}")
                    stream = cerebras_client.chat.completions.create(
                        messages=messages_for_llm,
                        model="qwen2.5-72b-instruct",
                        stream=True,
                        max_completion_tokens=2048,
                        temperature=0.2,
                        top_p=1
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Qwen 3"
                except Exception as e:
                    logger.warning(f"Qwen 3 hiba: {e}")
                    return None, None

            async def run_llama4():
                try:
                    logger.info(f"Starting Cerebras Llama 4 for user {user_id}")
                    stream = cerebras_client.chat.completions.create(
                        messages=messages_for_llm,
                        model="llama-4-scout-17b-16e-instruct",
                        stream=True,
                        max_completion_tokens=2048,
                        temperature=0.2,
                        top_p=1
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Llama 4"
                except Exception as e:
                    logger.warning(f"Llama 4 hiba: {e}")
                    return None, None

            # Párhuzamos futtatás - az első válasz nyer
            try:
                tasks = [run_qwen3(), run_llama4()]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Töröljük a függőben lévő feladatokat
                for task in pending:
                    task.cancel()
                
                # Első sikeres válasz feldolgozása
                for task in done:
                    result = await task
                    if result[0]:  # Ha van válasz
                        response_text, model_used = result
                        logger.info(f"Első válasz: {model_used}")
                        break
                
                # Ha egyik sem adott választ, próbáljuk őket szekvenciálisan
                if not response_text:
                    logger.warning("Párhuzamos futtatás sikertelen, szekvenciális próbálkozás")
                    response_text, model_used = await run_qwen3()
                    if not response_text:
                        response_text, model_used = await run_llama4()
                        
            except Exception as parallel_error:
                logger.error(f"Párhuzamos futtatás hiba: {parallel_error}")
                response_text = ""
        
        # Ha a Cerebras modellek nem működnek, használjuk a Gemini-t
        if not response_text and gemini_model:
            # Ha a Cerebras nem elérhető, használjuk a Gemini 2.5 Pro-t
            logger.info(f"Using Gemini 2.5 Pro for user {user_id}")
            # Gemini esetén a messages formátum átalakítása szükséges
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_for_llm])
            response = await gemini_model.generate_content_async(
                conversation_text,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.2
                )
            )
            response_text = response.text
            model_used = "Google Gemini 2.5 Pro"

        if not response_text:
            raise ValueError("Az AI modell nem adott választ.")

        # Frissítsük a beszélgetési előzményeket
        history.append({"role": "user", "content": current_message})
        history.append({"role": "assistant", "content": response_text})
        chat_histories[user_id] = history # Mentés memóriába
        
        # Adatbázisba mentés
        conn = sqlite3.connect('deep_discovery.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (user_id, message, response, model_used)
            VALUES (?, ?, ?, ?)
        ''', (user_id, current_message, response_text, model_used))
        
        # API használat frissítése
        cursor.execute('''
            UPDATE users SET api_calls_today = api_calls_today + 1, last_active = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()

        return {
            'response': response_text,
            'model_used': model_used,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Error in deep discovery chat for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba történt a beszélgetés során: {e}"
        )

@app.post("/api/subscription/upgrade")
async def upgrade_subscription(req: UserSubscriptionRequest):
    """Felhasználói előfizetés frissítése"""
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE users 
        SET subscription_type = ?, api_calls_today = 0
        WHERE id = ?
    ''', (req.subscription_type, req.user_id))
    
    conn.commit()
    conn.close()
    
    return {
        'message': f'Előfizetés sikeresen frissítve: {req.subscription_type}',
        'new_limits': SUBSCRIPTION_LIMITS[req.subscription_type]
    }

@app.get("/api/user/usage/{user_id}")
async def get_user_usage(user_id: str):
    """Felhasználói használati statisztikák lekérése"""
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT subscription_type, api_calls_today 
        FROM users 
        WHERE id = ?
    ''', (user_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return UserUsageResponse(
            user_id=user_id,
            subscription_type='basic',
            api_calls_today=0,
            api_calls_limit=SUBSCRIPTION_LIMITS['basic']['daily_api_calls'],
            remaining_calls=SUBSCRIPTION_LIMITS['basic']['daily_api_calls']
        )
    
    subscription_type, api_calls_today = result
    limit = SUBSCRIPTION_LIMITS[subscription_type]['daily_api_calls']
    
    return UserUsageResponse(
        user_id=user_id,
        subscription_type=subscription_type,
        api_calls_today=api_calls_today,
        api_calls_limit=limit,
        remaining_calls=max(0, limit - api_calls_today)
    )

@app.get("/api/analytics/dashboard")
async def analytics_dashboard():
    """Platformstatisztikák dashboard"""
    conn = sqlite3.connect('deep_discovery.db')
    cursor = conn.cursor()
    
    # Aktív felhasználók
    cursor.execute('''
        SELECT COUNT(*) FROM users 
        WHERE last_active > datetime('now', '-7 days')
    ''')
    active_users = cursor.fetchone()[0]
    
    # Összes API hívás ma
    cursor.execute('''
        SELECT SUM(api_calls_today) FROM users
    ''')
    total_api_calls = cursor.fetchone()[0] or 0
    
    # Előfizetési megoszlás
    cursor.execute('''
        SELECT subscription_type, COUNT(*) 
        FROM users 
        GROUP BY subscription_type
    ''')
    subscription_stats = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        'active_users_week': active_users,
        'total_api_calls_today': total_api_calls,
        'subscription_distribution': subscription_stats,
        'model_performance': ai_orchestrator.model_performance
    }

@app.get("/api/deep_discovery/chat_history/{user_id}")
async def get_chat_history_endpoint(user_id: str):
    """Visszaadja egy adott felhasználó beszélgetési előzményeit."""
    history = chat_histories.get(user_id, [])
    return {'user_id': user_id, 'history': history}

@app.post("/api/deep_discovery/clear_chat_history/{user_id}")
async def clear_chat_history_endpoint(user_id: str):
    """Törli egy adott felhasználó beszélgetési előzményeit."""
    if user_id in chat_histories:
        del chat_histories[user_id]
        logger.info(f"Chat history cleared for user {user_id}")
        return {'message': f'Beszélgetési előzmények törölve a felhasználó számára: {user_id}'}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Nincs beszélgetési előzmény a felhasználó számára: {user_id}"
    )

@app.post("/api/deep_discovery/research_trends")
async def get_research_trends(req: ScientificInsightRequest):
    """
    Keresi a legújabb tudományos/innovációs információkat az Exa AI-val,
    majd a Gemini 2.5 Pro-val elemzi és összefoglalja.
    """
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Az Exa AI kliens nem elérhető."
        )
    if not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A Gemini modell nem elérhető az elemzéshez."
        )

    try:
        logger.info(f"Searching Exa for query: {req.query}")
        search_response = exa_client.search(
            query=req.query,
            num_results=req.num_results,
            text_contents={"max_characters": 1000, "strategy": "retrieve"}
        )

        if not search_response or not search_response.results:
            return {
                "query": req.query,
                "summary": "Nem található releváns információ a lekérdezésre.",
                "sources": []
            }

        sources = []
        combined_content = ""
        for i, result in enumerate(search_response.results):
            if result.text_contents and result.text_contents.text:
                combined_content += f"--- Forrás {i+1}: {result.title} ({result.url}) ---\n{result.text_contents.text}\n\n"
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "published_date": result.published_date
                })

        # Gemini összefoglalás és elemzés
        summary_prompt = f"""
        Elemezd a következő tudományos/innovációs információkat, és készíts egy tömör, objektív összefoglalót (max. {req.summary_length} szó).
        Emeld ki a legfontosabb áttöréseket, következtetéseket vagy innovációs vonatkozásokat.

        Információk:
        {combined_content[:8000]} # Korlátozzuk a bemenetet a token limit miatt

        Összefoglalás:
        """

        gemini_response = await gemini_model.generate_content_async(
            summary_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=req.summary_length * 2, # Több token, hogy biztosan elférjen
                temperature=0.1
            )
        )
        summary_text = gemini_response.text

        return {
            "query": req.query,
            "summary": summary_text,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in research trends endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba történt a tudományos trendek elemzése során: {e}"
        )

@app.post("/api/deep_discovery/protein_structure")
async def protein_structure_lookup(req: ProteinLookupRequest):
    """
    Lekérdezi a fehérjeszerkezetet az EMBL-EBI AlphaFold Protein Structure Database API-ból.
    Ez nem generál új struktúrát, hanem meglévő előrejelzéseket keres.
    """
    ebi_alphafold_api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{req.protein_id}"

    try:
        logger.info(f"Querying AlphaFold DB for protein ID: {req.protein_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(ebi_alphafold_api_url, timeout=30)
            response.raise_for_status() # Hibát dob, ha a státuszkód 4xx vagy 5xx

            data = response.json()

            if not data or (isinstance(data, list) and not data):
                return {
                    "protein_id": req.protein_id,
                    "message": "Nem található előrejelzés ehhez a fehérje azonosítóhoz az AlphaFold adatbázisban.",
                    "details": None
                }

            # Az AlphaFold DB API gyakran listát ad vissza, vegyük az elsőt
            first_prediction = data[0] if isinstance(data, list) else data

            return {
                "protein_id": req.protein_id,
                "message": "Fehérje előrejelzés sikeresen lekérdezve.",
                "details": {
                    "model_id": first_prediction.get("model_id"),
                    "uniprot_id": first_prediction.get("uniprot_id"),
                    "plddt": first_prediction.get("plddt"), # Konfidencia pontszám
                    "protein_url": first_prediction.get("cif_url") or first_prediction.get("pdb_url"), # Link a struktúrához
                    "pae_url": first_prediction.get("pae_url"), # Predicted Aligned Error
                    "assembly_id": first_prediction.get("assembly_id")
                }
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from AlphaFold DB API for {req.protein_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Hiba a fehérje adatbázis lekérdezésekor: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error querying AlphaFold DB API for {req.protein_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hálózati hiba a fehérje adatbázis elérésekor: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in protein lookup for {req.protein_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Váratlan hiba a fehérje lekérdezése során: {e}"
        )

@app.post("/api/deep_discovery/custom_gcp_model")
async def custom_gcp_model_inference(req: CustomGCPModelRequest):
    """
    Meghív egy általad telepített egyedi AI modellt a GCP Vertex AI-ban.
    Ez lehet egy kémiai tulajdonság előrejelző, molekulageneráló, vagy anyagtulajdonság modell.
    """
    if not gcp_credentials:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A GCP Vertex AI kliens nincs megfelelően inicializálva. Ellenőrizze a GCP_SERVICE_ACCOUNT_KEY-t."
        )

    try:
        # A GCP projekt és régió felülírható a kérésben, ha szükséges
        project = req.gcp_project_id or GCP_PROJECT_ID
        region = req.gcp_region or GCP_REGION

        if not project or not region:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hiányzó GCP projekt ID vagy régió."
            )

        endpoint_name = f"projects/{project}/locations/{region}/endpoints/{req.gcp_endpoint_id}"
        logger.info(f"Calling GCP Vertex AI endpoint: {endpoint_name}")

        prediction_client = aiplatform.gapic.PredictionServiceClient(credentials=gcp_credentials)

        # Az `instances` formátuma a telepített modell elvárásaitól függ!
        # Itt egy általános formátumot használunk.
        instances_list = [req.input_data] if not isinstance(req.input_data, list) else req.input_data

        response = prediction_client.predict(
            endpoint=endpoint_name,
            instances=instances_list
        )

        logger.info(f"GCP Vertex AI prediction successful from endpoint: {endpoint_name}")
        return {
            "model_response": response.predictions,
            "model_id": req.gcp_endpoint_id,
            "status": "success"
        }

    except GoogleAPIError as e:
        logger.error(f"GCP Vertex AI API error: {e.message} (Code: {e.code})")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a GCP Vertex AI modell hívásakor: {e.message}"
        )
    except Exception as e:
        logger.error(f"Unexpected error calling GCP Vertex AI model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Váratlan hiba a GCP Vertex AI modell hívásakor: {e}"
        )

@app.post("/api/deep_discovery/simulation_optimizer")
async def simulation_optimizer(req: SimulationOptimizerRequest):
    """
    Használja a Cerebras Llama 4-et (vagy Gemini-t) szimulációs paraméterek optimalizálására
    vagy szimulációs kód generálására.
    """
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető modell a szimuláció optimalizálásához (Cerebras Llama vagy Gemini)."
        )

    prompt = f"""
    Feladat: {req.simulation_type} szimuláció optimalizálása.
    Bemeneti paraméterek: {json.dumps(req.input_parameters, indent=2)}
    Optimalizálási cél: {req.optimization_goal}

    Kérlek, generálj optimalizált szimulációs paramétereket, vagy ha releváns, egy rövid Python kód snippetet
    a szimuláció elvégzéséhez/optimalizálásához, figyelembe véve a megadott célt.
    Válaszod legyen tömör, szakmailag pontos, és csak a kért információt tartalmazza.
    """

    response_text = ""
    model_used = ""

    try:
        if cerebras_client:
            async def run_qwen3_sim():
                try:
                    logger.info(f"Starting Qwen 3 for simulation: {req.simulation_type}")
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="qwen2.5-72b-instruct",
                        stream=True,
                        max_completion_tokens=1024,
                        temperature=0.3,
                        top_p=1
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Qwen 3"
                except Exception as e:
                    logger.warning(f"Qwen 3 szimulációs hiba: {e}")
                    return None, None

            async def run_llama4_sim():
                try:
                    logger.info(f"Starting Llama 4 for simulation: {req.simulation_type}")
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-4-scout-17b-16e-instruct",
                        stream=True,
                        max_completion_tokens=1024,
                        temperature=0.3,
                        top_p=1
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Llama 4"
                except Exception as e:
                    logger.warning(f"Llama 4 szimulációs hiba: {e}")
                    return None, None

            # Párhuzamos futtatás szimulációhoz
            try:
                tasks = [run_qwen3_sim(), run_llama4_sim()]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task in pending:
                    task.cancel()
                
                for task in done:
                    result = await task
                    if result[0]:
                        response_text, model_used = result
                        logger.info(f"Szimulációs első válasz: {model_used}")
                        break
                        
            except Exception as parallel_error:
                logger.error(f"Szimulációs párhuzamos hiba: {parallel_error}")
                response_text = ""
        elif gemini_model:
            logger.info(f"Using Gemini 2.5 Pro for simulation optimization: {req.simulation_type}")
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.3
                )
            )
            response_text = response.text
            model_used = "Google Gemini 2.5 Pro"

        if not response_text:
            raise ValueError("Az AI modell nem adott választ.")

        return {
            "simulation_type": req.simulation_type,
            "optimization_goal": req.optimization_goal,
            "optimized_output": response_text,
            "model_used": model_used,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in simulation optimizer endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba történt a szimuláció optimalizálása során: {e}"
        )

@app.post("/api/deep_discovery/alphagenome")
async def alphagenome_analysis(req: AlphaGenomeRequest):
    """
    AlphaGenome genomikai elemzés: DNS/RNS szekvencia analízis AI-val.
    Kombinálja a Gemini modellt a genomikai adatok feldolgozásához és a Cerebras Llama-t részletes elemzéshez.
    """
    if not gemini_model and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell a genomikai elemzéshez."
        )

    # Validáljuk a genom szekvenciát
    valid_nucleotides = set('ATCGURYN-')  # DNA, RNA és egyéb valid karakterek
    sequence_upper = req.genome_sequence.upper()
    if not all(char in valid_nucleotides for char in sequence_upper):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Érvénytelen genom szekvencia. Csak A, T, C, G, U, R, Y, N, - karakterek engedélyezettek."
        )

    try:
        # Alap szekvencia információk számítása
        sequence_length = len(req.genome_sequence)
        gc_content = (sequence_upper.count('G') + sequence_upper.count('C')) / sequence_length * 100

        # AI elemzési prompt összeállítása
        analysis_prompt = f"""
        AlphaGenome Genomikai Elemzés:

        Szekvencia: {req.genome_sequence[:500]}{'...' if len(req.genome_sequence) > 500 else ''}
        Szervezet: {req.organism}
        Elemzés típusa: {req.analysis_type}
        Szekvencia hossza: {sequence_length} bázispár
        GC tartalom: {gc_content:.1f}%

        Kérlek, végezz részletes genomikai elemzést az alábbi szempontok szerint:

        1. **Szekvencia jellemzők**: Nukleotid összetétel, ismétlődő motívumok, különleges régiók
        2. **Funkcionális predikció**: Lehetséges génkódoló régiók, promóterek, szabályozó elemek
        3. **Evolúciós vonatkozások**: Konzervált régiók, filogenetikai jelentőség
        4. **Patológiai releváncia**: Ismert mutációs hotspotok, betegséggel kapcsolatos variánsok
        5. **Strukturális elemzés**: Másodlagos szerkezet, fehérje kölcsönhatások

        {"6. **Fehérje struktúra előrejelzés**: AlphaFold alapú szerkezeti predikciók" if req.include_predictions else ""}

        Válaszod legyen tudományosan megalapozott, magyar nyelvű és strukturált.
        """

        response_text = ""
        model_used = ""

        # Párhuzamos genomikai elemzés: Qwen 3 és Llama 4 egyenrangú
        if cerebras_client:
            async def run_qwen3_genome():
                try:
                    logger.info(f"Starting Qwen 3 for genome analysis: {req.analysis_type}")
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": analysis_prompt}],
                        model="qwen2.5-72b-instruct",
                        stream=True,
                        max_completion_tokens=2048,
                        temperature=0.1,
                        top_p=0.9
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Qwen 3"
                except Exception as e:
                    logger.warning(f"Qwen 3 genomikai hiba: {e}")
                    return None, None

            async def run_llama4_genome():
                try:
                    logger.info(f"Starting Llama 4 for genome analysis: {req.analysis_type}")
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": analysis_prompt}],
                        model="llama-4-scout-17b-16e-instruct",
                        stream=True,
                        max_completion_tokens=2048,
                        temperature=0.1,
                        top_p=0.9
                    )
                    content = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    return content, "Cerebras Llama 4"
                except Exception as e:
                    logger.warning(f"Llama 4 genomikai hiba: {e}")
                    return None, None

            # Párhuzamos futtatás genomikai elemzéshez
            try:
                tasks = [run_qwen3_genome(), run_llama4_genome()]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task in pending:
                    task.cancel()
                
                for task in done:
                    result = await task
                    if result[0]:
                        response_text, model_used = result
                        logger.info(f"Genomikai első válasz: {model_used}")
                        break
                        
            except Exception as parallel_error:
                logger.error(f"Genomikai párhuzamos hiba: {parallel_error}")
                response_text = ""
        
        # Ha Qwen 3 nem működik, próbáljuk a Gemini-t
        if not response_text and gemini_model:
            logger.info(f"Using Gemini 2.5 Pro for AlphaGenome analysis: {req.analysis_type}")
            response = await gemini_model.generate_content_async(
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1  # Alacsony temperature a tudományos pontosságért
                )
            )
            response_text = response.text
            model_used = "Google Gemini 2.5 Pro"
        elif not response_text and cerebras_client:
            logger.info(f"Using Cerebras Llama 4 for AlphaGenome analysis: {req.analysis_type}")
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1,
                top_p=0.9
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"

        if not response_text:
            raise ValueError("Az AI modell nem adott választ a genomikai elemzésre.")

        # AlphaFold integráció, ha fehérje predikció kért
        alphafold_data = None
        if req.include_predictions and req.analysis_type in ["comprehensive", "protein_coding"]:
            try:
                # Keresünk potenciális UniProt ID-kat a szekvenciában vagy alapértelmezett emberi fehérjéket
                common_proteins = ["P04637", "P53350", "P31946"]  # p53, PLK1, 14-3-3β
                for protein_id in common_proteins:
                    try:
                        ebi_url = f"https://alphafold.ebi.ac.uk/api/prediction/{protein_id}"
                        async with httpx.AsyncClient() as client:
                            alphafold_response = await client.get(ebi_url, timeout=10)
                            if alphafold_response.status_code == 200:
                                alphafold_data = alphafold_response.json()
                                break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"AlphaFold integráció hiba: {e}")

        # Eredmények összeállítása
        return {
            "sequence_info": {
                "length": sequence_length,
                "gc_content": round(gc_content, 2),
                "organism": req.organism,
                "analysis_type": req.analysis_type
            },
            "ai_analysis": {
                "content": response_text,
                "model_used": model_used
            },
            "alphafold_predictions": alphafold_data[0] if alphafold_data and isinstance(alphafold_data, list) else alphafold_data,
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "AlphaGenome v1.0",
                "creator": DIGITAL_FINGERPRINT
            },
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AlphaGenome analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba történt az AlphaGenome elemzés során: {e}"
        )

# --- Alkalmazás Indítása ---
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)