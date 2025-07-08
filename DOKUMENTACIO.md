# Deep Discovery AI - Jade Asszisztens Dokumentáció

## Alkalmazás áttekintése
A Deep Discovery AI platform egy fejlett mesterséges intelligencia alapú tudományos és innovációs katalizátor alkalmazás hibrid AI modellekkel.

## Hibrid AI Architektura
Az alkalmazás három AI modellt használ intelligens prioritási sorrendben:
1. **Cerebras Qwen 3** (qwen2.5-72b-instruct) - Elsődleges modell
2. **Cerebras Llama 4** (llama-4-scout-17b-16e-instruct) - Másodlagos modell
3. **Google Gemini 2.5 Pro** - Tartalék modell

## Fájlok és funkcióik

### main.py
```python
import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import httpx # Aszinkron HTTP kérésekhez
import logging

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
        gcp_credentials = None
else:
    logger.warning("GCP_SERVICE_ACCOUNT_KEY, GCP_PROJECT_ID, or GCP_REGION not found. GCP Vertex AI functionality will be limited.")

# Cerebras kliens (Qwen 3 + Llama 4 támogatással)
cerebras_client = None
if CEREBRAS_API_KEY:
    try:
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized successfully (Qwen 3 + Llama 4 support).")
    except Exception as e:
        logger.error(f"Error initializing Cerebras client: {e}")
else:
    logger.warning("CEREBRAS_API_KEY not found. Cerebras functionality will be limited.")

# Gemini kliens
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
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
    description="Hibrid AI platform Qwen 3, Llama 4, Gemini, Exa, AlphaFold és AlphaGenome modellekkel.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS konfiguráció
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

class AlphaGenomeRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=100, description="A teljes DNS vagy RNS szekvencia elemzésre.")
    organism: str = Field(..., description="A szervezet, amelyből a genom származik (pl. 'Homo sapiens').")
    analysis_type: str = Field(..., description="Az elemzés típusa ('átfogó', 'génkódoló régiók', 'funkcionális elemek').")
    include_predictions: bool = Field(default=False, description="Tartalmazzon-e fehérje struktúra előrejelzéseket (AlphaFold).")
```

### templates/index.html
Teljes frontend HTML fájl a chat alkalmazáshoz Apple Sign In integrációval és modern dark UI-val.

### templates/login.html  
Bejelentkezési oldal Apple Sign In és hagyományos email/jelszó opcióval.

### requirements.txt
Az alkalmazás összes Python függősége.

### pyproject.toml
A projekt konfigurációs fájlja.

## Hibrid AI Működése
1. **Chat funkció**: Qwen 3 ⚡ Llama 4 párhuzamos futtatás (gyorsabb nyer) → Gemini fallback
2. **Tudományos elemzés**: Qwen 3 ⚡ Llama 4 párhuzamos → Gemini fallback
3. **Szimuláció optimalizálás**: Qwen 3 ⚡ Llama 4 párhuzamos → Gemini fallback
4. **Genomikai elemzés**: Qwen 3 ⚡ Llama 4 párhuzamos → Gemini fallback

### Párhuzamos Futtatás Előnyei
- **Gyorsabb válaszidő**: Az első válaszadó modell eredménye kerül felhasználásra
- **Magasabb rendelkezésre állás**: Ha az egyik modell nem elérhető, a másik továbbra is működik
- **Optimalizált erőforrás-használat**: Automatikus feladat-megszakítás a gyorsabb válasz után

## API végpontok
- `/api/deep_discovery/chat` - Hibrid chat funkció
- `/api/deep_discovery/research_trends` - Tudományos trendek elemzése
- `/api/deep_discovery/protein_structure` - AlphaFold fehérje lekérdezés
- `/api/deep_discovery/simulation_optimizer` - Hibrid szimuláció optimalizálás
- `/api/deep_discovery/alphagenome` - Genomikai elemzés hibrid AI-val
- `/api/deep_discovery/custom_gcp_model` - Egyedi GCP modellek
- `/api/auth/apple` - Apple Sign In autentikáció
- `/health` - Rendszer állapot ellenőrzés

## Technológiai stack
- **Backend**: FastAPI + Python
- **AI Modellek**: Cerebras Qwen 3, Cerebras Llama 4, Google Gemini 2.5 Pro
- **Tudományos adatok**: Exa AI, AlphaFold DB, AlphaGenome
- **Cloud**: Google Cloud Platform Vertex AI
- **Frontend**: Modern HTML5 + CSS3 + JavaScript
- **Autentikáció**: Apple Sign In

## Telepítés és futtatás
```bash
# Függőségek telepítése
pip install -r requirements.txt

# Titkok beállítása Replit Secrets-ben:
# CEREBRAS_API_KEY
# GEMINI_API_KEY  
# EXA_API_KEY
# GCP_SERVICE_ACCOUNT_KEY
# GCP_PROJECT_ID
# GCP_REGION

# Alkalmazás futtatása
python main.py
```

A hibrid rendszer automatikusan választja ki a legjobb elérhető modellt minden feladathoz.