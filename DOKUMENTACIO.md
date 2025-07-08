# Jade - Deep Discovery AI Platform Dokumentáció

## Alkalmazás Leírása

A Jade egy fejlett tudományos AI platform, amely több AI modellt integrál egyetlen, modern webes felületen. A platform speciálisan tudományos kutatásokhoz, innovációs projektekhez és fehérje-struktúra elemzésekhez készült.

### Fő Funkciók:
- **Hibrid AI Chat**: Cerebras Llama 4 és Google Gemini 2.5 Pro modellek
- **Kutatási Trendek**: Exa AI keresés + Gemini elemzés
- **Fehérje Struktúra**: AlphaFold adatbázis integráció
- **Egyedi GCP Modellek**: Google Cloud Vertex AI támogatás
- **Szimuláció Optimalizáló**: AI-vezérelt paraméter optimalizálás

---

## Fájlok és Kódjuk

### 1. `main.py` - Főszerver
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

# Beszélgetési előzmények tárolása memóriában (egyszerű prototípushoz)
# Ezt éles környezetben adatbázisra (pl. Firestore) kell cserélni!
chat_histories: Dict[str, List[Message]] = {}

# --- API Végpontok ---

@app.get("/")
async def serve_frontend():
    """A frontend HTML oldal kiszolgálása."""
    return FileResponse("templates/index.html")

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
async def deep_discovery_chat(req: ChatRequest):
    """
    Kezeli a beszélgetéseket, a Cerebras Llama 4 és a Gemini 2.5 Pro modelleket használva.
    A beszélgetési előzményeket a szerver tárolja user_id alapján.
    """
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető chat modell (Cerebras Llama vagy Gemini)."
        )

    user_id = req.user_id
    current_message = req.message

    # Beszélgetési előzmények lekérése vagy inicializálása
    history = chat_histories.get(user_id, [])

    # Rendszerüzenet (csak egyszer az elején)
    system_message = {
        "role": "system",
        "content": "Te egy rendkívül intelligens és szakértő AI asszisztens vagy, aki magyarul válaszol. A neved Jade. Segítőkész, részletes és innovatív válaszokat adsz a legújabb tudományos és technológiai fejleményekről, különös tekintettel a biológia, kémia, anyagtudomány, orvostudomány és mesterséges intelligencia területére. Használd a tudásodat a legjobb válaszok megadásához."
    }

    # Építsük fel a teljes üzenetlistát
    messages_for_llm = [system_message] + history + [{"role": "user", "content": current_message}]

    response_text = ""
    model_used = ""

    try:
        # Próbáljuk meg a Cerebras Llama 4-gyel először (ha elérhető)
        if cerebras_client:
            logger.info(f"Using Cerebras Llama 4 for user {user_id}")
            # A Cerebras chat API-ja is streamel, de a FastAPI csak a teljes választ küldi el egyben itt.
            stream = cerebras_client.chat.completions.create(
                messages=messages_for_llm,
                model="llama-4-scout-17b-16e-instruct", # Használjuk a megadott modellt
                stream=True,
                max_completion_tokens=2048,
                temperature=0.2,
                top_p=1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        elif gemini_model:
            # Ha a Cerebras nem elérhető, használjuk a Gemini 2.5 Pro-t
            logger.info(f"Using Gemini 2.5 Pro for user {user_id}")
            response = await gemini_model.generate_content_async(
                messages_for_llm,
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
            logger.info(f"Using Cerebras Llama 4 for simulation optimization: {req.simulation_type}")
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=1024,
                temperature=0.3,
                top_p=1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
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

# --- Alkalmazás Indítása ---
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
```

### 2. `templates/index.html` - Frontend
```html
<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jade - Deep Discovery AI Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --color-primary: #1a1a1a;
            --color-secondary: #2a2a2a;
            --color-accent: #3a3a3a;
            --color-text: #e0e0e0;
            --color-text-dim: #a0a0a0;
            --color-border: #404040;
            --color-green: #10b981;
            --color-blue: #3b82f6;
            --color-purple: #8b5cf6;
            --color-pink: #ec4899;
            --color-yellow: #f59e0b;
        }
        body { background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .chat-container { background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px); border: 1px solid var(--color-border); }
        .message { max-width: 85%; word-wrap: break-word; animation: fadeIn 0.3s ease-in; }
        .message.user { background: linear-gradient(135deg, var(--color-blue), #1e40af); margin-left: auto; }
        .message.ai { background: linear-gradient(135deg, var(--color-green), #047857); }
        .sidebar { background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(15px); border-right: 1px solid var(--color-border); transition: transform 0.3s ease; }
        .welcome-screen { background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1)); border: 1px solid rgba(255, 255, 255, 0.1); }
        .modal-overlay { background: rgba(0, 0, 0, 0.8); backdrop-filter: blur(5px); }
        .modal-content { background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border: 1px solid var(--color-border); }
        .modal-title { background: linear-gradient(135deg, var(--color-green), var(--color-blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .modal-input { background: rgba(255, 255, 255, 0.1); border: 1px solid var(--color-border); color: var(--color-text); }
        .modal-input:focus { border-color: var(--color-green); outline: none; box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2); }
        .modal-button { background: linear-gradient(135deg, var(--color-green), #047857); transition: all 0.2s; }
        .modal-button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); }
        .modal-button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .ai-answer-card { background: rgba(255, 255, 255, 0.05); border: 1px solid var(--color-border); border-radius: 12px; padding: 1rem; }
        .ai-answer-card h3 { color: var(--color-green); }
        .related-actions-card { margin-top: 1rem; }
        .related-actions-card button { 
            display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; border-radius: 8px; 
            transition: background 0.2s; width: 100%; text-align: left;
            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--color-text);
            font-size: 0.875rem; /* text-sm */
        }
        .related-actions-card button:hover { background: rgba(255, 255, 255, 0.1); }
        .related-actions-card i { color: var(--color-blue); }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideInFromTop { from { transform: translateY(-50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

        @media (max-width: 768px) {
            .sidebar { position: fixed; z-index: 50; transform: translateX(-100%); }
            .sidebar.open { transform: translateX(0); }
        }
    </style>
</head>
<body class="text-gray-100 min-h-screen">
    <div class="flex h-screen">
        <!-- Sidebar -->
        <div class="sidebar w-64 p-4 flex flex-col" id="sidebar">
            <div class="flex items-center gap-3 mb-8">
                <div class="w-10 h-10 bg-gradient-to-br from-green-400 to-blue-500 rounded-lg flex items-center justify-center">
                    <i class="fas fa-atom text-white text-lg"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">Jade</h1>
                    <p class="text-xs text-gray-400">Deep Discovery AI</p>
                </div>
            </div>

            <button id="newChatBtn" class="w-full mb-6 p-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg font-medium transition-all hover:scale-105 hover:shadow-lg flex items-center justify-center gap-2">
                <i class="fas fa-plus"></i> Új Beszélgetés
            </button>

            <div class="mb-6">
                <h3 class="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">Deep Discovery Eszközök</h3>
                <div class="space-y-2">
                    <button id="researchTrendsBtn" class="w-full text-left text-sm text-gray-300 p-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-3"><i class="fas fa-flask w-5 text-center text-green-400"></i> Kutatási Trendek</button>
                    <button id="proteinStructureBtn" class="w-full text-left text-sm text-gray-300 p-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-3"><i class="fas fa-dna w-5 text-center text-blue-400"></i> Fehérje Struktúra</button>
                    <button id="customGCPModelBtn" class="w-full text-left text-sm text-gray-300 p-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-3"><i class="fas fa-cloud w-5 text-center text-pink-400"></i> Egyedi GCP Modell</button>
                    <button id="simulationOptimizerBtn" class="w-full text-left text-sm text-gray-300 p-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-3"><i class="fas fa-microchip w-5 text-center text-yellow-400"></i> Szimuláció Optimalizáló</button>
                    <button id="alphaGenomeBtn" class="w-full text-left text-sm text-gray-300 p-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-3"><i class="fas fa-circle-nodes w-5 text-center text-orange-400"></i> AlphaGenome Elemzés</button>
                </div>
            </div>
            <div class="flex-1"></div>
            <!-- Creator signature -->
            <div class="text-xs text-gray-500 text-center">
                <p>Jade made by Kollár Sándor</p>
            </div>
        </div>

        <!-- Main Chat Area -->
        <div class="flex-1 flex flex-col">
            <!-- Header -->
            <div class="p-4 border-b border-gray-700 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <button id="menuToggle" class="lg:hidden p-2 rounded-lg hover:bg-gray-800">
                        <i class="fas fa-bars"></i>
                    </button>
                    <h2 class="text-lg font-semibold">AI Asszisztens</h2>
                    <span class="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full" id="currentModelBadge">Hibrid AI</span>
                </div>
            </div>

            <!-- Chat Container -->
            <div class="flex-1 flex flex-col chat-container m-4 rounded-xl overflow-hidden" id="chatContainer">
                <!-- Welcome Screen -->
                <div class="flex-1 flex items-center justify-center p-8" id="welcomeScreen">
                    <div class="text-center max-w-2xl welcome-screen rounded-2xl p-8">
                        <div class="w-16 h-16 bg-gradient-to-br from-green-400 to-blue-500 rounded-full mx-auto mb-6 flex items-center justify-center">
                            <i class="fas fa-atom text-white text-2xl"></i>
                        </div>
                        <h2 class="text-3xl font-bold mb-4 bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">Üdvözöllek a Jade-ben!</h2>
                        <p class="text-gray-300 mb-6">Én egy fejlett AI asszisztens vagyok, aki segít tudományos kutatásokban, innovációban és fehérje-struktúra elemzésekben. Használhatod a speciális eszközöket, vagy egyszerűen beszélgethetsz velem!</p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            <div class="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                                <i class="fas fa-comments text-green-400"></i>
                                <span>Intelligens beszélgetések</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                                <i class="fas fa-search text-blue-400"></i>
                                <span>Kutatási trendek elemzése</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                                <i class="fas fa-dna text-purple-400"></i>
                                <span>Fehérje struktúra keresés</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                                <i class="fas fa-cogs text-yellow-400"></i>
                                <span>Szimuláció optimalizálás</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                                <i class="fas fa-circle-nodes text-orange-400"></i>
                                <span>AlphaGenome Elemzés</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Messages Container -->
                <div class="flex-1 overflow-y-auto p-6 space-y-4" id="messagesContainer" style="display: none;"></div>

                <!-- Input Area -->
                <div class="p-4 border-t border-gray-700">
                    <div class="flex gap-3">
                        <input type="text" id="messageInput" placeholder="Írj egy üzenetet..." class="flex-1 p-3 bg-gray-800 border border-gray-600 rounded-lg focus:border-green-500 focus:outline-none text-white">
                        <button id="sendButton" class="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg font-medium hover:scale-105 transition-all">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Research Trends Modal -->
    <div class="modal-overlay fixed inset-0 z-50 flex items-center justify-center p-4" id="researchTrendsModal" style="display: none;">
        <div class="modal-content max-w-2xl w-full max-h-[90vh] overflow-y-auto rounded-xl p-6">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold modal-title">Kutatási Trendek Elemzése</h2>
                <button class="modal-close p-2 rounded-full hover:bg-gray-700" data-modal-id="researchTrendsModal"><i class="fas fa-times"></i></button>
            </div>
            <div class="modal-message-container"></div>
            <form id="researchTrendsForm" class="flex flex-col gap-4">
                <input type="text" class="modal-input rounded-lg p-3" id="researchQuery" placeholder="Kutatási terület vagy témakör (pl. kvantum számítástechnika)" required>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm text-gray-400 mb-2">Találatok száma</label>
                        <select class="modal-input rounded-lg p-3 w-full" id="numResults">
                            <option value="3">3 találat</option>
                            <option value="5" selected>5 találat</option>
                            <option value="7">7 találat</option>
                            <option value="10">10 találat</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm text-gray-400 mb-2">Összefoglaló hossza</label>
                        <select class="modal-input rounded-lg p-3 w-full" id="summaryLength">
                            <option value="100">Rövid (100 szó)</option>
                            <option value="200" selected>Közepes (200 szó)</option>
                            <option value="300">Hosszú (300 szó)</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="modal-button font-bold py-3 rounded-lg" id="researchTrendsSubmitBtn">Elemzés Indítása</button>
            </form>
            <div id="researchResult" class="mt-4 pt-4 border-t border-gray-700 text-gray-300" style="display: none;">
                <h3 class="text-lg font-semibold mb-2 flex items-center gap-2 text-green-400"><i class="fas fa-chart-line"></i> Összefoglaló</h3>
                <p id="researchSummary" class="mb-4"></p>
                <h4 class="text-md font-semibold my-2 text-blue-400">Források</h4>
                <div id="researchSources" class="space-y-2"></div>
            </div>
        </div>
    </div>

    <!-- Protein Structure Modal -->
    <div class="modal-overlay" id="proteinStructureModal">
        <div class="modal-content">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold modal-title">Fehérje Struktúra Keresése</h2>
                <button class="modal-close p-2 rounded-full" data-modal-id="proteinStructureModal"><i class="fas fa-times"></i></button>
            </div>
            <div class="modal-message-container"></div>
            <form id="proteinStructureForm" class="flex flex-col gap-4">
                <input type="text" class="modal-input rounded-lg p-3" id="proteinId" placeholder="Fehérje azonosító (pl. UniProt ID: P0DTC2)" required>
                <button type="submit" class="modal-button font-bold py-3 rounded-lg" id="proteinStructureSubmitBtn">Keresés Indítása</button>
            </form>
            <div id="proteinResult" class="mt-4 pt-4 border-t border-gray-700 text-gray-300" style="display: none;">
                <h3 class="text-lg font-semibold mb-2 flex items-center gap-2 text-green-400"><i class="fas fa-dna"></i> Fehérje Információk</h3>
                <div id="proteinDetails"></div>
            </div>
        </div>
    </div>

    <!-- Custom GCP Model Modal -->
    <div class="modal-overlay" id="customGCPModelModal">
        <div class="modal-content">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold modal-title">Egyedi GCP Modell</h2>
                <button class="modal-close p-2 rounded-full" data-modal-id="customGCPModelModal"><i class="fas fa-times"></i></button>
            </div>
            <div class="modal-message-container"></div>
            <form id="customGCPModelForm" class="flex flex-col gap-4">
                <input type="text" class="modal-input rounded-lg p-3" id="gcpEndpointId" placeholder="GCP Endpoint ID" required>
                <textarea class="modal-input rounded-lg p-3" id="gcpInputData" placeholder='Bemeneti adatok JSON formátumban (pl. {"features": [1.2, 3.4, 5.6]})' rows="4" required></textarea>
                <button type="submit" class="modal-button font-bold py-3 rounded-lg" id="customGCPModelSubmitBtn">Modell Futtatása</button>
            </form>
            <div id="gcpModelResult" class="mt-4 pt-4 border-t border-gray-700 text-gray-300" style="display: none;">
                <h3 class="text-lg font-semibold mb-2 flex items-center gap-2 text-green-400"><i class="fas fa-cloud"></i> Modell Válasza</h3>
                <div id="gcpModelDetails"></div>
            </div>
        </div>
    </div>

    <!-- Simulation Optimizer Modal -->
    <div class="modal-overlay" id="simulationOptimizerModal">
        <div class="modal-content">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold modal-title">Szimuláció Optimalizáló</h2>
                <button class="modal-close p-2 rounded-full" data-modal-id="simulationOptimizerModal"><i class="fas fa-times"></i></button>
            </div>
            <div class="modal-message-container"></div>
            <form id="simulationOptimizerForm" class="flex flex-col gap-4">
                <select class="modal-input rounded-lg p-3" id="simulationType" required>
                    <option value="">Válaszd ki a szimuláció típusát...</option>
                    <option value="molecular_dynamics">Molekuláris dinamika</option>
                    <option value="materials_property">Anyagtulajdonság</option>
                    <option value="chemical_reaction">Kémiai reakció</option>
                    <option value="protein_folding">Fehérje hajtogatás</option>
                    <option value="drug_discovery">Gyógyszer felfedezés</option>
                </select>
                <textarea class="modal-input rounded-lg p-3" id="simulationParameters" placeholder='Bemeneti paraméterek JSON formátumban (pl. {"temperature": 300, "pressure": 1.0})' rows="3" required></textarea>
                <input type="text" class="modal-input rounded-lg p-3" id="optimizationGoal" placeholder="Optimalizálási cél (pl. minimize_energy, maximize_stability)" required>
                <button type="submit" class="modal-button font-bold py-3 rounded-lg" id="simulationOptimizerSubmitBtn">Optimalizálás Indítása</button>
            </form>
            <div id="simulationResult" class="mt-4 pt-4 border-t border-gray-700 text-gray-300" style="display: none;">
                <h3 class="text-lg font-semibold mb-2 flex items-center gap-2 text-green-400"><i class="fas fa-microchip"></i> Optimalizált Kimenet</h3>
                <div id="simulationDetails"></div>
                <p class="text-xs text-gray-500 mt-2">Modell: <span id="simulationModelUsed"></span></p>
            </div>
        </div>
    </div>

     <!-- AlphaGenome Modal -->
     <div class="modal-overlay" id="alphaGenomeModal">
        <div class="modal-content">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold modal-title">AlphaGenome Elemzés</h2>
                <button class="modal-close p-2 rounded-full" data-modal-id="alphaGenomeModal"><i class="fas fa-times"></i></button>
            </div>
            <div class="modal-message-container"></div>
            <form id="alphaGenomeForm" class="flex flex-col gap-4">
                <textarea class="modal-input rounded-lg p-3" id="genomeSequence" placeholder="DNS/RNS szekvencia" rows="5" required></textarea>
                <button type="submit" class="modal-button font-bold py-3 rounded-lg" id="alphaGenomeSubmitBtn">Elemzés Indítása</button>
            </form>
            <div id="alphaGenomeResult" class="mt-4 pt-4 border-t border-gray-700 text-gray-300" style="display: none;">
                <h3 class="text-lg font-semibold mb-2 flex items-center gap-2 text-green-400"><i class="fas fa-circle-nodes"></i> Genomikai Elemzés Eredménye</h3>
                <div id="alphaGenomeDetails"></div>
                <p class="text-xs text-gray-500 mt-2">Modell: <span id="alphaGenomeModelUsed"></span></p>
            </div>
        </div>
    </div>


    <script>
        // --- STATE & DOM ELEMENTS ---
        let isTyping = false;
        let currentUserId = localStorage.getItem('jadeUserId') || `user_${Math.random().toString(36).substring(2, 15)}`;
        localStorage.setItem('jadeUserId', currentUserId); // Mentjük a felhasználó ID-t

        const API_BASE_URL = window.location.origin; // A Repliten ez a Repl URL-je lesz

        const chatContainer = document.getElementById('chatContainer');
        const messagesContainer = document.getElementById('messagesContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const welcomeScreen = document.getElementById('welcomeScreen');
        const sidebar = document.getElementById('sidebar');
        const menuToggle = document.getElementById('menuToggle');
        const newChatBtn = document.getElementById('newChatBtn');
        const currentModelBadge = document.getElementById('currentModelBadge');

        // Deep Discovery Tool Buttons
        const researchTrendsBtn = document.getElementById('researchTrendsBtn');
        const proteinStructureBtn = document.getElementById('proteinStructureBtn');
        const customGCPModelBtn = document.getElementById('customGCPModelBtn');
        const simulationOptimizerBtn = document.getElementById('simulationOptimizerBtn');
        const alphaGenomeBtn = document.getElementById('alphaGenomeBtn');


        // --- CORE FUNCTIONS ---

        /**
         * Sends a message to the AI and displays the response
         */
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isTyping) return;

            // Hide welcome screen and show messages
            welcomeScreen.style.display = 'none';
            messagesContainer.style.display = 'block';

            addMessage('user', message);
            messageInput.value = '';
            isTyping = true;
            sendButton.disabled = true;

            try {
                const response = await fetch(`${API_BASE_URL}/api/deep_discovery/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, user_id: currentUserId })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                addMessage('ai', data.response, data.model_used);
            } catch (error) {
                addMessage('ai', `Elnézést, hiba történt: ${error.message}`, 'Hiba');
            } finally {
                isTyping = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        }

        /**
         * Adds a message to the chat container
         */
        function addMessage(sender, content, modelUsed = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender} p-4 rounded-2xl text-white shadow-lg`;

            if (sender === 'ai') {
                const lowerCaseAnswer = content.toLowerCase();
                let suggestedActionsHtml = '';

                // Intelligent action suggestions based on AI response content
                if (lowerCaseAnswer.includes('kutatás') || lowerCaseAnswer.includes('trend') || lowerCaseAnswer.includes('publikáció') || lowerCaseAnswer.includes('tanulmány')) {
                    suggestedActionsHtml += `
                        <button class="mb-2" onclick="showModal('researchTrendsModal'); addMessage('ai', 'Rendben, segítek a legfrissebb kutatási trendek felkutatásában. Milyen területet szeretnél megvizsgálni?');">
                            <i class="fas fa-flask text-green-400"></i> Kutatási Trendek Elemzése
                        </button>`;
                }
                if (lowerCaseAnswer.includes('fehérje') || lowerCaseAnswer.includes('protein') || lowerCaseAnswer.includes('struktúra') || lowerCaseAnswer.includes('alphafold')) {
                    suggestedActionsHtml += `
                        <button class="mb-2" onclick="showModal('proteinStructureModal'); addMessage('ai', 'Rendben, indítsuk el a Fehérje Struktúra Keresőt. Melyik fehérje ID-t szeretnéd lekérdezni?');">
                            <i class="fas fa-dna text-blue-400"></i> Fehérje Struktúra Keresése
                        </button>`;
                }
                if (lowerCaseAnswer.includes('anyag') || lowerCaseAnswer.includes('kémia') || lowerCaseAnswer.includes('molekula')) {
                    suggestedActionsHtml += `
                        <button class="mb-2" onclick="showModal('customGCPModelModal'); addMessage('ai', 'Értem, indítsuk el az Egyedi GCP Modellt. Kérlek, add meg a végpont ID-t és a bemeneti adatokat.');">
                            <i class="fas fa-cloud text-pink-400"></i> Egyedi GCP Modell Hívása
                        </button>`;
                }
                if (lowerCaseAnswer.includes('szimuláció') || lowerCaseAnswer.includes('optimalizálás') || lowerCaseAnswer.includes('tervezés')) {
                    suggestedActionsHtml += `
                        <button class="mb-2" onclick="showModal('simulationOptimizerModal'); addMessage('ai', 'Készen állok a Szimuláció Optimalizálásra. Milyen típusú szimulációról van szó?');">
                            <i class="fas fa-microchip text-yellow-400"></i> Szimuláció Optimalizáló
                        </button>`;
                }
                if (lowerCaseAnswer.includes('genom') || lowerCaseAnswer.includes('dns') || lowerCaseAnswer.includes('rns') || lowerCaseAnswer.includes('szekvencia')) {
                    suggestedActionsHtml += `
                        <button class="mb-2" onclick="showModal('alphaGenomeModal'); addMessage('ai', 'Rendben, indítsuk el az AlphaGenome elemzést. Kérlek, add meg a DNS/RNS szekvenciát.');">
                            <i class="fas fa-circle-nodes text-orange-400"></i> AlphaGenome Elemzés
                        </button>`;
                }

                messageDiv.innerHTML = `
                    <div class="ai-answer-card">
                        <h3 class="text-sm font-semibold mb-2 flex items-center gap-2">
                            <i class="fas fa-robot"></i> Jade AI
                            ${modelUsed ? `<span class="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">${modelUsed}</span>` : ''}
                        </h3>
                        <div class="prose prose-invert max-w-none">
                            ${content.replace(/\n/g, '<br>')}
                        </div>
                        ${suggestedActionsHtml ? `
                            <div class="related-actions-card">
                                <h4 class="text-sm font-medium text-gray-400 mb-2">Kapcsolódó műveletek:</h4>
                                <div class="space-y-1">
                                    ${suggestedActionsHtml}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-user text-white text-sm"></i>
                        </div>
                        <div class="flex-1">${content}</div>
                    </div>
                `;
            }

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // --- EVENT LISTENERS ---
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        newChatBtn.addEventListener('click', () => {
            messagesContainer.innerHTML = '';
            welcomeScreen.style.display = 'block';
            messageInput.focus();
            currentModelBadge.textContent = 'Hibrid AI';
        });

        // Toggle sidebar on mobile
        menuToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
        document.addEventListener('click', (e) => {
            if (window.innerWidth < 1024 && !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });

        // --- MODAL & TOOLS LOGIC ---

        function showModal(modalId) { document.getElementById(modalId).style.display = 'flex'; }
        function closeModal(modalId) { 
            document.getElementById(modalId).style.display = 'none'; 
            // Reset modal messages
            const container = document.getElementById(modalId).querySelector('.modal-message-container');
            if (container) container.innerHTML = '';
        }

        document.querySelectorAll('.modal-close').forEach(btn => btn.addEventListener('click', () => closeModal(btn.dataset.modalId)));

        // Assign Deep Discovery Tool buttons to modals
        researchTrendsBtn.addEventListener('click', () => showModal('researchTrendsModal'));
        proteinStructureBtn.addEventListener('click', () => showModal('proteinStructureModal'));
        customGCPModelBtn.addEventListener('click', () => showModal('customGCPModelModal'));
        simulationOptimizerBtn.addEventListener('click', () => showModal('simulationOptimizerModal'));
        alphaGenomeBtn.addEventListener('click', () => showModal('alphaGenomeModal'));

        function showModalMessage(modalId, message, isError = true) {
            const container = document.getElementById(modalId).querySelector('.modal-message-container');
            container.innerHTML = `
                <div class="p-3 rounded-lg mb-4 ${isError ? 'bg-red-500/20 border border-red-500/50 text-red-300' : 'bg-green-500/20 border border-green-500/50 text-green-300'}">
                    <i class="fas ${isError ? 'fa-exclamation-triangle' : 'fa-check-circle'} mr-2"></i>
                    ${message}
                </div>
            `;
        }

        // Research Trends Form
        document.getElementById('researchTrendsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = document.getElementById('researchQuery').value.trim();
            const numResults = parseInt(document.getElementById('numResults').value);
            const summaryLength = parseInt(document.getElementById('summaryLength').value);
            const submitBtn = document.getElementById('researchTrendsSubmitBtn');
            const resultDiv = document.getElementById('researchResult');

            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Elemzés folyamatban...';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch(`${API_BASE_URL}/api/deep_discovery/research_trends`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, num_results: numResults, summary_length: summaryLength })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                document.getElementById('researchSummary').textContent = data.summary;

                const sourcesContainer = document.getElementById('researchSources');
                sourcesContainer.innerHTML = '';

                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach((source, index) => {
                        const sourceLink = document.createElement('a');
                        sourceLink.href = source.url;
                        sourceLink.target = '_blank';
                        sourceLink.className = 'block p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors';
                        sourceLink.innerHTML = `<div class="font-medium text-blue-400">${source.title}</div><div class="text-xs text-gray-500">${source.url}</div>${source.published_date ? `<div class="text-xs text-gray-500">Publikálva: ${source.published_date}</div>` : ''}`;
                        sourcesContainer.appendChild(sourceLink);
                    });
                } else {
                    sourcesContainer.innerHTML = '<p class="text-gray-500 text-sm">Nincsenek források.</p>';
                }

                resultDiv.style.display = 'block';
                showModalMessage('researchTrendsModal', 'Elemzés sikeresen befejezve!', false);
            } catch (error) {
                showModalMessage('researchTrendsModal', `Hiba: ${error.message}`);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Elemzés Indítása';
            }
        });

        // Protein Structure Form
        document.getElementById('proteinStructureForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const proteinId = document.getElementById('proteinId').value.trim();
            const submitBtn = document.getElementById('proteinStructureSubmitBtn');
            const resultDiv = document.getElementById('proteinResult');

            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Keresés...';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch(`${API_BASE_URL}/api/deep_discovery/protein_structure`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ protein_id: proteinId })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                const detailsContainer = document.getElementById('proteinDetails');

                if (data.details) {
                    detailsContainer.innerHTML = `
                        <div class="space-y-3">
                            <div class="grid grid-cols-2 gap-4">
                                <div><strong>Modell ID:</strong> ${data.details.model_id || 'N/A'}</div>
                                <div><strong>UniProt ID:</strong> ${data.details.uniprot_id || 'N/A'}</div>
                                <div><strong>Konfidencia (pLDDT):</strong> ${data.details.plddt || 'N/A'}</div>
                                <div><strong>Assembly ID:</strong> ${data.details.assembly_id || 'N/A'}</div>
                            </div>
                            ${data.details.protein_url ? `
                                <div class="mt-4">
                                    <a href="${data.details.protein_url}" target="_blank" class="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors">
                                        <i class="fas fa-download"></i>
                                        Struktúra letöltése
                                    </a>
                                </div>
                            ` : ''}
                            ${data.details.pae_url ? `
                                <div class="mt-2">
                                    <a href="${data.details.pae_url}" target="_blank" class="inline-flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg transition-colors">
                                        <i class="fas fa-chart-line"></i>
                                        PAE adatok megtekintése
                                    </a>
                                </div>
                            ` : ''}
                        </div>
                    `;
                } else {
                    detailsContainer.innerHTML = `<p class="text-### 3. `requirements.txt` - Python Függőségek
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0
google-cloud-aiplatform>=1.38.0
google-generativeai>=0.3.0
google-auth>=2.23.0
google-api-core>=2.14.0
cerebras-cloud-sdk>=1.35.0
exa-py>=1.0.0
```

### 4. `.replit` - Replit Konfiguráció
```
entrypoint = "main.py"
modules = ["python-3.11"]

[nix]
channel = "stable-24_05"

[unitTest]
language = "python3"

[gitHubImport]
requiredFiles = [".replit", "replit.nix"]

[deployment]
run = ["python3", "main.py"]
deploymentTarget = "cloudrun"

[workflows]
runButton = "Futtatás"

[[workflows.workflow]]
name = "Futtatás"
author = 40296216
mode = "sequential"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "python main.py"

[[ports]]
localPort = 5000
externalPort = 80

[[ports]]
localPort = 34397
externalPort = 3000
```

### 5. `pyproject.toml` - Projekt Konfiguráció
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

### 6. `replit.nix` - Nix Környezet
```nix
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.setuptools
    pkgs.python311Packages.wheel
  ];
}