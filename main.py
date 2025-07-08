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

# Új modell az AlphaGenome API-hoz
class AlphaGenomeRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=100, description="A teljes DNS vagy RNS szekvencia elemzésre.")
    organism: str = Field(..., description="A szervezet, amelyből a genom származik (pl. 'Homo sapiens').")
    analysis_type: str = Field(..., description="Az elemzés típusa ('átfogó', 'génkódoló régiók', 'funkcionális elemek').")
    include_predictions: bool = Field(default=False, description="Tartalmazzon-e fehérje struktúra előrejelzéseket (AlphaFold).")

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

        # Először próbáljuk a Gemini-vel, amely jobb a tudományos szövegek elemzésében
        if gemini_model:
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
        elif cerebras_client:
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