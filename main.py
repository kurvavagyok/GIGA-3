
import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import logging
from datetime import datetime
import hashlib
import base64

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

# Naplózás konfigurálása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Digitális Ujjlenyomat ---
DIGITAL_FINGERPRINT = "Jade made by Kollár Sándor"
CREATOR_SIGNATURE = "SmFkZSBtYWRlIGJ5IEtvbGzDoXIgU8OhbmRvcg=="
CREATOR_HASH = "a7b4c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5"

# --- API Kulcsok betöltése ---
GCP_SERVICE_ACCOUNT_KEY_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")

# --- Kliensek inicializálása ---
gcp_credentials = None
if GCP_SERVICE_ACCOUNT_KEY_JSON and GCP_PROJECT_ID and GCP_REGION:
    try:
        info = json.loads(GCP_SERVICE_ACCOUNT_KEY_JSON)
        gcp_credentials = service_account.Credentials.from_service_account_info(info)
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=gcp_credentials)
        logger.info("GCP Vertex AI client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing GCP Vertex AI client: {e}")
        gcp_credentials = None

cerebras_client = None
if CEREBRAS_API_KEY:
    try:
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Cerebras client: {e}")

# Gemini 2.5 Pro inicializálása
gemini_model = None
gemini_25_pro = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        gemini_25_pro = genai.GenerativeModel('gemini-2.5-pro')
        logger.info("Gemini 1.5 Pro and 2.5 Pro clients initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini clients: {e}")

exa_client = None
if EXA_API_KEY:
    try:
        exa_client = Exa(api_key=EXA_API_KEY)
        logger.info("Exa client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Exa client: {e}")

# --- FastAPI alkalmazás ---
app = FastAPI(
    title="Jade - Deep Discovery AI Platform",
    description="Fejlett AI platform 150+ tudományos és innovációs szolgáltatással",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="templates"), name="static")

# --- Pydantic modellek ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(..., description="Felhasználó egyedi azonosítója")

class DeepResearchRequest(BaseModel):
    query: str = Field(..., description="Kutatási kérdés")
    user_id: str = Field(..., description="Felhasználó azonosító")

class SimpleAlphaRequest(BaseModel):
    service_name: str = Field(..., description="Az Alpha szolgáltatás neve")
    query: str = Field(..., description="Egyszerű szöveges kérés")
    details: str = Field(default="", description="További részletek (opcionális)")

class UniversalAlphaRequest(BaseModel):
    service_name: str = Field(..., description="Az Alpha szolgáltatás neve")
    input_data: Dict[str, Any] = Field(..., description="Bemeneti adatok")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Opcionális paraméterek")

class ScientificInsightRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Tudományos lekérdezés")
    num_results: int = Field(default=5, ge=1, le=10, description="Találatok száma")
    summary_length: int = Field(default=200, ge=50, le=500, description="Összefoglaló hossza")

class AdvancedExaRequest(BaseModel):
    query: str = Field(..., description="Keresési lekérdezés")
    type: str = Field(default="neural", description="Keresés típusa: neural, keyword, similarity")
    num_results: int = Field(default=10, ge=1, le=50, description="Találatok száma")
    include_domains: List[str] = Field(default=[], description="Csak ezeken a domaineken keressen")
    exclude_domains: List[str] = Field(default=[], description="Ezeket a domaineket zárja ki")
    start_crawl_date: Optional[str] = Field(None, description="Kezdő dátum (YYYY-MM-DD)")
    end_crawl_date: Optional[str] = Field(None, description="Befejező dátum (YYYY-MM-DD)")
    start_published_date: Optional[str] = Field(None, description="Publikálás kezdő dátuma")
    end_published_date: Optional[str] = Field(None, description="Publikálás befejező dátuma")
    include_text: List[str] = Field(default=[], description="Ezeket a szövegeket tartalmaznia kell")
    exclude_text: List[str] = Field(default=[], description="Ezeket a szövegeket nem tartalmazhatja")
    category: Optional[str] = Field(None, description="Kategória szűrő")
    subcategory: Optional[str] = Field(None, description="Alkategória szűrő")
    livecrawl: str = Field(default="always", description="Live crawl: always, never, when_necessary")
    text_contents_options: Dict[str, Any] = Field(default_factory=lambda: {
        "max_characters": 2000,
        "include_html_tags": False,
        "strategy": "comprehensive"
    })

class ExaSimilarityRequest(BaseModel):
    url: str = Field(..., description="Referencia URL")
    num_results: int = Field(default=10, ge=1, le=50, description="Hasonló találatok száma")
    category_weights: Dict[str, float] = Field(default={}, description="Kategória súlyok")
    exclude_source_domain: bool = Field(default=True, description="Forrás domain kizárása")

class ExaContentsRequest(BaseModel):
    ids: List[str] = Field(..., description="Exa result ID-k")
    summary: bool = Field(default=True, description="Összefoglaló generálása")
    highlights: Dict[str, Any] = Field(default_factory=dict, description="Kiemelés opciók")

class ProteinLookupRequest(BaseModel):
    protein_id: str = Field(..., description="Fehérje azonosító")

class CustomGCPModelRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="GCP modell bemeneti adatok")
    gcp_endpoint_id: str = Field(..., description="GCP végpont azonosító")
    gcp_project_id: Optional[str] = GCP_PROJECT_ID
    gcp_region: Optional[str] = GCP_REGION

class SimulationOptimizerRequest(BaseModel):
    simulation_type: str = Field(..., description="Szimuláció típusa")
    input_parameters: Dict[str, Any] = Field(..., description="Bemeneti paraméterek")
    optimization_goal: str = Field(..., description="Optimalizálási cél")

class AlphaGenomeRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=100, description="Genom szekvencia")
    organism: str = Field(..., description="Organizmus")
    analysis_type: str = Field(..., description="Elemzés típusa")
    include_predictions: bool = Field(default=False, description="Fehérje előrejelzések")

# Beszélgetési előzmények
chat_histories: Dict[str, List[Message]] = {}

# --- Alpha Services definíciója ---
ALPHA_SERVICES = {
    "biologiai_orvosi": {
        "AlphaMicrobiome": "Mikrobiom elemzés és baktériumközösség vizsgálat",
        "AlphaImmune": "Immunrendszer válaszok predikciója",
        "AlphaCardio": "Szívbetegségek kockázatelemzése",
        "AlphaNeuron": "Idegsejt aktivitás szimulálása",
        "AlphaVirus": "Vírus mutáció előrejelzése",
        "AlphaCell": "Sejtosztódás és növekedés modellezése",
        "AlphaMetabolism": "Anyagcsere útvonalak elemzése",
        "AlphaPharmaco": "Gyógyszer-receptor kölcsönhatások",
        "AlphaGene": "Génexpresszió előrejelzése",
        "AlphaProteomics": "Fehérje hálózatok elemzése",
        "AlphaToxicology": "Toxicitás és biztonság értékelése",
        "AlphaEpigenetics": "Epigenetikai változások predikciója",
        "AlphaBiomarker": "Biomarker azonosítás és validálás",
        "AlphaPathogen": "Kórokozó azonosítás és karakterizálás",
        "AlphaOncology": "Rák biomarkerek és terápiás célpontok",
        "AlphaEndocrine": "Hormonális szabályozás modellezése",
        "AlphaRespiratory": "Légzési rendszer betegségei",
        "AlphaNeurodegeneration": "Neurodegeneratív betegségek",
        "AlphaRegenerative": "Regeneratív medicina alkalmazások",
        "AlphaPersonalized": "Személyre szabott orvoslás",
        "AlphaBioengineering": "Biomérnöki rendszerek tervezése",
        "AlphaBioinformatics": "Bioinformatikai adatelemzés",
        "AlphaSystemsBiology": "Rendszerbiológiai modellezés",
        "AlphaSynthbio": "Szintetikus biológiai rendszerek",
        "AlphaLongevity": "Öregedés és hosszú élet kutatása"
    },
    "kemiai_anyagtudomanyi": {
        "AlphaCatalyst": "Katalizátor tervezés és optimalizálás",
        "AlphaPolymer": "Polimer tulajdonságok előrejelzése",
        "AlphaNanotech": "Nanomateriál szintézis és jellemzés",
        "AlphaChemSynthesis": "Kémiai szintézis útvonalak",
        "AlphaMaterial": "Anyagtulajdonságok predikciója",
        "AlphaSuperconductor": "Szupravezető anyagok kutatása",
        "AlphaSemiconductor": "Félvezető anyagok tervezése",
        "AlphaComposite": "Kompozit anyagok fejlesztése",
        "AlphaBattery": "Akkumulátor technológiák",
        "AlphaSolar": "Napelem hatékonyság optimalizálása",
        "AlphaCorrosion": "Korrózió és védelem elemzése",
        "AlphaAdhesive": "Ragasztó és kötőanyagok",
        "AlphaCrystal": "Kristályszerkezet előrejelzése",
        "AlphaLiquid": "Folyadék tulajdonságok modellezése",
        "AlphaGas": "Gázfázisú reakciók szimulálása",
        "AlphaSurface": "Felületi kémia és adszorpció",
        "AlphaElectrochemistry": "Elektrokémiai folyamatok",
        "AlphaPhotochemistry": "Fotokémiai reakciók",
        "AlphaThermodynamics": "Termodinamikai paraméterek",
        "AlphaKinetics": "Reakciókinetika modellezése",
        "AlphaSpectroscopy": "Spektroszkópiai adatelemzés",
        "AlphaChromatography": "Kromatográfiás szeparáció",
        "AlphaAnalytical": "Analitikai kémiai módszerek",
        "AlphaFormulation": "Formuláció és stabilitás",
        "AlphaGreen": "Zöld kémiai alternatívák"
    },
    "kornyezeti_fenntarthato": {
        "AlphaClimate": "Klímaváltozás modellezése",
        "AlphaOcean": "Óceáni rendszerek elemzése",
        "AlphaAtmosphere": "Légköri folyamatok szimulálása",
        "AlphaEcology": "Ökológiai rendszerek modellezése",
        "AlphaWater": "Víz minőség és kezelés",
        "AlphaSoil": "Talaj egészség és termékenység",
        "AlphaRenewable": "Megújuló energia optimalizálása",
        "AlphaCarbon": "Szén-dioxid befogás és tárolás",
        "AlphaWaste": "Hulladékgazdálkodás és újrahasznosítás",
        "AlphaBiodiversity": "Biodiverzitás védelem",
        "AlphaForest": "Erdészeti fenntarthatóság",
        "AlphaAgriculture": "Fenntartható mezőgazdaság",
        "AlphaPollution": "Környezetszennyezés elemzése",
        "AlphaConservation": "Természetvédelem stratégiák",
        "AlphaUrban": "Városi fenntarthatóság",
        "AlphaTransport": "Közlekedési rendszerek",
        "AlphaBuilding": "Épület energetika",
        "AlphaResource": "Erőforrás gazdálkodás",
        "AlphaLifecycle": "Életciklus elemzés",
        "AlphaCircular": "Körforgásos gazdaság",
        "AlphaEnvironmentalHealth": "Környezeti egészségügy",
        "AlphaWildlife": "Vadvilág monitoring",
        "AlphaMarine": "Tengeri ökoszisztémák",
        "AlphaDesertification": "Elsivatagosodás elleni küzdelem",
        "AlphaSustainability": "Fenntarthatósági metrikák"
    },
    "fizikai_asztrofizikai": {
        "AlphaQuantum": "Kvantumfizikai szimulációk",
        "AlphaParticle": "Részecskefizikai elemzések",
        "AlphaGravity": "Gravitációs hullámok elemzése",
        "AlphaCosmic": "Kozmikus sugárzás kutatása",
        "AlphaStellar": "Csillagfejlődés modellezése",
        "AlphaGalaxy": "Galaxisok dinamikája",
        "AlphaExoplanet": "Exobolygó karakterizálás",
        "AlphaPlasma": "Plazma fizika szimulációk",
        "AlphaOptics": "Optikai rendszerek tervezése",
        "AlphaCondensed": "Kondenzált anyag fizika",
        "AlphaSuperconductivity": "Szupravezetés mechanizmusai",
        "AlphaMagnetism": "Mágneses tulajdonságok",
        "AlphaThermodynamics": "Termodinamikai rendszerek",
        "AlphaFluid": "Folyadékdinamika szimulációk",
        "AlphaAcoustics": "Akusztikai jelenségek",
        "AlphaElectromagnetism": "Elektromágneses mezők",
        "AlphaNuclear": "Nukleáris folyamatok",
        "AlphaAtomic": "Atomfizikai számítások",
        "AlphaMolecular": "Molekuláris fizika",
        "AlphaSpectroscopy": "Spektroszkópiai elemzés",
        "AlphaLaser": "Lézer technológiák",
        "AlphaPhotonics": "Fotonika alkalmazások",
        "AlphaCryogenics": "Kriogén rendszerek",
        "AlphaVacuum": "Vákuum technológia",
        "AlphaInstrumentation": "Tudományos műszerek"
    },
    "technologiai_melymu": {
        "AlphaAI": "Mesterséges intelligencia architektúrák",
        "AlphaML": "Gépi tanulás optimalizálás",
        "AlphaNeural": "Neurális hálózatok tervezése",
        "AlphaRobotics": "Robotikai rendszerek",
        "AlphaAutonomy": "Autonóm rendszerek",
        "AlphaVision": "Számítógépes látás",
        "AlphaNLP": "Természetes nyelv feldolgozás",
        "AlphaOptimization": "Optimalizálási algoritmusok",
        "AlphaSimulation": "Szimulációs rendszerek",
        "AlphaModeling": "Matematikai modellezés",
        "AlphaControl": "Irányítástechnika",
        "AlphaSignal": "Jelfeldolgozás",
        "AlphaData": "Adatelemzés és big data",
        "AlphaNetwork": "Hálózati rendszerek",
        "AlphaSecurity": "Kiberbiztonsági elemzés",
        "AlphaCrypto": "Kriptográfiai protokollok",
        "AlphaBlockchain": "Blockchain technológiák",
        "AlphaIoT": "Internet of Things rendszerek",
        "AlphaEdge": "Edge computing optimalizálás",
        "AlphaCloud": "Felhő architektúrák",
        "AlphaHPC": "Nagy teljesítményű számítás",
        "AlphaDrone": "Drón technológiák",
        "AlphaSensor": "Szenzor hálózatok",
        "AlphaEmbedded": "Beágyazott rendszerek",
        "AlphaFPGA": "FPGA programozás"
    },
    "tarsadalmi_gazdasagi": {
        "AlphaEconomy": "Gazdasági modellek és előrejelzések",
        "AlphaMarket": "Piaci trendek elemzése",
        "AlphaFinance": "Pénzügyi kockázatelemzés",
        "AlphaSocial": "Társadalmi hálózatok elemzése",
        "AlphaPolicy": "Szakpolitikai hatáselemzés",
        "AlphaUrbanPlanning": "Városfejlesztés optimalizálása",
        "AlphaLogistics": "Logisztikai láncok",
        "AlphaSupplyChain": "Ellátási láncok optimalizálása",
        "AlphaManufacturing": "Gyártási folyamatok",
        "AlphaQuality": "Minőségbiztosítás",
        "AlphaRisk": "Kockázatelemzés és menedzsment",
        "AlphaDecision": "Döntéstámogató rendszerek",
        "AlphaStrategy": "Stratégiai tervezés",
        "AlphaInnovation": "Innovációs ökoszisztémák",
        "AlphaStartup": "Startup értékelés és mentoring",
        "AlphaEducation": "Oktatási rendszerek",
        "AlphaHealthcare": "Egészségügyi rendszerek",
        "AlphaCustomer": "Vásárlói viselkedés elemzése",
        "AlphaMarketing": "Marketing optimalizálás",
        "AlphaBrand": "Márka értékelés",
        "AlphaHR": "Humán erőforrás menedzsment",
        "AlphaLegal": "Jogi elemzések",
        "AlphaCompliance": "Megfelelőségi rendszerek",
        "AlphaEthics": "Etikai értékelések",
        "AlphaSustainableBusiness": "Fenntartható üzleti modellek"
    }
}

# --- Egyszerű Alpha Service Handler ---
async def handle_simple_alpha_service(service_name: str, query: str, details: str = "") -> Dict[str, Any]:
    """Egyszerű Alpha szolgáltatás kezelő szöveges bemenetnél"""
    
    # Keresés a kategóriákban
    service_category = None
    service_description = None
    
    for category, services in ALPHA_SERVICES.items():
        if service_name in services:
            service_category = category
            service_description = services[service_name]
            break
    
    if not service_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen Alpha szolgáltatás: {service_name}"
        )
    
    # Gemini 2.5 Pro használata elsőként
    selected_model = gemini_25_pro if gemini_25_pro else (gemini_model if gemini_model else cerebras_client)
    
    if not selected_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )
    
    # Prompt összeállítása
    prompt = f"""
    {service_name} Alpha Szolgáltatás Elemzés
    Kategória: {service_category}
    Szolgáltatás leírása: {service_description}
    
    Felhasználó kérése: {query}
    
    További részletek: {details if details else "Nincs további részlet"}
    
    Kérlek, végezz professzionális, tudományos elemzést és adj részletes válaszokat a megadott kérés alapján.
    A válaszod legyen strukturált, magyar nyelvű és gyakorlati szempontokat is tartalmazzon.
    Használd a legfrissebb tudományos információkat és módszereket.
    """
    
    try:
        response_text = ""
        model_used = ""
        
        if selected_model == gemini_25_pro:
            response = await gemini_25_pro.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 2.5 Pro"
            
        elif selected_model == gemini_model:
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"
            
        elif selected_model == cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        
        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": response_text,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a {service_name} szolgáltatás végrehajtása során: {e}"
        )

# --- Általános Alpha Service Handler ---
async def handle_alpha_service(service_name: str, input_data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Univerzális Alpha szolgáltatás kezelő"""
    
    # Keresés a kategóriákban
    service_category = None
    service_description = None
    
    for category, services in ALPHA_SERVICES.items():
        if service_name in services:
            service_category = category
            service_description = services[service_name]
            break
    
    if not service_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen Alpha szolgáltatás: {service_name}"
        )
    
    # Gemini 2.5 Pro használata elsőként
    selected_model = gemini_25_pro if gemini_25_pro else (gemini_model if gemini_model else cerebras_client)
    
    if not selected_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )
    
    # Prompt összeállítása
    prompt = f"""
    {service_name} Alpha Szolgáltatás
    Kategória: {service_category}
    Leírás: {service_description}
    
    Bemeneti adatok:
    {json.dumps(input_data, indent=2, ensure_ascii=False)}
    
    Paraméterek:
    {json.dumps(parameters or {}, indent=2, ensure_ascii=False)}
    
    Kérlek, végezz professzionális, tudományos elemzést és adj részletes válaszokat a megadott adatok alapján.
    A válaszod legyen strukturált, magyar nyelvű és gyakorlati szempontokat is tartalmazzon.
    """
    
    try:
        response_text = ""
        model_used = ""
        
        if selected_model == gemini_25_pro:
            response = await gemini_25_pro.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 2.5 Pro"
            
        elif selected_model == gemini_model:
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"
            
        elif selected_model == cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        
        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": response_text,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a {service_name} szolgáltatás végrehajtása során: {e}"
        )

# --- API Végpontok ---

@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")

@app.get("/api")
async def root_endpoint():
    return {
        "message": "Jade - Deep Discovery AI Platform",
        "version": app.version,
        "creator": DIGITAL_FINGERPRINT,
        "total_services": sum(len(services) for services in ALPHA_SERVICES.values()),
        "categories": list(ALPHA_SERVICES.keys())
    }

@app.get("/api/services")
async def get_services():
    """Minden Alpha szolgáltatás listázása kategóriák szerint"""
    return {
        "categories": ALPHA_SERVICES,
        "total_services": sum(len(services) for services in ALPHA_SERVICES.values())
    }

@app.get("/api/services/{category}")
async def get_services_by_category(category: str):
    """Egy kategória szolgáltatásainak listázása"""
    if category not in ALPHA_SERVICES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen kategória: {category}"
        )
    
    return {
        "category": category,
        "services": ALPHA_SERVICES[category]
    }

@app.post("/api/alpha/{service_name}")
async def execute_alpha_service(service_name: str, request: UniversalAlphaRequest):
    """Bármely Alpha szolgáltatás végrehajtása"""
    return await handle_alpha_service(
        service_name=service_name,
        input_data=request.input_data,
        parameters=request.parameters
    )

@app.post("/api/alpha/simple/{service_name}")
async def execute_simple_alpha_service(service_name: str, request: SimpleAlphaRequest):
    """Egyszerű Alpha szolgáltatás végrehajtása szöveges bemenetnél"""
    return await handle_simple_alpha_service(
        service_name=service_name,
        query=request.query,
        details=request.details
    )

@app.post("/api/deep_discovery/chat")
async def deep_discovery_chat(req: ChatRequest):
    """Chat funkcionalitás"""
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető chat modell"
        )

    user_id = req.user_id
    current_message = req.message
    history = chat_histories.get(user_id, [])

    system_message = {
        "role": "system",
        "content": f"""Te Jade vagy, egy fejlett AI asszisztens, aki magyarul válaszol. 
        Szakértő vagy tudományos és technológiai területeken.
        
        Rendelkezel {sum(len(services) for services in ALPHA_SERVICES.values())} Alpha szolgáltatással:
        - Biológiai/Orvosi: {len(ALPHA_SERVICES['biologiai_orvosi'])} szolgáltatás
        - Kémiai/Anyagtudományi: {len(ALPHA_SERVICES['kemiai_anyagtudomanyi'])} szolgáltatás
        - Környezeti/Fenntartható: {len(ALPHA_SERVICES['kornyezeti_fenntarthato'])} szolgáltatás
        - Fizikai/Asztrofizikai: {len(ALPHA_SERVICES['fizikai_asztrofizikai'])} szolgáltatás
        - Technológiai/Mélyműszaki: {len(ALPHA_SERVICES['technologiai_melymu'])} szolgáltatás
        - Társadalmi/Gazdasági: {len(ALPHA_SERVICES['tarsadalmi_gazdasagi'])} szolgáltatás
        
        Segítőkész, részletes és innovatív válaszokat adsz."""
    }

    messages_for_llm = [system_message] + history + [{"role": "user", "content": current_message}]

    try:
        response_text = ""
        model_used = ""

        # Gemini 2.5 Pro elsőként
        if gemini_25_pro:
            response = await gemini_25_pro.generate_content_async(
                '\n'.join([msg['content'] for msg in messages_for_llm]),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.2
                )
            )
            response_text = response.text
            model_used = "Gemini 2.5 Pro"
        elif cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=messages_for_llm,
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.2
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        elif gemini_model:
            response = await gemini_model.generate_content_async(
                '\n'.join([msg['content'] for msg in messages_for_llm]),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.2
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"

        history.append({"role": "user", "content": current_message})
        history.append({"role": "assistant", "content": response_text})
        chat_histories[user_id] = history

        return {
            'response': response_text,
            'model_used': model_used,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a beszélgetés során: {e}"
        )

@app.post("/api/deep_research")
async def deep_research(req: DeepResearchRequest):
    """Komplex deep research minden modellel"""
    if not exa_client or not gemini_25_pro:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Deep research szolgáltatás nem elérhető"
        )

    try:
        # 1. Exa keresés több típussal
        logger.info(f"Starting deep research for: {req.query}")
        
        # Neurális keresés
        neural_results = []
        try:
            neural_search = exa_client.search(
                query=req.query,
                type="neural",
                num_results=10,
                text_contents={"max_characters": 2000, "strategy": "comprehensive"},
                livecrawl="when_necessary"
            )
            neural_results = neural_search.results
        except Exception as e:
            logger.error(f"Neural search error: {e}")

        # Kulcsszavas keresés
        keyword_results = []
        try:
            keyword_search = exa_client.search(
                query=req.query,
                type="keyword",
                num_results=10,
                text_contents={"max_characters": 2000, "strategy": "comprehensive"}
            )
            keyword_results = keyword_search.results
        except Exception as e:
            logger.error(f"Keyword search error: {e}")

        # Tudományos domainekre fókuszált keresés
        scientific_domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "thelancet.com", "bmj.com", "plos.org",
            "ieee.org", "acm.org", "springer.com", "wiley.com", "biorxiv.org"
        ]
        
        scientific_results = []
        try:
            scientific_search = exa_client.search(
                query=req.query,
                type="neural",
                num_results=15,
                include_domains=scientific_domains,
                text_contents={"max_characters": 3000, "strategy": "comprehensive"},
                livecrawl="when_necessary"
            )
            scientific_results = scientific_search.results
        except Exception as e:
            logger.error(f"Scientific search error: {e}")

        # 2. Összes találat kombinálása
        all_results = neural_results + keyword_results + scientific_results
        
        # Duplikátumok eltávolítása URL alapján
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: getattr(x, 'score', 0), reverse=True)[:20]

        # 3. Tartalom összegyűjtése
        combined_content = ""
        sources = []
        
        for i, result in enumerate(sorted_results):
            if result.text_contents and result.text_contents.text:
                combined_content += f"--- Forrás {i+1}: {result.title} ({result.url}) ---\n{result.text_contents.text}\n\n"
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "published_date": result.published_date,
                    "score": getattr(result, 'score', 0)
                })

        # 4. Minden modell elemzése
        analyses = {}
        
        # Gemini 2.5 Pro elemzés
        if gemini_25_pro:
            try:
                gemini_prompt = f"""
                Készíts rendkívül részletes, tudományos színvonalú elemzést a következő témáról: {req.query}
                
                Felhasznált források:
                {combined_content[:15000]}
                
                Kérlek, adj átfogó, strukturált elemzést a következő szempontok szerint:
                1. Jelenlegi tudományos állás
                2. Főbb kutatási eredmények
                3. Kritikus pontok és viták
                4. Jövőbeli kutatási irányok
                5. Gyakorlati alkalmazások
                
                Használj szakmai terminológiákat, de magyarázd el őket is. Legyél részletes és precíz.
                """
                
                gemini_response = await gemini_25_pro.generate_content_async(
                    gemini_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=4000,
                        temperature=0.1
                    )
                )
                analyses["Gemini 2.5 Pro"] = gemini_response.text
            except Exception as e:
                logger.error(f"Gemini 2.5 Pro analysis error: {e}")

        # Cerebras elemzés
        if cerebras_client:
            try:
                cerebras_prompt = f"""
                Végezz mélyreható technológiai és innovációs elemzést a következő témáról: {req.query}
                
                Források:
                {combined_content[:12000]}
                
                Fókuszálj a technológiai aspektusokra, innovációs lehetőségekre és gyakorlati megvalósításra.
                """
                
                cerebras_response = ""
                stream = cerebras_client.chat.completions.create(
                    messages=[{"role": "user", "content": cerebras_prompt}],
                    model="llama-4-scout-17b-16e-instruct",
                    stream=True,
                    max_completion_tokens=3000,
                    temperature=0.1
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        cerebras_response += chunk.choices[0].delta.content
                
                analyses["Cerebras Llama 4"] = cerebras_response
            except Exception as e:
                logger.error(f"Cerebras analysis error: {e}")

        # Gemini 1.5 Pro kiegészítő elemzés
        if gemini_model:
            try:
                gemini15_prompt = f"""
                Adj kiegészítő perspektívát és összegző elemzést a következő témáról: {req.query}
                
                Már meglévő elemzések kiegészítéseként fókuszálj a:
                - Interdiszciplináris kapcsolatokra
                - Etikai megfontolásokra
                - Társadalmi hatásokra
                - Gazdasági vonatkozásokra
                
                Források: {combined_content[:10000]}
                """
                
                gemini15_response = await gemini_model.generate_content_async(
                    gemini15_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2000,
                        temperature=0.15
                    )
                )
                analyses["Gemini 1.5 Pro"] = gemini15_response.text
            except Exception as e:
                logger.error(f"Gemini 1.5 Pro analysis error: {e}")

        # 5. Végső szintézis Gemini 2.5 Pro-val
        final_synthesis = ""
        if gemini_25_pro and analyses:
            try:
                synthesis_prompt = f"""
                Készíts egy átfogó, tudományos szintézist a következő témáról: {req.query}
                
                Felhasználva a következő különböző AI modellek elemzéseit:
                {json.dumps(analyses, indent=2, ensure_ascii=False)}
                
                Eredeti források: {len(sources)} tudományos és szakmai forrás
                
                Készíts egy komplex, strukturált dokumentumot amely:
                1. Integrálja az összes perspektívát
                2. Kiemeli a konszenzust és az eltéréseket
                3. Azonosítja a legfontosabb megállapításokat
                4. Javasol konkrét lépéseket/következtetéseket
                5. Összegzi a legfontosabb forrásokat
                
                Ez legyen a legteljesebb, legpontosabb és leghasználhatóbb dokumentum a témában.
                """
                
                synthesis_response = await gemini_25_pro.generate_content_async(
                    synthesis_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=6000,
                        temperature=0.05
                    )
                )
                final_synthesis = synthesis_response.text
            except Exception as e:
                logger.error(f"Final synthesis error: {e}")

        return {
            "query": req.query,
            "total_sources": len(sources),
            "sources": sources,
            "individual_analyses": analyses,
            "final_synthesis": final_synthesis,
            "research_depth": "DEEP",
            "models_used": list(analyses.keys()),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in deep research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a mélyreható kutatás során: {e}"
        )

# Meglévő specializált végpontok megőrzése
@app.post("/api/exa/advanced_search")
async def exa_advanced_search(req: AdvancedExaRequest):
    """Fejlett Exa keresés minden paraméterrel"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        # Keresési paraméterek összeállítása
        search_params = {
            "query": req.query,
            "num_results": req.num_results,
            "text_contents": req.text_contents_options,
            "livecrawl": req.livecrawl
        }

        # Típus alapú keresés
        if req.type == "similarity" and hasattr(exa_client, 'find_similar'):
            # Hasonlóság alapú kereséshez URL szükséges
            search_params["type"] = "similarity"
        elif req.type == "keyword":
            search_params["type"] = "keyword"
        else:
            search_params["type"] = "neural"

        # Domain szűrők
        if req.include_domains:
            search_params["include_domains"] = req.include_domains
        if req.exclude_domains:
            search_params["exclude_domains"] = req.exclude_domains

        # Dátum szűrők
        if req.start_crawl_date:
            search_params["start_crawl_date"] = req.start_crawl_date
        if req.end_crawl_date:
            search_params["end_crawl_date"] = req.end_crawl_date
        if req.start_published_date:
            search_params["start_published_date"] = req.start_published_date
        if req.end_published_date:
            search_params["end_published_date"] = req.end_published_date

        # Szöveg szűrők
        if req.include_text:
            search_params["include_text"] = req.include_text
        if req.exclude_text:
            search_params["exclude_text"] = req.exclude_text

        # Kategória szűrők
        if req.category:
            search_params["category"] = req.category
        if req.subcategory:
            search_params["subcategory"] = req.subcategory

        logger.info(f"Advanced Exa search with params: {search_params}")
        response = exa_client.search(**search_params)

        # Eredmények feldolgozása
        results = []
        for result in response.results:
            processed_result = {
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "published_date": result.published_date,
                "author": getattr(result, 'author', None),
                "score": getattr(result, 'score', None),
                "text_content": result.text_contents.text if result.text_contents else None,
                "highlights": getattr(result, 'highlights', None)
            }
            results.append(processed_result)

        return {
            "query": req.query,
            "search_type": req.type,
            "total_results": len(results),
            "results": results,
            "search_params": search_params,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in advanced Exa search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a fejlett Exa keresés során: {e}"
        )

@app.post("/api/exa/find_similar")
async def exa_find_similar(req: ExaSimilarityRequest):
    """Hasonló tartalmak keresése URL alapján"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        params = {
            "url": req.url,
            "num_results": req.num_results,
            "exclude_source_domain": req.exclude_source_domain,
            "text_contents": {
                "max_characters": 2000,
                "strategy": "comprehensive"
            }
        }

        if req.category_weights:
            params["category_weights"] = req.category_weights

        response = exa_client.find_similar(**params)

        results = []
        for result in response.results:
            processed_result = {
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "similarity_score": getattr(result, 'score', None),
                "published_date": result.published_date,
                "text_content": result.text_contents.text if result.text_contents else None
            }
            results.append(processed_result)

        return {
            "reference_url": req.url,
            "similar_results": results,
            "total_found": len(results),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in Exa similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a hasonlóság alapú keresés során: {e}"
        )

@app.post("/api/exa/get_contents")
async def exa_get_contents(req: ExaContentsRequest):
    """Részletes tartalom lekérése Exa result ID alapján"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        params = {
            "ids": req.ids,
            "text_contents": {
                "max_characters": 5000,
                "include_html_tags": True,
                "strategy": "comprehensive"
            }
        }

        if req.highlights:
            params["highlights"] = req.highlights

        response = exa_client.get_contents(**params)

        contents = []
        for content in response.contents:
            processed_content = {
                "id": content.id,
                "url": content.url,
                "title": content.title,
                "text": content.text_contents.text if content.text_contents else None,
                "html": getattr(content.text_contents, 'html', None) if content.text_contents else None,
                "highlights": getattr(content, 'highlights', None),
                "published_date": content.published_date,
                "author": getattr(content, 'author', None)
            }
            contents.append(processed_content)

        # AI összefoglaló generálása ha kért
        summary = ""
        if req.summary and (gemini_model or cerebras_client):
            combined_text = "\n\n".join([c["text"] for c in contents if c["text"]])[:10000]
            
            summary_prompt = f"""
            Készíts részletes összefoglalót a következő tartalmakról:
            
            {combined_text}
            
            Az összefoglaló legyen strukturált és informatív.
            """

            try:
                if gemini_model:
                    response = await gemini_model.generate_content_async(
                        summary_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=1000,
                            temperature=0.1
                        )
                    )
                    summary = response.text
                elif cerebras_client:
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="llama-4-scout-17b-16e-instruct",
                        stream=True,
                        max_completion_tokens=1000,
                        temperature=0.1
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            summary += chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = "Hiba az összefoglaló generálása során"

        return {
            "contents": contents,
            "total_contents": len(contents),
            "ai_summary": summary if req.summary else None,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error getting Exa contents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a tartalmak lekérése során: {e}"
        )

@app.post("/api/exa/neural_search")
async def exa_neural_search(query: str, domains: List[str] = [], exclude_domains: List[str] = [], num_results: int = 20):
    """Speciális neurális keresés tudományos tartalmakhoz"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    # Tudományos domainok alapértelmezetten
    if not domains:
        domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "thelancet.com", "bmj.com", "plos.org",
            "ieee.org", "acm.org", "springer.com", "wiley.com", "elsevier.com"
        ]

    try:
        response = exa_client.search(
            query=query,
            type="neural",
            num_results=num_results,
            include_domains=domains,
            exclude_domains=exclude_domains,
            text_contents={
                "max_characters": 3000,
                "strategy": "comprehensive"
            },
            livecrawl="when_necessary"
        )

        # Eredmények pontszám szerint rendezése
        results = sorted(
            response.results, 
            key=lambda x: getattr(x, 'score', 0), 
            reverse=True
        )

        processed_results = []
        for result in results:
            processed_result = {
                "title": result.title,
                "url": result.url,
                "score": getattr(result, 'score', 0),
                "published_date": result.published_date,
                "domain": result.url.split('/')[2] if '/' in result.url else result.url,
                "text_preview": result.text_contents.text[:500] + "..." if result.text_contents else None
            }
            processed_results.append(processed_result)

        return {
            "query": query,
            "neural_results": processed_results,
            "domains_searched": domains,
            "total_results": len(processed_results),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in neural search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a neurális keresés során: {e}"
        )

@app.post("/api/deep_discovery/research_trends")
async def get_research_trends(req: ScientificInsightRequest):
    if not exa_client or not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI vagy Gemini nem elérhető"
        )

    try:
        # Fejlett Exa keresés tudományos domainekkel
        scientific_domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "thelancet.com", "bmj.com", "plos.org",
            "ieee.org", "acm.org", "springer.com", "wiley.com", "biorxiv.org"
        ]
        
        search_response = exa_client.search(
            query=req.query,
            type="neural",
            num_results=req.num_results,
            include_domains=scientific_domains,
            text_contents={
                "max_characters": 2000, 
                "strategy": "comprehensive",
                "include_html_tags": False
            },
            livecrawl="when_necessary",
            start_published_date="2020-01-01"  # Friss kutatások
        )

        if not search_response or not search_response.results:
            return {
                "query": req.query,
                "summary": "Nem található releváns információ",
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

        summary_prompt = f"""
        Elemezd a következő tudományos információkat és készíts összefoglalót (max. {req.summary_length} szó):

        {combined_content[:8000]}

        Összefoglalás:
        """

        response = await gemini_model.generate_content_async(
            summary_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=req.summary_length * 2,
                temperature=0.1
            )
        )

        return {
            "query": req.query,
            "summary": response.text,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in research trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a kutatási trendek elemzése során: {e}"
        )

@app.post("/api/deep_discovery/protein_structure")
async def protein_structure_lookup(req: ProteinLookupRequest):
    ebi_alphafold_api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{req.protein_id}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(ebi_alphafold_api_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or (isinstance(data, list) and not data):
                return {
                    "protein_id": req.protein_id,
                    "message": "Nem található előrejelzés",
                    "details": None
                }

            first_prediction = data[0] if isinstance(data, list) else data

            return {
                "protein_id": req.protein_id,
                "message": "Sikeres lekérdezés",
                "details": {
                    "model_id": first_prediction.get("model_id"),
                    "uniprot_id": first_prediction.get("uniprot_id"),
                    "plddt": first_prediction.get("plddt"),
                    "protein_url": first_prediction.get("cif_url") or first_prediction.get("pdb_url"),
                    "pae_url": first_prediction.get("pae_url"),
                    "assembly_id": first_prediction.get("assembly_id")
                }
            }

    except Exception as e:
        logger.error(f"Error in protein lookup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a fehérje lekérdezése során: {e}"
        )

@app.post("/api/deep_discovery/custom_gcp_model")
async def custom_gcp_model_inference(req: CustomGCPModelRequest):
    if not gcp_credentials:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP Vertex AI nem elérhető"
        )

    try:
        project = req.gcp_project_id or GCP_PROJECT_ID
        region = req.gcp_region or GCP_REGION
        endpoint_name = f"projects/{project}/locations/{region}/endpoints/{req.gcp_endpoint_id}"

        prediction_client = aiplatform.gapic.PredictionServiceClient(credentials=gcp_credentials)
        instances_list = [req.input_data] if not isinstance(req.input_data, list) else req.input_data

        response = prediction_client.predict(
            endpoint=endpoint_name,
            instances=instances_list
        )

        return {
            "model_response": response.predictions,
            "model_id": req.gcp_endpoint_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in GCP model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a GCP modell hívásakor: {e}"
        )

@app.post("/api/deep_discovery/simulation_optimizer")
async def simulation_optimizer(req: SimulationOptimizerRequest):
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető modell"
        )

    prompt = f"""
    {req.simulation_type} szimuláció optimalizálása:
    Paraméterek: {json.dumps(req.input_parameters, indent=2)}
    Cél: {req.optimization_goal}
    
    Generálj optimalizált paramétereket vagy Python kódot.
    """

    try:
        response_text = ""
        model_used = ""

        if cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=1024,
                temperature=0.3
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        elif gemini_model:
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.3
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"

        return {
            "simulation_type": req.simulation_type,
            "optimization_goal": req.optimization_goal,
            "optimized_output": response_text,
            "model_used": model_used,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in simulation optimizer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a szimuláció optimalizálásában: {e}"
        )

@app.post("/api/deep_discovery/alphagenome")
async def alphagenome_analysis(req: AlphaGenomeRequest):
    if not gemini_model and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    valid_nucleotides = set('ATCGURYN-')
    sequence_upper = req.genome_sequence.upper()
    if not all(char in valid_nucleotides for char in sequence_upper):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Érvénytelen genom szekvencia"
        )

    try:
        sequence_length = len(req.genome_sequence)
        gc_content = (sequence_upper.count('G') + sequence_upper.count('C')) / sequence_length * 100

        analysis_prompt = f"""
        AlphaGenome Genomikai Elemzés:
        Szekvencia: {req.genome_sequence[:500]}{'...' if len(req.genome_sequence) > 500 else ''}
        Szervezet: {req.organism}
        Elemzés: {req.analysis_type}
        Hossz: {sequence_length} bp
        GC: {gc_content:.1f}%
        
        Végezz részletes genomikai elemzést magyar nyelven.
        """

        response_text = ""
        model_used = ""

        if gemini_model:
            response = await gemini_model.generate_content_async(
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"
        elif cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"

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
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in AlphaGenome: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba az AlphaGenome elemzésben: {e}"
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
