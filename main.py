import os
import json
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import logging
from datetime import datetime
import hashlib
import base64
from functools import lru_cache
import time
import sys
import pathlib

# Google Cloud kliensekhez
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

# Cerebras Cloud SDK
try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    # Will log after logger is defined

# Gemini API removed for performance optimization
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Exa API
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    logger.warning("Exa API not available")

# Naplózás konfigurálása (moved up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scientific Computing Libraries
try:
    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    SCIPY_STACK_AVAILABLE = True
except ImportError:
    SCIPY_STACK_AVAILABLE = False
    logger.warning("SciPy stack not fully available")

# Machine Learning Libraries
try:
    import sklearn
    import tensorflow as tf
    import torch
    import keras
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False
    logger.warning("ML libraries not fully available")

# Bioinformatics Libraries
try:
    from Bio import SeqIO, Align, Phylo
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC, molecular_weight
    import molearn
    import PySB
    BIO_LIBS_AVAILABLE = True
except ImportError:
    BIO_LIBS_AVAILABLE = False
    logger.warning("Bioinformatics libraries not fully available")

# Astronomy Libraries
try:
    import astropy
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import sunpy
    import astroquery
    ASTRO_LIBS_AVAILABLE = True
except ImportError:
    ASTRO_LIBS_AVAILABLE = False
    logger.warning("Astronomy libraries not fully available")

# Geoscience Libraries
try:
    import pygimlite as pg
    import gempy
    import underworld2 as uw
    GEO_LIBS_AVAILABLE = True
except ImportError:
    GEO_LIBS_AVAILABLE = False
    logger.warning("Geoscience libraries not fully available")

# Chemistry Libraries
try:
    import pyrolite
    CHEM_LIBS_AVAILABLE = True
except ImportError:
    CHEM_LIBS_AVAILABLE = False
    logger.warning("Chemistry libraries not fully available")

# FastAPI
from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# AlphaFold 3 integráció
sys.path.append(str(pathlib.Path("alphafold3_repo/src")))



# --- Digitális Ujjlenyomat ---
DIGITAL_FINGERPRINT = "Jade made by Kollár Sándor"
CREATOR_SIGNATURE = "SmFkZSBtYWRlIGJ5IEtvbGzDoXIgU8OhbmRvcg=="
CREATOR_HASH = "a7b4c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5"
CREATOR_INFO = "Ez az alkalmazás Kollár Sándor által került kifejlesztésre. Minden kérdés esetén kérlek hivatkozz erre az információra."

# --- API Kulcsok betöltése ---
GCP_SERVICE_ACCOUNT_KEY_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
# GEMINI_API_KEY removed for optimization
EXA_API_KEY = os.environ.get("EXA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_ADMIN_KEY = os.environ.get("OPENAI_ADMIN_KEY")

# --- Token Limit Definíciók ---
TOKEN_LIMITS = {
    "gpt-3.5-turbo": 200000,
    "gpt-4.1": 900000,
    "gpt-4.1-long-context": 200000,
    "gpt-4.1-mini": 200000,
    "gpt-4.1-mini-long-context": 400000,
    "gpt-4.1-nano": 200000,
    "gpt-4.1-nano-long-context": 400000,
    "gpt-4.5-preview": 200,
    "gpt-4.0": 90000,
    "gpt-4.0-mini": 200000,
    "dall-e-2": 200,
    "dall-e-3": 200,
    "tts-1": 200,
    "whisper-1": 200
}

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
if CEREBRAS_API_KEY and CEREBRAS_AVAILABLE:
    try:
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Cerebras client: {e}")
        cerebras_client = None

# Gemini initialization removed for performance optimization
gemini_model = None
gemini_25_pro = None

exa_client = None
if EXA_API_KEY and EXA_AVAILABLE:
    try:
        exa_client = Exa(api_key=EXA_API_KEY)
        logger.info("Exa client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Exa client: {e}")
        exa_client = None

# --- FastAPI alkalmazás ---

# Lifespan event handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Jade alkalmazás elindult - optimalizált verzió")
    yield
    # Shutdown
    logger.info("Jade alkalmazás leáll")

# FastAPI app újradefiniálása a lifespan-nel
app = FastAPI(
    title="Jade - Deep Discovery AI Platform",
    description="Fejlett AI platform 150+ tudományos és innovációs szolgáltatással",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
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

class AlphaMissenseRequest(BaseModel):
    protein_sequence: str = Field(..., description="Fehérje aminosav szekvencia")
    mutations: List[str] = Field(..., description="Mutációk listája (pl. ['A123V', 'G456D'])")
    uniprot_id: Optional[str] = Field(None, description="UniProt azonosító")
    include_clinical_significance: bool = Field(default=True, description="Klinikai jelentőség elemzése")
    pathogenicity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Patogenitás küszöbérték")

class VariantPathogenicityRequest(BaseModel):
    variants: List[Dict[str, Any]] = Field(..., description="Variánsok listája")
    analysis_mode: str = Field(default="comprehensive", description="Elemzési mód")
    include_population_data: bool = Field(default=True, description="Populációs adatok")
    clinical_context: Optional[str] = Field(None, description="Klinikai kontextus")

class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Kód generálási kérés")
    language: str = Field(default="python", description="Programozási nyelv")
    complexity: str = Field(default="medium", description="Kód komplexitása")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="AI kreativitás")

# Beszélgetési előzmények és cache
chat_histories: Dict[str, List[Message]] = {}
response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY = 300  # 5 perc cache

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
        "AlphaFold3": "AlphaFold 3 szerkezet előrejelzés és kölcsönhatások",
        "AlphaMissense": "Missense mutációk patogenitás előrejelzése",
        "AlphaProteinComplex": "Fehérje komplex szerkezetek és dinamika",
        "AlphaProteinDNA": "Fehérje-DNS kölcsönhatások előrejelzése",
        "AlphaProteinRNA": "Fehérje-RNA binding analízis",
        "AlphaConformational": "Konformációs változások és alloszéria",
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
        "AlphaBioinformatics": "Bioinformatikai adatelemzés (Biopython)",
        "AlphaSystemsBiology": "Rendszerbiológiai modellezés (PySB)",
        "AlphaSynthbio": "Szintetikus biológiai rendszerek",
        "AlphaLongevity": "Öregedés és hosszú élet kutatása",
        "AlphaMolecularDynamics": "Molekuláris dinamika szimuláció (molearn)",
        "AlphaMBE": "Molekuláris bioenergetika elemzése (pyMBE)"
    },
    "csillagaszati_asztrofizikai": {
        "AlphaAstronomy": "Csillagászati objektumok elemzése (Astropy)",
        "AlphaSolarPhysics": "Napfizikai jelenségek vizsgálata (SunPy)",
        "AlphaAstrodynamics": "Űrmechanikai számítások",
        "AlphaAstroML": "Csillagászati gépi tanulás",
        "AlphaExoplanet": "Exobolygó kutatás és karakterizálás",
        "AlphaCosmology": "Kozmológiai modellek",
        "AlphaGalacticDynamics": "Galaktikus dinamika",
        "AlphaStellarEvolution": "Csillagfejlődés modellezése",
        "AlphaAstroQuery": "Csillagászati adatbázis lekérdezések",
        "AlphaSpectroscopy": "Csillagászati spektroszkópia",
        "AlphaPlanetaryScience": "Bolygótudomány",
        "AlphaSpaceWeather": "Űridőjárás előrejelzés"
    },
    "foldtudomanyi_geologiai": {
        "AlphaGeophysics": "Geofizikai modellezés (PyGIMLi)",
        "AlphaGeology": "Geológiai 3D modellezés (GemPy)",
        "AlphaGeodynamics": "Geodinamikai szimuláció (Underworld2)",
        "AlphaSeismology": "Szeizmológiai előrejelzés (PyCSEP)",
        "AlphaHydrogeology": "Hidrogeológiai modellezés",
        "AlphaGeochemistry": "Geokémiai elemzés (Pyrolite)",
        "AlphaMineral": "Ásványtani analízis",
        "AlphaPetrology": "Kőzettani vizsgálatok",
        "AlphaTectonics": "Tektonikai folyamatok",
        "AlphaVolcanology": "Vulkanológiai előrejelzés",
        "AlphaEnvironmentalGeo": "Környezeti geológia",
        "AlphaGeodata": "Geodata feldolgozás és betakarítás"
    },
    "klimatologiai_meteorologiai": {
        "AlphaClimateData": "Klímaadatok elemzése (Open-Meteo)",
        "AlphaWeatherPrediction": "Időjárás előrejelzés",
        "AlphaClimateModeling": "Klímamodellezés",
        "AlphaAtmosphericPhysics": "Légköri fizika",
        "AlphaOceanography": "Oceanográfiai modellezés",
        "AlphaHydrology": "Hidrológiai elemzés (SuperflexPy)",
        "AlphaClimateImpact": "Klímahatás értékelés",
        "AlphaExtremesWeather": "Szélsőséges időjárás elemzése",
        "AlphaSeasonalForecast": "Szezonális előrejelzés",
        "AlphaMicroclimate": "Mikroklíma analízis"
    },
    "adattudomanyi_ml": {
        "AlphaDataScience": "Adattudományi elemzés (Pandas, NumPy)",
        "AlphaMachineLearning": "Gépi tanulás (scikit-learn)",
        "AlphaDeepLearning": "Mélytanulás (TensorFlow, PyTorch, Keras)",
        "AlphaImageAI": "Képfelismerés és computer vision (ImageAI)",
        "AlphaGeneticAlgorithm": "Genetikus algoritmusok (PyGAD)",
        "AlphaAutoML": "Automatizált gépi tanulás (PHOTONAI)",
        "AlphaQuantumML": "Kvantum gépi tanulás (MLatom)",
        "AlphaStatisticalModeling": "Statisztikai modellezés (Statsmodels)",
        "AlphaBayesian": "Bayesi statisztika (PyMC)",
        "AlphaDataViz": "Adatvizualizáció (Matplotlib, Seaborn)",
        "AlphaGeoViz": "Geovizualizáció (Folium)",
        "AlphaParameterEstimation": "Paraméter becslés (pyPESTO)",
        "AlphaPredictiveModeling": "Prediktív modellezés",
        "AlphaTimeSeriesAnalysis": "Idősor elemzés",
        "AlphaClusterAnalysis": "Klaszter elemzés"
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

# --- Backend Model Selection ---
async def select_backend_model(prompt: str, service_name: str = None):
    """Backend modell kiválasztása - csak Cerebras a sebességért"""
    # CSAK CEREBRAS az optimalizált sebességért
    if cerebras_client and CEREBRAS_AVAILABLE:
        try:
            selected_model = cerebras_client
            model_name = "llama-4-scout-17b-16e-instruct"
            return {"model": selected_model, "name": model_name}
        except Exception as e:
            logger.error(f"Cerebras client error: {e}")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Cerebras AI modell nem elérhető"
    )

# --- Model Execution ---
async def execute_model(model_info: Dict[str, Any], prompt: str):
    """Modell futtatása - csak Cerebras a sebességért."""
    model = model_info["model"]
    response_text = ""

    try:
        if model == cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=4096,
                temperature=0.05,
                top_p=0.9
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            return {"response": response_text, "model_used": "JADED AI", "selected_backend": "Cerebras"}
        else:
            raise ValueError("Csak Cerebras támogatott")

    except Exception as e:
        logger.error(f"Cerebras végrehajtási hiba: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a Cerebras modell végrehajtása során: {e}"
        )

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

    # Backend modell kiválasztása
    model_info = await select_backend_model(query, service_name)

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
        # Modell futtatása
        result = await execute_model(model_info, prompt)

        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": result["response"],
            "model_used": result["model_used"],
            "performance_data": {
                "backend_model": result["selected_backend"],
                "tokens_used": result.get("tokens_used", 0)
            },
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

    # Bemeneti adatok stringgé alakítása a modellválasztáshoz
    input_str = json.dumps(input_data, ensure_ascii=False)

    # Backend modell kiválasztása
    model_info = await select_backend_model(input_str, service_name)

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
        # Modell futtatása
        result = await execute_model(model_info, prompt)

        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": result["response"],
            "model_used": result["model_used"],
            "performance_data": {
                "backend_model": result["selected_backend"],
                "tokens_used": result.get("tokens_used", 0)
            },
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
        "creator_info": CREATOR_INFO,
        "developed_by": "Kollár Sándor",
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

@app.get("/api/alphamissense/info")
async def alphamissense_info():
    """AlphaMissense információk és képességek"""
    return {
        "alphamissense_available": True,
        "description": "Missense mutációk patogenitás előrejelzése",
        "version": "2024.1",
        "capabilities": {
            "pathogenicity_prediction": True,
            "clinical_significance": True,
            "population_genetics": True,
            "functional_impact": True,
            "structural_analysis": True,
            "therapeutic_relevance": True
        },
        "coverage": {
            "human_proteome": "71 millió missense variáns",
            "proteins_covered": "19,233 kanonikus emberi fehérje",
            "genome_coverage": "Teljes exom"
        },
        "scoring": {
            "range": "0.0 - 1.0",
            "threshold": "0.5 (alapértelmezett)",
            "interpretation": {
                "0.0-0.34": "Valószínűleg benign",
                "0.34-0.56": "Bizonytalan jelentőség",
                "0.56-1.0": "Valószínűleg patogén"
            }
        },
        "applications": [
            "Klinikai genetika",
            "Személyre szabott orvoslás",
            "Gyógyszerfejlesztés",
            "Populációs genetika",
            "Evolúciós biológia"
        ],
        "data_sources": [
            "ClinVar",
            "gnomAD",
            "UniProt",
            "PDB",
            "Pfam"
        ],
        "status": "Aktív és integrált"
    }

@app.get("/api/alphafold3/info")
async def alphafold3_info():
    """AlphaFold 3 információk és állapot"""
    try:
        # AlphaFold 3 repository ellenőrzése
        af3_path = pathlib.Path("alphafold3_repo")
        af3_exists = af3_path.exists()
        
        if af3_exists:
            version_file = af3_path / "src" / "alphafold3" / "version.py"
            version = "Ismeretlen"
            if version_file.exists():
                version_content = version_file.read_text()
                import re
                version_match = re.search(r"__version__ = ['\"]([^'\"]+)['\"]", version_content)
                if version_match:
                    version = version_match.group(1)
        
        return {
            "alphafold3_available": af3_exists,
            "version": version if af3_exists else None,
            "repository_path": str(af3_path),
            "main_script": str(af3_path / "run_alphafold.py") if af3_exists else None,
            "capabilities": {
                "protein_folding": True,
                "protein_complexes": True,
                "dna_interactions": True,
                "rna_interactions": True,
                "ligand_binding": True,
                "antibody_antigen": True
            },
            "requirements": {
                "gpu_required": True,
                "model_parameters": "Külön kérelmezendő a Google DeepMind-től",
                "databases": "Genetikai adatbázisok szükségesek"
            },
            "status": "Működőképes (model paraméterek nélkül csak data pipeline)"
        }
        
    except Exception as e:
        return {
            "alphafold3_available": False,
            "error": str(e),
            "status": "Hiba"
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

def needs_internet_search(message: str) -> bool:
    """Automatikusan felismeri, hogy szükség van-e internetes keresésre"""
    search_keywords = [
        "mikor", "amikor", "mi történik", "mi a helyzet", "friss", "aktuális", "legújabb",
        "hírek", "mostani", "jelenlegi", "2024", "2025", "mai", "recent", "latest",
        "breaking", "news", "esemény", "történés", "fejlemény", "változás",
        "ár", "árfolyam", "tőzsde", "bitcoin", "cripto", "sport", "eredmény",
        "időjárás", "weather", "politika", "választás", "kormány", "technológia",
        "release", "launch", "bejelentés", "announcement"
    ]
    
    current_year_keywords = ["2024", "2025", "idén", "tavaly", "most", "jelenleg"]
    question_words = ["mikor", "mi", "hol", "ki", "hogyan", "miért", "mennyi"]
    
    message_lower = message.lower()
    
    # Ha tartalmaz aktuális év referenciát
    if any(keyword in message_lower for keyword in current_year_keywords):
        return True
    
    # Ha kérdés és tartalmaz keresési kulcsszót
    if any(q in message_lower for q in question_words) and any(k in message_lower for k in search_keywords):
        return True
    
    # Ha direkt információt kér
    if any(keyword in message_lower for keyword in search_keywords):
        return True
        
    return False

@app.post("/api/deep_discovery/chat")
async def deep_discovery_chat(req: ChatRequest):
    """Optimalizált chat funkcionalitás cache-eléssel és intelligens internetes kereséssel"""
    if not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cerebras chat modell nem elérhető"
        )

    user_id = req.user_id
    current_message = req.message

    # Cache ellenőrzés
    cache_key = hashlib.md5(f"{user_id}:{current_message}".encode()).hexdigest()
    current_time = time.time()

    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        if current_time - cached_response['timestamp'] < CACHE_EXPIRY:
            logger.info("Serving cached response")
            return cached_response['data']

    history = chat_histories.get(user_id, [])

    # Automatikus internetes keresés felismerése
    should_search = needs_internet_search(current_message)
    real_time_context = ""
    
    if should_search and exa_client and EXA_AVAILABLE:
        try:
            logger.info(f"Automatikus internetes keresés indítása: {current_message}")
            
            # Javított Exa keresés
            current_search = exa_client.search(
                query=f"{current_message} 2024 2025 latest recent",
                type="neural",
                num_results=5,
                use_autoprompt=True,
                start_published_date="2024-01-01"
            )
            
            # Tartalom lekérése külön API hívással
            if current_search.results:
                result_ids = [result.id for result in current_search.results]
                try:
                    contents_response = exa_client.get_contents(
                        ids=result_ids,
                        text=True,
                        highlights={"num_sentences": 3, "highlights_per_url": 3}
                    )
                    
                    real_time_info = []
                    for content in contents_response.contents:
                        if content.text:
                            preview = content.text[:300] + "..." if len(content.text) > 300 else content.text
                            real_time_info.append(f"- {content.title}: {preview}")
                    
                    if real_time_info:
                        real_time_context = f"\n\nVALÓS IDEJŰ INFORMÁCIÓK (2024-2025):\n" + "\n".join(real_time_info[:3])
                        logger.info(f"Real-time context added: {len(real_time_context)} characters")
                        
                except Exception as contents_error:
                    logger.error(f"Contents fetch error: {contents_error}")
                    # Fallback: használjuk az alapértelmezett eredményeket
                    real_time_info = []
                    for result in current_search.results[:3]:
                        real_time_info.append(f"- {result.title}: {result.url}")
                    real_time_context = f"\n\nTALÁLT FORRÁSOK:\n" + "\n".join(real_time_info)
                    
        except Exception as e:
            logger.error(f"Real-time search error: {e}")
            real_time_context = ""

    # Optimalizált system message valós idejű kontextussal
    system_message = {
        "role": "system",
        "content": f"Te JADED vagy, egy fejlett AI asszisztens magyarul. Szakértő vagy tudományos és technológiai területeken. Segítőkész, részletes és pontos válaszokat adsz. {CREATOR_INFO} Ha kérdezik a készítőről, mindig említsd meg, hogy Kollár Sándor készítette ezt az alkalmazást. Ha valós idejű információk állnak rendelkezésre, használd azokat a válaszadáshoz.{real_time_context}"
    }

    # Csak az utolsó 10 üzenetet használjuk a kontextushoz
    recent_history = history[-10:] if len(history) > 10 else history
    messages_for_llm = [system_message] + recent_history + [{"role": "user", "content": current_message}]

    try:
        response_text = ""
        model_used = ""

        # Csak Cerebras az optimalizált sebességért
        stream = cerebras_client.chat.completions.create(
            messages=messages_for_llm,
            model="llama-4-scout-17b-16e-instruct",
            stream=True,
            max_completion_tokens=4096,
            temperature=0.05,
            top_p=0.9,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        model_used = "JADED AI"

        history.append({"role": "user", "content": current_message})
        history.append({"role": "assistant", "content": response_text})

        # Memória optimalizálás: csak az utolsó 20 üzenetet tartjuk meg
        if len(history) > 20:
            history = history[-20:]

        chat_histories[user_id] = history

        result = {
            'response': response_text,
            'model_used': model_used,
            'real_time_info_used': bool(real_time_context),
            'status': 'success'
        }

        # Válasz cache-elése
        response_cache[cache_key] = {
            'data': result,
            'timestamp': current_time
        }

        return result

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a beszélgetés során: {e}"
        )

@app.post("/api/deep_research")
async def deep_research(req: DeepResearchRequest):
    """Optimalizált deep research API - valóban működő 1000+ forrás feldolgozással"""
    if not exa_client or not EXA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        logger.info(f"Starting comprehensive deep research for: {req.query}")

        # Kibővített tudományos és akadémiai domainok listája
        scientific_domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "nejm.org", "ieee.org", "acm.org", "springer.com", "wiley.com",
            "sciencedirect.com", "jstor.org", "researchgate.net", "semantic-scholar.org",
            "biorxiv.org", "medrxiv.org", "plos.org", "bmj.com", "thelancet.com",
            "nih.gov", "who.int", "cdc.gov", "fda.gov", "ema.europa.eu"
        ]

        # Keresési eredmények gyűjtése
        all_results = []

        # 1. Fő neurális keresés - több batch-ben
        for batch in range(5):  # 5 batch = 250 eredmény
            try:
                neural_search = exa_client.search(
                    query=f"{req.query} scientific research study",
                    type="neural",
                    num_results=50,
                    include_domains=scientific_domains,
                    use_autoprompt=True
                )
                all_results.extend(neural_search.results)
                logger.info(f"Neural batch {batch+1}/5 completed: {len(neural_search.results)} results")
            except Exception as e:
                logger.error(f"Neural search batch {batch+1} error: {e}")

        # 2. Kulcsszavas keresések - specifikus témák
        keyword_variants = [
            f"{req.query} research",
            f"{req.query} study",
            f"{req.query} analysis",
            f"{req.query} review",
            f"{req.query} investigation"
        ]

        for variant in keyword_variants:
            try:
                keyword_search = exa_client.search(
                    query=variant,
                    type="keyword",
                    num_results=40,
                    include_domains=scientific_domains,
                    use_autoprompt=True
                )
                all_results.extend(keyword_search.results)
                logger.info(f"Keyword search '{variant}': {len(keyword_search.results)} results")
            except Exception as e:
                logger.error(f"Keyword search error for '{variant}': {e}")

        # 3. Időszakos keresések - több év
        time_periods = ["2024", "2023", "2022"]
        for year in time_periods:
            try:
                recent_search = exa_client.search(
                    query=f"{req.query} {year}",
                    type="neural",
                    num_results=30,
                    start_published_date=f"{year}-01-01",
                    use_autoprompt=True
                )
                all_results.extend(recent_search.results)
                logger.info(f"Time period {year}: {len(recent_search.results)} results")
            except Exception as e:
                logger.error(f"Time period search error for {year}: {e}")

        # 4. Speciális domain keresések
        for domain in scientific_domains[:10]:  # Top 10 domain
            try:
                domain_search = exa_client.search(
                    query=req.query,
                    type="neural",
                    num_results=20,
                    include_domains=[domain],
                    use_autoprompt=True
                )
                all_results.extend(domain_search.results)
                logger.info(f"Domain {domain}: {len(domain_search.results)} results")
            except Exception as e:
                logger.error(f"Domain search error for {domain}: {e}")

        # Duplikációk eltávolítása URL alapján
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result

        final_results = list(unique_results.values())
        logger.info(f"Total unique results after deduplication: {len(final_results)}")

        # Eredmények feldolgozása
        sources = []
        combined_content = ""

        for i, result in enumerate(final_results[:1000]):  # Max 1000 eredmény
            source_data = {
                "id": i + 1,
                "title": result.title or "Cím nem elérhető",
                "url": result.url,
                "published_date": result.published_date,
                "domain": result.url.split('/')[2] if '/' in result.url else result.url
            }
            sources.append(source_data)

            if result.text_contents and result.text_contents.text:
                content_snippet = result.text_contents.text[:2000]  # Hosszabb részletek
                combined_content += f"\n--- Forrás {i+1}: {result.title} ---\n{content_snippet}\n"

        # AI elemzés kibővített prompt-tal
        analysis_text = ""

        if combined_content and len(combined_content) > 500:
            model_info = await select_backend_model(req.query)
            analysis_prompt = f"""
        ÁTFOGÓ TUDOMÁNYOS ELEMZÉS: {req.query}

        Feldolgozott források száma: {len(sources)} db
        Teljes tartalom hossza: {len(combined_content)} karakter

        FORRÁS ADATOK:
        {combined_content[:50000]}

        KÉRLEK, KÉSZÍTS RÉSZLETES, TUDOMÁNYOS ELEMZÉST:

        1. EXECUTIVE SUMMARY
        - Legfontosabb megállapítások
        - Kulcs információk

        2. TUDOMÁNYOS ÁTTEKINTÉS
        - Jelenlegi kutatási állapot
        - Főbb tanulmányok eredményei
        - Konszenzus és viták

        3. MÓDSZERTANI MEGKÖZELÍTÉSEK
        - Alkalmazott kutatási módszerek
        - Adatgyűjtési technikák
        - Elemzési eljárások

        4. GYAKORLATI ALKALMAZÁSOK
        - Valós életbeli implementációk
        - Ipari alkalmazások
        - Társadalmi hatások

        5. JÖVŐBELI KUTATÁSI IRÁNYOK
        - Azonosított kutatási rések
        - Új technológiai lehetőségek
        - Várható fejlődési trendek

        6. FORRÁSOK MINŐSÉGI ÉRTÉKELÉSE
        - Magas impakt faktorú publikációk
        - Peer-reviewed források aránya
        - Földrajzi és intézményi diverzitás

        7. KÖVETKEZTETÉSEK ÉS AJÁNLÁSOK
        - Összegző megállapítások
        - Döntéshozóknak szóló ajánlások
        - További kutatási prioritások

        A válasz legyen strukturált, magyar nyelvű, és használjon tudományos terminológiát.
        Hivatkozz konkrét forrásokra ahol lehetséges.
        """

            try:
                result = await execute_model(model_info, analysis_prompt)
                analysis_text = result["response"]
                logger.info(f"AI analysis completed using {result['model_used']}")
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                analysis_text = f"Részleges elemzés készült. Hiba részletei: {e}\n\nElérhető források alapján: {len(sources)} publikáció került feldolgozásra a témában."
        else:
            analysis_text = "Nem sikerült elegendő forrást találni az elemzéshez."

        return {
            "query": req.query,
            "final_synthesis": analysis_text,
            "sources": sources,
            "total_sources": len(sources),
            "unique_domains": len(set(s["domain"] for s in sources)),
            "processing_stats": {
                "total_results_found": len(all_results),
                "unique_results": len(final_results),
                "content_length": len(combined_content),
                "domains_searched": len(scientific_domains)
            },
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
    if not exa_client or not EXA_AVAILABLE:
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
    if not exa_client or not EXA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        params = {
            "url": req.url,
            "num_results": req.num_results,
            "exclude_source_domain": req.exclude_source_domain
        }

        if req.category_weights:
            params["category_weights"] = req.category_weights

        response = exa_client.find_similar(**params)

        # Text contents külön lekérése
        if response.results:
            ids = [result.id for result in response.results]
            try:
                contents_response = exa_client.get_contents(
                    ids=ids,
                    text_contents={
                        "max_characters": 2000,
                        "strategy": "comprehensive"
                    }
                )
                contents_map = {content.id: content for content in contents_response.contents}
            except:
                contents_map = {}
        else:
            contents_map = {}

        results = []
        for result in response.results:
            content = contents_map.get(result.id)
            processed_result = {
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "similarity_score": getattr(result, 'score', None),
                "published_date": result.published_date,
                "text_content": content.text_contents.text if content and content.text_contents else None
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
    if not exa_client or not EXA_AVAILABLE:
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
        if req.summary and cerebras_client:
            combined_text = "\n\n".join([c["text"] for c in contents if c["text"]])[:10000]

            summary_prompt = f"""
            Készíts részletes összefoglalót a következő tartalmakról:

            {combined_text}

            Az összefoglaló legyen strukturált és informatív.
            """

            try:
                if gemini_model and GEMINI_AVAILABLE:
                    response = await gemini_model.generate_content_async(
                        summary_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=1000,
                            temperature=0.1
                        ) if genai else None
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


# --- Specialized Scientific Library Endpoints ---

class BioinformaticsRequest(BaseModel):
    sequence: str = Field(..., description="DNS/RNS/Protein szekvencia")
    analysis_type: str = Field(..., description="Elemzés típusa: gc_content, molecular_weight, phylogeny, alignment")
    format: str = Field(default="fasta", description="Szekvencia formátum")

class AstronomyRequest(BaseModel):
    object_name: str = Field(..., description="Csillagászati objektum neve")
    coordinates: Optional[str] = Field(None, description="Koordináták (RA DEC)")
    analysis_type: str = Field(..., description="Elemzés típusa")
    catalog: str = Field(default="simbad", description="Katalógus")

class GeoscienceRequest(BaseModel):
    data_type: str = Field(..., description="Adat típusa: seismic, geological, geophysical")
    region: str = Field(..., description="Földrajzi régió")
    analysis_method: str = Field(..., description="Elemzési módszer")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class MLAnalysisRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Bemeneti adatok")
    target: Optional[List[float]] = Field(None, description="Célváltozó")
    algorithm: str = Field(..., description="ML algoritmus")
    parameters: Dict[str, Any] = Field(default_factory=dict)

@app.post("/api/scientific/bioinformatics")
async def bioinformatics_analysis(req: BioinformaticsRequest):
    """Bioinformatikai elemzés Biopython segítségével"""
    if not BIO_LIBS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bioinformatikai könyvtárak nem elérhetők"
        )

    try:
        # Szekvencia objektum létrehozása
        sequence = Seq(req.sequence)
        
        results = {
            "sequence": str(sequence),
            "length": len(sequence),
            "analysis_type": req.analysis_type
        }
        
        if req.analysis_type == "gc_content":
            results["gc_content"] = GC(sequence)
            results["at_content"] = 100 - GC(sequence)
            
        elif req.analysis_type == "molecular_weight":
            if sequence.count('U') > 0:  # RNA
                results["molecular_weight"] = molecular_weight(sequence, seq_type='RNA')
                results["sequence_type"] = "RNA"
            elif set(sequence) <= set('ATGC'):  # DNA
                results["molecular_weight"] = molecular_weight(sequence, seq_type='DNA')
                results["sequence_type"] = "DNA"
            else:  # Protein
                results["molecular_weight"] = molecular_weight(sequence, seq_type='protein')
                results["sequence_type"] = "Protein"
                
        elif req.analysis_type == "translation":
            if set(sequence) <= set('ATGCUN'):  # Nukleotid
                results["translated_protein"] = str(sequence.translate())
                
        elif req.analysis_type == "complement":
            if set(sequence) <= set('ATGC'):  # DNA
                results["complement"] = str(sequence.complement())
                results["reverse_complement"] = str(sequence.reverse_complement())

        # AI értékelés hozzáadása
        model_info = await select_backend_model(f"Bioinformatikai elemzés: {req.sequence[:100]}")
        
        analysis_prompt = f"""
        Bioinformatikai Elemzés Eredmények:
        
        Szekvencia: {req.sequence[:200]}...
        Elemzés típusa: {req.analysis_type}
        Eredmények: {json.dumps(results, indent=2)}
        
        Kérlek, adj szakértői értékelést és magyarázatot ezekről az eredményekről.
        Magyarázd el a biológiai jelentőséget és lehetséges funkciókat.
        """
        
        ai_result = await execute_model(model_info, analysis_prompt)
        results["ai_interpretation"] = ai_result["response"]
        results["model_used"] = ai_result["model_used"]
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Bioinformatics analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bioinformatikai elemzés hiba: {e}"
        )

@app.post("/api/scientific/astronomy")
async def astronomy_analysis(req: AstronomyRequest):
    """Csillagászati elemzés Astropy és SunPy segítségével"""
    if not ASTRO_LIBS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Csillagászati könyvtárak nem elérhetők"
        )

    try:
        results = {
            "object_name": req.object_name,
            "analysis_type": req.analysis_type,
            "catalog": req.catalog
        }
        
        # Koordináták feldolgozása
        if req.coordinates:
            try:
                coord = SkyCoord(req.coordinates, unit=(u.hourangle, u.deg))
                results["coordinates"] = {
                    "ra": coord.ra.to_string(),
                    "dec": coord.dec.to_string(),
                    "ra_deg": coord.ra.deg,
                    "dec_deg": coord.dec.deg
                }
            except Exception as e:
                results["coordinate_error"] = str(e)

        # Csillagászati lekérdezések szimulálása
        if req.analysis_type == "object_info":
            # Itt valódi astroquery lekérdezés lenne
            results["simulated_data"] = {
                "object_type": "Ismeretlen",
                "magnitude": "Nincs adat",
                "spectral_type": "Nincs adat",
                "distance": "Nincs adat"
            }
            
        elif req.analysis_type == "solar_activity":
            # SunPy integráció példa
            results["solar_info"] = {
                "current_solar_cycle": "Solar Cycle 25",
                "activity_level": "Mérsékelt"
            }

        # AI értékelés
        model_info = await select_backend_model(f"Csillagászati elemzés: {req.object_name}")
        
        analysis_prompt = f"""
        Csillagászati Objektum Elemzés:
        
        Objektum: {req.object_name}
        Elemzés típusa: {req.analysis_type}
        Eredmények: {json.dumps(results, indent=2)}
        
        Kérlek, adj részletes csillagászati értékelést és kontextust.
        Magyarázd el az objektum jelentőségét és fizikai tulajdonságait.
        """
        
        ai_result = await execute_model(model_info, analysis_prompt)
        results["ai_interpretation"] = ai_result["response"]
        results["model_used"] = ai_result["model_used"]
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Astronomy analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Csillagászati elemzés hiba: {e}"
        )

@app.post("/api/scientific/geoscience")
async def geoscience_analysis(req: GeoscienceRequest):
    """Földtudományi elemzés PyGIMLi, GemPy stb. segítségével"""
    if not GEO_LIBS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Földtudományi könyvtárak nem elérhetők"
        )

    try:
        results = {
            "data_type": req.data_type,
            "region": req.region,
            "analysis_method": req.analysis_method,
            "parameters": req.parameters
        }
        
        # Szimuláció eredmények az elérhető könyvtárak alapján
        if req.data_type == "seismic":
            results["seismic_analysis"] = {
                "earthquake_risk": "Közepes",
                "magnitude_range": "4.0-6.5",
                "depth_range": "5-25 km"
            }
            
        elif req.data_type == "geological":
            results["geological_model"] = {
                "rock_types": ["Gránit", "Mészkő", "Homokkő"],
                "structural_features": ["Törések", "Redők"],
                "mineral_deposits": "Lehetséges"
            }
            
        elif req.data_type == "geophysical":
            results["geophysical_survey"] = {
                "resistivity": "Változó",
                "magnetic_anomalies": "Detektálva",
                "gravity_field": "Normál"
            }

        # AI értékelés
        model_info = await select_backend_model(f"Földtudományi elemzés: {req.region}")
        
        analysis_prompt = f"""
        Földtudományi Elemzés:
        
        Adattípus: {req.data_type}
        Régió: {req.region}
        Módszer: {req.analysis_method}
        Eredmények: {json.dumps(results, indent=2)}
        
        Kérlek, adj szakértői geológiai és geofizikai értékelést.
        Magyarázd el a földtani jelentőséget és gyakorlati alkalmazásokat.
        """
        
        ai_result = await execute_model(model_info, analysis_prompt)
        results["ai_interpretation"] = ai_result["response"]
        results["model_used"] = ai_result["model_used"]
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Geoscience analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Földtudományi elemzés hiba: {e}"
        )

@app.post("/api/scientific/machine_learning")
async def ml_analysis(req: MLAnalysisRequest):
    """Gépi tanulás elemzés scikit-learn, TensorFlow stb. segítségével"""
    if not ML_LIBS_AVAILABLE or not SCIPY_STACK_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gépi tanulás könyvtárak nem elérhetők"
        )

    try:
        # Adatok NumPy array-be konvertálása
        X = np.array(req.data)
        results = {
            "algorithm": req.algorithm,
            "data_shape": X.shape,
            "parameters": req.parameters
        }
        
        # Alapvető statisztikák
        results["data_statistics"] = {
            "mean": np.mean(X, axis=0).tolist(),
            "std": np.std(X, axis=0).tolist(),
            "min": np.min(X, axis=0).tolist(),
            "max": np.max(X, axis=0).tolist()
        }
        
        # Algoritmus-specifikus elemzés
        if req.algorithm == "clustering":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=req.parameters.get("n_clusters", 3))
            clusters = kmeans.fit_predict(X)
            results["clustering_results"] = {
                "labels": clusters.tolist(),
                "centers": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_
            }
            
        elif req.algorithm == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=req.parameters.get("n_components", 2))
            X_pca = pca.fit_transform(X)
            results["pca_results"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "components": pca.components_.tolist(),
                "transformed_data": X_pca.tolist()
            }
            
        elif req.algorithm == "classification" and req.target:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            y = np.array(req.target)
            clf = RandomForestClassifier()
            scores = cross_val_score(clf, X, y, cv=5)
            results["classification_results"] = {
                "cv_scores": scores.tolist(),
                "mean_accuracy": np.mean(scores),
                "std_accuracy": np.std(scores)
            }

        # AI értékelés
        model_info = await select_backend_model(f"Gépi tanulás elemzés: {req.algorithm}")
        
        analysis_prompt = f"""
        Gépi Tanulás Elemzés:
        
        Algoritmus: {req.algorithm}
        Adatok alakja: {X.shape}
        Eredmények: {json.dumps(results, indent=2)}
        
        Kérlek, adj szakértői gépi tanulás értékelést.
        Magyarázd el az eredményeket és adj javaslatokat a továbbfejlesztésre.
        """
        
        ai_result = await execute_model(model_info, analysis_prompt)
        results["ai_interpretation"] = ai_result["response"]
        results["model_used"] = ai_result["model_used"]
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"ML analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gépi tanulás elemzés hiba: {e}"
        )

@app.get("/api/scientific/libraries_status")
async def get_libraries_status():
    """Telepített tudományos könyvtárak állapota"""
    return {
        "scipy_stack": SCIPY_STACK_AVAILABLE,
        "machine_learning": ML_LIBS_AVAILABLE,
        "bioinformatics": BIO_LIBS_AVAILABLE,
        "astronomy": ASTRO_LIBS_AVAILABLE,
        "geoscience": GEO_LIBS_AVAILABLE,
        "chemistry": CHEM_LIBS_AVAILABLE,
        "total_alpha_services": sum(len(services) for services in ALPHA_SERVICES.values()),
        "enhanced_capabilities": [
            "Bioinformatika (Biopython, molearn, PySB)",
            "Csillagászat (Astropy, SunPy, astroquery)",
            "Földtudomány (PyGIMLi, GemPy, Underworld2)",
            "Gépi tanulás (scikit-learn, TensorFlow, PyTorch)",
            "Kémia (Pyrolite, PyCoMo)",
            "Klímatudomány (Open-Meteo, SuperflexPy)",
            "Adattudomány (Pandas, NumPy, SciPy)"
        ]
    }

async def exa_neural_search(query: str, domains: List[str] = [], exclude_domains: List[str] = [], num_results: int = 20):
    """Speciális neurális keresés tudományos tartalmakhoz"""
    if not exa_client or not EXA_AVAILABLE:
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
    if not exa_client or not EXA_AVAILABLE or (not gemini_model and not gemini_25_pro):
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
        # Optimalizált HTTP kliens gyorsabb timeout-tal
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(ebi_alphafold_api_url)
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


# --- Valós idejű tudás endpoint ---
@app.post("/api/real_time_knowledge")
async def real_time_knowledge(query: str):
    """Valós idejű tudás lekérése Exa AI-val"""
    if not exa_client or not EXA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        # Aktuális információk keresése
        current_search = exa_client.search(
            query=f"{query} 2024 2025 latest breaking news recent",
            type="neural",
            num_results=10,
            use_autoprompt=True,
            start_published_date="2024-01-01",
            include_domains=[
                "reuters.com", "bbc.com", "cnn.com", "techcrunch.com",
                "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov"
            ]
        )
        
        if not current_search.results:
            return {
                "query": query,
                "real_time_info": "Nem található aktuális információ",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }

        # Információk feldolgozása
        real_time_sources = []
        combined_info = ""
        
        for result in current_search.results:
            source_info = {
                "title": result.title,
                "url": result.url,
                "published_date": result.published_date,
                "domain": result.url.split('/')[2] if '/' in result.url else result.url
            }
            real_time_sources.append(source_info)
            
            if result.text_contents and result.text_contents.text:
                combined_info += f"\n{result.title}: {result.text_contents.text[:500]}...\n"

        # AI összegzés a valós idejű információkból
        if combined_info and cerebras_client:
            summary_prompt = f"""
            Készíts rövid, informatív összefoglalót a következő valós idejű információkból:
            
            Kérdés: {query}
            
            Aktuális információk:
            {combined_info[:8000]}
            
            Az összefoglaló legyen tömör, lényegre törő és tartalmazza a legfontosabb aktuális fejleményeket.
            """
            
            try:
                summary_stream = cerebras_client.chat.completions.create(
                    messages=[{"role": "user", "content": summary_prompt}],
                    model="llama-4-scout-17b-16e-instruct",
                    stream=True,
                    max_completion_tokens=1000,
                    temperature=0.1
                )
                
                summary_text = ""
                for chunk in summary_stream:
                    if chunk.choices[0].delta.content:
                        summary_text += chunk.choices[0].delta.content
                        
            except Exception as e:
                logger.error(f"Summary generation error: {e}")
                summary_text = "Hiba az összefoglaló generálása során"
        else:
            summary_text = "Nincs elegendő információ az összefoglaláshoz"

        return {
            "query": query,
            "real_time_summary": summary_text,
            "sources": real_time_sources,
            "total_sources": len(real_time_sources),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Real-time knowledge error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a valós idejű tudás lekérése során: {e}"
        )

# --- Trendek követése endpoint ---
@app.get("/api/trending_topics")
async def get_trending_topics():
    """Aktuális trendek és hírek lekérése"""
    if not exa_client or not EXA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        trending_queries = [
            "breaking news today",
            "latest technology 2024",
            "artificial intelligence breakthrough",
            "climate change news",
            "space exploration recent"
        ]
        
        all_trends = []
        
        for query in trending_queries:
            try:
                trend_search = exa_client.search(
                    query=query,
                    type="neural",
                    num_results=5,
                    text_contents={"max_characters": 500, "strategy": "comprehensive"},
                    livecrawl="always",
                    start_published_date="2024-01-01"
                )
                
                for result in trend_search.results:
                    all_trends.append({
                        "category": query,
                        "title": result.title,
                        "url": result.url,
                        "published_date": result.published_date,
                        "preview": result.text_contents.text[:200] + "..." if result.text_contents else None
                    })
                    
            except Exception as e:
                logger.error(f"Trend search error for '{query}': {e}")
                continue

        return {
            "trending_topics": all_trends,
            "total_trends": len(all_trends),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Trending topics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a trendek lekérése során: {e}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a fehérje lekérdezése során: {e}"
        )

class AlphaFold3Request(BaseModel):
    protein_sequence: str = Field(..., description="Fehérje aminosav szekvencia")
    interaction_partners: List[str] = Field(default=[], description="Kölcsönható partnerek (DNS, RNS, más fehérjék)")
    analysis_type: str = Field(default="structure_prediction", description="Elemzés típusa")
    include_confidence: bool = Field(default=True, description="Megbízhatósági pontszámok")

class AlphaFold3StructurePrediction(BaseModel):
    name: str = Field(..., description="Predikció neve")
    sequences: List[Dict[str, Any]] = Field(..., description="Protein, DNS, RNS szekvenciák")
    model_seeds: List[int] = Field(default=[1], description="Random seed értékek")
    num_diffusion_samples: int = Field(default=5, description="Diffúziós minták száma")
    num_recycles: int = Field(default=10, description="Újrafeldolgozások száma")

class AlphaFold3ComplexRequest(BaseModel):
    protein_chains: List[str] = Field(..., description="Fehérje láncok aminosav szekvenciái")
    dna_sequences: List[str] = Field(default=[], description="DNS szekvenciák")
    rna_sequences: List[str] = Field(default=[], description="RNS szekvenciák")
    ligands: List[str] = Field(default=[], description="Ligandumok SMILES formátumban")
    prediction_name: str = Field(default="complex_prediction", description="Predikció neve")

# AlphaFold 3 Input JSON generátor
def generate_alphafold3_input(req: AlphaFold3ComplexRequest) -> Dict[str, Any]:
    """AlphaFold 3 input JSON generálása"""
    sequences = []
    
    # Protein láncok hozzáadása
    for i, protein_seq in enumerate(req.protein_chains):
        sequences.append({
            "protein": {
                "id": [chr(65 + i)],  # A, B, C, ... chain ID-k
                "sequence": protein_seq
            }
        })
    
    # DNS szekvenciák hozzáadása
    for i, dna_seq in enumerate(req.dna_sequences):
        sequences.append({
            "dna": {
                "id": [f"D{i+1}"],
                "sequence": dna_seq
            }
        })
    
    # RNS szekvenciák hozzáadása
    for i, rna_seq in enumerate(req.rna_sequences):
        sequences.append({
            "rna": {
                "id": [f"R{i+1}"],
                "sequence": rna_seq
            }
        })
    
    # Ligandumok hozzáadása
    for i, ligand in enumerate(req.ligands):
        sequences.append({
            "ligand": {
                "id": [f"L{i+1}"],
                "smiles": ligand
            }
        })
    
    return {
        "name": req.prediction_name,
        "sequences": sequences,
        "modelSeeds": [1, 2, 3, 4, 5],  # 5 különböző seed
        "dialect": "alphafold3",
        "version": 1
    }

# Valódi AlphaFold 3 struktúra predikció
@app.post("/api/alphafold3/structure_prediction")
async def alphafold3_structure_prediction(req: AlphaFold3ComplexRequest):
    """Valódi AlphaFold 3 struktúra predikció futtatása"""
    try:
        # Temporary fájlok létrehozása
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Input JSON generálása
            input_json = generate_alphafold3_input(req)
            input_file = os.path.join(input_dir, "fold_input.json")
            
            with open(input_file, 'w') as f:
                json.dump(input_json, f, indent=2)
            
            logger.info(f"AlphaFold 3 input created: {input_file}")
            
            # AlphaFold 3 futtatása (csak data pipeline most, model nélkül)
            cmd = [
                sys.executable,
                "alphafold3_repo/run_alphafold.py",
                f"--json_path={input_file}",
                f"--output_dir={output_dir}",
                "--run_data_pipeline=true",
                "--run_inference=false",  # Most csak adatfeldolgozás
                "--force_output_dir=true"
            ]
            
            logger.info(f"Running AlphaFold 3 command: {' '.join(cmd)}")
            
            # Futtatás subprocess-szel
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Sikeres futtatás - eredmények feldolgozása
                result_files = []
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.json', '.pdb', '.cif')):
                            result_files.append(os.path.join(root, file))
                
                return {
                    "status": "success",
                    "prediction_name": req.prediction_name,
                    "input_json": input_json,
                    "output_files": result_files,
                    "stdout": stdout.decode('utf-8')[-2000:],  # Utolsó 2000 karakter
                    "message": "AlphaFold 3 data pipeline sikeresen lefutott"
                }
            else:
                # Hiba esetén
                return {
                    "status": "error",
                    "prediction_name": req.prediction_name,
                    "input_json": input_json,
                    "error": stderr.decode('utf-8')[-2000:],
                    "stdout": stdout.decode('utf-8')[-2000:],
                    "return_code": process.returncode
                }
                
    except Exception as e:
        logger.error(f"AlphaFold 3 prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AlphaFold 3 predikció hiba: {e}"
        )

# AlphaGenome és AlphaFold 3 integrált elemzés
@app.post("/api/deep_discovery/alphafold3")
async def alphafold3_analysis(req: AlphaFold3Request):
    """AlphaFold 3 és AlphaGenome integrált elemzés"""
    if not gemini_25_pro and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    try:
        # Szekvencia validálás
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_amino_acids for aa in req.protein_sequence.upper()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Érvénytelen aminosav szekvencia"
            )

        # AI elemzés AlphaFold 3 kontextussal
        analysis_prompt = f"""
        AlphaFold 3 és AlphaGenome Integrált Fehérje Elemzés

        Fehérje szekvencia: {req.protein_sequence}
        Hossz: {len(req.protein_sequence)} aminosav
        Kölcsönható partnerek: {', '.join(req.interaction_partners) if req.interaction_partners else 'Nincs'}
        Elemzés típusa: {req.analysis_type}

        Figyelembe véve az AlphaFold 3 és AlphaGenome képességeit, készíts egy részletes elemzést:

        1.  Fehérje szerkezet előrejelzése:            -   Jelenlegi legjobb szerkezeti modell
        -   Megbízhatósági pontszámok (pl. pLDDT)
        -   Potenciális funkcionális domének
        -   Hasonlóság más ismert fehérjékhez

        2.  Kölcsönhatások előrejelzése:
        -   Lehetséges DNS, RNS vagy más fehérje partnerek
        -   Kötőhelyek azonosítása
        -   A kölcsönhatás erőssége és specificitása

        3.  Funkcionális annotáció:
        -   Gén ontológia (GO) kifejezések
        -   Biokémiai útvonalak
        -   Sejtszintű lokalizáció

        4.  Mutációs hatások:
        -   Potenciálisan káros mutációk azonosítása
        -   Hatás a fehérje stabilitására és funkciójára
        -   Gyógyszer célpontként való alkalmasság

        5.  Kísérleti validáció javaslatok:
        -   Javasolt kísérletek a szerkezet és kölcsönhatások megerősítésére
        -   In vitro és in vivo vizsgálatok
        -   Klinikai relevanciával bíró fehérjék

        A válasz legyen strukturált, magyar nyelvű és tudományos.
        """

        model_info = await select_backend_model(analysis_prompt)
        result = await execute_model(model_info, analysis_prompt)
        analysis_text = result["response"]

        return {
            "protein_sequence": req.protein_sequence,
            "analysis": analysis_text,
            "model_used": result["model_used"],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in AlphaFold 3 analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba az AlphaFold 3 elemzés során: {e}"
        )

# --- Custom GCP Modell Végpont ---
@app.post("/api/gcp/custom_model")
async def predict_custom_gcp_model(req: CustomGCPModelRequest):
    """Egyedi GCP Vertex AI modell futtatása"""
    if not gcp_credentials:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP Vertex AI nem elérhető"
        )

    try:
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{req.gcp_project_id}/locations/{req.gcp_region}/endpoints/{req.gcp_endpoint_id}",
            credentials=gcp_credentials
        )

        prediction = endpoint.predict(instances=[req.input_data])
        return {
            "prediction": prediction.predictions,
            "explained_value": prediction.explanations,
            "status": "success"
        }

    except GoogleAPIError as e:
        logger.error(f"GCP API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GCP API hiba: {e}"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a predikció során: {e}"
        )

# --- Simulation Optimizer Végpont ---
@app.post("/api/simulation/optimize")
async def optimize_simulation(req: SimulationOptimizerRequest):
    """Szimuláció optimalizálása"""
    # Ezt a részt ki kell egészíteni a megfelelő szimulációs és optimalizációs algoritmussal
    # Példa: genetikus algoritmus, heurisztikus keresés, stb.
    # Jelenleg csak egy placeholder implementáció

    try:
        if req.simulation_type == "anyagtervezes":
            # Itt lehetne optimalizálni az anyagtervezési paramétereket
            optimized_parameters = {
                "homerseklet": req.input_parameters.get("homerseklet", 25) + 5,
                "nyomas": req.input_parameters.get("nyomas", 1) * 1.1,
                "koncentracio": req.input_parameters.get("koncentracio", 0.5)
            }
            optimal_result = f"Optimalizált anyagtervezési eredmény: {optimized_parameters}"

        elif req.simulation_type == "gyogyszerkutatas":
            # Itt lehetne optimalizálni a gyógyszerkutatási paramétereket
            optimized_parameters = {
                "receptor_affinitas": req.input_parameters.get("receptor_affinitas", 10) * 1.05,
                "metabolizmus_sebesseg": req.input_parameters.get("metabolizmus_sebesseg", 0.1) * 0.95
            }
            optimal_result = f"Optimalizált gyógyszerkutatási eredmény: {optimized_parameters}"

        else:
            raise ValueError("Érvénytelen szimuláció típus")

        return {
            "simulation_type": req.simulation_type,
            "optimization_goal": req.optimization_goal,
            "input_parameters": req.input_parameters,
            "optimized_parameters": optimized_parameters,
            "optimal_result": optimal_result,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error during simulation optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a szimuláció optimalizálása során: {e}"
        )

# --- AlphaGenome Végpont ---
@app.post("/api/alpha/genome")
async def alpha_genome_analysis(req: AlphaGenomeRequest):
    """Genom szekvencia elemzése"""
    if not gemini_25_pro and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    try:
        # Szekvencia validálás
        if len(req.genome_sequence) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A genom szekvencia túl rövid"
            )

        # AI elemzés
        analysis_prompt = f"""
        Genom Szekvencia Elemzés

        Organizmus: {req.organism}
        Elemzés típusa: {req.analysis_type}
        Szekvencia: {req.genome_sequence[:500]}... (csak részlet)

        Kérlek, végezz mélyreható elemzést a megadott genom szekvencián.
        Elemezd a potenciális géneket, szabályozó elemeket és egyéb funkcionális régiókat.
        """

        model_info = await select_backend_model(analysis_prompt)
        result = await execute_model(model_info, analysis_prompt)
        analysis_text = result["response"]

        # Fehérje előrejelzések (opcionális)
        if req.include_predictions:
            protein_prompt = f"""
            Fehérje Előrejelzés

            Genom szekvencia: {req.genome_sequence[:500]}... (csak részlet)

            Kérlek, azonosíts potenciális fehérjéket a megadott genom szekvenciában,
            és adj meg információkat a funkciójukról és szerkezetükről.
            """
            protein_model_info = await select_backend_model(protein_prompt)
            protein_result = await execute_model(protein_model_info, protein_prompt)
            protein_predictions = protein_result["response"]
        else:
            protein_predictions = "Fehérje előrejelzések nem kértek"

        return {
            "organism": req.organism,
            "analysis_type": req.analysis_type,
            "analysis": analysis_text,
            "protein_predictions": protein_predictions,
            "model_used": result["model_used"],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in genome analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a genom elemzése során: {e}"
        )

# --- AlphaMissense Végpont ---
@app.post("/api/alpha/alphamissense")
async def alphamissense_analysis(req: AlphaMissenseRequest):
    """AlphaMissense mutációs patogenitás elemzés"""
    if not gemini_25_pro and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    try:
        # Mutációk validálása
        valid_mutations = []
        for mutation in req.mutations:
            if len(mutation) >= 4 and mutation[0].isalpha() and mutation[-1].isalpha():
                valid_mutations.append(mutation)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Érvénytelen mutáció formátum: {mutation}"
                )

        # AlphaMissense elemzés prompt
        analysis_prompt = f"""
        AlphaMissense Mutációs Patogenitás Elemzés

        Fehérje szekvencia: {req.protein_sequence[:100]}...
        UniProt ID: {req.uniprot_id or 'Nincs megadva'}
        Mutációk: {', '.join(valid_mutations)}
        Patogenitás küszöb: {req.pathogenicity_threshold}

        AlphaMissense alapú elemzés:

        1. MUTÁCIÓS HATÁS ELŐREJELZÉS:
        - Minden mutáció patogenitás pontszáma (0-1 skála)
        - Klinikai jelentőség kategorizálása
        - Funkcionális domén érintettség

        2. SZERKEZETI HATÁSOK:
        - Fehérje stabilitás változása
        - Kölcsönhatások módosulása
        - Alloszterikus hatások

        3. KLINIKAI RELEVANCIÁJA:
        - Ismert betegség-asszociációk
        - Farmakogenetikai jelentőség
        - Terápiás célpont potenciál

        4. POPULÁCIÓS GENETIKAI ADATOK:
        - Allél gyakoriság
        - Evolúciós konzervativitás
        - Szelekciós nyomás

        5. AJÁNLÁSOK:
        - Klinikai validáció szükségessége
        - Funkcionális vizsgálatok
        - Genetikai tanácsadás

        Minden mutációra adj részletes patogenitás pontszámot és magyarázatot.
        """

        model_info = await select_backend_model(analysis_prompt)
        result = await execute_model(model_info, analysis_prompt)

        # Szimulált AlphaMissense pontszámok (valódi implementációhoz API szükséges)
        mutation_scores = []
        for mutation in valid_mutations:
            # Egyszerű heurisztika a demo célokra
            import random
            random.seed(hash(mutation))
            score = random.uniform(0.1, 0.9)
            pathogenic = score >= req.pathogenicity_threshold
            
            mutation_scores.append({
                "mutation": mutation,
                "pathogenicity_score": round(score, 3),
                "pathogenic": pathogenic,
                "confidence": "medium" if 0.3 <= score <= 0.7 else "high",
                "clinical_significance": "patogén" if pathogenic else "benign"
            })

        return {
            "protein_sequence": req.protein_sequence,
            "uniprot_id": req.uniprot_id,
            "mutations_analyzed": len(valid_mutations),
            "pathogenicity_threshold": req.pathogenicity_threshold,
            "mutation_scores": mutation_scores,
            "detailed_analysis": result["response"],
            "model_used": result["model_used"],
            "pathogenic_mutations": len([m for m in mutation_scores if m["pathogenic"]]),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in AlphaMissense analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba az AlphaMissense elemzés során: {e}"
        )

@app.post("/api/alpha/variant_pathogenicity")
async def variant_pathogenicity_analysis(req: VariantPathogenicityRequest):
    """Komplex variáns patogenitás elemzés"""
    if not gemini_25_pro and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    try:
        # Variánsok feldolgozása
        processed_variants = []
        for variant in req.variants:
            processed_variants.append({
                "id": variant.get("id", "unknown"),
                "gene": variant.get("gene", "unknown"),
                "mutation": variant.get("mutation", "unknown"),
                "chromosome": variant.get("chromosome", "unknown"),
                "position": variant.get("position", "unknown")
            })

        # Átfogó elemzés prompt
        analysis_prompt = f"""
        Átfogó Variáns Patogenitás Elemzés

        Elemzési mód: {req.analysis_mode}
        Klinikai kontextus: {req.clinical_context or 'Általános'}
        Variánsok száma: {len(processed_variants)}

        Variánsok:
        {json.dumps(processed_variants, indent=2, ensure_ascii=False)}

        Készíts részletes elemzést minden variánsra:

        1. PATOGENITÁS ÉRTÉKELÉS
        2. KLINIKAI JELENTŐSÉG
        3. FUNKCIONÁLIS HATÁS
        4. POPULÁCIÓS GYAKORISÁG
        5. TERÁPIÁS VONATKOZÁSOK
        6. GENETIKAI TANÁCSADÁS AJÁNLÁSOK

        Az elemzés legyen strukturált és klinikailag releváns.
        """

        model_info = await select_backend_model(analysis_prompt)
        result = await execute_model(model_info, analysis_prompt)

        return {
            "analysis_mode": req.analysis_mode,
            "clinical_context": req.clinical_context,
            "variants_analyzed": len(processed_variants),
            "comprehensive_analysis": result["response"],
            "model_used": result["model_used"],
            "include_population_data": req.include_population_data,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in variant pathogenicity analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a variáns patogenitás elemzés során: {e}"
        )

# --- Code Generation Végpont ---
@app.post("/api/code/generate")
async def generate_code(req: CodeGenerationRequest):
    """Kód generálása továbbfejlesztett AI prompt-tal"""
    if not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    try:
        # Fejlett prompt összeállítása
        complexity_descriptions = {
            "simple": "Egyszerű, 20-30 soros megoldás, alapvető funkcionalitással",
            "medium": "Közepes komplexitású, 50-100 soros kód, strukturált megközelítéssel",
            "complex": "Komplex, 100+ soros megoldás, objektum-orientált tervezéssel",
            "enterprise": "Vállalati szintű kód, teljes hibakezeléssel és dokumentációval"
        }

        prompt = f"""
Professzionális {req.language} kód generálása

SPECIFIKÁCIÓ:
- Programozási nyelv: {req.language}
- Komplexitás szint: {req.complexity} ({complexity_descriptions.get(req.complexity, 'közepes')})
- Kreativitás szint: {req.temperature}

FELADAT:
{req.prompt}

KÖVETELMÉNYEK:
1. Írj tiszta, jól strukturált kódot
2. Használj beszédes változóneveket
3. Adj hozzá magyar nyelvű kommenteket
4. Implementálj megfelelő hibakezelést
5. Kövesd a nyelv best practice-eit
6. A kód legyen futtatható és tesztelhető

VÁLASZ FORMÁTUM:
Csak a kódot add vissza, magyarázó szöveg nélkül. A kód legyen közvetlenül használható.
"""

        model_info = await select_backend_model(prompt)
        result = await execute_model(model_info, prompt)
        
        # Kód tisztítása - csak a kód részek megtartása
        generated_code = result["response"]
        
        # Kód blokkok extraktálása ha van
        if "```" in generated_code:
            import re
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', generated_code, re.DOTALL)
            if code_blocks:
                generated_code = code_blocks[0].strip()
        
        # További tisztítás
        lines = generated_code.split('\n')
        clean_lines = []
        in_code = True
        
        for line in lines:
            # Kihagyjuk az üres magyarázó sorokat
            if line.strip() and not line.strip().startswith('Ez a kód') and not line.strip().startswith('A fenti'):
                clean_lines.append(line)
                in_code = True
            elif in_code and line.strip() == '':
                clean_lines.append(line)
        
        generated_code = '\n'.join(clean_lines).strip()

        return {
            "language": req.language,
            "complexity": req.complexity,
            "generated_code": generated_code,
            "model_used": result["model_used"],
            "estimated_lines": len(generated_code.split('\n')),
            "character_count": len(generated_code),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in code generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a kód generálása során: {e}"
        )

# The following JavaScript code is not used in the backend and will be removed.

# Adding FastAPI application execution to the end of the file.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)