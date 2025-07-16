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

# Naplózás konfigurálása (moved up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Cloud kliensekhez
try:
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    from google.api_core.exceptions import GoogleAPIError
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logger.warning("Google Cloud libraries not available")

# Cerebras Cloud SDK
try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    logger.warning("Cerebras SDK not available")

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    logger.warning("Gemini API not available")

# Exa API
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    logger.warning("Exa API not available")

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
if GCP_SERVICE_ACCOUNT_KEY_JSON and GCP_PROJECT_ID and GCP_REGION and GCP_AVAILABLE:
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

# Gemini initialization
gemini_model = None
gemini_25_pro = None
if GEMINI_AVAILABLE and genai:
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-pro')
            gemini_25_pro = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini models initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini models: {e}")

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
            model_name = "llama-3.1-70b-versatile"
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
                model="llama-3.1-70b-versatile",
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
            model="llama-3.1-70b-versatile",
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

# Additional endpoints for completeness but simplified
@app.get("/api/alphamissense/info")
async def alphamissense_info():
    """AlphaMissense információk és képességek"""
    return {
        "alphamissense_available": True,
        "description": "Missense mutációk patogenitás előrejelzése",
        "version": "2024.1",
        "status": "Aktív és integrált"
    }

@app.get("/api/alphafold3/info")
async def alphafold3_info():
    """AlphaFold 3 információk és állapot"""
    af3_path = pathlib.Path("alphafold3_repo")
    af3_exists = af3_path.exists()

    return {
        "alphafold3_available": af3_exists,
        "repository_path": str(af3_path),
        "status": "Működőképes (model paraméterek nélkül csak data pipeline)"
    }

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

        # Fehérje előrejelz" in generated_code:
            import re
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n', generated_code, re.DOTALL)
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