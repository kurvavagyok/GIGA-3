# Jade - Deep Discovery AI Platform Dokumentáció

## Alkalmazás Leírása

A Jade egy fejlett tudományos AI platform, amely több AI modellt integrál egyetlen, modern webes felületen. A platform speciálisan tudományos kutatásokhoz, innovációs projektekhez és fehérje-struktúra elemzésekhez készült.

### Fő Funkciók:
- **Hibrid AI Chat**: Cerebras Llama 4 és Google Gemini 1.5/2.5 Pro modellek
- **Deep Research**: Többmodelles elemzés párhuzamos feldolgozással
- **Kutatási Trendek**: Exa AI keresés + Gemini elemzés
- **Fehérje Struktúra**: AlphaFold adatbázis integráció
- **Egyedi GCP Modellek**: Google Cloud Vertex AI támogatás
- **Szimuláció Optimalizáló**: AI-vezérelt paraméter optimalizálás
- **AlphaGenome Elemzés**: DNS/RNS szekvencia elemzés
- **150+ Alpha Szolgáltatás**: Tudományos és innovációs szolgáltatások

---

## Alpha Szolgáltatások (150+)

### 1. Biológiai/Orvosi Kategória
- **AlphaGenome**: Genomikai elemzés és szekvencia vizsgálat
- **AlphaProtein**: Fehérje struktúra és funkció predikció
- **AlphaImmune**: Immunrendszer modellezés
- **AlphaCardio**: Szív-érrendszeri elemzések
- **AlphaNeuro**: Idegrendszeri szimulációk
- **AlphaMicrobiome**: Mikrobiom elemzés
- **AlphaPathogen**: Kórokozók vizsgálata
- **AlphaVaccine**: Vakcina fejlesztés támogatása
- **AlphaDrug**: Gyógyszer-molekula tervezés
- **AlphaOncology**: Onkológiai kutatás
- **AlphaMedical**: Orvosi diagnosztika
- **AlphaBiomarker**: Biomarker azonosítás

### 2. Kémiai/Anyagtudományi Kategória
- **AlphaCatalyst**: Katalizátor tervezés és optimalizálás
- **AlphaPolymer**: Polimer tulajdonságok előrejelzése
- **AlphaNanotech**: Nanomateriál szintézis
- **AlphaMaterial**: Anyagtulajdonságok predikciója
- **AlphaSuperconductor**: Szupravezető anyagok
- **AlphaSemiconductor**: Félvezető anyagok
- **AlphaComposite**: Kompozit anyagok
- **AlphaBattery**: Akkumulátor technológiák
- **AlphaSolar**: Napelem optimalizálás
- **AlphaCorrosion**: Korrózió elemzés

### 3. Környezeti/Fenntartható Kategória
- **AlphaClimate**: Klímaváltozás modellezése
- **AlphaOcean**: Óceáni rendszerek elemzése
- **AlphaAtmosphere**: Légköri folyamatok
- **AlphaEcology**: Ökológiai rendszerek
- **AlphaWater**: Víz minőség és kezelés
- **AlphaRenewable**: Megújuló energia
- **AlphaCarbon**: Szén-dioxid befogás
- **AlphaWaste**: Hulladékgazdálkodás
- **AlphaBiodiversity**: Biodiverzitás védelem

### 4. Fizikai/Asztrofizikai Kategória
- **AlphaQuantum**: Kvantumfizikai szimulációk
- **AlphaParticle**: Részecskefizika
- **AlphaGravity**: Gravitációs hullámok
- **AlphaCosmic**: Kozmikus sugárzás
- **AlphaStellar**: Csillagfejlődés
- **AlphaGalaxy**: Galaxisok dinamikája
- **AlphaExoplanet**: Exobolygó karakterizálás
- **AlphaPlasma**: Plazma fizika

### 5. Technológiai/Mélyműszaki Kategória
- **AlphaAI**: Mesterséges intelligencia architektúrák
- **AlphaML**: Gépi tanulás optimalizálás
- **AlphaNeural**: Neurális hálózatok
- **AlphaRobotics**: Robotikai rendszerek
- **AlphaAutonomy**: Autonóm rendszerek
- **AlphaVision**: Számítógépes látás
- **AlphaNLP**: Természetes nyelv feldolgozás
- **AlphaOptimization**: Optimalizálási algoritmusok

### 6. Társadalmi/Gazdasági Kategória
- **AlphaEconomy**: Gazdasági modellezés
- **AlphaMarket**: Piacanalízis
- **AlphaFinance**: Pénzügyi elemzés
- **AlphaTrade**: Kereskedelmi stratégiák
- **AlphaSupply**: Ellátási lánc optimalizálás
- **AlphaRisk**: Kockázatelemzés
- **AlphaBehavior**: Viselkedés elemzés
- **AlphaSocial**: Társadalmi hálózatok

---

## Technikai Részletek

### API Végpontok

#### 1. Chat Rendszer
```python
@app.post("/api/deep_discovery/chat")
async def deep_discovery_chat(req: ChatRequest):
```
- **Funkció**: Hibrid AI chat cache-eléssel
- **Modellek**: Cerebras Llama 4, Gemini 1.5/2.5 Pro
- **Beszélgetési előzmények**: Felhasználó ID alapú tárolás

#### 2. Deep Research
```python
@app.post("/api/deep_research")
async def deep_research(req: DeepResearchRequest):
```
- **Funkció**: Többmodelles párhuzamos elemzés
- **Folyamat**: Exa keresés → Párhuzamos AI elemzés → Végső szintézis
- **Modellek**: Cerebras, Gemini 1.5 Pro, Gemini 2.5 Pro

#### 3. Kutatási Trendek
```python
@app.post("/api/deep_discovery/research_trends")
async def get_research_trends(req: ScientificInsightRequest):
```
- **Funkció**: Tudományos trendek elemzése
- **Workflow**: Exa keresés → Gemini elemzés → Strukturált összefoglaló

#### 4. Fehérje Struktúra
```python
@app.post("/api/deep_discovery/protein_structure")
async def protein_structure_lookup(req: ProteinLookupRequest):
```
- **Funkció**: AlphaFold adatbázis lekérdezés
- **API**: EMBL-EBI AlphaFold Protein Structure Database
- **Adatok**: pLDDT konfidencia, PDB/CIF struktúrák

#### 5. AlphaGenome Elemzés
```python
@app.post("/api/deep_discovery/alphagenome")
async def alphagenome_analysis(req: AlphaGenomeRequest):
```
- **Funkció**: DNS/RNS szekvencia elemzés
- **Elemzések**: GC tartalom, ORF keresés, kozak szekvencia
- **Validáció**: Szekvencia hossz és karakter ellenőrzés

#### 6. Szimuláció Optimalizáló
```python
@app.post("/api/deep_discovery/simulation_optimizer")
async def simulation_optimizer(req: SimulationOptimizerRequest):
```
- **Funkció**: AI-vezérelt paraméter optimalizálás
- **Típusok**: Molekuláris dinamika, anyagtulajdonságok
- **Modellek**: Cerebras Llama 4 vagy Gemini

#### 7. Backend Modell Kiválasztás
```python
async def select_backend_model(prompt: str, service_name: str = None):
```
- **Funkció**: Automatikus modell kiválasztás
- **Logika**: Teljesítmény és token limit alapú
- **Optimalizálás**: Prompt komplexitás értékelés

### Optimalizációs Funkciók

#### Cache Rendszer
```python
response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY = 300  # 5 perc
```
- **Funkció**: Válasz gyorsítótár
- **Lejárat**: 5 perces automatikus törlés
- **Kulcs**: Prompt hash alapú

#### Exa Keresés Optimalizálás
```python
async def advanced_exa_search(query: str, search_type: str = "neural"):
```
- **Típusok**: Neurális, kulcsszavas, hasonlóság alapú
- **Tartalom**: Automatikus extrakció és tisztítás
- **Források**: Tudományos publikációk priorizálása

### Biztonsági Funkciók

#### Digitális Ujjlenyomat
```python
DIGITAL_FINGERPRINT = "Jade made by Kollár Sándor"
CREATOR_SIGNATURE = "SmFkZSBtYWRlIGJ5IEtvbGzDoXIgU8OhbmRvcg=="
CREATOR_HASH = "a7b4c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5"
```

#### Környezeti Változók
- **GCP_SERVICE_ACCOUNT_KEY**: Google Cloud hitelesítés
- **CEREBRAS_API_KEY**: Cerebras Cloud SDK
- **GEMINI_API_KEY**: Google Gemini API
- **EXA_API_KEY**: Exa keresési platform

### Frontend Felület

#### Modern UI Komponensek
- **Fekete háttér**: Elegáns, modern megjelenés
- **Responsive design**: Mobil és desktop optimalizálás
- **Modális ablakok**: Speciális funkciók kezelése
- **Valós idejű chat**: Dinamikus tartalom betöltés
- **Animációk**: Smooth felhasználói élmény

#### Funkcionális Eszközök
- **Új beszélgetés**: Chat előzmények törlése
- **Eszköz gombok**: Gyors hozzáférés Alpha szolgáltatásokhoz
- **Sidebar navigáció**: Összes funkció egy helyen
- **Válasz műveletek**: Kontextus alapú javaslatok

---

## Gyakorlati Alkalmazások

### Tudományos Kutatás
- Automatizált irodalomkutatás
- Hipotézis generálás
- Adatelemzés támogatás
- Publikáció előkészítés

### Innovációs Fejlesztés
- Új anyagok tervezése
- Technológiai megoldások
- Szabadalmi kutatás
- Versenyelemzés

### Orvosi Alkalmazások
- Diagnosztikai támogatás
- Gyógyszer fejlesztés
- Genomikai elemzés
- Személyre szabott medicina

### Környezeti Monitoring
- Klímaváltozás elemzés
- Ökológiai rendszerek
- Fenntarthatósági értékelés
- Környezeti hatásvizsgálat

---

## Telepítés és Konfigurálás

### Szükséges Függőségek
```bash
pip install -r requirements.txt
```

### Környezeti Változók Beállítása
```bash
export GCP_SERVICE_ACCOUNT_KEY="your_gcp_key"
export CEREBRAS_API_KEY="your_cerebras_key"
export GEMINI_API_KEY="your_gemini_key"
export EXA_API_KEY="your_exa_key"
```

### Alkalmazás Indítása
```bash
python main.py
```

### Port Konfiguráció
- **Fejlesztési port**: 5000
- **Produkciós port**: 80/443 (automatikus forwarding)

---

## Verziótörténet

### v2.0.0 (Jelenlegi)
- 150+ Alpha szolgáltatás hozzáadása
- Deep Research funkció
- Backend modell kiválasztás
- Fejlett cache rendszer
- AlphaGenome elemzés
- Optimalizált Exa keresés

### v1.0.0 (Alapverzió)
- Alapvető chat funkcionalitás
- Kutatási trendek
- Fehérje struktúra keresés
- GCP modell integráció
- Szimuláció optimalizáló

---

## Támogatás és Fejlesztés

**Fejlesztő**: Kollár Sándor  
**Platform**: Replit  
**Technológiai stack**: Python, FastAPI, Modern Web Technologies  
**AI Modellek**: Cerebras Llama 4, Google Gemini 1.5/2.5 Pro, Exa AI  

A Jade platform folyamatosan fejlődik, új Alpha szolgáltatásokkal és funkcionalitásokkal bővül a tudományos közösség igényei szerint.