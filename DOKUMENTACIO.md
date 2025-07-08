
# Jade - Deep Discovery AI Platform

## Egyszerű magyarázat

### Mi ez?
A Jade egy intelligens chatbot, ami speciálisan tudományos kutatásokhoz készült. Olyan, mint a ChatGPT, de tudományos szuperképességekkel.

### Mit tud?

#### 🗣️ Beszélgetés
- Válaszol kérdéseidre magyarul
- Megjegyzi a korábbi beszélgetéseket
- Több AI "agy"-at használ a legjobb válaszokhoz

#### 🧪 Kutatási Trendek
- Megkeresi a legfrissebb tudományos cikkeket
- Összefoglalja az újdonságokat
- Megmutatja a forrásokat

#### 🧬 Fehérje Struktúra Keresés
- Fehérjék 3D szerkezetét mutatja meg
- AlphaFold adatbázist használ
- Letölthető struktúra fájlok

#### ☁️ Egyedi AI Modellek
- Google Cloud modellek futtatása
- Speciális tudományos számítások
- Kémiai elemzések

#### ⚙️ Szimuláció Optimalizáló
- Tudományos szimulációk beállítása
- Paraméterek optimalizálása
- Kód generálás

### Hogyan használd?

1. **Nyisd meg a weboldalt**
2. **Írj egy üzenetet** a chat mezőbe
3. **Használd a speciális gombokat** a bal oldali menüben
4. **Fedezz fel** új tudományos információkat!

### Kinek való?
- Kutatók, tudósok
- Egyetemisták
- Bárki, aki kíváncsi a tudományra

## Fájlok leírása

### `main.py`
**Mit csinál:** A szerver, ami futtatja az egész programot
**Tartalom:**
- FastAPI webszerver
- 4 AI modell integrálása:
  - Cerebras Llama 4 (chat)
  - Google Gemini 2.5 Pro (elemzés)
  - Exa AI (keresés)
  - Google Cloud Vertex AI (egyedi modellek)
- 6 API végpont a különböző funkciókhoz
- Beszélgetési előzmények kezelése
- Hibakezelés és naplózás

**Főbb funkciók:**
- `/api/deep_discovery/chat` - Beszélgetés
- `/api/deep_discovery/research_trends` - Kutatási trendek
- `/api/deep_discovery/protein_structure` - Fehérje keresés
- `/api/deep_discovery/custom_gcp_model` - GCP modellek
- `/api/deep_discovery/simulation_optimizer` - Szimuláció optimalizálás

### `templates/index.html`
**Mit csinál:** Teljesen újratervezett webes felhasználói felület
**Új funkciók:**
- **Üdvözlő képernyő**: Intelligens javaslatok gyors kezdéshez
- **Lebegő eszköztár**: Gyors hozzáférés a speciális funkciókhoz
- **Okos gyorsműveletek**: Az AI válaszok alapján javasolt következő lépések
- **Folyamatos chat élmény**: Oldalsáv nélküli, fókuszált beszélgetés
- **Responsív dizájn**: Mobilra és asztali gépekre optimalizált

**Felhasználói elemek:**
- Központi chat ablak teljes képernyős élménnyel
- Intelligens üdvözlő képernyő példakérdésekkel
- Lebegő eszköz gombok a jobb oldalon
- Gyorsműveletek az AI válaszok alatt
- Modernizált modális ablakok
- Valós idejű gépelés indikátor
- Automatikus textarea méretezés

### `.replit`
**Mit csinál:** Replit konfiguráció
**Tartalom:**
- Python 3.11 modul használata
- `main.py` mint belépési pont
- Cloud Run deployment beállítása
- Futtatási workflow definiálása
- Port 5000 beállítása

### `pyproject.toml`
**Mit csinál:** Python projekt beállítások
**Tartalom:**
- Projekt metaadatok
- Python 3.11+ követelmény
- Cerebras Cloud SDK függőség

### Hiányzó fájlok (telepítendők)
- `requirements.txt` - További Python csomagok
- `replit.nix` - Nix környezet konfiguráció

## API kulcsok (Replit Secrets-ben)
- `CEREBRAS_API_KEY` - Cerebras AI modellekhez
- `GEMINI_API_KEY` - Google Gemini modellhez
- `EXA_API_KEY` - Exa kereséshez
- `GCP_SERVICE_ACCOUNT_KEY` - Google Cloud hozzáféréshez
- `GCP_PROJECT_ID` - Google Cloud projekt
- `GCP_REGION` - Google Cloud régió
