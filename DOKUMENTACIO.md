# Deep Discovery AI - Univerzális Tudományos Platform

## Alkalmazás áttekintése
A Deep Discovery AI egy teljes körű, piaci értékű mesterséges intelligencia platform hibrid AI modellekkel, amely tudományos kutatást, innovációt és üzleti alkalmazásokat szolgál ki előfizetéses modellel.

## Univerzális Rendszer Architektúra

### 1. Hibrid AI Orchestrator
- **Intelligens modell-váltás**: Automatikus optimális modell kiválasztás
- **Teljesítmény monitoring**: Valós idejű modell teljesítmény követés
- **Költségoptimalizálás**: Dinamikus modell allokáció költség alapján
- **Fallback rendszer**: Automatikus átváltás sikertelen kérések esetén

### 2. Piaci Monetizáció
- **Előfizetéses szintek**:
  - Basic: 50 API hívás/nap, alap modellek
  - Pro: 500 API hívás/nap, fejlett modellek, párhuzamos kérések
  - Enterprise: 10.000 API hívás/nap, teljes hozzáférés, prioritás
- **Rate limiting**: Automatikus használat korlátozás
- **Usage tracking**: Részletes használati statisztikák

### 3. Adattár és Cache Rendszer
- **SQLite adatbázis**: Felhasználók, előzmények, cache
- **Intelligens cache**: 1 órás cache kutatási eredményekhez
- **Használati analytics**: Teljes körű monitoring és riportolás

### 4. Deployment Optimalizáció
- **Replit Autoscale**: Automatikus skálázás terhelés alapján
- **Stateless architektúra**: Horizontális skálázhatóság
- **Monitoring**: Valós idejű rendszer állapot követés

## Hibrid AI Modellek
1. **Cerebras Qwen 3** (qwen2.5-72b-instruct) - Gyors általános feladatok
2. **Cerebras Llama 4** (llama-4-scout-17b-16e-instruct) - Beszélgetés és elemzés
3. **Google Gemini 2.5 Pro** - Tudományos kutatás és komplex feladatok

## Fájlok és funkcióik

### main.py
```python
# Teljes hibrid AI platform implementáció
# - AIOrchestrator osztály intelligens modell-váltáshoz
# - SQLite adatbázis perzisztens tároláshoz
# - Rate limiting és felhasználói limitek
# - Cache rendszer optimalizált teljesítményhez
# - Piaci előfizetési szintek kezelése
# - Analitikai dashboard API
```

### templates/index.html
Modern, reszponzív chat interfész Apple Sign In integrációval, vendég móddal és real-time funkciókkal.

**Főbb funkciók:**
- Apple Sign In integráció
- **Vendég mód**: Fiók nélküli használat lehetősége
- Hibrid AI chat interfész
- Real-time üzenetküldés
- Reszponzív design

### templates/admin.html
```html
<!-- Fejlett admin dashboard -->
<!-- - Valós idejű statisztikák -->
<!-- - Interaktív diagramok -->
<!-- - Modell teljesítmény monitoring -->
<!-- - Felhasználói aktivitás követés -->
```

### templates/login.html
Bejelentkezési oldal Apple Sign In és hagyományos email/jelszó opcióval.

### requirements.txt
Az alkalmazás összes Python függősége optimalizálva.

## Piaci Értékajánlat

### Tudományos Kutatás
- **Genom elemzés**: AlphaGenome funkciók fejlett DNS/RNA szekvencia elemzéshez
- **Fehérje kutatás**: AlphaFold integráció szerkezeti biológiához
- **Kutatási trendek**: Exa AI alapú naprakész tudományos információk

### Üzleti Alkalmazások
- **Biotechnológia**: Gyógyszeripari K+F támogatás
- **Oktatás**: Tudományos képzések és kurzusok
- **Konzultáció**: Szakértői elemzések és innovációs tanácsadás

### Technológiai Előnyök
- **Párhuzamos AI**: Három modell egyidejű futtatása
- **Költséghatékonyság**: Intelligens modell-váltás
- **Skálázhatóság**: Replit Autoscale használata
- **Megbízhatóság**: Többszintű fallback rendszer

## API végpontok

### Alap szolgáltatások
- `/api/deep_discovery/chat` - Hibrid chat funkció
- `/api/deep_discovery/research_trends` - Tudományos trendek
- `/api/deep_discovery/protein_structure` - AlphaFold fehérje lekérdezés
- `/api/deep_discovery/alphagenome` - Genomikai elemzés

### Piaci funkciók
- `/api/subscription/upgrade` - Előfizetés frissítés
- `/api/user/usage/{user_id}` - Felhasználói használat
- `/api/analytics/dashboard` - Platform statisztikák

### Autentikáció
- `/api/auth/apple` - Apple Sign In
- `/health` - Rendszer állapot

## Telepítés és futtatás

### Replit Secrets beállítása
```bash
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_gemini_key
EXA_API_KEY=your_exa_key
GCP_SERVICE_ACCOUNT_KEY=your_gcp_json
GCP_PROJECT_ID=your_project_id
GCP_REGION=us-central1
```

### Deployment
```bash
# Autoscale deployment Replit-en
# Automatikus skálázás 1-10 instance között
# 0.5 vCPU, 1GB RAM / instance