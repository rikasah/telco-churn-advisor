# Telco Churn Advisor

Final project AI Engineer — prediksi customer churn + agent chat (tool-calling, RAG) di atas data telco.

## Struktur

```
.
├── docker-compose.yml
├── ingestion/         # fetch + cleaning + validasi + scheduler (APScheduler)
├── backend/           # FastAPI: training script, /predict, /chat, chroma store
│   ├── train.py       # stratified split, log MLflow runs, register champion model
│   ├── app.py         # FastAPI app: /health, /predict, /chat
│   ├── model.py        # load champion model dari MLflow, scoring
│   ├── agent.py        # tool-calling loop (OpenRouter) untuk /chat
│   ├── rag.py          # chunk + embed FAQ docs ke Chroma
│   └── docs/           # FAQ & kebijakan retensi untuk RAG
├── frontend/           # Streamlit chat
├── mlflow/             # MLflow tracking server
├── postgres/           # Postgres image
├── docs/
│   └── injection_tests.md   # hasil percobaan prompt injection nyata
└── README.md
```

## Kontrak API

```
POST /predict
{ "customer_id": "C001" }
→ { "churn_probability": 0.72, "risk_level": "high" }

POST /chat
{ "customer_id": "C001", "message": "kenapa saya berisiko pindah?" }
→ { "reply": "...", "sources": ["faq_kontrak.pdf#2"], "tool_calls": ["predict_churn"] }
```

## Menjalankan

```bash
cp .env.example .env
# isi OPENROUTER_API_KEY di .env (dibutuhkan agar /chat bisa memanggil LLM)
docker compose up --build
```

Ingestion otomatis mengisi Postgres begitu container jalan. Untuk melatih model dan
mengisi MLflow Model Registry (dibutuhkan sebelum `/predict` bisa menjawab):

```bash
docker compose exec backend python train.py
```

Service yang tersedia setelah `up` (port host bisa diubah lewat `.env`):

| Service    | URL                                |
|------------|-------------------------------------|
| Frontend   | http://localhost:8502               |
| Backend    | http://localhost:8000/docs          |
| MLflow UI  | http://localhost:5001               |
| Ingestion  | http://localhost:8001/health        |
| Postgres   | localhost:5432                      |

(Port default MLflow/Frontend digeser dari 5000/8501 karena sering bentrok dengan
AirPlay Receiver macOS dan project Streamlit lain di mesin dev.)

## Deploy publik (Cloudflare Tunnel)

```bash
brew install cloudflared   # sekali saja
./scripts/tunnel.sh
```

Script akan print URL publik `https://<random>.trycloudflare.com` — itu yang dibagikan
ke evaluator. Catatan soal restart:
- `docker compose down` lalu `up` lagi **tidak** mengubah URL — tunnel cuma nge-forward
  ke `localhost:8502`, begitu container hidup lagi URL yang sama otomatis jalan lagi.
- URL baru hanya dibutuhkan kalau proses `cloudflared`-nya sendiri mati (Ctrl+C, tutup
  terminal, atau restart Mac). Jalankan `./scripts/tunnel.sh` lagi untuk dapat URL baru.
- Untuk stop tunnel: `pkill -f 'cloudflared tunnel'`.

Jalankan `./scripts/tunnel.sh` sesaat sebelum sesi demo.

## Status implementasi

- ✅ Ingestion otomatis (IBM Telco Churn dataset → Postgres, on-startup + hourly)
- ✅ Training dengan stratified split, PR-AUC & recall di-log ke MLflow, model
  terbaik diregister sebagai `telco_churn_model@champion`
- ✅ `/predict` memuat model dari MLflow Registry dan benar-benar dipanggil
- ✅ `/chat`: agent tool-calling manual (OpenRouter) yang memutuskan sendiri kapan
  memanggil `predict_churn` / `retrieve_docs`, grounded dengan sumber terlihat
- ✅ Prompt injection: 2 percobaan nyata (direct + indirect via RAG) — lihat
  [docs/injection_tests.md](docs/injection_tests.md)
- ⬜ Belum: hardening tambahan (rate limiting, review dokumen sebelum masuk index)
