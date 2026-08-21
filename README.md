# Telco Churn Advisor

Final project AI Engineer — prediksi customer churn + agent chat (tool-calling, RAG) di atas data telco.

## Struktur

```
.
├── docker-compose.yml
├── ingestion/         # script cleaning + validasi + scheduler (APScheduler)
├── backend/           # FastAPI: training script, /predict, /chat, chroma store
│   ├── train.py
│   ├── app.py
│   └── docs/          # PDF/markdown FAQ & T&C untuk RAG
├── frontend/          # Streamlit chat
├── mlflow/            # MLflow tracking server
├── postgres/          # Postgres image
├── docs/
│   └── injection_tests.md
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
docker compose up --build
```

Service yang tersedia setelah `up`:

| Service    | URL                              |
|------------|-----------------------------------|
| Frontend   | http://localhost:8501             |
| Backend    | http://localhost:8000/docs        |
| MLflow UI  | http://localhost:5000             |
| Ingestion  | http://localhost:8001/health      |
| Postgres   | localhost:5432                    |

Status: skeleton — endpoint tersedia dan mengembalikan respons kosong/placeholder, logika training/agent/ingestion belum diimplementasikan.
