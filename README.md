# Telco Churn Advisor

Final project AI Engineer — prediksi customer churn + agent chat (tool-calling, RAG) di atas data telco.

## Live Demo & Presentasi

- **Aplikasi live**: https://initiative-treating-fda-titten.trycloudflare.com/
- **Slide presentasi (PPT)**: https://docs.google.com/presentation/d/15sEXZMKRODHdeamQnghaanRVn4Z6B0B7JNvgvorcQAs/edit?usp=sharing

(URL aplikasi berubah tiap kali tunnel di-restart — kalau link di atas sudah
tidak aktif, lihat bagian "Deploy publik (Cloudflare Tunnel)" di bawah untuk
cara mendapatkan URL baru.)

## Struktur

```
.
├── docker-compose.yml
├── .env.example              # salin ke .env, isi OPENROUTER_API_KEY
├── ingestion/                # fetch + cleaning + validasi + scheduler (APScheduler)
│   ├── pipeline.py
│   └── main.py
├── backend/                  # FastAPI: training, /predict, /chat, chroma store
│   ├── app.py                 # FastAPI app: /health, /predict, /chat, + endpoint admin
│   ├── train.py                # stratified split + 5-fold CV, log MLflow runs, champion/challenger gate
│   ├── model.py                 # load champion model dari MLflow Registry, scoring
│   ├── agent.py                  # tool-calling loop (OpenRouter) untuk /chat
│   ├── explain.py                 # SHAP explainability untuk satu pelanggan
│   ├── rag.py                      # chunk + embed FAQ docs ke Chroma
│   ├── system_status.py             # kesehatan sistem, statistik agregat, top-risk
│   ├── logging_store.py              # catat tiap request /predict & /chat
│   ├── eval_rag.py                    # evaluasi kualitas retrieval (hit_rate@3, MRR)
│   ├── eval_agent_scope.py             # verifikasi tool mana yang dipanggil per prompt
│   └── docs/                            # FAQ & kebijakan retensi untuk RAG
├── frontend/                  # Streamlit: Chat, Analytics, Explain (navigasi sidebar)
├── mlflow/                     # image MLflow tracking server
├── postgres/                    # image Postgres
├── tests/                        # unit test (pytest), tanpa dependency layanan eksternal
├── scripts/tunnel.sh              # buka URL publik via Cloudflare Tunnel
├── docs/
│   └── injection_tests.md          # 5 percobaan prompt injection nyata + mitigasi
└── README.md
```

## Kontrak API

Endpoint ini adalah kontrak inti proyek dan tidak diubah bentuknya oleh
fitur tambahan apapun (lihat "Status implementasi" untuk endpoint admin
lain yang bersifat aditif):

```
POST /predict
{ "customer_id": "7590-VHVEG" }
→ { "churn_probability": 0.72, "risk_level": "high" }

POST /chat
{ "customer_id": "7590-VHVEG", "message": "kenapa saya berisiko pindah?" }
→ { "reply": "...", "sources": ["kebijakan_retensi.md#0"], "tool_calls": ["predict_churn", "explain_prediction", "retrieve_docs"] }
```

## Menjalankan

Ada dua jenis perintah yang perlu dibedakan: `docker compose up` untuk
**menyalakan seluruh sistem**, dan `docker compose exec` untuk
**menjalankan satu perintah sekali jalan** di dalam container yang sudah
hidup (training, evaluasi, dsb).

### 1. Setup awal (sekali saja)

```bash
cp .env.example .env
# buka .env, isi OPENROUTER_API_KEY (dibutuhkan supaya /chat bisa memanggil LLM)
```

### 2. Menyalakan seluruh sistem

```bash
docker compose up --build
```

Ini menyalakan kelima service sekaligus: `postgres`, `mlflow`, `backend`,
`frontend`, `ingestion`. Ingestion otomatis mengisi Postgres begitu
container-nya jalan (terjadwal ulang tiap 60 menit) — tidak ada langkah
manual yang dibutuhkan untuk data. Tunggu sampai semua service berstatus
`healthy` sebelum lanjut ke langkah berikutnya (`docker compose ps` untuk
cek).

Untuk menyalakan di background (tidak menahan terminal): tambahkan `-d`,
lalu pantau log dengan `docker compose logs -f <nama service>`.

### 3. Melatih model (sekali, sebelum /predict bisa menjawab)

`/predict` butuh model yang sudah terdaftar di MLflow Model Registry.
Jalankan lewat `exec`, bukan `up`, karena ini perintah sekali-jalan di
dalam container `backend` yang sudah hidup:

```bash
docker compose exec backend python train.py
```

Ini melatih 5 kandidat model, mengevaluasi tiap kandidat dengan stratified
5-fold cross-validation (bukan cuma satu train/test split — pada data
imbalance seperti ini, satu split acak saja bisa memberi angka PR-AUC yang
menyesatkan tergantung kebetulan pembagian datanya), lalu mempromosikan
kandidat terbaik jadi `@champion` HANYA jika benar-benar mengungguli
champion yang sedang aktif.

### 4. Perintah `exec` lain yang tersedia

Semua ini juga dijalankan di container `backend` yang sudah hidup, sama
polanya seperti training:

```bash
# Evaluasi kualitas retrieval RAG (hit_rate@3, MRR)
docker compose exec backend python eval_rag.py

# Verifikasi tool mana yang dipanggil agent untuk kumpulan prompt yang
# sudah dikurasi (termasuk grup prompt yang SENGAJA tidak seharusnya
# memanggil tool apapun)
docker compose exec backend python eval_agent_scope.py

# Buka shell di dalam container backend untuk debug manual
docker compose exec backend bash
```

## Service yang tersedia

Port host bisa diubah lewat `.env` kalau bentrok dengan service lain di
mesin kamu:

| Service    | URL                                |
|------------|-------------------------------------|
| Frontend   | http://localhost:8502               |
| Backend    | http://localhost:8000/docs          |
| MLflow UI  | http://localhost:5001               |
| Ingestion  | http://localhost:8001/health        |
| Postgres   | localhost:5432                      |

(Port default MLflow/Frontend digeser dari 5000/8501 karena sering bentrok dengan
AirPlay Receiver macOS dan project Streamlit lain di mesin dev.)

## Troubleshooting

**`docker compose up` gagal karena port sudah dipakai**
Ubah port host di `.env` (misalnya `BACKEND_PORT`, `FRONTEND_PORT`,
`MLFLOW_PORT`), lalu `docker compose up --build` lagi. Port container di
dalam tidak berubah, jadi service tetap saling terhubung normal.

**`/predict` mengembalikan error "model not found" / 404 aneh**
Model belum pernah dilatih. Jalankan `docker compose exec backend python
train.py` (langkah 3 di atas) dulu — ini langkah manual yang memang HARUS
dilakukan sekali sebelum `/predict` punya model untuk dipanggil.

**`docker compose exec backend ...` gagal dengan "service is not running"**
Container `backend` belum sehat. Cek `docker compose ps` — kalau
statusnya bukan `healthy`, tunggu beberapa detik lagi atau cek
`docker compose logs backend` untuk lihat errornya.

**`/chat` selalu error atau timeout**
Cek `OPENROUTER_API_KEY` sudah terisi benar di `.env`, lalu restart
backend: `docker compose restart backend`. Tanpa API key ini, endpoint
`/predict` tetap jalan normal (tidak butuh LLM), tapi `/chat` tidak akan
bisa memanggil model bahasa.

**Data tidak masuk-masuk ke halaman Analytics**
Ingestion butuh waktu beberapa detik setelah container pertama kali
hidup. Cek `curl http://localhost:8001/health` dan
`curl http://localhost:8001/status` untuk lihat status job terakhir.

**Container lama numpuk / mau mulai bersih dari nol**
```bash
docker compose down -v   # -v juga menghapus volume Postgres & MLflow data
docker compose up --build
```
Perhatikan `-v` menghapus data ter-ingest dan riwayat MLflow, jadi
langkah 3 (training) perlu diulang setelahnya.

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
- ✅ Training dengan stratified split + stratified 5-fold cross-validation, PR-AUC &
  recall di-log ke MLflow, model terbaik diregister sebagai `telco_churn_model@champion`
  hanya kalau benar-benar mengungguli champion sebelumnya
- ✅ `/predict` memuat model dari MLflow Registry dan benar-benar dipanggil
- ✅ `/chat`: agent tool-calling manual (OpenRouter) dengan 6 tools (prediksi, SHAP,
  RAG, dan 3 tools statistik lintas seluruh basis pelanggan), memutuskan sendiri kapan
  masing-masing dipanggil — diverifikasi lewat `eval_agent_scope.py`, termasuk kumpulan
  prompt yang sengaja TIDAK seharusnya memanggil tool apapun
- ✅ Prompt injection: 5 percobaan nyata (direct, indirect via RAG-poisoning, roleplay,
  fake history, base64 obfuscation), semuanya gagal dieksploitasi — lihat
  [docs/injection_tests.md](docs/injection_tests.md)
- ✅ Dashboard Analytics interaktif (navigasi sidebar konsisten di semua halaman,
  perbandingan churn rate antar beberapa kategori sekaligus)
- ⬜ Belum: hardening tambahan (rate limiting, autentikasi endpoint admin, review
  dokumen sebelum masuk index)
