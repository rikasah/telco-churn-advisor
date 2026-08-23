"""Streamlit UI for Telco Churn Advisor: chat, analytics, and explainability."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Telco Churn Advisor", layout="wide")
st.title("Telco Churn Advisor")

tab_chat, tab_analytics, tab_explain = st.tabs(["Chat", "Analytics", "Explain"])

# ============================================================
# Chat
# ============================================================
with tab_chat:
    customer_id = st.text_input("Customer ID", value="7590-VHVEG")

    st.caption(
        "Agent ini bisa lebih dari sekadar melaporkan skor churn -- bisa menjelaskan "
        "alasannya, menjawab pertanyaan kebijakan, atau sekadar menyapa. Coba salah satu "
        "contoh di bawah, atau ketik pertanyaan sendiri."
    )
    example_questions = [
        "Kenapa saya berisiko pindah dari layanan ini?",
        "Apa saja penawaran retensi untuk pelanggan berisiko tinggi?",
        "Apa bedanya kontrak bulanan dan kontrak tahunan?",
        "Halo, apa kabar?",
    ]
    example_cols = st.columns(len(example_questions))
    clicked_example = None
    for col, q in zip(example_cols, example_questions):
        if col.button(q, use_container_width=True):
            clicked_example = q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = clicked_example or st.chat_input(
        "Tanya soal customer ini, kebijakan retensi, atau apa saja..."
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"customer_id": customer_id, "message": prompt},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data.get("reply", "")
                st.write(reply)
                if data.get("sources"):
                    st.caption(f"Sources: {', '.join(data['sources'])}")
                if data.get("tool_calls"):
                    st.caption(f"Tools used: {', '.join(data['tool_calls'])}")
            except requests.RequestException as e:
                reply = f"Error calling backend: {e}"
                st.error(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

# ============================================================
# Analytics
# ============================================================
with tab_analytics:
    st.caption("Ringkasan penggunaan sistem secara live -- diambil dari log request nyata di Postgres.")

    if st.button("Refresh"):
        st.rerun()

    try:
        stats = requests.get(f"{BACKEND_URL}/stats", timeout=15).json()
    except requests.RequestException as e:
        st.error(f"Tidak bisa mengambil statistik: {e}")
        stats = None

    try:
        sys_status = requests.get(f"{BACKEND_URL}/system-status", timeout=15).json()
    except requests.RequestException as e:
        st.error(f"Tidak bisa mengambil status sistem: {e}")
        sys_status = None

    if sys_status:
        st.subheader("Kesehatan data & model")
        data_status = sys_status["data"]
        champion = sys_status["champion"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total pelanggan di database", f"{data_status['total_customers']:,}")
        last_ingested = data_status["last_ingested_at"]
        c2.metric("Data terakhir masuk", last_ingested[:19].replace("T", " ") if last_ingested else "-")
        job = data_status["last_ingestion_job"]
        job_status = job["status"] if job else "unknown"
        c3.metric("Status job ingestion terakhir", job_status)
        if champion:
            c4.metric("Model champion", f"v{champion['version']} (PR-AUC {champion['pr_auc']:.3f})")
        else:
            c4.metric("Model champion", "belum ada")

    if stats:
        st.subheader("Trafik")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total request", stats["total_requests"])
        col2.metric("/predict calls", stats["predict_count"])
        col3.metric("/chat calls", stats["chat_count"])

        latency = stats.get("latency", {})
        if latency:
            lat_cols = st.columns(len(latency))
            for col, (endpoint, l) in zip(lat_cols, latency.items()):
                col.metric(f"Latency {endpoint} (avg / p95)", f"{l['avg_ms']:.0f}ms / {l['p95_ms']:.0f}ms")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribusi risk level")
            risk = stats["risk_level_distribution"]
            if risk:
                df_risk = pd.DataFrame({"risk_level": list(risk.keys()), "jumlah": list(risk.values())})
                st.bar_chart(df_risk, x="risk_level", y="jumlah")
            else:
                st.info("Belum ada data /predict.")

        with col_b:
            st.subheader("Tools yang paling sering dipanggil agent")
            tools = stats["top_tools"]
            if tools:
                df_tools = pd.DataFrame({"tool": list(tools.keys()), "jumlah": list(tools.values())})
                st.bar_chart(df_tools, x="tool", y="jumlah")
            else:
                st.info("Belum ada data /chat dengan tool call.")

        st.subheader("Request terakhir")
        if stats["recent"]:
            st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True)
        else:
            st.info("Belum ada request tercatat.")

    st.subheader("Tren PR-AUC antar run training")
    try:
        trend = requests.get(f"{BACKEND_URL}/model-trend", timeout=15).json()
    except requests.RequestException as e:
        st.error(f"Tidak bisa mengambil riwayat training: {e}")
        trend = []
    if trend:
        df_trend = pd.DataFrame(trend)
        df_trend["run_order"] = range(1, len(df_trend) + 1)
        chart = (
            alt.Chart(df_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("run_order:O", title="Urutan run"),
                y=alt.Y("pr_auc:Q", title="PR-AUC", scale=alt.Scale(zero=False)),
                tooltip=["run_name", "pr_auc"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Belum ada run training tercatat.")

    st.subheader("Pelanggan dengan risiko churn tertinggi saat ini")
    st.caption("Dihitung on-demand dari model champion terhadap seluruh basis pelanggan.")
    n_top = st.slider("Jumlah pelanggan ditampilkan", 5, 30, 10)
    if st.button("Hitung top-risk"):
        try:
            top_risk = requests.get(f"{BACKEND_URL}/top-risk", params={"n": n_top}, timeout=60).json()
            st.dataframe(pd.DataFrame(top_risk), use_container_width=True)
        except requests.RequestException as e:
            st.error(f"Gagal menghitung: {e}")

# ============================================================
# Explain
# ============================================================
with tab_explain:
    st.caption("Bongkar prediksi model untuk satu pelanggan menjadi faktor konkret (SHAP).")
    explain_customer_id = st.text_input("Customer ID untuk dijelaskan", value="7590-VHVEG", key="explain_cid")

    if st.button("Jelaskan prediksi"):
        try:
            pred_resp = requests.post(
                f"{BACKEND_URL}/predict", json={"customer_id": explain_customer_id}, timeout=30
            )
            pred_resp.raise_for_status()
            pred = pred_resp.json()
            st.metric("Churn probability", f"{pred['churn_probability']:.1%}", pred["risk_level"])

            exp_resp = requests.post(
                f"{BACKEND_URL}/explain", json={"customer_id": explain_customer_id}, timeout=60
            )
            exp_resp.raise_for_status()
            factors = exp_resp.json()["faktor_utama"]

            df_factors = pd.DataFrame(factors)
            df_factors["signed_impact"] = df_factors.apply(
                lambda r: r["besar_pengaruh"] if r["arah"] == "meningkatkan risiko" else -r["besar_pengaruh"],
                axis=1,
            )
            df_factors["label"] = df_factors["faktor"] + " = " + df_factors["nilai_pelanggan"]

            chart = (
                alt.Chart(df_factors)
                .mark_bar()
                .encode(
                    x=alt.X("signed_impact:Q", title="Pengaruh terhadap risiko churn"),
                    y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
                    color=alt.condition(
                        alt.datum.signed_impact > 0,
                        alt.value("#C94A4A"),
                        alt.value("#3E9B5C"),
                    ),
                    tooltip=["faktor", "nilai_pelanggan", "arah", "besar_pengaruh"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Merah = meningkatkan risiko churn. Hijau = menurunkan risiko churn.")
        except requests.RequestException as e:
            st.error(f"Gagal menjelaskan prediksi: {e}")
